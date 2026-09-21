import inspect
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TOTAL_BATCH_SIZE = 480


def ce_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, vocab = logits.shape
    return F.cross_entropy(logits.reshape(bsz * seqlen, vocab).float(), targets.reshape(bsz * seqlen))


@torch.no_grad()
def parameter_summary(model: nn.Module, *, sharded: bool = False) -> dict:
    local_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    local_attn = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and ".attn." in name)
    local_emb = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and ("wte." in name or "wpe." in name)
    )

    if sharded and dist_ready():
        device = next(model.parameters()).device
        counts = torch.tensor([local_total, local_attn, local_emb], device=device, dtype=torch.long)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        total, attn, emb = int(counts[0].item()), int(counts[1].item()), int(counts[2].item())
    else:
        total, attn, emb = local_total, local_attn, local_emb

    return {
        "total_params": total,
        "attn_params": attn,
        "non_attn_params": total - attn,
        "attn_pct": 100.0 * attn / max(1, total),
        "embedding_params": emb,
    }


@torch.no_grad()
def global_param_norm(model: nn.Module, *, sharded: bool = False) -> float:
    local_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            local_sq += p.detach().float().pow(2).sum().item()
    if sharded and dist_ready():
        t = torch.tensor(local_sq, device=next(model.parameters()).device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return math.sqrt(t.item())
    return math.sqrt(local_sq)


@torch.no_grad()
def memory_stats(device: torch.device) -> dict:
    return {
        "memory/allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "memory/reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "memory/peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "memory/peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    v = str(value).strip()
    if v == "" or v.lower() in {"none", "null"}:
        return None
    return v


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_barrier(device: torch.device | None = None):
    if not dist_ready():
        return
    if device is not None and device.type == "cuda" and device.index is not None:
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


@torch.no_grad()
def reduce_scalar(value: float, device: torch.device, op: str = "mean") -> float:
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    if not dist_ready():
        return float(t.item())
    if op == "sum":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    elif op == "mean":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    elif op == "max":
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    elif op == "min":
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"unknown reduce op: {op}")
    return float(t.item())


@torch.no_grad()
def average_tensors_(tensors):
    if not dist_ready():
        return
    world_size = dist.get_world_size()
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(world_size)


def aggregate_stats(local_stats: dict, device: torch.device, global_tokens_per_step: int) -> dict:
    aggregated = {}
    for key, value in local_stats.items():
        if key == "step":
            aggregated[key] = int(value)
        elif key in {"perf/step_ms", "perf/hessian_ms", "perf/fwd_bwd_ms", "perf/elapsed_hours"} or key.startswith(
            "memory/"
        ):
            aggregated[key] = reduce_scalar(value, device, op="max")
        elif key == "lr":
            aggregated[key] = float(value)
        else:
            aggregated[key] = reduce_scalar(value, device, op="mean")

    aggregated["tokens_seen"] = int((aggregated["step"] + 1) * global_tokens_per_step)
    aggregated["perf/train_tokens_step"] = int(global_tokens_per_step)
    aggregated["perf/tokens_per_s"] = global_tokens_per_step / max(1e-12, aggregated["perf/step_ms"] / 1000.0)

    if "train/loss" in aggregated:
        aggregated["train/ppl"] = math.exp(min(20.0, aggregated["train/loss"]))
    if "eval/train_loss" in aggregated:
        aggregated["eval/train_ppl"] = math.exp(min(20.0, aggregated["eval/train_loss"]))
    if "eval/loss" in aggregated:
        aggregated["eval/ppl"] = math.exp(min(20.0, aggregated["eval/loss"]))

    return aggregated


def apply_hf_env(args):
    if normalize_optional_str(args.hf_token):
        os.environ["HF_TOKEN"] = normalize_optional_str(args.hf_token)
    if normalize_optional_str(args.hf_home):
        os.environ["HF_HOME"] = normalize_optional_str(args.hf_home)
    if normalize_optional_str(args.hf_hub_cache):
        os.environ["HF_HUB_CACHE"] = normalize_optional_str(args.hf_hub_cache)
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.hf_disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if args.hf_debug:
        os.environ["HF_DEBUG"] = "1"


def resolve_batching(args, world_size: int):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if args.grad_accum_steps is not None and args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    total_batch_size = args.total_batch_size
    if args.grad_accum_steps is None:
        if total_batch_size is None:
            total_batch_size = DEFAULT_TOTAL_BATCH_SIZE
        denom = args.batch_size * world_size
        if total_batch_size % denom != 0:
            raise ValueError(
                "--total-batch-size must be divisible by batch_size * world_size "
                f"(got total={total_batch_size}, batch_size={args.batch_size}, world_size={world_size})"
            )
        args.grad_accum_steps = total_batch_size // denom
    else:
        effective_total_batch_size = args.batch_size * args.grad_accum_steps * world_size
        if total_batch_size is None:
            total_batch_size = effective_total_batch_size
        elif effective_total_batch_size != total_batch_size:
            raise ValueError(
                "--total-batch-size does not match batch_size * grad_accum_steps * world_size "
                f"(got total={total_batch_size}, effective={effective_total_batch_size})"
            )

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    elif args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be >= 1")

    args.total_batch_size = total_batch_size
    return args


def init_distributed(args):
    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")

        torch.cuda.set_device(local_rank)

        init_kwargs = {
            "backend": "nccl",
            "init_method": "env://",
        }
        if "device_id" in inspect.signature(dist.init_process_group).parameters:
            init_kwargs["device_id"] = local_rank

        dist.init_process_group(**init_kwargs)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
        if device.type == "cuda":
            if device.index is None:
                torch.cuda.set_device(0)
                device = torch.device("cuda", 0)
            else:
                torch.cuda.set_device(device.index)

    return rank, local_rank, world_size, distributed, rank == 0, device


def init_wandb(args, config: dict, summary: dict, is_master: bool):
    if not args.wandb or not is_master:
        return None

    import wandb

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        group=args.wandb_group if args.wandb_group else None,
        name=args.run_name,
        config={**config, **summary},
        tags=args.wandb_tags.split(",") if args.wandb_tags else None,
    )
    return wandb
