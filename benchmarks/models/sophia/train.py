import argparse
import json
import math
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from benchmarks.models.gqa import LlamaAttention
from benchmarks.models.language_model_benchmark import (
    ATTN_BACKENDS,
    SOPHIA_PRESETS,
    CausalSelfAttention,
    GPTNeoXAttention,
    model_backends,
    model_from_preset,
)
from benchmarks.harness import runtime_provenance
from benchmarks.models.sophia.data import DeviceBatchStream, create_batch_loader, resolve_dataset_args
from benchmarks.models.sophia.optimizer import AdaHessian, FlashSophiaH, SophiaH
from benchmarks.models.sophia.runtime import (
    aggregate_stats,
    apply_hf_env,
    average_tensors_,
    ce_loss_from_logits,
    dist_barrier,
    dist_ready,
    global_param_norm,
    init_distributed,
    init_wandb,
    memory_stats,
    normalize_optional_str,
    parameter_summary,
)
from benchmarks.models.sophia.runtime import reduce_scalar, resolve_batching


SECOND_ORDER_OPTIMIZERS = frozenset({"sophiah", "flash-sophiah", "adahessian"})
BOB_ATTENTION_BACKENDS = frozenset(
    backend for backend in ATTN_BACKENDS if backend.startswith("bob")
)


def _first_order_impl(backend: str) -> str:
    if backend in BOB_ATTENTION_BACKENDS:
        return "native_flash_internal"
    return backend


@dataclass(frozen=True)
class AttentionRouting:
    ordinary_backend: str
    hessian_backend: str
    first_order_impl: str

    @property
    def requires_hessian_switch(self) -> bool:
        return self.ordinary_backend != self.hessian_backend


def resolve_attention_routing(
    *,
    optimizer: str,
    attn_backend: str,
    non_bob_attn_backend: str | None,
) -> AttentionRouting:
    """Resolve ordinary/HVP dispatch without bypassing a BoB wrapper."""
    if attn_backend not in ATTN_BACKENDS:
        raise ValueError(f"unknown attention backend: {attn_backend!r}")
    if non_bob_attn_backend is not None and non_bob_attn_backend not in ATTN_BACKENDS:
        raise ValueError(f"unknown non-BoB attention backend: {non_bob_attn_backend!r}")

    second_order = optimizer in SECOND_ORDER_OPTIMIZERS
    if not second_order:
        if non_bob_attn_backend is not None:
            raise ValueError(
                "--non-bob-attn-backend is only valid with a second-order optimizer"
            )
        return AttentionRouting(
            attn_backend,
            attn_backend,
            _first_order_impl(attn_backend),
        )

    if attn_backend == "flash":
        raise ValueError(
            "a second-order optimizer cannot use --attn-backend=flash for its HVP; "
            "choose math or a bob-family backend"
        )

    if attn_backend in BOB_ATTENTION_BACKENDS:
        if non_bob_attn_backend not in {None, attn_backend}:
            raise ValueError(
                "--non-bob-attn-backend cannot override a bob-family backend; "
                "the BoB wrapper already uses native FlashAttention for forward and "
                "first backward"
            )
        return AttentionRouting(
            attn_backend,
            attn_backend,
            "native_flash_internal",
        )

    ordinary_backend = non_bob_attn_backend or attn_backend
    return AttentionRouting(
        ordinary_backend,
        attn_backend,
        _first_order_impl(ordinary_backend),
    )


def validate_hessian_schedule(
    *,
    optimizer: str,
    hess_interval: int,
    hutch_samples: int,
) -> None:
    if optimizer not in SECOND_ORDER_OPTIMIZERS:
        return
    if hess_interval < 1:
        raise ValueError("--hess-interval must be >= 1 for a second-order optimizer")
    if hutch_samples < 1:
        raise ValueError("--hutch-samples must be >= 1 for a second-order optimizer")


@contextmanager
def use_attention_backend(model: torch.nn.Module, backend: str):
    """Temporarily route every supported attention block through ``backend``."""
    if backend not in ATTN_BACKENDS:
        raise ValueError(f"unknown attention backend: {backend!r}")

    attention_modules = [
        module
        for module in model.modules()
        if isinstance(module, (CausalSelfAttention, GPTNeoXAttention, LlamaAttention))
    ]
    if not attention_modules:
        raise ValueError("model has no supported attention modules")

    previous_backends = [module.backend for module in attention_modules]
    for module in attention_modules:
        module.backend = backend
    try:
        yield
    finally:
        for module, previous_backend in zip(attention_modules, previous_backends):
            module.backend = previous_backend


def hessian_attention_context(model: torch.nn.Module, routing: AttentionRouting):
    if not routing.requires_hessian_switch:
        return nullcontext()
    return use_attention_backend(model, routing.hessian_backend)


def take_batch_prefix(idx: torch.Tensor, targets: torch.Tensor, take: int):
    if take <= 0:
        raise ValueError("take must be > 0")
    return idx[:take], targets[:take]


def build_hutch_micro_batches(
    step_micro_batches,
    hutch_batch_size: int,
    extra_stream: DeviceBatchStream | None = None,
):
    if hutch_batch_size < 1:
        raise ValueError("hutch_batch_size must be >= 1")

    batches = []
    sequences_left = hutch_batch_size

    for idx, targets in step_micro_batches:
        if sequences_left == 0:
            break
        take = min(sequences_left, idx.size(0))
        if take > 0:
            batches.append(take_batch_prefix(idx, targets, take))
            sequences_left -= take

    while sequences_left > 0:
        if extra_stream is None:
            raise RuntimeError("not enough sequences to build Hutchinson micro-batches")
        idx, targets = extra_stream.next_batch()
        take = min(sequences_left, idx.size(0))
        batches.append(take_batch_prefix(idx, targets, take))
        sequences_left -= take

    return batches


def build_hutch_monolithic_batch(
    stream: DeviceBatchStream,
    hutch_batch_size: int,
):
    if hutch_batch_size < 1:
        raise ValueError("hutch_batch_size must be >= 1")

    idx_parts = []
    target_parts = []
    sequences_left = hutch_batch_size

    while sequences_left > 0:
        idx, targets = stream.next_batch()
        take = min(sequences_left, idx.size(0))
        idx_part, target_part = take_batch_prefix(idx, targets, take)
        idx_parts.append(idx_part)
        target_parts.append(target_part)
        sequences_left -= take

    return torch.cat(idx_parts, dim=0), torch.cat(target_parts, dim=0)


def get_lr(step: int, peak_lr: float, min_lr: float, warmup_steps: int, lr_decay_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= lr_decay_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / float(max(1, lr_decay_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def hutchinson_diag_estimate(model: torch.nn.Module, params, micro_batches, num_samples: int):
    estimates = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    total_weight = 0.0

    for idx, targets in micro_batches:
        batch_weight = float(idx.size(0))
        total_weight += batch_weight

        for _ in range(num_samples):
            logits = model(idx)
            loss = ce_loss_from_logits(logits, targets)
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            probes = [torch.randn_like(p, dtype=p.dtype) for p in params]
            proj = sum((g * z).sum() for g, z in zip(grads, probes))
            hvp = torch.autograd.grad(proj, params, retain_graph=False, create_graph=False)

            for acc, hv, z in zip(estimates, hvp, probes):
                acc.add_((hv * z).float(), alpha=batch_weight / max(1.0, num_samples))

    if total_weight == 0.0:
        raise ValueError("no hutchinson batches provided")

    inv_total = 1.0 / total_weight
    for acc in estimates:
        acc.mul_(inv_total)
    return estimates


def _rademacher_probes(params):
    probes = []
    for p in params:
        probe = torch.empty_like(p, memory_format=torch.preserve_format)
        probe.bernoulli_(0.5).mul_(2).sub_(1)
        probes.append(probe)
    return probes


def flash_hutchinson_update(
    model: torch.nn.Module,
    optimizer: FlashSophiaH,
    params,
    micro_batches,
    num_samples: int,
):
    total_weight = sum(float(idx.size(0)) for idx, _targets in micro_batches)
    if total_weight == 0.0:
        raise ValueError("no hutchinson batches provided")

    optimizer.begin_hessian_update()
    for idx, targets in micro_batches:
        batch_weight = float(idx.size(0))
        alpha = batch_weight / (total_weight * max(1, num_samples))
        for _ in range(num_samples):
            logits = model(idx)
            loss = ce_loss_from_logits(logits, targets)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            probes = _rademacher_probes(params)
            proj = sum((g * z).sum() for g, z in zip(grads, probes))
            hvp = torch.autograd.grad(proj, params, retain_graph=False, create_graph=False)
            optimizer.accumulate_hessian_sample(
                hvp,
                probes,
                alpha=alpha,
                average_distributed=dist_ready(),
            )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    eval_batches,
    amp_dtype: torch.dtype,
    device_type: str,
) -> float:
    model.eval()
    losses = []
    amp_enabled = device_type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}

    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        for idx, targets in eval_batches:
            logits = model(idx)
            loss = ce_loss_from_logits(logits, targets)
            losses.append(loss.item())

    model.train()
    return sum(losses) / max(1, len(losses))


def should_run_eval(step: int, max_steps: int, eval_every: int, *, eval_on_start: bool = True) -> bool:
    if step == max_steps - 1:
        return True
    if step == 0:
        return eval_on_start
    return eval_every > 0 and (step + 1) % eval_every == 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument(
        "--preset",
        default="gpt2",
        choices=SOPHIA_PRESETS,
    )
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument(
        "--attn-backend",
        type=str,
        default="flash",
        choices=tuple(ATTN_BACKENDS),
        help="HVP attention backend for second-order runs; model backend otherwise",
    )
    p.add_argument(
        "--non-bob-attn-backend",
        type=str,
        default=None,
        choices=tuple(ATTN_BACKENDS),
        help=(
            "ordinary train/eval backend for a non-BoB second-order reference "
            "(for example, Flash ordinary passes with a math HVP); a bob-family "
            "backend cannot be overridden because it already wires in FlashAttention"
        ),
    )
    p.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "sophiah", "flash-sophiah", "adahessian"]
    )
    p.add_argument("--compile", action="store_true")

    p.add_argument("--dataset", type=str, default="fineweb", choices=["fineweb", "openwebtext", "custom"])
    p.add_argument("--dataset-id", type=str, default="")
    p.add_argument("--dataset-name", type=str, default="")
    p.add_argument("--dataset-revision", type=str, default="")
    p.add_argument("--dataset-data-dir", type=str, default="")
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--tokenizer-name", type=str, default="gpt2")
    p.add_argument(
        "--tokenizer-backend",
        choices=("tiktoken", "huggingface"),
        default="tiktoken",
    )
    p.add_argument("--tokenizer-revision", type=str, default="")
    p.add_argument("--train-token-budget", type=int, default=1_000_000_000)
    p.add_argument("--val-buckets", type=int, default=10)
    p.add_argument("--shuffle-buffer", type=int, default=1_000)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--total-batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--eval-on-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-train", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=2)

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--beta1", type=float, default=None)
    p.add_argument("--beta2", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--hessian-power", type=float, default=None)
    p.add_argument("--eps", type=float, default=None)

    p.add_argument("--hess-interval", type=int, default=10)
    p.add_argument("--hutch-batch-size", type=int, default=32)
    p.add_argument("--hutch-samples", type=int, default=1)
    p.add_argument("--hutch-mode", type=str, default="microbatch", choices=["microbatch", "monolithic"])
    p.add_argument("--optimizer-stats-every", type=int, default=None)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=1000)

    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--hf-home", type=str, default=os.environ.get("HF_HOME", ""))
    p.add_argument("--hf-hub-cache", type=str, default=os.environ.get("HF_HUB_CACHE", ""))
    p.add_argument("--hf-offline", action="store_true")
    p.add_argument("--hf-disable-xet", action="store_true")
    p.add_argument("--hf-debug", action="store_true")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="fineweb1b-bob")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-group", type=str, default="")
    p.add_argument("--wandb-tags", type=str, default="")

    p.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return p.parse_args()


def training_defaults(args):
    if args.optimizer == "adamw":
        args.lr = 6e-4 if args.lr is None else args.lr
        args.min_lr = 0.05 * args.lr if args.min_lr is None else args.min_lr
        args.weight_decay = 0.1 if args.weight_decay is None else args.weight_decay
        args.beta1 = 0.9 if args.beta1 is None else args.beta1
        args.beta2 = 0.95 if args.beta2 is None else args.beta2
        args.gamma = 0.01 if args.gamma is None else args.gamma
        args.hessian_power = 1.0 if args.hessian_power is None else args.hessian_power
        args.eps = 1e-12 if args.eps is None else args.eps
    elif args.optimizer in {"sophiah", "flash-sophiah"}:
        args.lr = 6e-4 if args.lr is None else args.lr
        args.min_lr = 0.05 * args.lr if args.min_lr is None else args.min_lr
        args.weight_decay = 0.2 if args.weight_decay is None else args.weight_decay
        args.beta1 = 0.96 if args.beta1 is None else args.beta1
        args.beta2 = 0.99 if args.beta2 is None else args.beta2
        args.gamma = 0.01 if args.gamma is None else args.gamma
        args.hessian_power = 1.0 if args.hessian_power is None else args.hessian_power
        args.eps = 1e-12 if args.eps is None else args.eps
    else:
        args.lr = 6e-4 if args.lr is None else args.lr
        args.min_lr = 0.05 * args.lr if args.min_lr is None else args.min_lr
        args.weight_decay = 0.1 if args.weight_decay is None else args.weight_decay
        args.beta1 = 0.9 if args.beta1 is None else args.beta1
        args.beta2 = 0.999 if args.beta2 is None else args.beta2
        args.hessian_power = 1.0 if args.hessian_power is None else args.hessian_power
        args.eps = 1e-8 if args.eps is None else args.eps
    return args


def recorded_args(args) -> dict[str, object]:
    """Return reproducibility arguments without persisting credentials."""

    return {key: value for key, value in vars(args).items() if key != "hf_token"}


def main():
    args = resolve_dataset_args(training_defaults(parse_args()))
    apply_hf_env(args)

    routing = resolve_attention_routing(
        optimizer=args.optimizer,
        attn_backend=args.attn_backend,
        non_bob_attn_backend=args.non_bob_attn_backend,
    )
    # sdpa_flash_public explicitly requests native FlashAttention and rejects
    # fp32. Validate that constraint before the first forward pass.
    if args.dtype == "float32" and routing.ordinary_backend == "flash":
        raise ValueError(
            "--dtype float32 is incompatible with the explicit 'flash' "
            "attention backend; native FlashAttention requires fp16/bf16 "
            "(use math or a bob-family backend for fp32)"
        )
    validate_hessian_schedule(
        optimizer=args.optimizer,
        hess_interval=args.hess_interval,
        hutch_samples=args.hutch_samples,
    )
    args.ordinary_attn_backend = routing.ordinary_backend
    args.hessian_attn_backend = routing.hessian_backend
    args.first_order_impl = routing.first_order_impl

    rank, local_rank, world_size, distributed, is_master, device = init_distributed(args)

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This script is intended for CUDA training")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = resolve_batching(args, world_size)
    cfg, model_factory = model_from_preset(args.preset)
    if routing.ordinary_backend not in model_backends(args.preset):
        raise ValueError(
            f"{args.preset} does not support ordinary backend "
            f"{routing.ordinary_backend!r}"
        )
    if routing.hessian_backend not in model_backends(args.preset):
        raise ValueError(
            f"{args.preset} does not support Hessian backend "
            f"{routing.hessian_backend!r}"
        )
    if args.block_size is not None:
        if args.block_size < 1:
            raise ValueError("--block-size must be >= 1")
        cfg.block_size = args.block_size
    local_sequences_per_step = args.batch_size * args.grad_accum_steps
    global_sequences_per_step = local_sequences_per_step * world_size
    local_tokens_per_step = local_sequences_per_step * cfg.block_size
    global_tokens_per_step = global_sequences_per_step * cfg.block_size
    max_steps = math.ceil(args.train_token_budget / global_tokens_per_step)
    lr_decay_steps = max_steps

    if args.hutch_batch_size < 1:
        raise ValueError("--hutch-batch-size must be >= 1")
    if args.optimizer_stats_every is None:
        args.optimizer_stats_every = args.log_every if args.optimizer == "flash-sophiah" else 1
    elif args.optimizer_stats_every < 0:
        raise ValueError("--optimizer-stats-every must be >= 0")

    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.num_workers > 0 and args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1 when --num-workers > 0")
    if args.num_workers == 0 and args.prefetch_factor is not None and args.prefetch_factor != 2:
        print("warning: --prefetch-factor is ignored when --num-workers=0")

    amp_dtype = getattr(torch, args.dtype)
    amp_enabled = amp_dtype in {torch.float16, torch.bfloat16}
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    raw_model = model_factory(cfg, routing.ordinary_backend).to(
        device=device, dtype=amp_dtype
    ).train()

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    ddp_model = None
    if distributed:
        ddp_model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        train_model = ddp_model
        model_for_io = raw_model
    else:
        train_model = raw_model
        model_for_io = raw_model

    if args.compile:
        train_model = torch.compile(train_model)

    params = [p for p in model_for_io.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sophiah":
        optimizer = SophiaH(
            params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            eps=args.eps,
        )
    elif args.optimizer == "flash-sophiah":
        optimizer = FlashSophiaH(
            params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            eps=args.eps,
        )
    else:
        optimizer = AdaHessian(
            params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            hessian_power=args.hessian_power,
        )
    second_order_optimizer = isinstance(optimizer, (SophiaH, AdaHessian))

    train_loader = create_batch_loader(
        args=args,
        cfg=cfg,
        split_kind="train",
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
    )
    train_stream = DeviceBatchStream(train_loader, device)
    val_loader = create_batch_loader(
        args=args,
        cfg=cfg,
        split_kind="val",
        batch_size=args.eval_batch_size,
        rank=rank,
        world_size=world_size,
    )
    val_stream = DeviceBatchStream(val_loader, device)
    val_eval_batches = [val_stream.next_batch() for _ in range(args.eval_batches)]
    train_eval_batches = None
    hutch_stream = None

    if args.eval_train:
        train_eval_loader = create_batch_loader(
            args=args,
            cfg=cfg,
            split_kind="train",
            batch_size=args.eval_batch_size,
            rank=rank,
            world_size=world_size,
            seed=args.seed + 20_000,
        )
        train_eval_stream = DeviceBatchStream(train_eval_loader, device)
        train_eval_batches = [train_eval_stream.next_batch() for _ in range(args.eval_batches)]

    if second_order_optimizer:
        hutch_loader = None
        hutch_seed = args.seed + 10_000
        if args.hutch_mode == "monolithic":
            hutch_loader = create_batch_loader(
                args=args,
                cfg=cfg,
                split_kind="train",
                batch_size=args.hutch_batch_size,
                rank=rank,
                world_size=world_size,
                seed=hutch_seed,
            )
        elif args.hutch_batch_size > local_sequences_per_step:
            hutch_loader = create_batch_loader(
                args=args,
                cfg=cfg,
                split_kind="train",
                batch_size=args.batch_size,
                rank=rank,
                world_size=world_size,
                seed=hutch_seed,
            )

        if hutch_loader is not None:
            hutch_stream = DeviceBatchStream(hutch_loader, device)

    summary = parameter_summary(model_for_io)
    provenance = runtime_provenance(device)
    run_dir = Path(args.out_dir) / args.run_name
    metrics_handle = None
    wandb = None

    if is_master:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    **recorded_args(args),
                    "world_size": world_size,
                    "local_tokens_per_step": local_tokens_per_step,
                    "global_tokens_per_step": global_tokens_per_step,
                    "max_steps": max_steps,
                    "model_config": asdict(cfg),
                    "provenance": provenance,
                    **summary,
                },
                f,
                indent=2,
            )
        metrics_handle = open(run_dir / "metrics.jsonl", "w", encoding="utf-8")
        wandb = init_wandb(
            args,
            {
                **recorded_args(args),
                "world_size": world_size,
                "local_tokens_per_step": local_tokens_per_step,
                "global_tokens_per_step": global_tokens_per_step,
                "max_steps": max_steps,
                "model_config": asdict(cfg),
                "provenance": provenance,
            },
            summary,
            is_master=True,
        )
        dataset_label = args.dataset_id
        dataset_name = normalize_optional_str(args.dataset_name)
        dataset_revision = normalize_optional_str(args.dataset_revision)
        if dataset_name is not None:
            dataset_label = f"{dataset_label}/{dataset_name}"
        if dataset_revision is not None:
            dataset_label = f"{dataset_label}@{dataset_revision}"

        print(f"run_name={args.run_name}")
        print(
            f"optimizer={args.optimizer} "
            f"ordinary_backend={routing.ordinary_backend} "
            f"hessian_backend={routing.hessian_backend} "
            f"first_order_impl={routing.first_order_impl}"
        )
        if second_order_optimizer:
            print(f"hutch_mode={args.hutch_mode} hutch_batch_size={args.hutch_batch_size}")
        print(f"dataset={dataset_label}")
        print(f"streaming={args.streaming}")
        print(f"world_size={world_size}")
        print(f"batch_size={args.batch_size}")
        print(f"grad_accum_steps={args.grad_accum_steps}")
        print(f"total_batch_size={args.total_batch_size}")
        print(f"eval_on_start={args.eval_on_start}")
        print(f"eval_train={args.eval_train}")
        print(f"local_tokens_per_step={local_tokens_per_step:,}")
        print(f"global_tokens_per_step={global_tokens_per_step:,}")
        print(f"max_steps={max_steps:,}")
        print(f"total_params={summary['total_params']:,}")
        print(f"attn_params={summary['attn_params']:,} ({summary['attn_pct']:.2f}%)")

    dist_barrier(device)

    optimizer.zero_grad(set_to_none=True)
    start_time = time.perf_counter()
    best_eval_loss = float("inf")

    for step in range(max_steps):
        lr = get_lr(step, args.lr, args.min_lr, args.warmup_steps, lr_decay_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        torch.cuda.reset_peak_memory_stats(device)
        step_t0 = time.perf_counter()

        hess_ms = 0.0
        hess_tokens = 0
        needs_hutch = second_order_optimizer and step % args.hess_interval == 0

        if needs_hutch and args.hutch_mode == "monolithic":
            hutch_idx, hutch_targets = build_hutch_monolithic_batch(
                hutch_stream,
                args.hutch_batch_size,
            )
            torch.cuda.synchronize(device)
            hess_t0 = time.perf_counter()
            with hessian_attention_context(model_for_io, routing):
                if isinstance(optimizer, FlashSophiaH):
                    flash_hutchinson_update(
                        model=model_for_io,
                        optimizer=optimizer,
                        params=params,
                        micro_batches=[(hutch_idx, hutch_targets)],
                        num_samples=args.hutch_samples,
                    )
                else:
                    hessian_estimates = hutchinson_diag_estimate(
                        model=model_for_io,
                        params=params,
                        micro_batches=[(hutch_idx, hutch_targets)],
                        num_samples=args.hutch_samples,
                    )
                    average_tensors_(hessian_estimates)
                    optimizer.update_hessian(hessian_estimates)
            torch.cuda.synchronize(device)
            hess_ms = 1000.0 * (time.perf_counter() - hess_t0)
            hess_tokens = args.hutch_batch_size * cfg.block_size * world_size

        micro_batches = [train_stream.next_batch() for _ in range(args.grad_accum_steps)]

        if needs_hutch and args.hutch_mode == "microbatch":
            hutch_micro_batches = build_hutch_micro_batches(
                micro_batches,
                args.hutch_batch_size,
                extra_stream=hutch_stream,
            )

            torch.cuda.synchronize(device)
            hess_t0 = time.perf_counter()
            with hessian_attention_context(model_for_io, routing):
                if isinstance(optimizer, FlashSophiaH):
                    flash_hutchinson_update(
                        model=model_for_io,
                        optimizer=optimizer,
                        params=params,
                        micro_batches=hutch_micro_batches,
                        num_samples=args.hutch_samples,
                    )
                else:
                    hessian_estimates = hutchinson_diag_estimate(
                        model=model_for_io,
                        params=params,
                        micro_batches=hutch_micro_batches,
                        num_samples=args.hutch_samples,
                    )
                    average_tensors_(hessian_estimates)
                    optimizer.update_hessian(hessian_estimates)
            torch.cuda.synchronize(device)
            hess_ms = 1000.0 * (time.perf_counter() - hess_t0)
            hess_tokens = args.hutch_batch_size * cfg.block_size * world_size

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_idx, (idx, targets) in enumerate(micro_batches):
            sync_now = (not distributed) or (micro_idx == args.grad_accum_steps - 1)
            sync_ctx = nullcontext() if sync_now else ddp_model.no_sync()
            with sync_ctx:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    logits = train_model(idx)
                    loss = ce_loss_from_logits(logits, targets)
                loss_accum += loss.item()
                scaled_loss = loss / args.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model_for_io.parameters(), args.grad_clip).item()

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        train_loss = loss_accum / args.grad_accum_steps

        local_stats = {
            "step": step,
            "train/loss": train_loss,
            "lr": lr,
            "grad/global_norm": grad_norm,
            "param/global_norm": global_param_norm(model_for_io),
            "perf/step_ms": step_ms,
            "perf/hessian_ms": hess_ms,
            "perf/fwd_bwd_ms": step_ms - hess_ms,
            "perf/hutch_tokens": hess_tokens,
            "perf/elapsed_hours": (time.perf_counter() - start_time) / 3600.0,
            **memory_stats(device),
        }

        should_collect_optimizer_stats = args.optimizer_stats_every > 0 and (
            step % args.optimizer_stats_every == 0 or step == max_steps - 1
        )
        if should_collect_optimizer_stats and hasattr(optimizer, "logging_stats"):
            local_stats.update(optimizer.logging_stats())

        if should_run_eval(step, max_steps, args.eval_every, eval_on_start=args.eval_on_start):
            if train_eval_batches is not None:
                local_stats["eval/train_loss"] = evaluate(
                    model_for_io,
                    train_eval_batches,
                    amp_dtype,
                    device.type,
                )
            local_stats["eval/loss"] = evaluate(
                model_for_io,
                val_eval_batches,
                amp_dtype,
                device.type,
            )

        stats = aggregate_stats(local_stats, device, global_tokens_per_step)
        stats["perf/hutch_tokens"] = int(hess_tokens)
        stats["world_size"] = world_size

        improved = False
        if "eval/loss" in stats and stats["eval/loss"] < best_eval_loss:
            best_eval_loss = stats["eval/loss"]
            improved = True

        def _save_checkpoint(path):
            if not is_master:
                return
            ckpt_payload = {
                "step": step,
                "tokens_seen": stats["tokens_seen"],
                "best_eval_loss": best_eval_loss,
                "args": recorded_args(args),
                "model_config": asdict(cfg),
                "world_size": world_size,
                "model": model_for_io.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(ckpt_payload, path)

        if improved:
            _save_checkpoint(run_dir / "best.pt")

        if step % args.save_every == 0 or step == max_steps - 1:
            _save_checkpoint(run_dir / "last.pt")

        if is_master:
            row = {
                "run_name": args.run_name,
                "optimizer": args.optimizer,
                "backend": args.attn_backend,
                "non_bob_backend": args.non_bob_attn_backend,
                "ordinary_backend": routing.ordinary_backend,
                "hessian_backend": routing.hessian_backend,
                "first_order_impl": routing.first_order_impl,
                **stats,
            }
            metrics_handle.write(json.dumps(row) + "\n")
            metrics_handle.flush()
            if wandb is not None:
                wandb.log(row, step=step)
            if step % args.log_every == 0:
                eval_msg = f" | eval_loss={stats['eval/loss']:.4f}" if "eval/loss" in stats else ""
                print(
                    f"step={step:6d} | "
                    f"loss={stats['train/loss']:.4f} | "
                    f"lr={lr:.6g} | "
                    f"step_ms={stats['perf/step_ms']:8.2f} | "
                    f"tok/s={stats['perf/tokens_per_s']:10.1f} | "
                    f"peak_mb={stats['memory/peak_allocated_mb']:8.1f}"
                    f"{eval_msg}"
                )

    final_best_eval_loss = reduce_scalar(best_eval_loss, device, op="min") if math.isfinite(best_eval_loss) else None
    final_summary = {
        "run_name": args.run_name,
        "backend": args.attn_backend,
        "non_bob_backend": args.non_bob_attn_backend,
        "ordinary_backend": routing.ordinary_backend,
        "hessian_backend": routing.hessian_backend,
        "first_order_impl": routing.first_order_impl,
        "world_size": world_size,
        "steps": max_steps,
        "tokens_seen": max_steps * global_tokens_per_step,
        "best_eval_loss": final_best_eval_loss,
        "best_eval_ppl": math.exp(min(20.0, final_best_eval_loss)) if final_best_eval_loss is not None else None,
        "metrics_path": str(run_dir / "metrics.jsonl") if is_master else None,
    }

    if is_master:
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)
        print(json.dumps(final_summary, indent=2))

    if metrics_handle is not None:
        metrics_handle.close()
    if wandb is not None:
        wandb.finish()
    if dist_ready():
        dist_barrier(device)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
