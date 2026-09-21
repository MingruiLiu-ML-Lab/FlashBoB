"""Language-model definitions and the model benchmark command-line runner."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bob import sdpa_bob
from bob.attention import sdpa_math
from benchmarks.harness import PROVENANCE_FIELDS, dtype_name, runtime_provenance, source_provenance, write_rows
from benchmarks.harness import cleanup_device, run_with_oom_capture, sync_device, time_and_peak
from benchmarks.baselines import sdpa_hvp_manual, sdpa_hvp_semi_manual
from benchmarks.models.rotary import apply_rotary


def sdpa_flash(q, k, v, *, is_causal=True, scale=None, window_size=0):
    if window_size:
        raise NotImplementedError("native FlashAttention does not accept a sliding window")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=q.shape[1] != k.shape[1],
        )


ATTN_BACKENDS = {
    "flash": sdpa_flash,
    "math": sdpa_math,
    "bob": sdpa_bob,
    "hvp_manual": sdpa_hvp_manual,
    "hvp_semi_manual": sdpa_hvp_semi_manual,
}


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head


def gpt2_config(preset: str) -> GPTConfig:
    if preset == "gpt2":
        return GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)
    if preset == "gpt2-medium":
        return GPTConfig(vocab_size=50257, block_size=1024, n_layer=24, n_head=16, n_embd=1024)
    if preset == "gpt2-large":
        return GPTConfig(vocab_size=50257, block_size=1024, n_layer=36, n_head=20, n_embd=1280)
    if preset == "gpt2-xl":
        return GPTConfig(vocab_size=50257, block_size=1024, n_layer=48, n_head=25, n_embd=1600)
    if preset == "mini":
        return GPTConfig(vocab_size=50257, block_size=1024, n_layer=4, n_head=8, n_embd=512)
    raise ValueError(f"unknown preset: {preset}")


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        cfg: GPTConfig,
        backend: str,
        *,
        window_size: int = 0,
    ):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.head_dim
        self.backend = backend
        self.window_size = int(window_size)

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()

        y = ATTN_BACKENDS[self.backend](
            q,
            k,
            v,
            is_causal=True,
            window_size=self.window_size,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(
        self,
        cfg: GPTConfig,
        backend: str,
        *,
        window_size: int = 0,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(
            cfg,
            backend,
            window_size=window_size,
        )
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        cfg: GPTConfig,
        backend: str,
        *,
        window_size: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList(
            [
                Block(
                    cfg,
                    backend,
                    window_size=window_size,
                )
                for _ in range(cfg.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        assert T <= self.cfg.block_size, f"seq_len {T} exceeds block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


@dataclass
class GPTNeoXConfig:
    """GPT-NeoX configuration for the public Pythia model shapes."""

    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    intermediate_size: int
    rotary_pct: float = 0.25
    rotary_base: float = 10_000.0
    layer_norm_eps: float = 1e-5
    bias: bool = True

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head

    @property
    def rotary_dim(self) -> int:
        dimension = int(self.head_dim * self.rotary_pct)
        if dimension % 2:
            raise ValueError("rotary dimension must be even")
        return dimension


def pythia_config(preset: str) -> GPTNeoXConfig:
    if preset == "pythia-160m":
        return GPTNeoXConfig(
            vocab_size=50304,
            block_size=2048,
            n_layer=12,
            n_head=12,
            n_embd=768,
            intermediate_size=3072,
        )
    if preset == "pythia-1.4b":
        return GPTNeoXConfig(
            vocab_size=50304,
            block_size=2048,
            n_layer=24,
            n_head=16,
            n_embd=2048,
            intermediate_size=8192,
        )
    if preset == "pythia-160m-d128":
        # Exact Pythia-160M parameter geometry with half as many attention
        # heads. This D=128 shape control differs from the released
        # checkpoint's D=64 attention configuration.
        return GPTNeoXConfig(
            vocab_size=50304,
            block_size=2048,
            n_layer=12,
            n_head=6,
            n_embd=768,
            intermediate_size=3072,
        )
    if preset == "pythia-410m-d128":
        # Pythia-410M parameter geometry with eight rather than sixteen
        # heads. Head count does not change its projection matrix shapes.
        return GPTNeoXConfig(
            vocab_size=50304,
            block_size=2048,
            n_layer=24,
            n_head=8,
            n_embd=1024,
            intermediate_size=4096,
        )
    raise ValueError(f"unknown preset: {preset}")


def _rope_cache(
    sequence_length: int,
    rotary_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    embeddings = torch.cat((angles, angles), dim=-1)
    return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


class GPTNeoXAttention(nn.Module):
    def __init__(self, config: GPTNeoXConfig, backend: str):
        super().__init__()
        self.num_heads = config.n_head
        self.hidden_size = config.n_embd
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.backend = backend
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, hidden = hidden_states.shape
        q, k, v = self.qkv(hidden_states).split(self.hidden_size, dim=-1)
        q = q.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary(q, cosine, sine, self.rotary_dim).contiguous()
        k = apply_rotary(k, cosine, sine, self.rotary_dim).contiguous()
        v = v.contiguous()
        attended = ATTN_BACKENDS[self.backend](q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, hidden)
        return self.output(attended)


class GPTNeoXBlock(nn.Module):
    def __init__(self, config: GPTNeoXConfig, backend: str):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = GPTNeoXAttention(config, backend)
        self.mlp_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        return (
            hidden_states
            + self.attn(self.attention_norm(hidden_states), cosine, sine)
            + self.mlp(self.mlp_norm(hidden_states))
        )


class GPTNeoX(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
        backend: str,
        *,
        window_size: int = 0,
    ):
        super().__init__()
        if window_size:
            raise NotImplementedError(
                "the Pythia/GPT-NeoX benchmark supports dense causal attention only"
            )
        self.cfg = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            GPTNeoXBlock(config, backend) for _ in range(config.n_layer)
        )
        self.final_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length > self.cfg.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block size {self.cfg.block_size}"
            )
        hidden_states = self.embedding(tokens)
        cosine, sine = _rope_cache(
            sequence_length,
            self.cfg.rotary_dim,
            self.cfg.rotary_base,
            tokens.device,
            hidden_states.dtype,
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, cosine, sine)
        return self.output(self.final_norm(hidden_states))


MODEL_PRESETS = (
    "mini",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "pythia-160m",
    "pythia-160m-d128",
    "pythia-410m-d128",
    "pythia-1.4b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "smollm2-135m",
    "smollm2-360m",
    "qwen2.5-1.5b",
    "granite-3.1-1b-a400m",
)
SOPHIA_PRESETS = tuple(
    preset
    for preset in MODEL_PRESETS
    if preset not in {"qwen2.5-1.5b", "granite-3.1-1b-a400m"}
)


def model_from_preset(preset: str):
    if preset.startswith("pythia-"):
        return pythia_config(preset), GPTNeoX
    if preset.startswith(("llama-", "smollm2-")):
        from benchmarks.models.gqa import Llama, llama_config

        return llama_config(preset), Llama
    if preset in {"qwen2.5-1.5b", "granite-3.1-1b-a400m"}:
        from benchmarks.models.gqa import transformers_config, TransformersCausalLM

        return transformers_config(preset), TransformersCausalLM
    return gpt2_config(preset), GPT


def model_backends(preset: str) -> tuple[str, ...]:
    """Backends with the tensor semantics required by one model family."""

    if preset in {"qwen2.5-1.5b", "granite-3.1-1b-a400m"}:
        return ("bob", "math")
    if preset.startswith(("llama-", "smollm2-")):
        return ("bob", "flash", "math", "hvp_manual", "hvp_semi_manual")
    return tuple(ATTN_BACKENDS)



DEFAULT_REFS = {
    "first": ("flash",),
    "second": ("math", "hvp_manual", "hvp_semi_manual"),
}
SWEEP_FIELDNAMES = [
    "preset",
    "order",
    "batch_size",
    "device",
    "dtype",
    "seq_len",
    "candidate",
    "ref",
    "candidate_status",
    "ref_status",
    "candidate_ms",
    "ref_ms",
    "speedup",
    "candidate_peak_mib",
    "ref_peak_mib",
    "mem_ratio",
    "total_params",
    "attn_params",
    "non_attn_params",
    "attn_pct",
    "candidate_warmup_iters",
    "candidate_rep_iters",
    "ref_warmup_iters",
    "ref_rep_iters",
    "error",
    "source_sha256",
    "git_diff_sha256",
    "git_status_sha256",
    *PROVENANCE_FIELDS,
]


def make_model(
    cfg,
    backend: str,
    device: str,
    dtype: torch.dtype,
    seed: int,
    *,
    window_size: int = 0,
    model_factory=GPT,
) -> nn.Module:
    """Build one deterministic benchmark arm without resident peer models."""

    torch.manual_seed(seed)
    model = model_factory(
        cfg,
        backend,
        window_size=window_size,
    )
    model.to(device=device, dtype=dtype).eval()
    model._benchmark_named_params = tuple(benchmark_named_params(model))
    return model


def make_lm_batch(batch_size: int, seq_len: int, vocab_size: int, device: str, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    tokens = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len + 1),
        device=device,
        dtype=torch.long,
        generator=g,
    )
    idx = tokens[:, :-1]
    targets = tokens[:, 1:]
    return idx, targets


def ce_loss(model: nn.Module, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(idx)
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B * T, V).float(), targets.reshape(B * T))


def checksum(tensors) -> torch.Tensor:
    total = None
    for t in tensors:
        val = t.reshape(-1)[0].float()
        total = val if total is None else total + val
    return total


def make_probe(named_params, seed: int):
    g = torch.Generator(device=named_params[0][1].device)
    g.manual_seed(seed)
    return {
        alias_key: torch.randn(
            param.shape, device=param.device, dtype=param.dtype, generator=g
        )
        for _, param, alias_key in named_params
    }


def benchmark_named_params(model: nn.Module):
    named = getattr(model, "_benchmark_named_params", None)
    if named is None:
        return [
            (name, param, param_alias_key(param))
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
    return [(name, param, alias_key) for name, param, alias_key in named if param.requires_grad]


def param_alias_key(param: torch.Tensor):
    storage = param.untyped_storage()
    return (
        storage.data_ptr(),
        param.storage_offset(),
        tuple(param.shape),
        tuple(param.stride()),
        param.dtype,
        param.device.type,
        param.device.index,
    )


def grad_from_named_params(
    output: torch.Tensor,
    named_params,
    *,
    retain_graph: bool,
    create_graph: bool,
    context: str,
    zero_unused: bool = False,
):
    params = tuple(param for _, param, _ in named_params)
    grads = torch.autograd.grad(
        output,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    if all(grad is not None for grad in grads):
        return named_params, list(grads)
    if zero_unused:
        grads = [
            torch.zeros_like(param) if grad is None else grad
            for (_, param, _), grad in zip(named_params, grads)
        ]
        return named_params, grads

    used_aliases = {
        alias_key
        for (_, _, alias_key), grad in zip(named_params, grads)
        if grad is not None
    }
    kept_named_params = []
    kept_grads = []
    unexpected = []
    for (name, param, alias_key), grad in zip(named_params, grads):
        if grad is not None:
            kept_named_params.append((name, param, alias_key))
            kept_grads.append(grad)
        elif param.numel() == 0 or alias_key in used_aliases:
            continue
        else:
            unexpected.append(name)

    if unexpected:
        details = ", ".join(unexpected[:8])
        if len(unexpected) > 8:
            details += ", ..."
        raise RuntimeError(f"{context}: unexpected unused params: {details}")

    return kept_named_params, kept_grads


def first_order_step(model: nn.Module, idx: torch.Tensor, targets: torch.Tensor):
    params = [param for _, param, _ in benchmark_named_params(model)]
    loss = ce_loss(model, idx, targets)
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    return checksum(grads)


def second_order_step(model: nn.Module, idx: torch.Tensor, targets: torch.Tensor, probe):
    named_params = benchmark_named_params(model)
    loss = ce_loss(model, idx, targets)
    named_params, grads = grad_from_named_params(
        loss,
        named_params,
        retain_graph=True,
        create_graph=True,
        context="second_order_step(first_grad)",
    )

    active_probe = [probe[alias_key] for _, _, alias_key in named_params]
    proj = sum((grad * vector).sum() for grad, vector in zip(grads, active_probe))
    _, second = grad_from_named_params(
        proj,
        named_params,
        retain_graph=False,
        create_graph=False,
        context="second_order_step(hvp)",
        zero_unused=True,
    )
    return checksum(second)


@torch.no_grad()
def parameter_summary(model: nn.Module):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attn = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and (".attn." in name or ".self_attn." in name)
    )
    return {
        "total": total,
        "attn": attn,
        "non_attn": total - attn,
        "attn_pct": 100.0 * attn / total,
    }


def run_one(model, idx, targets, order: str, probe_seed: int, warmup_ms: int, rep_ms: int):
    if order == "first":
        fn = lambda: first_order_step(model, idx, targets)
    else:
        probe = make_probe(benchmark_named_params(model), probe_seed)
        fn = lambda: second_order_step(model, idx, targets, probe)

    device = idx.device
    _ = fn()
    sync_device(device)
    sample_start = time.perf_counter()
    _ = fn()
    sync_device(device)
    sample_ms = max(1e-3, (time.perf_counter() - sample_start) * 1000.0)
    warmup = max(1, min(25, round(warmup_ms / sample_ms)))
    iters = max(3, min(100, round(rep_ms / sample_ms)))
    ms, peak_mib = time_and_peak(fn, warmup=warmup, iters=iters, device=device)

    return {
        "ms": ms,
        "peak_mb": peak_mib,
        "warmup_iters": warmup,
        "rep_iters": iters,
    }


def _run_backend(
    cfg,
    backend,
    idx,
    targets,
    order,
    probe_seed,
    warmup_ms,
    rep_ms,
    device,
    dtype,
    seed,
    model_factory,
):
    model = None
    try:
        model = make_model(
            cfg, backend, device, dtype, seed, model_factory=model_factory
        )
        return run_one(model, idx, targets, order, probe_seed, warmup_ms, rep_ms)
    finally:
        del model
        cleanup_device(torch.device(device))


def run_sweep(
    cfg,
    seq_lens,
    batch_size: int,
    order: str,
    candidate_backend: str,
    refs: list[str],
    device: str,
    dtype: torch.dtype,
    seed: int,
    warmup_ms: int,
    rep_ms: int,
    ref_max_seq_len: dict[str, int] | None = None,
    model_factory=GPT,
):
    provenance = {
        **runtime_provenance(torch.device(device)),
        **source_provenance(Path(__file__).resolve().parents[2]),
    }

    with torch.device("meta"):
        summary_model = model_factory(cfg, candidate_backend)
    summary = parameter_summary(summary_model)
    del summary_model
    print(f"total_params={summary['total']:,}")
    print(f"attn_params={summary['attn']:,} ({summary['attn_pct']:.2f}%)")
    print(f"non_attn_params={summary['non_attn']:,}")

    ref_cap_note = ("  (" + ", ".join(f"{r} capped at N={n}" for r, n in ref_max_seq_len.items()) + ")"
                    ) if ref_max_seq_len else ""
    print(f"\norder={order}  {candidate_backend}  vs  {', '.join(refs)}{ref_cap_note}")
    print(
        f"{'N':>6}  {'ref':>16}  {'ref_ms':>10}  {'cand_ms':>10}  "
        f"{'speedup':>8}  {'ref_mb':>10}  {'cand_mb':>10}  {'mem_ratio':>10}"
    )
    print("-" * 94)

    rows = []

    def base_row(seq_len, ref):
        return {
            "seq_len": seq_len,
            "candidate": candidate_backend,
            "ref": ref,
            "candidate_status": "pending",
            "ref_status": "pending",
            "candidate_ms": None,
            "ref_ms": None,
            "speedup": None,
            "candidate_peak_mib": None,
            "ref_peak_mib": None,
            "mem_ratio": None,
            "total_params": summary["total"],
            "attn_params": summary["attn"],
            "non_attn_params": summary["non_attn"],
            "attn_pct": summary["attn_pct"],
            "candidate_warmup_iters": None,
            "candidate_rep_iters": None,
            "ref_warmup_iters": None,
            "ref_rep_iters": None,
            "error": None,
            **provenance,
        }

    for i, seq_len in enumerate(seq_lens):
        batch_seed = seed + i
        probe_seed = seed + 10_000 + i

        idx, targets = make_lm_batch(batch_size, seq_len, cfg.vocab_size, device, batch_seed)

        candidate_stats, candidate_error = run_with_oom_capture(
            lambda: _run_backend(
                cfg,
                candidate_backend,
                idx,
                targets,
                order,
                probe_seed,
                warmup_ms,
                rep_ms,
                device,
                dtype,
                seed,
                model_factory,
            ),
            device=torch.device(device),
        )
        if candidate_error is not None:
            for ref in refs:
                row = base_row(seq_len, ref)
                row.update(
                    candidate_status="oom",
                    ref_status="skipped",
                    error=f"candidate:{candidate_error}",
                )
                rows.append(row)
            del idx, targets
            cleanup_device(torch.device(device))
            continue

        for ref in refs:
            if ref_max_seq_len and seq_len > ref_max_seq_len.get(ref, float("inf")):
                row = base_row(seq_len, ref)
                row.update(
                    candidate_status="ok",
                    ref_status="capped",
                    candidate_ms=candidate_stats["ms"],
                    candidate_peak_mib=candidate_stats["peak_mb"],
                    candidate_warmup_iters=candidate_stats["warmup_iters"],
                    candidate_rep_iters=candidate_stats["rep_iters"],
                )
                rows.append(row)
                print(
                    f"{seq_len:6d}  {ref:>16}  {'N/A':>10}  "
                    f"{candidate_stats['ms']:10.4f}  {'N/A':>8}  "
                    f"{'N/A':>10}  {candidate_stats['peak_mb']:10.2f}  {'N/A':>10}"
                )
                continue

            ref_stats, ref_error = run_with_oom_capture(
                lambda ref=ref: _run_backend(
                    cfg,
                    ref,
                    idx,
                    targets,
                    order,
                    probe_seed,
                    warmup_ms,
                    rep_ms,
                    device,
                    dtype,
                    seed,
                    model_factory,
                ),
                device=torch.device(device),
            )
            if ref_error is not None:
                row = base_row(seq_len, ref)
                row.update(
                    candidate_status="ok",
                    ref_status="oom",
                    candidate_ms=candidate_stats["ms"],
                    candidate_peak_mib=candidate_stats["peak_mb"],
                    candidate_warmup_iters=candidate_stats["warmup_iters"],
                    candidate_rep_iters=candidate_stats["rep_iters"],
                    error=f"ref_{ref}:{ref_error}",
                )
                rows.append(row)
                continue
            speedup = ref_stats["ms"] / candidate_stats["ms"]
            mem_ratio = (
                ref_stats["peak_mb"] / candidate_stats["peak_mb"]
                if candidate_stats["peak_mb"] > 0
                else float("nan")
            )

            row = base_row(seq_len, ref)
            row.update(
                candidate_status="ok",
                ref_status="ok",
                candidate_ms=candidate_stats["ms"],
                ref_ms=ref_stats["ms"],
                speedup=speedup,
                candidate_peak_mib=candidate_stats["peak_mb"],
                ref_peak_mib=ref_stats["peak_mb"],
                mem_ratio=mem_ratio,
                candidate_warmup_iters=candidate_stats["warmup_iters"],
                candidate_rep_iters=candidate_stats["rep_iters"],
                ref_warmup_iters=ref_stats["warmup_iters"],
                ref_rep_iters=ref_stats["rep_iters"],
            )
            rows.append(row)

            print(
                f"{seq_len:6d}  {ref:>16}  "
                f"{ref_stats['ms']:10.4f}  "
                f"{candidate_stats['ms']:10.4f}  "
                f"{speedup:8.2f}  "
                f"{ref_stats['peak_mb']:10.2f}  "
                f"{candidate_stats['peak_mb']:10.2f}  "
                f"{mem_ratio:10.2f}"
            )

        del idx, targets
        cleanup_device(torch.device(device))

    return rows


def annotate_sweep_rows(rows, *, preset: str, order: str, batch_size: int, device: str, dtype: torch.dtype):
    for row in rows:
        row["preset"] = preset
        row["order"] = order
        row["batch_size"] = batch_size
        row["device"] = torch.device(device).type
        row["dtype"] = dtype_name(dtype)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preset", type=str, default="gpt2", choices=MODEL_PRESETS
    )
    p.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--order", type=str, default="second", choices=["first", "second"])
    p.add_argument(
        "--candidate-backend",
        type=str,
        default="bob",
        choices=tuple(ATTN_BACKENDS),
        help="Backend to benchmark as the candidate implementation.",
    )
    p.add_argument(
        "--refs",
        type=str,
        default=None,
        help=(
            "Comma-separated reference backends to compare the candidate backend against. "
            "Defaults: first-order='flash'; second-order='math,hvp_manual,hvp_semi_manual'. "
            f"Choices: {','.join(ATTN_BACKENDS)}."
        ),
    )
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--warmup-ms", type=int, default=25)
    p.add_argument("--rep-ms", type=int, default=100)
    p.add_argument(
        "--ref-max-seq-len",
        type=str,
        action="append",
        default=None,
        metavar="REF=N",
        help="Cap a specific ref backend at seq_len N (e.g. --ref-max-seq-len math=1024). Repeatable.",
    )
    p.add_argument("--csv", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = args.device

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    cfg, model_factory = model_from_preset(args.preset)
    supported_backends = set(model_backends(args.preset))
    if args.candidate_backend not in supported_backends:
        raise ValueError(
            f"{args.preset} does not support backend {args.candidate_backend!r}; "
            f"choose from {sorted(supported_backends)}"
        )
    if args.vocab_size is not None:
        cfg.vocab_size = args.vocab_size
    cfg.block_size = max(seq_lens)
    dtype = getattr(torch, args.dtype)

    if args.refs is None:
        refs = [
            reference
            for reference in DEFAULT_REFS[args.order]
            if reference != args.candidate_backend and reference in supported_backends
        ]
        if not refs:
            refs = [
                backend
                for backend in ("bob", "math", "flash")
                if backend != args.candidate_backend and backend in supported_backends
            ][:1]
    else:
        refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    unknown = [
        reference
        for reference in refs
        if reference not in supported_backends or reference == args.candidate_backend
    ]
    if unknown:
        raise ValueError(f"unknown/forbidden ref backends: {unknown}")
    if not refs:
        raise ValueError("no reference backends remain after excluding the candidate backend")
    # sdpa_flash_public explicitly requests native FlashAttention and requires
    # CUDA. Reject unsupported benchmark arms before starting the sweep.
    if not device.startswith("cuda") and "flash" in {args.candidate_backend, *refs}:
        raise ValueError(
            "the explicit 'flash' backend requires a CUDA device; "
            f"got --device {device!r} (use math or a bob-family backend)"
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows = run_sweep(
        cfg=cfg,
        seq_lens=seq_lens,
        batch_size=args.batch_size,
        order=args.order,
        candidate_backend=args.candidate_backend,
        refs=refs,
        device=device,
        dtype=dtype,
        seed=args.seed,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        ref_max_seq_len={
            r: int(n)
            for entry in (args.ref_max_seq_len or [])
            for r, n in [entry.split("=", 1)]
        } or None,
        model_factory=model_factory,
    )
    annotate_sweep_rows(
        rows,
        preset=args.preset,
        order=args.order,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
    )
    if args.csv:
        write_rows(rows, path=args.csv, fieldnames=SWEEP_FIELDNAMES)


if __name__ == "__main__":
    main()
