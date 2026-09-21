"""Shared attention benchmark operations.

The benchmark surface mirrors the package surface: one candidate operator,
``bob.sdpa_bob``, and one explicit PyTorch reference. Shape alone selects MHA,
GQA/MQA, or rectangular attention.
"""

import math
from collections.abc import Callable

import torch

from bob import sdpa_bob
from benchmarks.baselines import sdpa_hvp_manual, sdpa_hvp_semi_manual


OUTPUT_NAMES = ("out", "dQ", "dK", "dV", "Qbar", "Kbar", "Vbar", "dObar")


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: int = 0,
    q_offset: int | None = None,
    k_offset: int = 0,
) -> torch.Tensor:
    """Materialized SDPA reference for every supported geometry."""

    query_length, key_length = q.shape[-2], k.shape[-2]
    query_heads, key_heads = q.shape[1], k.shape[1]
    if query_heads % key_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if query_heads != key_heads:
        repeats = query_heads // key_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        q_offset = key_length - query_length if q_offset is None else q_offset
        query_positions = torch.arange(query_length, device=q.device) + q_offset
        key_positions = torch.arange(key_length, device=q.device) + k_offset
        distance = query_positions[:, None] - key_positions[None, :]
        mask = distance >= 0
        if window_size:
            mask &= distance < window_size
        scores = scores.masked_fill(~mask, float("-inf"))
    elif window_size:
        raise NotImplementedError("sliding windows require causal attention")
    return torch.softmax(scores, dim=-1).to(v.dtype) @ v


def make_case(
    *,
    batch: int,
    query_heads: int,
    key_heads: int,
    query_length: int,
    key_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(shape: tuple[int, ...], *, requires_grad: bool = False) -> torch.Tensor:
        return torch.randn(
            shape,
            dtype=dtype,
            device=device,
            generator=generator,
            requires_grad=requires_grad,
        )

    q_shape = (batch, query_heads, query_length, head_dim)
    kv_shape = (batch, key_heads, key_length, head_dim)
    inputs = (
        rand(q_shape, requires_grad=True),
        rand(kv_shape, requires_grad=True),
        rand(kv_shape, requires_grad=True),
        rand(q_shape, requires_grad=True),
    )
    bars = (rand(q_shape), rand(kv_shape), rand(kv_shape))
    return inputs, bars


def derivative_outputs(
    attention: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    bars: tuple[torch.Tensor, ...],
    *,
    order: str,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Return forward, first-gradient, and optionally second-gradient outputs."""

    q, k, v, grad_out = inputs
    out = attention(q, k, v, **kwargs)
    first = torch.autograd.grad(
        out,
        (q, k, v),
        grad_out,
        create_graph=order == "second",
    )
    values = (out, *first)
    names = OUTPUT_NAMES[:4]
    if order == "second":
        second = torch.autograd.grad(first, (q, k, v, grad_out), bars)
        values += second
        names = OUTPUT_NAMES
    return {name: value.detach() for name, value in zip(names, values)}


BACKENDS: dict[str, Callable[..., torch.Tensor]] = {
    "bob": sdpa_bob,
    "math": attention_reference,
    "hvp-manual": sdpa_hvp_manual,
    "hvp-semi-manual": sdpa_hvp_semi_manual,
}


__all__ = ["BACKENDS", "OUTPUT_NAMES", "attention_reference", "derivative_outputs", "make_case"]
