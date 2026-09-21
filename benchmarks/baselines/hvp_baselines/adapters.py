"""Adapters so HF-style HVP functions can be slotted into gpt_2.ATTN_BACKENDS.

Upstream signature (HuggingFace AttentionInterface):
    fn(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
       is_causal=None, **kwargs) -> (attn_output[B, T, H, D], None)

gpt_2.ATTN_BACKENDS signature:
    fn(q, k, v, is_causal=True) -> out[B, H, T, D]
"""
from types import SimpleNamespace

import torch

from hvp_baselines import hvp_manual, hvp_semi_manual


def _fake_module(is_causal: bool) -> SimpleNamespace:
    return SimpleNamespace(is_causal=is_causal)


def _wrap(hvp_fn):
    def backend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool = True, **kwargs) -> torch.Tensor:
        ws = kwargs.pop("window_size", 0)
        scale = kwargs.pop("scale", None)
        if ws not in (0, None):
            raise NotImplementedError(
                f"{hvp_fn.__name__} adapter does not support sliding-window attention (window_size={ws})"
            )
        out, _ = hvp_fn(
            _fake_module(is_causal),
            q, k, v,
            attention_mask=None,
            dropout=0.0,
            scaling=scale,
            is_causal=is_causal,
        )
        return out.transpose(1, 2).contiguous()
    backend.__name__ = f"sdpa_{hvp_fn.__name__}"
    return backend


sdpa_hvp_manual = _wrap(hvp_manual)
sdpa_hvp_semi_manual = _wrap(hvp_semi_manual)
