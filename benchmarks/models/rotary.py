"""Shared rotary-position operation for local benchmark models."""

import torch


def apply_rotary(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    rotary_dim = tensor.shape[-1] if rotary_dim is None else rotary_dim
    rotated, remainder = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
    first, second = rotated.chunk(2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    cosine = cosine[None, None, :, :]
    sine = sine[None, None, :, :]
    return torch.cat((rotated * cosine + rotated_half * sine, remainder), dim=-1)
