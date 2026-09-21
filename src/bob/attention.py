"""Attention forward and first backward for equal query and key lengths.

PyTorch supplies the forward and the first backward. This module wraps them and
returns the quantities the FlashBoB second backward consumes: the attention
output `O`, the log-sum-exp `L`, and the first-order gradients `dQ`, `dK`, and
`dV`. It contains no BoB variant selection.

CUDA calls private `torch.ops.aten` entry points. Dense attention uses cuDNN
SDPA or aten FlashAttention, chosen by compute capability; see
`CUDNN_MIN_COMPUTE_CAPABILITY`. Sliding-window attention uses
`_efficient_attention_forward`, which accepts a window argument. CPU calls
PyTorch's math SDPA and does not enter the custom autograd bridge.

The `torch.ops.aten` entry points are private and their positional signatures
vary across PyTorch releases. A changed signature produces incorrect gradients
without raising an exception. Supported versions are fixed by the pinned
environment.

Rectangular attention is implemented in `bob.kernels.rectangular`.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import CausalVariant

from bob.kernels.common import normalize_window_size

# aten's window support rides on the "upper left" causal variant
SWA_MASK_TYPE = int(CausalVariant.UPPER_LEFT)

# Which fused operator supplies the dense forward and first backward. cuDNN's
# backward was faster than aten flash in historical H100 and B200 measurements,
# although this checkout does not contain their artifacts. On SM89, aten flash
# is 15% to 25% faster on the first backward and 5% to 11% faster on the full
# second-order step. The front end is therefore selected by compute capability.
# The numerical comparison and evidence limits are recorded in docs/support.md.
CUDNN_MIN_COMPUTE_CAPABILITY = (9, 0)


@lru_cache(maxsize=None)
def dense_front_end_is_cudnn(device_index: int) -> bool:
    """Whether dense attention on this device should use cuDNN over aten flash."""
    return torch.cuda.get_device_capability(device_index) >= CUDNN_MIN_COMPUTE_CAPABILITY


@dataclass
class FlashAux:
    """Forward outputs and aten state required by the first backward.

    `lse` is the backend's log-sum-exp tensor. cuDNN may return
    `[B, H, M, 1]`; `lse_rows` converts it to the `[B, H, M]` FP32
    natural-log form every FlashBoB kernel consumes.
    """

    out: torch.Tensor
    lse: torch.Tensor
    cum_seq_q: torch.Tensor | None
    cum_seq_k: torch.Tensor | None
    max_q: int
    max_k: int
    philox_seed: torch.Tensor
    philox_offset: torch.Tensor

    def meta(self) -> tuple:
        """The non-output state, as the opaque tuple the autograd bridges carry."""
        return (self.cum_seq_q, self.cum_seq_k, self.max_q, self.max_k,
                self.philox_seed, self.philox_offset)


def lse_rows(lse: torch.Tensor, m: int) -> torch.Tensor:
    """`[B, H, M]` FP32 natural-log statistic from any backend's `lse` tensor."""
    if lse.ndim == 4:
        lse = lse.squeeze(-1)
    return lse[..., :m].contiguous().float()


def aux_from_meta(out, lse, meta) -> FlashAux:
    cum_q, cum_k, max_q, max_k, seed, offset = meta
    return FlashAux(out=out, lse=lse, cum_seq_q=cum_q, cum_seq_k=cum_k, max_q=int(max_q),
                    max_k=int(max_k), philox_seed=seed, philox_offset=offset)


def effective_window(seq_len: int, window_size: int | None, *, is_causal: bool) -> int:
    """Window size reduced to 0 when it covers the whole sequence.

    A causal window at least as wide as the sequence is equivalent to causal
    masking, and 0 selects the dense causal path.
    """
    window_size = normalize_window_size(window_size)
    if is_causal and window_size >= seq_len:
        return 0
    return window_size


def build_swa_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
    *,
    is_causal: bool = True,
) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    row, col = idx[:, None], idx[None, :]
    if not is_causal:
        if window_size > 0:
            raise NotImplementedError("window_size is only implemented for causal attention")
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    mask = row >= col
    if window_size > 0:
        mask = mask & ((row - col) < window_size)
    return mask


def sdpa_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: int = 0,
) -> torch.Tensor:
    """PyTorch math SDPA with the requested square mask."""
    window_size = effective_window(q.shape[-2], window_size, is_causal=is_causal)
    attn_bias = None
    if window_size > 0:
        mask = build_swa_mask(q.shape[-2], window_size, q.device, is_causal=is_causal)
        attn_bias = torch.zeros(mask.shape, device=q.device, dtype=q.dtype)
        attn_bias = attn_bias.masked_fill(~mask, float("-inf"))
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=is_causal and window_size == 0,
            scale=scale,
        )


def _cudnn_forward(q, k, v, *, is_causal, scale) -> FlashAux:
    """cuDNN fused attention for dense MHA, GQA, and MQA."""
    out, lse, cum_q, cum_k, max_q, max_k, seed, offset, _ = torch.ops.aten._scaled_dot_product_cudnn_attention.default(
        q, k, v, None, True, 0.0, is_causal, False, scale=scale,
    )
    return FlashAux(
        out=out, lse=lse, cum_seq_q=cum_q, cum_seq_k=cum_k, max_q=int(max_q),
        max_k=int(max_k), philox_seed=seed, philox_offset=offset,
    )


def _cudnn_backward(grad_out, q, k, v, aux, *, is_causal, scale):
    return torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(
        grad_out, q, k, v, aux.out, aux.lse, aux.philox_seed, aux.philox_offset, None,
        aux.cum_seq_q, aux.cum_seq_k, aux.max_q, aux.max_k, 0.0, is_causal, scale=scale,
    )


def _flash_forward(q, k, v, *, is_causal, scale) -> FlashAux:
    """aten's FlashAttention for dense MHA, GQA, and MQA."""
    out, lse, cum_q, cum_k, max_q, max_k, seed, offset, _ = torch.ops.aten._scaled_dot_product_flash_attention.default(
        q, k, v, dropout_p=0.0, is_causal=is_causal, return_debug_mask=False, scale=scale,
    )
    return FlashAux(
        out=out, lse=lse, cum_seq_q=cum_q, cum_seq_k=cum_k, max_q=int(max_q),
        max_k=int(max_k), philox_seed=seed, philox_offset=offset,
    )


def _flash_backward(grad_out, q, k, v, aux, *, is_causal, scale):
    return torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        grad_out, q, k, v, aux.out, aux.lse, aux.cum_seq_q, aux.cum_seq_k,
        aux.max_q, aux.max_k, 0.0, is_causal, aux.philox_seed, aux.philox_offset,
        scale=scale,
    )[:3]


def _swa_forward(q, k, v, *, is_causal, window_size, scale) -> FlashAux:
    if not is_causal:
        raise NotImplementedError("native sliding-window attention is only implemented for causal attention")
    # this aten op wants [B, N, H, D]
    q_n, k_n, v_n = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    out, lse, seed, offset, max_q, max_k = torch.ops.aten._efficient_attention_forward(
        q_n,
        k_n,
        v_n,
        None,
        None,
        None,
        None,
        None,
        0.0,
        SWA_MASK_TYPE,
        True,
        scale=scale,
        seqlen_k=None,
        window_size=window_size,
    )
    return FlashAux(
        out=out.transpose(1, 2).contiguous(),
        lse=lse,
        cum_seq_q=None,
        cum_seq_k=None,
        max_q=int(max_q),
        max_k=int(max_k),
        philox_seed=seed,
        philox_offset=offset,
    )


def _swa_backward(grad_out, q, k, v, aux, *, is_causal, window_size, scale):
    if not is_causal:
        raise NotImplementedError("native sliding-window attention is only implemented for causal attention")
    g_n, q_n, k_n, v_n, o_n = (t.transpose(1, 2).contiguous() for t in (grad_out, q, k, v, aux.out))
    dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
        g_n,
        q_n,
        k_n,
        v_n,
        None,
        o_n,
        None,
        None,
        aux.max_q,
        aux.max_k,
        aux.lse,
        0.0,
        aux.philox_seed,
        aux.philox_offset,
        SWA_MASK_TYPE,
        False,
        scale=scale,
        num_splits_key=None,
        window_size=window_size,
        shared_storage_dqdkdv=False,
    )
    return tuple(t.transpose(1, 2).contiguous() for t in (dq, dk, dv))


def forward(q, k, v, *, is_causal, window_size, scale) -> FlashAux:
    """Attention forward for the CUDA custom-autograd path."""
    window_size = normalize_window_size(window_size)
    if q.device.type != "cuda":
        raise NotImplementedError("the FlashBoB custom-autograd path requires CUDA")
    if window_size > 0:
        return _swa_forward(
            q,
            k,
            v,
            is_causal=is_causal,
            window_size=window_size,
            scale=scale,
        )
    if dense_front_end_is_cudnn(q.device.index):
        return _cudnn_forward(q, k, v, is_causal=is_causal, scale=scale)
    return _flash_forward(q, k, v, is_causal=is_causal, scale=scale)


def first_backward(grad_out, q, k, v, aux, *, is_causal, window_size, scale):
    """First-order dQ, dK, and dV corresponding to ``forward``."""
    window_size = normalize_window_size(window_size)
    if q.device.type != "cuda":
        raise NotImplementedError("the FlashBoB custom-autograd path requires CUDA")
    if window_size > 0:
        return _swa_backward(
            grad_out,
            q,
            k,
            v,
            aux,
            is_causal=is_causal,
            window_size=window_size,
            scale=scale,
        )
    # must match whichever operator produced aux in forward
    if dense_front_end_is_cudnn(q.device.index):
        return _cudnn_backward(grad_out, q, k, v, aux, is_causal=is_causal, scale=scale)
    return _flash_backward(grad_out, q, k, v, aux, is_causal=is_causal, scale=scale)
