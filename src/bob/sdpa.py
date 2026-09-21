"""Scaled dot-product attention with a Triton FlashBoB second backward."""

import math
import numbers
import operator

import torch
from torch.autograd import Function

from bob import attention
from bob.attention import effective_window, sdpa_math
from bob.kernels import gqa, rectangular, square

__all__ = ["sdpa_bob"]

CUDA_DTYPES = (torch.float16, torch.bfloat16)
CUDA_HEAD_DIMS = (32, 64, 128)
CPU_DTYPES = (*CUDA_DTYPES, torch.float32)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    scale: float | None,
    window_size: int,
    q_offset: int | None,
    k_offset: int,
) -> int:
    """Validate the complete public contract before selecting a backend."""
    if torch._C._are_functorch_transforms_active():
        raise NotImplementedError(
            "torch.func transforms are unsupported; use torch.autograd reverse "
            "mode through second order"
        )

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None:
            raise NotImplementedError(
                "forward-mode AD is unsupported; use torch.autograd reverse "
                "mode through second order"
            )
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must be rank 4 [B, H, N, D], got shape {tuple(tensor.shape)}"
            )

    if any(dimension <= 0 for tensor in (q, k, v) for dimension in tensor.shape):
        raise ValueError(
            "q, k, and v dimensions must all be positive, got "
            f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if v.shape != k.shape:
        raise ValueError(
            f"v shape {tuple(v.shape)} must exactly match k shape {tuple(k.shape)}"
        )
    if q.shape[0] != k.shape[0]:
        raise ValueError(
            "q, k, and v must have the same batch size, got "
            f"{q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            "q, k, and v must have the same head dimension, got "
            f"{q.shape[-1]}, {k.shape[-1]}, and {v.shape[-1]}"
        )
    if q.shape[1] % k.shape[1]:
        raise ValueError(
            f"query heads H_Q={q.shape[1]} must be divisible by KV heads H_KV={k.shape[1]}"
        )
    if not (q.device == k.device == v.device):
        raise ValueError(
            "q, k, and v must be on the same device, got "
            f"q={q.device} k={k.device} v={v.device}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            "q, k, and v must have the same dtype, got "
            f"q={q.dtype} k={k.dtype} v={v.dtype}"
        )

    supported_dtypes = CUDA_DTYPES if q.device.type == "cuda" else CPU_DTYPES
    if q.dtype not in supported_dtypes:
        names = ", ".join(str(dtype) for dtype in supported_dtypes)
        raise TypeError(f"q, k, and v dtype must be one of {{{names}}}, got {q.dtype}")
    if q.device.type == "cuda" and q.shape[-1] not in CUDA_HEAD_DIMS:
        raise ValueError(
            f"CUDA head dimension D must be one of {CUDA_HEAD_DIMS}, got {q.shape[-1]}"
        )

    if not isinstance(is_causal, bool):
        raise TypeError(f"is_causal must be bool, got {type(is_causal).__name__}")
    window_size = _nonnegative_integer("window_size", window_size)
    if window_size > 0 and not is_causal:
        raise NotImplementedError("window_size>0 requires is_causal=True")
    _optional_nonnegative_integer("q_offset", q_offset)
    _nonnegative_integer("k_offset", k_offset)

    if scale is not None:
        if isinstance(scale, bool) or not isinstance(scale, numbers.Real):
            raise TypeError(f"scale must be a finite positive number, got {scale!r}")
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError(f"scale must be a finite positive number, got {scale!r}")
    return window_size


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {value!r}") from error
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _optional_nonnegative_integer(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(name, value)


def _check_gqa_support(q: torch.Tensor, k: torch.Tensor, *, window_size: int) -> None:
    if q.device.type != "cuda":
        raise NotImplementedError("FlashBoB GQA/MQA requires CUDA")
    if q.dtype != torch.bfloat16:
        raise TypeError("FlashBoB GQA/MQA requires bfloat16")
    if window_size > 0:
        raise NotImplementedError("FlashBoB GQA/MQA does not support sliding windows")
    if q.shape[1] // k.shape[1] > 32:
        raise NotImplementedError("FlashBoB GQA/MQA supports grouping factors up to 32")


def _first_backward(q, k, v, dO, out, lse, meta, *, q_offset, k_offset,
                    is_causal, window_size, scale):
    """Run the first attention backward on contiguous saved tensors."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    dO, out, lse = dO.contiguous(), out.contiguous(), lse.contiguous()
    if q_offset is not None:
        grads = rectangular.rect_first_backward(
            dO,
            q,
            k,
            v,
            out,
            lse,
            q_offset=q_offset,
            k_offset=k_offset,
            is_causal=is_causal,
            window_size=window_size,
            scale=scale,
        )
        return (q, k, v, dO, out, lse), grads

    aux = attention.aux_from_meta(out, lse, meta)
    grads = attention.first_backward(
        dO,
        q,
        k,
        v,
        aux,
        is_causal=is_causal,
        window_size=window_size,
        scale=scale,
    )
    return (q, k, v, dO, out, lse), grads


def _second_backward(q, k, v, out, lse, dO, dQ, ddQ, ddK, ddV, *,
                     q_offset, k_offset, is_causal, window_size, scale):
    zero_tangents = (ddQ is None, ddK is None, ddV is None)
    ddQ = q if ddQ is None else ddQ.contiguous()
    ddK = k if ddK is None else ddK.contiguous()
    ddV = v if ddV is None else ddV.contiguous()

    if q_offset is not None:
        return rectangular.bob_2pass_rect(
            q,
            k,
            v,
            out,
            lse,
            dO,
            ddQ,
            ddK,
            ddV,
            q_offset=q_offset,
            k_offset=k_offset,
            causal=is_causal,
            window_size=window_size,
            scale=scale,
            bf16_source_dots=q.dtype == torch.bfloat16,
            dQ=dQ,
            reuse_dq=True,
            zero_tangents=zero_tangents,
        )
    if q.shape[1] != k.shape[1]:
        return gqa.bob_gqa(
            q,
            k,
            v,
            out,
            lse,
            dO,
            ddQ,
            ddK,
            ddV,
            causal=is_causal,
            dQ=dQ.contiguous(),
            zero_tangents=zero_tangents,
            scale=scale,
        )
    return square.bob_2pass(
        q,
        k,
        v,
        out,
        lse,
        dO,
        ddQ,
        ddK,
        ddV,
        causal=is_causal,
        window_size=window_size,
        dQ=dQ.contiguous(),
        reuse_dq=True,
        bf16_source_dots=q.dtype == torch.bfloat16,
        zero_tangents=zero_tangents,
        scale=scale,
    )


class BoBSecondOrder(Function):
    """Differentiate PyTorch's first attention backward with Triton."""

    @staticmethod
    def forward(ctx, q, k, v, dO, out, lse, meta, q_offset, k_offset,
                is_causal, window_size, scale):
        ctx.set_materialize_grads(False)
        (q, k, v, dO, out, lse), (dQ, dK, dV) = _first_backward(
            q,
            k,
            v,
            dO,
            out,
            lse,
            meta,
            q_offset=q_offset,
            k_offset=k_offset,
            is_causal=is_causal,
            window_size=window_size,
            scale=scale,
        )
        if q_offset is None:
            lse = attention.lse_rows(lse, q.shape[-2])
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            dO,
            dQ,
        )
        ctx.q_offset = q_offset
        ctx.k_offset = k_offset
        ctx.is_causal = is_causal
        ctx.window_size = window_size
        ctx.scale = scale
        return dQ, dK, dV

    @staticmethod
    def backward(ctx, ddQ, ddK, ddV):
        q, k, v, out, lse, dO, dQ = ctx.saved_tensors
        grads = _second_backward(
            q,
            k,
            v,
            out,
            lse,
            dO,
            dQ,
            ddQ,
            ddK,
            ddV,
            q_offset=ctx.q_offset,
            k_offset=ctx.k_offset,
            is_causal=ctx.is_causal,
            window_size=ctx.window_size,
            scale=ctx.scale,
        )
        return (*grads, None, None, None, None, None, None, None, None)


class SDPAWithBoB(Function):
    @staticmethod
    def forward(ctx, q, k, v, q_offset, k_offset, is_causal, window_size, scale):
        if q_offset is None:
            aux = attention.forward(
                q,
                k,
                v,
                is_causal=is_causal,
                window_size=window_size,
                scale=scale,
            )
            out, lse, meta = aux.out, aux.lse, aux.meta()
        else:
            out, lse = rectangular.rect_forward(
                q,
                k,
                v,
                q_offset=q_offset,
                k_offset=k_offset,
                is_causal=is_causal,
                window_size=window_size,
                scale=scale,
            )
            meta = None
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.meta = meta
        ctx.q_offset = q_offset
        ctx.k_offset = k_offset
        ctx.is_causal = is_causal
        ctx.window_size = window_size
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dO):
        q, k, v, out, lse = ctx.saved_tensors
        if not torch.is_grad_enabled():
            dQ, dK, dV = _first_backward(
                q,
                k,
                v,
                dO,
                out,
                lse,
                ctx.meta,
                q_offset=ctx.q_offset,
                k_offset=ctx.k_offset,
                is_causal=ctx.is_causal,
                window_size=ctx.window_size,
                scale=ctx.scale,
            )[1]
            return dQ, dK, dV, None, None, None, None, None
        dQ, dK, dV = BoBSecondOrder.apply(
            q,
            k,
            v,
            dO,
            out,
            lse,
            ctx.meta,
            ctx.q_offset,
            ctx.k_offset,
            ctx.is_causal,
            ctx.window_size,
            ctx.scale,
        )
        return dQ, dK, dV, None, None, None, None, None


def _infer_q_offset(q_len: int, k_len: int, *, q_offset: int | None,
                    k_offset: int, is_causal: bool) -> int:
    if q_offset is not None:
        return q_offset
    if is_causal and q_len != k_len:
        return k_offset + max(0, k_len - q_len)
    return k_offset


def _effective_rect_window(q_len: int, q_offset: int, k_offset: int,
                           window_size: int) -> int:
    if window_size >= q_offset - k_offset + q_len:
        return 0
    return window_size


def sdpa_bob(
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
    """Compute SDPA with a FlashBoB second backward.

    Tensors use [batch, heads, sequence, head_dim] layout. CUDA supports FP16
    and BF16 with head dimensions 32, 64, or 128. Square BF16 attention also
    supports grouped and multi-query heads. Grouped heads cannot be combined
    with rectangular geometry or a sliding window.

    CPU calls PyTorch's differentiable math implementation. It is intended as
    a correctness fallback and accepts any positive head dimension.
    """
    window_size = _validate_inputs(
        q,
        k,
        v,
        is_causal=is_causal,
        scale=scale,
        window_size=window_size,
        q_offset=q_offset,
        k_offset=k_offset,
    )
    k_offset = operator.index(k_offset)
    q_offset = _infer_q_offset(
        q.shape[-2],
        k.shape[-2],
        q_offset=None if q_offset is None else operator.index(q_offset),
        k_offset=k_offset,
        is_causal=is_causal,
    )
    is_rectangular = q.shape[-2] != k.shape[-2] or q_offset != k_offset
    is_grouped = q.shape[1] != k.shape[1]

    if is_rectangular:
        if is_grouped:
            raise NotImplementedError(
                "FlashBoB GQA/MQA does not support rectangular attention"
            )
        window_size = _effective_rect_window(
            q.shape[-2], q_offset, k_offset, window_size
        )
        if q.device.type != "cuda":
            return rectangular.sdpa_math_rect(
                q,
                k,
                v,
                q_offset=q_offset,
                k_offset=k_offset,
                is_causal=is_causal,
                window_size=window_size,
                scale=scale,
            )
        return SDPAWithBoB.apply(
            q, k, v, q_offset, k_offset, is_causal, window_size, scale
        )

    window_size = effective_window(
        q.shape[-2], window_size, is_causal=is_causal
    )
    if is_grouped:
        _check_gqa_support(q, k, window_size=window_size)
    if q.device.type != "cuda":
        return sdpa_math(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=scale,
            window_size=window_size,
        )
    return SDPAWithBoB.apply(
        q, k, v, None, None, is_causal, window_size, scale
    )
