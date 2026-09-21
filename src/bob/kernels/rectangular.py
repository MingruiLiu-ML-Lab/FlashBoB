"""Rectangular attention with unequal query and key lengths at absolute offsets.

Applies to cross-attention and cached-prefix decoding, where the query block
covers rows `[q_offset, q_offset + M)` and the keys cover
`[k_offset, k_offset + N_KV)`. Masking is on absolute positions, so this module
passes offset-shifted indices into the shared tile bodies where the square path
passes tile-local indices.

The Triton kernels, the launcher, the flash-attn fast path, and the math
fallback are all defined here. `flash_attn` is imported inside the call that
uses it rather than at module scope. It is the only optional dependency in
`bob`, and a module-scope import would require it for callers that never
reach this path.

The rectangular path reuses the first-order $dQ = \\tau XK$, which the
rectangular first backward returns on both the flash-attn slice path and the
autograd fallback.

Inputs are consumed at storage precision. Outputs are FP32.
"""

import math

import torch
import triton
import triton.language as tl

from . import common
from .common import COL_PASS_CONFIGS, ROW_PASS_CONFIGS

FLASH_RECT_DTYPES = (torch.float16, torch.bfloat16)


def grad_via_autograd(forward_fn, grad_out, q, k, v):
    """First-order dQ, dK, and dV from differentiating `forward_fn`.

    `forward_fn` is applied to detached clones of q, k, and v, leaving the
    caller's autograd graph unchanged. Only the rectangular math fallback
    below uses it; it lives here so `kernels/` never imports the top-level package.
    """
    with torch.enable_grad():
        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        out = forward_fn(qr, kr, vr)
        return torch.autograd.grad(out, (qr, kr, vr), grad_out, retain_graph=False, create_graph=False)


# ---------------------------------------------------------------------------
# masks and math fallback
# ---------------------------------------------------------------------------


def build_rectangular_swa_mask(
    q_len: int,
    k_len: int,
    q_offset: int,
    k_offset: int,
    window_size: int,
    device: torch.device,
    *,
    is_causal: bool = True,
) -> torch.Tensor:
    row_abs = torch.arange(q_len, device=device) + int(q_offset)
    col_abs = torch.arange(k_len, device=device) + int(k_offset)
    if not is_causal:
        if window_size > 0:
            raise NotImplementedError("window_size>0 requires is_causal=True")
        return torch.ones((q_len, k_len), dtype=torch.bool, device=device)
    mask = row_abs[:, None] >= col_abs[None, :]
    if window_size > 0:
        mask = mask & ((row_abs[:, None] - col_abs[None, :]) < window_size)
    return mask


def sdpa_math_rect(q, k, v, *, q_offset, k_offset, is_causal, window_size, scale):
    """Masked SDPA over the rectangular block, using the math backend."""
    bias = None
    if is_causal or window_size > 0:
        mask = build_rectangular_swa_mask(
            q.shape[-2],
            k.shape[-2],
            q_offset,
            k_offset,
            window_size,
            q.device,
            is_causal=is_causal,
        )
        bias = torch.zeros(mask.shape, device=q.device, dtype=q.dtype)
        bias = bias.masked_fill(~mask, float("-inf")).view(1, 1, q.shape[-2], k.shape[-2])
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F

    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False, scale=scale)


# ---------------------------------------------------------------------------
# flash-attn fast path (imported lazily, only when eligible)
# ---------------------------------------------------------------------------


def flash_rect_eligible(q, k, v, *, q_offset, k_offset, is_causal, window_size) -> bool:
    """Whether the flash-attn slice path applies.

    The path requires the query block to fall entirely inside the key range, so
    that the rectangle reduces to causal attention over a contiguous K/V slice.
    """
    if not (is_causal and q.device.type == "cuda"):
        return False
    if q.dtype not in FLASH_RECT_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    B, H, M, D = q.shape
    Bk, Hk, N_KV, Dk = k.shape
    # flash-attn serves grouped and multi-query heads natively
    if (B, D) != (Bk, Dk) or Hk < 1 or H % Hk:
        return False
    if D > 256 or (D % 8) != 0 or window_size < 0:
        return False
    delta = int(q_offset) - int(k_offset)
    return 0 <= delta and delta + M <= N_KV


def flash_rect_slice_bounds(M, q_offset, k_offset, window_size) -> tuple[int, int]:
    delta = int(q_offset) - int(k_offset)
    j_lo = max(0, delta - (window_size - 1)) if window_size > 0 else 0
    return j_lo, delta + M


def _flash_ops():
    """Import the flash-attn forward and backward entry points.

    The import is local to this function, so `import bob` succeeds without
    flash-attn installed and only an eligible rectangular call requires it.
    """
    from flash_attn.flash_attn_interface import (
        _wrapped_flash_attn_backward as bwd,
        _wrapped_flash_attn_forward as fwd,
    )

    return fwd, bwd


def flash_rect_forward(q, k, v, *, q_offset, k_offset, window_size, scale):
    fwd, _ = _flash_ops()
    D = q.shape[-1]
    j_lo, j_hi = flash_rect_slice_bounds(q.shape[2], q_offset, k_offset, window_size)
    q_f = q.transpose(1, 2).contiguous()
    k_f = k[:, :, j_lo:j_hi, :].transpose(1, 2).contiguous()
    v_f = v[:, :, j_lo:j_hi, :].transpose(1, 2).contiguous()
    softmax_scale = float(scale) if scale is not None else (1.0 / math.sqrt(D))
    w_left, w_right = (window_size - 1, 0) if window_size > 0 else (-1, -1)
    out_f, lse, _, _ = fwd(q_f, k_f, v_f, 0.0, softmax_scale, True, w_left, w_right, 0.0, None, False)
    return out_f.transpose(1, 2).contiguous(), lse


def flash_rect_backward(dO, q, k, v, out, lse, *, q_offset, k_offset, window_size, scale):
    _, bwd = _flash_ops()
    D = q.shape[-1]
    j_lo, j_hi = flash_rect_slice_bounds(q.shape[2], q_offset, k_offset, window_size)
    q_f = q.transpose(1, 2).contiguous()
    k_f = k[:, :, j_lo:j_hi, :].transpose(1, 2).contiguous()
    v_f = v[:, :, j_lo:j_hi, :].transpose(1, 2).contiguous()
    out_f = out.transpose(1, 2).contiguous()
    dO_f = dO.transpose(1, 2).contiguous()
    dq_f, dk_f, dv_f = (torch.empty_like(t) for t in (q_f, k_f, v_f))
    softmax_scale = float(scale) if scale is not None else (1.0 / math.sqrt(D))
    w_left, w_right = (window_size - 1, 0) if window_size > 0 else (-1, -1)
    bwd(
        dO_f,
        q_f,
        k_f,
        v_f,
        out_f,
        lse,
        dq_f,
        dk_f,
        dv_f,
        0.0,
        softmax_scale,
        True,
        w_left,
        w_right,
        0.0,
        None,
        False,
        None,
    )
    # the slice only covered [j_lo, j_hi); the rest of K/V received no gradient
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dk[:, :, j_lo:j_hi, :] = dk_f.transpose(1, 2)
    dv[:, :, j_lo:j_hi, :] = dv_f.transpose(1, 2)
    return dq_f.transpose(1, 2).contiguous(), dk, dv


def math_forward_rect(q, k, v, *, q_offset, k_offset, is_causal, window_size, scale):
    """Rectangular forward for ineligible shapes. Returns (out, lse).

    Both outputs are derived from a single `[M, N_KV]` score tensor.
    """
    scale = scale if scale is not None else (1.0 / math.sqrt(q.shape[-1]))
    scores = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * float(scale)
    if is_causal or window_size > 0:
        mask = build_rectangular_swa_mask(
            q.shape[-2],
            k.shape[-2],
            q_offset,
            k_offset,
            window_size,
            q.device,
            is_causal=is_causal,
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse.unsqueeze(-1))
    return (probs @ v.float()).to(v.dtype), lse


def rect_forward(q, k, v, *, q_offset, k_offset, is_causal, window_size, scale):
    """Rectangular forward. Returns (out, lse).

    Uses the flash-attn slice path when `flash_rect_eligible` accepts the
    shapes, and `math_forward_rect` otherwise.
    """
    if flash_rect_eligible(q, k, v, q_offset=q_offset, k_offset=k_offset, is_causal=is_causal, window_size=window_size):
        return flash_rect_forward(q, k, v, q_offset=q_offset, k_offset=k_offset, window_size=window_size, scale=scale)
    return math_forward_rect(
        q, k, v, q_offset=q_offset, k_offset=k_offset, is_causal=is_causal, window_size=window_size, scale=scale
    )


def rect_first_backward(dO, q, k, v, out, lse, *, q_offset, k_offset, is_causal, window_size, scale):
    if flash_rect_eligible(q, k, v, q_offset=q_offset, k_offset=k_offset, is_causal=is_causal, window_size=window_size):
        return flash_rect_backward(
            dO, q, k, v, out, lse, q_offset=q_offset, k_offset=k_offset, window_size=window_size, scale=scale
        )
    return grad_via_autograd(
        lambda a, b, c: sdpa_math_rect(
            a,
            b,
            c,
            q_offset=q_offset,
            k_offset=k_offset,
            is_causal=is_causal,
            window_size=window_size,
            scale=scale,
        ),
        dO,
        q,
        k,
        v,
    )


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=ROW_PASS_CONFIGS,
    key=[
        "M_BUCKET",
        "N_KV_BUCKET",
        "D_MODEL",
        "IS_CAUSAL",
        "WINDOW_SIZE",
        "DOT_INPUT_PRECISION",
        "USE_BF16_SOURCE_DOTS",
        "REUSE_DQ",
        "ZERO_UQ",
        "ZERO_UK",
        "ZERO_UV",
    ],
    prune_configs_by={"early_config_prune": common.pruner("row")},
)
@triton.jit
def _row_pass_kernel_rect(
    Qbar_ptr,
    dObar_ptr,
    Alpha_ptr,
    E_ptr,
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    dO_ptr,
    dQ_ptr,
    dQbar_ptr,
    dKbar_ptr,
    dVbar_ptr,
    L_ptr,
    D_ptr,
    stride_bh_q,
    stride_q_n,
    stride_bh_kv,
    stride_kv_n,
    stride_d,
    M,
    N_KV,
    Q_OFFSET,
    K_OFFSET,
    scale,
    # bucketed sequence length: the autotune cache key only, never indexing.
    # see `common.autotune_length_bucket`.
    M_BUCKET: tl.constexpr,
    N_KV_BUCKET: tl.constexpr,
    D_MODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    USE_BF16_SOURCE_DOTS: tl.constexpr,
    REUSE_DQ: tl.constexpr,
    ZERO_UQ: tl.constexpr,
    ZERO_UK: tl.constexpr,
    ZERO_UV: tl.constexpr,
    TILE_DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid0 = tl.program_id(0)
    group_id = pid0 // GROUP_M
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid0 % group_size)
    pid_bh = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_MODEL)
    mask_m = off_m < M
    base_q = pid_bh * stride_bh_q
    base_kv = pid_bh * stride_bh_kv

    # tiles used to arrive already widened, because the launcher upcast the whole
    # tensor in HBM. They now arrive at storage precision, so the dot operands
    # take TILE_DTYPE and O_m, which is only ever used in elementwise arithmetic
    # inside finish_row_outputs, converts to FP32 explicitly.
    Q_m = tl.load(
        Q_ptr + base_q + off_m[:, None] * stride_q_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
    ).to(TILE_DTYPE)
    dO_m = tl.load(
        dO_ptr + base_q + off_m[:, None] * stride_q_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
    ).to(TILE_DTYPE)
    dQbar_m = tl.load(
        dQbar_ptr + base_q + off_m[:, None] * stride_q_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
    ).to(TILE_DTYPE)

    L_m = tl.load(L_ptr + pid_bh * M + off_m, mask=mask_m, other=0.0)
    # $D_i$ is row-local, so it is formed here from the O tile instead of by a
    # separate delta launch. O is reloaded in the epilogue rather than held live
    # across the column sweep.
    O_m = tl.load(
        O_ptr + base_q + off_m[:, None] * stride_q_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
    )
    D_m = common.row_delta(dO_m, O_m)

    acc_alpha = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_E_circ = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_dO_circ = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)
    acc_Omega = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)
    acc_B = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)
    acc_R_shift = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)

    # clipping is in absolute coordinates, then shifted back into KV-local space
    m_abs_start = pid_m * BLOCK_M + Q_OFFSET
    start_col_block = 0
    end_col_block = tl.cdiv(N_KV, BLOCK_N)
    if IS_CAUSAL:
        if WINDOW_SIZE > 0:
            earliest_col = tl.maximum(0, m_abs_start - (WINDOW_SIZE - 1) - K_OFFSET)
            start_col_block = earliest_col // BLOCK_N
        latest_col = (pid_m + 1) * BLOCK_M + Q_OFFSET - K_OFFSET
        end_col_block = tl.minimum(end_col_block, tl.cdiv(latest_col, BLOCK_N))

    off_m_abs = off_m + Q_OFFSET

    for n_idx in range(start_col_block, end_col_block):
        off_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        off_n_abs = off_n + K_OFFSET
        mask_n = off_n < N_KV
        col_ptrs = off_n[:, None] * stride_kv_n + off_d[None, :] * stride_d

        K_n = tl.load(K_ptr + base_kv + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
        V_n = tl.load(V_ptr + base_kv + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
        # a zero tangent is never read, so it is never fetched; the K/V tile
        # stands in only to keep arity uniform across the constexpr branches
        if ZERO_UK:
            dKbar_n = K_n
        else:
            dKbar_n = tl.load(dKbar_ptr + base_kv + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
        if ZERO_UV:
            dVbar_n = K_n
        else:
            dVbar_n = tl.load(dVbar_ptr + base_kv + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)

        (
            acc_alpha,
            acc_E_circ,
            acc_dO_circ,
            acc_Omega,
            acc_B,
            acc_R_shift,
        ) = common.row_pass_body(
            K_n,
            V_n,
            dKbar_n,
            dVbar_n,
            Q_m,
            dO_m,
            dQbar_m,
            L_m,
            D_m,
            off_m_abs,
            off_n_abs,
            mask_m,
            mask_n,
            acc_alpha,
            acc_E_circ,
            acc_dO_circ,
            acc_Omega,
            acc_B,
            acc_R_shift,
            scale,
            IS_CAUSAL=IS_CAUSAL,
            WINDOW_SIZE=WINDOW_SIZE,
            DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
            USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
            REUSE_DQ=REUSE_DQ,
            ZERO_UQ=ZERO_UQ,
            ZERO_UK=ZERO_UK,
            ZERO_UV=ZERO_UV,
        )

    # epilogue operands, loaded after the sweep rather than held through it
    O_m = tl.load(
        O_ptr + base_q + off_m[:, None] * stride_q_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
    ).to(tl.float32)
    if REUSE_DQ:
        dQ_m = tl.load(
            dQ_ptr + base_q + off_m[:, None] * stride_q_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
        ).to(tl.float32)
    else:
        dQ_m = Q_m
    alpha_m, E_m, Qbar_m, dObar_m = common.finish_row_outputs(
        acc_alpha,
        acc_E_circ,
        acc_dO_circ,
        acc_Omega,
        acc_B,
        acc_R_shift,
        O_m,
        dQ_m,
        D_m,
        scale,
        REUSE_DQ=REUSE_DQ,
    )

    out_ptrs = base_q + off_m[:, None] * stride_q_n + off_d[None, :] * stride_d
    tl.store(Qbar_ptr + out_ptrs, Qbar_m, mask=mask_m[:, None])
    tl.store(dObar_ptr + out_ptrs, dObar_m, mask=mask_m[:, None])
    tl.store(Alpha_ptr + pid_bh * M + off_m, alpha_m, mask=mask_m)
    tl.store(E_ptr + pid_bh * M + off_m, E_m, mask=mask_m)
    # published for the column pass, which streams rows it does not own
    tl.store(D_ptr + pid_bh * M + off_m, D_m, mask=mask_m)


@triton.autotune(
    configs=COL_PASS_CONFIGS,
    key=[
        "M_BUCKET",
        "N_KV_BUCKET",
        "D_MODEL",
        "IS_CAUSAL",
        "WINDOW_SIZE",
        "DOT_INPUT_PRECISION",
        "USE_BF16_SOURCE_DOTS",
        "ZERO_UQ",
        "ZERO_UK",
        "ZERO_UV",
    ],
    prune_configs_by={"early_config_prune": common.pruner("col")},
)
@triton.jit
def _col_pass_kernel_rect(
    Kbar_ptr,
    Vbar_ptr,
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    dQbar_ptr,
    dKbar_ptr,
    dVbar_ptr,
    L_ptr,
    D_ptr,
    Alpha_ptr,
    E_ptr,
    stride_bh_q,
    stride_q_n,
    stride_bh_kv,
    stride_kv_n,
    stride_d,
    M,
    N_KV,
    Q_OFFSET,
    K_OFFSET,
    scale,
    # bucketed sequence length: the autotune cache key only, never indexing.
    # see `common.autotune_length_bucket`.
    M_BUCKET: tl.constexpr,
    N_KV_BUCKET: tl.constexpr,
    D_MODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    USE_BF16_SOURCE_DOTS: tl.constexpr,
    ZERO_UQ: tl.constexpr,
    ZERO_UK: tl.constexpr,
    ZERO_UV: tl.constexpr,
    TILE_DTYPE: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    num_pid_n = tl.cdiv(N_KV, BLOCK_N)
    pid0 = tl.program_id(0)
    group_id = pid0 // GROUP_N
    first_pid_n = group_id * GROUP_N
    group_size = tl.minimum(num_pid_n - first_pid_n, GROUP_N)
    pid_n = first_pid_n + (pid0 % group_size)
    pid_bh = tl.program_id(1)

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_n_abs = off_n + K_OFFSET
    off_d = tl.arange(0, D_MODEL)
    mask_n = off_n < N_KV
    base_q = pid_bh * stride_bh_q
    base_kv = pid_bh * stride_bh_kv
    col_ptrs = off_n[:, None] * stride_kv_n + off_d[None, :] * stride_d

    K_n = tl.load(K_ptr + base_kv + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
    V_n = tl.load(V_ptr + base_kv + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
    # a zero tangent is never read, so it is never fetched; the K/V tile
    # stands in only to keep arity uniform across the constexpr branches
    if ZERO_UK:
        dKbar_n = K_n
    else:
        dKbar_n = tl.load(dKbar_ptr + base_kv + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
    if ZERO_UV:
        dVbar_n = K_n
    else:
        dVbar_n = tl.load(dVbar_ptr + base_kv + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)

    acc_Kbar = tl.zeros([BLOCK_N, D_MODEL], dtype=tl.float32)
    acc_Vbar = tl.zeros([BLOCK_N, D_MODEL], dtype=tl.float32)

    n_abs_start = pid_n * BLOCK_N + K_OFFSET
    n_abs_end = n_abs_start + BLOCK_N
    start_m = 0
    end_m = tl.cdiv(M, BLOCK_M)
    if IS_CAUSAL:
        m_lo = tl.maximum(0, n_abs_start - Q_OFFSET)
        start_m = m_lo // BLOCK_M
        if WINDOW_SIZE > 0:
            m_hi = (n_abs_end - 1) + WINDOW_SIZE + 1 - Q_OFFSET
            end_m = tl.minimum(end_m, tl.cdiv(m_hi, BLOCK_M))

    for m_idx in range(start_m, end_m):
        off_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = off_m < M
        off_m_abs = off_m + Q_OFFSET
        row_ptrs = off_m[:, None] * stride_q_n + off_d[None, :] * stride_d

        Q_m = tl.load(Q_ptr + base_q + row_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)
        dO_m = tl.load(dO_ptr + base_q + row_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)
        dQbar_m = tl.load(dQbar_ptr + base_q + row_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)

        L_m = tl.load(L_ptr + pid_bh * M + off_m, mask=mask_m, other=0.0)
        D_m = tl.load(D_ptr + pid_bh * M + off_m, mask=mask_m, other=0.0)
        alpha_m = tl.load(Alpha_ptr + pid_bh * M + off_m, mask=mask_m, other=0.0)
        E_m = tl.load(E_ptr + pid_bh * M + off_m, mask=mask_m, other=0.0)

        valid_mn = common.attention_mask(
            off_m_abs,
            off_n_abs,
            mask_m,
            mask_n,
            IS_CAUSAL=IS_CAUSAL,
            WINDOW_SIZE=WINDOW_SIZE,
        )

        acc_Kbar, acc_Vbar = common.col_pass_body(
            Q_m,
            dO_m,
            dQbar_m,
            K_n,
            V_n,
            dKbar_n,
            dVbar_n,
            L_m,
            D_m,
            alpha_m,
            E_m,
            valid_mn,
            acc_Kbar,
            acc_Vbar,
            scale,
            DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
            USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
            ZERO_UQ=ZERO_UQ,
            ZERO_UK=ZERO_UK,
            ZERO_UV=ZERO_UV,
        )

    out_ptrs = base_kv + off_n[:, None] * stride_kv_n + off_d[None, :] * stride_d
    tl.store(Kbar_ptr + out_ptrs, acc_Kbar, mask=mask_n[:, None])
    tl.store(Vbar_ptr + out_ptrs, acc_Vbar, mask=mask_n[:, None])


# ---------------------------------------------------------------------------
# launcher
# ---------------------------------------------------------------------------


def bob_2pass_rect(
    Q,
    K,
    V,
    O,
    L,
    dO,
    dQ_bar,
    dK_bar,
    dV_bar,
    *,
    q_offset: int = 0,
    k_offset: int = 0,
    causal: bool = True,
    window_size: int = 0,
    scale: float | None = None,
    bf16_source_dots: bool = False,
    dQ: torch.Tensor | None = None,
    reuse_dq: bool = False,
    zero_tangents: tuple[bool, bool, bool] = (False, False, False),
):
    """Rectangular two-pass second backward. Returns (Qbar, Kbar, Vbar, dObar).

    Inputs are consumed at storage precision. Outputs are FP32.

    `reuse_dq` uses the first-order $dQ = \\tau XK$ returned by the rectangular
    first backward in place of re-accumulating it, as the square path does.
    `zero_tangents` marks which of the input gradients `(U_Q, U_K, U_V)` are
    exactly zero, with the meaning given in `square.bob_2pass`.
    """
    zero_uq, zero_uk, zero_uv = zero_tangents
    if reuse_dq and dQ is None:
        raise ValueError("reuse_dq=True requires the first-order dQ")
    if Q.device.type != "cuda":
        raise NotImplementedError("the rectangular FlashBoB kernel requires CUDA")
    if bf16_source_dots and any(t.dtype != torch.bfloat16 for t in (Q, K, V, dO, dQ_bar, dK_bar, dV_bar)):
        raise ValueError("bf16_source_dots=True requires BF16 source tensors")

    window_size = common.normalize_window_size(window_size)

    B, H, M, D = Q.shape
    Bk, Hk, N_KV, Dk = K.shape
    if (B, H, D) != (Bk, Hk, Dk) or V.shape != K.shape:
        raise ValueError("Q, K, and V must have matching batch, head, and feature dimensions")
    if any(t.shape != Q.shape for t in (O, dO, dQ_bar)):
        raise ValueError("O, dO, and dQ_bar must match Q.shape")
    if dK_bar.shape != K.shape or dV_bar.shape != V.shape:
        raise ValueError("dK_bar and dV_bar must match K.shape and V.shape")
    if L.shape != (B, H, M):
        raise ValueError(f"L must have shape {(B, H, M)}, got {tuple(L.shape)}")
    score_scale = D**-0.5 if scale is None else float(scale)
    BH = B * H

    # inputs stay at storage precision: the eight FP32 copies this used to
    # materialize were nine HBM round trips of pure conversion traffic before
    # either kernel launched, and they forced every tl.dot into TF32. For FP32
    # inputs .float() was already a no-op, so this changes only low-precision calls.
    Qf, Kf, Vf = Q.contiguous(), K.contiguous(), V.contiguous()
    Of, dOf = O.contiguous(), dO.contiguous()
    dQb = dQ_bar.contiguous()
    dKb = dK_bar.contiguous()
    dVb = dV_bar.contiguous()
    Lf = (L.float() * common.LOG2E_HOST).contiguous()

    # $D_i = \langle dO_i, O_i \rangle$ accumulates in FP32 inside the row pass,
    # which already loads the O and dO tiles it needs. No separate delta launch.
    tile_dtype = tl.bfloat16 if bf16_source_dots else tl.float32
    D_vec = torch.empty((B, H, M), device=Q.device, dtype=torch.float32)
    # alias Q when dQ is unused so the kernel always has a valid pointer
    dQf = dQ.contiguous() if reuse_dq else Qf

    Qbar = torch.empty((B, H, M, D), device=Q.device, dtype=torch.float32)
    Kbar = torch.empty((B, H, N_KV, D), device=Q.device, dtype=torch.float32)
    Vbar = torch.empty((B, H, N_KV, D), device=Q.device, dtype=torch.float32)
    dObar = torch.empty((B, H, M, D), device=Q.device, dtype=torch.float32)
    alpha = torch.empty((B, H, M), device=Q.device, dtype=torch.float32)
    E = torch.empty((B, H, M), device=Q.device, dtype=torch.float32)

    strides = (M * D, D, N_KV * D, D, 1)

    _row_pass_kernel_rect[lambda META: (triton.cdiv(M, META["BLOCK_M"]), BH)](
        Qbar,
        dObar,
        alpha,
        E,
        Qf,
        Kf,
        Vf,
        Of,
        dOf,
        dQf,
        dQb,
        dKb,
        dVb,
        Lf,
        D_vec,
        *strides,
        M,
        N_KV,
        int(q_offset),
        int(k_offset),
        M_BUCKET=common.autotune_length_bucket(M),
        N_KV_BUCKET=common.autotune_length_bucket(N_KV),
        D_MODEL=D,
        scale=score_scale,
        WINDOW_SIZE=window_size,
        IS_CAUSAL=causal,
        DOT_INPUT_PRECISION="tf32",
        USE_BF16_SOURCE_DOTS=bf16_source_dots,
        REUSE_DQ=reuse_dq,
        ZERO_UQ=zero_uq,
        ZERO_UK=zero_uk,
        ZERO_UV=zero_uv,
        TILE_DTYPE=tile_dtype,
    )

    _col_pass_kernel_rect[lambda META: (triton.cdiv(N_KV, META["BLOCK_N"]), BH)](
        Kbar,
        Vbar,
        Qf,
        Kf,
        Vf,
        dOf,
        dQb,
        dKb,
        dVb,
        Lf,
        D_vec,
        alpha,
        E,
        *strides,
        M,
        N_KV,
        int(q_offset),
        int(k_offset),
        M_BUCKET=common.autotune_length_bucket(M),
        N_KV_BUCKET=common.autotune_length_bucket(N_KV),
        D_MODEL=D,
        scale=score_scale,
        WINDOW_SIZE=window_size,
        IS_CAUSAL=causal,
        DOT_INPUT_PRECISION="tf32",
        USE_BF16_SOURCE_DOTS=bf16_source_dots,
        ZERO_UQ=zero_uq,
        ZERO_UK=zero_uk,
        ZERO_UV=zero_uv,
        TILE_DTYPE=tile_dtype,
    )

    return Qbar, Kbar, Vbar, dObar
