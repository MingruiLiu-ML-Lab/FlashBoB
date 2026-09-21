"""Square BoB kernels and their launcher.

Applies to equal query and key lengths with zero offsets. The algorithm has two
passes: a row pass producing (alpha, E, D, Qbar, dObar) and a column pass
producing (Kbar, Vbar). Each kernel is defined next to the launcher that
invokes it.

The dispatched configuration reuses the first-order dQ. `reuse_dq=False`
selects the accumulating form and is available for ablation. BF16 source dots
and the zero-input-gradient specializations are independent flags.

Tensors are passed at storage precision. `bf16_source_dots` determines the
precision of a loaded tile. When disabled, tiles are converted to FP32 after
load and every `tl.dot` runs in TF32. When enabled, tiles remain at storage
precision and the dots run on the BF16 tensor cores.

$D$ is computed inside the row pass from the O and dO tiles it loads, then
written for the column pass.
"""

import torch
import triton
import triton.language as tl

from . import common
from .common import COL_PASS_CONFIGS, ROW_PASS_CONFIGS

# ---------------------------------------------------------------------------
# two-pass kernels
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=ROW_PASS_CONFIGS,
    key=[
        "N_BUCKET",
        "D_MODEL",
        "IS_CAUSAL",
        "WINDOW_SIZE",
        "DOT_INPUT_PRECISION",
        "REUSE_DQ",
        "USE_BF16_SOURCE_DOTS",
        "ZERO_UQ",
        "ZERO_UK",
        "ZERO_UV",
    ],
    prune_configs_by={"early_config_prune": common.pruner("row")},
)
@triton.jit
def _row_pass_kernel(
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
    stride_bh,
    stride_n,
    stride_d,
    N,
    scale,
    # bucketed sequence length: the autotune cache key only, never indexing.
    # see `common.autotune_length_bucket`.
    N_BUCKET: tl.constexpr,
    D_MODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    REUSE_DQ: tl.constexpr,
    USE_BF16_SOURCE_DOTS: tl.constexpr,
    ZERO_UQ: tl.constexpr,
    ZERO_UK: tl.constexpr,
    ZERO_UV: tl.constexpr,
    TILE_DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(N, BLOCK_M)
    pid0 = tl.program_id(0)
    group_id = pid0 // GROUP_M
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid0 % group_size)
    pid_bh = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_MODEL)
    mask_m = off_m < N
    base = pid_bh * stride_bh

    # the address expression is recomputed at each use rather than hoisted: a
    # hoisted [BLOCK_M, D_MODEL] index tensor stays live across the whole column
    # loop, and the register pressure costs more than the repeated arithmetic

    # TILE_DTYPE is the storage dtype on the BF16 path, so dot operands reach the
    # BF16 tensor cores instead of being widened to FP32 first. O_m is only ever
    # used in elementwise FP32 arithmetic, so it converts unconditionally.
    Q_m = tl.load(
        Q_ptr + base + off_m[:, None] * stride_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
    ).to(TILE_DTYPE)
    dO_m = tl.load(
        dO_ptr + base + off_m[:, None] * stride_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
    ).to(TILE_DTYPE)
    dQbar_m = tl.load(
        dQbar_ptr + base + off_m[:, None] * stride_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
    ).to(TILE_DTYPE)

    L_m = tl.load(L_ptr + pid_bh * N + off_m, mask=mask_m, other=0.0)

    # $D_i$ is row-local, so this program can form it from the O tile it is
    # about to need anyway. That retires the separate delta launch and the
    # O(N) D write-then-read through HBM. O itself does NOT stay live across the
    # column sweep; it is reloaded in the epilogue, its only other use.
    O_m = tl.load(O_ptr + base + off_m[:, None] * stride_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0)
    D_m = common.row_delta(dO_m, O_m)

    acc_alpha = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_E_circ = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_dO_circ = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)
    acc_Omega = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)
    acc_B = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)
    acc_R_shift = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)

    # clip the column range before traversal so masked-out tiles are never
    # loaded, rather than loading them and discarding under the mask
    start_col_block = 0
    end_col_block = tl.cdiv(N, BLOCK_N)
    if IS_CAUSAL:
        if WINDOW_SIZE > 0:
            earliest_col = tl.maximum(0, pid_m * BLOCK_M - WINDOW_SIZE + 1)
            start_col_block = earliest_col // BLOCK_N
        end_col_block = tl.minimum(
            end_col_block,
            tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N),
        )

    for n_idx in range(start_col_block, end_col_block):
        off_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = off_n < N
        col_ptrs = off_n[:, None] * stride_n + off_d[None, :] * stride_d

        K_n = tl.load(K_ptr + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
        V_n = tl.load(V_ptr + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
        # a zero tangent is never read, so it is never fetched; the K/V tile
        # stands in only to keep arity uniform across the constexpr branches
        if ZERO_UK:
            dKbar_n = K_n
        else:
            dKbar_n = tl.load(dKbar_ptr + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
        if ZERO_UV:
            dVbar_n = K_n
        else:
            dVbar_n = tl.load(dVbar_ptr + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)

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
            off_m,
            off_n,
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
        O_ptr + base + off_m[:, None] * stride_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
    ).to(tl.float32)
    # loaded only when it is used; otherwise dQ_ptr aliases Q_ptr
    if REUSE_DQ:
        dQ_m = tl.load(
            dQ_ptr + base + off_m[:, None] * stride_n + off_d[None, :] * stride_d, mask=mask_m[:, None], other=0.0
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

    out_ptrs = base + off_m[:, None] * stride_n + off_d[None, :] * stride_d
    tl.store(Qbar_ptr + out_ptrs, Qbar_m, mask=mask_m[:, None])
    tl.store(dObar_ptr + out_ptrs, dObar_m, mask=mask_m[:, None])
    tl.store(Alpha_ptr + pid_bh * N + off_m, alpha_m, mask=mask_m)
    tl.store(E_ptr + pid_bh * N + off_m, E_m, mask=mask_m)
    # D is published for the column pass, which streams row tiles it does not own
    # and cannot re-derive D without re-reading O. Each row tile belongs to
    # exactly one program, so this is a single write per row, never a race.
    tl.store(D_ptr + pid_bh * N + off_m, D_m, mask=mask_m)


@triton.autotune(
    configs=COL_PASS_CONFIGS,
    key=[
        "N_BUCKET",
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
def _col_pass_kernel(
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
    stride_bh,
    stride_n,
    stride_d,
    N,
    scale,
    # bucketed sequence length: the autotune cache key only, never indexing.
    # see `common.autotune_length_bucket`.
    N_BUCKET: tl.constexpr,
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
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid0 = tl.program_id(0)
    group_id = pid0 // GROUP_N
    first_pid_n = group_id * GROUP_N
    group_size = tl.minimum(num_pid_n - first_pid_n, GROUP_N)
    pid_n = first_pid_n + (pid0 % group_size)
    pid_bh = tl.program_id(1)

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, D_MODEL)
    mask_n = off_n < N
    base = pid_bh * stride_bh
    col_ptrs = off_n[:, None] * stride_n + off_d[None, :] * stride_d

    K_n = tl.load(K_ptr + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
    V_n = tl.load(V_ptr + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
    # a zero tangent is never read, so it is never fetched; the K/V tile
    # stands in only to keep arity uniform across the constexpr branches
    if ZERO_UK:
        dKbar_n = K_n
    else:
        dKbar_n = tl.load(dKbar_ptr + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
    if ZERO_UV:
        dVbar_n = K_n
    else:
        dVbar_n = tl.load(dVbar_ptr + base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)

    acc_Kbar = tl.zeros([BLOCK_N, D_MODEL], dtype=tl.float32)
    acc_Vbar = tl.zeros([BLOCK_N, D_MODEL], dtype=tl.float32)

    start_m = 0
    end_m = tl.cdiv(N, BLOCK_M)
    if IS_CAUSAL:
        start_m = (pid_n * BLOCK_N) // BLOCK_M
        if WINDOW_SIZE > 0:
            last_row = pid_n * BLOCK_N + BLOCK_N - 1 + WINDOW_SIZE
            end_m = tl.minimum(end_m, tl.cdiv(last_row + 1, BLOCK_M))

    for m_idx in range(start_m, end_m):
        off_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = off_m < N
        row_ptrs = off_m[:, None] * stride_n + off_d[None, :] * stride_d

        Q_m = tl.load(Q_ptr + base + row_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)
        dO_m = tl.load(dO_ptr + base + row_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)
        dQbar_m = tl.load(dQbar_ptr + base + row_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)

        L_m = tl.load(L_ptr + pid_bh * N + off_m, mask=mask_m, other=0.0)
        D_m = tl.load(D_ptr + pid_bh * N + off_m, mask=mask_m, other=0.0)
        alpha_m = tl.load(Alpha_ptr + pid_bh * N + off_m, mask=mask_m, other=0.0)
        E_m = tl.load(E_ptr + pid_bh * N + off_m, mask=mask_m, other=0.0)

        valid_mn = common.attention_mask(
            off_m,
            off_n,
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

    tl.store(Kbar_ptr + base + col_ptrs, acc_Kbar, mask=mask_n[:, None])
    tl.store(Vbar_ptr + base + col_ptrs, acc_Vbar, mask=mask_n[:, None])


# ---------------------------------------------------------------------------
# two-pass launcher
# ---------------------------------------------------------------------------


def bob_2pass(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    L: torch.Tensor,
    dO: torch.Tensor,
    dQ_bar: torch.Tensor,
    dK_bar: torch.Tensor,
    dV_bar: torch.Tensor,
    *,
    causal: bool = True,
    window_size: int = 0,
    dQ: torch.Tensor | None = None,
    reuse_dq: bool = False,
    bf16_source_dots: bool = False,
    zero_tangents: tuple[bool, bool, bool] = (False, False, False),
    scale: float | None = None,
):
    """Square two-pass second backward. Returns (Qbar, Kbar, Vbar, dObar).

    Tensors are `[B, H, N, D]` and contiguous. Low-precision inputs remain at
    storage precision in HBM, and tiles are converted after load according to
    `bf16_source_dots`.

    `zero_tangents` marks which of the input gradients `(U_Q, U_K, U_V)` on
    `dQ`, `dK`, and `dV` are exactly zero. Autograd supplies an exact zero when
    the differentiated scalar does not depend on the matching first-order
    gradient. Each marked entry removes its source GEMM and its Layer-C
    counterpart from both passes. A marked tensor is not read; any tensor of the
    correct shape is accepted in its place.
    """
    zero_uq, zero_uk, zero_uv = zero_tangents
    if Q.device.type != "cuda":
        raise NotImplementedError("the FlashBoB kernel requires CUDA")
    if reuse_dq and dQ is None:
        raise ValueError("reuse_dq=True requires the first-order dQ")
    if bf16_source_dots and any(t.dtype != torch.bfloat16 for t in (Q, K, V, dO, dQ_bar, dK_bar, dV_bar)):
        raise ValueError("bf16_source_dots=True requires BF16 source tensors")
    # the kernels index L at `pid_bh * N`, so a row-padded L (which is what the
    # sliding-window aten forward returns for N not a multiple of 32) reads the
    # wrong rows for every head after the first. `rectangular.py` already
    # checked this shape; the square path previously omitted the check.
    if L.shape != Q.shape[:-1]:
        raise ValueError(f"L must have shape {tuple(Q.shape[:-1])}, got {tuple(L.shape)}")

    window_size = common.normalize_window_size(window_size)

    # on the BF16 path tiles stay at storage precision all the way into tl.dot;
    # otherwise every tile widens to FP32 after load and the dots run in TF32
    tile_dtype = tl.bfloat16 if bf16_source_dots else tl.float32

    B, H, N, D = Q.shape
    score_scale = D**-0.5 if scale is None else float(scale)
    BH = B * H
    stride_bh, stride_n, stride_d = N * D, D, 1

    Qf, Kf, Vf = Q.contiguous(), K.contiguous(), V.contiguous()
    Of, dOf = O.contiguous(), dO.contiguous()
    dQb, dKb, dVb = dQ_bar.contiguous(), dK_bar.contiguous(), dV_bar.contiguous()
    # alias Q when dQ is unused so the kernel always has a valid pointer
    dQf = dQ.contiguous() if reuse_dq else Qf

    D_vec = torch.empty((B, H, N), device=Q.device, dtype=torch.float32)
    # L is converted into log2 units once here, per row, instead of the
    # kernels rescaling every score tile
    Lf = (L.float() * common.LOG2E_HOST).contiguous()
    Qbar = torch.empty_like(Q)
    Kbar = torch.empty_like(K)
    Vbar = torch.empty_like(V)
    dObar = torch.empty_like(dO)
    alpha = torch.empty((B, H, N), device=Q.device, dtype=torch.float32)
    E = torch.empty((B, H, N), device=Q.device, dtype=torch.float32)

    # no delta launch: the row pass computes D from the O and dO tiles it already
    # loads and publishes it here for the column pass. That removes one launch
    # and the delta kernel's full O and dO read of HBM.
    _row_pass_kernel[lambda META: (triton.cdiv(N, META["BLOCK_M"]), BH)](
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
        stride_bh,
        stride_n,
        stride_d,
        N=N,
        N_BUCKET=common.autotune_length_bucket(N),
        D_MODEL=D,
        scale=score_scale,
        WINDOW_SIZE=window_size,
        IS_CAUSAL=causal,
        DOT_INPUT_PRECISION="tf32",
        REUSE_DQ=reuse_dq,
        USE_BF16_SOURCE_DOTS=bf16_source_dots,
        ZERO_UQ=zero_uq,
        ZERO_UK=zero_uk,
        ZERO_UV=zero_uv,
        TILE_DTYPE=tile_dtype,
    )

    _col_pass_kernel[lambda META: (triton.cdiv(N, META["BLOCK_N"]), BH)](
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
        stride_bh,
        stride_n,
        stride_d,
        N=N,
        N_BUCKET=common.autotune_length_bucket(N),
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
