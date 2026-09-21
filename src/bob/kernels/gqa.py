"""Grouped-query and multi-query attention.

The query side is indexed as `[B, H_KV, N*G, D]`, where packed row `g*N + token`
selects group member `g` at position `token`. One K/V head then serves its whole
query group from a single load. The column pass walks every packed row for its
KV head, and the group reduction into Kbar and Vbar follows from that traversal
without repeated K/V tensors and without atomics.

The packing is head-major by design. Packed row `g*N + token` is exactly the
row offset of token `token` of query head `h_kv*G + g` inside the caller's
contiguous `[B, H_Q, N, D]` buffer, so a tile of consecutive packed rows is a
tile of consecutive tokens of one head and every query-side load is the same
contiguous access the square kernels make. The token-major convention
`token*G + g` that this module used before gathered each row of a tile from a
different head, `N*D` elements apart, and made BLOCK_M mean a fraction of a
token, so the row tile had to align to G. Head-major addressing is what lets
these kernels tile and clip exactly like the square pair.

Query-side tensors keep their caller `[B, H_Q, N, D]` shape: no tensor is
packed or unpacked. Materializing a packed copy requires six permute and
contiguous passes on input and two on output, measured at 5.5% of the call.
`alpha`, `E`, and `D` are stored in packed row order and are internal to this
pair of kernels.

Causal masking compares token positions rather than packed row indices. A row
tile now spans one head, so the token range of a tile is contiguous and the
column clip is exactly the square path's.

Two specializations apply:

- Split columns. Short sequences leave SMs idle, so the column pass is divided
  across `SPLITS` programs that write FP32 partials, which a reduction kernel
  combines.
- Zero input gradients. Each of U_Q, U_K, and U_V that autograd leaves
  undefined removes its source GEMM and its Layer-C counterpart from both
  passes.

BLOCK_M counts packed rows, which head-major packing makes tokens of one head,
so it no longer has to divide the group size and can reach the square pair's
widest row tile.
"""

import torch
import triton
import triton.language as tl

from . import common

# The square pair's tile shapes, at this module's warp count and pipeline depth.
#
# BLOCK_M is the lever that matters here. A row program streams one BLOCK_N x D
# tile each of K, V, dKbar and dVbar per column step and the causal area fixes
# the number of column steps, so the K/V traffic of the whole pass is
# proportional to $D / \mathrm{BLOCK\_M}$. Capping BLOCK_M at 32 while the
# square pair reaches 128 left the GQA pair at 1.70x the square pair on expanded
# K/V on H100 at N=2048, B=2, Hq=32, Hkv=16, d=64, even after the tiles were
# loaded at storage precision in a historical H100 measurement.
#
# Sweeping warp count and pipeline depth as well, the way `common`
# does, would take these families from 26 configurations to 198. The campaign
# pays that cold autotune once per persisted geometry (677 s at N=32768 for the
# 16 configurations this module had), so warps and stages stay fixed here.
ROW_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "GROUP_M": gm}, num_warps=4, num_stages=2)
    for bm, bn in ((16, 16), (16, 32), (32, 16), (32, 32), (64, 16), (64, 32), (128, 32))
    for gm in (4, 8)
]

COL_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "GROUP_N": 8}, num_warps=4, num_stages=ns)
    for bm, bn in ((16, 16), (16, 32), (32, 16), (32, 32), (64, 32), (128, 32))
    for ns in (2, 3)
]


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=ROW_CONFIGS,
    key=[
        "N_BUCKET",
        "G",
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
def _packed_row_pass(
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
    N,
    N_PACKED,
    # bucketed sequence length: the autotune cache key only, never indexing.
    # see `common.autotune_length_bucket`.
    N_BUCKET: tl.constexpr,
    SCORE_SCALE,
    G: tl.constexpr,
    H_KV: tl.constexpr,
    ZERO_UQ: tl.constexpr,
    ZERO_UK: tl.constexpr,
    ZERO_UV: tl.constexpr,
    D_MODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    USE_BF16_SOURCE_DOTS: tl.constexpr,
    REUSE_DQ: tl.constexpr,
    TILE_DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tiles_per_head = tl.cdiv(N, BLOCK_M)
    num_pid_m = G * tiles_per_head
    pid0 = tl.program_id(0)
    group_id = pid0 // GROUP_M
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid0 % group_size)
    pid_bhkv = tl.program_id(1)

    scale = SCORE_SCALE

    # head-major packing: one row tile is one head's tokens, so the tile never
    # straddles a group member and the causal clip is on a contiguous range
    g_m = pid_m // tiles_per_head
    tile_m = pid_m % tiles_per_head
    token_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_m = g_m * N + token_m  # the packed row index, for the internal stats
    off_d = tl.arange(0, D_MODEL)
    mask_m = token_m < N

    # Query-side tensors are addressed in their ORIGINAL [B, H_Q, N, D] layout.
    # Packed row g*N + token is token `token` of group member g, whose address is
    # b*H_Q*N*D + (h_kv*G + g)*N*D + token*D + d, i.e. the packed row index times
    # D past the group's base. Materializing a packed copy instead cost six
    # permute+contiguous passes on the way in and two on the way out; this is the
    # same arithmetic done in the address computation, and it loads contiguously.
    b_idx = pid_bhkv // H_KV
    h_kv = pid_bhkv % H_KV
    q_head = b_idx * (G * H_KV * N * D_MODEL) + h_kv * G * (N * D_MODEL)
    q_ptrs = q_head + off_m[:, None] * D_MODEL + off_d[None, :]

    stats_head = b_idx * (G * H_KV * N) + h_kv * G * N
    stats_row = stats_head + off_m
    # alpha, E, and D stay in packed row order: they are internal to this pair of
    # kernels and never leave, so there is nothing to unpack
    stats_base = pid_bhkv * N_PACKED
    kv_base = pid_bhkv * N * D_MODEL

    # TILE_DTYPE is the storage dtype on the BF16 path, exactly as in the square
    # kernels. Loading these at FP32 left the Layer-C dots on
    # the TF32 path, because `common.row_pass_body` casts P, PF, PP and P dP to
    # `Q_m.dtype` before `tl.dot`, and doubled every live tile's registers.
    Q_m = tl.load(Q_ptr + q_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)
    dO_m = tl.load(dO_ptr + q_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)
    dQbar_m = tl.load(dQbar_ptr + q_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)
    # O is needed only for D here; the epilogue reloads it rather than holding an
    # FP32 tile live across the column sweep
    O_m = tl.load(O_ptr + q_ptrs, mask=mask_m[:, None], other=0.0)

    L_m = tl.load(L_ptr + stats_row, mask=mask_m, other=0.0)
    # D is row-local; forming it here retires the separate packed delta launch
    D_m = common.row_delta(dO_m, O_m)

    acc_alpha = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_E_circ = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_dO_circ = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)
    acc_Omega = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)
    acc_B = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)
    acc_R_shift = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)

    # a row tile is BLOCK_M consecutive tokens of one head
    start_col_block = 0
    end_col_block = tl.cdiv(N, BLOCK_N)
    if IS_CAUSAL:
        first_token = tile_m * BLOCK_M
        last_token_exclusive = tl.minimum(N, first_token + BLOCK_M)
        if WINDOW_SIZE > 0:
            earliest_col = tl.maximum(0, first_token - WINDOW_SIZE + 1)
            start_col_block = earliest_col // BLOCK_N
        end_col_block = tl.minimum(end_col_block, tl.cdiv(last_token_exclusive, BLOCK_N))

    for n_idx in range(start_col_block, end_col_block):
        off_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = off_n < N
        col_ptrs = off_n[:, None] * D_MODEL + off_d[None, :]

        K_n = tl.load(K_ptr + kv_base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
        V_n = tl.load(V_ptr + kv_base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
        # a zero tangent is never read, so it is never fetched; the K/V tile
        # stands in only to keep arity uniform across the constexpr branches
        if ZERO_UK:
            dKbar_n = K_n
        else:
            dKbar_n = tl.load(dKbar_ptr + kv_base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
        if ZERO_UV:
            dVbar_n = V_n
        else:
            dVbar_n = tl.load(dVbar_ptr + kv_base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)

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
            token_m,
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
    O_m = tl.load(O_ptr + q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    # loaded only when it is used; otherwise dQ_ptr aliases Q_ptr
    if REUSE_DQ:
        dQ_m = tl.load(dQ_ptr + q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    else:
        dQ_m = Q_m.to(tl.float32)
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

    # written straight into [B, H_Q, N, D], so there is nothing to unpack after
    tl.store(Qbar_ptr + q_ptrs, Qbar_m, mask=mask_m[:, None])
    tl.store(dObar_ptr + q_ptrs, dObar_m, mask=mask_m[:, None])
    tl.store(Alpha_ptr + stats_base + off_m, alpha_m, mask=mask_m)
    tl.store(E_ptr + stats_base + off_m, E_m, mask=mask_m)
    tl.store(D_ptr + stats_base + off_m, D_m, mask=mask_m)


@triton.autotune(
    configs=COL_CONFIGS,
    key=[
        "N_BUCKET",
        "G",
        "SPLITS",
        "D_MODEL",
        "IS_CAUSAL",
        "WINDOW_SIZE",
        "USE_BF16_SOURCE_DOTS",
        "ZERO_UQ",
        "ZERO_UK",
        "ZERO_UV",
    ],
    prune_configs_by={"early_config_prune": common.pruner("col")},
)
@triton.jit
def _packed_col_pass(
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
    N,
    N_PACKED,
    # bucketed sequence length: the autotune cache key only, never indexing.
    # see `common.autotune_length_bucket`.
    N_BUCKET: tl.constexpr,
    SCORE_SCALE,
    G: tl.constexpr,
    H_KV: tl.constexpr,
    ZERO_UQ: tl.constexpr,
    ZERO_UK: tl.constexpr,
    ZERO_UV: tl.constexpr,
    SPLITS: tl.constexpr,
    D_MODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    USE_BF16_SOURCE_DOTS: tl.constexpr,
    TILE_DTYPE: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid0 = tl.program_id(0)
    group_id = pid0 // GROUP_N
    first_pid_n = group_id * GROUP_N
    group_size = tl.minimum(num_pid_n - first_pid_n, GROUP_N)
    pid_n = first_pid_n + (pid0 % group_size)
    pid_bhkv = tl.program_id(1)
    pid_split = tl.program_id(2)

    scale = SCORE_SCALE
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, D_MODEL)
    mask_n = off_n < N

    kv_base = pid_bhkv * N * D_MODEL
    out_base = (pid_split * tl.num_programs(1) + pid_bhkv) * N * D_MODEL
    col_ptrs = off_n[:, None] * D_MODEL + off_d[None, :]

    K_n = tl.load(K_ptr + kv_base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
    V_n = tl.load(V_ptr + kv_base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
    # a zero tangent is never read, so it is never fetched; the K/V tile
    # stands in only to keep arity uniform across the constexpr branches
    if ZERO_UK:
        dKbar_n = K_n
    else:
        dKbar_n = tl.load(dKbar_ptr + kv_base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)
    if ZERO_UV:
        dVbar_n = V_n
    else:
        dVbar_n = tl.load(dVbar_ptr + kv_base + col_ptrs, mask=mask_n[:, None], other=0.0).to(TILE_DTYPE)

    acc_Kbar = tl.zeros([BLOCK_N, D_MODEL], dtype=tl.float32)
    acc_Vbar = tl.zeros([BLOCK_N, D_MODEL], dtype=tl.float32)

    # see _packed_row_pass: query-side tensors keep their [B, H_Q, N, D] layout
    b_idx = pid_bhkv // H_KV
    h_kv = pid_bhkv % H_KV
    q_head = b_idx * (G * H_KV * N * D_MODEL) + h_kv * G * (N * D_MODEL)
    stats_head = b_idx * (G * H_KV * N) + h_kv * G * N
    stats_base = pid_bhkv * N_PACKED
    # head-major packing: walk each group member's token tiles in turn, so every
    # row tile is BLOCK_M consecutive tokens of one head and loads contiguously
    tiles_per_head = tl.cdiv(N, BLOCK_M)
    start_tile = 0
    end_tile = tiles_per_head
    if IS_CAUSAL:
        start_tile = (pid_n * BLOCK_N) // BLOCK_M
        if WINDOW_SIZE > 0:
            last_token_exclusive = tl.minimum(N, pid_n * BLOCK_N + BLOCK_N + WINDOW_SIZE)
            end_tile = tl.minimum(end_tile, tl.cdiv(last_token_exclusive, BLOCK_M))
    tiles_per_member = tl.maximum(end_tile - start_tile, 0)

    # stride by SPLITS so each split program covers a disjoint set of row tiles
    for m_idx in range(pid_split, G * tiles_per_member, SPLITS):
        g_m = m_idx // tiles_per_member
        token_m = (start_tile + m_idx % tiles_per_member) * BLOCK_M + tl.arange(0, BLOCK_M)
        off_m = g_m * N + token_m
        mask_m = token_m < N
        row_ptrs = q_head + off_m[:, None] * D_MODEL + off_d[None, :]

        Q_m = tl.load(Q_ptr + row_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)
        dO_m = tl.load(dO_ptr + row_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)
        dQbar_m = tl.load(dQbar_ptr + row_ptrs, mask=mask_m[:, None], other=0.0).to(TILE_DTYPE)

        L_m = tl.load(L_ptr + stats_head + off_m, mask=mask_m, other=0.0)
        D_m = tl.load(D_ptr + stats_base + off_m, mask=mask_m, other=0.0)
        alpha_m = tl.load(Alpha_ptr + stats_base + off_m, mask=mask_m, other=0.0)
        E_m = tl.load(E_ptr + stats_base + off_m, mask=mask_m, other=0.0)

        valid_mn = common.attention_mask(
            token_m,
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

    tl.store(Kbar_ptr + out_base + col_ptrs, acc_Kbar, mask=mask_n[:, None])
    tl.store(Vbar_ptr + out_base + col_ptrs, acc_Vbar, mask=mask_n[:, None])


@triton.jit
def _reduce_split_columns(
    Kbar_ptr,
    Vbar_ptr,
    Kpartial_ptr,
    Vpartial_ptr,
    TOTAL,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    acc_k = tl.zeros([BLOCK], dtype=tl.float32)
    acc_v = tl.zeros([BLOCK], dtype=tl.float32)
    for split in range(SPLITS):
        s = split * TOTAL + offsets
        acc_k += tl.load(Kpartial_ptr + s, mask=mask, other=0.0)
        acc_v += tl.load(Vpartial_ptr + s, mask=mask, other=0.0)
    tl.store(Kbar_ptr + offsets, acc_k, mask=mask)
    tl.store(Vbar_ptr + offsets, acc_v, mask=mask)


# splitting is only worth its overhead once a program's row sweep is long enough
# to dominate the FP32 partial write and the reduction pass that follow it
MIN_SPLIT_SWEEP_ROWS = 512


def _default_splits(B: int, h_kv: int, N: int, G: int, device: torch.device) -> int:
    """Smallest power-of-two split that fills approximately two SM waves.

    Two quantities determine the result. `estimated_programs` is the parallelism
    the column grid has without splitting, evaluated at the column pass's usual
    BLOCK_N. The sweep length is the work per program, which is `N * G` packed
    rows, since a program walks every query row of every group member.

    Splitting is applied when the grid underfills the device and the sweep is at
    least `MIN_SPLIT_SWEEP_ROWS` rows.
    """
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    estimated_programs = B * h_kv * triton.cdiv(N, 32)
    if estimated_programs >= 2 * sm_count:
        return 1
    if N * G < MIN_SPLIT_SWEEP_ROWS:
        return 1
    required = triton.cdiv(2 * sm_count, max(1, estimated_programs))
    # beyond four parts the reduction traffic outweighs the extra parallelism
    return min(4, max(1, 1 << (required - 1).bit_length()))


def _validate(Q, K, V, O, L, dO, dQ_bar, dK_bar, dV_bar):
    if Q.ndim != 4 or K.ndim != 4:
        raise ValueError("Q and K must be rank-4 tensors")
    if Q.device.type != "cuda":
        raise RuntimeError("packed GQA requires CUDA tensors")
    B, h_q, N, D = Q.shape
    if K.shape[0] != B or K.shape[2:] != (N, D):
        raise ValueError("K must share Q's batch, sequence, and head dimension")
    h_kv = K.shape[1]
    if h_q % h_kv:
        raise ValueError(f"H_Q={h_q} must be divisible by H_KV={h_kv}")
    G = h_q // h_kv
    if not 1 <= G <= 32:
        raise NotImplementedError("packed GQA supports 1 <= G <= 32")
    if D not in (32, 64, 128):
        raise NotImplementedError("packed GQA supports D in {32, 64, 128}")
    q_shape, kv_shape = (B, h_q, N, D), (B, h_kv, N, D)
    for name, t in (("O", O), ("dO", dO), ("dQ_bar", dQ_bar)):
        if t.shape != q_shape:
            raise ValueError(f"{name} must have shape {q_shape}")
    for name, t in (("V", V), ("dK_bar", dK_bar), ("dV_bar", dV_bar)):
        if t.shape != kv_shape:
            raise ValueError(f"{name} must have shape {kv_shape}")
    if L.shape != (B, h_q, N):
        raise ValueError(f"L must have shape {(B, h_q, N)}")
    tensors = (Q, K, V, O, L, dO, dQ_bar, dK_bar, dV_bar)
    if any(t.device != Q.device for t in tensors):
        raise ValueError("all tensors must be on one CUDA device")
    if len({t.dtype for t in tensors if t is not L}) != 1:
        raise ValueError("all vector tensors must have one dtype")
    if Q.dtype != torch.bfloat16:
        raise TypeError("packed GQA requires BF16 vectors")
    return B, h_q, h_kv, N, D, G


# ---------------------------------------------------------------------------
# launchers
# ---------------------------------------------------------------------------


def _gqa_core(
    Q,
    K,
    V,
    O,
    L,
    dO,
    dQ_bar,
    dK_bar,
    dV_bar,
    dQ,
    *,
    causal: bool = True,
    zero_tangents: tuple[bool, bool, bool] = (False, False, False),
    scale: float | None = None,
):
    """BF16 reuse-dQ GQA kernel boundary.

    Query-side tensors keep their `[B, H_Q, N, D]` layout. The kernels address
    packed row m as token `m // G` of group member `m % G`, so no tensor is
    packed or unpacked. `alpha`, `E`, and `D` are stored in packed row order and
    are internal to this pair of kernels.
    """
    zero_uq, zero_uk, zero_uv = zero_tangents
    B, h_q, N, D = Q.shape
    h_kv = K.shape[1]
    G = h_q // h_kv
    if not 1 <= G <= 32:
        raise NotImplementedError("GQA supports 1 <= G <= 32")
    n_packed = N * G
    score_scale = D**-0.5 if scale is None else float(scale)
    splits = _default_splits(B, h_kv, N, G, Q.device)
    # the square launcher's rule: tiles stay at storage precision on the BF16 path
    tile_dtype = tl.bfloat16

    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    O, dO, dQ_bar = O.contiguous(), dO.contiguous(), dQ_bar.contiguous()
    dK_bar, dV_bar, dQ = dK_bar.contiguous(), dV_bar.contiguous(), dQ.contiguous()
    # log2 units, matching common._reconstruct_p
    Lf = (L.float() * common.LOG2E_HOST).contiguous()

    Qbar = torch.empty_like(Q)
    dObar = torch.empty_like(Q)
    packed_stats = (B, h_kv, n_packed)
    alpha = torch.empty(packed_stats, device=Q.device, dtype=torch.float32)
    E = torch.empty_like(alpha)
    D_vec = torch.empty_like(alpha)

    # one row tile per (group member, token tile): head-major packing keeps a tile
    # inside one head even when BLOCK_M does not divide N
    _packed_row_pass[lambda META: (G * triton.cdiv(N, META["BLOCK_M"]), B * h_kv)](
        Qbar,
        dObar,
        alpha,
        E,
        Q,
        K,
        V,
        O,
        dO,
        dQ,
        dQ_bar,
        dK_bar,
        dV_bar,
        Lf,
        D_vec,
        N,
        n_packed,
        common.autotune_length_bucket(N),
        score_scale,
        G=G,
        H_KV=h_kv,
        ZERO_UQ=zero_uq,
        ZERO_UK=zero_uk,
        ZERO_UV=zero_uv,
        D_MODEL=D,
        WINDOW_SIZE=0,
        IS_CAUSAL=causal,
        DOT_INPUT_PRECISION="tf32",
        USE_BF16_SOURCE_DOTS=True,
        REUSE_DQ=True,
        TILE_DTYPE=tile_dtype,
    )

    Kbar = torch.empty_like(K)
    Vbar = torch.empty_like(V)
    if splits == 1:
        Kpartials, Vpartials = Kbar, Vbar
    else:
        partial_shape = (splits, B, h_kv, N, D)
        Kpartials = torch.empty(partial_shape, device=K.device, dtype=torch.float32)
        Vpartials = torch.empty_like(Kpartials)

    _packed_col_pass[lambda META: (triton.cdiv(N, META["BLOCK_N"]), B * h_kv, splits)](
        Kpartials,
        Vpartials,
        Q,
        K,
        V,
        dO,
        dQ_bar,
        dK_bar,
        dV_bar,
        Lf,
        D_vec,
        alpha,
        E,
        N,
        n_packed,
        common.autotune_length_bucket(N),
        score_scale,
        G=G,
        H_KV=h_kv,
        ZERO_UQ=zero_uq,
        ZERO_UK=zero_uk,
        ZERO_UV=zero_uv,
        SPLITS=splits,
        D_MODEL=D,
        WINDOW_SIZE=0,
        IS_CAUSAL=causal,
        DOT_INPUT_PRECISION="tf32",
        USE_BF16_SOURCE_DOTS=True,
        TILE_DTYPE=tile_dtype,
    )
    if splits > 1:
        total = B * h_kv * N * D
        _reduce_split_columns[(triton.cdiv(total, 256),)](
            Kbar,
            Vbar,
            Kpartials,
            Vpartials,
            total,
            SPLITS=splits,
            BLOCK=256,
            num_warps=4,
        )
    return Qbar, Kbar, Vbar, dObar


def bob_gqa(
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
    causal: bool = True,
    dQ: torch.Tensor,
    zero_tangents: tuple[bool, bool, bool] = (False, False, False),
    scale: float | None = None,
):
    """GQA/MQA second backward. Returns (Qbar, Kbar, Vbar, dObar).

    Query-side tensors are `[B, H_Q, N, D]` and KV-side tensors are
    `[B, H_KV, N, D]`, with `G = H_Q // H_KV` between 1 and 32. Requires BF16
    and causal or non-causal dense masking.
    """
    _validate(Q, K, V, O, L, dO, dQ_bar, dK_bar, dV_bar)
    if dQ.shape != Q.shape or dQ.device != Q.device or dQ.dtype != Q.dtype:
        raise ValueError("dQ must match Q's shape, device, and dtype")
    return _gqa_core(
        Q,
        K,
        V,
        O,
        L,
        dO,
        dQ_bar,
        dK_bar,
        dV_bar,
        dQ,
        causal=causal,
        zero_tangents=zero_tangents,
        scale=scale,
    )
