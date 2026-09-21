"""Triton components shared by the square, GQA, and rectangular BoB kernels.

Contents:

- the autotune configuration space and its pruning policy;
- the attention mask;
- the dot-precision policy;
- the row-pass and column-pass tile algebra;
- the delta reduction $D_i = \\langle dO_i, O_i \\rangle$.

Traversal order, output ownership, pointer arithmetic, and launch policy are
specific to each algorithm and are defined in `square.py`, `gqa.py`, and
`rectangular.py`.

The tile algebra, for one head, with $\\tau$ the softmax scale:

    S = tau Q K^T,   P = softmax(S),   O = P V

Given input gradients $U_Q, U_K, U_V$ on $dQ$, $dK$, $dV$:

    D_i        = <dO_i, O_i>
    dP         = dO V^T
    F          = tau (U_Q K^T + Q U_K^T)
    C          = dO U_V^T
    alpha_i    = sum_j P_ij F_ij
    dP_shift   = dP - D
    Pbar_circ  = C + F * dP_shift
    E_circ_i   = sum_j P_ij Pbar_circ_ij
    E_i        = E_circ_i - alpha_i D_i
"""

import functools

import triton
import triton.language as tl


# This value must be a tl.constexpr because @triton.jit bodies cannot read
# plain module-level floating-point globals. The original kernels redeclared
# `log2e` inside each kernel.
LOG2E = tl.constexpr(1.4426950408889634)
# the same constant for host-side use: launchers pre-scale L into log2 units so
# the kernels never rescale a score tile
LOG2E_HOST = 1.4426950408889634


# ---------------------------------------------------------------------------
# autotune configuration space
# ---------------------------------------------------------------------------

# The row body issues eleven dots per column tile and spills at d=128 even under
# REUSE_DQ, so warp count is an independent tuning variable:
# more warps cut per-thread register demand, fewer raise per-warp ILP. It is
# swept only for the tile shapes where it can plausibly matter, because the
# search space is multiplicative and cold-start autotuning is already the
# dominant first-call cost.
def _row_configs():
    return [
        triton.Config(
            {"BLOCK_M": block_m, "BLOCK_N": block_n, "GROUP_M": group_m},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_m, block_n in (
            (16, 16), (32, 16), (32, 32), (32, 64), (64, 16), (64, 32), (128, 32),
        )
        for num_stages in (1, 2, 3)
        for group_m in (4, 8)
        for num_warps in _warp_choices()
    ]


def _warp_choices():
    """Warp counts included in the autotune space.

    2 and 4 warps, plus 8 for devices whose tile caps reach the shapes that need
    it. 8 warps was measured at $d=32$, $64$, and $128$ on SM89 and was not
    selected in any of those cases, so `_tile_caps` drops it there and that
    device's search space is unchanged. It is selected on SM100, where the row
    pass runs a $64\times32$ tile that 4 warps cannot feed.

    Impact of sweeping warp count: 13% at $d=128$ and 8% at $d=64$, measured
    across three independent processes. Autotune cold start increases from 2.3 s
    to approximately 9 s per configuration key. The configuration space is the
    product of the swept dimensions.
    """
    return (2, 4, 8)


def _col_configs():
    # the column pass has larger live row tiles and less tolerance for wide N
    # at high D, so this family stays narrower than the row family.
    return [
        triton.Config(
            {"BLOCK_M": block_m, "BLOCK_N": block_n, "GROUP_N": group_n},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_m, block_n in ((16, 16), (16, 32), (32, 32), (64, 32),
                                 (128, 32), (64, 64))
        for num_stages in (2, 3)
        for group_n in (4, 8)
        for num_warps in _warp_choices()
    ]


ROW_PASS_CONFIGS = _row_configs()
COL_PASS_CONFIGS = _col_configs()

# Above this length, a historical run measured 1,909 seconds for one autotune
# pass at $N=32{,}768$, and per-configuration runtime grows as $N^2$.
# Bucketing the length in the autotune cache key here
# reuses the 32K winner at every longer length, so the long-context baseline is
# the schedule the autotuner picked where tuning is still affordable rather than
# a hand-picked tile. The bucket only helps inside one process: a fresh worker
# tunes again at its true length unless the winner is persisted separately.
AUTOTUNE_LENGTH_CEILING = 32768


def autotune_length_bucket(n: int) -> int:
    """The sequence length as the Triton autotune cache key sees it.

    Below the ceiling this is the identity, so shorter shapes keep tuning per
    length. The tiles are shape-agnostic for correctness -- every load and store
    is masked against the true `N` -- so reusing a longer shape's winner is a
    schedule choice, never a numerics one.
    """
    return min(int(n), AUTOTUNE_LENGTH_CEILING)


LEGACY_SMEM_LIMIT = 95 * 1024


def _smem_limit(device=None) -> int:
    """Per-block shared-memory limit for one device.

    `BOB_SMEM_LIMIT` overrides the value: `legacy` selects the historical
    95 KiB budget, `auto` or an unset value queries the device, and an integer
    sets the limit in bytes.
    """
    import os

    raw = os.environ.get("BOB_SMEM_LIMIT", "").strip().lower()
    if raw == "legacy":
        return LEGACY_SMEM_LIMIT
    if raw and raw != "auto":
        value = int(raw)
        if value <= 0:
            raise ValueError(f"BOB_SMEM_LIMIT must be positive, got {value}")
        return value

    import torch

    if not torch.cuda.is_available():
        return LEGACY_SMEM_LIMIT
    props = torch.cuda.get_device_properties(device)
    return int(
        getattr(props, "shared_memory_per_block_optin", None)
        or getattr(props, "shared_memory_per_block", LEGACY_SMEM_LIMIT)
    )


# heuristic accumulator-element ceiling used only to bound the search space
REGISTER_LIMIT = 16 * 1024


def _tile_caps(smem_limit: int, cc_major: int = 0):
    """Search-space caps for one device: (tile area, column $B_c$, accumulator
    elements, warps).

    These bound autotune cost. They are not resource limits -- the shared-memory
    and accumulator budgets below are, and Triton rejects anything that still
    does not fit. The constants were sized when SM89 was the only target, where
    99 KiB of shared memory keeps the score tile small, and applying them to
    SM90 and later rejected tiles that were never timed: the column pass gains
    1.25x on H100 and B200 from a $64\times64$ tile the SM89 caps forbid
    outright, and the row pass from $32\times32$ / $64\times32$.

    The 1024 area on the small-memory branch is itself a correction rather than
    the historical 512: $32\times32$ measures 1.107x over the selected
    $16\times16$ on an RTX 4090 Laptop, paired in-process, with FP64-oracle
    error equal to it in every output.
    """
    # SM90 and SM100 share the wide space. The one column tile SM100 still gets
    # wrong under the convert-before-transpose form is excluded by its own
    # guard below rather than by narrowing these caps, so every configuration
    # that is correct there stays reachable.
    if smem_limit >= 160 * 1024:
        return 4096, 64, 32 * 1024, 8
    return 1024, 32, REGISTER_LIMIT, 4


def _cc_major(device=None) -> int:
    """Compute-capability major version, or 0 when there is no CUDA device.

    `_tile_caps` needs it because shared-memory capacity alone does not separate
    SM90 from SM100, and one column configuration is correct on the first and
    wrong on the second.
    """
    import torch

    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_capability(device)[0]


def _streamed_itemsize(named_args) -> int:
    """Bytes per element of the operand tiles staged in shared memory.

    The `[B, H, N, D]` operands are tiled over `D_MODEL` and multi-buffered.
    `L`, `D`, `alpha`, and `E` are per-row FP32 vectors and occupy no streamed
    tile; the GQA split-K partials are rank 5. Rank selects the operands that
    apply.

    The return value is the widest matching operand, which bounds the
    requirement for a mixed-precision call. With no matching operand the return
    value is 4, the FP32 element size.
    """
    return max(
        (
            value.element_size()
            for value in named_args.values()
            if getattr(value, "ndim", None) == 4 and hasattr(value, "element_size")
        ),
        default=4,
    )


def _specialization_flag(name, named_args, kwargs):
    """Read a constexpr specialization flag from either argument mapping.

    Triton passes runtime arguments and constexprs to `early_config_prune`
    through different mappings depending on the call form, so both are searched.
    `D_MODEL` is resolved the same way.
    """
    value = named_args.get(name, kwargs.get(name))
    return bool(value) if value is not None else False


def _prune_configs(
    configs,
    named_args,
    *,
    axis: str,
    num_live_tiles: int,
    accum_factor: int,
    allow_stage1: bool,
    **kwargs,
):
    """Drop configurations that cannot fit shared memory or registers.

    `axis` selects which block dimension the program owns and which one its
    inner loop streams. A row program owns BLOCK_M rows and streams BLOCK_N
    columns; a column program owns BLOCK_N columns and streams BLOCK_M rows.
    Shared memory scales with the streamed dimension, accumulators with the
    owned one.
    """
    d_model = int(named_args.get("D_MODEL", kwargs.get("D_MODEL", 0)))
    if d_model == 0:
        return configs

    device = next(
        (
            value.device
            for value in named_args.values()
            if hasattr(value, "device") and value.device.type == "cuda"
        ),
        None,
    )
    smem_limit = _smem_limit(device)
    # `acc_R_shift` is dead under REUSE_DQ, and Triton does drop it: the spill
    # stack for the d=128 row kernel falls from 1872 to 1024 bytes when the flag
    # is set (measured with cuobjdump -res-usage). Budgeting for it anyway made
    # this reject wide-BLOCK_M configurations that now fit.
    if axis == "row" and _specialization_flag("REUSE_DQ", named_args, kwargs):
        accum_factor = max(1, accum_factor - 1)
    # tiles are staged at storage precision, so a BF16 call needs half the shared
    # memory an FP32 call does; budgeting 4 bytes unconditionally rejected
    # configurations that fit. Note this guard is not what binds at D_MODEL>=128,
    # where the tile-area guard below rejects first regardless of dtype.
    itemsize = _streamed_itemsize(named_args)
    cc = _cc_major(device)
    max_area, max_col_bn, accum_ceiling, max_warps = _tile_caps(smem_limit, cc)

    pruned = []
    for cfg in configs:
        block_m = cfg.kwargs["BLOCK_M"]
        block_n = cfg.kwargs["BLOCK_N"]
        streamed, owned = (block_n, block_m) if axis == "row" else (block_m, block_n)

        if d_model >= 128 and block_m * block_n > max_area:
            continue
        # the column pass keeps more live row tiles, so wide N is worse there --
        # but "wide" is a function of the shared memory the device actually has
        if axis == "col" and d_model >= 128 and block_n > max_col_bn:
            continue
        if cfg.num_warps > max_warps:
            continue
        # Convert-before-transpose in `col_pass_body` fixed every SM100 column
        # tile except this one: BLOCK_N=64 with BLOCK_M>=64 at 4 or more warps
        # still returns a $\widetilde K$ 1.4 relative L2 from the FP64 oracle,
        # while BLOCK_N=32 at the same BLOCK_M and warp count, and BLOCK_N=64 at
        # 2 warps, are all correct. SM90 is correct for all 96 configurations
        # tested. This applies to every d_model because the
        # caps above are d-conditional and would leave this reachable at d=64.
        if (axis == "col" and cc >= 10
                and block_n >= 64 and block_m >= 64 and cfg.num_warps >= 4):
            continue
        # GradMem 128k gate, Triton 3.4 / B200: BF16-source BM64/BN32
        # corrupts Qbar/dObar at 4 and 8 warps, stages 2 and 3 (N=4096,
        # 32768). BM32/BN32 passes the same FP64 oracle. Keep D128 row
        # tiles at <=32 until wider configurations are separately validated.
        # d=64 fails the same way: the autotuner's BM128/BN32, 8-warp row tile
        # gives ddQ/ddO 0.69-0.75/1.18-1.27 relative L2 from FP64 at N=32768.
        # Explicit tiles with BLOCK_M<=32 produce approximately 3e-3 relative
        # L2 and pass the same check. d=32 is untested.
        if (axis == "row" and cc >= 10 and d_model >= 64 and block_m > 32
                and _specialization_flag("USE_BF16_SOURCE_DOTS", named_args, kwargs)):
            continue
        if cfg.num_stages == 1 and not allow_stage1:
            continue
        if num_live_tiles * cfg.num_stages * streamed * d_model * itemsize > smem_limit:
            continue
        if accum_factor * owned * d_model > accum_ceiling:
            continue

        pruned.append(cfg)

    if pruned:
        return pruned
    # Falling back to `configs[0]` could hand back a configuration this function
    # had just rejected -- including num_stages=1, which the row policy forbids
    # outright rather than as a budget heuristic. Fall back to the cheapest
    # configuration that still satisfies the hard guard instead, and fail loudly
    # if even that does not exist rather than launching something illegal.
    legal = [c for c in configs if (allow_stage1 or c.num_stages > 1)
             and not (axis == "row" and cc >= 10 and d_model >= 64
                      and c.kwargs["BLOCK_M"] > 32
                      and _specialization_flag("USE_BF16_SOURCE_DOTS", named_args, kwargs))]
    if not legal:
        raise RuntimeError(
            f"no safe {axis}-pass configuration at D_MODEL={d_model}"
        )
    return [
        min(
            legal,
            key=lambda c: (
                c.num_stages
                * (c.kwargs["BLOCK_N"] if axis == "row" else c.kwargs["BLOCK_M"])
                * d_model,
                c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"],
            ),
        )
    ]


# (axis, num_live_tiles, accum_factor, allow_stage1)
PRUNE_POLICY = {
    "row": ("row", 4, 4, False),
    "col": ("col", 3, 2, False),
    "alpha_3pass": ("row", 2, 1, True),
    "row_3pass": ("row", 4, 3, True),
    "col_3pass": ("col", 3, 2, True),
}


def pruner(name: str):
    """`early_config_prune` callable for a named pass."""
    axis, num_live_tiles, accum_factor, allow_stage1 = PRUNE_POLICY[name]
    return functools.partial(
        _prune_configs,
        axis=axis,
        num_live_tiles=num_live_tiles,
        accum_factor=accum_factor,
        allow_stage1=allow_stage1,
    )


def normalize_window_size(window_size: int | None) -> int:
    if window_size is None:
        return 0
    window_size = int(window_size)
    if window_size < 0:
        raise ValueError(f"window_size must be >= 0, got {window_size}")
    return window_size


# ---------------------------------------------------------------------------
# tile primitives
# ---------------------------------------------------------------------------

@triton.jit
def attention_mask(
    off_m,
    off_n,
    mask_m,
    mask_n,
    IS_CAUSAL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    """Validity mask for one (row, column) tile.

    `off_m` and `off_n` are absolute positions. The square and GQA paths have
    zero offsets and pass tile-local indices. The rectangular path passes
    indices shifted by `Q_OFFSET` and `K_OFFSET`.
    """
    valid = mask_m[:, None] & mask_n[None, :]
    if IS_CAUSAL:
        valid = valid & (off_m[:, None] >= off_n[None, :])
        if WINDOW_SIZE > 0:
            valid = valid & ((off_m[:, None] - off_n[None, :]) < WINDOW_SIZE)
    return valid


@triton.jit
def mask_scores(
    qk,
    off_m,
    off_n,
    mask_m,
    mask_n,
    IS_CAUSAL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    valid = attention_mask(
        off_m, off_n, mask_m, mask_n,
        IS_CAUSAL=IS_CAUSAL,
        WINDOW_SIZE=WINDOW_SIZE,
    )
    return tl.where(valid, qk, float("-inf")), valid


@triton.jit
def source_dot(
    left,
    right,
    DOT_INPUT_PRECISION: tl.constexpr,
    USE_BF16_SOURCE_DOTS: tl.constexpr,
):
    if USE_BF16_SOURCE_DOTS:
        # these operands originated in bf16 storage, so this selects the bf16
        # tensor-core path without quantizing a newly derived fp32 quantity
        return tl.dot(left.to(tl.bfloat16), right.to(tl.bfloat16))
    return tl.dot(left, right, input_precision=DOT_INPUT_PRECISION)


@triton.jit
def _reconstruct_p(S_mn, L_m, valid):
    r"""Softmax tile reconstructed from log2-scaled scores and a log2-scaled LSE.

    Both operands are premultiplied by $\log_2 e$. The caller scales the score
    tile by `scale * LOG2E`, and the launcher converts `L` on the host. The
    reconstruction is then a single subtraction followed by `exp2`.
    """
    P_mn = tl.exp2(S_mn - L_m[:, None])
    return tl.where(valid, P_mn, 0.0)


# ---------------------------------------------------------------------------
# row pass
# ---------------------------------------------------------------------------

@triton.jit
def row_pass_body(
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
    IS_CAUSAL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    USE_BF16_SOURCE_DOTS: tl.constexpr,
    REUSE_DQ: tl.constexpr,
    ZERO_UQ: tl.constexpr,
    ZERO_UK: tl.constexpr,
    ZERO_UV: tl.constexpr,
):
    r"""Accumulate one column tile into the row-pass accumulators.

    `acc_R_shift` is accumulated when `REUSE_DQ` is false. When `REUSE_DQ` is
    true it is returned unchanged, which keeps the return arity uniform across
    the constexpr branches.

    `ZERO_UQ`, `ZERO_UK`, and `ZERO_UV` mark input gradients that are exactly
    zero, the value autograd supplies when the differentiated scalar does not
    depend on the matching first-order gradient. Each flag removes one source
    GEMM and its Layer-C counterpart:

        ZERO_UQ   removes $U_Q K^T$
        ZERO_UK   removes $Q U_K^T$ and $(P \odot X) U_K$
        ZERO_UV   removes $C = dO U_V^T$ and $P U_V$

    A tile whose flag is set is not read. Any tile of the correct shape is
    accepted in its place.
    """
    # S feeds nothing but the softmax reconstruction, so it is scaled straight
    # into log2 units. F below keeps the natural scale.
    S_mn = source_dot(
        Q_m,
        tl.trans(K_n),
        DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
        USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
    ) * (scale * LOG2E)
    S_mn, valid = mask_scores(
        S_mn, off_m, off_n, mask_m, mask_n,
        IS_CAUSAL=IS_CAUSAL,
        WINDOW_SIZE=WINDOW_SIZE,
    )
    P_mn = _reconstruct_p(S_mn, L_m, valid)

    dP_mn = source_dot(
        dO_m,
        tl.trans(V_n),
        DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
        USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
    )

    # F = tau (U_Q K^T + Q U_K^T); each half disappears with its own tangent
    if ZERO_UQ and ZERO_UK:
        F_mn = tl.zeros(dP_mn.shape, dtype=tl.float32)
    elif ZERO_UK:
        F_mn = source_dot(
            dQbar_m,
            tl.trans(K_n),
            DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
            USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
        ) * scale
    elif ZERO_UQ:
        F_mn = source_dot(
            Q_m,
            tl.trans(dKbar_n),
            DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
            USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
        ) * scale
    else:
        F_mn = (
            source_dot(
                dQbar_m,
                tl.trans(K_n),
                DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
                USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
            )
            + source_dot(
                Q_m,
                tl.trans(dKbar_n),
                DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
                USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
            )
        ) * scale

    dP_shift = dP_mn - D_m[:, None]
    PF = P_mn * F_mn

    if ZERO_UV:
        # C = 0, so P * (C + F * dP_shift) reduces to (P * F) * dP_shift. The
        # grouping is kept as-is: it is algebraically equal to the general form
        # but not bit-identical, and the zero-tangent path was written this way.
        PP_circ = PF * dP_shift
    else:
        C_mn = source_dot(
            dO_m,
            tl.trans(dVbar_n),
            DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
            USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
        )
        PP_circ = P_mn * (C_mn + F_mn * dP_shift)

    acc_alpha += tl.sum(PF, axis=1)
    acc_E_circ += tl.sum(PP_circ, axis=1)

    P_c = P_mn.to(Q_m.dtype)
    PF_c = PF.to(Q_m.dtype)
    PP_c = PP_circ.to(Q_m.dtype)
    PdPs_c = (P_mn * dP_shift).to(Q_m.dtype)

    if ZERO_UV:
        acc_dO_circ += tl.dot(PF_c, V_n, input_precision=DOT_INPUT_PRECISION)
    else:
        acc_dO_circ += (
            tl.dot(P_c, dVbar_n, input_precision=DOT_INPUT_PRECISION)
            + tl.dot(PF_c, V_n, input_precision=DOT_INPUT_PRECISION)
        )
    if ZERO_UK:
        acc_Omega += tl.dot(PP_c, K_n, input_precision=DOT_INPUT_PRECISION)
    else:
        acc_Omega += (
            tl.dot(PP_c, K_n, input_precision=DOT_INPUT_PRECISION)
            + tl.dot(PdPs_c, dKbar_n, input_precision=DOT_INPUT_PRECISION)
        )

    acc_B += tl.dot(P_c, K_n, input_precision=DOT_INPUT_PRECISION)

    if not REUSE_DQ:
        acc_R_shift += tl.dot(PdPs_c, K_n, input_precision=DOT_INPUT_PRECISION)

    return acc_alpha, acc_E_circ, acc_dO_circ, acc_Omega, acc_B, acc_R_shift


@triton.jit
def finish_row_outputs(
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
    REUSE_DQ: tl.constexpr,
):
    """Convert the row accumulators into (alpha, E, Qbar, dObar).

    `O_m` and `dQ_m` are epilogue operands, loaded after the column sweep. Each
    is read once, in this function, so a `[BLOCK_M, D_MODEL]` tile held across
    the sweep would occupy registers for the duration of the loop.

    `dQ_m` is read when `REUSE_DQ` is set. Otherwise any tile of the correct
    shape is accepted in its place.
    """
    alpha_m = acc_alpha
    E_m = acc_E_circ - alpha_m * D_m
    dObar_m = acc_dO_circ - alpha_m[:, None] * O_m

    if REUSE_DQ:
        Qbar_m = (
            (acc_Omega - acc_E_circ[:, None] * acc_B) * scale
            - alpha_m[:, None] * dQ_m
        )
    else:
        Qbar_m = (
            acc_Omega
            - alpha_m[:, None] * acc_R_shift
            - acc_E_circ[:, None] * acc_B
        ) * scale

    return alpha_m, E_m, Qbar_m, dObar_m


# ---------------------------------------------------------------------------
# column pass
# ---------------------------------------------------------------------------

@triton.jit
def col_pass_body(
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
    DOT_INPUT_PRECISION: tl.constexpr,
    USE_BF16_SOURCE_DOTS: tl.constexpr,
    ZERO_UQ: tl.constexpr,
    ZERO_UK: tl.constexpr,
    ZERO_UV: tl.constexpr,
):
    """Accumulate one row tile into the column-pass accumulators.

    `valid_mn` is supplied by the caller. The square path masks on tile-local
    indices, the rectangular path on absolute offsets, and the GQA path on the
    token index of each packed row. This is the only masking difference among
    the column kernels.

    `ZERO_UQ`, `ZERO_UK`, and `ZERO_UV` have the same meaning as in
    `row_pass_body`. With `ZERO_UQ` and `ZERO_UK` both set, $F = 0$ and
    $\\alpha = 0$, so $h = 0$ and the `acc_Vbar` contribution is zero.
    """
    S_mn = source_dot(
        Q_m,
        tl.trans(K_n),
        DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
        USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
    ) * (scale * LOG2E)
    S_mn = tl.where(valid_mn, S_mn, float("-inf"))
    P_mn = _reconstruct_p(S_mn, L_m, valid_mn)

    dP_mn = source_dot(
        dO_m,
        tl.trans(V_n),
        DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
        USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
    )

    if ZERO_UQ and ZERO_UK:
        F_mn = tl.zeros(dP_mn.shape, dtype=tl.float32)
    elif ZERO_UK:
        F_mn = source_dot(
            dQbar_m,
            tl.trans(K_n),
            DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
            USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
        )
    elif ZERO_UQ:
        F_mn = source_dot(
            Q_m,
            tl.trans(dKbar_n),
            DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
            USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
        )
    else:
        F_mn = source_dot(
            dQbar_m,
            tl.trans(K_n),
            DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
            USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
        ) + source_dot(
            Q_m,
            tl.trans(dKbar_n),
            DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
            USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
        )
    if ZERO_UV:
        C_mn = tl.zeros(dP_mn.shape, dtype=tl.float32)
    else:
        C_mn = source_dot(
            dO_m,
            tl.trans(dVbar_n),
            DOT_INPUT_PRECISION=DOT_INPUT_PRECISION,
            USE_BF16_SOURCE_DOTS=USE_BF16_SOURCE_DOTS,
        )
    F_mn *= scale

    dP_shift = dP_mn - D_m[:, None]
    h_mn = F_mn - alpha_m[:, None]
    Pbar_circ_mn = C_mn + F_mn * dP_shift
    Pbar_mn = Pbar_circ_mn - alpha_m[:, None] * dP_mn
    dS_mn = P_mn * dP_shift
    Sbar_mn = P_mn * (Pbar_mn - E_m[:, None])

    # the derived tiles are FP32 accumulator products while the operands sit at
    # storage precision on the BF16 path, and tl.dot rejects a mixed-dtype pair.
    # The row body already converts its derived operands this way; this is the
    # column body's missing counterpart, and it is an identity for FP32 tiles.
    # With U_Q = U_K = 0 we get F = 0 and alpha = 0, hence h = 0 and Vbar
    # vanishes identically. Verified against the FP64 oracle, which returns an
    # exact zero for a V-only tangent.
    # Convert before the transpose. `trans(X).to(bf16)` and
    # `trans(X.to(bf16))` hold the same values because elementwise rounding
    # commutes with a permutation, but they use different layout paths. The
    # required path converts first and then transposes at storage precision.
    # Transposing the FP32 register tile first produces an incorrect Kbar on
    # SM100 for BLOCK_M >= 64 at 4 or more warps.
    Pt_c = tl.trans((P_mn * h_mn).to(dO_m.dtype))
    if not (ZERO_UQ and ZERO_UK):
        acc_Vbar += tl.dot(Pt_c, dO_m, input_precision=DOT_INPUT_PRECISION)
    # the second term is dS^T U_Q, which vanishes with U_Q
    Sbar_t = tl.trans(Sbar_mn.to(Q_m.dtype))
    if ZERO_UQ:
        acc_Kbar += tl.dot(
            Sbar_t, Q_m, input_precision=DOT_INPUT_PRECISION,
        ) * scale
    else:
        acc_Kbar += (
            tl.dot(Sbar_t, Q_m, input_precision=DOT_INPUT_PRECISION)
            + tl.dot(tl.trans(dS_mn.to(dQbar_m.dtype)), dQbar_m,
                     input_precision=DOT_INPUT_PRECISION)
        ) * scale

    return acc_Kbar, acc_Vbar


# ---------------------------------------------------------------------------
# delta reduction
# ---------------------------------------------------------------------------

@triton.jit
def row_delta(dO_m, O_m):
    """$D_i = \\langle dO_i, O_i \\rangle$ for one row tile, accumulated in FP32.

    Computing the reduction inside a row program removes a separate kernel
    launch and one $O(N)$ HBM write and read per two-pass call.
    """
    return tl.sum(dO_m.to(tl.float32) * O_m.to(tl.float32), axis=1)
