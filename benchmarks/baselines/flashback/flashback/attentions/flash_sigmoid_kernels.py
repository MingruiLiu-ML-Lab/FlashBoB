import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from ..pallas_utils import tile_dot, triton_compiler_params
from functools import partial
from jax.experimental.pallas import triton as plgpu

DEFAULT_MASK_VALUE = -0.95 * float(np.finfo(np.dtype("float16")).max)

def check_shape(shape, expected_shape, name):
    assert shape == expected_shape, f"{name} shape: observed {shape} != expected {expected_shape}"

def blocking_for_inputs(RD_tensors, CD_tensors, RD_O_tensors, CD_O_tensors,
                        block_R, block_C, grid_over):
    '''
    CD_O_tensors: output tensors, None if no preloaded output
    RD_O_tensors: output tensors, None if no preloaded output
    execute pallas with inputs RD_tensors + CD_tensors and outputs RD_O_tensors + CD_O_tensors
    '''
    B, R, H, D = RD_tensors[0].shape
    C = CD_tensors[0].shape[1]

    # checks
    for i, tensor in enumerate(RD_tensors):
        check_shape(tensor.shape, (B, R, H, D), f'{i}th tensor')

    for i, tensor in enumerate(CD_tensors):
        check_shape(tensor.shape, (B, C, H, D), f'{i}th tensor')

    assert R % block_R == 0, f"{R} % {block_R} != 0"
    assert C % block_C == 0, f"{C} % {block_C} != 0"
    assert grid_over in ['R', 'C'], f"grid_over must be 'R' or 'C'"

    # build grid + blockspecs + default out shapes
    if grid_over == 'R':
        grid = (B, R // block_R, H)
        RD_blockspec = pl.BlockSpec((None, block_R, None, D), lambda b, r, h: (b, r, h, 0))
        CD_blockspec = pl.BlockSpec((None, C, None, D),  lambda b, r, h: (b, 0, h, 0))
    else:
        grid = (B, C // block_C, H)
        RD_blockspec = pl.BlockSpec((None, R, None, D), lambda b, c, h: (b, 0, h, 0))
        CD_blockspec = pl.BlockSpec((None, block_C, None, D),  lambda b, c, h: (b, c, h, 0))

    RD_out_shape = jax.ShapeDtypeStruct((B, R, H, D), jnp.float32)
    CD_out_shape = jax.ShapeDtypeStruct((B, C, H, D), jnp.float32)

    # build out_shapes and in_specs
    # for each out tensor: add a placeholder if None; otherwise accumulate in the input
    in_specs_RD = [RD_blockspec for _ in RD_tensors]
    in_specs_CD = [CD_blockspec for _ in CD_tensors]
    out_shape = []
    io_map = {}

    # final args:
    # RD_inputs, CD_inputs, RD_acc_outputs, CD_acc_outputs
    in_tensors = list(RD_tensors + CD_tensors)
    in_specs = in_specs_RD + in_specs_CD
    out_specs = []

    for o in RD_O_tensors:
        out_specs.append(RD_blockspec)
        if o is None:
            out_shape.append(RD_out_shape)
        else:
            out_shape.append(o)
            in_specs.append(RD_blockspec)
            in_tensors.append(o)

            # get index of output tensor and index of last tensor added
            oidx = len(out_shape) - 1
            iidx = len(in_tensors) - 1
            io_map[iidx] = oidx

    for o in CD_O_tensors:
        out_specs.append(CD_blockspec)
        if o is None:
            # just add shape; memory allocated by pallas
            out_shape.append(CD_out_shape)
        else:
            # otherwise: accumulate in corresponding input
            out_shape.append(o)
            in_specs.append(CD_blockspec)
            in_tensors.append(o)

            # get index of output tensor and index of last tensor added
            oidx = len(out_shape) - 1
            iidx = len(in_tensors) - 1
            io_map[iidx] = oidx

    return grid, in_specs, out_specs, out_shape, in_tensors, io_map


def approx_tanh(x: jax.Array) -> jax.Array:
  r"""Elementwise approximate hyperbolic tangent: :math:`\mathrm{tanh}(x)`.

  See
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh.
  """
  if x.dtype == jnp.float16:
    asm = "tanh.approx.f16 $0, $1;"
    constraint = "h"
  elif x.dtype == jnp.bfloat16:
    asm = "tanh.approx.bf16 $0, $1;"
    constraint = "h"
  elif x.dtype == jnp.float32:
    asm = "tanh.approx.f32 $0, $1;"
    constraint = "f"
  else:
    raise TypeError(f"approx_tanh does not accept {x.dtype} arrays")

  elementwise_inline_asm = plgpu.elementwise_inline_asm
  [result] = elementwise_inline_asm(
      asm,
      args=[x],
      constraints=f"={constraint},{constraint}",
      pack=1,
      result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
  )
  return result


def tanh(x):
    return approx_tanh(x)

@partial(jax.jit, static_argnames=['causal', 'bias', 'softmax_factor',
                                   'block', 'precision', 'num_stages',
                                   'num_warps'])
def sigmoid_attn_bwdbwd(Q, K, V, dO, dot_dQ, dot_dK, dot_dV, causal, bias,
                        softmax_factor, block, precision,
                        num_stages, num_warps):
    # need to load up:
    # row by row: Q, dO, dot_dQ
    # col by col: K, V, dot_dK, dot_dV
    # need to write out:
    # row by row: dot_Q, dot_dO
    # col by col: dot_K, dot_V
    # therefore: block by columns;
    block_R, block_C = block, block

    B, R, H, D = Q.shape
    C = K.shape[1]
    check_shape(K.shape, (B, C, H, D), 'K shape')
    check_shape(V.shape, (B, C, H, D), 'V shape')
    check_shape(dO.shape, (B, R, H, D), 'dO shape')
    check_shape(dot_dQ.shape, (B, R, H, D), 'dot_dQ shape')
    check_shape(dot_dK.shape, (B, C, H, D), 'dot_dK shape')
    check_shape(dot_dV.shape, (B, C, H, D), 'dot_dV shape')

    def bwdbwd_kernel(Q_ref, dO_ref, dot_dQ_ref, K_ref, V_ref, dot_dK_ref,
                      dot_dV_ref, dot_Q_ref, dot_dO_ref, _, __, dot_K_ref,
                      dot_V_ref):
        # load in columnwise shitters
        k, v = K_ref[...], V_ref[...]
        dot_dk, dot_dv = dot_dK_ref[...], dot_dV_ref[...]

        check_shape(k.shape, (block_C, D), 'k shape')
        check_shape(v.shape, (block_C, D), 'v shape')
        check_shape(dot_dk.shape, (block_C, D), 'dot_dk shape')
        check_shape(dot_dv.shape, (block_C, D), 'dot_dv shape')

        start_c = pl.program_id(axis=1)
        acc_dot_k = jnp.zeros((block_C, D), dtype=jnp.float32)
        acc_dot_v = jnp.zeros((block_C, D), dtype=jnp.float32)

        dot = partial(tile_dot, precision=precision)

        def loop_body(start_r, accs):
            start_r_abs = start_r * block_R
            r_slice = pl.Slice(start_r_abs, block_R)
            row_idxs = (r_slice, slice(None))
            q = pl.load(Q_ref, row_idxs)
            do = pl.load(dO_ref, row_idxs)
            dot_dq = pl.load(dot_dQ_ref, row_idxs)

            dot_ds = dot(q, dot_dk.T) + dot(dot_dq, k.T)
            if softmax_factor is not None:
                dot_ds = dot_ds * softmax_factor

            # setup from backward pass
            s = make_s(q, k, precision, bias, softmax_factor, causal, start_r,
                       block_R, start_c, block_C)
            # s_tanh = tanh(s * 0.5)

            # # # prev: 5 ops
            # # p = 0.5 * (1. + s_tanh) # 2 ops with s_tanh
            # # q1m_tanh = 0.25 * (1. - s_tanh**2) # 3 ops with s_tanh

            # new: 4 ops
            # halftanh = 0.5 * s_tanh
            # halftanh2 = halftanh**2
            # p = 0.5 + halftanh
            # q1m_tanh = 0.25 - halftanh2

            p = jax.nn.sigmoid(s)
            q1m_tanh = p - p**2 #

            dp = dot(do, v.T)
            ds = dp * q1m_tanh

            # onto the backward pass
            dot_dp = q1m_tanh * dot_ds
            # (-dP * S_tanh) * dot_dp
            # dot_s_from_ds = dot_dp * -dp * (1. - 2 * p) #  (-dp * s_tanh)
            # dot_s_from_ds = -dot_dp * dp * s_tanh
            # dot_s_from_ds = -dot_dp * dp * s_tanh
            dot_s_from_ds = dot_dp * dp * (1. - 2 * p) #  (-dp * s_tanh)

            dot_v = dot(dot_dp.T, do)
            dot_do_from_dp = dot(dot_dp, v)
            dot_do_from_dv = dot(p, dot_dv)

            dot_do = dot_do_from_dp + dot_do_from_dv

            dot_p_from_dv = dot(do, dot_dv.T)
            dot_p = dot_p_from_dv

            dot_k_from_dq = dot(ds.T, dot_dq)
            dot_q_from_dk = dot(ds, dot_dk)

            fadot_ds_post_from_p = dot_p * q1m_tanh
            fadot_ds_pre = fadot_ds_post_from_p + dot_s_from_ds

            dot_k_from_p = dot(fadot_ds_pre.T, q)
            dot_q_from_p = dot(fadot_ds_pre, k)

            dot_k = (dot_k_from_dq + dot_k_from_p)
            dot_q = (dot_q_from_dk + dot_q_from_p)

            if softmax_factor is not None:
                dot_k = dot_k * softmax_factor
                dot_q = dot_q * softmax_factor

            pl.atomic_add(dot_Q_ref, r_slice, dot_q.astype(dot_Q_ref.dtype))
            pl.atomic_add(dot_dO_ref, r_slice, dot_do.astype(dot_dO_ref.dtype))
            acc_dot_dk, acc_dot_dv = accs
            return acc_dot_dk + dot_k, acc_dot_dv + dot_v

        num_row_blocks = R // block_R
        start_block = start_c if causal else 0
        acc_dot_dk, acc_dot_dv = jax.lax.fori_loop(start_block, num_row_blocks, loop_body, (acc_dot_k, acc_dot_v))
        dot_K_ref[...] = acc_dot_dk
        dot_V_ref[...] = acc_dot_dv

    dot_Q_output = jnp.zeros((B, R, H, D), dtype=jnp.float32)
    dot_dO_output = jnp.zeros((B, R, H, D), dtype=jnp.float32)
    RD_outputs = [dot_Q_output, dot_dO_output] # dot_Q, dot_dO
    CD_outputs = [None, None] # dot_K, dot_V
    RD_inputs = [Q, dO, dot_dQ]
    CD_inputs = [K, V, dot_dK, dot_dV]

    ret = blocking_for_inputs(RD_inputs, CD_inputs, RD_outputs,
                              CD_outputs, block_R, block_C, grid_over='C')
    grid, in_specs, out_specs, out_shape, in_tensors, io_map = ret
    dot_Q, dot_dO, dot_K, dot_V = pl.pallas_call(bwdbwd_kernel,
                                                out_shape=out_shape,
                                                grid=grid,
                                                compiler_params=triton_compiler_params(
                                                    num_stages=num_stages,
                                                    num_warps=num_warps,
                                                ),
                                                input_output_aliases=io_map,
                                                in_specs=in_specs,
                                                out_specs=out_specs)(*in_tensors)
    return dot_Q, dot_dO, dot_K, dot_V


# last arguments are for accumulating
@partial(jax.jit, static_argnames=['causal', 'bias', 'softmax_factor',
                                   'block', 'precision', 'num_stages',
                                   'num_warps'])
def sigmoid_attn_bwd(Q, K, V, dO, causal, bias, softmax_factor, block=None,
                     precision=None, acc_dQ=None, acc_dK=None, acc_dV=None,
                     num_stages=None, num_warps=None):
    assert num_stages is not None
    assert num_warps is not None
    assert precision is not None
    assert block is not None
    B, R, H, D = Q.shape
    C = K.shape[1]
    block_C, block_R = block, block

    def bwd_kernel(*args):
        if acc_dK is None:
            Q_ref, dO_ref, K_ref, V_ref, dQ_ref, _, dK_ref, dV_ref = args
        else:
            Q_ref, dO_ref, K_ref, V_ref, dQ_ref, dK_ref, dV_ref, _, _, _ = args


        k = K_ref[...]
        v = V_ref[...]

        # assert q.shape == (1, block_R, 1, D), f'q shape {q.shape} != {(block_B, block_R, block_H, D)}'
        check_shape(k.shape, (block_C, D), 'k shape')
        check_shape(v.shape, (block_C, D), 'v shape')

        # batch, row, head, dim
        # acc_dq = jnp.zeros((block_R, D), dtype=jnp.float32)
        acc_dk = jnp.zeros((block_C, D), dtype=jnp.float32)
        acc_dv = jnp.zeros((block_C, D), dtype=jnp.float32)
        start_c = pl.program_id(axis=1)

        dot = partial(tile_dot, precision=precision)

        def loop_body(start_r, accs):
            start_r_abs = start_r * block_R
            r_slice = pl.Slice(start_r_abs, block_R)
            row_idxs = (r_slice, slice(None))
            q = pl.load(Q_ref, row_idxs)
            do = pl.load(dO_ref, row_idxs)

            s = make_s(q, k, precision, bias, softmax_factor, causal, start_r,
                       block_R, start_c, block_C)

            # P = jax.nn.sigmoid(s)
            # s_tanh = tanh(s * 0.5)
            # half_tanh = 0.5 * s_tanh
            # P = 0.5 + half_tanh
            # dP = dot(do, v.T)
            # dS = dP * (0.25 - half_tanh**2) # 3 ops
            # # dS = dP * 0.25 * (1. - s_tanh**2) 4 ops
            # dS = dP * P * (1. - P)

            # OLD WITHOUT TANH
            P = jax.nn.sigmoid(s)
            dP = dot(do, v.T)
            dS = dP * (P - P**2)

            if softmax_factor is not None:
                dS = dS * softmax_factor

            dQ = dot(dS, k)
            dK = dot(dS.T, q)
            dV = dot(P.T, do)

            acc_dk, acc_dv = accs
            acc_dk = acc_dk + dK
            acc_dv = acc_dv + dV

            pl.atomic_add(dQ_ref, r_slice, dQ.astype(dQ_ref.dtype))
            return (acc_dk, acc_dv)

        num_row_blocks = R // block_R
        start_block = start_c if causal else 0

        acc_dk, acc_dv = jax.lax.fori_loop(start_block, num_row_blocks, loop_body, (acc_dk, acc_dv))
        dK_ref[...] = acc_dk.astype(dK_ref.dtype)
        dV_ref[...] = acc_dv.astype(dV_ref.dtype)

    RD_inputs = [Q, dO]
    CD_inputs = [K, V]

    # outputs:
    all_none = acc_dQ is None and acc_dK is None and acc_dV is None
    all_not_none = acc_dQ is not None and acc_dK is not None and acc_dV is not None
    assert all_none or all_not_none, "either all or none of the outputs must be None"

    RD_outputs = [acc_dQ if acc_dQ is not None else jnp.zeros((B, R, H, D), dtype=jnp.float32)] # dQ
    CD_outputs = [acc_dK, acc_dV] # dK, dV

    ret = blocking_for_inputs(RD_inputs, CD_inputs, RD_outputs,
                              CD_outputs, block_R, block_C, grid_over='C')
    grid, in_specs, out_specs, out_shape, in_tensors, io_map = ret
    dQ, dK, dV = pl.pallas_call(bwd_kernel,
                                out_shape=out_shape,
                                grid=grid,
                                compiler_params=triton_compiler_params(
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                ),
                                input_output_aliases=io_map,
                                in_specs=in_specs,
                                out_specs=out_specs)(*in_tensors)
    return dQ, dK, dV

def make_s(q, k, precision, bias, softmax_factor, causal, start_r, block_R,
           start_c, block_C):
    qkT = tile_dot(q, k.T, precision=precision)

    if softmax_factor is not None:
        qkT = qkT * softmax_factor

    if bias is not None:
        qkT = qkT + bias

    # row indices: correspond to outputs
    # col indices: correspond to inputs
    # causal mask: include only if row index <= col index
    if causal:
        row_indices = start_r * block_R + jnp.arange(block_R)
        col_indices = start_c * block_C + jnp.arange(block_C)
        causal_mask = row_indices[:, None] >= col_indices[None, :]
        qkT = jnp.where(causal_mask, qkT, DEFAULT_MASK_VALUE)

    return qkT

@partial(jax.jit, static_argnames=['causal', 'bias', 'softmax_factor',
                                   'block', 'precision',
                                   'num_stages', 'num_warps'])
def sigmoid_attn_fwd(Q, K, V, *, causal, bias, softmax_factor, block,
                     precision, num_stages, num_warps):
    # in shapes:
    # - Q: (B, R, H, D)
    # - K: (B, C, H, D)
    # - V: (B, C, H, D)
    # out shape:
    # - O: (B, R, H, D)
    _, _, _, D = Q.shape
    C = K.shape[1]
    block_R, block_C = block, block

    def fwd_kernel(Q_ref, K_ref, V_ref, O_ref):
        q = Q_ref[...]

        # assert q.shape == (1, block_R, 1, D), f'q shape {q.shape} != {(block_B, block_R, block_H, D)}'
        check_shape(q.shape, (block_R, D), 'q shape')

        # batch, row, head, dim
        acc_o = jnp.zeros((block_R, D), dtype=jnp.float32)
        start_r = pl.program_id(axis=1)

        def body(start_c, acc_o):
            start_c_abs = start_c * block_C
            c_slice = pl.Slice(start_c_abs, block_C)
            col_idxs = (c_slice, slice(None))
            k = pl.load(K_ref, col_idxs)
            v = pl.load(V_ref, col_idxs)

            qkT = make_s(q, k, precision, bias, softmax_factor, causal, start_r,
                         block_R, start_c, block_C)
            qkT = jax.nn.sigmoid(qkT)
            # qkT = 0.5 * (1. + jnp.tanh(qkT * 0.5))

            this_o = tile_dot(qkT, v, precision=precision)
            acc_o = acc_o + this_o
            return acc_o

        # this is across columns in the in the keys (which are inputs)
        # optimization: only loop across the columns for valid blocks if causal
        if causal:
            num_col_blocks = start_r + 1
        else:
            num_col_blocks = C // block_C

        acc_o = jax.lax.fori_loop(0, num_col_blocks, body, acc_o)
        O_ref[...] = acc_o

    ret = blocking_for_inputs([Q], [K, V], [None], [], block_R, block_C,
                              grid_over='R')

    grid, in_specs, out_specs, out_shape, in_tensors, io_map = ret
    return pl.pallas_call(fwd_kernel,
                          out_shape=out_shape,
                          grid=grid,
                          input_output_aliases=io_map,
                          compiler_params=triton_compiler_params(
                              num_stages=num_stages,
                              num_warps=num_warps,
                          ),
                          in_specs=in_specs,
                          out_specs=out_specs)(*in_tensors)[0]

