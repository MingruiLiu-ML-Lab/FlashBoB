import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax import lax
import functools
import numpy as np
from ..pallas_utils import tile_dot, triton_compiler_params

DEFAULT_MASK_VALUE = -0.95 * float(np.finfo(np.dtype("float16")).max)

def fused_kernel(
    q_ref, k_ref, v_ref, dO_ref, dot_dQ_ref, dot_dK_ref, dot_dV_ref, m_ref, l_ref, delta_ref,
    dot_K_ref, dot_V_ref, # zero-initialized outputs
    dot_Q_ref, dot_K_dummy_ref, dot_V_dummy_ref, dot_dO_ref,
    num_heads, head_dim, block_rows, block_cols, causal, attn_softcap,
    softmax_factor, precision):

    seq_len = k_ref.shape[0]

    row_idx = pl.program_id(2)

    block_Q = pl.load(q_ref, (pl.dslice(None), pl.dslice(None)))
    block_dot_dQ = pl.load(dot_dQ_ref, (pl.dslice(None), pl.dslice(None)))
    block_dO = pl.load(dO_ref, (pl.dslice(None), pl.dslice(None)))

    block_m = pl.load(m_ref, (pl.dslice(None),))
    block_l = pl.load(l_ref, (pl.dslice(None),))
    block_delta = pl.load(delta_ref, (pl.dslice(None),))
    block_lPdP = block_delta * block_l

    z_1 = jnp.zeros((block_rows,), dtype=jnp.float32)
    z_21 = jnp.zeros((block_rows,), dtype=jnp.float32)
    z_22 = jnp.zeros((block_rows,), dtype=jnp.float32)
    z_u = jnp.zeros((block_rows,), dtype=jnp.float32)

    def loop_body(apply):
        def f(block_start_idx, carry):

            span_q = row_idx * block_rows + jnp.arange(block_rows)
            span_k = block_start_idx * block_cols + jnp.arange(block_cols)
            causal_mask = span_q[:, None] >= span_k[None, :]

            block_K = pl.load(k_ref, (pl.dslice(block_start_idx * block_cols, block_cols), pl.dslice(None)))
            block_V = pl.load(v_ref, (pl.dslice(block_start_idx * block_cols, block_cols), pl.dslice(None)))
            block_dot_dK = pl.load(dot_dK_ref, (pl.dslice(block_start_idx * block_cols, block_cols), pl.dslice(None)))
            block_dot_dV = pl.load(dot_dV_ref, (pl.dslice(block_start_idx * block_cols, block_cols), pl.dslice(None)))

            block_S = tile_dot(block_Q, block_K.T, precision)
            if softmax_factor is not None:
                block_S = block_S * softmax_factor

            if attn_softcap is not None:
                block_S_tanh = jnp.tanh(block_S / attn_softcap)
                block_S = block_S_tanh * attn_softcap
            else:
                block_S_tanh = None

            if causal:
                block_S = jnp.where(causal_mask, block_S, DEFAULT_MASK_VALUE)

            block_P = jnp.exp(block_S - block_m[..., None]) / block_l[..., None]
            block_dP = tile_dot(block_dO, block_V.T, precision)

            block_dS_post = block_P * (block_dP - block_lPdP[:, None])
            if attn_softcap is not None:
                block_dS_pre = block_dS_post * (1 - block_S_tanh**2)
            else:
                block_dS_pre = block_dS_post

            block_dot_P_from_dV = tile_dot(block_dO, block_dot_dV.T, precision)
            
            block_dot_dS_pre = tile_dot(block_Q, block_dot_dK.T, precision) + tile_dot(block_dot_dQ, block_K.T, precision)
            if softmax_factor is not None:
                block_dot_dS_pre *= softmax_factor

            if attn_softcap is not None:
                block_dot_dS_post = block_dot_dS_pre * (1 - block_S_tanh**2)
            else:
                block_dot_dS_post = block_dot_dS_pre

            return apply(carry=carry, block_start_idx=block_start_idx, 
                         block_K=block_K,
                         block_V=block_V,
                         block_dot_dK=block_dot_dK,
                         block_P=block_P,
                         block_dP=block_dP,
                         block_lPdP=block_lPdP,
                         block_dot_P_from_dV=block_dot_P_from_dV,
                         block_dot_dV=block_dot_dV,
                         block_dot_dS_post=block_dot_dS_post,
                         block_dot_dS_pre=block_dot_dS_pre,
                         block_S_tanh=block_S_tanh,
                         block_dS_post=block_dS_post,
                         block_dS_pre=block_dS_pre)
        return f

    def calculate_sums(carry, block_start_idx,
                       block_K, block_V, block_dot_dK, block_P, block_dP,
                       block_lPdP, block_dot_P_from_dV,
                       block_dot_dV, block_dot_dS_post, block_dot_dS_pre,
                       block_S_tanh, block_dS_post, block_dS_pre):
        z_1, z_21, z_22, z_u = carry

        V_1 = block_dot_dS_post * block_dP - block_dot_dS_post * block_lPdP[:, None]
        V_2 = -block_dP
        block_P_hadamard_dot_dS_post = block_P * block_dot_dS_post

        z_1 += block_P_hadamard_dot_dS_post.sum(axis=-1)
        V_1 += block_dot_P_from_dV

        W_11 = block_P * V_1
        W_12 = block_P * V_2

        z_21 += W_11.sum(axis=-1)
        z_22 += W_12.sum(axis=-1)
        z_u = z_1

        return z_1, z_21, z_22, z_u

    if causal:
        num_rows = (row_idx + 1) * block_rows
        loop_limit = pl.cdiv(num_rows, block_cols)
    else:
        loop_limit = pl.cdiv(seq_len, block_cols)

    z_1, z_21, z_22, z_u = lax.fori_loop(0, loop_limit, loop_body(calculate_sums), (z_1, z_21, z_22, z_u))

    z_22 = z_22 * z_1
    z2_tot = z_21 + z_22
    block_P_hadamard_dot_P_sum = z2_tot
    block_P_hadamard_dot_dS_post_sum = z_1

    dot_Q_accum = jnp.zeros((block_rows, head_dim), dtype=jnp.float32)
    dot_K_accum = jnp.zeros((block_rows, head_dim), dtype=jnp.float32)
    dot_V_accum = jnp.zeros((block_rows, head_dim), dtype=jnp.float32)
    dot_dO_accum = jnp.zeros((block_rows, head_dim), dtype=jnp.float32)

    def finalize_row(carry, block_start_idx,
                     block_K, block_V, block_dot_dK, block_P, block_dP,
                     block_lPdP, block_dot_P_from_dV, block_dot_dV,
                     block_dot_dS_post, block_dot_dS_pre,
                     block_S_tanh, block_dS_post, block_dS_pre):

        dot_Q_accum, dot_K_accum, dot_V_accum, dot_dO_accum = carry

        if softmax_factor is not None:
            block_dS_pre *= softmax_factor

        block_dot_Q_from_dK = tile_dot(block_dS_pre, block_dot_dK, precision) # * softmax_factor
        block_dot_K_from_dQ = tile_dot(block_dS_pre.T, block_dot_dQ, precision) # * softmax_factor
        
        dot_Q_accum += block_dot_Q_from_dK

        col_slice = pl.dslice(block_start_idx * block_cols, block_cols)

        block_dot_dP = block_P * block_dot_dS_post - block_P * block_P_hadamard_dot_dS_post_sum[:, None]

        block_dot_V = tile_dot(block_dot_dP.T, block_dO, precision)
        pl.atomic_add(dot_V_ref, (col_slice, pl.dslice(None)), block_dot_V.astype(dot_V_ref.dtype))

        block_dot_dO_from_dP = tile_dot(block_dot_dP, block_V, precision)
        block_dot_dO_from_dV = tile_dot(block_P, block_dot_dV, precision)
        dot_dO_accum += block_dot_dO_from_dP + block_dot_dO_from_dV

        block_dot_P_from_dS = block_dot_dS_post * block_dP - \
                              block_dot_dS_post * block_lPdP[:, None] - \
                              block_dP * block_P_hadamard_dot_dS_post_sum[:, None]

        block_dot_P = block_dot_P_from_dV + block_dot_P_from_dS

        block_fadot_dS_post_from_P = block_P * block_dot_P - block_P * block_P_hadamard_dot_P_sum[:, None]

        if attn_softcap is not None:
            block_dot_tanh = -2 * block_S_tanh * block_dS_post * block_dot_dS_pre
            block_fadot_dS_post_from_dS_pre = block_dot_tanh / attn_softcap
        else:
            block_fadot_dS_post_from_dS_pre = 0

        block_fadot_dS_post = block_fadot_dS_post_from_dS_pre + block_fadot_dS_post_from_P
        if attn_softcap is not None:
            block_fadot_dS_pre = block_fadot_dS_post * (1 - block_S_tanh**2)
        else:
            block_fadot_dS_pre = block_fadot_dS_post

        if softmax_factor is not None:
            block_fadot_dS_pre *= softmax_factor

        block_dot_K_from_P = tile_dot(block_fadot_dS_pre.T, block_Q, precision) # * softmax_factor
        block_dot_Q_from_P = tile_dot(block_fadot_dS_pre, block_K, precision) # * softmax_factor

        block_dot_K = (block_dot_K_from_dQ + block_dot_K_from_P)

        pl.atomic_add(dot_K_ref, (col_slice, pl.dslice(None)), block_dot_K.astype(dot_K_ref.dtype))

        dot_Q_accum += block_dot_Q_from_P

        return dot_Q_accum, dot_K_accum, dot_V_accum, dot_dO_accum

    dot_Q_accum, dot_K_accum, dot_V_accum, dot_dO_accum = lax.fori_loop(0, loop_limit, loop_body(finalize_row), 
                                                                        (dot_Q_accum, dot_K_accum, dot_V_accum, dot_dO_accum))
    row_idx = pl.program_id(2)
    row_slice = pl.dslice(row_idx * block_rows, block_rows)
    pl.store(dot_Q_ref, (pl.dslice(None), pl.dslice(None)), dot_Q_accum.astype(dot_Q_ref.dtype))
    pl.atomic_add(dot_K_ref, (row_slice, pl.dslice(None)), dot_K_accum.astype(dot_K_ref.dtype))
    pl.atomic_add(dot_V_ref, (row_slice, pl.dslice(None)), dot_V_accum.astype(dot_V_ref.dtype))
    pl.store(dot_dO_ref, (pl.dslice(None), pl.dslice(None)), dot_dO_accum.astype(dot_dO_ref.dtype))

# dirty hack bc pallas reversed the order of these things
def RevBlockSpec(index_map, block_spec):
    assert isinstance(block_spec, tuple)
    assert callable(index_map)
    return pl.BlockSpec(block_spec, index_map)

def launch_fused(q, k, v, dO, dot_dQ, dot_dK, dot_dV, m, l, delta,
                 causal, attn_softcap, softmax_factor, precision, block, num_stages, num_warps):
    # print("NOTE! Using Jax convention of batch_size, seq_len, num_heads, head_dim")
    batch_size, seq_len, num_heads, head_dim = q.shape
    # print(f"{batch_size=} {seq_len=} {num_heads=} {head_dim=}")

    block_rows = block
    block_cols = block
    grid = (batch_size, num_heads, pl.cdiv(seq_len, block_rows))
    # print(f"{grid=}")

    # do not need atomic adds
    dot_Q = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    dot_dO = jax.ShapeDtypeStruct(shape=dO.shape, dtype=dO.dtype)

    # need atomic adds
    dot_K = jnp.zeros(shape=k.shape, dtype=k.dtype)
    dot_V = jnp.zeros(shape=v.shape, dtype=v.dtype)

    # tmp assertions
    assert dot_Q.shape == dot_dQ.shape == q.shape
    assert dot_K.shape == dot_dK.shape == k.shape
    assert dot_V.shape == dot_dV.shape == v.shape
    assert dot_dO.shape == dO.shape

    assert m.shape == l.shape == delta.shape
    assert q.shape == dO.shape

    kernel = functools.partial(fused_kernel,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               block_rows=block_rows,
                               block_cols=block_cols,
                               causal=causal,
                               attn_softcap=attn_softcap,
                               softmax_factor=softmax_factor,
                               precision=precision)

    is_mha = len(k.shape) == 4
    if is_mha:
        k_spec = RevBlockSpec(lambda b, h, r: (b, 0, h, 0), (None, seq_len, None, head_dim))
        v_spec = RevBlockSpec(lambda b, h, r: (b, 0, h, 0), (None, seq_len, None, head_dim))

        dot_dK_spec = RevBlockSpec(lambda b, h, r: (b, 0, h, 0), (None, seq_len, None, head_dim))
        dot_dV_spec = RevBlockSpec(lambda b, h, r: (b, 0, h, 0), (None, seq_len, None, head_dim))

        dot_K_spec = RevBlockSpec(lambda b, h, r: (b, 0, h, 0), (None, seq_len, None, head_dim))
        dot_V_spec = RevBlockSpec(lambda b, h, r: (b, 0, h, 0), (None, seq_len, None, head_dim))

    else:
        k_spec = RevBlockSpec(lambda b, h, r: (b, 0, 0), (None, seq_len, head_dim))
        v_spec = RevBlockSpec(lambda b, h, r: (b, 0, 0), (None, seq_len, head_dim))

        dot_dK_spec = RevBlockSpec(lambda b, h, r: (b, 0, 0), (None, seq_len, head_dim))
        dot_dV_spec = RevBlockSpec(lambda b, h, r: (b, 0, 0), (None, seq_len, head_dim))

        dot_K_spec = RevBlockSpec(lambda b, h, r: (b, 0, 0), (None, seq_len, head_dim))
        dot_V_spec = RevBlockSpec(lambda b, h, r: (b, 0, 0), (None, seq_len, head_dim))

    in_specs = [
        RevBlockSpec(lambda b, h, r: (b, r, h, 0), (None, block_rows, None, head_dim)), # q
        k_spec, v_spec,

        RevBlockSpec(lambda b, h, r: (b, r, h, 0), (None, block_rows, None, head_dim)), # dO

        RevBlockSpec(lambda b, h, r: (b, r, h, 0), (None, block_rows, None, head_dim)), # dot_dQ
        dot_dK_spec, dot_dV_spec,

        RevBlockSpec(lambda b, h, r: (b, h, r), (None, None, block_rows)), # m
        RevBlockSpec(lambda b, h, r: (b, h, r), (None, None, block_rows)), # l
        RevBlockSpec(lambda b, h, r: (b, h, r), (None, None, block_rows)), # delta

        dot_K_spec, dot_V_spec
    ]

    dot_Q, dot_K, dot_V, dot_dO = pl.pallas_call(
        kernel=kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=[
            RevBlockSpec(lambda b, h, r: (b, r, h, 0), (None, block_rows, None, head_dim)), # dot_Q
            dot_K_spec,
            dot_V_spec,
            RevBlockSpec(lambda b, h, r: (b, r, h, 0), (None, block_rows, None, head_dim)), # dot_dO
        ],
        compiler_params=triton_compiler_params(
            num_stages=num_stages, num_warps=num_warps
        ),
        out_shape = [ dot_Q, dot_K, dot_V, dot_dO ],
        input_output_aliases = { 10: 1, 11: 2 },
        name="fused_attn_bwd_bwd"
    )(q, k, v, dO, dot_dQ, dot_dK, dot_dV, m, l, delta, dot_K, dot_V)

    return dot_Q, dot_K, dot_V, dot_dO

def fused_attn_bwd_bwd(saved, args, precision, block, num_stages, num_warps):
    dot_dQ, dot_dK, dot_dV = args
    Q, K, V, dO, m, l, delta, causal, attn_softcap, softmax_factor = saved
    dot_Q, dot_K, dot_V, dot_dO = launch_fused(q=Q, k=K, v=V, dO=dO, 
                                               dot_dQ=dot_dQ, dot_dK=dot_dK,
                                               dot_dV=dot_dV, 
                                               m=m, l=l, delta=delta, 
                                               causal=causal, 
                                               attn_softcap=attn_softcap, 
                                               softmax_factor=softmax_factor,
                                               precision=precision,
                                               block=block,
                                               num_stages=num_stages,
                                               num_warps=num_warps)
    return dot_Q, dot_K, dot_V, dot_dO
