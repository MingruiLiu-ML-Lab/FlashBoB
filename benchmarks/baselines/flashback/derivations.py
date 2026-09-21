'''
This file includes both (a) the backwards pass and (b) the backwards-over-backwards
pass of attention written as "math" (in Jax). We include tests at the bottom: run
this file via

    python derivations.py

to see them! 
'''


import jax
import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
from functools import partial

import numpy as np
DEFAULT_MASK_VALUE = -0.95 * float(np.finfo(np.dtype("float16")).max)

def ref_make_S(Q, K, causal):
    S = jnp.matmul(Q, K.transpose((0,1,3,2)))
    if causal:
        span_q = jnp.arange(Q.shape[-2])
        span_k = jnp.arange(K.shape[-2])
        causal_mask = span_q[None, None, :, None] >= span_k[None, None, None, :]
        S = jnp.where(causal_mask, S, DEFAULT_MASK_VALUE)

    return S

# ! SOME JAX BOILERPLATE
def attn_bwd_fwd(Q, K, V, dO, causal):
    res, bck_saved = attention_backward(Q, K, V, dO, causal)
    return res, bck_saved

def attn_bwd(Q, K, V, dO, causal):
    out, _ = attn_bwd_fwd(Q, K, V, dO, causal)
    return out

@jax.custom_vjp
def attn_bwd_custom(Q, K, V, dO, causal):
    out, _ = attn_bwd_fwd(Q, K, V, dO, causal)
    return out

# ! ATTENTION BACKWARDS PASS
def attention_backward(Q, K, V, dO, causal):
    S = ref_make_S(Q, K, causal)
    P = jax.nn.softmax(S, axis=-1)

    dP = jnp.matmul(dO, V.transpose((0,1,3,2)))

    PdP = P * dP
    colsums = jnp.einsum('bhqk->bhq', PdP)
    dS = P * (dP - colsums[:, :, :, None])

    dQ = jnp.matmul(dS, K)
    dK = jnp.matmul(dS.transpose((0,1,3,2)), Q)
    dV = jnp.matmul(P.transpose((0,1,3,2)), dO)

    return (dQ, dK, dV), (Q, K, V, dS, dO, P, dP, causal)

# ! ATTENTION BACKWARDS OVER BACKWARDS
def attn_bwd_bwd(saved, args):
    dot_dQ, dot_dK, dot_dV = args
    Q, K, V, dS, dO, P, dP, causal = saved

    # FIRST ROUND: dot_dS
    # depends on dK, dQ
    dot_dS = jnp.matmul(Q, dot_dK.transpose(0,1,3,2)) + jnp.matmul(dot_dQ, K.transpose(0,1,3,2))

    # SECOND ROUND: dot_dP, dot_P
    dot_dP = P * dot_dS - P * (P * dot_dS).sum(axis=-1)[..., None]

    # dot_P: depends on both dot_dV and dot_dS
    dot_P_from_dV = jnp.matmul(dO, dot_dV.transpose(0,1,3,2))
    # dot_P_from_dS = (dot_dS * dP - dot_dS * (P * dP).sum(axis=-1)[..., None] - (P * dP * dot_dS).sum(axis=-1)[..., None] )
    # ?... - t2 - t1
    dot_P_from_dS = dot_dS * dP - dot_dS * (P * dP).sum(axis=-1)[..., None] - dP * (P * dot_dS).sum(axis=-1)[..., None]
    dot_P = dot_P_from_dV + dot_P_from_dS 

    # THIRD ROUND: dot_dO, dot_Q, dot_K, dot_V
    # >> First lets do dot_K, dot_Q
    #    dot_K: from both dot dQ and dot_P
    #    dot_Q: from both dot dK and dot_P
    #    first we do the non flash attention terms
    dot_K_from_dQ = jnp.matmul(dS.transpose(0,1,3,2), dot_dQ)
    dot_Q_from_dK = jnp.matmul(dS, dot_dK)

    #    now we do the flash attention terms
    fadot_dS = P * dot_P - P * (P * dot_P).sum(axis=-1)[..., None]
    dot_K_from_P = jnp.matmul(fadot_dS.transpose(0,1,3,2), Q)
    dot_Q_from_P = jnp.matmul(fadot_dS, K)
    dot_K = dot_K_from_dQ + dot_K_from_P
    dot_Q = dot_Q_from_dK + dot_Q_from_P

    # >> Now lets do dot_V and dot_dO
    dot_V = jnp.matmul(dot_dP.transpose(0,1,3,2), dO)
    dot_dO_from_dP = jnp.matmul(dot_dP, V)
    dot_dO_from_dV = jnp.matmul(P, dot_dV)
    dot_dO = dot_dO_from_dP + dot_dO_from_dV

    return dot_Q, dot_K, dot_V, dot_dO, None

# ! JAX CUSTOM VJPS
attn_bwd_custom.defvjp(attn_bwd_fwd, attn_bwd_bwd)

B = 1
SEQ_LEN = 32
N_HEADS = 1
HEAD_DIM = 8

dtype = jnp.float64

q = jax.random.uniform(jax.random.PRNGKey(0), (B, N_HEADS, SEQ_LEN//2, HEAD_DIM), dtype=dtype)
k = jax.random.uniform(jax.random.PRNGKey(1), (B, N_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype)
v = jax.random.uniform(jax.random.PRNGKey(2), (B, N_HEADS, SEQ_LEN, HEAD_DIM//2), dtype=dtype)

def vec_params(x):
    x = jax.tree_util.tree_flatten(x)[0]
    return jnp.concatenate([p.flatten() for p in x])

sample_dO = jax.random.normal(jax.random.PRNGKey(10), (B, N_HEADS, SEQ_LEN//2, HEAD_DIM//2), dtype=dtype)

def random_project_output(*x, fwd=None):
    out = fwd(*x)
    z = 0
    for o in out:
        z += jnp.sum(o * jax.random.uniform(jax.random.PRNGKey(6), o.shape) + 0.5)

    return out

CAUSAL = True

jacobian_ad = jax.jacrev(partial(random_project_output, fwd=partial(attn_bwd, causal=CAUSAL)), argnums=[0, 1, 2, 3])(q, k, v, sample_dO)
jacobian_custom = jax.jacrev(partial(random_project_output, fwd=partial(attn_bwd_custom, causal=CAUSAL)), argnums=[0, 1, 2, 3])(q, k, v, sample_dO)

input_names = ['q', 'k', 'v', 'dO']
output_names = ['dq', 'dk', 'dv']

#! TESTING

passed_count = 0
total = len(jacobian_ad) * len(jacobian_ad[0]) * len(jacobian_ad[0][0])
for output_ad, output_custom, output_name in zip(jacobian_ad, jacobian_custom, output_names):
    for input_ad, input_custom, input_name in zip(output_ad, output_custom, input_names):
        msg = f'd{output_name}/d{input_name}'
        diff = jnp.abs(input_ad - input_custom)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        abs_gt = jnp.abs(input_ad).max().item()
        abs_custom = jnp.abs(input_custom).max().item()
        passed = max_diff < 1e-10
        check_or_fail_emolji = '✅' if passed else '❌'
        passed_count += int(passed)
        print(f"{check_or_fail_emolji} | {msg}: {max_diff = } {mean_diff = } {abs_gt = } {abs_custom = }")

print('PASSED:', passed_count, '/', total)
