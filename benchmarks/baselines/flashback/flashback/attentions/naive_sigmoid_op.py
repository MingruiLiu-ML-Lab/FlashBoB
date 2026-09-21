from ..pallas_utils import Precision
from jax import numpy as jnp
import jax
from functools import partial
import numpy as np

DEFAULT_MASK_VALUE = -0.95 * float(np.finfo(np.dtype("float16")).max)

def make_causal_mask(l):
    span_q = jnp.arange(l)
    span_k = jnp.arange(l)
    causal_mask = span_q[None, None, :, None] >= span_k[None, None, None, :]
    return causal_mask

def infer_precision(precision):
    # jax.lax.Precision.HIGH: tf32
    # jax.lax.Precision.HIGHEST: fp32
    dot_prec = jax.lax.Precision.HIGH

    if precision == Precision.FP16:
        input_dtype = jnp.float16
    elif precision == Precision.BF16:
        input_dtype = jnp.bfloat16
    elif precision == Precision.TF32_ROUND:
        input_dtype = jnp.float32
    elif precision == Precision.TF32_TRUNC:
        raise ValueError("TF32_TRUNC not supported")
    else:
        assert precision == Precision.FP32
        input_dtype = jnp.float32
        dot_prec = jax.lax.Precision.HIGHEST

    return input_dtype, dot_prec

def naive_sigmoid_mha(Q, K, V, sm_scale=None, causal=None, bias=None, precision='tf32-round'):
    B, R, H, D = Q.shape
    C = K.shape[1]
    assert C == R, f"need R and C to be the same, got {R=} and {C=}"

    if sm_scale is None:
        sm_scale = float(Q.shape[-1])**(-0.5)

    if bias is None:
        bias = float(np.log(R) * -5)

    data_prec, dot_prec = infer_precision(precision)
    assert Q.dtype == K.dtype == V.dtype, "Q, K, V must have same dtype"
    assert Q.dtype == data_prec, f"Q, K, V must be of dtype {data_prec}"
    sequence_length = Q.shape[1]

    einsum = partial(jnp.einsum, precision=dot_prec)

    assert len(K.shape) == 4
    S = einsum('brhd,bchd->bhrc', Q, K)

    if sm_scale is not None:
        S = S * sm_scale

    if bias is not None:
        S = S + bias

    if causal:
        causal_mask = make_causal_mask(sequence_length)
        S = jnp.where(causal_mask, S, DEFAULT_MASK_VALUE)

    P = jax.nn.sigmoid(S)
    O = einsum('bhrc,bchd->brhd', P, V)
    return O
