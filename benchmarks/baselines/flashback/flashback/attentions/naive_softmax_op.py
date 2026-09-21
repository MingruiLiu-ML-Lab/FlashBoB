from functools import partial
import jax
import jax.numpy as jnp

from .naive_sigmoid_op import DEFAULT_MASK_VALUE, make_causal_mask, infer_precision
from ..pallas_utils import Precision

def naive_softmax_mha(Q, K, V, sm_scale=None, causal=True,
                      precision=Precision.TF32_ROUND):
    data_prec, dot_prec = infer_precision(precision)
    assert Q.dtype == K.dtype == V.dtype, "Q, K, V must have same dtype"
    assert Q.dtype == data_prec, f"Q, K, V must be of dtype {data_prec}"
    sequence_length = Q.shape[1]

    if sm_scale is None:
        sm_scale = float(Q.shape[-1])**(-0.5)

    einsum = partial(jnp.einsum, precision=dot_prec)

    assert len(K.shape) == 4
    S = einsum('brhd,bchd->bhrc', Q, K)

    if sm_scale is not None:
        S = S * sm_scale

    if causal:
        causal_mask = make_causal_mask(sequence_length)
        S = jnp.where(causal_mask, S, DEFAULT_MASK_VALUE)

    P = jax.nn.softmax(S, axis=-1)
    O = einsum('bhrc,bchd->brhd', P, V)
    return O

