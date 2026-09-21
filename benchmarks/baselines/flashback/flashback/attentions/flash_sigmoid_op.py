import jax
from jax._src.lax.lax import canonicalize_precision, Precision as JPrecision
from functools import partial
import jax.numpy as jnp
from functools import cache
from jax import core
from ..pallas_utils import Precision
from ..autotune import AUTOTUNE, autotuned
from .flash_sigmoid_kernels import sigmoid_attn_fwd, sigmoid_attn_bwd, sigmoid_attn_bwdbwd
import numpy as np

BLOCKS = [16, 32, 64, 128]
STAGES = [1, 2, 3]
WARPS = [2, 4, 8, 16]

def filter_fn(cfg, args, kwargs):
    # ensure edge cases of strange args and kwargs combos
    if len(args) > 1:
        Q, K = args[:2]
    elif len(args) == 1:
        Q = args[0]
        K = kwargs['K']
    else:
        Q = kwargs['Q']
        K = kwargs['K']

    R = Q.shape[1]
    C = K.shape[1]
    if cfg['block'] > C or cfg['block'] > R:
        return False

    return True

at_sigmoid_attn_fwd = autotuned(sigmoid_attn_fwd, block=BLOCKS,
                                num_stages=STAGES,
                                num_warps=WARPS, _filter=filter_fn)

at_sigmoid_attn_bwd = autotuned(sigmoid_attn_bwd, block=BLOCKS,
                                num_stages=STAGES,
                                num_warps=WARPS, _filter=filter_fn)

at_sigmoid_attn_bwdbwd = autotuned(sigmoid_attn_bwdbwd,
                                   block=BLOCKS,
                                   num_stages=STAGES,
                                   num_warps=WARPS, _filter=filter_fn)

op_cache = {}
def make_operator(sm_scale, bias, causal, precision):
    # x = core.concrete_or_error(None, x, "The problem arose with the x argument.")
    sm_scale = core.concrete_or_error(None, sm_scale, "The sm_scale argument must be concrete.")
    bias = core.concrete_or_error(None, bias, "The bias argument must be concrete.")
    causal = core.concrete_or_error(None, causal, "The causal argument must be concrete.")
    precision = core.concrete_or_error(None, precision, "The precision argument must be concrete.")

    if hasattr(sm_scale, 'to_concrete_value'):
        sm_scale = sm_scale.to_concrete_value()

    if hasattr(bias, 'to_concrete_value'):
        bias = bias.to_concrete_value()

    if hasattr(causal, 'to_concrete_value'):
        causal = causal.to_concrete_value()

    if hasattr(precision, 'to_concrete_value'):
        precision = precision.to_concrete_value()

    sm_scale = float(sm_scale) if sm_scale is not None else None
    bias = float(bias) if bias is not None else None
    causal = bool(causal)

    cache_key = (sm_scale, bias, causal, precision)
    if cache_key in op_cache:
        return op_cache[cache_key]

    def _attention_forward(Q, K, V):
        O, saved = base_attention_forward(Q, K, V)
        return (O, saved), saved

    def _attention_backward(saved, dO):
        q, k, v = saved
        dO, dsave = dO

        acc_dQ, acc_dK, acc_dV = dsave[:3]
        acc_dQ = jax.lax.stop_gradient(acc_dQ)
        acc_dK = jax.lax.stop_gradient(acc_dK)
        acc_dV = jax.lax.stop_gradient(acc_dV)

        # accumulate into acc_dQ, acc_dK, acc_dV
        dQ, dK, dV = _attention_backward_unwrapped(q, k, v, dO, acc_dQ, acc_dK, acc_dV)
        return dQ, dK, dV

    def _full_attention_backward_unwrapped(q, k, v, dO, acc_dQ, acc_dK, acc_dV):
        dQ, dK, dV = at_sigmoid_attn_bwd(q, k, v, dO, causal, bias, sm_scale,
                                      block=AUTOTUNE, precision=precision,
                                      acc_dQ=acc_dQ, acc_dK=acc_dK, acc_dV=acc_dV,
                                      num_stages=AUTOTUNE, num_warps=AUTOTUNE)
        new_saved = (q, k, v, dO)
        return (dQ, dK, dV), new_saved

    @jax.custom_vjp
    def _attention_backward_unwrapped(q, k, v, dO, acc_dQ, acc_dK, acc_dV):
        return _full_attention_backward_unwrapped(q=q, k=k, v=v, dO=dO,
                                                  acc_dQ=acc_dQ, acc_dK=acc_dK,
                                                  acc_dV=acc_dV)[0]

    def _attention_backward_unwrapped_backward(saved, dots):
        dot_dQ, dot_dK, dot_dV = dots
        q, k, v, dO = saved
        ret = at_sigmoid_attn_bwdbwd(q, k, v, dO, dot_dQ, dot_dK, dot_dV,
                                  causal, bias, sm_scale, block=AUTOTUNE,
                                  precision=precision, num_stages=AUTOTUNE,
                                  num_warps=AUTOTUNE)
        edot_Q, edot_dO, edot_K, edot_V = ret

        return edot_Q, edot_K, edot_V, edot_dO, None, None, None

    _attention_backward_unwrapped.defvjp(_full_attention_backward_unwrapped,
                                         _attention_backward_unwrapped_backward)

    @jax.custom_vjp
    def base_attention_forward(Q, K, V):
        O = at_sigmoid_attn_fwd(Q, K, V, causal=causal, bias=bias,
                                softmax_factor=sm_scale, block=AUTOTUNE,
                                precision=precision, num_stages=AUTOTUNE,
                                num_warps=AUTOTUNE)
        saved = Q, K, V
        return O, saved

    base_attention_forward.defvjp(_attention_forward, _attention_backward)

    def attention_forward(Q, K, V):
        return base_attention_forward(Q, K, V)[0]

    # return attention_forward
    op_cache[cache_key] = attention_forward
    return attention_forward


def infer_precision(Q, K, V):
    assert Q.dtype == K.dtype == V.dtype, "Q, K, V must have same dtype"
    dtype = Q.dtype
    if dtype == np.float32 or dtype == jnp.float32:
        precision = Precision.FP32
    elif dtype == np.float16 or dtype == jnp.float16:
        precision = Precision.FP16
    elif dtype == jnp.bfloat16:
        precision = Precision.BF16
    else:
        raise ValueError(f"Invalid dtype {dtype}, {type(dtype)}")

    if precision == Precision.FP32:
        tf32_precisions = [JPrecision.DEFAULT, JPrecision.HIGH, None]
        if canonicalize_precision(None) in tf32_precisions:
            precision = Precision.TF32_ROUND

    return precision


def sigmoid_mha(Q, K, V, sm_scale, bias, causal, precision):
    attention_forward = make_operator(sm_scale, bias, causal, precision)
    return attention_forward(Q, K, V)
