import jax
from functools import partial
import jax.numpy as jnp
from functools import cache
from ..autotune import AUTOTUNE, autotuned
from .flash_softmax_kernels import _mha_backward, _mha_forward
from .flash_softmax_bob_kernel import fused_attn_bwd_bwd
from ..pallas_utils import Precision
from .flash_sigmoid_op import BLOCKS, STAGES, WARPS

def clean_backward(Q, K, V, sm_scale, saved, dO, causal, attn_softcap, precision,
                  block, num_stages, num_warps):
    dQ, dK, dV, delta = _mha_backward(sm_scale=sm_scale, causal=causal,
                                      block_q=block, block_k=block,
                                      backward_pass_impl="triton", num_warps=num_warps,
                                      num_stages=num_stages, grid=None, interpret=False,
                                      res=saved, do=dO, debug=False,
                                      attn_softcap=attn_softcap,
                                      precision=precision)
    return dQ, dK, dV, delta

def clean_forward(Q, K, V, sm_scale, causal, attn_softcap, precision,
                  block, num_stages, num_warps):
    return _mha_forward(q=Q, k=K, v=V, segment_ids=None,
                                sm_scale=sm_scale,
                                causal=causal, attn_softcap=attn_softcap,
                                block_q=block, block_k=block,
                                backward_pass_impl="triton",
                                num_warps=num_warps, num_stages=num_stages, grid=None,
                                interpret=False, debug=False, precision=precision)

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

    if C % cfg['block'] != 0 or R % cfg['block'] != 0:
        return False

    return True

def filter_fn_bwdbwd(cfg, args, kwargs):
    # ensure edge cases of strange args and kwargs combos
    if len(args) >= 1:
        saved = args[0]
    else:
        saved = kwargs['saved']

    return filter_fn(cfg, saved, {})


at_attn_fwd = autotuned(clean_forward, block=BLOCKS, num_stages=STAGES,
                        num_warps=WARPS, _filter=filter_fn)

# at_attn_bwd = autotuned(sigmoid_attn_bwd, block=BLOCKS, num_stages=STAGES,
at_attn_bwd = autotuned(clean_backward, block=BLOCKS, num_stages=STAGES,
                        num_warps=WARPS, _filter=filter_fn)

at_attn_bwdbwd = autotuned(fused_attn_bwd_bwd, block=BLOCKS, num_stages=STAGES,
                        num_warps=WARPS, _filter=filter_fn_bwdbwd)

@cache
def make_operator(sm_scale, attn_softcap, causal, precision):
    def _attention_forward(Q, K, V):
        O, saved = base_attention_forward(Q, K, V)
        return (O, saved), saved

    def _attention_backward(saved, dO):
        q, k, v, segment_ids, out, l, m = saved
        dO, dsave = dO

        out = jax.lax.stop_gradient(out)
        l = jax.lax.stop_gradient(l)
        m = jax.lax.stop_gradient(m)

        dQ, dK, dV = _attention_backward_unwrapped(q, k, v, dO, out, l, m)
        dQ_save, dK_save, dV_save = dsave[:3]
        dQ = dQ + dQ_save
        dK = dK + dK_save
        dV = dV + dV_save
        return dQ, dK, dV

    def _full_attention_backward_unwrapped(q, k, v, dO, out, l, m):
        # segment ids is None
        saved = q, k, v, None, out, l, m
        dQ, dK, dV, delta = at_attn_bwd(q, k, v, sm_scale, saved, dO, causal,
                                           attn_softcap, precision)
        new_saved = (q, k, v, dO, m, l, delta)
        return (dQ, dK, dV), new_saved

    @jax.custom_vjp
    def _attention_backward_unwrapped(q, k, v, dO, out, l, m):
        return _full_attention_backward_unwrapped(q=q, k=k, v=v, dO=dO,
                                                  out=out, l=l, m=m)[0]

    def _attention_backward_unwrapped_backward(saved, dots):
        dot_dQ, dot_dK, dot_dV = dots
        (q, k, v, dO, m, l, delta) = saved
        esaved = q, k, v, dO, m, l, delta, causal, attn_softcap, sm_scale
        # args = esaved, (dot_dQ, dot_dK, dot_dV), precision
        # edot_Q, edot_K, edot_V, edot_dO = fused_attn_bwd_bwd(*args)
        args = esaved, (dot_dQ, dot_dK, dot_dV), precision
        edot_Q, edot_K, edot_V, edot_dO = at_attn_bwdbwd(*args)
        return edot_Q, edot_K, edot_V, edot_dO, None, None, None

    _attention_backward_unwrapped.defvjp(_full_attention_backward_unwrapped, _attention_backward_unwrapped_backward)

    @jax.custom_vjp
    def base_attention_forward(Q, K, V):
        O, saved = at_attn_fwd(Q, K, V, sm_scale, causal, attn_softcap, precision)
        return O, saved

    base_attention_forward.defvjp(_attention_forward, _attention_backward)

    def attention_forward(Q, K, V):
        # get shapes
        B, R, H, D = Q.shape
        C = K.shape[1]
        D_v = V.shape[-1]

        # assert D in [32, 64]
        assert C == R, f"need R and C to be the same, got {R=} and {C=}"
        assert D_v == D, f"need D and D_v to be the same, got {D=} and {D_v=}"

        assert H < R, f'probably need H < R, got {H=} and {R=} (heads should be less than sequence length..?)'

        if C % 32 != 0:
            print("🚨🚨 WARNING 🚨🚨: C is not divisible by 32, got", C)

        return base_attention_forward(Q, K, V)[0]

    return attention_forward


def softmax_mha(Q, K, V, sm_scale=None, causal=True, precision=Precision.TF32_ROUND):
    attn_softcap = None
    operator = make_operator(sm_scale=sm_scale, attn_softcap=attn_softcap, causal=causal,
                             precision=precision)
    return operator(Q, K, V)
