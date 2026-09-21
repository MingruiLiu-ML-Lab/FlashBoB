import jax
import jax.numpy as jnp
import numpy as np
from flashback.pallas_utils import Precision
from flashback.ops import sigmoid_mha
from functools import partial

def main():
    T = 256
    R, C, B, H, D = T, T, 4, 4, 32
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    Q = jax.random.normal(k1, (B, R, H, D), dtype=jnp.float32) * 0.01
    K = jax.random.normal(k2, (B, C, H, D), dtype=jnp.float32) * 0.01
    V = jax.random.normal(k3, (B, C, H, D), dtype=jnp.float32) * 0.01

    sm_scale = 1.0
    bias = float(np.log(R) * -5)

    fwd = jax.jit(partial(sigmoid_mha, sm_scale=sm_scale, bias=bias))
    o = fwd(Q, K, V)

    bck = jax.jit(jax.grad(lambda Q, K, V: fwd(Q, K, V).sum(), argnums=[0, 1, 2]))
    dQ, dK, dV = bck(Q, K, V)

    reduced_back = lambda Q, K, V: sum(d.sum() for d in bck(Q, K, V))
    bckbck = jax.jit(jax.grad(reduced_back, argnums=[0, 1, 2]))
    ddQ, ddK, ddV = bckbck(Q, K, V)

if __name__ == '__main__':
    main()
