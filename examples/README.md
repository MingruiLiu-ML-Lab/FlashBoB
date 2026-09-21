# Examples

Each script runs against an installed package and prints its results. All
scripts require a CUDA device with bfloat16 support and use head dimension 64,
which is within the supported set of 32, 64, and 128.

```bash
pip install -e '.[native,benchmarks]'
python examples/<script>.py
```

| Script | Shows | Extra requirement | Runtime |
|---|---|---|---|
| [second_order_attention.py](second_order_attention.py) | The smallest complete call: forward, first gradients, second gradients | none | 13 s |
| [gqa_second_order_attention.py](gqa_second_order_attention.py) | Grouped-query attention through the same function | none | 15 s |
| [swap_attention_backend.py](swap_attention_backend.py) | Replacing `F.scaled_dot_product_attention` in an existing module, and what it costs numerically | none | 13 s |
| [flash_attn_drop_in.py](flash_attn_drop_in.py) | A `flash_attn_func`-shaped wrapper compared with FlashAttention | `.[native]` | 13 s |
| [huggingface_backend.py](huggingface_backend.py) | Registering the operator as `attn_implementation="bob"` for compatible Transformers models | `.[benchmarks]` | 15 s |
| [hessian_vector_product.py](hessian_vector_product.py) | Time and peak memory of `Hv` compared with PyTorch's math SDPA backend | none | 52 s |

Runtimes are wall clock on an RTX 4090 Laptop including Triton autotuning on
first use, which dominates every script except the last.

## Swapping an existing attention implementation

For calls that use only `[B, H, N, D]` tensors and the `is_causal` and `scale`
keywords, replacing `torch.nn.functional.scaled_dot_product_attention` changes
one call:

```diff
-  y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
+  y = sdpa_bob(q, k, v, is_causal=True)
```

The forward and first backward use fused PyTorch ATen operators on the dense
CUDA route. FlashBoB makes the first backward differentiable and supplies the
second backward.

For arguments supported by FlashBoB, the main differences from
`flash_attn_func` are tensor layout and sliding-window notation:

| `flash_attn_func` | `sdpa_bob` |
|---|---|
| `[B, N, H, D]` | `[B, H, N, D]` |
| `causal=` | `is_causal=` |
| `softmax_scale=` | `scale=` |
| `window_size=(left, right)` | `window_size=left + 1`, backward-looking only |

Grouped-query and multi-query attention need no separate function in either
library: pass K and V with fewer heads than Q, with the query head count
divisible by the key/value head count.

## What the numbers mean

The comparison scripts report **relative L2 error against an FP32 reference**
rather than a single `allclose` tolerance, because a global absolute tolerance
on bfloat16 gradients depends strongly on the gradient magnitudes. Where an
example asserts an accuracy bound, FlashBoB's error must be at most twice the
PyTorch baseline's error on the same inputs.

Timings use CUDA events after warmup and report the median. These
single-configuration measurements illustrate the examples. Paper-scale
reproduction uses the harness and methodology in
[../benchmarks/README.md](../benchmarks/README.md).
