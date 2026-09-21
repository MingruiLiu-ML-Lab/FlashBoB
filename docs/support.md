# Support contract

`bob.sdpa_bob` is the only public function:

```python
sdpa_bob(
    q,
    k,
    v,
    *,
    is_causal=False,
    scale=None,
    window_size=0,
    q_offset=None,
    k_offset=0,
)
```

The query has shape `[B, H_Q, N_Q, D]`; the key and value have shape
`[B, H_KV, N_KV, D]`. The key and value shapes must match. All three inputs
must have the same batch size, head dimension, dtype, and device. Rectangular
attention permits `N_Q != N_KV` and requires `H_Q == H_KV`. GQA and MQA permit
`H_Q > H_KV` and require `N_Q == N_KV`.

## CUDA routes

| Geometry | Dtype | Head dimension | Additional restrictions |
|---|---|---|---|
| Square MHA | FP16, BF16 | 32, 64, 128 | Causal, noncausal, or causal window |
| Rectangular MHA | FP16, BF16 | 32, 64, 128 | Equal head counts; eligible causal calls require `flash-attn` |
| Square GQA/MQA | BF16 | 32, 64, 128 | `H_Q / H_KV <= 32`; no finite window |

Dense CUDA attention uses fused private ATen operators for the forward and
first backward. Sliding-window attention uses ATen's efficient-attention
operators. Eligible causal rectangular calls use `flash-attn`; other
rectangular calls use the PyTorch math implementation. The second backward
uses the Triton kernels in this package.

### Dense front end

The dense forward and first backward run on one of two fused operators, chosen
by compute capability:

| Compute capability | Operator | Basis |
|---|---|---|
| 9.0 and above | `_scaled_dot_product_cudnn_attention` | Historical H100 and B200 measurements favored cuDNN; this checkout does not contain their artifacts |
| Below 9.0 | `_scaled_dot_product_flash_attention` | On SM89, cuDNN is 15% to 25% slower on the first backward and 5% to 11% slower on the whole second-order step |

On SM89, the two dense operators produced second-order outputs with relative
L2 differences near `1.3e-03` from 512 to 16384 tokens. Operator selection
changes the schedule and latency while preserving the public API. Sliding
windows use `_efficient_attention_forward`; rectangular attention follows the
routes described above.

The SM89 figures were measured with `python -m benchmarks.front_end`. That
comparison changes the fused operator while holding the Triton second
backward, input tensors, and timing order fixed. The H100 and B200 figures are
historical project measurements without matching artifacts in this checkout.
The compute-capability threshold therefore requires confirmation on SM90 and
SM100 before it is treated as performance-optimal on those architectures.

CPU MHA and rectangular calls use a PyTorch math fallback. CPU GQA/MQA is not
supported.

## Masks and offsets

`window_size=0` means full attention. A positive window includes the current
key and the preceding `window_size - 1` positions. Finite windows require
`is_causal=True`.

Offsets assign absolute positions:

```text
query position = local query index + q_offset
key position   = local key index + k_offset
```

When `q_offset` is omitted for rectangular causal attention, the operator uses
`k_offset + max(0, N_KV - N_Q)`, which right-aligns the query block. Explicit
offsets and windows must be nonnegative integers. `scale` must be finite and
positive when supplied.

## Differentiation

Reverse-mode differentiation is supported through second order. Call the first
`torch.autograd.grad` with `create_graph=True`. Dropout, a user-provided bias,
forward-mode AD, and `torch.func` transforms are outside the supported API.

Inputs may be noncontiguous. Outputs and gradients are returned in ordinary
contiguous PyTorch layout and retain the input dtype and device.

## Software compatibility

The supported configurations are PyTorch 2.8 with Triton 3.4 and PyTorch 2.9
with Triton 3.5. CUDA forward paths call private ATen operators, so
compatibility with other PyTorch releases is not assumed. Expanding this range
requires rerunning the CUDA correctness suite.
