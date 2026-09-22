# FlashBoB

FlashBoB adds second-order reverse-mode differentiation to scaled dot-product
attention (SDPA). Its public API is one PyTorch function that supports
multi-head attention (MHA), grouped-query attention (GQA), multi-query
attention (MQA), unequal query and key lengths, and causal sliding windows.

```python
from bob import sdpa_bob
```

## Install

```bash
pip install .
```

Install the `native` extra to use FlashAttention for eligible rectangular CUDA
calls and to run the FlashAttention comparison example:

```bash
pip install '.[native]'
```

## Use

Each input has shape `[batch, heads, sequence, head_dim]`. Set
`create_graph=True` when computing the first gradients so PyTorch retains the
graph needed for the second gradients.

```python
import torch
from bob import sdpa_bob

q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

out = sdpa_bob(q, k, v, is_causal=True)
dq, dk, dv = torch.autograd.grad(out.square().mean(), (q, k, v),
                                 create_graph=True)
ddq, ddk, ddv = torch.autograd.grad(
    dq.square().mean() + dk.square().mean() + dv.square().mean(),
    (q, k, v),
)
```

GQA and MQA use the same function. The query head count may be an integer
multiple of the key/value head count:

```python
q = torch.randn(1, 32, 4096, 64, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
k = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
v = torch.randn_like(k, requires_grad=True)
out = sdpa_bob(q, k, v, is_causal=True)
```

For a right-aligned query block, the position offset is inferred from the
sequence lengths:

```python
q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
k = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
v = torch.randn_like(k, requires_grad=True)
out = sdpa_bob(q, k, v, is_causal=True, window_size=256)
```

Pass `q_offset` and `k_offset` when the tensors use different absolute
positions.

## Examples

For SDPA calls that use only `q`, `k`, `v`, `is_causal`, and `scale`, the
replacement changes one call:

```diff
-  y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
+  y = sdpa_bob(q, k, v, is_causal=True)
```

On CUDA, the forward and first backward use fused PyTorch ATen operators for
dense and sliding-window attention. FlashBoB makes the first backward
differentiable and computes the second backward with Triton kernels.

```bash
python examples/swap_attention_backend.py   # replace SDPA in a module, with error bounds
python examples/flash_attn_drop_in.py       # flash_attn_func layout and window conventions
python examples/huggingface_backend.py      # register bob with compatible Transformers models
python examples/hessian_vector_product.py   # time and memory of Hv against the math backend
```

[examples/README.md](examples/README.md) lists every script, what it
demonstrates, and the conversion table between this API and
`flash_attn_func`.

## Support

- CPU: differentiable PyTorch reference.
- CUDA MHA: FP16 or BF16, head dimension 32, 64, or 128.
- CUDA GQA/MQA: BF16, equal query and key lengths, group size at most 32.
- Reverse-mode differentiation through second order.
- No dropout or user-provided attention bias.

See [docs/support.md](docs/support.md) for the complete input, route, and
software compatibility requirements.

## Verify

```bash
pip install -e '.[benchmarks,test]'
pytest -q tests/unit tests/benchmarks
```

On a supported CUDA system, run the bounded correctness and benchmark suite:

```bash
./benchmarks/reproduce.sh --profile minimal all
```

The wheel contains only the `bob` package from `src/bob`. Model experiments,
preserved comparison implementations, and full reproduction commands are in
[benchmarks/README.md](benchmarks/README.md).

```text
src/bob/       public operator and Triton kernels
examples/      runnable examples, including drop-in replacement patterns
tests/         public behavior and numerical checks
benchmarks/    experiment runners and preserved comparison implementations
```

## Citation

If you use FlashBoB in your work, please cite our paper:

```bibtex
@misc{givans2026flashbob,
      title={FlashBoB: I/O-Efficient Exact Backward-over-Backward for Softmax Attention}, 
      author={Anthony Givans and Michael Crawshaw and Mingrui Liu},
      year={2026},
      eprint={2609.24089},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.24089}, 
}
```
