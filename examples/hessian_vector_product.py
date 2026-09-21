"""Compare Hessian-vector products over model parameters.

A Hessian-vector product is $Hv = \\nabla_\\theta (\\nabla_\\theta L \\cdot v)$. The
inner term requires the first gradient to remain in the autograd graph. Every
operation on the evaluated path, including attention, must therefore support
second-order reverse-mode differentiation.

In the supported PyTorch versions, fused SDPA kernels do not provide a second
backward. The PyTorch math backend does, but it materializes the
`[B, H, N, N]` attention matrix. This example compares that backend with
FlashBoB on latency and peak allocated memory.

Run: python examples/hessian_vector_product.py
"""

import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bob import sdpa_bob


def sdpa_math(q, k, v, *, is_causal=False, scale=None):
    """PyTorch math SDPA, which supports a second backward with $O(N^2)$ memory."""
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, attention):
        super().__init__()
        self.n_head, self.head_dim, self.attention = n_head, dim // n_head, attention
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        B, N, C = x.shape
        h = self.norm(x)
        q, k, v = (
            t.view(B, N, self.n_head, self.head_dim).transpose(1, 2)
            for t in self.qkv(h).split(C, dim=2)
        )
        y = self.attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, N, C))
        return x + self.mlp(x)


def hessian_vector_product(model, x, vectors):
    """`Hv` at the current parameters, with `v` supplied per parameter tensor."""
    params = list(model.parameters())
    loss = model(x).float().square().mean()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return torch.autograd.grad(
        sum((g.float() * v).sum() for g, v in zip(grads, vectors)), params
    )


def measure(attention, seq_len, dim, n_head, layers, warmup, iters):
    torch.manual_seed(0)
    model = nn.Sequential(*[Block(dim, n_head, attention) for _ in range(layers)])
    model = model.cuda().to(torch.bfloat16)
    x = torch.randn(1, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    vectors = [torch.randn_like(p, dtype=torch.float32) for p in model.parameters()]

    step = lambda: hessian_vector_product(model, x, vectors)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    # `step` closes over the model, so deleting the locals here would not free
    # anything; the caller empties the cache once this frame has gone away
    return statistics.median(samples), torch.cuda.max_memory_allocated() / 2**20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    props = torch.cuda.get_device_properties(0)
    print(f"{props.name} | torch {torch.__version__} | bf16 causal | "
          f"B=1 dim={args.dim} heads={args.heads} layers={args.layers} | "
          f"median of {args.iters} after {args.warmup} warmup")
    print(f"{'N':>7}{'math ms':>10}{'math MiB':>11}{'bob ms':>10}{'bob MiB':>10}"
          f"{'speedup':>9}{'memory':>9}")
    for seq_len in args.seq_lens:
        row = {}
        for name, attention in (("math", sdpa_math), ("bob", sdpa_bob)):
            try:
                row[name] = measure(
                    attention, seq_len, args.dim, args.heads, args.layers,
                    args.warmup, args.iters,
                )
            except torch.OutOfMemoryError:
                row[name] = None
            torch.cuda.empty_cache()
        if row["bob"] is None:
            print(f"{seq_len:>7}  FlashBoB ran out of memory")
            continue
        bob_ms, bob_mib = row["bob"]
        if row["math"] is None:
            print(f"{seq_len:>7}{'OOM':>10}{'OOM':>11}{bob_ms:>10.2f}{bob_mib:>10.0f}"
                  f"{'n/a':>9}{'n/a':>9}")
            continue
        math_ms, math_mib = row["math"]
        print(f"{seq_len:>7}{math_ms:>10.2f}{math_mib:>11.0f}{bob_ms:>10.2f}{bob_mib:>10.0f}"
              f"{math_ms / bob_ms:>8.2f}x{math_mib / bob_mib:>8.2f}x")


if __name__ == "__main__":
    main()
