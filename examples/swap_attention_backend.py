"""Swap PyTorch SDPA for FlashBoB in an existing attention module.

For calls that use `[B, H, N, D]` tensors plus `is_causal` and `scale`,
`sdpa_bob` can replace `torch.nn.functional.scaled_dot_product_attention` at
the call site. This example compares their first-order numerical error and
shows that FlashBoB also provides a second backward.

Both implementations are compared with the same FP32 math reference. The
reported values are relative errors against that reference.

Run: python examples/swap_attention_backend.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bob import sdpa_bob


def sdpa_reference(q, k, v, *, is_causal=False, scale=None):
    """PyTorch math SDPA used as the twice-differentiable reference."""
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)


class CausalSelfAttention(nn.Module):
    """Ordinary causal self-attention with a swappable attention callable.

    `attention` is called as `attention(q, k, v, is_causal=True)` on
    `[B, H, N, D]` tensors, the contract `F.scaled_dot_product_attention` and
    `sdpa_bob` both satisfy.
    """

    def __init__(self, dim: int, n_head: int, attention=F.scaled_dot_product_attention):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.attention = attention
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = (
            t.view(B, N, self.n_head, self.head_dim).transpose(1, 2)
            for t in self.qkv(x).split(C, dim=2)
        )
        y = self.attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, N, C))


def relative_error(values, exact) -> float:
    """Return the largest per-tensor relative L2 error."""
    return max(
        float((value.float() - target).norm() / target.norm().clamp_min(1e-12))
        for value, target in zip(values, exact)
    )


def first_and_second_grads(model: nn.Module, x: torch.Tensor):
    """`(dW, d/dW of ||dW||^2)` for every parameter, i.e. one Hessian-vector product."""
    params = list(model.parameters())
    loss = model(x).float().square().mean()
    first = torch.autograd.grad(loss, params, create_graph=True)
    second = torch.autograd.grad(sum(g.float().square().sum() for g in first), params)
    # first still carries the graph that produced second; callers only compare values
    return tuple(g.detach() for g in first), second


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    torch.manual_seed(0)
    B, N, C, H = 2, 512, 512, 8
    x = torch.randn(B, N, C, device="cuda", dtype=torch.bfloat16)

    # Compare three attention implementations with identical model weights.
    baseline = CausalSelfAttention(C, H).cuda().to(torch.bfloat16)
    swapped = CausalSelfAttention(C, H, attention=sdpa_bob).cuda().to(torch.bfloat16)
    swapped.load_state_dict(baseline.state_dict())
    reference = CausalSelfAttention(C, H, attention=sdpa_reference).cuda().float()
    reference.load_state_dict({n: p.float() for n, p in baseline.state_dict().items()})

    exact_first, exact_second = first_and_second_grads(reference, x.float())

    print("== forward and first backward ==")
    errors = {}
    for name, model in (("torch sdpa", baseline), ("sdpa_bob", swapped)):
        loss = model(x).float().square().mean()
        grads = torch.autograd.grad(loss, list(model.parameters()))
        errors[name] = relative_error(grads, exact_first)
        print(f"  {name:11s} relative L2 error vs fp32 = {errors[name]:.3e}")

    # Apply the same relative-error criterion used by this repository's examples.
    assert errors["sdpa_bob"] <= 2 * errors["torch sdpa"], errors
    print("  sdpa_bob is within 2x of the torch sdpa error")

    print("== second backward ==")
    for name, model in (("torch sdpa", baseline), ("sdpa_bob", swapped)):
        try:
            _, second = first_and_second_grads(model, x)
        except RuntimeError as error:
            print(f"  {name:11s} RuntimeError: {error}")
            continue
        print(
            f"  {name:11s} relative L2 error vs fp32 = "
            f"{relative_error(second, exact_second):.3e}"
        )


if __name__ == "__main__":
    main()
