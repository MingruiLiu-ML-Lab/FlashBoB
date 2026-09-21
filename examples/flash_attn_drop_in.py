"""Use FlashBoB where code already calls `flash_attn_func`.

FlashAttention takes `[B, N, H, D]`; FlashBoB takes `[B, H, N, D]`. The wrapper
below transposes the inputs and output and converts the supported keyword
arguments:

    flash_attn window_size=(left, right)  ->  bob window_size=left + 1, right=0
    flash_attn causal=                    ->  bob is_causal=
    flash_attn softmax_scale=             ->  bob scale=

For GQA, pass K and V with fewer heads than Q.

The transposes produce noncontiguous tensors, matching the layout passed by a
`flash_attn_func` call site.

Requires: pip install '.[native]'
Run: python examples/flash_attn_drop_in.py
"""

import torch
from flash_attn import flash_attn_func

from bob import sdpa_bob


def flash_bob_func(q, k, v, *, softmax_scale=None, causal=False, window_size=(-1, -1)):
    """`flash_attn_func`'s signature and `[B, N, H, D]` layout, backed by FlashBoB."""
    left, right = window_size
    if right not in (0, -1):
        raise NotImplementedError("FlashBoB windows look backward only, so right must be 0")
    out = sdpa_bob(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=causal,
        scale=softmax_scale,
        window_size=0 if left < 0 else left + 1,
    )
    return out.transpose(1, 2)


def forward_and_grads(attention, q, k, v, grad_out, **kwargs):
    out = attention(q, k, v, **kwargs)
    grads = torch.autograd.grad(out, (q, k, v), grad_out)
    return out.detach(), grads


def relative_error(values, expected) -> float:
    return max(
        float((value.float() - target.float()).norm() / target.float().norm().clamp_min(1e-12))
        for value, target in zip(values, expected)
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    torch.manual_seed(0)
    cases = (
        ("mha    ", 8, 8, dict(causal=True)),
        ("gqa    ", 8, 2, dict(causal=True)),
        ("mqa    ", 8, 1, dict(causal=True)),
        ("swa-256", 8, 8, dict(causal=True, window_size=(255, 0))),
    )
    B, N, D = 2, 1024, 64

    print(f"B={B} N={N} D={D} bf16, relative L2 error of sdpa_bob vs flash_attn_func")
    for name, heads, kv_heads, kwargs in cases:
        q = torch.randn(B, N, heads, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, N, kv_heads, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        grad_out = torch.randn(B, N, heads, D, device="cuda", dtype=torch.bfloat16)

        expected_out, expected_grads = forward_and_grads(
            flash_attn_func, q, k, v, grad_out, **kwargs
        )
        actual_out, actual_grads = forward_and_grads(
            flash_bob_func, q, k, v, grad_out, **kwargs
        )
        print(
            f"  {name}  out {relative_error([actual_out], [expected_out]):.2e}"
            f"   dq/dk/dv {relative_error(actual_grads, expected_grads):.2e}"
        )

    print()
    print("second backward comparison:")
    q = torch.randn(B, N, 8, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = torch.randn_like(q, requires_grad=True), torch.randn_like(q, requires_grad=True)
    for name, attention in (("flash_attn_func", flash_attn_func), ("flash_bob_func ", flash_bob_func)):
        first = torch.autograd.grad(
            attention(q, k, v, causal=True).float().square().mean(),
            (q, k, v),
            create_graph=True,
        )
        try:
            second = torch.autograd.grad(sum(g.float().square().sum() for g in first), (q, k, v))
        except RuntimeError as error:
            # flash-attn's backward is not itself differentiable, so the
            # first-order gradients come back detached from the graph
            print(f"  {name}  no second backward: {error}")
            continue
        print(f"  {name}  ok, shapes {[tuple(g.shape) for g in second]}")


if __name__ == "__main__":
    main()
