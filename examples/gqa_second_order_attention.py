"""Second-order grouped-query attention through the same function.

GQA and MQA need no separate entry point. Pass K and V with fewer heads than Q
and FlashBoB indexes the key/value heads directly rather than expanding them,
so the memory cost stays proportional to the KV head count.

CUDA GQA requires bfloat16 and a grouping factor of at most 32.

Run: python examples/gqa_second_order_attention.py
"""

import torch

from bob import sdpa_bob


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    torch.manual_seed(0)
    # 32 query heads over 8 key/value heads is a grouping factor of 4
    q = torch.randn(1, 32, 4096, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)

    output = sdpa_bob(q, k, v, is_causal=True)
    first = torch.autograd.grad(output.float().square().mean(), (q, k, v), create_graph=True)
    second = torch.autograd.grad(sum(g.float().square().sum() for g in first), (q, k, v))

    print("output:      ", tuple(output.shape))
    print("first grads: ", [tuple(g.shape) for g in first])
    print("second grads:", [tuple(g.shape) for g in second])


if __name__ == "__main__":
    main()
