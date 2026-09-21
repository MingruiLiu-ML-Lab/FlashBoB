"""Minimal FlashBoB second-order attention example."""

import torch

from bob import sdpa_bob


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    shape = (2, 8, 1024, 64)
    tensors = [
        torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    q, k, v = tensors

    output = sdpa_bob(q, k, v, is_causal=True)
    first = torch.autograd.grad(
        output.float().square().mean(),
        tensors,
        create_graph=True,
    )
    second = torch.autograd.grad(
        sum(gradient.float().square().sum() for gradient in first),
        tensors,
    )

    print("output:", tuple(output.shape))
    print("first gradients:", [tuple(gradient.shape) for gradient in first])
    print("second gradients:", [tuple(gradient.shape) for gradient in second])


if __name__ == "__main__":
    main()
