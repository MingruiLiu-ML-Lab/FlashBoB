import subprocess
import sys
from pathlib import Path

import torch

import bob
from benchmarks.attention import attention_reference


def _second_grad(operator, q, k, v, grad_out, bars, **kwargs):
    out = operator(q, k, v, **kwargs)
    first = torch.autograd.grad(out, (q, k, v), grad_out, create_graph=True)
    return torch.autograd.grad(first, (q, k, v, grad_out), bars)


def test_public_surface_contains_exactly_one_operator():
    assert bob.__all__ == ["sdpa_bob"]
    assert callable(bob.sdpa_bob)


def test_importing_bob_does_not_import_flash_attn():
    # a fresh interpreter is the only honest place to ask this: sys.modules is
    # shared with every earlier test in the session, and an eligible rectangular
    # call imports flash-attn at runtime by design
    probe = subprocess.run(
        [sys.executable, "-c", "import sys; import bob; print('flash_attn' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(bob.__file__).parents[1],
    )
    assert probe.stdout.strip() == "False"


def test_cpu_square_second_gradient_matches_materialized_reference():
    generator = torch.Generator().manual_seed(0)
    shape = (1, 2, 5, 4)
    tensors = tuple(
        torch.randn(shape, dtype=torch.float32, generator=generator).requires_grad_()
        for _ in range(4)
    )
    bars = tuple(
        torch.randn(shape, dtype=torch.float32, generator=generator)
        for _ in range(3)
    )
    kwargs = {"is_causal": True, "scale": 0.7}

    expected = _second_grad(attention_reference, *tensors, bars, **kwargs)
    actual = _second_grad(bob.sdpa_bob, *tensors, bars, **kwargs)

    for reference, result in zip(expected, actual):
        torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-5)


def test_cpu_rectangular_second_gradient_matches_materialized_reference():
    generator = torch.Generator().manual_seed(1)
    q_shape = (1, 2, 3, 4)
    kv_shape = (1, 2, 6, 4)
    q = torch.randn(q_shape, dtype=torch.float32, generator=generator).requires_grad_()
    k = torch.randn(kv_shape, dtype=torch.float32, generator=generator).requires_grad_()
    v = torch.randn(kv_shape, dtype=torch.float32, generator=generator).requires_grad_()
    grad_out = torch.randn(q_shape, dtype=torch.float32, generator=generator).requires_grad_()
    bars = (
        torch.randn(q_shape, dtype=torch.float32, generator=generator),
        torch.randn(kv_shape, dtype=torch.float32, generator=generator),
        torch.randn(kv_shape, dtype=torch.float32, generator=generator),
    )
    kwargs = {"is_causal": True, "window_size": 4, "q_offset": 3, "k_offset": 0}

    expected = _second_grad(attention_reference, q, k, v, grad_out, bars, **kwargs)
    actual = _second_grad(bob.sdpa_bob, q, k, v, grad_out, bars, **kwargs)

    for reference, result in zip(expected, actual):
        torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-5)
