import pytest
import torch

from bob import sdpa_bob
from benchmarks.attention import attention_reference


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _case(query_heads, key_heads, query_length, key_length, head_dim, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q_shape = (1, query_heads, query_length, head_dim)
    kv_shape = (1, key_heads, key_length, head_dim)
    rand = lambda shape: torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    tensors = tuple(
        value.requires_grad_(True)
        for value in (rand(q_shape), rand(kv_shape), rand(kv_shape), rand(q_shape))
    )
    bars = (rand(q_shape), rand(kv_shape), rand(kv_shape))
    return tensors, bars


def _second_grad(operator, tensors, bars, kwargs):
    q, k, v, grad_out = tensors
    out = operator(q, k, v, **kwargs)
    first = torch.autograd.grad(out, (q, k, v), grad_out, create_graph=True)
    return torch.autograd.grad(first, (q, k, v, grad_out), bars)


@pytest.mark.parametrize(
    ("shape", "kwargs"),
    [
        ((2, 2, 64, 64, 64), {"is_causal": True}),
        ((2, 2, 48, 48, 128), {"is_causal": False}),
        ((2, 2, 64, 64, 64), {"is_causal": True, "window_size": 16}),
        ((8, 2, 64, 64, 64), {"is_causal": True}),
        ((8, 1, 64, 64, 64), {"is_causal": True}),
        (
            (2, 2, 32, 64, 64),
            {"is_causal": True, "window_size": 32, "q_offset": 32, "k_offset": 0},
        ),
    ],
)
def test_second_gradient_matches_materialized_reference(shape, kwargs):
    tensors, bars = _case(*shape)
    expected = _second_grad(attention_reference, tensors, bars, kwargs)
    actual = _second_grad(sdpa_bob, tensors, bars, kwargs)

    for reference, result in zip(expected, actual):
        torch.testing.assert_close(result.float(), reference.float(), rtol=0.08, atol=0.08)


def test_undefined_kv_tangents_do_not_change_the_query_tangent_result():
    tensors, bars = _case(2, 2, 64, 64, 64)
    q, k, v, grad_out = tensors
    out = sdpa_bob(q, k, v, is_causal=True)
    first = torch.autograd.grad(out, (q, k, v), grad_out, create_graph=True)
    actual = torch.autograd.grad((first[0] * bars[0]).sum(), (q, k, v, grad_out))

    reference_inputs = tuple(
        value.detach().clone().requires_grad_(True) for value in tensors
    )
    rq, rk, rv, reference_grad_out = reference_inputs
    ref_out = attention_reference(rq, rk, rv, is_causal=True)
    ref_first = torch.autograd.grad(
        ref_out, (rq, rk, rv), reference_grad_out, create_graph=True
    )
    expected = torch.autograd.grad(
        (ref_first[0] * bars[0]).sum(), reference_inputs
    )

    for reference, result in zip(expected, actual):
        torch.testing.assert_close(result.float(), reference.float(), rtol=0.08, atol=0.08)
