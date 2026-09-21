import pytest
import torch

from bob import sdpa_bob


def _inputs():
    return tuple(torch.randn(1, 2, 4, 8) for _ in range(3))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scale": 0.0}, "scale"),
        ({"scale": float("nan")}, "scale"),
        ({"window_size": -1}, "window"),
        ({"window_size": 1.5}, "window"),
        ({"window_size": True}, "window"),
        ({"is_causal": False, "window_size": 2}, "causal"),
        ({"q_offset": -1}, "offset"),
        ({"q_offset": 1.5}, "offset"),
        ({"k_offset": True}, "offset"),
    ],
)
def test_invalid_scalar_arguments_fail_at_the_public_boundary(kwargs, message):
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=message):
        sdpa_bob(*_inputs(), **kwargs)


def test_mismatched_key_and_value_shapes_are_rejected():
    q, k, _ = _inputs()
    v = torch.randn(1, 2, 5, 8)
    with pytest.raises(ValueError, match="shape"):
        sdpa_bob(q, k, v)


def test_mixed_input_dtypes_are_rejected():
    q, k, v = _inputs()
    with pytest.raises(ValueError, match="dtype"):
        sdpa_bob(q, k.double(), v)


def test_query_heads_must_be_divisible_by_key_heads():
    q = torch.randn(1, 3, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="divisible"):
        sdpa_bob(q, k, v)


def test_cpu_gqa_reports_the_unsupported_route():
    q = torch.randn(1, 4, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn_like(k)
    with pytest.raises(NotImplementedError, match="CUDA"):
        sdpa_bob(q, k, v, is_causal=True)
