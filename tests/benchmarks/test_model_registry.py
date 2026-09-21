import pytest
import torch

from benchmarks.models.gqa import Llama, LlamaConfig
from benchmarks.models.language_model_benchmark import (
    MODEL_PRESETS,
    SOPHIA_PRESETS,
    GPTNeoX,
    model_backends,
    model_from_preset,
)


@pytest.mark.parametrize(
    ("preset", "layers", "query_heads", "kv_heads", "head_dim", "parameters"),
    [
        ("llama-3.2-1b", 16, 32, 8, 64, 1_235_814_400),
        ("llama-3.2-3b", 28, 24, 8, 128, 3_212_749_824),
        ("smollm2-135m", 30, 9, 3, 64, 134_515_008),
        ("smollm2-360m", 32, 15, 5, 64, 361_821_120),
    ],
)
def test_local_gqa_presets_match_published_architectures(
    preset, layers, query_heads, kv_heads, head_dim, parameters
):
    config, model_factory = model_from_preset(preset)

    with torch.device("meta"):
        model = model_factory(config, "math")

    assert model_factory is Llama
    assert (
        config.n_layer,
        config.n_head,
        config.n_kv_head,
        config.head_dim,
    ) == (layers, query_heads, kv_heads, head_dim)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters


@pytest.mark.parametrize(
    ("preset", "layers", "hidden", "query_heads", "kv_heads", "parameters"),
    [
        ("qwen2.5-1.5b", 28, 1536, 12, 2, 1_543_714_304),
        ("granite-3.1-1b-a400m", 24, 1024, 16, 8, 1_334_625_280),
    ],
)
def test_transformers_presets_match_published_architectures(
    preset, layers, hidden, query_heads, kv_heads, parameters
):
    pytest.importorskip("transformers")
    config, model_factory = model_from_preset(preset)

    with torch.device("meta"):
        model = model_factory(config, "math")

    assert (
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
    ) == (layers, hidden, query_heads, kv_heads, hidden // query_heads)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters


@pytest.mark.parametrize(
    ("preset", "heads", "head_dim", "parameters"),
    [
        ("pythia-160m-d128", 6, 128, 162_322_944),
        ("pythia-410m-d128", 8, 128, 405_334_016),
    ],
)
def test_pythia_d128_controls_keep_the_expected_parameter_geometry(
    preset, heads, head_dim, parameters
):
    config, model_factory = model_from_preset(preset)
    with torch.device("meta"):
        model = model_factory(config, "math")

    assert model_factory is GPTNeoX
    assert config.n_head == heads
    assert config.head_dim == head_dim
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters


def test_tiny_gqa_model_runs_through_the_shared_math_backend():
    config = LlamaConfig(
        vocab_size=31,
        block_size=8,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        n_embd=16,
        intermediate_size=32,
        use_llama3_rope_scaling=False,
    )
    model = Llama(config, "math")
    tokens = torch.tensor([[1, 5, 3, 9], [2, 7, 4, 6]])

    logits = model(tokens)

    assert logits.shape == (2, 4, 31)
    assert torch.isfinite(logits).all()


def test_registry_exposes_every_published_model():
    assert {
        "llama-3.2-1b",
        "llama-3.2-3b",
        "smollm2-135m",
        "smollm2-360m",
        "qwen2.5-1.5b",
        "granite-3.1-1b-a400m",
        "pythia-160m-d128",
        "pythia-410m-d128",
    } <= set(MODEL_PRESETS)


def test_sophia_uses_the_shared_registry_for_mutable_local_models():
    assert {
        "llama-3.2-1b",
        "llama-3.2-3b",
        "smollm2-135m",
        "smollm2-360m",
        "pythia-160m-d128",
        "pythia-410m-d128",
    } <= set(SOPHIA_PRESETS)
    assert "qwen2.5-1.5b" not in SOPHIA_PRESETS
    assert "granite-3.1-1b-a400m" not in SOPHIA_PRESETS


def test_model_families_advertise_only_compatible_backends():
    assert model_backends("qwen2.5-1.5b") == ("bob", "math")
    assert "flash" in model_backends("llama-3.2-1b")
    assert "bob" in model_backends("pythia-160m-d128")
