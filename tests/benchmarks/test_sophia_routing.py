import unittest
from argparse import Namespace
from unittest.mock import patch

import torch

from benchmarks.models.gqa import Llama, LlamaAttention, LlamaConfig
from benchmarks.models.language_model_benchmark import (
    ATTN_BACKENDS,
    CausalSelfAttention,
    GPT,
    GPTConfig,
    GPTNeoX,
    GPTNeoXConfig,
    model_from_preset,
)
from benchmarks.models.sophia.data import encode_text_for_training
from benchmarks.models.sophia.runtime import ce_loss_from_logits
from benchmarks.models.sophia.train import (
    BOB_ATTENTION_BACKENDS,
    evaluate,
    hessian_attention_context,
    hutchinson_diag_estimate,
    recorded_args,
    resolve_attention_routing,
    use_attention_backend,
    validate_hessian_schedule,
)


class TokenizerFake:
    special_tokens_set = {"<eos>"}

    def encode(self, text, **kwargs):
        assert text == "sample"
        if "allowed_special" in kwargs:
            assert kwargs["allowed_special"] == self.special_tokens_set
        else:
            assert kwargs == {"add_special_tokens": False}
        return [3, 1, 4]


def test_training_text_encoding_supports_both_tokenizer_interfaces():
    for backend in ("tiktoken", "huggingface"):
        tokens = encode_text_for_training(TokenizerFake(), "sample", backend=backend)
        assert tokens.tolist() == [3, 1, 4]
        assert str(tokens.dtype) == "int32"


def test_recorded_args_excludes_hugging_face_credentials():
    args = Namespace(hf_token="secret", dataset_id="example/dataset")

    assert recorded_args(args) == {"dataset_id": "example/dataset"}


def _tiny_gpt(backend: str = "flash") -> GPT:
    cfg = GPTConfig(
        vocab_size=17,
        block_size=4,
        n_layer=1,
        n_head=2,
        n_embd=8,
    )
    return GPT(cfg, backend)


def test_pythia_presets_use_the_gpt_neox_model_family():
    config, model_factory = model_from_preset("pythia-160m")

    assert model_factory is GPTNeoX
    assert config.head_dim == 64
    assert config.rotary_dim == 16


def test_tiny_gpt_neox_preserves_token_geometry():
    config = GPTNeoXConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=32,
        intermediate_size=64,
    )
    model = GPTNeoX(config, "math")
    tokens = torch.randint(0, config.vocab_size, (2, 6))

    assert model(tokens).shape == (2, 6, config.vocab_size)


def test_sophia_backend_context_routes_and_restores_gqa_models():
    config = LlamaConfig(
        vocab_size=31,
        block_size=4,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        n_embd=16,
        intermediate_size=32,
        use_llama3_rope_scaling=False,
    )
    model = Llama(config, "flash")

    with use_attention_backend(model, "math"):
        assert [
            module.backend
            for module in model.modules()
            if isinstance(module, LlamaAttention)
        ] == ["math"]

    assert [
        module.backend
        for module in model.modules()
        if isinstance(module, LlamaAttention)
    ] == ["flash"]


def _attention_backends(model: torch.nn.Module) -> list[str]:
    return [
        module.backend
        for module in model.modules()
        if isinstance(module, CausalSelfAttention)
    ]


def _attention_spy(calls, name):
    def attention(q, k, v, **_kwargs):
        calls.append(name)
        return q + k + v

    return attention


class SophiaBackendRoutingTests(unittest.TestCase):
    def test_resolver_builds_hybrid_math_and_persistent_bob_routes(self):
        math_routing = resolve_attention_routing(
            optimizer="sophiah",
            attn_backend="math",
            non_bob_attn_backend="flash",
        )
        self.assertEqual(math_routing.ordinary_backend, "flash")
        self.assertEqual(math_routing.hessian_backend, "math")
        self.assertEqual(math_routing.first_order_impl, "flash")
        self.assertTrue(math_routing.requires_hessian_switch)

        for backend in sorted(BOB_ATTENTION_BACKENDS):
            with self.subTest(backend=backend):
                bob_routing = resolve_attention_routing(
                    optimizer="sophiah",
                    attn_backend=backend,
                    non_bob_attn_backend=None,
                )
                self.assertEqual(bob_routing.ordinary_backend, backend)
                self.assertEqual(bob_routing.hessian_backend, backend)
                self.assertEqual(
                    bob_routing.first_order_impl,
                    "native_flash_internal",
                )
                self.assertFalse(bob_routing.requires_hessian_switch)

    def test_resolver_rejects_bob_override_and_flash_hvp(self):
        for backend in sorted(BOB_ATTENTION_BACKENDS):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "cannot override a bob-family"):
                    resolve_attention_routing(
                        optimizer="sophiah",
                        attn_backend=backend,
                        non_bob_attn_backend="flash",
                    )

        with self.assertRaisesRegex(ValueError, "cannot use --attn-backend=flash"):
            resolve_attention_routing(
                optimizer="sophiah",
                attn_backend="flash",
                non_bob_attn_backend=None,
            )

        with self.assertRaisesRegex(ValueError, "only valid with a second-order optimizer"):
            resolve_attention_routing(
                optimizer="adamw",
                attn_backend="flash",
                non_bob_attn_backend="math",
            )

    def test_hessian_schedule_validation(self):
        with self.assertRaisesRegex(ValueError, "--hess-interval must be >= 1"):
            validate_hessian_schedule(
                optimizer="sophiah",
                hess_interval=0,
                hutch_samples=1,
            )
        with self.assertRaisesRegex(ValueError, "--hutch-samples must be >= 1"):
            validate_hessian_schedule(
                optimizer="sophiah",
                hess_interval=10,
                hutch_samples=0,
            )

    def test_context_restores_backend_after_exception_and_when_nested(self):
        model = _tiny_gpt()
        self.assertEqual(_attention_backends(model), ["flash"])

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with use_attention_backend(model, "bob"):
                self.assertEqual(_attention_backends(model), ["bob"])
                with use_attention_backend(model, "math"):
                    self.assertEqual(_attention_backends(model), ["math"])
                self.assertEqual(_attention_backends(model), ["bob"])
                raise RuntimeError("sentinel")

        self.assertEqual(_attention_backends(model), ["flash"])

    def test_hybrid_math_uses_flash_for_ordinary_train_and_eval(self):
        torch.manual_seed(17)
        routing = resolve_attention_routing(
            optimizer="sophiah",
            attn_backend="math",
            non_bob_attn_backend="flash",
        )
        model = _tiny_gpt(routing.ordinary_backend)
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        calls = []

        with patch.dict(
            ATTN_BACKENDS,
            {
                "flash": _attention_spy(calls, "flash"),
                "math": _attention_spy(calls, "math"),
            },
        ):
            with hessian_attention_context(model, routing):
                estimates = hutchinson_diag_estimate(
                    model,
                    params,
                    [(idx, targets)],
                    num_samples=1,
                )

            self.assertTrue(calls)
            self.assertEqual(set(calls), {"math"})
            self.assertTrue(all(torch.isfinite(estimate).all() for estimate in estimates))
            self.assertEqual(_attention_backends(model), ["flash"])

            calls.clear()
            logits = model(idx)
            ce_loss_from_logits(logits, targets).backward()
            self.assertTrue(calls)
            self.assertEqual(set(calls), {"flash"})

            calls.clear()
            evaluate(model, [(idx, targets)], torch.float32, "cpu")
            self.assertTrue(calls)
            self.assertEqual(set(calls), {"flash"})

    def test_bob_backend_stays_installed_for_hvp_train_and_eval(self):
        torch.manual_seed(17)
        routing = resolve_attention_routing(
            optimizer="sophiah",
            attn_backend="bob",
            non_bob_attn_backend=None,
        )
        model = _tiny_gpt(routing.ordinary_backend)
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        calls = []

        with patch.dict(ATTN_BACKENDS, {"bob": _attention_spy(calls, "bob")}):
            with hessian_attention_context(model, routing):
                estimates = hutchinson_diag_estimate(
                    model,
                    params,
                    [(idx, targets)],
                    num_samples=1,
                )

            self.assertTrue(calls)
            self.assertEqual(set(calls), {"bob"})
            self.assertTrue(all(torch.isfinite(estimate).all() for estimate in estimates))
            self.assertEqual(_attention_backends(model), ["bob"])

            calls.clear()
            logits = model(idx)
            ce_loss_from_logits(logits, targets).backward()
            self.assertTrue(calls)
            self.assertEqual(set(calls), {"bob"})
            self.assertEqual(_attention_backends(model), ["bob"])

            calls.clear()
            evaluate(model, [(idx, targets)], torch.float32, "cpu")
            self.assertTrue(calls)
            self.assertEqual(set(calls), {"bob"})
            self.assertEqual(_attention_backends(model), ["bob"])


if __name__ == "__main__":
    unittest.main()
