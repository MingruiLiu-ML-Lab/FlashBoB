import unittest

import torch

from benchmarks.models.language_model_benchmark import CausalSelfAttention, GPT, GPTConfig
from benchmarks.models.sophia.runtime import ce_loss_from_logits
from benchmarks.models.sophia.train import (
    hessian_attention_context,
    hutchinson_diag_estimate,
    resolve_attention_routing,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SophiaCudaRoutingTests(unittest.TestCase):
    def test_hybrid_math_and_persistent_bob_routing(self):
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        cfg = GPTConfig(
            vocab_size=32,
            block_size=16,
            n_layer=1,
            n_head=1,
            n_embd=32,
        )
        idx = torch.arange(16, device="cuda").remainder(cfg.vocab_size)[None, :]
        targets = (idx + 1).remainder(cfg.vocab_size)

        cases = (
            ("math", "flash"),
            ("bob", None),
        )
        for hessian_backend, non_bob_backend in cases:
            with self.subTest(
                hessian_backend=hessian_backend,
                non_bob_backend=non_bob_backend,
            ):
                routing = resolve_attention_routing(
                    optimizer="sophiah",
                    attn_backend=hessian_backend,
                    non_bob_attn_backend=non_bob_backend,
                )
                model = GPT(cfg, routing.ordinary_backend).to(
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                params = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]

                with hessian_attention_context(model, routing):
                    estimates = hutchinson_diag_estimate(
                        model,
                        params,
                        [(idx, targets)],
                        num_samples=1,
                    )

                self.assertTrue(
                    all(torch.isfinite(estimate).all() for estimate in estimates)
                )
                self.assertEqual(
                    {
                        module.backend
                        for module in model.modules()
                        if isinstance(module, CausalSelfAttention)
                    },
                    {routing.ordinary_backend},
                )

                model.zero_grad(set_to_none=True)
                loss = ce_loss_from_logits(model(idx), targets)
                loss.backward()
                torch.cuda.synchronize()
                self.assertTrue(all(parameter.grad is not None for parameter in params))
                self.assertTrue(
                    all(torch.isfinite(parameter.grad).all() for parameter in params)
                )
                self.assertEqual(
                    {
                        module.backend
                        for module in model.modules()
                        if isinstance(module, CausalSelfAttention)
                    },
                    {routing.ordinary_backend},
                )


if __name__ == "__main__":
    unittest.main()
