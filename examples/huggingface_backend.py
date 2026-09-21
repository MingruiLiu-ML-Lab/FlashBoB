"""Register FlashBoB as a Transformers attention implementation.

After registration, a compatible model can load with
`attn_implementation="bob"`. Its attention calls then use FlashBoB without
changes to the model implementation.

Two registrations are needed:

- `AttentionInterface` supplies the attention call itself.
- `AttentionMaskInterface` tells Transformers how to build the mask. Reusing
  the SDPA mask builder matters: it returns `None` for an unpadded causal
  batch. FlashBoB can represent that case with `is_causal` and does not accept
  a materialized additive attention mask.

Requires: pip install '.[benchmarks]'
Run: python examples/huggingface_backend.py
"""

from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, LlamaConfig
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.modeling_utils import AttentionInterface

from bob import sdpa_bob


def bob_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    is_causal=None,
    **kwargs,
):
    """Transformers attention entry point backed by `sdpa_bob`.

    `query` is `[B, H_Q, N, D]` and `key`/`value` are `[B, H_KV, N, D]`;
    Transformers expects `[B, N, H_Q, D]` back, plus attention weights that a
    FlashBoB does not return. Grouped heads retain their input shapes because
    FlashBoB indexes KV heads directly without expanding them.
    """
    if attention_mask is not None:
        raise NotImplementedError(
            "FlashBoB takes no attention bias, so padded batches are unsupported; "
            "use an unpadded causal batch"
        )
    if dropout:
        raise NotImplementedError("FlashBoB does not implement attention dropout")
    out = sdpa_bob(
        query,
        key,
        value,
        is_causal=True if is_causal is None else is_causal,
        scale=scaling,
    )
    return out.transpose(1, 2).contiguous(), None


def hessian_vector_product(model, tokens):
    """`H v` for the whole model at `v = grad`, the shape a second-order optimizer needs."""
    params = [p for p in model.parameters() if p.requires_grad]
    loss = model(input_ids=tokens, labels=tokens).loss
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return torch.autograd.grad(sum((g * g).sum() for g in grads), params)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    AttentionInterface.register("bob", bob_attention_forward)
    AttentionMaskInterface.register("bob", sdpa_mask)

    # Build a small Llama from configuration without downloading weights.
    # Head dimension 64 and bfloat16 are supported on CUDA. Eight query heads
    # over two KV heads exercises the GQA route.
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=1024,
    )
    torch.manual_seed(0)
    tokens = torch.randint(0, config.vocab_size, (2, 512), device="cuda")

    # Use the same architecture and weights for both attention implementations.
    # Each model needs a separate configuration because Transformers stores the
    # implementation choice on the configuration and resolves it during forward.
    reference = AutoModelForCausalLM.from_config(
        deepcopy(config), attn_implementation="sdpa"
    ).cuda().to(torch.bfloat16)
    model = AutoModelForCausalLM.from_config(
        deepcopy(config), attn_implementation="bob"
    ).cuda().to(torch.bfloat16)
    model.load_state_dict(reference.state_dict())

    for name, candidate in (("sdpa", reference), ("bob", model)):
        loss = candidate(input_ids=tokens, labels=tokens).loss
        print(f"{name:5s} attn={candidate.config._attn_implementation:5s} "
              f"loss={float(loss.detach()):.4f}", end="  ")
        try:
            curvature = hessian_vector_product(candidate, tokens)
        except RuntimeError as error:
            print(f"no Hessian-vector product: {error}")
            continue
        total = sum(float(h.float().norm()) for h in curvature)
        print(f"Hv over {len(curvature)} tensors, sum of norms {total:.4f}")


if __name__ == "__main__":
    main()
