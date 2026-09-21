"""GQA language-model architectures used by the model benchmarks.

The Llama and SmolLM definitions are local, randomly initialized models with
the released architecture shapes. Qwen and Granite use lazily imported
Transformers implementations so the core package does not depend on
Transformers.
"""

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bob import sdpa_bob
from benchmarks.models.rotary import apply_rotary


LLAMA_PRESETS = (
    "llama-3.2-1b",
    "llama-3.2-3b",
    "smollm2-135m",
    "smollm2-360m",
)
TRANSFORMERS_PRESETS = ("qwen2.5-1.5b", "granite-3.1-1b-a400m")


@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    block_size: int = 131072
    n_layer: int = 28
    n_head: int = 24
    n_kv_head: int = 8
    n_embd: int = 3072
    intermediate_size: int = 8192
    rope_theta: float = 500000.0
    rope_factor: float = 32.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_context: int = 8192
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_llama3_rope_scaling: bool = True

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head

    @property
    def group_size(self) -> int:
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        return self.n_head // self.n_kv_head


def llama_config(preset: str) -> LlamaConfig:
    if preset == "llama-3.2-1b":
        return LlamaConfig(
            n_layer=16,
            n_head=32,
            n_kv_head=8,
            n_embd=2048,
            intermediate_size=8192,
        )
    if preset == "llama-3.2-3b":
        return LlamaConfig()
    if preset == "smollm2-135m":
        return LlamaConfig(
            vocab_size=49152,
            block_size=8192,
            n_layer=30,
            n_head=9,
            n_kv_head=3,
            n_embd=576,
            intermediate_size=1536,
            rope_theta=100000.0,
            initializer_range=1.0 / 24.0,
            use_llama3_rope_scaling=False,
        )
    if preset == "smollm2-360m":
        return LlamaConfig(
            vocab_size=49152,
            block_size=8192,
            n_layer=32,
            n_head=15,
            n_kv_head=5,
            n_embd=960,
            intermediate_size=2560,
            rope_theta=100000.0,
            use_llama3_rope_scaling=False,
        )
    raise ValueError(f"unknown Llama-family preset: {preset!r}")


def _rope_cache(
    sequence_length: int,
    config: LlamaConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    exponent = (
        torch.arange(0, config.head_dim, 2, device=device, dtype=torch.float32)
        / config.head_dim
    )
    inverse_frequency = 1.0 / (config.rope_theta**exponent)

    if config.use_llama3_rope_scaling:
        wavelength = 2.0 * math.pi / inverse_frequency
        low_wavelength = config.rope_original_context / config.rope_low_freq_factor
        high_wavelength = config.rope_original_context / config.rope_high_freq_factor
        scaled = inverse_frequency / config.rope_factor
        smooth = (
            config.rope_original_context / wavelength - config.rope_low_freq_factor
        ) / (config.rope_high_freq_factor - config.rope_low_freq_factor)
        smoothed = (1.0 - smooth) * scaled + smooth * inverse_frequency
        inverse_frequency = torch.where(
            wavelength > low_wavelength, scaled, inverse_frequency
        )
        medium = (wavelength <= low_wavelength) & (wavelength >= high_wavelength)
        inverse_frequency = torch.where(medium, smoothed, inverse_frequency)

    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequency)
    embeddings = torch.cat((angles, angles), dim=-1)
    return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        dtype = tensor.dtype
        normalized = tensor.float()
        normalized *= torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return self.weight * normalized.to(dtype)


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        backend: str,
        *,
        window_size: int = 0,
    ):
        super().__init__()
        if backend not in {"bob", "flash", "math", "hvp_manual", "hvp_semi_manual"}:
            raise ValueError(f"unsupported GQA backend: {backend!r}")
        if window_size:
            raise NotImplementedError("the GQA model benchmark uses square dense attention")
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.backend = backend

        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.n_embd, config.n_kv_head * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.n_embd, config.n_kv_head * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, sequence, self.n_head, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch, sequence, self.n_kv_head, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, sequence, self.n_kv_head, self.head_dim
        )
        query = apply_rotary(query.transpose(1, 2).contiguous(), cosine, sine)
        key = apply_rotary(key.transpose(1, 2).contiguous(), cosine, sine)
        value = value.transpose(1, 2).contiguous()

        if self.backend == "bob":
            attended = sdpa_bob(query, key, value, is_causal=True)
        elif self.backend == "flash":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=True,
                    enable_gqa=True,
                )
        else:
            # The reference and imported HVP baselines are MHA operators. Their
            # Match PyTorch SDPA's explicit grouped-query expansion.
            group_size = self.n_head // self.n_kv_head
            key = key.repeat_interleave(group_size, dim=1)
            value = value.repeat_interleave(group_size, dim=1)
            from benchmarks.models.language_model_benchmark import ATTN_BACKENDS

            attended = ATTN_BACKENDS[self.backend](
                query, key, value, is_causal=True
            )

        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, self.n_embd
        )
        return self.o_proj(attended)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(tensor)) * self.up_proj(tensor))


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig, backend: str, **attention_options):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = LlamaAttention(config, backend, **attention_options)
        self.post_attention_layernorm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.input_layernorm(hidden_states), cosine, sine
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Llama(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        backend: str,
        *,
        window_size: int = 0,
    ):
        super().__init__()
        self.cfg = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(
            LlamaBlock(
                config,
                backend,
                window_size=window_size,
            )
            for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length > self.cfg.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block size {self.cfg.block_size}"
            )
        hidden_states = self.embed_tokens(tokens)
        cosine, sine = _rope_cache(
            sequence_length, self.cfg, tokens.device, hidden_states.dtype
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, cosine, sine)
        return self.lm_head(self.norm(hidden_states))


def _register_flashbob_attention() -> None:
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    def flashbob_attention(
        _module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        **_kwargs,
    ):
        if attention_mask is not None:
            raise RuntimeError("FlashBoB model benchmarks require an implicit causal mask")
        output = sdpa_bob(query, key, value, is_causal=True, scale=scaling)
        return output.transpose(1, 2).contiguous(), None

    def implicit_causal_mask(**kwargs):
        if kwargs.get("attention_mask") is not None:
            raise RuntimeError("padded or custom masks are outside this benchmark")
        return None

    AttentionInterface.register("bob_hvp", flashbob_attention)
    AttentionMaskInterface.register("bob_hvp", implicit_causal_mask)


def transformers_config(preset: str):
    if preset == "qwen2.5-1.5b":
        from transformers import Qwen2Config

        config = Qwen2Config.from_dict(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "attention_bias": True,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151643,
                "hidden_act": "silu",
                "hidden_size": 1536,
                "initializer_range": 0.02,
                "intermediate_size": 8960,
                "max_position_embeddings": 131072,
                "max_window_layers": 28,
                "model_type": "qwen2",
                "num_attention_heads": 12,
                "num_hidden_layers": 28,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000.0,
                "sliding_window": 32768,
                "tie_word_embeddings": True,
                "use_cache": False,
                "use_sliding_window": False,
                "vocab_size": 151936,
            }
        )
    elif preset == "granite-3.1-1b-a400m":
        from transformers.models.granitemoe import GraniteMoeConfig

        config = GraniteMoeConfig.from_dict(
            {
                "architectures": ["GraniteMoeForCausalLM"],
                "attention_bias": False,
                "attention_dropout": 0.1,
                "attention_multiplier": 0.015625,
                "bos_token_id": 0,
                "embedding_multiplier": 12.0,
                "eos_token_id": 0,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "initializer_range": 0.02,
                "intermediate_size": 512,
                "logits_scaling": 6.0,
                "max_position_embeddings": 131072,
                "model_type": "granitemoe",
                "num_attention_heads": 16,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 8,
                "num_local_experts": 32,
                "output_router_logits": False,
                "pad_token_id": 0,
                "residual_multiplier": 0.22,
                "rms_norm_eps": 1e-6,
                "rope_scaling": None,
                "rope_theta": 1_500_000.0,
                "router_aux_loss_coef": 0.001,
                "tie_word_embeddings": True,
                "torch_dtype": "bfloat16",
                "use_cache": False,
                "vocab_size": 49152,
            }
        )
    else:
        raise ValueError(f"unknown Transformers preset: {preset!r}")
    config.benchmark_preset = preset
    config.block_size = config.max_position_embeddings
    return config


class TransformersCausalLM(nn.Module):
    """Uniform benchmark adapter for Qwen2 and Granite MoE."""

    def __init__(
        self,
        config,
        backend: str,
        *,
        window_size: int = 0,
    ):
        super().__init__()
        if backend not in {"bob", "math"}:
            raise ValueError("Qwen and Granite benchmarks support only bob and math")
        if window_size:
            raise NotImplementedError("the Transformers benchmarks use square dense attention")
        config = copy.deepcopy(config)
        if backend == "bob":
            _register_flashbob_attention()
            config._attn_implementation = "bob_hvp"
        else:
            config._attn_implementation = "eager"

        if config.benchmark_preset == "qwen2.5-1.5b":
            from transformers import Qwen2ForCausalLM

            self.model = Qwen2ForCausalLM(config)
        else:
            from transformers.models.granitemoe import GraniteMoeForCausalLM

            self.model = GraniteMoeForCausalLM(config)
        self.cfg = config

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens).logits


__all__ = [
    "LLAMA_PRESETS",
    "TRANSFORMERS_PRESETS",
    "Llama",
    "LlamaAttention",
    "LlamaConfig",
    "TransformersCausalLM",
    "llama_config",
    "transformers_config",
]
