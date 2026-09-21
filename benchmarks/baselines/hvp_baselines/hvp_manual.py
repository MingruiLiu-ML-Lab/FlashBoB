import torch

from transformers.integrations.sdpa_attention import repeat_kv

from typing import Optional


class SDPA_FullManualBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_out, q, k, v, attn_mask, is_causal, dropout_mask, dropout_scale, scale):
        ctx.save_for_backward(grad_out, q, k, v, dropout_mask)
        ctx.attn_mask = attn_mask
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.dropout_scale = dropout_scale
        d = q.size(-1)
        scale = scale if (scale is not None) else (1 / d ** 0.5)

        scores = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(q.dtype)
            scores = scores + attn_mask
        if is_causal:
            S, L = scores.size(-2), scores.size(-1)
            causal = torch.triu(scores.new_full((S, L), float("-inf")), diagonal=1).to(q.dtype)
            scores = scores + causal
        
        P = torch.softmax(scores, dim=-1)
        # use cached dropout mask
        if dropout_mask is not None:
            P = P * dropout_mask * dropout_scale

        dV  = P.transpose(-2, -1) @ grad_out
        dP  = grad_out @ v.transpose(-2, -1)
        Ssm = (dP * P).sum(dim=-1, keepdim=True)
        dS  = (dP - Ssm) * P
        dQ  = dS @ k * scale
        dK  = dS.transpose(-2, -1) @ q * scale
        return dQ, dK, dV, None, None, None, None
    
    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v, *_):
        grad_out, q, k, v, dropout_mask = ctx.saved_tensors
        attn_mask, is_causal = ctx.attn_mask, ctx.is_causal
        dropout_scale, scale = ctx.dropout_scale, ctx.scale

        d = q.size(-1)
        scale = scale if (scale is not None) else (1 / d ** 0.5)

        scores = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(q.dtype)
            scores = scores + attn_mask
        if is_causal:
            S, L = scores.size(-2), scores.size(-1)
            causal = torch.triu(scores.new_full((S, L), float("-inf")), diagonal=1).to(q.dtype)
            scores = scores + causal

        P = torch.softmax(scores, dim=-1)
        # use cached dropout mask
        if dropout_mask is not None:
            P = P * dropout_mask * dropout_scale

        dP  = grad_out @ v.transpose(-2, -1)
        Ssm = (dP * P).sum(dim=-1, keepdim=True)
        dS  = (dP - Ssm) * P

        bar_dS = (grad_q @ k.transpose(-2, -1) + q @ grad_k.transpose(-2, -1)) * scale
        beta = (bar_dS * P).sum(dim=-1, keepdim=True)
        bar_dP = P * (bar_dS - beta)
        bar_P  = grad_out @ grad_v.transpose(-2, -1) + bar_dS * (dP - Ssm) - dP * beta

        tmp = (bar_P * P).sum(dim=-1, keepdim=True)
        bar_scores = (bar_P - tmp) * P

        dgrad_grad_out = P @ grad_v + bar_dP @ v
        dgrad_q        = (dS @ grad_k + bar_scores @ k) * scale
        dgrad_k        = (dS.transpose(-2, -1) @ grad_q + bar_scores.transpose(-2, -1) @ q) * scale
        dgrad_v        = bar_dP.transpose(-2, -1) @ grad_out

        return dgrad_grad_out, dgrad_q, dgrad_k, dgrad_v, None, None, None, None, None
    

class SDPA_FullManual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        ctx.attn_mask = attn_mask
        ctx.is_causal = bool(is_causal)
        ctx.scale = scale

        d = q.size(-1)
        scale = scale if (scale is not None) else (1 / d ** 0.5)
        scores = (q @ k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(q.dtype)
            scores = scores + attn_mask

        if is_causal:
            S, L = scores.size(-2), scores.size(-1)
            scores = scores + torch.triu(scores.new_full((S, L), float("-inf")), diagonal=1)

        P = torch.softmax(scores, dim=-1)
        if dropout_p and dropout_p > 0.0:
            dropout_mask = (torch.rand_like(P) > dropout_p).to(P.dtype)
            dropout_scale = 1.0 / (1.0 - dropout_p)
        else:
            dropout_mask = None
            dropout_scale = None
        if dropout_mask is not None:
            P = P * dropout_mask * dropout_scale

        ctx.dropout_scale = dropout_scale
        ctx.save_for_backward(q, k, v, dropout_mask)
        
        out = P @ v
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, dropout_mask = ctx.saved_tensors
        attn_mask, is_causal = ctx.attn_mask, ctx.is_causal
        scale, dropout_scale = ctx.scale, ctx.dropout_scale
        return SDPA_FullManualBwd.apply(grad_out, q, k, v, attn_mask, is_causal, dropout_mask, dropout_scale, scale)


def hvp_manual(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        if attention_mask.size(1) < query.size(1):
            attention_mask = attention_mask.expand(-1, query.size(1), -1, -1)
    
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    attn_output = SDPA_FullManual.apply(
        query, key, value, attention_mask, dropout, is_causal, scaling,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
