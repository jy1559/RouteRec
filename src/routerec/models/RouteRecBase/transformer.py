"""
Transformer encoder backbone for RouteRecBase.

Provides:
  - ``TransformerBlock``  — single encoder layer (self-attn + FFN).
  - ``MoEFFNBlock``       — Transformer layer where FFN is replaced with a
                            small MoE (4 expert FFNs + router).
  - ``TransformerEncoder`` — stack of L blocks with causal + padding mask.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routers import Router, load_balance_loss


# -----------------------------------------------------------------------
# Multi-Head Self-Attention
# -----------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attn_dropout: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(float(dropout if attn_dropout is None else attn_dropout))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x    : [B, T, d_model]
            mask : [B, 1, T, T] or [1, 1, T, T]
                   bool mask (True=block) or additive mask (0 / -inf).
        Returns:
            [B, T, d_model]
        """
        B, T, _ = x.shape
        Q = self.Wq(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Use PyTorch SDPA fast path when available (Flash/MemEff kernels on GPU).
        if hasattr(F, "scaled_dot_product_attention"):
            attn_mask = mask
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask < 0
            out = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
            return self.Wo(out)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)  # [B, H, T, T]
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                scores = scores + mask  # additive mask: 0 for attend, -inf for block
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.Wo(out)


# -----------------------------------------------------------------------
# Feed-Forward Network (standard)
# -----------------------------------------------------------------------

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------
# Standard Transformer Block
# -----------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, attn_dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN transformer
        h = x + self.drop1(self.attn(self.ln1(x), mask))
        h = h + self.drop2(self.ffn(self.ln2(h)))
        return h


# -----------------------------------------------------------------------
# MoE-FFN Transformer Block (optional: ffn_moe=True)
# -----------------------------------------------------------------------

class MoEFFN(nn.Module):
    """FFN replaced by a small MoE of 4 FFN experts + router.
    
    This is the *Transformer-internal MoE* variant (optional extension).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        top_k: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            PositionwiseFFN(d_model, d_ff, dropout)
            for _ in range(n_experts)
        ])
        self.router = Router(
            d_in=d_model,
            n_experts=n_experts,
            d_hidden=d_model // 2,
            top_k=top_k,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, d_model]
        Returns:
            out: [B, T, d_model]
            gate_weights: [B, T, K]
        """
        gate_weights, gate_logits = self.router(x)  # [B, T, K]
        expert_outs = torch.stack([e(x) for e in self.experts], dim=-2)  # [B,T,K,d]
        out = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=-2)     # [B,T,d]
        return out, gate_weights


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE-FFN instead of standard FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_ffn_experts: int = 4,
        ffn_top_k: Optional[int] = None,
        dropout: float = 0.1,
        attn_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, attn_dropout)
        self.moe_ffn = MoEFFN(d_model, d_ff, n_ffn_experts, ffn_top_k, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Returns:
            h            : [B, T, d_model]
            ffn_gate_w   : [B, T, K]   — MoE-FFN gating weights (for logging).
        """
        h = x + self.drop1(self.attn(self.ln1(x), mask))
        ffn_out, ffn_gate_w = self.moe_ffn(self.ln2(h))
        h = h + ffn_out
        return h, ffn_gate_w


# -----------------------------------------------------------------------
# Full Transformer Encoder
# -----------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks with causal + padding masking.

    Supports two modes:
      - ``ffn_moe=False`` (default) → standard TransformerBlock.
      - ``ffn_moe=True``             → MoETransformerBlock (FFN is MoE).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        attn_dropout: Optional[float] = None,
        ffn_moe: bool = False,
        n_ffn_experts: int = 4,
        ffn_top_k: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.ffn_moe = ffn_moe
        d_ff = d_ff or 4 * d_model

        if ffn_moe:
            self.layers = nn.ModuleList([
                MoETransformerBlock(d_model, n_heads, d_ff, n_ffn_experts, ffn_top_k, dropout, attn_dropout)
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, dropout, attn_dropout)
                for _ in range(n_layers)
            ])

        self.final_ln = nn.LayerNorm(d_model)

    @staticmethod
    def build_mask(
        item_seq: torch.Tensor,
        pad_value: int = 0,
    ) -> torch.Tensor:
        """Build combined causal + padding mask (additive style).

        Args:
            item_seq: [B, T] — item id sequence (0 = padding).
        Returns:
            mask: [B, 1, T, T] bool mask (True = block).
        """
        B, T = item_seq.shape
        device = item_seq.device

        # Causal mask (upper triangular → block future)
        causal = torch.triu(
            torch.ones(T, T, device=device), diagonal=1
        ).bool()  # True = block

        # Padding mask: positions where item_seq == pad_value → block
        pad_mask = (item_seq == pad_value)  # [B, T], True = pad

        # Expand: pad positions should not be attended to (as keys)
        # [B, 1, 1, T] — block all queries from attending to pad keys
        pad_mask_key = pad_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]

        # Combine: block if causal OR pad
        combined = causal.unsqueeze(0).unsqueeze(0) | pad_mask_key  # [B,1,T,T]
        return combined

    def forward(
        self,
        x: torch.Tensor,
        item_seq: torch.Tensor,
    ):
        """
        Args:
            x        : [B, T, d_model] — token embeddings.
            item_seq : [B, T]          — item ids (for mask building).
        Returns:
            h : [B, T, d_model]
            ffn_moe_weights : dict {layer_idx: [B,T,K]} if ffn_moe else empty dict.
        """
        mask = self.build_mask(item_seq)
        ffn_moe_weights = {}

        h = x
        if self.ffn_moe:
            for i, layer in enumerate(self.layers):
                h, fw = layer(h, mask)
                ffn_moe_weights[f"layer_{i}"] = fw
        else:
            for layer in self.layers:
                h = layer(h, mask)

        h = self.final_ln(h)
        return h, ffn_moe_weights
