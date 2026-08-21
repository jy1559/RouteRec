"""Parallel stage merge router for RouteRec."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .routers import Router


class StageMergeRouter(nn.Module):
    """Learn stage-wise merge weights from base hidden + feature context."""

    def __init__(
        self,
        *,
        n_stages: int,
        n_features: int,
        d_model: int,
        d_feat_emb: int,
        d_router_hidden: int,
        dropout: float,
        top_k: Optional[int],
        use_hidden: bool = True,
        use_feature: bool = True,
    ):
        super().__init__()
        if n_stages <= 0:
            raise ValueError(f"n_stages must be > 0, got {n_stages}")
        if not (use_hidden or use_feature):
            raise ValueError("StageMergeRouter requires at least one input source.")

        self.n_stages = int(n_stages)
        self.use_hidden = bool(use_hidden)
        self.use_feature = bool(use_feature)
        self.top_k = None if top_k is None or int(top_k) <= 0 else int(top_k)

        router_in_dim = 0
        if self.use_hidden:
            router_in_dim += int(d_model)
        if self.use_feature:
            self.feature_proj = nn.Linear(int(n_features), int(d_feat_emb))
            router_in_dim += int(d_feat_emb)
        else:
            self.feature_proj = None

        self.router = Router(
            d_in=router_in_dim,
            n_experts=self.n_stages,
            d_hidden=int(d_router_hidden),
            top_k=self.top_k,
            dropout=float(dropout),
        )

    def _build_input(self, hidden: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        inputs = []
        if self.use_hidden:
            inputs.append(hidden)
        if self.use_feature:
            assert self.feature_proj is not None
            inputs.append(self.feature_proj(feat))
        return inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        temperature: float,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        router_in = self._build_input(hidden, feat)
        return self.router(router_in, temperature=temperature, top_k=top_k)
