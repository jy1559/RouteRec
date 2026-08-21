"""Expert MLP modules for hidden-aware stage MoE.

Each expert consumes:
- token hidden state (optional)
- expert-specific feature subset (optional, projected to d_feat_emb)

and produces a d_model-dim residual update candidate.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ExpertMLP(nn.Module):
    """Single hidden-aware expert.

    Architecture:
      feature_proj: Linear(d_feat_in -> d_feat_emb)  (optional)
      net: Linear(d_in -> d_hidden) -> GELU -> Dropout -> Linear(d_hidden -> d_out)
    """

    def __init__(
        self,
        d_feat_in: int,
        d_model: int,
        d_feat_emb: int,
        d_hidden: int,
        d_out: int,
        use_hidden: bool = True,
        use_feature: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not (use_hidden or use_feature):
            raise ValueError("ExpertMLP requires at least one input source.")

        self.use_hidden = bool(use_hidden)
        self.use_feature = bool(use_feature)

        if self.use_feature:
            self.feature_proj = nn.Linear(d_feat_in, d_feat_emb)
        else:
            self.feature_proj = None

        d_in = (d_model if self.use_hidden else 0) + (d_feat_emb if self.use_feature else 0)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        expert_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden: [*, d_model]
            expert_feat: [*, d_feat_in]
        Returns:
            [*, d_out]
        """
        inputs = []
        if self.use_hidden:
            inputs.append(hidden)
        if self.use_feature:
            inputs.append(self.feature_proj(expert_feat))
        x = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
        return self.net(x)


class ExpertGroup(nn.Module):
    """A group of K hidden-aware experts for one MoE stage.

    Each expert can have a different input dimension (different number of
    assigned features), but they all produce the same ``d_out`` dimension.

    Parameters
    ----------
    expert_feature_dims : list[int]
        Feature input dimensions for each expert (length = K).
    d_model : int
        Hidden state dimension.
    d_feat_emb : int
        Feature embedding dimension.
    d_hidden : int
        Hidden dimension inside each expert MLP.
    d_out : int
        Output dimension shared by all experts.
    use_hidden : bool
        Whether experts use hidden state.
    use_feature : bool
        Whether experts use feature embedding.
    dropout : float
        Dropout probability inside expert MLPs.
    """

    def __init__(
        self,
        expert_feature_dims: List[int],
        d_model: int,
        d_feat_emb: int,
        d_hidden: int,
        d_out: int,
        use_hidden: bool = True,
        use_feature: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = len(expert_feature_dims)
        self.experts = nn.ModuleList([
            ExpertMLP(
                d_feat_in=d_in,
                d_model=d_model,
                d_feat_emb=d_feat_emb,
                d_hidden=d_hidden,
                d_out=d_out,
                use_hidden=use_hidden,
                use_feature=use_feature,
                dropout=dropout,
            )
            for d_in in expert_feature_dims
        ])

    def forward(
        self,
        hidden: torch.Tensor,
        expert_inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Run all experts and stack outputs.

        Args:
            hidden: [B, (T,) d_model]
            expert_inputs: list of K tensors, each [B, (T,) d_in_k].
        Returns:
            [B, (T,) K, d_out]  — stacked expert embeddings.
        """
        # Each expert_k(input_k) → [*, d_out]
        outputs = [
            expert(hidden, inp)  # [*, d_out]
            for expert, inp in zip(self.experts, expert_inputs)
        ]
        return torch.stack(outputs, dim=-2)  # [*, K, d_out]
