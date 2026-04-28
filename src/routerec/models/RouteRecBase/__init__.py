"""
RouteRecBase — 3-Stage Hierarchical Mixture-of-Experts Sequential Recommender.

Modules:
    feature_config  — Feature column → stage/expert mapping.
    experts         — Expert MLP modules.
    routers         — Gating network (dense / top-k) + aux losses.
    moe_stages      — MoEStage, HierarchicalMoE (Macro→Mid→Micro).
    transformer     — Transformer backbone (standard + MoE-FFN variant).
    featured_moe    — Main RecBole SequentialRecommender class.
    logging_utils   — Expert weight logging / analysis utilities.
"""

from .featured_moe import RouteRecBase

__all__ = ["RouteRecBase"]
