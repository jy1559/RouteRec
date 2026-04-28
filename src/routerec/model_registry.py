"""Model presets and bounded search spaces for the RouteRec paper repository."""

from __future__ import annotations

from copy import deepcopy

from .datasets import normalize_dataset_name


PAPER_BASELINES = (
    "SASRec",
    "GRU4Rec",
    "TiSASRec",
    "DuoRec",
    "SIGMA",
    "BSARec",
    "FEARec",
    "DIFSR",
    "FAME",
)

LOCAL_BASELINE_IMPLEMENTATIONS = (
    "BSARec",
    "DIFSR",
    "DuoRec",
    "FAME",
    "FDSA",
    "FEARec",
    "SIGMA",
    "TiSASRec",
)

DATASET_LR_INTERVALS = {
    "amazon_beauty": (1.5e-4, 2.0e-3),
    "beauty": (1.5e-4, 2.0e-3),
    "foursquare": (1.5e-4, 2.2e-3),
    "KuaiRecLargeStrictPosV2_0.2": (3.0e-4, 5.0e-3),
    "lastfm0.03": (8.0e-5, 1.2e-3),
    "movielens1m": (1.5e-4, 2.2e-3),
    "retail_rocket": (1.5e-4, 2.2e-3),
}

ROUTEREC_DEFAULT = {
    "model": "RouteRec",
    "MAX_ITEM_LIST_LENGTH": 20,
    "embedding_size": 192,
    "hidden_size": 192,
    "d_ff": 384,
    "d_expert_hidden": 192,
    "d_router_hidden": 64,
    "d_feat_emb": 16,
    "expert_scale": 3,
    "hidden_dropout_prob": 0.15,
    "fixed_hidden_dropout_prob": 0.15,
    "attn_dropout_prob": 0.10,
    "learning_rate": 5.5e-4,
    "weight_decay": 1.0e-6,
    "lr_scheduler_type": "warmup_cosine",
    "macro_history_window": 5,
    "route_consistency_lambda": 2.5e-4,
    "z_loss_lambda": 1.0e-4,
    "router_impl": "learned",
    "router_use_feature": True,
    "router_use_hidden": True,
    "router_feature_proj_dim": 0,
    "stage_feature_dropout_prob": 0.03,
    "stage_family_dropout_prob": {"macro": 0.02, "mid": 0.02, "micro": 0.02},
    "layer_layout": ["layer", "layer", "layer"],
    "stage_router_granularity": {"macro": "session", "mid": "session", "micro": "token"},
}

ROUTEREC_DATASET_PRESETS = {
    "KuaiRecLargeStrictPosV2_0.2": {
        "MAX_ITEM_LIST_LENGTH": 20,
        "embedding_size": 224,
        "hidden_size": 224,
        "d_ff": 448,
        "d_expert_hidden": 224,
        "d_router_hidden": 64,
        "d_feat_emb": 20,
        "hidden_dropout_prob": 0.12,
        "fixed_hidden_dropout_prob": 0.15,
        "attn_dropout_prob": 0.07,
        "learning_rate": 5.476e-4,
        "weight_decay": 1.6e-6,
        "route_consistency_lambda": 1.2e-3,
        "z_loss_lambda": 1.0e-4,
        "stage_feature_dropout_prob": 0.03,
    },
    "amazon_beauty": {
        "MAX_ITEM_LIST_LENGTH": 20,
        "embedding_size": 192,
        "hidden_size": 192,
        "d_ff": 384,
        "d_expert_hidden": 192,
        "d_router_hidden": 32,
        "d_feat_emb": 16,
        "hidden_dropout_prob": 0.18,
        "fixed_hidden_dropout_prob": 0.20,
        "attn_dropout_prob": 0.12,
        "learning_rate": 5.672e-4,
        "weight_decay": 5.0e-7,
        "route_consistency_lambda": 2.5e-4,
        "z_loss_lambda": 2.0e-4,
        "stage_feature_dropout_prob": 0.10,
    },
    "foursquare": {
        "MAX_ITEM_LIST_LENGTH": 30,
        "embedding_size": 160,
        "hidden_size": 160,
        "d_ff": 320,
        "d_expert_hidden": 160,
        "d_router_hidden": 32,
        "d_feat_emb": 8,
        "hidden_dropout_prob": 0.12,
        "fixed_hidden_dropout_prob": 0.15,
        "attn_dropout_prob": 0.05,
        "learning_rate": 9.5164e-4,
        "weight_decay": 1.2e-6,
        "route_consistency_lambda": 2.5e-4,
        "z_loss_lambda": 2.0e-4,
        "stage_feature_dropout_prob": 0.10,
    },
    "lastfm0.03": {
        "MAX_ITEM_LIST_LENGTH": 30,
        "embedding_size": 224,
        "hidden_size": 224,
        "d_ff": 448,
        "d_expert_hidden": 224,
        "d_router_hidden": 96,
        "d_feat_emb": 12,
        "hidden_dropout_prob": 0.12,
        "fixed_hidden_dropout_prob": 0.14,
        "attn_dropout_prob": 0.12,
        "learning_rate": 4.983e-4,
        "weight_decay": 5.0e-7,
        "route_consistency_lambda": 2.5e-4,
        "z_loss_lambda": 1.0e-4,
        "stage_feature_dropout_prob": 0.03,
    },
    "movielens1m": {
        "MAX_ITEM_LIST_LENGTH": 10,
        "embedding_size": 128,
        "hidden_size": 128,
        "d_ff": 256,
        "d_expert_hidden": 128,
        "d_router_hidden": 96,
        "d_feat_emb": 16,
        "hidden_dropout_prob": 0.16,
        "fixed_hidden_dropout_prob": 0.15,
        "attn_dropout_prob": 0.07,
        "learning_rate": 1.29037e-3,
        "weight_decay": 1.0e-6,
        "route_consistency_lambda": 1.2e-3,
        "z_loss_lambda": 4.0e-4,
        "stage_feature_dropout_prob": 0.03,
    },
    "retail_rocket": {
        "MAX_ITEM_LIST_LENGTH": 20,
        "embedding_size": 192,
        "hidden_size": 192,
        "d_ff": 384,
        "d_expert_hidden": 192,
        "d_router_hidden": 64,
        "d_feat_emb": 8,
        "hidden_dropout_prob": 0.16,
        "fixed_hidden_dropout_prob": 0.14,
        "attn_dropout_prob": 0.12,
        "learning_rate": 5.6966e-4,
        "weight_decay": 1.0e-6,
        "route_consistency_lambda": 2.5e-4,
        "z_loss_lambda": 4.0e-4,
        "stage_feature_dropout_prob": 0.03,
    },
}

BASELINE_DEFAULTS = {
    "SASRec": {"learning_rate": 7.0e-4, "MAX_ITEM_LIST_LENGTH": 20, "hidden_size": 128, "n_layers": 2, "n_heads": 2, "hidden_dropout_prob": 0.15, "attn_dropout_prob": 0.10},
    "GRU4Rec": {"learning_rate": 3.0e-3, "MAX_ITEM_LIST_LENGTH": 20, "hidden_size": 128, "n_layers": 1, "dropout_prob": 0.20},
    "TiSASRec": {"learning_rate": 5.0e-4, "MAX_ITEM_LIST_LENGTH": 20, "hidden_size": 128, "n_layers": 2, "n_heads": 2, "hidden_dropout_prob": 0.15, "time_span": 128},
    "DuoRec": {"learning_rate": 4.0e-4, "MAX_ITEM_LIST_LENGTH": 20, "hidden_size": 128, "n_layers": 2, "n_heads": 2, "hidden_dropout_prob": 0.15, "tau": 0.20, "lmd": 0.05, "lmd_sem": 0.05},
    "SIGMA": {"learning_rate": 3.0e-4, "MAX_ITEM_LIST_LENGTH": 20, "hidden_size": 128, "n_layers": 2, "n_heads": 2, "hidden_dropout_prob": 0.15},
    "BSARec": {"learning_rate": 5.0e-4, "MAX_ITEM_LIST_LENGTH": 20, "hidden_size": 128, "n_layers": 2, "n_heads": 2, "hidden_dropout_prob": 0.15, "alpha": 0.50, "c": 3},
    "FEARec": {"learning_rate": 3.5e-4, "MAX_ITEM_LIST_LENGTH": 20, "hidden_size": 128, "n_layers": 2, "n_heads": 2, "hidden_dropout_prob": 0.15, "tau": 0.20, "semantic_weight": 0.05},
    "DIFSR": {"learning_rate": 8.0e-4, "MAX_ITEM_LIST_LENGTH": 20, "hidden_size": 128, "n_layers": 2, "n_heads": 2, "hidden_dropout_prob": 0.15, "attribute_hidden_size": 128, "fusion_type": "gate"},
    "FAME": {"learning_rate": 8.0e-4, "MAX_ITEM_LIST_LENGTH": 20, "hidden_size": 128, "n_layers": 2, "n_heads": 4, "hidden_dropout_prob": 0.15, "num_experts": 4},
}

PAPER_BOUNDED_GRID = {
    "shared": {
        "weight_decay": [5e-7, 1e-6, 1e-5, 5e-5, 1e-4, 1.5e-4],
        "max_history_length": [10, 20, 30, 50],
        "width_embedding": [64, 96, 112, 128, 160, 192],
        "inner_width": [128, 192, 224, 256, 320, 384],
        "layers": [1, 2, 3, 4],
        "heads": [1, 2, 4, 8],
        "hidden_dropout": [0.10, 0.12, 0.13, 0.15, 0.18],
        "recurrent_dropout": [0.10, 0.15, 0.20, 0.25, 0.30],
        "attention_dropout": [0.06, 0.08, 0.10, 0.12, 0.15, 0.20],
    },
    "baseline_additions": {
        "TiSASRec": {"time_span": [64, 128, 256, 384, 512]},
        "GRU4Rec": {"dropout_prob": [0.10, 0.15, 0.20, 0.25, 0.30]},
        "DuoRec": {"tau": [0.16, 0.18, 0.20, 0.22, 0.24], "contrastive_weight": [0.02, 0.03, 0.04, 0.05, 0.06], "semantic_weight": [0.0, 0.04, 0.05, 0.08, 0.10]},
        "FEARec": {"tau": [0.16, 0.18, 0.20, 0.22, 0.24], "contrastive_weight": [0.02, 0.03, 0.04, 0.05, 0.06], "semantic_weight": [0.04, 0.05, 0.08, 0.10, 0.12]},
        "BSARec": {"alpha": [0.35, 0.50, 0.55, 0.70], "c": [2, 3, 5, 7]},
        "DIFSR": {"attribute_hidden_size": [96, 128, 160, 192], "lambda_attr": [0.08, 0.09, 0.10, 0.12, 0.14], "fusion_type": ["gate", "sum", "concat"]},
        "FDSA": {"attribute_hidden_size": [96, 128, 160, 192], "lambda_attr": [0.09, 0.10, 0.12, 0.14, 0.15]},
        "FAME": {"num_experts": [2, 3, 4, 5, 6]},
    },
    "routerec_additions": {
        "backbone_depth": [1, 2, 3],
        "dropout": [0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.24],
        "weight_decay": [3.125e-7, 5e-7, 6e-7, 1e-6, 1.5e-6, 2e-6, 5e-6],
        "expert_scale": [2, 3, 4],
        "expert_scale_kuairec": [2, 3, 4, 5],
        "router_width": [32, 64, 96, 128],
        "feature_dropout": [0.0, 0.03, 0.05, 0.10],
        "attention_dropout": [0.05, 0.06, 0.08, 0.10, 0.12],
        "route_consistency_lambda": [0.0, 2.5e-4, 5e-4, 8e-4, 1.2e-3],
        "z_loss_lambda": [0.0, 5e-5, 1e-4, 2e-4],
        "cue_bank_size": [10, 12, 16, 24],
    },
}


def recommended_routerec_config(dataset: str | None = None) -> dict:
    dataset_name = normalize_dataset_name(dataset) if dataset else None
    config = deepcopy(ROUTEREC_DEFAULT)
    if dataset_name and dataset_name in ROUTEREC_DATASET_PRESETS:
        config.update(deepcopy(ROUTEREC_DATASET_PRESETS[dataset_name]))
    if dataset_name and dataset_name in DATASET_LR_INTERVALS and "learning_rate_range" not in config:
        config["learning_rate_range"] = list(DATASET_LR_INTERVALS[dataset_name])
    return config