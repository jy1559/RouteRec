"""
FDSA - Feature-level Deeper Self-Attention Network for Sequential Recommendation.

Reference:
    Tingting Zhang et al. "Feature-level Deeper Self-Attention Network for
    Sequential Recommendation." IJCAI 2019.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer, TransformerEncoder, VanillaAttention
from recbole.model.loss import BPRLoss


def _resolve_selected_features(config, dataset) -> List[str]:
    raw = config["selected_features"] if "selected_features" in config else []
    if isinstance(raw, str):
        raw = [raw]
    feats = [str(x).strip() for x in (raw or []) if str(x).strip()]
    if feats:
        return feats

    item_feat = getattr(dataset, "item_feat", None)
    if item_feat is None:
        return []
    cols = set(getattr(item_feat, "columns", []) or [])
    if "category" in cols:
        return ["category"]
    return []


class FDSA(SequentialRecommender):
    """FDSA model with safe feature fallback for feature_added_v4 pipelines."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.selected_features = _resolve_selected_features(config, dataset)
        self.pooling_mode = config["pooling_mode"] if "pooling_mode" in config else "mean"
        self.device = config["device"]
        self.num_feature_field = len(self.selected_features)

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset,
            self.hidden_size,
            self.selected_features,
            self.pooling_mode,
            self.device,
        )

        self.item_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.feature_att_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        self.feature_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss_type must be one of ['BPR', 'CE']")

        self.apply(self._init_weights)
        self.other_parameter_name = ["feature_embed_layer"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _build_feature_emb(self, item_seq, position_embedding):
        if self.num_feature_field <= 0:
            return torch.zeros_like(position_embedding)

        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding["item"]
        dense_embedding = dense_embedding["item"]

        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)

        if not feature_table:
            return torch.zeros_like(position_embedding)

        feature_table = torch.cat(feature_table, dim=-2)
        feature_emb, _ = self.feature_att_layer(feature_table)
        feature_emb = feature_emb + position_embedding
        feature_emb = self.LayerNorm(feature_emb)
        return self.dropout(feature_emb)

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)

        feature_trm_input = self._build_feature_emb(item_seq, position_embedding)

        attention_mask = self.get_attention_mask(item_seq)
        item_output = self.item_trm_encoder(
            item_trm_input, attention_mask, output_all_encoded_layers=True
        )[-1]
        feature_output = self.feature_trm_encoder(
            feature_trm_input, attention_mask, output_all_encoded_layers=True
        )[-1]

        item_output = self.gather_indexes(item_output, item_seq_len - 1)
        feature_output = self.gather_indexes(feature_output, item_seq_len - 1)

        output_concat = torch.cat((item_output, feature_output), dim=-1)
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        return self.dropout(output)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            return self.loss_fct(pos_score, neg_score)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        return torch.mul(seq_output, test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        return torch.matmul(seq_output, test_items_emb.transpose(0, 1))
