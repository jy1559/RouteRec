"""TiSASRec implementation for RecBole-style sequential recommendation.

Adapted from the official TiSASRec TensorFlow implementation and ported to
PyTorch/RecBole interfaces used in this repository.
"""

import math

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class TiSASMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout_prob)
        self.out_dropout = nn.Dropout(dropout_prob)

    def _split_heads(self, x):
        # [B, L, H] -> [B, nH, L, dH]
        bsz, seqlen, _ = x.size()
        x = x.view(bsz, seqlen, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        # [B, nH, L, dH] -> [B, L, H]
        bsz, _, seqlen, _ = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bsz, seqlen, self.hidden_size)

    def forward(self, x, time_k, time_v, abs_pos_k, abs_pos_v, attn_mask):
        # x: [B, L, H]
        # time_k/time_v: [B, L, L, H]
        # abs_pos_k/abs_pos_v: [B, L, H]
        # attn_mask: [B, 1, L, L] (0 or large negative)
        bsz, seqlen, _ = x.size()

        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        abs_k = self._split_heads(abs_pos_k)
        abs_v = self._split_heads(abs_pos_v)

        time_k = time_k.view(bsz, seqlen, seqlen, self.num_heads, self.head_dim)
        time_k = time_k.permute(0, 3, 1, 2, 4)
        time_v = time_v.view(bsz, seqlen, seqlen, self.num_heads, self.head_dim)
        time_v = time_v.permute(0, 3, 1, 2, 4)

        scores_item = torch.matmul(q, k.transpose(-2, -1))
        scores_pos = torch.matmul(q, abs_k.transpose(-2, -1))
        scores_time = (q.unsqueeze(3) * time_k).sum(dim=-1)

        scores = (scores_item + scores_pos + scores_time) / math.sqrt(self.head_dim)
        scores = scores + attn_mask

        probs = torch.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        context_item = torch.matmul(probs, v)
        context_pos = torch.matmul(probs, abs_v)
        context_time = (probs.unsqueeze(-1) * time_v).sum(dim=-2)

        context = context_item + context_pos + context_time
        context = self._merge_heads(context)
        context = self.out(context)
        context = self.out_dropout(context)
        return context


class TiSASRecBlock(nn.Module):
    def __init__(self, hidden_size, inner_size, num_heads, dropout_prob):
        super().__init__()
        self.attn_layer_norm = nn.LayerNorm(hidden_size)
        self.attn = TiSASMultiHeadAttention(hidden_size, num_heads, dropout_prob)

        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x, time_k, time_v, abs_pos_k, abs_pos_v, attn_mask, pad_mask):
        x = x + self.attn(self.attn_layer_norm(x), time_k, time_v, abs_pos_k, abs_pos_v, attn_mask)
        x = x * pad_mask
        x = x + self.ffn(self.ffn_layer_norm(x))
        x = x * pad_mask
        return x


class TiSASRec(SequentialRecommender):
    input_type = "point"

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        def cfg_get(key, default=None):
            try:
                return config[key]
            except Exception:
                return default

        self.n_layers = int(cfg_get("n_layers", cfg_get("num_layers", 2)))
        self.n_heads = int(cfg_get("n_heads", cfg_get("num_heads", 2)))
        self.hidden_size = int(cfg_get("hidden_size", cfg_get("embedding_size", 64)))
        self.inner_size = int(cfg_get("inner_size", self.hidden_size * 4))
        self.hidden_dropout_prob = float(cfg_get("hidden_dropout_prob", 0.2))
        self.initializer_range = float(cfg_get("initializer_range", 0.02))
        self.time_span = int(cfg_get("time_span", 256))
        self.loss_type = str(cfg_get("loss_type", "CE"))

        time_field = str(cfg_get("TIME_FIELD", "timestamp"))
        list_suffix = str(cfg_get("LIST_SUFFIX", "_list"))
        self.time_seq_field = f"{time_field}{list_suffix}"

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.abs_pos_key_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.abs_pos_value_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.time_key_embedding = nn.Embedding(self.time_span + 1, self.hidden_size)
        self.time_value_embedding = nn.Embedding(self.time_span + 1, self.hidden_size)

        self.input_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.emb_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.blocks = nn.ModuleList(
            [
                TiSASRecBlock(
                    hidden_size=self.hidden_size,
                    inner_size=self.inner_size,
                    num_heads=self.n_heads,
                    dropout_prob=self.hidden_dropout_prob,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_time_seq(self, interaction, item_seq):
        raw = interaction.interaction.get(self.time_seq_field)
        if raw is None:
            return torch.zeros_like(item_seq)

        time_seq = raw
        if not torch.is_tensor(time_seq):
            time_seq = torch.as_tensor(time_seq, device=item_seq.device)
        time_seq = time_seq.to(item_seq.device)
        if time_seq.dim() > 2:
            time_seq = time_seq.squeeze(-1)
        return time_seq.long()

    def _build_time_matrix(self, item_seq, time_seq):
        # Pairwise time-interval matrix clipped by time_span.
        time_diff = (time_seq.unsqueeze(-1) - time_seq.unsqueeze(-2)).abs()
        time_diff = torch.clamp(time_diff, max=self.time_span)

        valid = (item_seq > 0)
        valid_pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        time_diff = time_diff * valid_pair.long()
        return time_diff

    def forward(self, item_seq, item_seq_len, interaction=None):
        bsz, seqlen = item_seq.size()
        pad_mask = (item_seq > 0).unsqueeze(-1).to(dtype=self.item_embedding.weight.dtype)

        x = self.item_embedding(item_seq)
        x = self.input_dropout(x)
        x = x * pad_mask

        pos_ids = torch.arange(seqlen, dtype=torch.long, device=item_seq.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(item_seq)
        abs_pos_k = self.emb_dropout(self.abs_pos_key_embedding(pos_ids))
        abs_pos_v = self.emb_dropout(self.abs_pos_value_embedding(pos_ids))

        if interaction is None:
            time_seq = torch.zeros_like(item_seq)
        else:
            time_seq = self._get_time_seq(interaction, item_seq)
        time_matrix = self._build_time_matrix(item_seq, time_seq)
        time_k = self.emb_dropout(self.time_key_embedding(time_matrix))
        time_v = self.emb_dropout(self.time_value_embedding(time_matrix))

        attn_mask = self.get_attention_mask(item_seq)

        for block in self.blocks:
            x = block(x, time_k, time_v, abs_pos_k, abs_pos_v, attn_mask, pad_mask)

        x = self.final_layer_norm(x)
        output = self.gather_indexes(x, item_seq_len - 1)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, interaction=interaction)
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
        seq_output = self.forward(item_seq, item_seq_len, interaction=interaction)
        test_item_emb = self.item_embedding(test_item)
        return torch.mul(seq_output, test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, interaction=interaction)
        test_items_emb = self.item_embedding.weight
        return torch.matmul(seq_output, test_items_emb.transpose(0, 1))
