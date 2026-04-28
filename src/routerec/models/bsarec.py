"""
BSARec - Beyond Self-Attention based Sequential Recommendation

Implements BSARec model that augments Transformer encoder with frequency domain analysis.
Uses Fourier transform to capture high-frequency signals reflecting short-term user interests.

Paper: "An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention"
(AAAI 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
import math


class FrequencyLayer(nn.Module):
    """
    Frequency layer aligned with the official BSARec implementation.

    A short low-pass component is preserved up to cutoff ``c`` and the
    remaining high-frequency residual is scaled by a learnable beta term.
    """

    def __init__(self, hidden_size, dropout=0.1, c=3):
        super(FrequencyLayer, self).__init__()
        self.hidden_size = hidden_size
        self.out_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.c = max(1, int(c))
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        freq_bins = x_fft.shape[1]
        cutoff = min(freq_bins, max(1, self.c // 2 + 1))

        low_pass = x_fft.clone()
        if cutoff < freq_bins:
            low_pass[:, cutoff:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm="ortho")
        high_pass = x - low_pass
        sequence_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        hidden_states = self.out_dropout(sequence_fft)
        return self.norm(hidden_states + x)


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Self-Attention"""
    
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Expand mask for multi-head: (batch, 1, seq_len, seq_len) for broadcasting
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        
        output = self.fc_out(context)
        return output


class BSARecLayer(nn.Module):
    """
    BSARec Transformer layer with frequency enhancement.
    Combines self-attention with frequency domain analysis.
    """
    
    def __init__(self, hidden_size, num_heads=4, dropout=0.1, max_seq_len=50, alpha=0.5, c=3, inner_size=None):
        super(BSARecLayer, self).__init__()

        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.frequency = FrequencyLayer(hidden_size, dropout=dropout, c=c)
        ff_inner = int(inner_size) if inner_size is not None else hidden_size * 4
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_inner, hidden_size),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.alpha = float(alpha)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len) or None
        
        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        attn_out = self.attention(x, x, x, mask)
        gsp = self.attn_norm(x + self.attn_dropout(attn_out))
        dsp = self.frequency(x)
        x = self.alpha * dsp + (1.0 - self.alpha) * gsp
        ff_out = self.feed_forward(x)
        return self.ff_norm(x + ff_out)


class BSARec(SequentialRecommender):
    """
    BSARec: Beyond Self-Attention based Sequential Recommendation
    
    Combines Transformer with Fourier frequency analysis to capture
    both low-frequency trends and high-frequency short-term patterns.
    """
    
    input_type = "point"  # RecBole requirement
    
    def __init__(self, config, dataset):
        super(BSARec, self).__init__(config, dataset)
        
        self.n_items = dataset.item_num
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers'] if 'num_layers' in config else 2
        self.num_heads = config['num_heads'] if 'num_heads' in config else 4
        self.dropout_prob = config['hidden_dropout_prob'] if 'hidden_dropout_prob' in config else 0.1
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH'] if 'MAX_ITEM_LIST_LENGTH' in config else 50
        self.bsarec_alpha = config['bsarec_alpha'] if 'bsarec_alpha' in config else 0.5
        self.bsarec_c = config['bsarec_c'] if 'bsarec_c' in config else 3
        self.inner_size = config['inner_size'] if 'inner_size' in config else self.hidden_size * 4
        
        # Item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        
        # Projection to hidden size
        if self.embedding_size != self.hidden_size:
            self.embedding_proj = nn.Linear(self.embedding_size, self.hidden_size)
        else:
            self.embedding_proj = None
        
        # Positional embedding
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        
        # BSARec transformer layers with frequency enhancement
        self.transformer_layers = nn.ModuleList([
            BSARecLayer(
                self.hidden_size,
                self.num_heads,
                self.dropout_prob,
                self.max_seq_length,
                alpha=self.bsarec_alpha,
                c=self.bsarec_c,
                inner_size=self.inner_size,
            )
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_layer = nn.Linear(self.hidden_size, self.embedding_size)
        
        self.loss_type = config['loss_type'] if 'loss_type' in config else 'CE'
        self.to(self.device)
    
    def forward(self, item_seq, item_seq_len):
        """
        Args:
            item_seq: (batch_size, seq_len) long tensor
            item_seq_len: (batch_size,) long tensor
        
        Returns:
            seq_output: (batch_size, hidden_size) - representation for last item
        """
        batch_size, seq_len = item_seq.shape
        
        # Embed items
        item_emb = self.item_embedding(item_seq)  # (batch_size, seq_len, embedding_size)
        
        # Project to hidden size
        if self.embedding_proj is not None:
            x = self.embedding_proj(item_emb)  # (batch_size, seq_len, hidden_size)
        else:
            x = item_emb
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb
        
        # Create attention mask (causal + padding)
        mask = self._get_attention_mask(item_seq, item_seq_len)
        
        # Pass through BSARec layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Get last valid output
        last_hidden = x[range(batch_size), item_seq_len - 1]  # (batch_size, hidden_size)
        
        # Project to embedding space
        seq_output = self.output_layer(last_hidden)  # (batch_size, embedding_size)
        
        return seq_output
    
    def _get_attention_mask(self, item_seq, item_seq_len):
        """
        Create attention mask for causal attention (look-ahead prevention)
        and padding mask.
        
        Args:
            item_seq: (batch_size, seq_len)
            item_seq_len: (batch_size,)
        
        Returns:
            mask: (batch_size, seq_len, seq_len) or (batch_size, seq_len)
        """
        batch_size, seq_len = item_seq.shape
        
        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=item_seq.device))
        
        # Padding mask
        padding_mask = torch.arange(seq_len, device=item_seq.device).unsqueeze(0) < item_seq_len.unsqueeze(1)
        
        # Combine masks
        mask = causal_mask.unsqueeze(0) * padding_mask.unsqueeze(1)  # (batch_size, seq_len, seq_len)
        
        return mask
    
    def calculate_loss(self, interaction):
        """
        Calculate CE loss for next item prediction.
        
        Args:
            interaction: RecBole interaction dict
        
        Returns:
            loss: scalar
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        
        # CE loss with all items
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = F.cross_entropy(logits, pos_items)
        
        return loss
    
    def predict(self, interaction):
        """
        Predict scores for a specific item (used when full_sort_predict raises NotImplementedError).
        
        Args:
            interaction: RecBole interaction dict with ITEM_ID
        
        Returns:
            scores: (batch_size,) - score for the specific item
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        
        return scores
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items (used for full-sort evaluation).
        This avoids the expensive repeat_interleave fallback in RecBole.
        
        Args:
            interaction: RecBole interaction dict
        
        Returns:
            scores: (batch_size, n_items)
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len)
        
        # Compute logits for all items
        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        
        return scores
