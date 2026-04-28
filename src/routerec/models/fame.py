"""
FAME - Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation

Implements FAME model that uses multi-head prediction with MoE (Mixture of Experts)
in the self-attention layer to capture multi-faceted user preferences.

Paper: "Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation"
(WSDM 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
import math


class MoEMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Mixture-of-Experts (MoE) Query Generation.
    
    Each head has N experts for query generation. A router network determines
    the importance weight of each expert and aggregates their outputs.
    """
    
    def __init__(self, hidden_size, num_heads=4, num_experts=4, dropout=0.1):
        super(MoEMultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # MoE: Multiple query experts per head
        # W_Q^(h)(n) for each head h and expert n
        self.query_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_size, self.head_dim, bias=False)
                for _ in range(num_experts)
            ])
            for _ in range(num_heads)
        ])
        
        # Shared key and value projections per head
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Router network for each head: determines expert importance
        # W_exp^(h): (num_experts * head_dim) -> num_experts
        self.routers = nn.ModuleList([
            nn.Linear(num_experts * self.head_dim, num_experts, bias=False)
            for _ in range(num_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len, seq_len) or None
        
        Returns:
            output: (batch_size, seq_len, num_heads, head_dim) - per-head outputs
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute shared key and value for all heads
        K = self.key(x)  # (batch, seq_len, hidden_size)
        V = self.value(x)  # (batch, seq_len, hidden_size)
        
        # Reshape K, V for multi-head
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch, seq, heads, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Process each head with MoE
        head_outputs = []
        
        for h in range(self.num_heads):
            k_h = K[:, :, h, :]  # (batch, seq_len, head_dim)
            v_h = V[:, :, h, :]
            
            # Compute expert query representations
            expert_representations = []  # List of (batch, seq_len, head_dim)
            
            for n in range(self.num_experts):
                q_expert = self.query_experts[h][n](x)  # (batch, seq_len, head_dim)
                
                # Compute attention for this expert
                # scores: (batch, seq_len, seq_len)
                scores = torch.matmul(q_expert, k_h.transpose(-2, -1)) / self.scale
                
                if mask is not None:
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
                    scores = scores.masked_fill(mask == 0, float('-inf'))
                
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                # Expert representation: f_i(n)^(h)
                expert_out = torch.matmul(attn_weights, v_h)  # (batch, seq_len, head_dim)
                expert_representations.append(expert_out)
            
            # Stack expert outputs: (batch, seq_len, num_experts, head_dim)
            expert_stack = torch.stack(expert_representations, dim=2)
            
            # Router: compute importance weights for each expert
            # Concatenate expert representations for router input
            router_input = expert_stack.view(batch_size, seq_len, -1)  # (batch, seq_len, num_experts * head_dim)
            expert_weights = self.routers[h](router_input)  # (batch, seq_len, num_experts)
            expert_weights = F.softmax(expert_weights, dim=-1)  # β_i(n)^(h)
            
            # Aggregate expert outputs: f_i^(h) = sum_n β_i(n)^(h) * f_i(n)^(h)
            # expert_weights: (batch, seq_len, num_experts, 1)
            expert_weights = expert_weights.unsqueeze(-1)
            head_out = (expert_stack * expert_weights).sum(dim=2)  # (batch, seq_len, head_dim)
            
            head_outputs.append(head_out)
        
        # Stack head outputs: (batch, seq_len, num_heads, head_dim)
        output = torch.stack(head_outputs, dim=2)
        
        return output


class FAMELayer(nn.Module):
    """
    FAME Transformer layer with MoE attention.
    """
    
    def __init__(self, hidden_size, num_heads=4, num_experts=4, dropout=0.1):
        super(FAMELayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # MoE Multi-Head Attention
        self.moe_attention = MoEMultiHeadAttention(hidden_size, num_heads, num_experts, dropout)
        
        # Output projection to combine heads (for intermediate layers)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
        # Shared FFN' for each head (operates on head_dim)
        self.head_ffn = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.head_dim, self.head_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, return_per_head=False):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len, seq_len) or None
            return_per_head: if True, return per-head outputs for the final layer
        
        Returns:
            If return_per_head:
                output: (batch_size, seq_len, num_heads, head_dim)
            Else:
                output: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # MoE attention: (batch, seq_len, num_heads, head_dim)
        attn_out = self.moe_attention(x, mask)
        
        if return_per_head:
            # For final layer: apply FFN' per head and return per-head outputs
            # attn_out: (batch, seq_len, num_heads, head_dim)
            
            # Residual connection with input split into heads
            x_heads = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
            attn_out = x_heads + self.dropout(attn_out)
            
            # Apply shared FFN' to each head
            ffn_out = self.head_ffn(attn_out)  # (batch, seq_len, num_heads, head_dim)
            output = self.norm2(attn_out + ffn_out)
            
            return output
        else:
            # For intermediate layers: combine heads and return full hidden
            combined = attn_out.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
            combined = self.fc_out(combined)
            output = self.norm1(x + self.dropout(combined))
            
            return output


class FAME(SequentialRecommender):
    """
    FAME: Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation
    
    Uses multi-head prediction with MoE attention to capture multi-faceted
    user preferences. Each head captures a distinct facet of items, and
    a gating mechanism dynamically determines head importance.
    """
    
    input_type = "point"  # RecBole requirement
    
    def __init__(self, config, dataset):
        super(FAME, self).__init__(config, dataset)
        
        self.n_items = dataset.item_num
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers'] if 'num_layers' in config else 2
        self.num_heads = config['num_heads'] if 'num_heads' in config else 4
        self.num_experts = config['num_experts'] if 'num_experts' in config else 4
        self.fame_moe_last_k_layers = config['fame_moe_last_k_layers'] if 'fame_moe_last_k_layers' in config else 1
        self.dropout_prob = config['hidden_dropout_prob'] if 'hidden_dropout_prob' in config else 0.1
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH'] if 'MAX_ITEM_LIST_LENGTH' in config else 50
        self.fame_moe_last_k_layers = max(1, min(self.num_layers, int(self.fame_moe_last_k_layers)))
        
        self.head_dim = self.hidden_size // self.num_heads
        
        # Item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        
        # Projection to hidden size
        if self.embedding_size != self.hidden_size:
            self.embedding_proj = nn.Linear(self.embedding_size, self.hidden_size)
        else:
            self.embedding_proj = None
        
        # Positional embedding
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        
        # Use MoE attention for the final k layers. The default k=1 preserves
        # the original FAME behavior while allowing step2 recovery sweeps to
        # probe deeper MoE usage when needed.
        self.transformer_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i < self.num_layers - self.fame_moe_last_k_layers:
                self.transformer_layers.append(
                    StandardTransformerLayer(self.hidden_size, self.num_heads, self.dropout_prob)
                )
            else:
                self.transformer_layers.append(
                    FAMELayer(self.hidden_size, self.num_heads, self.num_experts, self.dropout_prob)
                )
        
        # Facet projection: W_f^(h) for each head (d -> d')
        # Maps item embeddings to head-specific sub-embeddings
        self.facet_projections = nn.ModuleList([
            nn.Linear(self.embedding_size, self.head_dim, bias=False)
            for _ in range(self.num_heads)
        ])
        
        # Head gating mechanism: determines importance of each head
        # Input: concatenated head outputs (d = num_heads * head_dim)
        # Output: importance scores for each head (num_heads)
        self.head_gate = nn.Linear(self.hidden_size, self.num_heads)
        
        self.loss_type = config['loss_type'] if 'loss_type' in config else 'CE'
        self.to(self.device)
    
    def forward(self, item_seq, item_seq_len):
        """
        Args:
            item_seq: (batch_size, seq_len) long tensor
            item_seq_len: (batch_size,) long tensor
        
        Returns:
            seq_output: (batch_size, embedding_size) - representation for prediction
        """
        batch_size, seq_len = item_seq.shape
        
        # Embed items
        item_emb = self.item_embedding(item_seq)  # (batch_size, seq_len, embedding_size)
        
        # Project to hidden size
        if self.embedding_proj is not None:
            x = self.embedding_proj(item_emb)
        else:
            x = item_emb
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb
        
        # Create attention mask
        mask = self._get_attention_mask(item_seq, item_seq_len)
        
        # Pass through transformer layers
        for i, layer in enumerate(self.transformer_layers):
            if i < self.num_layers - self.fame_moe_last_k_layers:
                x = layer(x, mask)
            else:
                return_heads = i == self.num_layers - 1
                x = layer(x, mask, return_per_head=return_heads)
        
        # Get last valid position per-head outputs: F_t^(h)
        # x: (batch, seq_len, num_heads, head_dim)
        last_indices = (item_seq_len - 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        last_indices = last_indices.expand(-1, 1, self.num_heads, self.head_dim)
        last_hidden = x.gather(1, last_indices).squeeze(1)  # (batch, num_heads, head_dim)
        
        # Compute head importance: g = softmax(W_g * concat(F_t^(1), ..., F_t^(H)) + b_g)
        concat_heads = last_hidden.view(batch_size, -1)  # (batch, hidden_size)
        head_gate_scores = self.head_gate(concat_heads)  # (batch, num_heads)
        head_weights = F.softmax(head_gate_scores, dim=-1)  # g~
        
        return last_hidden, head_weights
    
    def _get_attention_mask(self, item_seq, item_seq_len):
        """Create causal attention mask."""
        batch_size, seq_len = item_seq.shape
        
        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=item_seq.device))
        
        # Padding mask
        padding_mask = torch.arange(seq_len, device=item_seq.device).unsqueeze(0) < item_seq_len.unsqueeze(1)
        
        # Combine masks
        mask = causal_mask.unsqueeze(0) * padding_mask.unsqueeze(1)
        
        return mask
    
    def calculate_loss(self, interaction):
        """
        Calculate CE loss using multi-head prediction.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Get per-head outputs and head weights
        last_hidden, head_weights = self.forward(item_seq, item_seq_len)
        # last_hidden: (batch, num_heads, head_dim)
        # head_weights: (batch, num_heads)
        
        batch_size = item_seq.size(0)
        
        # Get target item embedding
        target_emb = self.item_embedding(pos_items)  # (batch, embedding_size)
        
        # Compute per-head scores for target item
        # P^(h)(v) = x_v^(h)^T * F_t^(h)
        head_scores = []
        for h in range(self.num_heads):
            # Project target to head subspace
            target_sub = self.facet_projections[h](target_emb)  # (batch, head_dim)
            head_out = last_hidden[:, h, :]  # (batch, head_dim)
            score = (target_sub * head_out).sum(dim=-1)  # (batch,)
            head_scores.append(score)
        
        head_scores = torch.stack(head_scores, dim=-1)  # (batch, num_heads)
        
        # Weighted sum of head scores
        pos_score = (head_weights * head_scores).sum(dim=-1)  # (batch,)
        
        # Full softmax loss: need scores for all items
        all_item_emb = self.item_embedding.weight  # (n_items, embedding_size)
        
        # Compute per-head logits for all items
        all_logits = torch.zeros(batch_size, self.n_items, device=item_seq.device)
        
        for h in range(self.num_heads):
            # Project all items to head subspace
            all_item_sub = self.facet_projections[h](all_item_emb)  # (n_items, head_dim)
            head_out = last_hidden[:, h, :]  # (batch, head_dim)
            
            # Compute scores: (batch, n_items)
            head_logits = torch.matmul(head_out, all_item_sub.T)
            
            # Weight by head importance
            all_logits += head_weights[:, h:h+1] * head_logits
        
        loss = F.cross_entropy(all_logits, pos_items)
        
        return loss
    
    def predict(self, interaction):
        """
        Predict score for a specific item.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        last_hidden, head_weights = self.forward(item_seq, item_seq_len)
        
        # Get test item embedding
        test_emb = self.item_embedding(test_item)  # (batch, embedding_size)
        
        # Compute per-head scores
        head_scores = []
        for h in range(self.num_heads):
            test_sub = self.facet_projections[h](test_emb)
            head_out = last_hidden[:, h, :]
            score = (test_sub * head_out).sum(dim=-1)
            head_scores.append(score)
        
        head_scores = torch.stack(head_scores, dim=-1)
        scores = (head_weights * head_scores).sum(dim=-1)
        
        return scores
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items (full-sort evaluation).
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        last_hidden, head_weights = self.forward(item_seq, item_seq_len)
        
        batch_size = item_seq.size(0)
        all_item_emb = self.item_embedding.weight  # (n_items, embedding_size)
        
        # Compute weighted logits across heads
        all_logits = torch.zeros(batch_size, self.n_items, device=item_seq.device)
        
        for h in range(self.num_heads):
            all_item_sub = self.facet_projections[h](all_item_emb)  # (n_items, head_dim)
            head_out = last_hidden[:, h, :]  # (batch, head_dim)
            head_logits = torch.matmul(head_out, all_item_sub.T)  # (batch, n_items)
            all_logits += head_weights[:, h:h+1] * head_logits
        
        return all_logits


class StandardTransformerLayer(nn.Module):
    """Standard transformer layer for intermediate layers."""
    
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(StandardTransformerLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        attn_out = self.fc_out(context)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x
