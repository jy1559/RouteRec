"""
DIF-SR - Decoupled Side Information Fusion for Sequential Recommendation

Implements DIF-SR model that decouples side information fusion in the
attention layer instead of early integration in the input stage.

Paper: "Decoupled Side Information Fusion for Sequential Recommendation"
(SIGIR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
import math


class DIFMultiHeadAttention(nn.Module):
    """
    Decoupled Multi-Head Attention that separates attention calculation
    for item embeddings, position embeddings, and attribute embeddings.
    """
    
    def __init__(self, n_heads, hidden_size, attribute_hidden_sizes, 
                 dropout=0.1, fusion_type='sum', max_len=50):
        super(DIFMultiHeadAttention, self).__init__()
        
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads
        self.fusion_type = fusion_type
        self.max_len = max_len
        
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        
        # Item Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Position Q, K
        self.query_p = nn.Linear(hidden_size, hidden_size)
        self.key_p = nn.Linear(hidden_size, hidden_size)
        
        # Attribute Q, K for each attribute
        self.attribute_hidden_sizes = attribute_hidden_sizes
        self.n_attributes = len(attribute_hidden_sizes)
        self.attribute_head_dims = [size // n_heads for size in attribute_hidden_sizes]
        
        self.attribute_query_layers = nn.ModuleList([
            nn.Linear(attr_size, attr_size) 
            for attr_size in attribute_hidden_sizes
        ])
        self.attribute_key_layers = nn.ModuleList([
            nn.Linear(attr_size, attr_size) 
            for attr_size in attribute_hidden_sizes
        ])
        
        # Fusion layer
        if fusion_type == 'concat':
            total_len = max_len * (2 + self.n_attributes)
            self.fusion_layer = nn.Linear(total_len, max_len)
        elif fusion_type == 'gate':
            self.fusion_layer = nn.Linear(max_len, 1)
        
        # Output
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
    
    def transpose_for_scores(self, x, head_dim=None):
        """Reshape for multi-head attention."""
        if head_dim is None:
            head_dim = self.head_dim
        new_shape = x.size()[:-1] + (self.n_heads, head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
    
    def forward(self, item_hidden, attribute_embeds, position_embed, attention_mask):
        """
        Args:
            item_hidden: (batch, seq_len, hidden_size) - item hidden states
            attribute_embeds: list of (batch, seq_len, attr_hidden_size) - attribute embeddings
            position_embed: (batch, seq_len, hidden_size) - position embeddings
            attention_mask: (batch, 1, 1, seq_len) - attention mask
        
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = item_hidden.shape
        
        # Item attention scores
        item_Q = self.transpose_for_scores(self.query(item_hidden))
        item_K = self.transpose_for_scores(self.key(item_hidden))
        item_V = self.transpose_for_scores(self.value(item_hidden))
        item_scores = torch.matmul(item_Q, item_K.transpose(-1, -2))  # (batch, heads, seq, seq)
        
        # Position attention scores
        pos_Q = self.transpose_for_scores(self.query_p(position_embed))
        pos_K = self.transpose_for_scores(self.key_p(position_embed))
        pos_scores = torch.matmul(pos_Q, pos_K.transpose(-1, -2))  # (batch, heads, seq, seq)
        
        # Attribute attention scores
        attribute_scores_list = []
        for i, (attr_embed, attr_Q_layer, attr_K_layer) in enumerate(
            zip(attribute_embeds, self.attribute_query_layers, self.attribute_key_layers)
        ):
            # Handle possible extra dimension from FeatureSeqEmbLayer
            if attr_embed.dim() == 4:
                attr_embed = attr_embed.squeeze(-2)  # (batch, seq_len, attr_size)
            
            attr_Q = self.transpose_for_scores(attr_Q_layer(attr_embed), self.attribute_head_dims[i])
            attr_K = self.transpose_for_scores(attr_K_layer(attr_embed), self.attribute_head_dims[i])
            attr_scores = torch.matmul(attr_Q, attr_K.transpose(-1, -2))
            attribute_scores_list.append(attr_scores)
        
        # Fuse attention scores
        if self.fusion_type == 'sum':
            attention_scores = item_scores + pos_scores
            for attr_scores in attribute_scores_list:
                attention_scores = attention_scores + attr_scores
        elif self.fusion_type == 'concat':
            # Concatenate all scores along last dimension
            all_scores = [item_scores, pos_scores] + attribute_scores_list
            all_scores = torch.cat([s.unsqueeze(-1) for s in all_scores], dim=-1)
            # (batch, heads, seq, seq, n_sources)
            all_scores = all_scores.view(batch_size, self.n_heads, seq_len, -1)
            attention_scores = self.fusion_layer(all_scores)
        elif self.fusion_type == 'gate':
            # Gate-based fusion
            all_scores = torch.stack([item_scores, pos_scores] + attribute_scores_list, dim=-1)
            # (batch, heads, seq, seq, n_sources)
            gate_input = all_scores.transpose(-1, -2)  # (batch, heads, seq, n_sources, seq)
            gates = torch.sigmoid(self.fusion_layer(gate_input))  # (batch, heads, seq, n_sources, 1)
            gates = gates.transpose(-1, -2)  # (batch, heads, seq, 1, n_sources)
            attention_scores = (all_scores * gates).sum(dim=-1)
        
        # Scale and mask
        attention_scores = attention_scores / self.scale
        attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, item_V)  # (batch, heads, seq, head_dim)
        
        # Reshape back
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, -1)  # (batch, seq, hidden_size)
        
        # Output projection
        output = self.dense(context)
        output = self.dropout(output)
        output = self.layernorm(output + item_hidden)
        
        return output


class DIFTransformerLayer(nn.Module):
    """
    DIF Transformer layer with decoupled attention and FFN.
    """
    
    def __init__(self, n_heads, hidden_size, attribute_hidden_sizes,
                 inner_size=256, dropout=0.1, fusion_type='sum', max_len=50):
        super(DIFTransformerLayer, self).__init__()
        
        self.dif_attention = DIFMultiHeadAttention(
            n_heads, hidden_size, attribute_hidden_sizes, dropout, fusion_type, max_len
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.layernorm = nn.LayerNorm(hidden_size)
    
    def forward(self, item_hidden, attribute_embeds, position_embed, attention_mask):
        attention_output = self.dif_attention(
            item_hidden, attribute_embeds, position_embed, attention_mask
        )
        ffn_output = self.ffn(attention_output)
        output = self.layernorm(attention_output + ffn_output)
        return output


class DIFSR(SequentialRecommender):
    """
    DIF-SR: Decoupled Side Information Fusion for Sequential Recommendation
    
    Moves side information from input to attention layer and decouples
    the attention calculation of various side information and item representation.
    """
    
    input_type = "point"  # RecBole requirement
    
    def __init__(self, config, dataset):
        super(DIFSR, self).__init__(config, dataset)
        
        self.n_items = dataset.item_num
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers'] if 'num_layers' in config else 2
        self.num_heads = config['num_heads'] if 'num_heads' in config else 4
        self.inner_size = config['inner_size'] if 'inner_size' in config else self.hidden_size * 4
        self.dropout_prob = config['hidden_dropout_prob'] if 'hidden_dropout_prob' in config else 0.1
        self.fusion_type = config['fusion_type'] if 'fusion_type' in config else 'sum'
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH'] if 'MAX_ITEM_LIST_LENGTH' in config else 50
        
        # Selected features for side information
        # Default to 'item_category' if available, otherwise empty
        self.selected_features = config['selected_features'] if 'selected_features' in config else []
        if isinstance(self.selected_features, str):
            self.selected_features = [self.selected_features]
        self.selected_features = [str(x).strip() for x in self.selected_features if str(x).strip()]

        # If not explicitly set, infer a practical default from item metadata.
        if not self.selected_features:
            item_feat = getattr(dataset, "item_feat", None)
            item_cols = set(getattr(item_feat, "columns", []) or [])
            if "category" in item_cols:
                self.selected_features = ["category"]
        
        # Attribute hidden sizes (same as hidden_size by default)
        attr_hidden_size = config['attribute_hidden_size'] if 'attribute_hidden_size' in config else self.hidden_size
        self.attribute_hidden_sizes = [attr_hidden_size] * len(self.selected_features)
        
        # Item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        
        # Projection to hidden size
        if self.embedding_size != self.hidden_size:
            self.embedding_proj = nn.Linear(self.embedding_size, self.hidden_size)
        else:
            self.embedding_proj = None
        
        # Position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        
        # Attribute embeddings
        self.attribute_embeddings = nn.ModuleDict()
        self.n_attributes = {}
        
        # Store item feature lookup table (item_id -> attribute_id)
        self.item_to_attr = {}
        
        valid_features = []
        for i, feat in enumerate(self.selected_features):
            if feat in dataset.field2token_id:
                n_values = len(dataset.field2token_id[feat])
                self.n_attributes[feat] = n_values
                self.attribute_embeddings[feat] = nn.Embedding(
                    n_values, self.attribute_hidden_sizes[i], padding_idx=0
                )
                valid_features.append(feat)
                
                # Build item_id -> attribute lookup table from item_feat
                if dataset.item_feat is not None and feat in dataset.item_feat.columns:
                    item_ids = dataset.item_feat['item_id'].numpy()
                    feat_ids = dataset.item_feat[feat].numpy()
                    # Create lookup tensor (item_id -> feat_id)
                    lookup = torch.zeros(self.n_items, dtype=torch.long)
                    for item_id, feat_id in zip(item_ids, feat_ids):
                        if item_id < self.n_items:
                            lookup[item_id] = feat_id
                    self.register_buffer(f'{feat}_lookup', lookup)
        self.selected_features = valid_features
        self.attribute_hidden_sizes = [attr_hidden_size] * len(self.selected_features)
        
        # DIF Transformer layers
        self.transformer_layers = nn.ModuleList([
            DIFTransformerLayer(
                self.num_heads, self.hidden_size, self.attribute_hidden_sizes,
                self.inner_size, self.dropout_prob, self.fusion_type, self.max_seq_length
            )
            for _ in range(self.num_layers)
        ])
        
        # Layer normalization
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Output projection
        if self.hidden_size != self.embedding_size:
            self.output_proj = nn.Linear(self.hidden_size, self.embedding_size)
        else:
            self.output_proj = None
        
        # Attribute predictor (auxiliary task)
        self.use_attribute_predictor = config['use_attribute_predictor'] if 'use_attribute_predictor' in config else False
        if self.use_attribute_predictor and len(self.selected_features) > 0:
            self.attribute_predictors = nn.ModuleDict()
            for feat in self.selected_features:
                if feat in self.n_attributes:
                    self.attribute_predictors[feat] = nn.Linear(
                        self.hidden_size, self.n_attributes[feat]
                    )
        
        self.lambda_attr = config['lambda_attr'] if 'lambda_attr' in config else 0.1
        self.loss_type = config['loss_type'] if 'loss_type' in config else 'CE'
        self.to(self.device)
    
    def get_attention_mask(self, item_seq):
        """Generate causal attention mask."""
        batch_size, seq_len = item_seq.shape
        
        # Padding mask
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=item_seq.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine masks
        extended_attention_mask = extended_attention_mask * causal_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def get_attribute_embeddings(self, item_seq, interaction):
        """Get attribute embeddings for the item sequence using item_feat lookup."""
        attribute_embeds = []
        
        for i, feat in enumerate(self.selected_features):
            if feat in self.attribute_embeddings:
                # Use lookup table to get attribute ids from item_seq
                lookup_name = f'{feat}_lookup'
                if hasattr(self, lookup_name):
                    lookup = getattr(self, lookup_name)
                    # item_seq: (batch, seq_len) -> feat_seq: (batch, seq_len)
                    feat_seq = lookup[item_seq]  # Use item_id to lookup attribute_id
                    attr_emb = self.attribute_embeddings[feat](feat_seq)
                else:
                    # No lookup table, use zero embeddings as fallback
                    batch_size, seq_len = item_seq.shape
                    attr_emb = torch.zeros(
                        batch_size, seq_len, self.attribute_hidden_sizes[i],
                        device=item_seq.device
                    )
                attribute_embeds.append(attr_emb)
        
        return attribute_embeds
    
    def forward(self, item_seq, item_seq_len, interaction=None):
        """
        Args:
            item_seq: (batch_size, seq_len) long tensor
            item_seq_len: (batch_size,) long tensor
            interaction: RecBole interaction dict for attribute access
        
        Returns:
            seq_output: (batch_size, embedding_size)
        """
        batch_size, seq_len = item_seq.shape
        
        # Item embeddings
        item_emb = self.item_embedding(item_seq)
        if self.embedding_proj is not None:
            item_hidden = self.embedding_proj(item_emb)
        else:
            item_hidden = item_emb
        
        item_hidden = self.layernorm(item_hidden)
        item_hidden = self.dropout(item_hidden)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        position_embed = self.position_embedding(positions)
        
        # Attribute embeddings
        if interaction is not None:
            attribute_embeds = self.get_attribute_embeddings(item_seq, interaction)
        else:
            attribute_embeds = [
                torch.zeros(batch_size, seq_len, attr_size, device=item_seq.device)
                for attr_size in self.attribute_hidden_sizes
            ]
        
        # Attention mask
        attention_mask = self.get_attention_mask(item_seq)
        
        # Pass through DIF transformer layers
        hidden = item_hidden
        for layer in self.transformer_layers:
            hidden = layer(hidden, attribute_embeds, position_embed, attention_mask)
        
        # Get last valid output
        last_hidden = hidden[range(batch_size), item_seq_len - 1]
        
        # Project to embedding space
        if self.output_proj is not None:
            seq_output = self.output_proj(last_hidden)
        else:
            seq_output = last_hidden
        
        return seq_output
    
    def calculate_loss(self, interaction):
        """
        Calculate CE loss for next item prediction, optionally with attribute prediction loss.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len, interaction)
        
        # CE loss
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = F.cross_entropy(logits, pos_items)
        
        # Attribute prediction loss (auxiliary task)
        if self.use_attribute_predictor and hasattr(self, 'attribute_predictors'):
            for feat in self.selected_features:
                if feat in self.attribute_predictors:
                    lookup_name = f'{feat}_lookup'
                    if hasattr(self, lookup_name):
                        lookup = getattr(self, lookup_name)
                        target_feat = lookup[pos_items]
                        attr_logits = self.attribute_predictors[feat](seq_output)
                        attr_loss = F.cross_entropy(attr_logits, target_feat, ignore_index=0)
                        loss = loss + self.lambda_attr * attr_loss
        
        return loss
    
    def predict(self, interaction):
        """
        Predict score for a specific item.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len, interaction)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        
        return scores
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items (full-sort evaluation).
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len, interaction)
        
        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        
        return scores
