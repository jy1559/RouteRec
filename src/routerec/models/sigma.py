"""
SIGMA - Selective Gated Mamba for Sequential Recommendation

Implements SIGMA model that uses Partially Flipped Mamba (PF-Mamba) with
Dense Selective Gate (DS Gate) and Feature Extract GRU (FE-GRU).

Paper: "SIGMA: Selective Gated Mamba for Sequential Recommendation"
(arXiv 2408.11451)

Note: This is a simplified implementation without mamba-ssm package.
For optimal performance, install mamba-ssm and use the official selective scan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
import math


class MambaBlock(nn.Module):
    """
    Mamba block with selective state space model.
    Uses sequential scan for SSM computation (can be replaced with mamba-ssm for speed).
    """
    
    def __init__(self, hidden_size, state_size=16, conv_kernel=4, expand=2, dropout=0.1):
        super(MambaBlock, self).__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.expand = expand
        self.inner_size = hidden_size * expand
        
        # Input projection
        self.in_proj = nn.Linear(hidden_size, self.inner_size * 2, bias=False)
        
        # Conv1d for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_size,
            out_channels=self.inner_size,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.inner_size
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.inner_size, state_size + state_size + self.inner_size, bias=False)
        
        # A parameter (diagonal, log-space for stability)
        A = torch.arange(1, state_size + 1, dtype=torch.float32).unsqueeze(0).expand(self.inner_size, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.inner_size))
        
        # dt projection
        self.dt_proj = nn.Linear(self.inner_size, self.inner_size, bias=True)
        
        # Initialize dt_proj for stability
        with torch.no_grad():
            dt_init_std = 0.02
            nn.init.normal_(self.dt_proj.weight, std=dt_init_std)
            dt = torch.exp(torch.rand(self.inner_size) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_proj.bias.copy_(dt)
        
        # Output projection
        self.out_proj = nn.Linear(self.inner_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def ssm_scan(self, x, dt, A, B, C):
        """
        Efficient SSM approximation using exponential moving average.
        
        This is a fast approximation that captures the essence of SSM:
        - Exponential decay of past information
        - Selective input gating via B
        - Selective output via C
        
        For exact computation, use mamba-ssm package.
        
        Args:
            x: (batch, seq_len, inner_size) - input
            dt: (batch, seq_len, inner_size) - delta time
            A: (inner_size, state_size) - state matrix (negative)
            B: (batch, seq_len, state_size) - input matrix
            C: (batch, seq_len, state_size) - output matrix
        
        Returns:
            y: (batch, seq_len, inner_size)
        """
        batch, seq_len, inner_size = x.shape
        
        # Compute decay factor from dt and A
        # Use sigmoid to ensure stability (0 < alpha < 1)
        alpha = torch.sigmoid(dt * 0.1)  # (batch, seq_len, inner_size)
        
        # Input gating via B (average over state_size)
        B_gate = torch.sigmoid(B.mean(dim=-1, keepdim=True))  # (batch, seq_len, 1)
        gated_input = (1 - alpha) * B_gate * x  # (batch, seq_len, inner_size)
        
        # EMA-style computation using conv1d for efficiency
        # This approximates: h[t] = alpha * h[t-1] + (1-alpha) * input[t]
        # Using exponentially weighted moving average
        
        # Simple approach: weighted cumsum with decaying weights
        # For speed, use a simple EMA approximation
        y = torch.zeros_like(x)
        h = torch.zeros(batch, inner_size, device=x.device, dtype=x.dtype)
        
        # Vectorized EMA using cumsum trick (stable version)
        # Clamp alpha for numerical stability
        alpha_clamped = alpha.clamp(0.01, 0.99)
        
        for t in range(seq_len):
            h = alpha_clamped[:, t] * h + gated_input[:, t]
            y[:, t] = h
        
        # Output gating via C
        C_gate = torch.sigmoid(C.mean(dim=-1, keepdim=True))  # (batch, seq_len, 1)
        y = y * C_gate
        
        return y
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection with gate
        xz = self.in_proj(x)  # (batch, seq_len, inner_size * 2)
        x_proj, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, inner_size)
        
        # Conv1d for local context
        x_conv = x_proj.transpose(1, 2)  # (batch, inner_size, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal conv
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, inner_size)
        x_conv = F.silu(x_conv)
        
        # SSM parameters from input
        x_ssm = self.x_proj(x_conv)  # (batch, seq_len, state_size + state_size + inner_size)
        B = x_ssm[:, :, :self.state_size]  # (batch, seq_len, state_size)
        C = x_ssm[:, :, self.state_size:2*self.state_size]  # (batch, seq_len, state_size)
        dt_input = x_ssm[:, :, 2*self.state_size:]  # (batch, seq_len, inner_size)
        
        # Compute dt
        dt = F.softplus(self.dt_proj(dt_input))  # (batch, seq_len, inner_size)
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # (inner_size, state_size)
        
        # SSM scan
        y = self.ssm_scan(x_conv, dt, A, B, C)  # (batch, seq_len, inner_size)
        
        # Add skip connection with D
        y = y + self.D * x_conv
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)
        
        return output


class DenseSelectiveGate(nn.Module):
    """
    Dense Selective Gate (DS Gate) for allocating weights between
    forward and backward Mamba directions.
    """
    
    def __init__(self, hidden_size, dropout=0.1):
        super(DenseSelectiveGate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        self.delta = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            gate: (batch, seq_len, hidden_size) - gate weights
        """
        # Dense + Conv1d
        g = self.dense(x)  # (batch, seq_len, hidden_size)
        g = g.transpose(1, 2)  # (batch, hidden_size, seq_len)
        g = self.conv1d(g)
        g = g.transpose(1, 2)  # (batch, seq_len, hidden_size)
        
        # Forget gate + SiLU
        delta = self.delta(g)
        forget_gate = torch.sigmoid(delta)
        silu_gate = F.silu(delta)
        
        gate = silu_gate + forget_gate
        return self.dropout(gate)


class FeatureExtractGRU(nn.Module):
    """
    Feature Extract (FE) for capturing short-term dependencies.
    Uses stacked Conv1d layers for efficiency.
    """
    
    def __init__(self, hidden_size, dropout=0.1):
        super(FeatureExtractGRU, self).__init__()
        # Efficient Conv1d stack
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        c = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        c = F.silu(self.conv1(c))
        c = self.conv2(c)
        c = c.transpose(1, 2)  # (batch, seq_len, hidden_size)
        
        output = self.norm(c + x)  # Residual connection
        output = self.dropout(output)
        
        return output


class GMambaBlock(nn.Module):
    """
    G-Mamba Block with Bidirectional Mamba (PF-Mamba) + Conv-based feature extraction.
    Uses partially flipped sequence for bidirectional context modeling.
    """
    
    def __init__(self, hidden_size, state_size=16, conv_kernel=4, expand=2, 
                 remaining_ratio=0.5, dropout=0.1):
        super(GMambaBlock, self).__init__()
        self.hidden_size = hidden_size
        self.remaining_ratio = remaining_ratio
        
        # Bidirectional Mamba (PF-Mamba)
        self.mamba_forward = MambaBlock(hidden_size, state_size, conv_kernel, expand, dropout)
        self.mamba_backward = MambaBlock(hidden_size, state_size, conv_kernel, expand, dropout)
        
        # DS Gate for direction weighting
        self.ds_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Conv-based feature extraction (efficient FE-GRU replacement)
        self.fe_conv = FeatureExtractGRU(hidden_size, dropout)
        
        # Mixing layer
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.mix_linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def partial_flip(self, x):
        """Partially flip sequence: flip first part, keep last part."""
        batch, seq_len, hidden = x.shape
        r = max(1, int(seq_len * self.remaining_ratio))
        n = seq_len - r
        
        if n <= 0:
            return x
        
        # Flip first n, keep last r
        return torch.cat([x[:, :n].flip(dims=[1]), x[:, n:]], dim=1)
    
    def forward(self, x, seq_len):
        """
        Args:
            x: (batch, seq_len, hidden_size) - embedded sequence
            seq_len: (batch,) - actual sequence lengths
        
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        # DS Gate for direction weighting
        gate_weights = self.ds_gate(x)  # (batch, seq_len, hidden_size)
        
        # Forward Mamba
        m_forward = self.mamba_forward(x)
        
        # Backward Mamba (on flipped sequence)
        x_flipped = self.partial_flip(x)
        m_backward = self.mamba_backward(x_flipped)
        # Restore original order after backward pass
        m_backward = self.partial_flip(m_backward)
        
        # Combine with gating
        m_combined = gate_weights * m_forward + (1 - gate_weights) * m_backward
        
        # Conv for short-term patterns
        f_out = self.fe_conv(x)
        
        # Gated mixing between Mamba and Conv
        gate = self.gate(torch.cat([m_combined, f_out], dim=-1))
        z = gate * m_combined + (1 - gate) * f_out
        
        z = self.mix_linear(z)
        z = self.norm(z + x)  # Residual
        z = self.dropout(z)
        
        return z


class SIGMA(SequentialRecommender):
    """
    SIGMA: Selective Gated Mamba for Sequential Recommendation
    
    Uses PF-Mamba (Partially Flipped Mamba) with DS Gate (Dense Selective Gate)
    and FE-GRU (Feature Extract GRU) to handle context modeling and short
    sequence modeling challenges.
    """
    
    input_type = "point"  # RecBole requirement
    
    def __init__(self, config, dataset):
        super(SIGMA, self).__init__(config, dataset)
        
        self.n_items = dataset.item_num
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers'] if 'num_layers' in config else 1
        self.state_size = config['state_size'] if 'state_size' in config else 16
        self.conv_kernel = config['conv_kernel'] if 'conv_kernel' in config else 4
        self.expand = config['expand'] if 'expand' in config else 2
        self.remaining_ratio = config['remaining_ratio'] if 'remaining_ratio' in config else 0.2
        self.dropout_prob = config['hidden_dropout_prob'] if 'hidden_dropout_prob' in config else 0.1
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH'] if 'MAX_ITEM_LIST_LENGTH' in config else 50
        
        # Item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        
        # Projection to hidden size
        if self.embedding_size != self.hidden_size:
            self.embedding_proj = nn.Linear(self.embedding_size, self.hidden_size)
        else:
            self.embedding_proj = None
        
        # Positional embedding
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        
        # G-Mamba blocks
        self.gmamba_layers = nn.ModuleList([
            GMambaBlock(
                self.hidden_size, self.state_size, self.conv_kernel, 
                self.expand, self.remaining_ratio, self.dropout_prob
            )
            for _ in range(self.num_layers)
        ])
        
        # PFFN (Position-wise Feed-Forward Network)
        self.pffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(self.dropout_prob)
        )
        
        # Layer normalization
        self.layernorm = nn.LayerNorm(self.hidden_size)
        
        # Output projection
        if self.hidden_size != self.embedding_size:
            self.output_proj = nn.Linear(self.hidden_size, self.embedding_size)
        else:
            self.output_proj = None
        
        self.loss_type = config['loss_type'] if 'loss_type' in config else 'CE'
        self.to(self.device)
    
    def forward(self, item_seq, item_seq_len):
        """
        Args:
            item_seq: (batch_size, seq_len) long tensor
            item_seq_len: (batch_size,) long tensor
        
        Returns:
            seq_output: (batch_size, embedding_size) - representation for last item
        """
        batch_size, seq_len = item_seq.shape
        
        # Embed items
        item_emb = self.item_embedding(item_seq)  # (batch, seq_len, embedding_size)
        
        # Project to hidden size
        if self.embedding_proj is not None:
            x = self.embedding_proj(item_emb)
        else:
            x = item_emb
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb
        
        # Pass through G-Mamba layers
        for layer in self.gmamba_layers:
            x = layer(x, item_seq_len)
        
        # PFFN with residual
        residual = x
        x = self.pffn(x)
        x = self.layernorm(x + residual)
        
        # Get last valid output
        last_hidden = x[range(batch_size), item_seq_len - 1]  # (batch, hidden_size)
        
        # Project to embedding space
        if self.output_proj is not None:
            seq_output = self.output_proj(last_hidden)
        else:
            seq_output = last_hidden
        
        return seq_output
    
    def calculate_loss(self, interaction):
        """
        Calculate CE loss for next item prediction.
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
        Predict score for a specific item.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        
        return scores
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items (full-sort evaluation).
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len)
        
        # Compute logits for all items
        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        
        return scores
