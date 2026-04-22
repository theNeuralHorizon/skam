"""Self-Attention anomaly scorer -- shared between training and inference.

Architecture
------------
Linear(5 → 32) → MultiheadAttention(d=32, heads=4)
→ LayerNorm → mean-pool → Linear(32 → 16) → ReLU → Linear(16 → 1) → Sigmoid

Input : (batch, seq_length, 5)
Output: (batch, 1) -- anomaly probability in [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttentionScorer(nn.Module):
    """Temporal anomaly scorer using multi-head self-attention."""

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_length, input_dim)

        Returns
        -------
        Tensor of shape (batch, 1) -- anomaly probability
        """
        h = self.input_proj(x)  # (B, T, d_model)
        attn_out, _ = self.attention(h, h, h)  # self-attention
        h = self.layer_norm(h + attn_out)  # residual + norm
        pooled = h.mean(dim=1)  # (B, d_model) -- mean pool over time
        return self.classifier(pooled)  # (B, 1)
