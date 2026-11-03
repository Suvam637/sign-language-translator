"""PyTorch sequence classifier for dynamic (moving) signs: BiGRU + attention pooling."""
from __future__ import annotations

import torch
from torch import nn


class SignGRU(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, T, F)
        h, _ = self.gru(self.proj(x))                             # (B, T, 2H)
        w = torch.softmax(self.attn(h), dim=1)                    # attention over time
        return self.head((w * h).sum(dim=1))                      # (B, C)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
