"""Neural-network architectures."""

from __future__ import annotations

import torch
from torch import nn


class CNNLSTM(nn.Module):
    """Extract local temporal patterns before LSTM sequence summarization."""

    def __init__(
        self,
        n_features: int,
        conv_channels: int = 32,
        hidden_size: int = 48,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels),
            nn.Dropout(0.15),
        )
        self.lstm = nn.LSTM(conv_channels, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.20), nn.Linear(hidden_size, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        convolved = self.conv(inputs.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(convolved)
        return self.head(sequence[:, -1, :])

