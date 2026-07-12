"""Tabular neural networks for CIC IoT-DIAD experiments."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class TabularMLP(nn.Module):
    """A compact MLP suitable for CPU baselines and LFighter's final-layer analysis."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 8,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        layers = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden

        self.features = nn.Sequential(*layers)
        # Keep classifier as the final layer. LFighter can later inspect its
        # class-specific weights and bias without changing its core principle.
        self.classifier = nn.Linear(previous, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
