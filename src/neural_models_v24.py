"""Neural tabular models for CIC IoT-DIAD reliability screening V2.4."""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class PlainMLP(nn.Module):
    """Compact MLP with an explicit final classifier for LFighter compatibility."""

    def __init__(self, input_dim: int, num_classes: int = 8, dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.LayerNorm(width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.LayerNorm(width),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResidualMLP(nn.Module):
    """Residual MLP designed for stable CPU training and FL output-gradient analysis."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 8,
        width: int = 256,
        depth: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(depth)])
        self.penultimate = nn.Sequential(
            nn.Linear(width, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.penultimate(x)
        return self.classifier(x)


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularCNN1D(nn.Module):
    """Small 1D CNN used as an architecture ablation for ordered feature vectors."""

    def __init__(self, input_dim: int, num_classes: int = 8, dropout: float = 0.15) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ConvResidualBlock(64, dropout),
            ConvResidualBlock(64, dropout),
        )
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.penultimate = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.blocks(self.stem(x))
        x = torch.cat([self.pool_avg(x).squeeze(-1), self.pool_max(x).squeeze(-1)], dim=1)
        x = self.penultimate(x)
        return self.classifier(x)


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class alpha weights."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.detach().clone().float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - target_probs).clamp_min(1e-8) ** self.gamma) * target_log_probs
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


def build_model(
    architecture: Literal["mlp", "resmlp", "cnn1d"],
    input_dim: int,
    num_classes: int,
) -> nn.Module:
    if architecture == "mlp":
        return PlainMLP(input_dim=input_dim, num_classes=num_classes)
    if architecture == "resmlp":
        return ResidualMLP(input_dim=input_dim, num_classes=num_classes)
    if architecture == "cnn1d":
        return TabularCNN1D(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture}")
