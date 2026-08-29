# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumEncoder(nn.Module):
    """
    1D CNN Encoder with GroupNorm and LayerNorm for binary risk classification.
    """
    def __init__(self, input_dim=561, hidden_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_block(x).squeeze(-1)
        embed = self.fc(h)
        norm = torch.norm(embed, p=2, dim=1, keepdim=True)
        return embed / torch.clamp(norm, min=1e-3)


class BinaryClassifier(nn.Module):
    """
    Binary risk classification head with optional encoder freezing.
    """
    def __init__(self, encoder, input_dim=256, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        with torch.set_grad_enabled(not all(p.requires_grad is False for p in self.encoder.parameters())):
            embed = self.encoder(x)
        logit = self.classifier(embed)
        return logit.squeeze(-1)


class MultiClassSpectrumEncoder(nn.Module):
    """
    1D CNN Encoder with BatchNorm for 9-class NPS classification.
    """
    def __init__(self, input_dim=561, hidden_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_block(x).squeeze(-1)
        embed = self.fc(h)
        return F.normalize(embed, p=2, dim=1)


class MultiClassClassifier(nn.Module):
    """
    Multi-class classification head for 9 NPS categories.
    """
    def __init__(self, encoder, num_classes=9, input_dim=256, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        embed = self.encoder(x)
        logits = self.classifier(embed)
        return logits
