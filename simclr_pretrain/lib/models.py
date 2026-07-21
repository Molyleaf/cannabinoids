import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrumEncoder(nn.Module):
    """1D-CNN 质谱编码器"""
    def __init__(self, input_dim=561, hidden_dim=256):
        super().__init__()
        # 使用 GroupNorm(num_groups=8) 替换 BatchNorm1d 以彻底解决 BN 信息泄露
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
        # 使用 LayerNorm 替换 BatchNorm1d
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_block(x).squeeze(-1)
        embed = self.fc(h)
        # 引入 safe L2 归一化保护，防止极小模长向量的反向梯度爆炸
        norm = torch.norm(embed, p=2, dim=1, keepdim=True)
        return embed / torch.clamp(norm, min=1e-3)


class ProjectionHead(nn.Module):
    """投影头"""
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=64):
        super().__init__()
        # 使用 LayerNorm 替换 BatchNorm1d
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

