import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from simclr_pretrain.lib.models import SpectrumEncoder

class BinaryClassifier(nn.Module):
    """二分类模型（挂载在质谱编码器之上）"""
    def __init__(self, encoder, input_dim=256, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        
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
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        if self.freeze_encoder:
            with torch.no_grad():
                embed = self.encoder(x)
        else:
            embed = self.encoder(x)
            
        logit = self.classifier(embed)
        return logit.squeeze(-1)


def load_pretrained_encoder(encoder_path, input_dim=561, hidden_dim=256, device='cuda'):
    """实例化 SpectrumEncoder 并加载预训练权重"""
    dev = torch.device(device if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu')
    encoder = SpectrumEncoder(input_dim=input_dim, hidden_dim=hidden_dim).to(dev)
    
    encoder_path = Path(encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"未找到预训练模型文件: {encoder_path}")
        
    checkpoint = torch.load(str(encoder_path), map_location=dev)
    
    if isinstance(checkpoint, dict):
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    # 清理 key 前缀
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith('module.'):
            new_k = new_k[7:]
        if new_k.startswith('encoder.'):
            new_k = new_k[8:]
        cleaned_state_dict[new_k] = v
        
    missing_keys, unexpected_keys = encoder.load_state_dict(cleaned_state_dict, strict=False)
    print(f"  [OK] 成功加载预训练权重: {encoder_path}")
    if missing_keys:
        print(f"    [WARN] 缺失键: {missing_keys}")
    if unexpected_keys:
        print(f"    [WARN] 忽略多余键: {unexpected_keys[:3]}...")
        
    return encoder
