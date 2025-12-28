import torch
import torch.nn as nn

class ConsistencyDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_dim=20): # 预测未来 20 根 K 线
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )

    def generate_path(self, latent):
        # 直接映射：Latent -> 未来价格路径 (Single-step)
        return self.net(latent)