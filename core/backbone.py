import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMambaBlock(nn.Module):
    """
    纯 PyTorch 实现的简化版 Mamba 逻辑 (S6)
    无需 NVCC 编译，Windows 环境直接运行
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: [Batch, Seq_Len, D_Model]
        (batch, seq_len, d_model) = x.shape
        xz = self.in_proj(x) # [B, S, D_inner*2]
        x, z = xz.chunk(2, dim=-1) # 分为两条路径

        # 卷积路径
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = F.silu(x)

        # 简单的 SSM 模拟 (S6 核心逻辑的简化版)
        x_norm = F.silu(self.dt_proj(x))
        y = x * x_norm # 模拟选择性扫描

        # 门控路径融合
        out = y * F.silu(z)
        return self.out_proj(out)

class MambaEncoder(nn.Module):
    def __init__(self, d_model=128, n_layers=4, n_quantiles=5):
        super().__init__()
        # 使用我们自己写的 SimpleMambaBlock 替换官方 mamba-ssm
        self.layers = nn.ModuleList([
            SimpleMambaBlock(d_model=d_model) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.quantile_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, n_quantiles)
        )
        self.diff_head = nn.Linear(d_model, d_model)

    def forward(self, x, return_latent=True):
        for layer in self.layers:
            x = layer(x) + x
        
        latent = self.ln(x[:, -1, :])
        if return_latent:
            return latent, self.quantile_head(latent), self.diff_head(latent)
        return latent