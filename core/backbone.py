import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMambaBlock(nn.Module):
    """
    Simplified causal Mamba (S6-like), pure PyTorch, Windows safe
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # depthwise causal conv
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x[:, :, :S]  # ✅ causal cut
        x = x.transpose(1, 2)

        x = F.silu(x)
        gate = torch.sigmoid(self.dt_proj(x))
        y = x * gate * torch.sigmoid(z)

        return self.out_proj(y)


class MambaEncoder(nn.Module):
    def __init__(self, input_dim=4, d_model=128, n_layers=4, n_quantiles=5):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList(
            [SimpleMambaBlock(d_model) for _ in range(n_layers)]
        )

        self.ln = nn.LayerNorm(d_model)

        # Tail-risk shaping head (NOT for trading path)
        self.quantile_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, n_quantiles)
        )

        # ✅ bounded diffusion residual
        self.diff_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x)

        latent = self.ln(x[:, -1])
        return latent, self.quantile_head(latent), self.diff_head(latent)
