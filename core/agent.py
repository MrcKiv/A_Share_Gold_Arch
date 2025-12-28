import torch
import torch.nn as nn

class IQNAgent(nn.Module):
    def __init__(self, latent_dim=128, action_dim=3, K=32):
        super().__init__()
        self.K = K  # 分位数采样数
        self.phi = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.fc_q = nn.Linear(latent_dim, action_dim)

    def get_q_dist(self, latent, taus):
        # latent: [B, L], taus: [B, K]
        # 1. 融合 Latent 与分位数嵌入
        phi_taus = self.phi(taus.unsqueeze(-1)) # [B, K, L]
        combined = latent.unsqueeze(1) * phi_taus # 特征融合
        
        # 2. 输出每个分位点下的动作价值
        quantile_values = self.fc_q(combined) # [B, K, Action]
        return quantile_values

    def select_action(self, latent, risk_kappa=1.5):
        # 此时 latent 必须已过 LN 和 Clipping
        batch_size = latent.size(0)
        taus = torch.rand(batch_size, self.K).to(latent.device)
        
        q_dist = self.get_q_dist(latent, taus) # [B, K, Action]
        
        # 风险敏感型决策：E[Q] - κ * Std[Q]
        q_mean = q_dist.mean(dim=1)
        q_std = q_dist.std(dim=1)
        risk_adjusted_q = q_mean - risk_kappa * q_std
        
        return risk_adjusted_q.argmax(dim=-1)