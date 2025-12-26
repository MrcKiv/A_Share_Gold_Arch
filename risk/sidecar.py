import torch
import torch.nn as nn

class IndependentRiskSidecar(nn.Module):
    """
    独立风控旁路：专注于预测未来风险的“非对称性”
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        # 独立的 MLP，不与 Agent 共享参数
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1) # 输出：未来 MDD (最大回撤) 的预期值
        )

    def forward(self, latent):
        """
        latent: 来自 Encoder 的 [Batch, Latent_Dim]
        输出结果经过 Sigmoid 缩放到 0-1 之间，代表百分比回撤
        """
        # 使用 Sigmoid 限制在 0-1，假设最大可能回撤是 100% (实际上 A 股通常设置 10%-20% 熔断)
        mdd_pred = torch.sigmoid(self.net(latent))
        return mdd_pred

    def get_risk_signal(self, latent, threshold=0.05):
        """
        实盘中调用的方法：如果预测回撤 > 5%，返回熔断信号
        """
        mdd_val = self.forward(latent)
        is_unsafe = mdd_val > threshold
        return is_unsafe, mdd_val