import torch
import torch.nn as nn
import torch.nn.functional as F

class IndustrialRiskLoss(nn.Module):
    def __init__(self, quantiles=[0.01, 0.05, 0.5, 0.95, 0.99], temp=0.07):
        super().__init__()
        self.quantiles = quantiles
        self.temp = temp

    def pinball_loss(self, q_pred, y_true):
        # q_pred: [B, n_quantiles], y_true: [B, 1]
        losses = []
        for i, tau in enumerate(self.quantiles):
            err = y_true - q_pred[:, i:i+1]
            loss = torch.max(tau * err, (tau - 1) * err)
            losses.append(loss.mean())
        return torch.stack(losses).mean()

    def info_nce_causal(self, z_anchor, z_pos, z_neg):
        # z_pos 为历史上同 regime 的样本，z_neg 为异 regime 样本
        pos_sim = F.cosine_similarity(z_anchor, z_pos) / self.temp
        neg_sim = F.cosine_similarity(z_anchor, z_neg) / self.temp
        logits = torch.stack([pos_sim, neg_sim], dim=1) # [B, 2]
        labels = torch.zeros(z_anchor.size(0), dtype=torch.long).to(z_anchor.device)
        return F.cross_entropy(logits, labels)

    def forward(self, diff_out, noise_true, q_pred, y_future, z_tuple, weights):
        l_diff = F.mse_loss(diff_out, noise_true)
        l_tail = self.pinball_loss(q_pred, y_future)
        l_regime = self.info_nce_causal(*z_tuple)
        
        # 总损失 = λ1*Diff + λ2*Tail + λ3*Regime
        return weights[0]*l_diff + weights[1]*l_tail + weights[2]*l_regime