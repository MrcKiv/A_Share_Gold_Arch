import torch
import torch.nn as nn
import torch.nn.functional as F


class IndustrialRiskLoss(nn.Module):
    """
    Diffusion + Tail Risk (Pinball) + Regime Contrastive
    Numerically safe version
    """
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)

    def pinball_loss(self, q_pred, y):
        losses = []
        for i, tau in enumerate(self.quantiles.to(q_pred.device)):
            diff = y - q_pred[:, i:i+1]
            loss = torch.maximum(
                tau * diff,
                (tau - 1) * diff
            )
            losses.append(loss.mean())
        return sum(losses) / len(losses)

    def contrastive_loss(self, z_a, z_p, z_n, temperature=0.1):
        # ✅ latent normalization to avoid NaN
        z_a = z_a / (z_a.norm(dim=-1, keepdim=True) + 1e-6)
        z_p = z_p / (z_p.norm(dim=-1, keepdim=True) + 1e-6)
        z_n = z_n / (z_n.norm(dim=-1, keepdim=True) + 1e-6)

        pos = torch.sum(z_a * z_p, dim=-1) / temperature
        neg = torch.sum(z_a * z_n, dim=-1) / temperature

        logits = torch.stack([pos, neg], dim=1)
        labels = torch.zeros(z_a.size(0), dtype=torch.long, device=z_a.device)

        return F.cross_entropy(logits, labels)

    def forward(
        self,
        diff_out, noise_true,
        q_pred, y_future,
        z_tuple,
        weights
    ):
        w_diff, w_tail, w_reg = weights

        l_diff = F.mse_loss(diff_out, noise_true)

        l_tail = self.pinball_loss(q_pred, y_future)

        z_a, z_p, z_n = z_tuple
        l_reg = self.contrastive_loss(z_a, z_p, z_n)

        # ✅ RMS normalize to align gradient scales
        def norm(x): return x / (x.detach().abs().mean() + 1e-6)

        l_diff = norm(l_diff)
        l_tail = norm(l_tail)
        l_reg = norm(l_reg)

        return (
            w_diff * l_diff,
            w_tail * l_tail,
            w_reg * l_reg
        )
