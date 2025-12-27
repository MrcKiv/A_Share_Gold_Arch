import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.backbone import MambaEncoder
from core.loss import IndustrialRiskLoss
from data.pipeline import RegimeAwareDataset


def train_pretrain(stocks):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = RegimeAwareDataset(stocks)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = MambaEncoder().to(device)
    loss_fn = IndustrialRiskLoss(
        quantiles=[0.01, 0.05, 0.5, 0.95, 0.99]
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        for batch in loader:
            a = batch["anchor"].to(device)
            p = batch["pos"].to(device)
            n = batch["neg"].to(device)
            y = batch["y_future"].to(device)
            noise = batch["noise"].to(device)

            za, qa, da = model(a)
            zp, _, _ = model(p)
            zn, _, _ = model(n)

            w = [1.0, min(1.0, epoch / 5), 0.5]
            l1, l2, l3 = loss_fn(
                da, noise,
                qa, y,
                (za, zp, zn),
                w
            )
            loss = l1 + l2 + l3

            if torch.isnan(loss):
                opt.zero_grad()
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

    torch.save(model.state_dict(), "models/encoder_latest.pth")
