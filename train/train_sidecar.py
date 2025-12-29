import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.backbone import MambaEncoder
from risk.sidecar import IndependentRiskSidecar
from risk.sidecar_dataset import SidecarDataset


def train_sidecar(stocks, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Dataset & Loader
    # --------------------------------------------------
    dataset = SidecarDataset(
        stocks,
        seq_len=config['seq_len'],
        horizon=config['mdd_horizon']
    )

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # --------------------------------------------------
    # Models
    # --------------------------------------------------
    encoder = MambaEncoder(d_model=config['d_model']).to(device)
    encoder.load_state_dict(
        torch.load("models/encoder_latest.pth", map_location=device),
        strict=False
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    sidecar = IndependentRiskSidecar(
        latent_dim=config['d_model']
    ).to(device)

    optimizer = torch.optim.AdamW(
        sidecar.parameters(),
        lr=config['lr'],
        weight_decay=1e-4
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    print("🚧 开始训练 Risk Sidecar...")

    for epoch in range(config['epochs']):
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            x = batch["x"].to(device)
            mdd_true = batch["mdd"].to(device)

            with torch.no_grad():
                latent, _, _ = encoder(x)
                latent = F.layer_norm(latent, (latent.size(-1),))
                latent = torch.clamp(latent, -3.0, 3.0)

            mdd_pred = sidecar(latent).squeeze(-1)

            # MSE + L1 混合（金融风险更稳）
            loss = (
                F.mse_loss(mdd_pred, mdd_true)
                + 0.2 * F.l1_loss(mdd_pred, mdd_true)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sidecar.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | Sidecar Loss: {avg_loss:.6f}")

    torch.save(sidecar.state_dict(), "models/sidecar_latest.pth")
    print("✅ Risk Sidecar 训练完成并已保存")
