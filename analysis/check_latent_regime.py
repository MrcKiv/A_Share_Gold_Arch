import sys
import os

# ============================================================
# 强制注入项目根目录（不可失败，解决 core / data import）
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# 标准库 / 第三方库
# ============================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ============================================================
# 项目内模块
# ============================================================
from core.backbone import MambaEncoder
from data.pipeline import RegimeAwareDataset
from data.loader import AShareDataLoader
from data.processor import FinancialFeatureEngineer


def main():
    # --------------------------------------------------------
    # 0. 基本配置
    # --------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples = 1000          # 最多抽取多少个样本做 PCA
    min_valid_samples = 50      # 至少需要多少个有效 latent

    print(f"🧠 Using device: {device}")

    # --------------------------------------------------------
    # 1. 加载 Encoder（兼容旧 checkpoint）
    # --------------------------------------------------------
    encoder = MambaEncoder().to(device)

    ckpt_path = os.path.join(PROJECT_ROOT, "models", "encoder_latest.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state, strict=False)  # ✅ 关键：兼容结构变更
    encoder.eval()

    print("✅ Encoder loaded (strict=False)")

    # --------------------------------------------------------
    # 2. 加载少量数据（只用于分析）
    # --------------------------------------------------------
    fe = FinancialFeatureEngineer(window_size=252)

    loader = AShareDataLoader(
        folder_path="data_source",   # ⚠️ 确保与你的 CSV 路径一致
        seq_len=60,
        feature_engineer=fe
    )

    stocks = loader.load_all_csv(limit=30)
    if len(stocks) == 0:
        raise RuntimeError("No valid stock data loaded.")

    dataset = RegimeAwareDataset(stocks)

    print(f"📊 Dataset size: {len(dataset)} samples")

    # --------------------------------------------------------
    # 3. 抽取 latent（带 NaN / Inf 防护）
    # --------------------------------------------------------
    latents = []
    regimes = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            sample = dataset[i]

            x = sample["anchor"].unsqueeze(0).to(device)  # [1, seq_len, dim]
            z, _, _ = encoder(x)

            z_np = z.cpu().numpy()[0]

            # ========== 工业级分析防护 ==========
            if not np.isfinite(z_np).all():
                continue  # 跳过 NaN / Inf latent

            latents.append(z_np)
            regimes.append(sample.get("regime", 0))

    latents = np.array(latents)
    regimes = np.array(regimes)

    print(f"✅ Collected {len(latents)} valid latents")

    if len(latents) < min_valid_samples:
        raise RuntimeError(
            f"Too few valid latents ({len(latents)}). "
            f"Encoder may be unstable on extreme samples."
        )

    # --------------------------------------------------------
    # 4. PCA 降维
    # --------------------------------------------------------
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(latents)

    explained = pca.explained_variance_ratio_.sum()
    print(f"📉 PCA explained variance (2D): {explained:.2%}")

    # --------------------------------------------------------
    # 5. 可视化
    # --------------------------------------------------------
    plt.figure(figsize=(8, 6))

    colors = {
        0: "green",   # low vol
        1: "blue",    # mid vol
        2: "red"      # high vol
    }

    for r in [0, 1, 2]:
        idx = regimes == r
        if idx.sum() == 0:
            continue
        plt.scatter(
            z_2d[idx, 0],
            z_2d[idx, 1],
            s=8,
            c=colors[r],
            label=f"Regime {r}",
            alpha=0.6
        )

    plt.legend()
    plt.title("Latent Regime Separation (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
