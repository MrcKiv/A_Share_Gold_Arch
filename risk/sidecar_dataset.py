import torch
from torch.utils.data import Dataset
import numpy as np


class SidecarDataset(Dataset):
    """
    Dataset for training Risk Sidecar.
    Label: future maximum drawdown (MDD)
    """

    def __init__(self, stocks, seq_len=60, horizon=20):
        self.seq_len = seq_len
        self.horizon = horizon
        self.samples = []

        for stock in stocks:
            prices = stock[:, 0]  # 假设第 0 维是 log_ret 或 close proxy
            length = len(prices)

            if length < seq_len + horizon + 1:
                continue

            for t in range(length - seq_len - horizon):
                window = stock[t : t + seq_len]
                future = prices[t + seq_len : t + seq_len + horizon]

                # ---------- 计算未来最大回撤 ----------
                peak = torch.cummax(future, dim=0).values
                drawdown = (future - peak) / (peak + 1e-6)
                mdd = drawdown.min().abs()   # 正数

                self.samples.append((window, mdd))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, mdd = self.samples[idx]
        return {
            "x": x,
            "mdd": torch.clamp(mdd, 0.0, 1.0)
        }
