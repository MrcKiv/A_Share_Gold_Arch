import torch
import numpy as np
from torch.utils.data import Dataset


class RegimeAwareDataset(Dataset):
    """
    Regime-aware dataset for representation learning.
    No future leakage into encoder weights.
    """

    def __init__(self, stocks, seq_len=60, lookahead=5):
        self.seq_len = seq_len
        self.lookahead = lookahead
        self.stocks = stocks

        self.samples = []
        self.regime_buckets = {0: [], 1: [], 2: []}

        for s_idx, tensor in enumerate(stocks):
            length = tensor.size(0)
            if length < seq_len + lookahead + 1:
                continue

            for i in range(length - seq_len - lookahead):
                future_ret = tensor[
                    i + seq_len : i + seq_len + lookahead, 0
                ]
                vol = torch.std(future_ret).item()

                if vol < 0.01:
                    regime = 0
                elif vol > 0.03:
                    regime = 2
                else:
                    regime = 1

                self.regime_buckets[regime].append(len(self.samples))
                self.samples.append((s_idx, i, regime))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_idx, start, regime = self.samples[idx]

        anchor = self.stocks[s_idx][start : start + self.seq_len]

        pos_idx = np.random.choice(self.regime_buckets[regime])
        p_s_idx, p_start, _ = self.samples[pos_idx]
        pos = self.stocks[p_s_idx][p_start : p_start + self.seq_len]

        neg_regime = np.random.choice([r for r in [0, 1, 2] if r != regime])
        neg_idx = np.random.choice(self.regime_buckets[neg_regime])
        n_s_idx, n_start, _ = self.samples[neg_idx]
        neg = self.stocks[n_s_idx][n_start : n_start + self.seq_len]

        y_future = self.stocks[s_idx][
            start + self.seq_len : start + self.seq_len + self.lookahead, 0
        ].sum()

        # Industrial stability clamp
        y_future = torch.clamp(y_future, -0.1, 0.1)

        noise = torch.randn(anchor.size(-1))

        return {
            "anchor": anchor,
            "pos": pos,
            "neg": neg,
            "y_future": y_future.unsqueeze(0),
            "noise": noise,
            "regime": regime  # <-- 必须加上这一行
        }
