# data/pipeline.py 增强版
class RegimeAwareDataset(Dataset):
    def __init__(self, stocks_tensors, seq_len=60, lookahead=5):
        self.samples = []
        # 增加一个字典，按波动率等级（Regime）存储索引，用于负采样
        self.regime_buckets = {0: [], 1: [], 2: []} # 0:低波, 1:中波, 2:高波

        for s_idx, tensor in enumerate(stocks_tensors):
            for i in range(len(tensor) - seq_len - lookahead):
                # 计算未来收益的波动作为 Regime 标签
                future_ret = tensor[i + seq_len : i + seq_len + lookahead, 0]
                vol = torch.std(future_ret).item()
                
                # 简单的分箱 (实际建议用 30/70 分位数)
                regime = 0 if vol < 0.01 else (2 if vol > 0.03 else 1)
                
                sample_info = (s_idx, i, regime)
                self.samples.append(sample_info)
                self.regime_buckets[regime].append(len(self.samples) - 1)
        
        self.stocks_tensors = stocks_tensors

    def __getitem__(self, idx):
        s_idx, start_idx, regime = self.samples[idx]
        anchor_x = self.stocks_tensors[s_idx][start_idx : start_idx + 60]
        
        # 构造正样本：从同一个 Regime 桶里随机抽一个
        pos_idx = np.random.choice(self.regime_buckets[regime])
        p_s_idx, p_start_idx, _ = self.samples[pos_idx]
        pos_x = self.stocks_tensors[p_s_idx][p_start_idx : p_start_idx + 60]
        
        # 构造负样本：从不同 Regime 桶里抽一个
        other_regimes = [r for r in [0, 1, 2] if r != regime]
        neg_regime = np.random.choice(other_regimes)
        neg_idx = np.random.choice(self.regime_buckets[neg_regime])
        n_s_idx, n_start_idx, _ = self.samples[neg_idx]
        neg_x = self.stocks_tensors[n_s_idx][n_start_idx : n_start_idx + 60]
        
        return anchor_x, pos_x, neg_x, regime