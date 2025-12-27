import pandas as pd
import numpy as np
import torch

class FinancialFeatureEngineer:
    def __init__(self, window_size=252):
        self.window_size = window_size

    def create_features(self, df, date_col='trade_date'):
        """
        针对 A 股表头优化，并增加数值稳定性保护
        """
        df = df.copy()
        df = df.sort_values(date_col).reset_index(drop=True)

        # 1. 基础特征：对数收益率 (增加 1e-9 保护防止价格为0或负值)
        # 仅处理收盘价大于 0 的有效数据
        df = df[df['close'] > 0].copy()
        df['log_ret'] = np.log(df['close'] / (df['close'].shift(1) + 1e-9) + 1e-9)
        df['log_ret'] = df['log_ret'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 2. A股停牌处理
        vol_col = 'vol' if 'vol' in df.columns else 'volume'
        df['is_suspended'] = ((df[vol_col] == 0) & (df['high'] == df['low'])).astype(float)

        # 3. 技术指标
        df['volatility'] = df['log_ret'].rolling(20).std()
        df['range_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)

        # 4. 严格因果归一化 (Rolling Z-Score) 
        # 增加 epsilon 防止除以 0，并使用 clip 限制离群值
        target_cols = ['log_ret', 'volatility', 'range_pos']
        
        for col in target_cols:
            rolling = df[col].rolling(window=self.window_size)
            mu = rolling.mean().shift(1)
            sigma = rolling.std().shift(1)
            # 这里的 1e-6 防止波动率为 0 导致的 NaN
            df[f'{col}_z'] = (df[col] - mu) / (sigma + 1e-6)
            # 工业级补丁：将特征限制在 [-5, 5] 之间，防止极端波动摧毁神经网络权重
            df[f'{col}_z'] = df[f'{col}_z'].clip(-5.0, 5.0)

        # 5. 清洗 NaN
        df = df.fillna(0)

        # 6. 计算未来 5 天滚动波动率标签 (用于 Regime 分类)
        df['future_vol_label'] = df['log_ret'].rolling(5).std().shift(-5)
        
        return df