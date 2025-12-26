import pandas as pd
import numpy as np
import torch

class FinancialFeatureEngineer:
    def __init__(self, window_size=252):
        self.window_size = window_size

    def create_features(self, df, date_col='trade_date'):
        """
        针对 A 股表头优化：trade_date, open, high, low, close, vol
        """
        df = df.copy()
        df = df.sort_values(date_col).reset_index(drop=True)

        # 1. 基础特征：对数收益率
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        # 2. A股停牌处理 (基于你提供的 'vol' 列名)
        # 逻辑：成交量为0 且 最高价等于最低价
        vol_col = 'vol' if 'vol' in df.columns else 'volume'
        df['is_suspended'] = ((df[vol_col] == 0) & (df['high'] == df['low'])).astype(float)

        # 3. 技术指标
        # 20日滚动波动率
        df['volatility'] = df['log_ret'].rolling(20).std()
        # 日内振幅位置
        df['range_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)

        # ---------------------------------------------------------
        # 4. 严格因果归一化 (Rolling Z-Score) 
        # ---------------------------------------------------------
        # 核心：使用 .shift(1) 确保 T 时刻的归一化只用到 T-1 之前的数据，杜绝泄露
        target_cols = ['log_ret', 'volatility', 'range_pos']
        
        for col in target_cols:
            rolling = df[col].rolling(window=self.window_size)
            mu = rolling.mean().shift(1)
            sigma = rolling.std().shift(1)
            df[f'{col}_z'] = (df[col] - mu) / (sigma + 1e-9)

        # 5. 清洗 NaN (由滚动窗口产生)
        df['log_ret_z'] = df['log_ret_z'].fillna(0)
        df = df.dropna().reset_index(drop=True)

        # 6. 计算 Regime 标签所需的未来波动率 (用于预训练对比学习)
        # 预测未来 5 天的波动率
        df['future_vol_label'] = df['log_ret'].rolling(5).std().shift(-5)
        
        return df