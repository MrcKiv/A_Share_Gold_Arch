import os
import pandas as pd
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

class AShareDataLoader:
    def __init__(self, folder_path, seq_len=60, feature_engineer=None):
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.fe = feature_engineer
        self.all_stocks_data = []

    def load_all_csv(self, limit=None):
        csv_files = glob(os.path.join(self.folder_path, "*.csv"))
        if limit:
            csv_files = csv_files[:limit]
        
        print(f"🚀 开始读取 {len(csv_files)} 只股票的数据...")
        
        for file in tqdm(csv_files):
            try:
                df = pd.read_csv(file)
                
                possible_date_cols = ['trade_date', 'date', 'Date']
                date_col = next((c for c in possible_date_cols if c in df.columns), None)
                
                if date_col is None: continue
                
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
                df = df.ffill().fillna(0)
                
                if self.fe:
                    df = self.fe.create_features(df, date_col=date_col)
                
                feature_cols = [c for c in df.columns if c.endswith('_z')] + ['is_suspended']
                
                if len(feature_cols) <= 1: continue
                
                stock_tensor = torch.FloatTensor(df[feature_cols].values)
                
                # 核心修正：检查最终的 Tensor 是否含有非法数值
                if torch.isnan(stock_tensor).any() or torch.isinf(stock_tensor).any():
                    # print(f"⚠️ 跳过 {file}: 检测到无效数值(NaN/Inf)")
                    continue
                
                if len(stock_tensor) > self.seq_len + 10:
                    self.all_stocks_data.append(stock_tensor)
                    
            except Exception as e:
                pass

        print(f"\n✅ 成功加载 {len(self.all_stocks_data)} 只股票的有效序列。")
        return self.all_stocks_data