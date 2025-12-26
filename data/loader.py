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
        """
        自动读取文件夹下所有 CSV 文件，支持 trade_date 表头
        """
        csv_files = glob(os.path.join(self.folder_path, "*.csv"))
        if limit:
            csv_files = csv_files[:limit]
        
        print(f"🚀 开始读取 {len(csv_files)} 只股票的数据...")
        
        for file in tqdm(csv_files):
            try:
                # 1. 读取数据
                df = pd.read_csv(file)
                
                # 2. 核心修正：自动识别日期列 (兼容 trade_date 或 date)
                possible_date_cols = ['trade_date', 'date', 'Date']
                date_col = next((c for c in possible_date_cols if c in df.columns), None)
                
                if date_col is None:
                    # print(f"⚠️ 跳过 {file}: 找不到日期列")
                    continue
                
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
                
                # 3. 核心修正：修复 fillna 警告
                df = df.ffill().fillna(0)
                
                # 4. 调用特征工程
                if self.fe:
                    # 传入 date_col 确保特征工程知道哪一列是日期
                    df = self.fe.create_features(df, date_col=date_col)
                
                # 5. 提取归一化后的特征列 (_z 结尾) 和 停牌标志
                feature_cols = [c for c in df.columns if c.endswith('_z')] + ['is_suspended']
                
                if len(feature_cols) <= 1: # 只有 is_suspended 则说明特征计算失败
                    continue
                
                # 转换为 Tensor [Time, Features]
                stock_tensor = torch.FloatTensor(df[feature_cols].values)
                
                # 6. 存储有效序列
                if len(stock_tensor) > self.seq_len + 10: # 预留点空间
                    self.all_stocks_data.append(stock_tensor)
                    
            except Exception as e:
                # print(f"⚠️ 跳过文件 {file}，原因: {e}")
                pass

        print(f"\n✅ 成功加载 {len(self.all_stocks_data)} 只股票的有效序列。")
        return self.all_stocks_data