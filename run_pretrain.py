import os
import torch
from data.loader import AShareDataLoader
from data.processor import FinancialFeatureEngineer
from train.pretrain import train_pretrain

def main():
    # 1. 初始化特征工程和加载器
    fe = FinancialFeatureEngineer(window_size=252)
    loader = AShareDataLoader(
        folder_path="data_source", 
        seq_len=60, 
        feature_engineer=fe
    )

    # 2. 加载股票数据 (先加载少量，如 100 只，确保流程跑通)
    print("正在加载数据...")
    stocks = loader.load_all_csv(limit=100)
    
    if not stocks:
        print("错误：未找到有效数据，请检查 data_source 目录。")
        return

    # 3. 确保模型保存目录存在
    os.makedirs("models", exist_ok=True)

    # 4. 开始预训练
    print("开始预训练...")
    train_pretrain(stocks)

if __name__ == "__main__":
    main()