import torch
from data.processor import FinancialFeatureEngineer
from data.loader import AShareDataLoader
from train.pretrain import train_pretrain
# from train.train_rl import run_rl_training # 假设你已准备好 RL 训练脚本

def main():
    # --- 1. 数据加载与特征工程 ---
    DATA_PATH = "./data_source/"  # 👈 你的 CSV 文件存放目录
    print("Step 1: 正在加载数据并进行特征工程...")
    
    fe = FinancialFeatureEngineer(window_size=252)
    loader = AShareDataLoader(DATA_PATH, seq_len=60, feature_engineer=fe)
    
    # 初次测试建议只读取 200 只股票，确认流程无误后再全量读取
    stocks_tensors = loader.load_all_csv(limit=200) 
    
    if not stocks_tensors:
        print("❌ 未加载到有效数据，请检查数据路径和 CSV 格式。")
        return

    # --- 2. 阶段一：Encoder 预训练 (Representation Learning) ---
    print("\nStep 2: 开始预训练 Encoder (Mamba + Diffusion + Risk Awareness)...")
    # 这步会保存模型到 ./models/encoder_latest.pth
    train_pretrain(stocks_tensors)
    
    print("\n✅ 预训练完成！模型已保存至 ./models/encoder_latest.pth")

    # --- 3. 阶段二：RL Agent 训练 (Decision Learning) ---
    # print("\nStep 3: 开始训练 RL 决策层...")
    # run_rl_training(stocks_tensors) 

if __name__ == "__main__":
    main()