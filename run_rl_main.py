import torch
from core.backbone import MambaEncoder
from core.agent import IQNAgent
from data.loader import AShareDataLoader
from data.processor import FinancialFeatureEngineer
from data.pipeline import RegimeAwareDataset
from train.train_rl import run_rl_training

def main():
    config = {
        'd_model': 128,
        'K': 32,
        'batch_size': 1024,
        'epochs': 20
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载数据
    fe = FinancialFeatureEngineer()
    loader = AShareDataLoader(folder_path="data_source", feature_engineer=fe)
    stocks = loader.load_all_csv(limit=50) 
    dataset = RegimeAwareDataset(stocks)

    # 2. 初始化模型
    encoder = MambaEncoder(d_model=config['d_model']).to(device)
    # 必须加载刚才预训练好的权重
    encoder.load_state_dict(torch.load("models/encoder_latest.pth", map_location=device), strict=False)
    
    agent = IQNAgent(latent_dim=config['d_model'], action_dim=3, K=config['K']).to(device)

    # 3. 开启 RL 训练
    run_rl_training(dataset, encoder, agent, config)

if __name__ == "__main__":
    main()