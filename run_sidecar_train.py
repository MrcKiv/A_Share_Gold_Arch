import os
from data.loader import AShareDataLoader
from data.processor import FinancialFeatureEngineer
from train.train_sidecar import train_sidecar


def main():
    config = {
        'd_model': 128,
        'seq_len': 60,
        'mdd_horizon': 20,
        'batch_size': 512,
        'epochs': 10,
        'lr': 3e-4,
    }

    fe = FinancialFeatureEngineer(window_size=252)
    loader = AShareDataLoader(
        folder_path="data_source",
        seq_len=config['seq_len'],
        feature_engineer=fe
    )

    print("📥 加载股票数据...")
    stocks = loader.load_all_csv(limit=100)

    if not stocks:
        raise RuntimeError("❌ 未加载到有效股票数据")

    os.makedirs("models", exist_ok=True)

    train_sidecar(stocks, config)


if __name__ == "__main__":
    main()
