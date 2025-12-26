import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.backbone import MambaEncoder
from core.loss import IndustrialRiskLoss
from data.pipeline import RegimeAwareDataset
import os

# --- 1. 配置超参数 ---
CONFIG = {
    'd_model': 128,
    'n_layers': 4,
    'batch_size': 512,  # 考虑到 Triplet 会占用 3 倍显存，建议从 512 开始
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': './models/encoder_latest.pth'
}

def train_pretrain(stocks_tensors):
    device = CONFIG['device']
    
    # --- 2. 准备组件 ---
    # 初始化 Dataset & DataLoader
    dataset = RegimeAwareDataset(stocks_tensors, seq_len=60, lookahead=5)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    
    # 初始化模型与损失函数
    model = MambaEncoder(d_model=CONFIG['d_model'], n_layers=CONFIG['n_layers']).to(device)
    criterion = IndustrialRiskLoss(quantiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # 建立模型目录
    os.makedirs('./models', exist_ok=True)
    
    print(f"🔥 预训练启动！设备: {device}, 样本总数: {len(dataset)}")

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        
        # 动态计算 Loss 权重 (Annealing 策略)
        # 前期主攻 Diffusion (w1), 中期引入 Regime (w3), 后期强化 Tail Risk (w2)
        w1 = 1.0 # Diffusion 始终保持
        w2 = min(1.0, epoch / (CONFIG['epochs'] * 0.5)) # Tail Risk 权重逐渐增加
        w3 = 0.5 # 对比学习保持常态
        weights = [w1, w2, w3]

        for i, batch in enumerate(dataloader):
            # 获取 Triplet 样本
            anchor = batch['anchor'].to(device) # [B, 60, D]
            pos = batch['pos'].to(device)
            neg = batch['neg'].to(device)
            y_future = batch['y_future'].to(device).unsqueeze(1)
            noise_true = batch['noise'].to(device)

            # --- 3. 前向传播 ---
            # 为了计算 InfoNCE，我们需要三者的 Latents
            # 但只有 Anchor 需要计算 Quantile 和 Diffusion Head
            z_anchor, q_pred, diff_out = model(anchor)
            z_pos, _, _ = model(pos)
            z_neg, _, _ = model(neg)

            # --- 4. 计算组合 Loss ---
            # 这里的 z_tuple 是 (anchor, pos, neg)
            l_diff, l_tail, l_regime = criterion(
                diff_out, noise_true,     # Diffusion 项
                q_pred, y_future,         # Pinball 项
                (z_anchor, z_pos, z_neg), # InfoNCE 项
                weights
            )
            
            loss = l_diff + l_tail + l_regime

            # --- 5. 优化 ---
            optimizer.zero_grad()
            loss.backward()
            
            # 工业级补丁：梯度裁剪，防止 Mamba 在处理极端序列时梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(f"Epoch [{epoch}/{CONFIG['epochs']}] Step [{i}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (Diff: {l_diff.item():.4f}, "
                      f"Tail: {l_tail.item():.4f}, Regime: {l_regime.item():.4f})")

        # 每个 Epoch 结束后保存一次
        torch.save(model.state_dict(), CONFIG['save_path'])
        print(f"✨ Epoch {epoch} 完成，平均 Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    # 这里接入你之前 load_all_csv 得到的 stocks_tensors
    # stocks_tensors = loader.load_all_csv()
    # train_pretrain(stocks_tensors)
    pass