import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def run_rl_training(dataset, encoder, agent, config):
    # 1. 彻底锁定 Encoder，RL 只训练决策头 (Agent)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.AdamW(agent.parameters(), lr=3e-4)
    loader = DataLoader(dataset, batch_size=config.get('batch_size', 1024), shuffle=True)
    
    print(f"📈 开始 RL 训练 (IQN 算法)...")
    
    for epoch in range(config.get('epochs', 50)):
        total_loss = 0
        for batch in loader:
            # 获取 Anchor 序列作为当前状态 S
            x = batch["anchor"].to(device)
            # 这里简化演示，使用 y_future 作为奖励 R (实际应根据 Action 计算收益)
            rewards = batch["y_future"].to(device) 
            
            # 2. 提取并稳定 Latent (Industrial Patch)
            with torch.no_grad():
                latent, _, _ = encoder(x)
                # 消除分布漂移并限制离群值
                latent = F.layer_norm(latent, (latent.size(-1),))
                latent = torch.clamp(latent, -3.0, 3.0)
            
            # 3. 计算 IQN 损失 (分位数回归)
            # 采样两组不同的分位数 taus
            taus = torch.rand(x.size(0), config['K']).to(device)
            # 获取当前状态的 Q 分布
            current_q_dist = agent.get_q_dist(latent, taus) 
            
            # 简化版 Huber Loss 目标 (针对金融数据的厚尾特性)
            # 实际 RL 中需对比 Q(s,a) 与 R + Q(s', a')，此处以拟合预期收益为例
            diff = rewards.unsqueeze(1) - current_q_dist # [B, K, Action]
            loss = (torch.abs(taus.unsqueeze(-1) - (diff < 0).float()) * F.huber_loss(diff, torch.zeros_like(diff), reduction='none')).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch} | RL Loss: {total_loss/len(loader):.6f}")

    # 4. 保存 Agent 权重
    torch.save(agent.state_dict(), "models/agent_latest.pth")
    print("✅ RL Agent 训练完成并已保存。")