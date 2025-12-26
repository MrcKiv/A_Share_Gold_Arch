def run_rl_training(dataloader, encoder, agent, config):
    # 1. 彻底锁定 Encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    
    for x, _, _ in dataloader:
        x = x.cuda()
        
        # 2. 提取并稳定 Latent (Industrial Patch)
        with torch.no_grad():
            latent, _, _ = encoder(x)
            latent = torch.nn.functional.layer_norm(latent, (latent.size(-1),))
            latent = torch.clamp(latent, -3.0, 3.0)
            
        # 3. IQN 学习逻辑 (简化描述)
        taus = torch.rand(x.size(0), config['K']).cuda()
        q_values = agent.get_q_dist(latent, taus)
        
        # 计算 Quantile Regression Loss 并更新 Agent...
        # loss = calculate_quantile_huber_loss(q_values, rewards, next_states)