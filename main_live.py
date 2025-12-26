import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from core.backbone import MambaEncoder
from core.agent import IQNAgent
from core.decoder import ConsistencyDecoder
from data.processor import FinancialFeatureEngineer

class LiveTradingSystem:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 初始化并加载模型
        self.encoder = MambaEncoder(d_model=config['d_model']).to(self.device)
        self.agent = IQNAgent(latent_dim=config['d_model'], action_dim=3).to(self.device)
        self.risk_decoder = ConsistencyDecoder(latent_dim=config['d_model']).to(self.device)
        
        self.load_checkpoints()
        
        # 2. 初始化环境状态
        self.position = 0  # 当前持仓: 0-空仓, 1-多仓
        self.cash = config['initial_cash']
        self.equity = config['initial_cash']
        self.trade_log = []

    def load_checkpoints(self):
        """加载训练好的权重"""
        try:
            self.encoder.load_state_dict(torch.load("models/encoder_latest.pth", map_location=self.device))
            self.agent.load_state_dict(torch.load("models/agent_latest.pth", map_location=self.device))
            self.encoder.eval()
            self.agent.eval()
            print("✅ 已成功加载所有模型权重。")
        except FileNotFoundError:
            print("⚠️ 未找到权重文件，请先运行 pretrain.py 和 train_rl.py")

    def apply_stability_patch(self, latent):
        """工业级稳定性补丁：LayerNorm + Clipping"""
        # 消除 Regime Shift 带来的分布漂移
        latent = torch.nn.functional.layer_norm(latent, (latent.size(-1),))
        # 抑制极端离群值，防止 RL Agent 做出疯狂决策
        latent = torch.clamp(latent, -self.config['clip_val'], self.config['clip_val'])
        return latent

    def calculate_costs(self, action, price):
        """
        计算 A 股交易成本
        - 佣金: 0.02% (万二)
        - 印花税: 0.05% (仅卖出时缴纳, 2024年标准)
        - 滑点: 假设 0.1% (模拟冲击成本)
        """
        cost = 0
        if action == 1: # 买入
            cost = price * (self.config['commission'] + self.config['slippage'])
        elif action == 2: # 卖出
            cost = price * (self.config['commission'] + self.config['stamp_duty'] + self.config['slippage'])
        return cost

    @torch.no_grad()
    def make_decision(self, market_seq):
        """
        核心双系统决策逻辑
        market_seq: [1, seq_len, features]
        """
        # --- System 1: 交易路径 ---
        latent, q_risk_head, _ = self.encoder(market_seq)
        latent = self.apply_stability_patch(latent)
        
        # IQN 决策：考虑风险敏感度 kappa
        # 返回动作：0-观望, 1-买入/持仓, 2-卖出/空仓
        action_idx = self.agent.select_action(latent, risk_kappa=self.config['risk_kappa'])
        
        # --- System 2: 风险旁路 (Sidecar) ---
        # 逻辑：利用 Quantile Head 预测未来 1% 分位点的预期收益
        # 如果预期回撤 > 熔断阈值，强制执行熔断动作 (Action 2)
        tail_risk = q_risk_head[0, 0].item() # 0.01 分位点
        
        final_action = action_idx.item()
        is_circuit_break = False
        
        if tail_risk < self.config['circuit_break_threshold']:
            final_action = 2 # 强制平仓/空仓
            is_circuit_break = True
            
        return final_action, tail_risk, is_circuit_break

    def run_inference(self, live_data_df):
        """
        运行实盘/回测推断循环
        live_data_df: 处理好的 DataFrame
        """
        print("🚀 开始推断逻辑...")
        # 假设 live_data_df 已经经过特征工程处理
        feature_cols = [c for c in live_data_df.columns if c.endswith('_z')]
        data_tensor = torch.FloatTensor(live_data_df[feature_cols].values).to(self.device)

        for t in range(self.config['seq_len'], len(data_tensor)):
            # 获取当前窗口
            window = data_tensor[t - self.config['seq_len'] : t].unsqueeze(0)
            current_price = live_data_df.iloc[t]['close']
            
            # 1. 做出决策
            action, risk_val, broken = self.make_decision(window)
            
            # 2. 模拟执行与成本计算 (此处简化逻辑)
            # 实际系统中需考虑 T+1 和 可用资金
            msg = "HOLD"
            if action == 1 and self.position == 0:
                cost = self.calculate_costs(1, current_price)
                self.position = 1
                msg = f"BUY at {current_price:.2f}"
            elif action == 2 and self.position == 1:
                cost = self.calculate_costs(2, current_price)
                self.position = 0
                msg = f"SELL at {current_price:.2f}"
            
            if broken:
                msg += " [!!! CIRCUIT BREAKER !!!]"

            # 打印日志（或写入数据库）
            if t % 10 == 0: # 减少打印频率
                print(f"Time: {t} | Action: {action} | TailRisk: {risk_val:.4f} | Msg: {msg}")

# --- 配置参数 ---
config = {
    'd_model': 128,
    'seq_len': 60,
    'clip_val': 3.0,
    'initial_cash': 1000000.0,
    'commission': 0.0002,
    'stamp_duty': 0.0005,
    'slippage': 0.001,
    'risk_kappa': 1.5,
    'circuit_break_threshold': -0.04, # 预期 1% 亏损超 4% 即熔断
}

# --- 启动 ---
if __name__ == "__main__":
    # 此处假设你已经加载了 csv 并生成了 df
    # df = pd.read_csv("your_processed_data.csv")
    # system = LiveTradingSystem(config)
    # system.run_inference(df)
    pass