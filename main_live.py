import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

# 核心组件导入
from core.backbone import MambaEncoder
from core.agent import IQNAgent
from core.decoder import ConsistencyDecoder
from data.processor import FinancialFeatureEngineer
from data.loader import AShareDataLoader  # 修复 NameError

class LiveTradingSystem:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 初始化模型架构
        self.encoder = MambaEncoder(d_model=config['d_model']).to(self.device)
        self.agent = IQNAgent(latent_dim=config['d_model'], action_dim=3).to(self.device)
        self.risk_decoder = ConsistencyDecoder(latent_dim=config['d_model']).to(self.device)
        
        # 加载训练好的权重
        self.load_checkpoints()
        
        # 2. 初始化环境状态
        self.position = 0  # 当前持仓: 0-空仓, 1-多仓
        self.cash = config['initial_cash']
        self.equity = config['initial_cash']
        self.trade_log = []

    def load_checkpoints(self):
        """加载预训练好的权重文件"""
        try:
            # 确保路径与训练时保存的路径一致
            if os.path.exists("models/encoder_latest.pth"):
                self.encoder.load_state_dict(torch.load("models/encoder_latest.pth", map_location=self.device))
                print("✅ 已加载 Encoder 权重")
            
            if os.path.exists("models/agent_latest.pth"):
                self.agent.load_state_dict(torch.load("models/agent_latest.pth", map_location=self.device))
                print("✅ 已加载 Agent 权重")
                
            self.encoder.eval()
            self.agent.eval()
        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}，请确保模型文件存在于 models/ 目录下。")

    def apply_stability_patch(self, latent):
        """工业级稳定性补丁：LayerNorm + Clipping"""
        latent = torch.nn.functional.layer_norm(latent, (latent.size(-1),))
        latent = torch.clamp(latent, -self.config['clip_val'], self.config['clip_val'])
        return latent

    def calculate_costs(self, action, price):
        """计算 A 股交易成本"""
        cost = 0
        if action == 1: # 买入: 佣金 + 滑点
            cost = price * (self.config['commission'] + self.config['slippage'])
        elif action == 2: # 卖出: 佣金 + 印花税 + 滑点
            cost = price * (self.config['commission'] + self.config['stamp_duty'] + self.config['slippage'])
        return cost

    @torch.no_grad()
    def make_decision(self, market_seq):
        """核心双系统决策逻辑"""
        # --- System 1: 交易路径 ---
        latent, q_risk_head, _ = self.encoder(market_seq)
        latent = self.apply_stability_patch(latent)
        
        # IQN 决策
        action_idx = self.agent.select_action(latent, risk_kappa=self.config['risk_kappa'])
        
        # --- System 2: 风险旁路 (Sidecar) ---
        # 利用 Quantile Head 预测未来 1% 分位点的预期收益
        tail_risk = q_risk_head[0, 0].item() 
        
        final_action = action_idx.item()
        is_circuit_break = False
        
        # 熔断逻辑：如果预期亏损超过阈值，强制卖出/空仓
        if tail_risk < self.config['circuit_break_threshold']:
            final_action = 2 
            is_circuit_break = True
            
        return final_action, tail_risk, is_circuit_break

    def run_inference(self, live_data_df):
        """运行模拟推断循环"""
        print(f"🚀 开始对 {len(live_data_df)} 条数据运行推断...")
        
        # 提取特征列（以 _z 结尾的归一化特征）
        feature_cols = [c for c in live_data_df.columns if c.endswith('_z')] + ['is_suspended']
        if not feature_cols:
            print("❌ 错误：未在 DataFrame 中找到特征列，请确保已运行 FinancialFeatureEngineer")
            return

        data_tensor = torch.FloatTensor(live_data_df[feature_cols].values).to(self.device)

        for t in range(self.config['seq_len'], len(data_tensor)):
            # 获取滑动窗口
            window = data_tensor[t - self.config['seq_len'] : t].unsqueeze(0)
            current_price = live_data_df.iloc[t]['close']
            
            # 1. 执行决策
            action, risk_val, broken = self.make_decision(window)
            
            # 2. 模拟执行日志
            msg = "HOLD"
            if action == 1 and self.position == 0:
                self.position = 1
                msg = f"BUY at {current_price:.2f}"
            elif action == 2 and self.position == 1:
                self.position = 0
                msg = f"SELL at {current_price:.2f}"
            
            if broken:
                msg += " [!!! CIRCUIT BREAKER TRIGGERED !!!]"

            if t % 20 == 0: # 每 20 个周期打印一次日志
                print(f"Time: {t} | Action: {action} | TailRisk: {risk_val:.4f} | Msg: {msg}")

# --- 全局配置参数 ---
config = {
    'd_model': 128,
    'seq_len': 60,
    'clip_val': 3.0,
    'initial_cash': 1000000.0,
    'commission': 0.0002,      # 万二佣金
    'stamp_duty': 0.0005,      # 千分之五印花税
    'slippage': 0.001,         # 千一滑点
    'risk_kappa': 1.5,         # 风险偏好系数
    'circuit_break_threshold': -0.04, # 预期跌幅超 4% 熔断
}

# --- 启动逻辑 ---
if __name__ == "__main__":
    # 1. 环境准备
    fe = FinancialFeatureEngineer(window_size=252)
    
    # 2. 读取演示数据（请确保 data_source 目录下有 csv 文件）
    data_path = "data_source"
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ 错误：在 {data_path} 目录下未找到任何 CSV 数据文件。")
    else:
        # 读取第一只股票进行演示
        sample_file = os.path.join(data_path, csv_files[0])
        print(f"📊 正在加载演示数据: {sample_file}")
        
        df = pd.read_csv(sample_file)
        # 执行特征工程
        df_processed = fe.create_features(df)
        
        # 3. 启动系统
        system = LiveTradingSystem(config)
        
        # 4. 执行回测/推断
        system.run_inference(df_processed)