import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

# ============================================================
# 核心组件导入
# ============================================================
from core.backbone import MambaEncoder
from core.agent import IQNAgent
from risk.sidecar import IndependentRiskSidecar
from data.processor import FinancialFeatureEngineer
from data.loader import AShareDataLoader


# ============================================================
# 实盘 / 回测系统
# ============================================================
class LiveTradingSystem:
    def __init__(self, config):
        self.risk_ema = 0.0
        self.in_circuit_break = False
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----------------------------------------------------
        # 1. 模型初始化
        # ----------------------------------------------------
        self.encoder = MambaEncoder(
            d_model=config['d_model']
        ).to(self.device)

        self.agent = IQNAgent(
            latent_dim=config['d_model'],
            action_dim=3,
            K=config['K']
        ).to(self.device)

        # 🔥 唯一风险熔断源
        self.risk_sidecar = IndependentRiskSidecar(
            latent_dim=config['d_model']
        ).to(self.device)

        self.load_checkpoints()

        # ----------------------------------------------------
        # 2. 账户状态
        # ----------------------------------------------------
        self.position = 0      # 0: 空仓, 1: 多仓
        self.cash = config['initial_cash']
        self.equity = config['initial_cash']
        self.trade_log = []

    # --------------------------------------------------------
    # 权重加载
    # --------------------------------------------------------
    def load_checkpoints(self):
        if os.path.exists("models/encoder_latest.pth"):
            self.encoder.load_state_dict(
                torch.load("models/encoder_latest.pth", map_location=self.device),
                strict=False
            )
            print("✅ 已加载 Encoder 权重")

        if os.path.exists("models/agent_latest.pth"):
            self.agent.load_state_dict(
                torch.load("models/agent_latest.pth", map_location=self.device)
            )
            print("✅ 已加载 Agent 权重")

        if os.path.exists("models/sidecar_latest.pth"):
            self.risk_sidecar.load_state_dict(
                torch.load("models/sidecar_latest.pth", map_location=self.device)
            )
            print("✅ 已加载 Risk Sidecar 权重")

        self.encoder.eval()
        self.agent.eval()
        self.risk_sidecar.eval()

    # --------------------------------------------------------
    # 工业级稳定性补丁
    # --------------------------------------------------------
    def apply_stability_patch(self, latent):
        latent = F.layer_norm(latent, (latent.size(-1),))
        latent = torch.clamp(
            latent,
            -self.config['clip_val'],
            self.config['clip_val']
        )
        return latent

    # --------------------------------------------------------
    # A 股交易成本
    # --------------------------------------------------------
    def calculate_costs(self, action, price):
        if action == 1:  # BUY
            return price * (self.config['commission'] + self.config['slippage'])
        elif action == 2:  # SELL
            return price * (
                self.config['commission']
                + self.config['stamp_duty']
                + self.config['slippage']
            )
        return 0.0

    # --------------------------------------------------------
    # 核心决策逻辑（Sidecar 主导熔断）
    # --------------------------------------------------------
    @torch.no_grad()
def make_decision(self, market_seq):
    # ---------- Encoder ----------
    latent, _, _ = self.encoder(market_seq)
    latent = self.apply_stability_patch(latent)

    # ---------- Agent ----------
    action_idx = self.agent.select_action(
        latent,
        risk_kappa=self.config['risk_kappa']
    )

    # ---------- Sidecar ----------
    mdd_pred = self.risk_sidecar(latent).item()

    # ---------- EMA 更新 ----------
    alpha = self.config['risk_ema_alpha']
    self.risk_ema = (
        alpha * mdd_pred
        + (1 - alpha) * self.risk_ema
    )

    # ---------- Hysteresis ----------
    if not self.in_circuit_break:
        if self.risk_ema > self.config['risk_high']:
            self.in_circuit_break = True
    else:
        if self.risk_ema < self.config['risk_low']:
            self.in_circuit_break = False

    final_action = action_idx.item()
    is_circuit_break = False

    if self.in_circuit_break:
        final_action = 2
        is_circuit_break = True

    return final_action, mdd_pred, self.risk_ema, is_circuit_break


    # --------------------------------------------------------
    # 推断 / 回测主循环
    # --------------------------------------------------------
    def run_inference(self, live_data_df):
        print(f"🚀 开始对 {len(live_data_df)} 条数据运行推断...")

        feature_cols = [c for c in live_data_df.columns if c.endswith('_z')] + ['is_suspended']
        if not feature_cols:
            raise RuntimeError("❌ 未找到特征列，请先运行 FinancialFeatureEngineer")

        data_tensor = torch.FloatTensor(
            live_data_df[feature_cols].values
        ).to(self.device)

        for t in range(self.config['seq_len'], len(data_tensor)):
            window = data_tensor[
                t - self.config['seq_len']: t
            ].unsqueeze(0)

            price = live_data_df.iloc[t]['close']

            action, mdd, mdd_ema, broken = self.make_decision(window)

            if t % 20 == 0:
                print(
                    f"Time {t:4d} | "
                    f"Action {action} | "
                    f"MDD {mdd:.3f} | "
                    f"EMA {mdd_ema:.3f} | "
                    f"{'[RISK ON]' if broken else ''}"
                )



# ============================================================
# 全局配置
# ============================================================
config = {
    'd_model': 128,
    'K': 32,
    'seq_len': 60,
    'clip_val': 3.0,

    'initial_cash': 1_000_000.0,

    # 交易成本（A 股）
    'commission': 0.0002,
    'stamp_duty': 0.0005,
    'slippage': 0.001,

    # 风险参数
    'risk_kappa': 1.5,
    'mdd_threshold': 0.05,   # 预测未来最大回撤 > 5% → 熔断

    # === Risk EMA ===
    'risk_ema_alpha': 0.05,     # EMA 平滑系数（慢=更稳）
    
    # === Hysteresis ===
    'risk_high': 0.06,          # 进入熔断
    'risk_low': 0.03,           # 解除熔断
}


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    fe = FinancialFeatureEngineer(window_size=252)

    data_path = "data_source"
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

    if not csv_files:
        raise RuntimeError("❌ data_source 目录下没有 CSV 数据")

    sample_file = os.path.join(data_path, csv_files[0])
    print(f"📊 正在加载演示数据: {sample_file}")

    df = pd.read_csv(sample_file)
    df = fe.create_features(df)

    system = LiveTradingSystem(config)
    system.run_inference(df)
