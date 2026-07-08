# /data/cye_temp/workspace/backtest_engine/config/Config.py
import os
from pathlib import Path

class Config:
    # 📂 数据路径 (只读)
    # DATA_DIR = "/data/train_data"
    # DATA_TEST_DIR = "/data/test_data"
    # RAW_PANEL = "/data/data_process/4.15_revision/raw_daily_panel.parquet"
    DATA_DIR = "c:/Users/yecha/workspace/data/train_data"
    DATA_TEST_DIR = "c:/Users/yecha/workspace/data/test_data"
    RAW_PANEL = "c:/Users/yecha/workspace/data/model_ready_panel_normalized_with_fwd_5d_label_and_backtest_fields_with_masks_rebuild_20150105_20260618.parquet"

    # 📂 工程路径 (可写)
    ROOT = Path(__file__).resolve().parents[1]
    OUT_DIR = ROOT / "output"
    FIG_DIR = OUT_DIR / "figures"
    PROC_DIR = ROOT / "processed_data/backtest"
    LOG_DIR  = ROOT / "log"
    MODEL_DIR = ROOT / "saved_models"

    # ⚙️ 回测结构参数 (🔑 已修正预热期)
    WARMUP_DAYS    = 0      # 🔑 预训练模型无需长预热，5天足够初始化状态
    REBALANCE_DAYS = 10
    TOP_K          = 50
    INITIAL_CAPITAL = 10_000_000.0
    COMMISSION_RATE = 0.0002

    # 💰 交易摩擦 (A股标准)
    SLIPPAGE = 0.000
    STAMP_TAX_RATE = 0.0000

    # 🤖 模型配置
    # MODELS = ['ElasticNet', 'XGBoost', 'LightGBM', 'Transformer', 'OptSharpe', 'BuyAndHoldAll']
    MODELS = ['ElasticNet', 'XGBoost', 'LightGBM', 'OptSharpe', 'BuyAndHoldAll']    
    # 🔍 SHAP 分析配置
    SHAP_SAMPLE_SIZE = 500