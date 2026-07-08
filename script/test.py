# /data/cye_temp/workspace/backtest_engine/script/test.py
import sys
import os
import logging
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.data_loader import load_panel_data, compute_real_returns, compute_ic_ir
from src.backtest_engine import BacktestEngine
from util.metrics import evaluate_and_plot
from config.Config import Config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def main():
    setup_logging()
    
    DATA_DIR = Config.DATA_DIR
    RAW_PANEL = Config.RAW_PANEL
    OUT_DIR = Config.OUT_DIR
    FIG_DIR = Config.FIG_DIR
    PROC_DIR = Config.PROC_DIR
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    yr_start, yr_end = 2016, 2023
    logging.info(f"📦 加载训练面板 ({yr_start}-{yr_end})...")
    df = load_panel_data(DATA_DIR, list(range(yr_start, yr_end + 1)), file_prefix="train")
    logging.info(f"✅ 基础面板: {df.shape[0]} 条")
    
    logging.info("📐 计算真实日收益率与收盘价...")
    df = compute_real_returns(RAW_PANEL, df)

    factor_cols = [c for c in df.columns if c.endswith('_MKT_Z') or c.endswith('_IND_Z')]
    logging.info(f"📊 计算 {len(factor_cols)} 个因子 IC/IR...")
    # ic_ir = compute_ic_ir(factor_cols, 'FWD_RET_5D_Z_P01_P99', df)
    # with open(OUT_DIR / 'factor_ic_ir.json', 'w', encoding='utf-8') as f:
    #     json.dump(ic_ir, f, indent=2, ensure_ascii=False)

    exclude = {'S_INFO_WINDCODE', 'TRADE_DT', 'SW_L1_CODE', 'FWD_RET_5D_Z_P01_P99', 
               'FEATURE_MASK', 'BUY_MASK', 'SELL_MASK', 'S_DQ_ADJCLOSE', 'ADJ_CLOSE_RET_1D'}
    feature_cols = [c for c in factor_cols if c not in exclude]
    
    logging.info(f"🚀 启动回测引擎 | 特征数: {len(feature_cols)} | 包含 Baseline: OptSharpe")



if __name__ == "__main__":
    main()