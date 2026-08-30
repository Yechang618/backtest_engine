# script/train_models_percentile.py
import sys
import os
import json
import logging
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features, compute_derived_factors
from config.Config import Config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def get_volatility_config(cfg: Config):
    """
    读取波动率数据，基于训练期(<=2024-12-31)计算 P30/P70 阈值，
    并返回全量月份波动率字典以供后续划分 Regime。
    """
    vol_monthly_path = cfg.OUT_DIR / "monthly_volatility.json"
    if not vol_monthly_path.exists():
        raise FileNotFoundError(f"❌ 找不到波动率数据文件: {vol_monthly_path}，请先运行 calculate_monthly_volatility.py")
        
    with open(vol_monthly_path, 'r', encoding='utf-8') as f:
        monthly_vol = json.load(f)
        
    all_monthly = monthly_vol.get('All', {})
    
    # 🔑 严谨性修复：仅使用训练期 (<=2024-12) 的数据计算 P30 和 P70，防止未来数据污染阈值
    train_period_vols = [vol for m, vol in all_monthly.items() if m <= '2024-12']
    if len(train_period_vols) < 10:
        raise ValueError("❌ 训练期波动率数据不足，无法计算分位数阈值！")
        
    p30 = float(np.percentile(train_period_vols, 30))
    p70 = float(np.percentile(train_period_vols, 70))
    
    logging.info(f"📊 基于训练期(<=2024-12)计算的波动率阈值 | P30: {p30:.6f} | P70: {p70:.6f}")
    return p30, p70, all_monthly

def assign_regime(month_str: str, p30: float, p70: float, all_monthly: dict):
    """根据固定阈值和全量月份字典，返回该月份的 Regime"""
    vol = all_monthly.get(month_str)
    if vol is None:
        return None
    if vol <= p30:
        return 'low'
    elif vol <= p70:
        return 'mid'
    else:
        return 'high'

def train_tree_models(model_name, model_type, X_train, y_train, X_eval=None, y_eval=None):
    """统一封装 XGBoost 和 LightGBM 的训练逻辑，支持验证集早停 (Early Stopping)"""
    eval_set = [(X_eval, y_eval)] if X_eval is not None else None
    
    if model_type == 'xgb':
        model = xgb.XGBRegressor(
            n_estimators=1000, max_depth=5, learning_rate=0.05, 
            random_state=42, verbosity=0
        )
        # 🔑 启用 Early Stopping
        model.fit(
            X_train, y_train, 
            eval_set=eval_set, 
            early_stopping_rounds=50 if eval_set else None, 
            verbose=False
        )
    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(
            n_estimators=1000, max_depth=5, learning_rate=0.05, 
            random_state=42, verbosity=-1
        )
        callbacks = []
        if eval_set:
            callbacks.extend([lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            
        model.fit(
            X_train, y_train, 
            eval_set=eval_set, 
            callbacks=callbacks
        )
    return model

def main():
    setup_logging()
    cfg = Config()
    
    logging.info("📦 加载并预处理面板数据 (扩展至 2026-03-31)...")
    df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2027)), file_prefix="train", load_train=True, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE') 
    
    # 🔑 扩展时间范围至 2026-03-31，以包含验证集
    df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-05-01')) & (df['TRADE_DT'] <= pd.to_datetime('2026-03-31'))].copy()
    
    feature_cols = extract_valid_features(df)
    cfg.FEATURE_COLS = feature_cols
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    
    # 过滤有效样本
    df = df[(df['FEATURE_MASK'] == 1)].dropna(subset=[label_col] + feature_cols)
    logging.info(f"📊 全局有效样本数: {len(df)} | 特征数: {len(feature_cols)}")
    
    # 🔑 1. 获取波动率配置 (P30, P70 及全量月份字典)
    p30, p70, all_monthly = get_volatility_config(cfg)
    
    # 🔑 2. 划分训练集与验证集
    train_end_date = pd.to_datetime('2024-12-31')
    val_start_date = pd.to_datetime('2025-01-01')
    val_end_date = pd.to_datetime('2026-03-31')
    
    train_df = df[df['TRADE_DT'] <= train_end_date].copy()
    val_df = df[(df['TRADE_DT'] >= val_start_date) & (df['TRADE_DT'] <= val_end_date)].copy()
    
    logging.info(f"📅 训练集范围: ~ {train_end_date.strftime('%Y-%m-%d')} | 样本数: {len(train_df)}")
    logging.info(f"📅 验证集范围: {val_start_date.strftime('%Y-%m-%d')} ~ {val_end_date.strftime('%Y-%m-%d')} | 样本数: {len(val_df)}")
    
    # 🔑 3. 为训练集和验证集打上 Regime 标签
    train_df['YEAR_MONTH'] = train_df['TRADE_DT'].dt.strftime('%Y-%m')
    train_df['VOL_REGIME'] = train_df['YEAR_MONTH'].apply(lambda m: assign_regime(m, p30, p70, all_monthly))
    train_df = train_df.dropna(subset=['VOL_REGIME']) # 剔除无法映射的月份
    
    val_df['YEAR_MONTH'] = val_df['TRADE_DT'].dt.strftime('%Y-%m')
    val_df['VOL_REGIME'] = val_df['YEAR_MONTH'].apply(lambda m: assign_regime(m, p30, p70, all_monthly))
    val_df = val_df.dropna(subset=['VOL_REGIME'])
    
    regimes = ['low', 'mid', 'high']
    trainers = {}
    
    # 🔑 4. 循环训练三个波动率区间的模型
    for regime in regimes:
        logging.info(f"🚀 开始训练 [{regime.upper()}] 波动率区间模型...")
        
        # 准备训练集
        train_subset = train_df[train_df['VOL_REGIME'] == regime]
        X_tr, y_tr = train_subset[feature_cols], train_subset[label_col]
        
        # 准备验证集
        val_subset = val_df[val_df['VOL_REGIME'] == regime]
        X_ev, y_ev = None, None
        
        # 设定验证集最小样本量阈值，防止因样本过少导致早停失效或报错
        MIN_VAL_SAMPLES = 500 
        if len(val_subset) >= MIN_VAL_SAMPLES:
            X_ev, y_ev = val_subset[feature_cols], val_subset[label_col]
            logging.info(f"  ✅ [{regime.upper()}] 验证集已启用 | 验证样本数: {len(X_ev)}")
        else:
            logging.warning(f"  ⚠️ [{regime.upper()}] 验证集样本不足 ({len(val_subset)} < {MIN_VAL_SAMPLES})，将不使用验证集进行早停。")
            
        logging.info(f"  训练集样本数: {len(X_tr)}")
        
        # 训练 XGBoost
        model_name_xgb = f"XGB_{regime}"
        trainers[model_name_xgb] = train_tree_models(model_name_xgb, 'xgb', X_tr, y_tr, X_ev, y_ev)
        
        # 训练 LightGBM
        model_name_lgbm = f"LGBM_{regime}"
        trainers[model_name_lgbm] = train_tree_models(model_name_lgbm, 'lgbm', X_tr, y_tr, X_ev, y_ev)
        
    # 🔑 5. 保存模型
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    save_path = os.path.join(cfg.MODEL_DIR, "percentile_sklearn_models.pkl")
    joblib.dump(trainers, save_path)
    logging.info(f"✅ 所有波动率区间模型已成功保存至: {save_path}")

if __name__ == "__main__":
    main()