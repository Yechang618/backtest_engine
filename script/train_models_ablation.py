# script/train_models_ablation.py
import sys
import os
import logging
import joblib
import json
import numpy as np
from pathlib import Path
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features, compute_derived_factors
from config.Config import Config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def load_volatility_config(cfg: Config):
    """读取波动率数据，计算 P30/P70 阈值及月份映射"""
    vol_monthly_path = cfg.OUT_DIR / "monthly_volatility.json"
    vol_pct_path = cfg.OUT_DIR / "volatility_percentiles.json"
    
    if not vol_monthly_path.exists() or not vol_pct_path.exists():
        raise FileNotFoundError(
            f"❌ 找不到波动率数据文件，请先运行 calculate_monthly_volatility.py 和 visualize_volatility.py\n"
            f"缺失: {vol_monthly_path} 或 {vol_pct_path}"
        )
        
    with open(vol_monthly_path, 'r', encoding='utf-8') as f:
        monthly_vol = json.load(f)
    with open(vol_pct_path, 'r', encoding='utf-8') as f:
        pct_data = json.load(f)
        
    all_pct = pct_data.get('All', {})
    p30 = float(all_pct.get('p30'))
    p70 = float(all_pct.get('p70'))
    
    if p30 is None or p70 is None:
        raise ValueError("❌ 无法在 volatility_percentiles.json 中找到 'All' 的 p30 或 p70 数据！")
        
    logging.info(f"📊 基于训练期(<=2024-12)计算的波动率阈值 | P30: {p30:.6f} | P70: {p70:.6f}")
    
    all_monthly = monthly_vol.get('All', {})
    month_to_regime = {}
    for month_str, vol in all_monthly.items():
        if vol <= p30: month_to_regime[month_str] = 'low'
        elif vol <= p70: month_to_regime[month_str] = 'mid'
        else: month_to_regime[month_str] = 'high'
            
    return month_to_regime

def train_tree_models(model_name, model_type, X_train, y_train, X_eval=None, y_eval=None):
    """统一封装 XGBoost 和 LightGBM 的训练逻辑，支持验证集早停"""
    eval_set = [(X_eval, y_eval)] if X_eval is not None else None
    
    if model_type == 'xgb':
        model = xgb.XGBRegressor(
            n_estimators=1000, max_depth=5, learning_rate=0.05, 
            random_state=42, verbosity=0,
            early_stopping_rounds=50 if X_eval is not None else None
        )
        model.fit(
            X_train, y_train, 
            eval_set=eval_set, 
            verbose=False
        )
    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(
            n_estimators=1000, max_depth=5, learning_rate=0.05, 
            random_state=42, verbosity=-1
        )
        callbacks = []
        if X_eval is not None:
            callbacks.extend([lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            
        if X_eval is not None:
            model.fit(X_train, y_train, eval_X=X_eval, eval_y=y_eval, callbacks=callbacks)
        else:
            model.fit(X_train, y_train)
    return model

def main():
    setup_logging()
    cfg = Config()
    
    json_path = cfg.ABLATION_FEATURE_JSON
    if not os.path.exists(json_path):
        logging.error(f"❌ 找不到消融特征文件: {json_path}，请先运行 shap_result_visual.py")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        ablation_features = json.load(f)
        
    logging.info(f"📦 加载消融特征配置 | 模型数: {len(ablation_features)}")
    
    logging.info("📦 加载并预处理数据 (扩展至 2026-03-31 以支持波动率验证集)...")
    df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2027)), file_prefix="train", load_train=True, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE') 
    
    # 限制全局时间范围至 2026-03-31
    df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-05-01')) & (df['TRADE_DT'] <= pd.to_datetime('2026-03-31'))].copy()
    
    all_feature_cols = extract_valid_features(df)
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    train_df = df[(df['FEATURE_MASK'] == 1)].dropna(subset=[label_col] + all_feature_cols)
    
    logging.info(f"📊 全局有效样本数: {len(train_df)} | 特征数: {len(all_feature_cols)}")
    
    # 🔑 定义时间窗口划分字典 (用于 -22, -23, -24)
    splits_time = {
        '22': {'train_end': '2022-05-31', 'eval_start': '2022-06-01', 'eval_end': '2024-08-31'},
        '23': {'train_end': '2023-05-31', 'eval_start': '2023-06-01', 'eval_end': '2024-08-31'},
        '24': {'train_end': '2024-08-31', 'eval_start': None, 'eval_end': None}
    }
    
    # 🔑 加载波动率配置 (用于 _low, _mid, _high)
    month_to_regime = load_volatility_config(cfg)
    
    trainers_ablation = {}
    
    for raw_model_name, feats in ablation_features.items():
        # 🔑 核心修复：标准化模型名称，兼容 "-low/-mid/-high" 与 "_low/_mid/_high"
        model_name = raw_model_name.replace('-low', '_low').replace('-mid', '_mid').replace('-high', '_high')
        if model_name != raw_model_name:
            logging.info(f"🔄 模型名称标准化: {raw_model_name} -> {model_name}")
            
        # 过滤出数据集中实际存在的特征
        valid_feats = [f for f in feats if f in all_feature_cols]
        if not valid_feats:
            logging.warning(f"⚠️ {model_name} 的消融特征在数据集中均不存在，跳过。")
            continue
            
        if 'XGB' in model_name:
            model_type = 'xgb'
        elif 'LGBM' in model_name:
            model_type = 'lgbm'
        else:
            logging.warning(f"⚠️ 无法识别 {model_name} 的模型类型，跳过。")
            continue
            
        X_tr, y_tr, X_ev, y_ev = None, None, None, None
        
        # 🔑 核心逻辑：根据标准化后的后缀判断数据划分方式
        if model_name.endswith(('-22', '-23', '-24')):
            suffix = model_name.split('-')[-1]
            if suffix not in splits_time:
                logging.warning(f"⚠️ 无法识别 {model_name} 的时间后缀 {suffix}，跳过。")
                continue
            dates = splits_time[suffix]
            
            train_mask = train_df['TRADE_DT'] <= pd.to_datetime(dates['train_end'])
            X_tr, y_tr = train_df[train_mask][valid_feats], train_df[train_mask][label_col]
            
            if dates['eval_start']:
                eval_mask = (train_df['TRADE_DT'] >= pd.to_datetime(dates['eval_start'])) & \
                            (train_df['TRADE_DT'] <= pd.to_datetime(dates['eval_end']))
                X_ev, y_ev = train_df[eval_mask][valid_feats], train_df[eval_mask][label_col]
                
        elif model_name.endswith(('_low', '_mid', '_high')):
            regime = model_name.split('_')[-1]
            logging.info(f"🔍 处理波动率区间模型: {model_name} | regime={regime}")
            
            # 训练集: <= 2024-12-31 且 VOL_REGIME == regime
            train_df_time = train_df[train_df['TRADE_DT'] <= '2024-12-31'].copy()
            train_df_time['YEAR_MONTH'] = train_df_time['TRADE_DT'].dt.strftime('%Y-%m')
            train_df_time['VOL_REGIME'] = train_df_time['YEAR_MONTH'].map(month_to_regime)
            
            # 🔑 诊断：打印月份分布
            regime_counts = train_df_time['VOL_REGIME'].value_counts()
            logging.info(f"  📊 训练集月份分布: {regime_counts.to_dict()}")
            
            subset_train = train_df_time[train_df_time['VOL_REGIME'] == regime]
            X_tr, y_tr = subset_train[valid_feats], subset_train[label_col]
            logging.info(f"  📊 训练集样本数: {len(X_tr)}")
            
            # 验证集: 2025-01-01 ~ 2026-03-31 且 VOL_REGIME == regime
            val_df_time = train_df[(train_df['TRADE_DT'] >= '2025-01-01') & (train_df['TRADE_DT'] <= '2026-03-31')].copy()
            val_df_time['YEAR_MONTH'] = val_df_time['TRADE_DT'].dt.strftime('%Y-%m')
            val_df_time['VOL_REGIME'] = val_df_time['YEAR_MONTH'].map(month_to_regime)
            
            val_regime_counts = val_df_time['VOL_REGIME'].value_counts()
            logging.info(f"  📊 验证集月份分布: {val_regime_counts.to_dict()}")
            
            subset_val = val_df_time[val_df_time['VOL_REGIME'] == regime]
            
            MIN_VAL_SAMPLES = 500
            if len(subset_val) >= MIN_VAL_SAMPLES:
                X_ev, y_ev = subset_val[valid_feats], subset_val[label_col]
                logging.info(f"  ✅ [{model_name}] 验证集已启用 | 验证样本数: {len(X_ev)}")
            else:
                logging.warning(f"  ⚠️ [{model_name}] 验证集样本不足 ({len(subset_val)} < {MIN_VAL_SAMPLES})，将不使用验证集进行早停。")
        else:
            logging.warning(f"⚠️ 无法识别 {model_name} 的后缀，跳过。")
            continue
        
        # 🔑 防御性检查：训练集为空则跳过
        if X_tr is None or len(X_tr) == 0:
            logging.error(f"❌ {model_name} 训练集为空，跳过训练。请检查月份映射和数据范围。")
            continue
            
        abl_model_name = f"{model_name}_ablation"
        logging.info(f"🚀 训练消融模型 {abl_model_name} | 特征数: {len(valid_feats)} | Train: {len(X_tr)}, Eval: {len(X_ev) if X_ev is not None else 'None'}")
        
        try:
            model = train_tree_models(abl_model_name, model_type, X_tr, y_tr, X_ev, y_ev)
            trainers_ablation[abl_model_name] = model
        except Exception as e:
            logging.error(f"❌ {abl_model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    save_path = os.path.join(cfg.MODEL_DIR, "ablation_sklearn_models.pkl")
    joblib.dump(trainers_ablation, save_path)
    logging.info(f"✅ 所有消融模型已保存至: {save_path}")
    logging.info(f"📦 成功训练的模型: {list(trainers_ablation.keys())}")

if __name__ == "__main__":
    main()