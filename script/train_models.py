# /data/cye_temp/workspace/backtest_engine/script/train_models.py
import sys
import os
import logging
import joblib
import torch
from pathlib import Path
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features, compute_derived_factors
from config.Config import Config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def save_models(trainers, model_dir, ablation=False):
    os.makedirs(model_dir, exist_ok=True)
    sklearn_trainers = {}
    for name, model in trainers.items():
        sklearn_trainers[name] = model
        
    suffix = "ablation_sklearn_models.pkl" if ablation else "sklearn_models.pkl"
    joblib.dump(sklearn_trainers, os.path.join(model_dir, suffix))
    logging.info(f"✅ 模型已保存至: {os.path.join(model_dir, suffix)}")

def train_tree_models(model_name, model_type, X_train, y_train, X_eval=None, y_eval=None):
    """统一封装 XGBoost 和 LightGBM 的训练逻辑，支持验证集早停"""
    eval_set = [(X_eval, y_eval)] if X_eval is not None else None
    
    if model_type == 'xgb':
        model = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)] if eval_set else []
        model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        
    return model

def main():
    setup_logging()
    cfg = Config()

    logging.info("📦 加载并预处理数据...")
    df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2025)), file_prefix="train", load_train=True, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE') 
    
    # 限制全局时间范围
    df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-06-01')) & (df['TRADE_DT'] <= pd.to_datetime('2024-08-31'))].copy()
    
    feature_cols = extract_valid_features(df)
    cfg.FEATURE_COLS = feature_cols
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    
    # 全量有效数据
    train_df = df[(df['FEATURE_MASK'] == 1)].dropna(subset=[label_col] + feature_cols)
    logging.info(f"📊 全局有效样本数: {len(train_df)}")

    # 🔑 定义时间切片配置
    splits = {
        '22': {'train_end': '2022-05-31', 'eval_start': '2022-06-01', 'eval_end': '2024-08-31'},
        '23': {'train_end': '2023-05-31', 'eval_start': '2023-06-01', 'eval_end': '2024-08-31'},
        '24': {'train_end': '2024-08-31', 'eval_start': None, 'eval_end': None}
    }

    # 构建训练任务列表: (模型名, 模型类型, 特征集, 时间切片)
    tasks = []
    for suffix, dates in splits.items():
        tasks.append((f'XGB-{suffix}', 'xgb', cfg.FEATURE_SELECTED_XGB, dates))
        tasks.append((f'LGBM-{suffix}', 'lgbm', cfg.FEATURE_SELECTED_LGBM, dates))

    # ==========================================
    # 1. 训练全量特征模型
    # ==========================================
    trainers = {}
    logging.info("🚀 开始训练全量特征细分模型...")
    
    # 顺便训练 ElasticNet (使用全量数据到 24 年底)
    X_en = train_df[train_df['TRADE_DT'] <= '2024-08-31'][feature_cols]
    y_en = train_df[train_df['TRADE_DT'] <= '2024-08-31'][label_col]
    trainers['ElasticNet'] = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=1000)
    trainers['ElasticNet'].fit(X_en, y_en)

    for model_name, model_type, feats, dates in tasks:
        # 切分训练集
        train_mask = train_df['TRADE_DT'] <= pd.to_datetime(dates['train_end'])
        X_tr, y_tr = train_df[train_mask][feats], train_df[train_mask][label_col]
        
        # 切分验证集
        X_ev, y_ev = None, None
        if dates['eval_start']:
            eval_mask = (train_df['TRADE_DT'] >= pd.to_datetime(dates['eval_start'])) & \
                        (train_df['TRADE_DT'] <= pd.to_datetime(dates['eval_end']))
            X_ev, y_ev = train_df[eval_mask][feats], train_df[eval_mask][label_col]
            
        logging.info(f"  训练 {model_name} | Train: {len(X_tr)}, Eval: {len(X_ev) if X_ev is not None else 'None'}")
        trainers[model_name] = train_tree_models(model_name, model_type, X_tr, y_tr, X_ev, y_ev)

    save_models(trainers, cfg.MODEL_DIR, ablation=False)

    # ==========================================
    # 2. 训练消融实验模型
    # ==========================================
    if cfg.SHAP_ABLATION:
        trainers_ablation = {}
        logging.info("🧪 开始训练消融实验细分模型...")
        
        for model_name, model_type, feats, dates in tasks:
            train_mask = train_df['TRADE_DT'] <= pd.to_datetime(dates['train_end'])
            X_tr, y_tr = train_df[train_mask][feats], train_df[train_mask][label_col]
            
            X_ev, y_ev = None, None
            if dates['eval_start']:
                eval_mask = (train_df['TRADE_DT'] >= pd.to_datetime(dates['eval_start'])) & \
                            (train_df['TRADE_DT'] <= pd.to_datetime(dates['eval_end']))
                X_ev, y_ev = train_df[eval_mask][feats], train_df[eval_mask][label_col]
                
            # 消融模型名称加后缀
            abl_model_name = f"{model_name}_ablation"
            logging.info(f"  训练 {abl_model_name} | Train: {len(X_tr)}, Eval: {len(X_ev) if X_ev is not None else 'None'}")
            trainers_ablation[abl_model_name] = train_tree_models(abl_model_name, model_type, X_tr, y_tr, X_ev, y_ev)

        save_models(trainers_ablation, cfg.MODEL_DIR, ablation=True)

    logging.info("✅ 所有细分模型训练流程全部完成！")

if __name__ == "__main__":
    main()