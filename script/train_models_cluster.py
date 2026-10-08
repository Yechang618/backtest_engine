# script/train_models_cluster.py
import sys
import os
import json
import logging
import joblib
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

def train_tree_models(model_name, model_type, X_train, y_train, X_eval=None, y_eval=None):
    eval_set = [(X_eval, y_eval)] if X_eval is not None else None
    if model_type == 'xgb':
        model = xgb.XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0,
                                   early_stopping_rounds=50 if X_eval is not None else None)
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(n_estimators=1000, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)] if X_eval is not None else []
        if X_eval is not None:
            model.fit(X_train, y_train, eval_X=X_eval, eval_y=y_eval, callbacks=callbacks)
        else:
            model.fit(X_train, y_train)
    return model

def main():
    setup_logging()
    cfg = Config()
    
    labels_path = cfg.OUT_DIR / "market_regime_labels.json"
    if not labels_path.exists():
        logging.error(f"❌ 找不到聚类标签文件: {labels_path}，请先运行 cluster_market_regimes.py")
        return
        
    with open(labels_path, 'r', encoding='utf-8') as f:
        regime_labels = json.load(f)
        
    logging.info("📦 加载面板数据并映射 Cluster 标签...")
    df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2027)), file_prefix="train", load_train=True, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE')
    df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-06-01')) & (df['TRADE_DT'] <= pd.to_datetime('2026-03-31'))].copy()
    
    df['YEAR_MONTH'] = df['TRADE_DT'].dt.strftime('%Y-%m')
    df['CLUSTER_ID'] = df['YEAR_MONTH'].map(regime_labels)
    df = df.dropna(subset=['CLUSTER_ID'])
    
    feature_cols = extract_valid_features(df)
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    train_df = df[(df['FEATURE_MASK'] == 1)].dropna(subset=[label_col] + feature_cols)
    
    train_period = train_df[train_df['TRADE_DT'] <= '2024-12-31']
    val_period = train_df[(train_df['TRADE_DT'] >= '2025-01-01') & (train_df['TRADE_DT'] <= '2026-03-31')]
    
    cluster_ids = sorted(train_period['CLUSTER_ID'].unique())
    trainers_cluster = {}
    
    for c_id in cluster_ids:
        logging.info(f"🚀 开始训练 Cluster {c_id} 的模型...")
        
        # 训练集
        tr_subset = train_period[train_period['CLUSTER_ID'] == c_id]
        X_tr, y_tr = tr_subset[feature_cols], tr_subset[label_col]
        
        # 验证集
        vl_subset = val_period[val_period['CLUSTER_ID'] == c_id]
        X_ev, y_ev = None, None
        if len(vl_subset) >= 500:
            X_ev, y_ev = vl_subset[feature_cols], vl_subset[label_col]
            logging.info(f"  ✅ 验证集已启用 | 样本数: {len(X_ev)}")
        else:
            logging.warning(f"  ⚠️ 验证集样本不足 ({len(vl_subset)})，不使用早停。")
            
        logging.info(f"  训练集样本数: {len(X_tr)}")
        
        # 训练 LGBM 和 XGB
        lgbm_name = f"LGBM_c{c_id + 1}"
        xgb_name = f"XGB_c{c_id + 1}"
        
        trainers_cluster[lgbm_name] = train_tree_models(lgbm_name, 'lgbm', X_tr, y_tr, X_ev, y_ev)
        trainers_cluster[xgb_name] = train_tree_models(xgb_name, 'xgb', X_tr, y_tr, X_ev, y_ev)
        
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    save_path = os.path.join(cfg.MODEL_DIR, "cluster_sklearn_models.pkl")
    joblib.dump(trainers_cluster, save_path)
    logging.info(f"✅ 所有 Cluster 模型已保存至: {save_path}")

if __name__ == "__main__":
    main()