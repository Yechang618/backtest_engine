# script/train_models_ablation.py
import sys
import os
import logging
import joblib
import json
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
        model = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)] if eval_set else []
        model.fit(X_train, y_train, eval_X=X_eval, eval_y=y_eval, callbacks=callbacks)
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
    
    logging.info("📦 加载并预处理数据...")
    df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2025)), file_prefix="train", load_train=True, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE') 
    df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-06-01')) & (df['TRADE_DT'] <= pd.to_datetime('2024-08-31'))].copy()
    
    all_feature_cols = extract_valid_features(df)
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    train_df = df[(df['FEATURE_MASK'] == 1)].dropna(subset=[label_col] + all_feature_cols)
    
    splits = {
        '22': {'train_end': '2022-05-31', 'eval_start': '2022-06-01', 'eval_end': '2024-08-31'},
        '23': {'train_end': '2023-05-31', 'eval_start': '2023-06-01', 'eval_end': '2024-08-31'},
        '24': {'train_end': '2024-08-31', 'eval_start': None, 'eval_end': None}
    }
    
    trainers_ablation = {}
    
    for model_name, feats in ablation_features.items():
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
            
        suffix = model_name.split('-')[-1] if '-' in model_name else '24'
        if suffix not in splits:
            logging.warning(f"⚠️ 无法识别 {model_name} 的时间后缀 {suffix}，跳过。")
            continue
            
        dates = splits[suffix]
        
        train_mask = train_df['TRADE_DT'] <= pd.to_datetime(dates['train_end'])
        X_tr, y_tr = train_df[train_mask][valid_feats], train_df[train_mask][label_col]
        
        X_ev, y_ev = None, None
        if dates['eval_start']:
            eval_mask = (train_df['TRADE_DT'] >= pd.to_datetime(dates['eval_start'])) & \
                        (train_df['TRADE_DT'] <= pd.to_datetime(dates['eval_end']))
            X_ev, y_ev = train_df[eval_mask][valid_feats], train_df[eval_mask][label_col]
            
        logging.info(f"🚀 训练消融模型 {model_name} | 特征数: {len(valid_feats)} | Train: {len(X_tr)}, Eval: {len(X_ev) if X_ev is not None else 'None'}")
        
        try:
            model = train_tree_models(model_name, model_type, X_tr, y_tr, X_ev, y_ev)
            trainers_ablation[model_name] = model
        except Exception as e:
            logging.error(f"❌ {model_name} 训练失败: {e}")
            
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    save_path = os.path.join(cfg.MODEL_DIR, "ablation_sklearn_models.pkl")
    joblib.dump(trainers_ablation, save_path)
    logging.info(f"✅ 消融模型已保存至: {save_path}")

if __name__ == "__main__":
    main()