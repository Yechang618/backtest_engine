# /data/cye_temp/workspace/backtest_engine/script/train_models.py
import sys
import os
import logging
import joblib
# import torch
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features
from config.Config import Config
# from src.transformer_model import PyTorchTabularRegressor, SimpleTabularTransformer
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def save_models(trainers, model_dir, ablation=False):
    """自定义保存逻辑：sklearn 用 joblib，PyTorch 用 torch.save"""
    os.makedirs(model_dir, exist_ok=True)
    sklearn_trainers = {}
    if ablation == False:
        logging.info(f"💾 保存模型至: {model_dir}")
        for name, model in trainers.items():
            if hasattr(model, 'state_dict'): # PyTorch 模型
                torch.save(model.state_dict(), os.path.join(model_dir, f"{name}_state.pt"))
                meta = {
                    'input_dim': model.input_dim, 'hidden_dim': model.hidden_dim,
                    'n_heads': model.n_heads, 'n_layers': model.n_layers,
                    'feature_names': model.feature_names
                }
                joblib.dump(meta, os.path.join(model_dir, f"{name}_meta.pkl"))
            else:
                sklearn_trainers[name] = model
    else:
        logging.info(f"💾 保存消融实验模型至: {model_dir}")
        for name, model in trainers.items():
            sklearn_trainers[name] = model
            
    if sklearn_trainers and ablation == False:
        joblib.dump(sklearn_trainers, os.path.join(model_dir, "sklearn_models.pkl"))
    elif sklearn_trainers and ablation == True:
        joblib.dump(sklearn_trainers, os.path.join(model_dir, "ablation_sklearn_models.pkl"))
    logging.info(f"✅ 所有模型已成功保存至: {model_dir}")

def main():
    setup_logging()
    cfg = Config()
    
    # 1. 仅加载训练集数据
    logging.info("📦 仅加载训练集数据...")
    # df = load_panel_data(cfg.DATA_DIR, None, list(range(2016, 2025)), file_prefix="train", load_train=True, load_test=False)
    df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2025)), file_prefix="train", load_train=True, load_test=True)
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    # Get df from 2016 to 2024
    df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-06-01')) & (df['TRADE_DT'] <= pd.to_datetime('2024-07-31'))].copy()
    feature_cols = extract_valid_features(df)
    cfg.FEATURE_COLS = feature_cols
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    
    logging.info(f"📊 训练集形状: {df.shape} | 特征数: {len(feature_cols)}")

    # 2. 准备训练数据 (严格过滤 FEATURE_MASK 和 NaN)
    # train_df = df[(df['FEATURE_MASK'] == 1) & (df['DATA_SOURCE'] == 'train')].dropna(subset=[label_col] + feature_cols)
    train_df = df[(df['FEATURE_MASK'] == 1) ].dropna(subset=[label_col] + feature_cols)
    print(f"Column names: {train_df.columns.tolist()}")
    X_train, y_train = train_df[feature_cols], train_df[label_col]
    logging.info(f"🔧 有效训练样本数: {len(X_train)}")

    # 3. 初始化并训练模型
    trainers = {}
    
    logging.info("🚀 开始训练 ElasticNet...")
    trainers['ElasticNet'] = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=1000)
    trainers['ElasticNet'].fit(X_train, y_train)

    logging.info("🚀 开始训练 XGBoost...")
    trainers['XGBoost'] = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
    trainers['XGBoost'].fit(X_train, y_train)

    logging.info("🚀 开始训练 LightGBM...")
    trainers['LightGBM'] = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
    trainers['LightGBM'].fit(X_train, y_train)

    # logging.info("🚀 开始训练 Transformer (Tabular)...")
    # # Transformer 训练可能较慢，此处使用轻量级配置
    # trainers['Transformer'] = PyTorchTabularRegressor(
    #     input_dim=len(feature_cols), hidden_dim=128, n_heads=4, n_layers=2, 
    #     epochs=30, batch_size=512, lr=1e-3
    # )
    # trainers['Transformer'].fit(X_train, y_train)

    # 4. 保存模型
    save_models(trainers, cfg.MODEL_DIR)
    logging.info("✅ 模型训练流程全部完成！")

    # 消融实验
    if cfg.SHAP_ABLATION and cfg.FEATURE_SELECTED:
        trainers_ablation = {}
        logging.info(f"🧪 消融实验模式开启 | 仅训练选定特征: {cfg.FEATURE_SELECTED}")
        ablation_train_df = train_df[cfg.FEATURE_SELECTED + [label_col]]
        X_ablation, y_ablation = ablation_train_df[cfg.FEATURE_SELECTED], ablation_train_df[label_col]
        
        logging.info("🚀 开始训练消融实验模型 (LightGBM)...")
        trainers_ablation['LightGBM'] = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
        trainers_ablation['LightGBM'].fit(X_ablation, y_ablation)
        
        # 保存消融实验模型
        # os.makedirs(cfg.MODEL_DIR, exist_ok=True)
        # joblib.dump(trainers_ablation['LightGBM'], os.path.join(cfg.MODEL_DIR, "ablation_lightgbm.pkl"))
        # logging.info(f"✅ 消融实验模型已保存至: {os.path.join(cfg.MODEL_DIR, 'ablation_lightgbm.pkl')}")

        # XGBoost 消融实验
        logging.info("🚀 开始训练消融实验模型 (XGBoost)...")
        trainers_ablation['XGBoost'] = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
        trainers_ablation['XGBoost'].fit(X_ablation, y_ablation)
        
        # 保存消融实验模型
        # joblib.dump(trainers_ablation['XGBoost'], os.path.join(cfg.MODEL_DIR, "ablation_xgboost.pkl"))
        # logging.info(f"✅ 消融实验模型已保存至: {os.path.join(cfg.MODEL_DIR, 'ablation_xgboost.pkl')}")

        # ElasticNet 消融实验
        logging.info("🚀 开始训练消融实验模型 (ElasticNet)...")
        trainers_ablation['ElasticNet'] = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=1000)
        trainers_ablation['ElasticNet'].fit(X_ablation, y_ablation)

        # 保存消融实验模型
        # joblib.dump(trainers_ablation['ElasticNet'], os.path.join(cfg.MODEL_DIR, "ablation_elasticnet.pkl"))
        # logging.info(f"✅ 消融实验模型已保存至: {os.path.join(cfg.MODEL_DIR, 'ablation_elasticnet.pkl')}")

        # 保存所有消融实验模型
        save_models(trainers_ablation, cfg.MODEL_DIR, ablation=True)

        # 5. 消融实验完成
        logging.info("✅ 消融实验训练流程全部完成！")


if __name__ == "__main__":
    main()