# /data/cye_temp/workspace/backtest_engine/script/train_models.py
import sys
import os
import logging
import joblib
# import torch
from pathlib import Path

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

def save_models(trainers, model_dir):
    """自定义保存逻辑：sklearn 用 joblib，PyTorch 用 torch.save"""
    os.makedirs(model_dir, exist_ok=True)
    sklearn_trainers = {}
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
            
    if sklearn_trainers:
        joblib.dump(sklearn_trainers, os.path.join(model_dir, "sklearn_models.pkl"))
    logging.info(f"✅ 所有模型已成功保存至: {model_dir}")

def main():
    setup_logging()
    cfg = Config()
    
    # 1. 仅加载训练集数据
    logging.info("📦 仅加载训练集数据...")
    df = load_panel_data(cfg.DATA_DIR, None, list(range(2016, 2024)), file_prefix="train", load_train=True, load_test=False)
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    feature_cols = extract_valid_features(df)
    cfg.FEATURE_COLS = feature_cols
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    
    logging.info(f"📊 训练集形状: {df.shape} | 特征数: {len(feature_cols)}")

    # 2. 准备训练数据 (严格过滤 FEATURE_MASK 和 NaN)
    train_df = df[(df['FEATURE_MASK'] == 1) & (df['DATA_SOURCE'] == 'train')].dropna(subset=[label_col] + feature_cols)
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

if __name__ == "__main__":
    main()