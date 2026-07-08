# /data/cye_temp/workspace/backtest_engine/model/baseline_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from typing import List
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, feature_cols, returns_history: dict):
        self.feature_cols = feature_cols
        self.returns_history = returns_history
        self.trainers = {
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1)
        }

    def train_models(self, X: pd.DataFrame, y: pd.Series, model_names: List[str]):
        valid_X = X[self.feature_cols].dropna()
        valid_y = y.reindex(valid_X.index).dropna()
        if len(valid_X) < 500: return
        
        for name in model_names:
            if name in self.trainers:
                try:
                    self.trainers[name].fit(valid_X, valid_y)
                    logger.info(f"🔧 模型 {name} 训练完成 | 样本: {len(valid_X)}")
                except Exception as e:
                    logger.warning(f"⚠️ {name} 训练失败: {e}")

    def predict_rank(self, model_name: str, X_day: pd.DataFrame) -> pd.Series:
        if model_name == 'OptSharpe':
            return self._calc_opt_sharpe_weights(X_day.index.tolist())
        
        valid = X_day[self.feature_cols].copy()
        preds = self.trainers[model_name].predict(valid)
        return pd.Series(preds, index=valid.index)

    def _calc_opt_sharpe_weights(self, valid_codes: List[str]) -> pd.Series:
        eligible = [c for c in valid_codes if len(self.returns_history.get(c, [])) >= 30]
        if len(eligible) < 10:
            return pd.Series(0.0, index=valid_codes)
            
        try:
            max_len = min(120, max(len(self.returns_history[c]) for c in eligible))
            aligned = {}
            for c in eligible:
                hist = self.returns_history[c][-max_len:]
                aligned[c] = [np.nan]*(max_len-len(hist)) + hist if len(hist) < max_len else hist
            ret_df = pd.DataFrame(aligned)
            mu = ret_df.mean()
            cov = ret_df.cov() + np.eye(len(eligible))*1e-6
            raw_w = np.linalg.solve(cov.values, mu.values)
            raw_w = np.maximum(raw_w, 0)
            weights = raw_w / (raw_w.sum() + 1e-8)
        except Exception:
            ret_df = pd.DataFrame({c: self.returns_history[c][-120:] for c in eligible})
            raw_w = np.maximum(ret_df.mean().values / (ret_df.var().values + 1e-8), 0)
            weights = raw_w / (raw_w.sum() + 1e-8)
            
        full = pd.Series(0.0, index=valid_codes)
        full[eligible] = weights
        return full