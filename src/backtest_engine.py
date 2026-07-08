# /data/cye_temp/workspace/backtest_engine/src/backtest_engine.py
import pandas as pd
import numpy as np
from typing import List, Dict
from collections import defaultdict
import logging
import os
from .backtest_core import PortfolioManager

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, df: pd.DataFrame, config, trainers: dict, label_col: str = 'label_1'):
        self.df = df[df['FEATURE_MASK'] == 1].copy()
        self.cfg = config
        self.feature_cols = config.FEATURE_COLS
        self.label_col = label_col
        self.portfolios = {m: PortfolioManager(config.INITIAL_CAPITAL, config.COMMISSION_RATE) for m in config.MODELS}
        self.returns_history = defaultdict(list)
        self.trainers = trainers
        self._baseline_init = False
        self._check_feature_alignment() # 🔑 新增：特征对齐校验
        logging.info(f"BacktestEngine 初始化完成 (加载预训练模型) | 模型: {list(self.trainers.keys())} | 样本数: {len(self.df)}")

    def _check_feature_alignment(self):
        """确保测试集特征顺序/名称与训练时一致"""
        for name, model in self.trainers.items():
            if hasattr(model, 'feature_names_in_'):
                trained_features = list(model.feature_names_in_)
                if set(trained_features) != set(self.feature_cols):
                    logger.warning(f"⚠️ {name} 训练特征({len(trained_features)})与测试集特征({len(self.feature_cols)})不匹配！")
                    # 自动按训练顺序重排
                    self.feature_cols = [c for c in trained_features if c in self.feature_cols]
                    logger.info(f"✅ 已自动对齐 {name} 特征列顺序 | 最终数量: {len(self.feature_cols)}")
                break

    def _calc_opt_sharpe_weights(self, valid_codes: List[str]) -> pd.Series:
        eligible = [c for c in valid_codes if len(self.returns_history.get(c, [])) >= 30]
        if len(eligible) < 10: return pd.Series(0.0, index=valid_codes)
        try:
            hist_data = {c: self.returns_history[c][-30:] for c in eligible}
            ret_df = pd.DataFrame(hist_data)
            mu, cov_matrix = ret_df.mean(), ret_df.cov()
            reg = np.eye(len(eligible)) * 1e-4 * np.trace(cov_matrix.values) / len(eligible)
            raw_w = np.linalg.solve(cov_matrix.values + reg, mu.values)
            raw_w = np.maximum(raw_w, 0)
            weights = raw_w / raw_w.sum() if raw_w.sum() > 1e-8 else np.zeros_like(raw_w)
        except Exception:
            ret_df = pd.DataFrame({c: self.returns_history[c][-30:] for c in eligible})
            raw_w = np.maximum(ret_df.mean().values / (ret_df.var().values + 1e-8), 0)
            weights = raw_w / (raw_w.sum() + 1e-8)
        full_scores = pd.Series(0.0, index=valid_codes)
        full_scores[eligible] = weights
        return full_scores

    def run(self) -> Dict[str, pd.DataFrame]:
        grouped = self.df.groupby('TRADE_DT')
        dates = sorted(grouped.groups.keys())
        day_cnt = 0
        results = {m: [] for m in self.cfg.MODELS}
        prev_prices = {}
        logger.info(f"🚀 启动样本外回测 | 交易日: {len(dates)} | 模型已冻结")

        for date in dates:
            daily = grouped.get_group(date).set_index('S_INFO_WINDCODE').copy()
            # price_dict: {code: adj_close_price} 用于更新投资组合价值和计算日收益率
            price_dict = daily['S_DQ_ADJCLOSE'].to_dict()
            day_cnt += 1

            for code in daily.index:
                # daily.index 是当日所有股票的代码列表，price_dict 存储了这些股票的收盘价，prev_prices 记录了前一天的收盘价以计算日收益率
                # returns_history 维护了每只股票最近 80 个交易日的收益率序列，用于 OptSharpe 模型的权重计算
                price = price_dict[code]
                daily_ret = (price - prev_prices[code]) / prev_prices[code] if code in prev_prices and prev_prices[code] > 1e-6 else 0.0
                self.returns_history[code].append(daily_ret)
                if len(self.returns_history[code]) > 80:
                    self.returns_history[code] = self.returns_history[code][-80:]
                prev_prices[code] = price

            if 'BuyAndHoldAll' in self.cfg.MODELS and not self._baseline_init:
                tradable_all = daily[daily.get('BUY_MASK', 1) == 1].index.tolist()
                if tradable_all:
                    self.portfolios['BuyAndHoldAll'].buy_universe_once(date, tradable_all, price_dict)
                    self._baseline_init = True

            # 🔑 修复：使用 <= 避免负数取模问题，且确保超过预热期才调仓
            if day_cnt <= self.cfg.WARMUP_DAYS:
                for m in self.cfg.MODELS:
                    nav = self.portfolios[m].update_daily(date, price_dict)
                    results[m].append({'TRADE_DT': date, 'Value': nav})
                continue

            # 调仓逻辑
            if (day_cnt - self.cfg.WARMUP_DAYS) % self.cfg.REBALANCE_DAYS == 0:
                tradable = daily[daily.get('BUY_MASK', 1) == 1].copy()
                if not tradable.empty:
                    for name in self.cfg.MODELS:
                        if name == 'BuyAndHoldAll': continue
                        if name == 'OptSharpe':
                            weights = self._calc_opt_sharpe_weights(tradable.index.tolist())
                            top50 = weights.nlargest(self.cfg.TOP_K).index.tolist()
                        else:
                            if name not in self.trainers: continue
                            try:
                                preds = self.trainers[name].predict(tradable[self.feature_cols])
                                top50 = pd.Series(preds, index=tradable.index).nlargest(self.cfg.TOP_K).index.tolist()
                            except Exception as e:
                                logger.error(f"❌ {name} 预测失败: {e}")
                                top50 = []
                        
                        if len(top50) == 0:
                            logger.warning(f"⚠️ {date.strftime('%Y-%m-%d')} | {name} 未生成有效 Top50 标的")
                        self.portfolios[name].rebalance(date, top50, price_dict)

            for m in self.cfg.MODELS:
                nav = self.portfolios[m].update_daily(date, price_dict)
                results[m].append({'TRADE_DT': date, 'Value': nav})
                
            if day_cnt % 50 == 0:
                logger.info(f"📊 进度: {date.strftime('%Y-%m-%d')} | 现金(EN): {self.portfolios['ElasticNet'].cash:,.0f}")

        return {k: pd.DataFrame(v) for k, v in results.items()}

    def analyze_shap(self, output_dir: str, sample_size: int = 500):
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("⚠️ 未找到 'shap' 库，跳过 SHAP 分析。")
            return

        logger.info("🔍 开始计算 SHAP 值 (仅限树模型)...")
        os.makedirs(output_dir, exist_ok=True)
        
        for name in ['XGBoost', 'LightGBM']:
            if name not in self.trainers: continue
            try:
                logger.info(f"  正在计算 {name} 的 SHAP 值...")
                X_background = self.df[self.feature_cols].dropna().head(sample_size)
                explainer = shap.TreeExplainer(self.trainers[name])
                shap_values = explainer.shap_values(X_background)
                
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_background, show=False, max_display=20)
                plt.title(f"{name} SHAP Feature Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_summary_{name}.png'), dpi=150)
                plt.close()
                
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                feature_importance = pd.Series(mean_abs_shap, index=self.feature_cols).sort_values(ascending=False).head(20)
                import json
                with open(os.path.join(output_dir, f'shap_importance_{name}.json'), 'w') as f:
                    json.dump({str(k): float(v) for k, v in feature_importance.to_dict().items()}, f, indent=4)
                logger.info(f"  ✅ {name} SHAP 分析完成！")
            except Exception as e:
                logger.warning(f"  ⚠️ {name} SHAP 计算失败: {e}")