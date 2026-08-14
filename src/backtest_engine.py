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
    def __init__(self, df: pd.DataFrame, config, trainers: dict, label_col: str = 'label_1', ablation=False):
        self.df = df[df['FEATURE_MASK'] == 1].copy()
        self.cfg = config
        self.ablation = ablation
        if self.ablation:
            self.feature_cols_lgbm = config.FEATURE_SELECTED_LGBM
            self.feature_cols_xgb = config.FEATURE_SELECTED_XGB
            self.feature_cols = config.FEATURE_SELECTED
        else:
            self.feature_cols_lgbm = config.FEATURE_COLS
            self.feature_cols_xgb = config.FEATURE_COLS
            self.feature_cols = config.FEATURE_COLS
        print(f"🔧 BacktestEngine 初始化 | 样本数: {len(self.df)} | 特征数: {len(self.feature_cols_lgbm)} | 标签列: {label_col}")
        self.label_col = label_col
        self.portfolios = {m: PortfolioManager(config.INITIAL_CAPITAL, config.COMMISSION_RATE) for m in config.MODELS}
        self.returns_history = defaultdict(list)
        self.trainers = trainers
        self._baseline_init = False

        # 🔑 功能一：每日 Rank IC 记录
        self.daily_ic_results = {m: [] for m in self.cfg.MODELS}
        
        # 🔑 功能二：动态切换策略状态
        self.dynamic_current_model = getattr(config, 'DYNAMIC_SWITCH_INIT_MODEL', 'OptSharpe')
        self.dynamic_ic_history = {m: [] for m in getattr(config, 'DYNAMIC_SWITCH_BASE_MODELS', [])}
        self.dynamic_b = getattr(config, 'DYNAMIC_SWITCH_B', 1.05)

        # 🔑 无未来函数设计：预测误差延迟结算机制
        self.mse_results = {m: [] for m in self.cfg.MODELS}
        self.prediction_cache = defaultdict(dict)  # 结构: {pred_date: {model_name: {code: pred_value}}}

        # 🔑 新增：记录 DynamicSwitch 每日使用的模型历史
        self.dynamic_switch_history = []

        self._check_feature_alignment() # 🔑 新增：特征对齐校验
        logging.info(f"BacktestEngine 初始化完成 (加载预训练模型) | 模型: {list(self.trainers.keys())} | 样本数: {len(self.df)}")

    def _check_feature_alignment(self):
        """确保测试集特征顺序/名称与训练时一致"""
        for name, model in self.trainers.items():
            if hasattr(model, 'feature_names_in_'):
                trained_features = list(model.feature_names_in_)
                if name == 'LightGBM' and set(trained_features) != set(self.feature_cols_lgbm):
                    logger.warning(f"⚠️ {name} 训练特征({len(trained_features)})与测试集特征({len(self.feature_cols_lgbm)})不匹配！")
                    # 自动按训练顺序重排
                    self.feature_cols_lgbm = [c for c in trained_features if c in self.feature_cols_lgbm]
                    logger.info(f"✅ 已自动对齐 {name} 特征列顺序 | 最终数量: {len(self.feature_cols_lgbm)}")
                elif name == 'XGBoost' and set(trained_features) != set(self.feature_cols_xgb):
                    logger.warning(f"⚠️ {name} 训练特征({len(trained_features)})与测试集特征({len(self.feature_cols_xgb)})不匹配！")
                    # 自动按训练顺序重排
                    self.feature_cols_xgb = [c for c in trained_features if c in self.feature_cols_xgb]
                    logger.info(f"✅ 已自动对齐 {name} 特征列顺序 | 最终数量: {len(self.feature_cols_xgb)}")
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
        # 🔑 新增：计算按周调仓日 (支持节假日自动顺延)
        rebalance_dates_set = None
        target_wd = getattr(self.cfg, 'REBALANCE_WEEKDAY', None)
        if target_wd is not None:
            dates_series = pd.Series(dates)
            iso_year = dates_series.dt.isocalendar().year
            iso_week = dates_series.dt.isocalendar().week
            weekday = dates_series.dt.weekday
            
            df_dates = pd.DataFrame({'date': dates, 'iso_year': iso_year, 'iso_week': iso_week, 'weekday': weekday})
            # 筛选每周中 weekday >= 目标星期 的交易日，取最小值（即第一个满足条件的交易日）
            valid_df = df_dates[df_dates['weekday'] >= target_wd]
            rebalance_dates = valid_df.groupby(['iso_year', 'iso_week'])['date'].min().values
            rebalance_dates_set = set(rebalance_dates)
            
            wd_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五'}
            logger.info(f"📅 启用按周调仓 | 目标星期: {wd_names.get(target_wd, target_wd)} | 实际调仓日数量: {len(rebalance_dates_set)}")
        else:
            logger.info(f"📅 启用按天数调仓 | 周期: {self.cfg.REBALANCE_DAYS} 天")

        logger.info(f"🚀 启动样本外回测 (无未来函数误差结算版) | 交易日: {len(dates)}")

        for date in dates:
            daily = grouped.get_group(date).set_index('S_INFO_WINDCODE').copy()
            Target = 'S_DQ_ADJCLOSE' 
            price_dict = daily[Target].to_dict()
            day_cnt += 1

            # 1. 更新收益历史 (用于 OptSharpe)
            for code in daily.index:
                price = price_dict[code]
                daily_ret = (price - prev_prices[code]) / prev_prices[code] if code in prev_prices and prev_prices[code] > 1e-6 else 0.0
                self.returns_history[code].append(daily_ret)
                if len(self.returns_history[code]) > 80:
                    self.returns_history[code] = self.returns_history[code][-80:]
                prev_prices[code] = price

            # 2. 基线策略初始化
            if 'BuyAndHoldAll' in self.cfg.MODELS and not self._baseline_init:
                tradable_all = daily[daily.get('BUY_MASK', 1) == 1].index.tolist()
                if tradable_all:
                    self.portfolios['BuyAndHoldAll'].buy_universe_once(date, tradable_all, price_dict)
                    self._baseline_init = True

            if day_cnt <= self.cfg.WARMUP_DAYS:
                for m in self.cfg.MODELS:
                    nav = self.portfolios[m].update_daily(date, price_dict)
                    results[m].append({'TRADE_DT': date, 'Value': nav})
                continue

            # 🔑 功能一：计算每日截面 Rank IC
            tradable = daily[daily.get('BUY_MASK', 1) == 1].copy()
            true_labels = tradable[self.label_col]
            valid_mask = true_labels.notna()
            
            if valid_mask.sum() > 30:  # 至少需要30只股票才能计算有效的截面IC
                valid_tradable = tradable[valid_mask]
                valid_true_labels = true_labels[valid_mask]
                
                for m in self.cfg.MODELS:
                    if m in ['BuyAndHoldAll']: 
                        continue
                    
                    # DynamicSwitch 的 IC 等于其当前底层模型的 IC
                    m_to_calc = self.dynamic_current_model if m == 'DynamicSwitch' else m
                    
                    if m_to_calc == 'OptSharpe':
                        preds = self._calc_opt_sharpe_weights(valid_tradable.index.tolist())
                    elif m_to_calc in self.trainers:
                        try:
                            preds = self.trainers[m_to_calc].predict(valid_tradable[self.feature_cols])
                        except:
                            continue
                    else:
                        continue
                        
                    preds_series = pd.Series(preds, index=valid_tradable.index)

                    # 🔑 修复：检查是否为常数数组，避免 ConstantInputWarning
                    if preds_series.std() < 1e-10 or valid_true_labels.std() < 1e-10:
                        continue  # 跳过常数情况

                    ic = valid_true_labels.corr(preds_series, method='spearman')
                    
                    if not np.isnan(ic):
                        self.daily_ic_results[m].append({'TRADE_DT': date, 'IC': ic})
                        
                        # 更新动态切换的历史 IC (仅保留最近10天)
                        if m in self.dynamic_ic_history:
                            self.dynamic_ic_history[m].append(ic)
                            if len(self.dynamic_ic_history[m]) > 10:
                                self.dynamic_ic_history[m] = self.dynamic_ic_history[m][-10:]

            # 3. 调仓与预测逻辑 (🔑 仅缓存，不计算误差)
            # 🔑 修改：使用预计算的调仓日集合进行判断
            if rebalance_dates_set is not None:
                is_rebalance_day = date in rebalance_dates_set
            else:
                is_rebalance_day = (day_cnt - self.cfg.WARMUP_DAYS) % self.cfg.REBALANCE_DAYS == 0
                
            if is_rebalance_day:
                # 🔑 功能二：动态切换逻辑
                if 'DynamicSwitch' in self.cfg.MODELS:
                    avg_ics = {}
                    for m in self.dynamic_ic_history:
                        if len(self.dynamic_ic_history[m]) >= 10:
                            avg_ics[m] = np.mean(self.dynamic_ic_history[m])
                        elif len(self.dynamic_ic_history[m]) > 0:
                            avg_ics[m] = np.mean(self.dynamic_ic_history[m]) # 不足10天用现有平均
                        else:
                            avg_ics[m] = -np.inf
                    
                    if avg_ics:
                        best_model = max(avg_ics, key=avg_ics.get)
                        best_ic = avg_ics[best_model]
                        
                        # 只有当当前模型有足够历史数据时才进行比较，防止初始阶段频繁切换
                        if len(self.dynamic_ic_history[self.dynamic_current_model]) >= 10:
                            current_ic = np.mean(self.dynamic_ic_history[self.dynamic_current_model])
                            if best_ic > self.dynamic_b * current_ic and best_model != self.dynamic_current_model:
                                logger.info(f"🔄 DynamicSwitch 切换模型: {self.dynamic_current_model} -> {best_model} (Avg IC: {current_ic:.4f} -> {best_ic:.4f})")
                                self.dynamic_current_model = best_model

                tradable = daily[daily.get('BUY_MASK', 1) == 1].copy()
                if not tradable.empty:
                    for name in self.cfg.MODELS:
                        if name == 'BuyAndHoldAll': continue

                        if name == 'DynamicSwitch':
                            selected_model = self.dynamic_current_model
                            if selected_model == 'OptSharpe':
                                weights = self._calc_opt_sharpe_weights(tradable.index.tolist())
                                top50 = weights.nlargest(self.cfg.TOP_K).index.tolist()
                            elif selected_model in self.trainers:
                                if selected_model == 'LightGBM_ablation':
                                    preds = self.trainers[selected_model].predict(tradable[self.feature_cols_lgbm])
                                elif selected_model == 'XGBoost_ablation':
                                    preds = self.trainers[selected_model].predict(tradable[self.feature_cols_xgb])
                                else:
                                    preds = self.trainers[selected_model].predict(tradable[self.feature_cols])
                                top50 = pd.Series(preds, index=tradable.index).nlargest(self.cfg.TOP_K).index.tolist()
                            else:
                                top50 = []
                        elif name == 'OptSharpe':
                            weights = self._calc_opt_sharpe_weights(tradable.index.tolist())
                            top50 = weights.nlargest(self.cfg.TOP_K).index.tolist()
                        else:
                            if name not in self.trainers: continue
                            try:
                                if name == 'LightGBM':
                                    preds = self.trainers[name].predict(tradable[self.feature_cols_lgbm])
                                elif name == 'XGBoost':
                                    preds = self.trainers[name].predict(tradable[self.feature_cols_xgb])
                                else:
                                    preds = self.trainers[name].predict(tradable[self.feature_cols])

                                # 🔑 核心：将预测值缓存，等待 T+i 日结算。此时绝不访问 label_{i}
                                self.prediction_cache[date][name] = dict(zip(tradable.index, preds))
                                
                                top50 = pd.Series(preds, index=tradable.index).nlargest(self.cfg.TOP_K).index.tolist()
                            except Exception as e:
                                logger.error(f"❌ {name} 预测失败: {e}")
                                top50 = []
                                
                        if len(top50) == 0:
                            logger.warning(f"⚠️ {date.strftime('%Y-%m-%d')} | {name} 未生成有效 Top50 标的")
                        self.portfolios[name].rebalance(date, top50, price_dict)

            # 4. 每日净值计算
            for m in self.cfg.MODELS:
                nav = self.portfolios[m].update_daily(date, price_dict)
                results[m].append({'TRADE_DT': date, 'Value': nav})

            # 🔑 新增：记录当日 DynamicSwitch 使用的模型
            if 'DynamicSwitch' in self.cfg.MODELS:
                self.dynamic_switch_history.append({
                    'TRADE_DT': date,
                    'Model': self.dynamic_current_model
                })
                
            # 5. 🔑 无未来函数误差结算 (Delayed Realized Error)
            # 计算需要结算的预测日 (T 日 = 当前 T+i 日 - i 个交易日)
            settle_idx = day_cnt - 1 - self.cfg.REBALANCE_DAYS
            if settle_idx >= 0:
                pred_date = dates[settle_idx]
                if pred_date in self.prediction_cache:
                    # 此时 pred_date 到 date 的真实收益已经客观实现
                    # 我们从 self.df 中读取 pred_date 的 label_{i} (它现在代表已实现的历史收益)
                    pred_day_data = self.df[self.df['TRADE_DT'] == pred_date].set_index('S_INFO_WINDCODE')
                    
                    for model_name, preds_dict in self.prediction_cache[pred_date].items():
                        sq_errors = []
                        abs_errors = []
                        for code, pred in preds_dict.items():
                            if code in pred_day_data.index:
                                true_label = pred_day_data.loc[code, self.label_col]
                                if not np.isnan(true_label):
                                    sq_errors.append((pred - true_label) ** 2)
                                    abs_errors.append(abs(pred - true_label))
                        
                        if sq_errors:
                            # 误差记录在结算日 (date)，而非预测日 (pred_date)
                            self.mse_results[model_name].append({
                                'TRADE_DT': date, 
                                'MSE': float(np.mean(sq_errors)),
                                'MAE': float(np.mean(abs_errors)),
                                'Sample_Count': len(sq_errors)
                            })
                    # 结算完成，释放内存
                    del self.prediction_cache[pred_date]

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