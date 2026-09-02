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
    def __init__(self, df: pd.DataFrame, config, trainers: dict, label_col: str = 'label_1', ablation=False, trade_pools: dict = None):
        self.df = df[df['FEATURE_MASK'] == 1].copy()
        self.cfg = config
        self.ablation = ablation
        self.trade_pools = trade_pools
        
        # 🔑 特征集初始化
        self.feature_cols = config.FEATURE_COLS  # 全量特征集 (用于没有专属特征的模型)
        if self.ablation:
            raw_selected = getattr(config, 'FEATURE_SELECTED', {})
            if isinstance(raw_selected, dict):
                self.feature_cols_ablation = raw_selected
                logging.info(f"✅ 消融模式已启用，成功加载 {len(self.feature_cols_ablation)} 个模型的专属特征集。")
            else:
                self.feature_cols_ablation = {}
        else:
            self.feature_cols_ablation = {}
            
        print(f"🔧 BacktestEngine 初始化 | 样本数: {len(self.df)} | 消融模式: {self.ablation} | 标签列: {label_col}")
        self.label_col = label_col
        self.portfolios = {m: PortfolioManager(config.INITIAL_CAPITAL, config.COMMISSION_RATE) for m in config.MODELS}
        self.returns_history = defaultdict(list)
        self.trainers = trainers
        self._baseline_init = False
        
        # 🔑 核心修复：引入独立字典，为每个模型单独缓存对齐后的特征集
        self.aligned_features = {} 
        
        # 历史价格缓存 (用于 Residual)
        self.adjclose_history = defaultdict(list)
        # SensitiveSwitch 状态与 Residual 缓存
        self.residual_cache = {}
        self.sensitive_current_model = getattr(config, 'SENSITIVE_SWITCH_INIT_MODEL', 'LGBM-24')
        self.sensitive_switch_ic_history = {m: [] for m in getattr(config, 'SENSITIVE_SWITCH_BASE_MODELS', [])}
        self.daily_resid_ic_results = {}
        self.sensitive_switch_history = []
        # DynamicSwitch 状态
        self.daily_ic_results = {m: [] for m in self.cfg.MODELS}
        self.dynamic_current_model = getattr(config, 'DYNAMIC_SWITCH_INIT_MODEL', 'OptSharpe')
        self.dynamic_ic_history = {m: [] for m in getattr(config, 'DYNAMIC_SWITCH_BASE_MODELS', [])}
        self.dynamic_b = getattr(config, 'DYNAMIC_SWITCH_B', 1.05)
        self.dynamic_switch_history = []
        # 无未来函数误差结算
        self.mse_results = {m: [] for m in self.cfg.MODELS}
        self.prediction_cache = defaultdict(dict)
        
        self._check_feature_alignment()
        if self.trade_pools:
            logging.info(f"🚫 启用股票池硬约束 | 覆盖月份数: {len(self.trade_pools)}")
        logging.info(f"BacktestEngine 初始化完成 | 模型: {list(self.trainers.keys())} | 样本数: {len(self.df)}")

    def _get_features_for_model(self, model_name: str) -> List[str]:
        """🔑 特征路由：优先使用独立缓存的对齐特征集，否则回退到默认逻辑"""
        # 1. 优先返回该模型专属的对齐特征集
        if model_name in self.aligned_features:
            return self.aligned_features[model_name]
        
        # 2. 如果没有缓存，使用默认路由逻辑
        if self.ablation and hasattr(self, 'feature_cols_ablation') and model_name in self.feature_cols_ablation:
            return self.feature_cols_ablation[model_name]
        
        # 3. 回退到全量特征集
        return self.feature_cols

    def _check_feature_alignment(self):
        """确保测试集特征顺序/名称与训练时一致，并独立缓存，防止互相覆盖"""
        for name, model in self.trainers.items():
            if hasattr(model, 'feature_names_in_'):
                target_feats = self._get_features_for_model(name)
                trained_features = list(model.feature_names_in_)
                if set(trained_features) != set(target_feats):
                    aligned_feats = [c for c in trained_features if c in target_feats]
                    logger.warning(f"⚠️ {name} 特征不匹配，已自动对齐至 {len(aligned_feats)} 个")
                    # 🔑 核心修复：存入独立字典，绝不覆盖全局的 self.feature_cols
                    self.aligned_features[name] = aligned_feats
                else:
                    self.aligned_features[name] = target_feats
            else:
                # 对于没有 feature_names_in_ 属性的模型，使用默认特征集
                self.aligned_features[name] = self._get_features_for_model(name)
                
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

    def _compute_style_and_industry(self, daily_df: pd.DataFrame):
        style_data = {}
        if 'S_DQ_CAPITAL' in daily_df.columns and 'S_DQ_CLOSE' in daily_df.columns:
            style_data['log_mcap'] = np.log1p(daily_df['S_DQ_CLOSE'] * daily_df['S_DQ_CAPITAL'] / 10000)
        elif 'PV_CAPITAL_LOG_MKT_Z' in daily_df.columns:
            style_data['log_mcap'] = daily_df['PV_CAPITAL_LOG_MKT_Z']
        else:
            style_data['log_mcap'] = np.nan
            
        if 'S_DQ_VOLUME' in daily_df.columns and 'S_DQ_CAPITAL' in daily_df.columns:
            style_data['turnover'] = daily_df['S_DQ_VOLUME'] / daily_df['S_DQ_CAPITAL']
        elif 'S_DQ_VOLUME' in daily_df.columns:
            style_data['turnover'] = daily_df['S_DQ_VOLUME']
        elif 'TURNOVER_SHARE_OF_MARKET_MKT_Z' in daily_df.columns:
            style_data['turnover'] = daily_df['TURNOVER_SHARE_OF_MARKET_MKT_Z']
        else:
            style_data['turnover'] = np.nan
            
        mom20_list, mom60_list, vol20_list, vol60_list = [], [], [], []
        for code in daily_df.index:
            hist = self.adjclose_history.get(code, [])
            if len(hist) >= 20:
                prices = np.array(hist)
                rets = np.diff(prices) / prices[:-1]
                mom20_list.append(prices[-1] / prices[-20] - 1)
                vol20_list.append(np.std(rets[-20:], ddof=0) if len(rets) >= 10 else np.nan)
            else:
                mom20_list.append(np.nan)
                vol20_list.append(np.nan)
            if len(hist) >= 60:
                mom60_list.append(prices[-1] / prices[-60] - 1)
                vol60_list.append(np.std(rets[-60:], ddof=0) if len(rets) >= 20 else np.nan)
            else:
                mom60_list.append(np.nan)
                vol60_list.append(np.nan)
                
        style_data['mom20'] = mom20_list
        style_data['mom60'] = mom60_list
        style_data['vol20'] = vol20_list
        style_data['vol60'] = vol60_list
        style_df = pd.DataFrame(style_data, index=daily_df.index)
        
        if 'SW_L1_NAME' in daily_df.columns:
            industry_col = daily_df['SW_L1_NAME']
        elif 'SW_L1_CODE' in daily_df.columns:
            industry_col = daily_df['SW_L1_CODE']
        else:
            industry_col = pd.Series('missing', index=daily_df.index)
            
        return style_df, industry_col

    def _compute_residuals(self, preds: pd.Series, style_df: pd.DataFrame, industry_col: pd.Series) -> pd.Series:
        df = pd.concat([preds.rename('pred'), style_df, industry_col.rename('ind')], axis=1).dropna()
        if len(df) < 100:
            return pd.Series(-1e6, index=preds.index)
            
        style_cols = ['log_mcap', 'turnover', 'vol20', 'vol60', 'mom20', 'mom60']
        for col in style_cols:
            std = df[col].std(ddof=0)
            if std > 1e-8:
                df[col] = (df[col] - df[col].mean()) / std
            else:
                df[col] = 0.0
                
        df['ind'] = df['ind'].fillna('missing').astype(str)
        dummies = pd.get_dummies(df['ind'], drop_first=True, dtype=float)
        X = pd.concat([df[style_cols], dummies], axis=1).values
        y = df['pred'].values
        
        if X.shape[1] >= len(df) - 5:
            return pd.Series(-1e6, index=preds.index)
            
        X = np.column_stack([np.ones(len(X)), X])
        try:
            XtX = X.T @ X
            Xty = X.T @ y
            beta = np.linalg.solve(XtX + 1e-8 * np.eye(XtX.shape[0]), Xty)
            resid = y - X @ beta
            full_resid = pd.Series(-1e6, index=preds.index)
            full_resid.loc[df.index] = resid
            return full_resid
        except Exception as e:
            logger.warning(f"⚠️ OLS 回归失败: {e}")
            return pd.Series(-1e6, index=preds.index)

    def run(self) -> Dict[str, pd.DataFrame]:
        grouped = self.df.groupby('TRADE_DT')
        dates = sorted(grouped.groups.keys())
        day_cnt = 0
        results = {m: [] for m in self.cfg.MODELS}
        prev_prices = {}
        
        rebalance_dates_set = None
        target_wd = getattr(self.cfg, 'REBALANCE_WEEKDAY', None)
        if target_wd is not None:
            dates_series = pd.Series(dates)
            iso_year = dates_series.dt.isocalendar().year
            iso_week = dates_series.dt.isocalendar().week
            weekday = dates_series.dt.weekday
            df_dates = pd.DataFrame({'date': dates, 'iso_year': iso_year, 'iso_week': iso_week, 'weekday': weekday})
            valid_df = df_dates[df_dates['weekday'] >= target_wd]
            rebalance_dates = valid_df.groupby(['iso_year', 'iso_week'])['date'].min().values
            rebalance_dates_set = set(rebalance_dates)
            wd_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五'}
            logger.info(f"📅 启用按周调仓 | 目标星期: {wd_names.get(target_wd, target_wd)} | 实际调仓日数量: {len(rebalance_dates_set)}")
        else:
            logger.info(f"📅 启用按天数调仓 | 周期: {self.cfg.REBALANCE_DAYS} 天")
            
        logger.info(f"🚀 启动样本外回测 (统一路由竞争版) | 交易日: {len(dates)}")
        
        for date in dates:
            daily = grouped.get_group(date).set_index('S_INFO_WINDCODE').copy()
            Target = 'S_DQ_ADJCLOSE' 
            price_dict = daily[Target].to_dict()
            day_cnt += 1
            
            for code in daily.index:
                price = price_dict[code]
                daily_ret = (price - prev_prices[code]) / prev_prices[code] if code in prev_prices and prev_prices[code] > 1e-6 else 0.0
                self.returns_history[code].append(daily_ret)
                if len(self.returns_history[code]) > 80:
                    self.returns_history[code] = self.returns_history[code][-80:]
                prev_prices[code] = price
                self.adjclose_history[code].append(price)
                if len(self.adjclose_history[code]) > 60:
                    self.adjclose_history[code] = self.adjclose_history[code][-60:]
                    
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
                
            tradable = daily[daily.get('BUY_MASK', 1) == 1].copy()
            if self.trade_pools is not None:
                current_month_str = date.strftime('%Y%m')
                allowed_codes = self.trade_pools.get(current_month_str, set())
                if allowed_codes:
                    tradable = tradable[tradable.index.isin(allowed_codes)]
                else:
                    tradable = pd.DataFrame()
                    
            true_labels = tradable[self.label_col]
            valid_mask = true_labels.notna()
            
            if not tradable.empty:
                style_df, industry_col = self._compute_style_and_industry(tradable)
                true_labels = tradable[self.label_col]
                valid_mask = true_labels.notna()
                
                base_models = set(getattr(self.cfg, 'SENSITIVE_SWITCH_BASE_MODELS', []) + 
                                  getattr(self.cfg, 'DYNAMIC_SWITCH_BASE_MODELS', []))
                for m in base_models:
                    if m not in self.trainers: continue
                    try:
                        feats = self.aligned_features.get(m, self._get_features_for_model(m))
                        preds_raw = pd.Series(self.trainers[m].predict(tradable[feats]), index=tradable.index)
                        resid = self._compute_residuals(preds_raw, style_df, industry_col)
                        self.residual_cache[m] = resid
                        
                        if valid_mask.sum() > 30:
                            valid_resid = resid[valid_mask]
                            valid_resid = valid_resid[valid_resid > -1e5]
                            if len(valid_resid) > 30:
                                ic = true_labels.loc[valid_resid.index].corr(valid_resid, method='spearman')
                                if not np.isnan(ic):
                                    if m in self.sensitive_switch_ic_history:
                                        self.sensitive_switch_ic_history[m].append(ic)
                                        if len(self.sensitive_switch_ic_history[m]) > 10:
                                            self.sensitive_switch_ic_history[m] = self.sensitive_switch_ic_history[m][-10:]
                                    if m not in self.daily_resid_ic_results:
                                        self.daily_resid_ic_results[m] = []
                                    self.daily_resid_ic_results[m].append({'TRADE_DT': date, 'IC': ic})
                    except Exception as e:
                        logger.error(f"❌ {m} Residual 计算失败: {e}")
                        
            if valid_mask.sum() > 30:
                valid_tradable = tradable[valid_mask]
                valid_true_labels = true_labels[valid_mask]
                for m in self.cfg.MODELS:
                    if m in ['BuyAndHoldAll']: continue
                    m_to_calc = self.dynamic_current_model if m == 'DynamicSwitch' else m
                    if m_to_calc == 'OptSharpe':
                        preds = self._calc_opt_sharpe_weights(valid_tradable.index.tolist())
                    elif m_to_calc in self.trainers:
                        try:
                            feats = self.aligned_features.get(m_to_calc, self._get_features_for_model(m_to_calc))
                            preds = self.trainers[m_to_calc].predict(valid_tradable[feats])
                        except: continue
                    else: continue
                    
                    preds_series = pd.Series(preds, index=valid_tradable.index)
                    if preds_series.std() < 1e-10 or valid_true_labels.std() < 1e-10:
                        continue
                    ic = valid_true_labels.corr(preds_series, method='spearman')
                    if not np.isnan(ic):
                        self.daily_ic_results[m].append({'TRADE_DT': date, 'IC': ic})
                        if m in self.dynamic_ic_history:
                            self.dynamic_ic_history[m].append(ic)
                            if len(self.dynamic_ic_history[m]) > 10:
                                self.dynamic_ic_history[m] = self.dynamic_ic_history[m][-10:]
                                
            is_rebalance_day = date in rebalance_dates_set if rebalance_dates_set else ((day_cnt - self.cfg.WARMUP_DAYS) % self.cfg.REBALANCE_DAYS == 0)
            
            if is_rebalance_day:
                if 'DynamicSwitch' in self.cfg.MODELS:
                    avg_ics = {}
                    for m in self.dynamic_ic_history:
                        if len(self.dynamic_ic_history[m]) >= 10:
                            avg_ics[m] = np.mean(self.dynamic_ic_history[m])
                        elif len(self.dynamic_ic_history[m]) > 0:
                            avg_ics[m] = np.mean(self.dynamic_ic_history[m])
                        else:
                            avg_ics[m] = -np.inf
                    if avg_ics:
                        best_model = max(avg_ics, key=avg_ics.get)
                        best_ic = avg_ics[best_model]
                        if len(self.dynamic_ic_history[self.dynamic_current_model]) >= 10:
                            current_ic = np.mean(self.dynamic_ic_history[self.dynamic_current_model])
                            if best_ic > self.dynamic_b * current_ic and best_model != self.dynamic_current_model:
                                logger.info(f"🔄 DynamicSwitch 切换模型: {self.dynamic_current_model} -> {best_model} (Avg IC: {current_ic:.4f} -> {best_ic:.4f})")
                                self.dynamic_current_model = best_model
                                
                if 'SensitiveSwitch' in self.cfg.MODELS:
                    avg_ics = {}
                    for m in self.sensitive_switch_ic_history:
                        if len(self.sensitive_switch_ic_history[m]) >= 5:
                            avg_ics[m] = np.mean(self.sensitive_switch_ic_history[m])
                    if avg_ics:
                        best_model = max(avg_ics, key=avg_ics.get)
                        if best_model != self.sensitive_current_model:
                            logger.info(f"🔄 SensitiveSwitch 切换模型: {self.sensitive_current_model} -> {best_model} (Avg Resid IC: {avg_ics.get(self.sensitive_current_model, 0):.4f} -> {avg_ics[best_model]:.4f})")
                            self.sensitive_current_model = best_model
                            
                if not tradable.empty:
                    for name in self.cfg.MODELS:
                        if name == 'BuyAndHoldAll': continue
                        
                        feats = self.aligned_features.get(name, self._get_features_for_model(name))
                        
                        if name == 'DynamicSwitch':
                            selected_model = self.dynamic_current_model
                            if selected_model == 'OptSharpe':
                                weights = self._calc_opt_sharpe_weights(tradable.index.tolist())
                                top50 = weights.nlargest(self.cfg.TOP_K).index.tolist()
                            elif selected_model in self.trainers:
                                sel_feats = self.aligned_features.get(selected_model, self._get_features_for_model(selected_model))
                                preds = self.trainers[selected_model].predict(tradable[sel_feats])
                                top50 = pd.Series(preds, index=tradable.index).nlargest(self.cfg.TOP_K).index.tolist()
                            else: top50 = []
                            
                        elif name == 'SensitiveSwitch':
                            selected_model = self.sensitive_current_model
                            if selected_model == 'OptSharpe':
                                weights = self._calc_opt_sharpe_weights(tradable.index.tolist())
                                top50 = weights.nlargest(self.cfg.TOP_K).index.tolist()
                            elif selected_model in self.trainers:
                                resid_scores = self.residual_cache.get(selected_model)
                                if resid_scores is not None and not resid_scores.empty:
                                    valid_resid = resid_scores[resid_scores > -1e5]
                                    if len(valid_resid) >= self.cfg.TOP_K:
                                        top50 = valid_resid.nlargest(self.cfg.TOP_K).index.tolist()
                                    else:
                                        top50 = resid_scores.nlargest(self.cfg.TOP_K).index.tolist()
                                else:
                                    top50 = []
                            else:
                                top50 = []
                                
                        elif name == 'OptSharpe':
                            weights = self._calc_opt_sharpe_weights(tradable.index.tolist())
                            top50 = weights.nlargest(self.cfg.TOP_K).index.tolist()
                            
                        else:
                            if name not in self.trainers: continue
                            try:
                                preds = self.trainers[name].predict(tradable[feats])
                                self.prediction_cache[date][name] = dict(zip(tradable.index, preds))
                                top50 = pd.Series(preds, index=tradable.index).nlargest(self.cfg.TOP_K).index.tolist()
                            except Exception as e:
                                logger.error(f"❌ {name} 预测失败: {e}")
                                top50 = []
                                
                        if len(top50) == 0 and name != 'BuyAndHoldAll':
                            if self.trade_pools is not None and not tradable.empty:
                                logger.info(f"ℹ️ {date.strftime('%Y-%m-%d')} | {name} 股票池内无有效标的，空仓")
                            else:
                                logger.warning(f"⚠️ {date.strftime('%Y-%m-%d')} | {name} 未生成有效标的")
                        self.portfolios[name].rebalance(date, top50, price_dict)
                        
            for m in self.cfg.MODELS:
                nav = self.portfolios[m].update_daily(date, price_dict)
                results[m].append({'TRADE_DT': date, 'Value': nav})
                
            if 'DynamicSwitch' in self.cfg.MODELS:
                self.dynamic_switch_history.append({'TRADE_DT': date, 'Model': self.dynamic_current_model})
            if 'SensitiveSwitch' in self.cfg.MODELS:
                self.sensitive_switch_history.append({'TRADE_DT': date, 'Model': self.sensitive_current_model})
                
            settle_idx = day_cnt - 1 - self.cfg.REBALANCE_DAYS
            if settle_idx >= 0:
                pred_date = dates[settle_idx]
                if pred_date in self.prediction_cache:
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
                            self.mse_results[model_name].append({
                                'TRADE_DT': date, 
                                'MSE': float(np.mean(sq_errors)),
                                'MAE': float(np.mean(abs_errors)),
                                'Sample_Count': len(sq_errors)
                            })
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
                feats = self.aligned_features.get(name, self._get_features_for_model(name))
                X_background = self.df[feats].dropna().head(sample_size)
                explainer = shap.TreeExplainer(self.trainers[name])
                shap_values = explainer.shap_values(X_background)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_background, show=False, max_display=20)
                plt.title(f"{name} SHAP Feature Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_summary_{name}.png'), dpi=150)
                plt.close()
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                feature_importance = pd.Series(mean_abs_shap, index=feats).sort_values(ascending=False).head(20)
                import json
                with open(os.path.join(output_dir, f'shap_importance_{name}.json'), 'w') as f:
                    json.dump({str(k): float(v) for k, v in feature_importance.to_dict().items()}, f, indent=4)
                logger.info(f"  ✅ {name} SHAP 分析完成！")
            except Exception as e:
                logger.warning(f"  ⚠️ {name} SHAP 计算失败: {e}")
# import pandas as pd
# import numpy as np
# from typing import List, Dict
# from collections import defaultdict
# import logging
# import os
# from .backtest_core import PortfolioManager

# logger = logging.getLogger(__name__)

# class BacktestEngine:
#     def __init__(self, df: pd.DataFrame, config, trainers: dict, label_col: str = 'label_1', ablation=False, trade_pools: dict = None):
#         self.df = df[df['FEATURE_MASK'] == 1].copy()
#         self.cfg = config
#         self.ablation = ablation
#         self.trade_pools = trade_pools

#         # 🔑 特征集初始化
#         self.feature_cols = config.FEATURE_COLS  # 全量特征集 (用于全量模型)
        
#         if self.ablation:
#             # 安全读取动态注入的字典
#             raw_selected = getattr(config, 'FEATURE_SELECTED', {})
#             if isinstance(raw_selected, dict):
#                 self.feature_cols_ablation = raw_selected
#                 logging.info(f"✅ 消融模式已启用，成功加载 {len(self.feature_cols_ablation)} 个模型的专属特征集。")
#             else:
#                 self.feature_cols_ablation = {}
#                 logger.warning("⚠️ 消融模式已启用，但 Config.FEATURE_SELECTED 不是字典，将回退使用全量特征！")
#         else:
#             self.feature_cols_ablation = {}
            
#         print(f"🔧 BacktestEngine 初始化 | 样本数: {len(self.df)} | 消融模式: {self.ablation} | 标签列: {label_col}")

        
#         # # 🔑 特征集初始化
#         # self.feature_cols = config.FEATURE_COLS  # 全量特征集 (用于全量模型)
#         # if self.ablation:
#         #     # 消融特征集 (字典: {model_name: [features]})
#         #     self.feature_cols_ablation = config.FEATURE_SELECTED if isinstance(config.FEATURE_SELECTED, dict) else {}
            
#         print(f"🔧 BacktestEngine 初始化 | 样本数: {len(self.df)} | 消融模式: {self.ablation} | 标签列: {label_col}")
#         self.label_col = label_col
        
#         self.portfolios = {m: PortfolioManager(config.INITIAL_CAPITAL, config.COMMISSION_RATE) for m in config.MODELS}
#         self.returns_history = defaultdict(list)
#         self.trainers = trainers
#         self._baseline_init = False
        
#         # 历史价格缓存 (用于 Residual)
#         self.adjclose_history = defaultdict(list)
        
#         # SensitiveSwitch 状态与 Residual 缓存
#         self.residual_cache = {}
#         self.sensitive_current_model = getattr(config, 'SENSITIVE_SWITCH_INIT_MODEL', 'LGBM-24')
#         self.sensitive_switch_ic_history = {m: [] for m in getattr(config, 'SENSITIVE_SWITCH_BASE_MODELS', [])}
#         self.daily_resid_ic_results = {}
#         self.sensitive_switch_history = []
        
#         # DynamicSwitch 状态
#         self.daily_ic_results = {m: [] for m in self.cfg.MODELS}
#         self.dynamic_current_model = getattr(config, 'DYNAMIC_SWITCH_INIT_MODEL', 'OptSharpe')
#         self.dynamic_ic_history = {m: [] for m in getattr(config, 'DYNAMIC_SWITCH_BASE_MODELS', [])}
#         self.dynamic_b = getattr(config, 'DYNAMIC_SWITCH_B', 1.05)
#         self.dynamic_switch_history = []
        
#         # 无未来函数误差结算
#         self.mse_results = {m: [] for m in self.cfg.MODELS}
#         self.prediction_cache = defaultdict(dict)
        
#         self._check_feature_alignment()
        
#         if self.trade_pools:
#             logging.info(f"🚫 启用股票池硬约束 | 覆盖月份数: {len(self.trade_pools)}")
#         logging.info(f"BacktestEngine 初始化完成 | 模型: {list(self.trainers.keys())} | 样本数: {len(self.df)}")

#     def _get_features_for_model(self, model_name: str) -> List[str]:
#         """🔑 特征路由：优先使用消融专属特征集，否则使用全量特征集"""
#         if self.ablation and hasattr(self, 'feature_cols_ablation') and model_name in self.feature_cols_ablation:
#             return self.feature_cols_ablation[model_name]
#         return self.feature_cols

#     def _check_feature_alignment(self):
#         """确保测试集特征顺序/名称与训练时一致"""
#         for name, model in self.trainers.items():
#             if hasattr(model, 'feature_names_in_'):
#                 target_feats = self._get_features_for_model(name)
#                 trained_features = list(model.feature_names_in_)
#                 if set(trained_features) != set(target_feats):
#                     aligned_feats = [c for c in trained_features if c in target_feats]
#                     logger.warning(f"⚠️ {name} 特征不匹配，已自动对齐至 {len(aligned_feats)} 个")
#                     # 更新对应的特征集
#                     if self.ablation and hasattr(self, 'feature_cols_ablation') and name in self.feature_cols_ablation:
#                         self.feature_cols_ablation[name] = aligned_feats
#                     else:
#                         self.feature_cols = aligned_feats

#     def _calc_opt_sharpe_weights(self, valid_codes: List[str]) -> pd.Series:
#         eligible = [c for c in valid_codes if len(self.returns_history.get(c, [])) >= 30]
#         if len(eligible) < 10: return pd.Series(0.0, index=valid_codes)
#         try:
#             hist_data = {c: self.returns_history[c][-30:] for c in eligible}
#             ret_df = pd.DataFrame(hist_data)
#             mu, cov_matrix = ret_df.mean(), ret_df.cov()
#             reg = np.eye(len(eligible)) * 1e-4 * np.trace(cov_matrix.values) / len(eligible)
#             raw_w = np.linalg.solve(cov_matrix.values + reg, mu.values)
#             raw_w = np.maximum(raw_w, 0)
#             weights = raw_w / raw_w.sum() if raw_w.sum() > 1e-8 else np.zeros_like(raw_w)
#         except Exception:
#             ret_df = pd.DataFrame({c: self.returns_history[c][-30:] for c in eligible})
#             raw_w = np.maximum(ret_df.mean().values / (ret_df.var().values + 1e-8), 0)
#             weights = raw_w / (raw_w.sum() + 1e-8)
#         full_scores = pd.Series(0.0, index=valid_codes)
#         full_scores[eligible] = weights
#         return full_scores

#     # 🔑 新增：Residual 计算核心模块
#     def _compute_style_and_industry(self, daily_df: pd.DataFrame):
#         """计算 6 个风格变量和行业哑变量，支持智能 Fallback"""
#         style_data = {}
        
#         # 1. log_mcap (市值)
#         if 'S_DQ_CAPITAL' in daily_df.columns and 'S_DQ_CLOSE' in daily_df.columns:
#             style_data['log_mcap'] = np.log1p(daily_df['S_DQ_CLOSE'] * daily_df['S_DQ_CAPITAL'] / 10000)
#         elif 'PV_CAPITAL_LOG_MKT_Z' in daily_df.columns:
#             style_data['log_mcap'] = daily_df['PV_CAPITAL_LOG_MKT_Z']  # Fallback: 使用已标准化的 log_mcap
#         else:
#             style_data['log_mcap'] = np.nan
            
#         # 2. turnover_proxy (换手率代理)
#         if 'S_DQ_VOLUME' in daily_df.columns and 'S_DQ_CAPITAL' in daily_df.columns:
#             style_data['turnover'] = daily_df['S_DQ_VOLUME'] / daily_df['S_DQ_CAPITAL']
#         elif 'S_DQ_VOLUME' in daily_df.columns:
#             style_data['turnover'] = daily_df['S_DQ_VOLUME']  # Fallback
#         elif 'TURNOVER_SHARE_OF_MARKET_MKT_Z' in daily_df.columns:
#             style_data['turnover'] = daily_df['TURNOVER_SHARE_OF_MARKET_MKT_Z']  # Fallback
#         else:
#             style_data['turnover'] = np.nan

#         # 3 & 4. vol20, vol60, mom20, mom60 (从 adjclose_history 计算)
#         mom20_list, mom60_list, vol20_list, vol60_list = [], [], [], []
#         for code in daily_df.index:
#             hist = self.adjclose_history.get(code, [])
#             if len(hist) >= 20:
#                 prices = np.array(hist)
#                 rets = np.diff(prices) / prices[:-1]
#                 mom20_list.append(prices[-1] / prices[-20] - 1)
#                 vol20_list.append(np.std(rets[-20:], ddof=0) if len(rets) >= 10 else np.nan)
#             else:
#                 mom20_list.append(np.nan)
#                 vol20_list.append(np.nan)
                
#             if len(hist) >= 60:
#                 mom60_list.append(prices[-1] / prices[-60] - 1)
#                 vol60_list.append(np.std(rets[-60:], ddof=0) if len(rets) >= 20 else np.nan)
#             else:
#                 mom60_list.append(np.nan)
#                 vol60_list.append(np.nan)
                
#         style_data['mom20'] = mom20_list
#         style_data['mom60'] = mom60_list
#         style_data['vol20'] = vol20_list
#         style_data['vol60'] = vol60_list
        
#         style_df = pd.DataFrame(style_data, index=daily_df.index)
        
#         # 5. 行业变量
#         if 'SW_L1_NAME' in daily_df.columns:
#             industry_col = daily_df['SW_L1_NAME']
#         elif 'SW_L1_CODE' in daily_df.columns:
#             industry_col = daily_df['SW_L1_CODE']
#         else:
#             industry_col = pd.Series('missing', index=daily_df.index)
            
#         return style_df, industry_col

#     def _compute_residuals(self, preds: pd.Series, style_df: pd.DataFrame, industry_col: pd.Series) -> pd.Series:
#         """OLS 回归计算 Residual 分数，缺失值填充 -1e6"""
#         df = pd.concat([preds.rename('pred'), style_df, industry_col.rename('ind')], axis=1).dropna()
        
#         # 条件 1: 有效样本少于 100 不回归
#         if len(df) < 100:
#             return pd.Series(-1e6, index=preds.index)
            
#         # Z-score 风格变量 (ddof=0)
#         style_cols = ['log_mcap', 'turnover', 'vol20', 'vol60', 'mom20', 'mom60']
#         for col in style_cols:
#             std = df[col].std(ddof=0)
#             if std > 1e-8:
#                 df[col] = (df[col] - df[col].mean()) / std
#             else:
#                 df[col] = 0.0
                
#         # 行业哑变量 (drop_first=True 去掉基准行业)
#         df['ind'] = df['ind'].fillna('missing').astype(str)
#         dummies = pd.get_dummies(df['ind'], drop_first=True, dtype=float)
        
#         X = pd.concat([df[style_cols], dummies], axis=1).values
#         y = df['pred'].values
        
#         # 条件 2: 设计矩阵列数 >= 样本数 - 5 不回归
#         if X.shape[1] >= len(df) - 5:
#             return pd.Series(-1e6, index=preds.index)
            
#         # 添加截距项
#         X = np.column_stack([np.ones(len(X)), X])
        
#         try:
#             # 手动 OLS: beta = (X'X)^-1 X'y (加入微小正则化防止奇异)
#             XtX = X.T @ X
#             Xty = X.T @ y
#             beta = np.linalg.solve(XtX + 1e-8 * np.eye(XtX.shape[0]), Xty)
#             resid = y - X @ beta
            
#             # 映射回原始 index，缺失填 -1e6
#             full_resid = pd.Series(-1e6, index=preds.index)
#             full_resid.loc[df.index] = resid
#             return full_resid
#         except Exception as e:
#             logger.warning(f"⚠️ OLS 回归失败: {e}")
#             return pd.Series(-1e6, index=preds.index)
        
#     def run(self) -> Dict[str, pd.DataFrame]:
#         grouped = self.df.groupby('TRADE_DT')
#         dates = sorted(grouped.groups.keys())
#         day_cnt = 0
#         results = {m: [] for m in self.cfg.MODELS}
#         prev_prices = {}
#         # 🔑 新增：计算按周调仓日 (支持节假日自动顺延)
#         rebalance_dates_set = None
#         target_wd = getattr(self.cfg, 'REBALANCE_WEEKDAY', None)
#         if target_wd is not None:
#             dates_series = pd.Series(dates)
#             iso_year = dates_series.dt.isocalendar().year
#             iso_week = dates_series.dt.isocalendar().week
#             weekday = dates_series.dt.weekday
            
#             df_dates = pd.DataFrame({'date': dates, 'iso_year': iso_year, 'iso_week': iso_week, 'weekday': weekday})
#             # 筛选每周中 weekday >= 目标星期 的交易日，取最小值（即第一个满足条件的交易日）
#             valid_df = df_dates[df_dates['weekday'] >= target_wd]
#             rebalance_dates = valid_df.groupby(['iso_year', 'iso_week'])['date'].min().values
#             rebalance_dates_set = set(rebalance_dates)
            
#             wd_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五'}
#             logger.info(f"📅 启用按周调仓 | 目标星期: {wd_names.get(target_wd, target_wd)} | 实际调仓日数量: {len(rebalance_dates_set)}")
#         else:
#             logger.info(f"📅 启用按天数调仓 | 周期: {self.cfg.REBALANCE_DAYS} 天")

#         logger.info(f"🚀 启动样本外回测 (无未来函数误差结算版) | 交易日: {len(dates)}")

#         for date in dates:
#             daily = grouped.get_group(date).set_index('S_INFO_WINDCODE').copy()
#             Target = 'S_DQ_ADJCLOSE' 
#             price_dict = daily[Target].to_dict()
#             day_cnt += 1

#              # 1. 更新收益历史 & 🔑 更新 adjclose_history (用于 Residual)
#             for code in daily.index:
#                 price = price_dict[code]
#                 daily_ret = (price - prev_prices[code]) / prev_prices[code] if code in prev_prices and prev_prices[code] > 1e-6 else 0.0
#                 self.returns_history[code].append(daily_ret)
#                 if len(self.returns_history[code]) > 80:
#                     self.returns_history[code] = self.returns_history[code][-80:]
#                 prev_prices[code] = price

#                 self.adjclose_history[code].append(price)
#                 if len(self.adjclose_history[code]) > 60:
#                     self.adjclose_history[code] = self.adjclose_history[code][-60:]

#             # 2. 基线策略初始化
#             if 'BuyAndHoldAll' in self.cfg.MODELS and not self._baseline_init:
#                 tradable_all = daily[daily.get('BUY_MASK', 1) == 1].index.tolist()
#                 if tradable_all:
#                     self.portfolios['BuyAndHoldAll'].buy_universe_once(date, tradable_all, price_dict)
#                     self._baseline_init = True

#             if day_cnt <= self.cfg.WARMUP_DAYS:
#                 for m in self.cfg.MODELS:
#                     nav = self.portfolios[m].update_daily(date, price_dict)
#                     results[m].append({'TRADE_DT': date, 'Value': nav})
#                 continue

#             # 🔑 核心修改：获取当日可交易股票，并应用股票池硬约束 (对所有模型全局生效)
#             tradable = daily[daily.get('BUY_MASK', 1) == 1].copy()
#             if self.trade_pools is not None:
#                 current_month_str = date.strftime('%Y%m')
#                 allowed_codes = self.trade_pools.get(current_month_str, set())
#                 if allowed_codes:
#                     tradable = tradable[tradable.index.isin(allowed_codes)]
#                 else:
#                     tradable = pd.DataFrame()  # 当月无股票池数据，强制空仓

#             # 🔑 修复：直接使用 isin 过滤。
#             # 即使 allowed_codes 为空集，Pandas 也会返回一个 0 行但保留原 columns 的 DataFrame，
#             # 从而避免 pd.DataFrame() 丢失列信息导致后续 KeyError。
#             tradable = tradable[tradable.index.isin(allowed_codes)]

#             # 🔑 功能一：计算每日截面 Rank IC (使用过滤后的 tradable)
#             true_labels = tradable[self.label_col]
#             valid_mask = true_labels.notna()

#             if not tradable.empty:
#                 style_df, industry_col = self._compute_style_and_industry(tradable)
#                 true_labels = tradable[self.label_col]
#                 valid_mask = true_labels.notna()
                
#                 # 计算所有基础模型的 Residual
#                 base_models = set(getattr(self.cfg, 'SENSITIVE_SWITCH_BASE_MODELS', []) + 
#                                   getattr(self.cfg, 'DYNAMIC_SWITCH_BASE_MODELS', []))
                
#                 for m in base_models:
#                     if m not in self.trainers: continue
#                     try:
#                         feats = self._get_features_for_model(m)
#                         preds_raw = pd.Series(self.trainers[m].predict(tradable[feats]), index=tradable.index)
#                         resid = self._compute_residuals(preds_raw, style_df, industry_col)
#                         self.residual_cache[m] = resid
                        
#                         # 计算 Residual Rank IC (用于 SensitiveSwitch 路由)
#                         if valid_mask.sum() > 30:
#                             valid_resid = resid[valid_mask]
#                             # 排除 -1e6 的无效残差
#                             valid_resid = valid_resid[valid_resid > -1e5]
#                             if len(valid_resid) > 30:
#                                 ic = true_labels.loc[valid_resid.index].corr(valid_resid, method='spearman')
#                                 if not np.isnan(ic):
#                                     if m in self.sensitive_switch_ic_history:
#                                         self.sensitive_switch_ic_history[m].append(ic)
#                                         if len(self.sensitive_switch_ic_history[m]) > 10:
#                                             self.sensitive_switch_ic_history[m] = self.sensitive_switch_ic_history[m][-10:]

#                                 # 🔑 新增：记录用于绘图的 Residual IC (包含日期)
#                                 if m not in self.daily_resid_ic_results:
#                                     self.daily_resid_ic_results[m] = []
#                                 self.daily_resid_ic_results[m].append({'TRADE_DT': date, 'IC': ic})

#                     except Exception as e:
#                         logger.error(f"❌ {m} Residual 计算失败: {e}")

#             if valid_mask.sum() > 30:  # 至少需要30只股票才能计算有效的截面IC
#                 valid_tradable = tradable[valid_mask]
#                 valid_true_labels = true_labels[valid_mask]
                
#                 for m in self.cfg.MODELS:
#                     if m in ['BuyAndHoldAll']: continue
#                     m_to_calc = self.dynamic_current_model if m == 'DynamicSwitch' else m
                    
#                     if m_to_calc == 'OptSharpe':
#                         preds = self._calc_opt_sharpe_weights(valid_tradable.index.tolist())
#                     elif m_to_calc in self.trainers:
#                         try:
#                             feats = self._get_features_for_model(m_to_calc) # 🔑 路由特征
#                             preds = self.trainers[m_to_calc].predict(valid_tradable[feats])
#                         except: continue
#                     else: continue
                        
#                     preds_series = pd.Series(preds, index=valid_tradable.index)

#                     # 🔑 修复：检查是否为常数数组，避免 ConstantInputWarning
#                     if preds_series.std() < 1e-10 or valid_true_labels.std() < 1e-10:
#                         continue  # 跳过常数情况

#                     ic = valid_true_labels.corr(preds_series, method='spearman')
                    
#                     if not np.isnan(ic):
#                         self.daily_ic_results[m].append({'TRADE_DT': date, 'IC': ic})
                        
#                         # 更新动态切换的历史 IC (仅保留最近10天)
#                         if m in self.dynamic_ic_history:
#                             self.dynamic_ic_history[m].append(ic)
#                             if len(self.dynamic_ic_history[m]) > 10:
#                                 self.dynamic_ic_history[m] = self.dynamic_ic_history[m][-10:]

#             # 3. 调仓与预测逻辑 (🔑 仅缓存，不计算误差)
#             # 🔑 修改：使用预计算的调仓日集合进行判断
#             # if rebalance_dates_set is not None:
#             #     is_rebalance_day = date in rebalance_dates_set
#             # else:
#             #     is_rebalance_day = (day_cnt - self.cfg.WARMUP_DAYS) % self.cfg.REBALANCE_DAYS == 0
                
#             is_rebalance_day = date in rebalance_dates_set if rebalance_dates_set else ((day_cnt - self.cfg.WARMUP_DAYS) % self.cfg.REBALANCE_DAYS == 0)
            
#             if is_rebalance_day:
#                 # 🔑 功能二：动态切换逻辑
#                 if 'DynamicSwitch' in self.cfg.MODELS:
#                     avg_ics = {}
#                     for m in self.dynamic_ic_history:
#                         if len(self.dynamic_ic_history[m]) >= 10:
#                             avg_ics[m] = np.mean(self.dynamic_ic_history[m])
#                         elif len(self.dynamic_ic_history[m]) > 0:
#                             avg_ics[m] = np.mean(self.dynamic_ic_history[m]) # 不足10天用现有平均
#                         else:
#                             avg_ics[m] = -np.inf
                    
#                     if avg_ics:
#                         best_model = max(avg_ics, key=avg_ics.get)
#                         best_ic = avg_ics[best_model]
                        
#                         # 只有当当前模型有足够历史数据时才进行比较，防止初始阶段频繁切换
#                         if len(self.dynamic_ic_history[self.dynamic_current_model]) >= 10:
#                             current_ic = np.mean(self.dynamic_ic_history[self.dynamic_current_model])
#                             if best_ic > self.dynamic_b * current_ic and best_model != self.dynamic_current_model:
#                                 logger.info(f"🔄 DynamicSwitch 切换模型: {self.dynamic_current_model} -> {best_model} (Avg IC: {current_ic:.4f} -> {best_ic:.4f})")
#                                 self.dynamic_current_model = best_model

#                 # 🔑 SensitiveSwitch 路由 (基于 Residual Rank IC)
#                 if 'SensitiveSwitch' in self.cfg.MODELS:
#                     avg_ics = {}
#                     for m in self.sensitive_switch_ic_history:
#                         if len(self.sensitive_switch_ic_history[m]) >= 5:  # 至少需要 5 天数据
#                             avg_ics[m] = np.mean(self.sensitive_switch_ic_history[m])
#                     if avg_ics:
#                         best_model = max(avg_ics, key=avg_ics.get)
#                         if best_model != self.sensitive_current_model:
#                             logger.info(f"🔄 SensitiveSwitch 切换模型: {self.sensitive_current_model} -> {best_model} (Avg Resid IC: {avg_ics.get(self.sensitive_current_model, 0):.4f} -> {avg_ics[best_model]:.4f})")
#                             self.sensitive_current_model = best_model

#                 # tradable = daily[daily.get('BUY_MASK', 1) == 1].copy()
#                 if not tradable.empty:
#                     for name in self.cfg.MODELS:
#                         if name == 'BuyAndHoldAll': continue

#                         # 🔑 统一使用路由方法获取特征和预测
#                         feats = self._get_features_for_model(name)
                        
#                         if name == 'DynamicSwitch':
#                             selected_model = self.dynamic_current_model
#                             if selected_model == 'OptSharpe':
#                                 weights = self._calc_opt_sharpe_weights(tradable.index.tolist())
#                                 top50 = weights.nlargest(self.cfg.TOP_K).index.tolist()
#                             elif selected_model in self.trainers:
#                                 sel_feats = self._get_features_for_model(selected_model)
#                                 preds = self.trainers[selected_model].predict(tradable[sel_feats])
#                                 top50 = pd.Series(preds, index=tradable.index).nlargest(self.cfg.TOP_K).index.tolist()
#                             else: top50 = []
                            
#                         # 🔑 核心修复：SensitiveSwitch 专属调仓逻辑
#                         elif name == 'SensitiveSwitch':
#                             selected_model = self.sensitive_current_model
#                             if selected_model == 'OptSharpe':
#                                 weights = self._calc_opt_sharpe_weights(tradable.index.tolist())
#                                 top50 = weights.nlargest(self.cfg.TOP_K).index.tolist()
#                             elif selected_model in self.trainers:
#                                 # 使用 Residual 分数进行排序
#                                 resid_scores = self.residual_cache.get(selected_model)
#                                 if resid_scores is not None and not resid_scores.empty:
#                                     # 过滤掉 -1e6 的无效残差，只取有效分数最高的 Top K
#                                     valid_resid = resid_scores[resid_scores > -1e5]
#                                     if len(valid_resid) >= self.cfg.TOP_K:
#                                         top50 = valid_resid.nlargest(self.cfg.TOP_K).index.tolist()
#                                     else:
#                                         # 如果有效分数不足 Top K，则用原始分数补齐
#                                         top50 = resid_scores.nlargest(self.cfg.TOP_K).index.tolist()
#                                 else:
#                                     top50 = []
#                             else:
#                                 top50 = []

#                         elif name == 'OptSharpe':
#                             weights = self._calc_opt_sharpe_weights(tradable.index.tolist())
#                             top50 = weights.nlargest(self.cfg.TOP_K).index.tolist()
#                         else:
#                             if name not in self.trainers: continue
#                             try:
#                                 preds = self.trainers[name].predict(tradable[feats]) # 🔑 使用路由后的特征
#                                 self.prediction_cache[date][name] = dict(zip(tradable.index, preds))
#                                 top50 = pd.Series(preds, index=tradable.index).nlargest(self.cfg.TOP_K).index.tolist()
#                             except Exception as e:
#                                 logger.error(f"❌ {name} 预测失败: {e}")
#                                 top50 = []
                                
#                         if len(top50) == 0 and name != 'BuyAndHoldAll':
#                             if self.trade_pools is not None and not tradable.empty:
#                                 logger.info(f"ℹ️ {date.strftime('%Y-%m-%d')} | {name} 股票池内无有效标的，空仓")
#                             else:
#                                 logger.warning(f"⚠️ {date.strftime('%Y-%m-%d')} | {name} 未生成有效标的")
                                
#                         self.portfolios[name].rebalance(date, top50, price_dict)

#             # 4. 每日净值计算
#             for m in self.cfg.MODELS:
#                 nav = self.portfolios[m].update_daily(date, price_dict)
#                 results[m].append({'TRADE_DT': date, 'Value': nav})

#             # 🔑 新增：记录当日 DynamicSwitch 使用的模型
#             if 'DynamicSwitch' in self.cfg.MODELS:
#                 self.dynamic_switch_history.append({
#                     'TRADE_DT': date,
#                     'Model': self.dynamic_current_model
#                 })

#             # 🔑 新增：记录当日 SensitiveSwitch 使用的模型
#             if 'SensitiveSwitch' in self.cfg.MODELS:
#                 self.sensitive_switch_history.append({
#                     'TRADE_DT': date,
#                     'Model': self.sensitive_current_model
#                 })

                
#             # 5. 🔑 无未来函数误差结算 (Delayed Realized Error)
#             # 计算需要结算的预测日 (T 日 = 当前 T+i 日 - i 个交易日)
#             settle_idx = day_cnt - 1 - self.cfg.REBALANCE_DAYS
#             if settle_idx >= 0:
#                 pred_date = dates[settle_idx]
#                 if pred_date in self.prediction_cache:
#                     # 此时 pred_date 到 date 的真实收益已经客观实现
#                     # 我们从 self.df 中读取 pred_date 的 label_{i} (它现在代表已实现的历史收益)
#                     pred_day_data = self.df[self.df['TRADE_DT'] == pred_date].set_index('S_INFO_WINDCODE')
                    
#                     for model_name, preds_dict in self.prediction_cache[pred_date].items():
#                         sq_errors = []
#                         abs_errors = []
#                         for code, pred in preds_dict.items():
#                             if code in pred_day_data.index:
#                                 true_label = pred_day_data.loc[code, self.label_col]
#                                 if not np.isnan(true_label):
#                                     sq_errors.append((pred - true_label) ** 2)
#                                     abs_errors.append(abs(pred - true_label))
                        
#                         if sq_errors:
#                             # 误差记录在结算日 (date)，而非预测日 (pred_date)
#                             self.mse_results[model_name].append({
#                                 'TRADE_DT': date, 
#                                 'MSE': float(np.mean(sq_errors)),
#                                 'MAE': float(np.mean(abs_errors)),
#                                 'Sample_Count': len(sq_errors)
#                             })
#                     # 结算完成，释放内存
#                     del self.prediction_cache[pred_date]

#             if day_cnt % 50 == 0:
#                 logger.info(f"📊 进度: {date.strftime('%Y-%m-%d')} | 现金(EN): {self.portfolios['ElasticNet'].cash:,.0f}")

#         return {k: pd.DataFrame(v) for k, v in results.items()}

#     def analyze_shap(self, output_dir: str, sample_size: int = 500):
#         try:
#             import shap
#             import matplotlib.pyplot as plt
#         except ImportError:
#             logger.warning("⚠️ 未找到 'shap' 库，跳过 SHAP 分析。")
#             return

#         logger.info("🔍 开始计算 SHAP 值 (仅限树模型)...")
#         os.makedirs(output_dir, exist_ok=True)
        
#         for name in ['XGBoost', 'LightGBM']:
#             if name not in self.trainers: continue
#             try:
#                 logger.info(f"  正在计算 {name} 的 SHAP 值...")
#                 X_background = self.df[self.feature_cols].dropna().head(sample_size)
#                 explainer = shap.TreeExplainer(self.trainers[name])
#                 shap_values = explainer.shap_values(X_background)
                
#                 plt.figure(figsize=(12, 8))
#                 shap.summary_plot(shap_values, X_background, show=False, max_display=20)
#                 plt.title(f"{name} SHAP Feature Importance")
#                 plt.tight_layout()
#                 plt.savefig(os.path.join(output_dir, f'shap_summary_{name}.png'), dpi=150)
#                 plt.close()
                
#                 mean_abs_shap = np.abs(shap_values).mean(axis=0)
#                 feature_importance = pd.Series(mean_abs_shap, index=self.feature_cols).sort_values(ascending=False).head(20)
#                 import json
#                 with open(os.path.join(output_dir, f'shap_importance_{name}.json'), 'w') as f:
#                     json.dump({str(k): float(v) for k, v in feature_importance.to_dict().items()}, f, indent=4)
#                 logger.info(f"  ✅ {name} SHAP 分析完成！")
#             except Exception as e:
#                 logger.warning(f"  ⚠️ {name} SHAP 计算失败: {e}")