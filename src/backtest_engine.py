# /data/cye_temp/workspace/backtest_engine/src/backtest_engine.py
import pandas as pd
import numpy as np
from typing import List, Dict
from collections import defaultdict
import logging
import os
from .backtest_core import PortfolioManager
import joblib
import json

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

        # 🔑 新增：DynamicSwitch2 (IC 加权集成) 状态
        self.dynamic_switch2_base_models = getattr(config, 'DYNAMIC_SWITCH2_BASE_MODELS', [])
        self.dynamic_switch2_current_weights = None
        self.dynamic_switch2_valid_models = []

        # 🔑 新增：DynamicSwitch_IR 状态
        self.dynamic_ir_current_model = getattr(config, 'DYNAMIC_SWITCH_IR_INIT_MODEL', 'LGBM-24')
        
        # 🔑 新增：DynamicSwitch2_IR 状态 (每次调仓动态计算权重，无需保存 current model)
        self.dynamic_switch2_ir_weights = None
        self.dynamic_switch2_ir_valid_models = []

        # 无未来函数误差结算
        self.mse_results = {m: [] for m in self.cfg.MODELS}
        self.prediction_cache = defaultdict(dict)
        
        self._check_feature_alignment()
        if self.trade_pools:
            logging.info(f"🚫 启用股票池硬约束 | 覆盖月份数: {len(self.trade_pools)}")
        logging.info(f"BacktestEngine 初始化完成 | 模型: {list(self.trainers.keys())} | 样本数: {len(self.df)}")

        # 🔑 新增：加载聚类模型与市场状态预测缓存

        self.last_month_amount = 1.0
        self.last_month_mcap = 1.0

        self.cluster_pipeline = None
        if hasattr(config, 'CLUSTER_MODEL_PKL') and os.path.exists(config.CLUSTER_MODEL_PKL):
            self.cluster_pipeline = joblib.load(config.CLUSTER_MODEL_PKL)
            logging.info("✅ 成功加载市场状态聚类模型 (KMeans)")
            
        # self.current_cluster_id = 0  # 默认 Cluster 0
        # self.last_processed_month = None
        # self.monthly_buffer = []    # 用于缓存当月每日数据以计算月度特征

        # 🔑 新增：加载市场状态聚类标签 (直接查表，无需实时计算)
        self.current_cluster_id = 0
        self.last_processed_month = None
        self.month_to_cluster = {}
        
        labels_path = getattr(config, 'MARKET_REGIME_LABELS', None)
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.month_to_cluster = json.load(f)
            logging.info(f"✅ 成功加载市场状态标签 | 覆盖月份数: {len(self.month_to_cluster)}")
        else:
            logging.warning("⚠️ 未找到 market_regime_labels.json，ClusterRegime 策略将默认使用 Cluster 0")

    # def _compute_monthly_features(self, buffer_dfs: list) -> np.ndarray:
    #     """利用缓存的每日数据计算全市场宏观特征"""
    #     if not buffer_dfs or self.cluster_pipeline is None:
    #         return None
            
    #     df_month = pd.concat(buffer_dfs)
    #     if 'daily_ret' not in df_month.columns:
    #         df_month['daily_ret'] = df_month.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].pct_change()
            
    #     # 1. Volatility
    #     vol = df_month.groupby('S_INFO_WINDCODE')['daily_ret'].std().mean()
    #     # 2. Turnover
    #     if 'S_DQ_CAPITAL' in df_month.columns:
    #         turnover = (df_month['S_DQ_VOLUME'] / df_month['S_DQ_CAPITAL']).mean()
    #     else:
    #         turnover = 0.0
    #     # 3. Amount & 4. MCap (这里简化为当月总和，环比需要历史数据，此处用对数代替以保持一致性)
    #     amount = np.log1p(df_month['S_DQ_AMOUNT'].sum()) if 'S_DQ_AMOUNT' in df_month.columns else 0.0
    #     if 'S_DQ_CAPITAL' in df_month.columns:
    #         mcap = np.log1p((df_month['S_DQ_CLOSE'] * df_month['S_DQ_CAPITAL']).sum())
    #     else:
    #         mcap = 0.0

    #     # 注意：训练时使用的是 amount_ret 和 mcap_ret。为了在回测中实时计算，
    #     # 我们维护一个全局的 last_amount 和 last_mcap 来计算环比。
    #     # (为简化代码，此处假设引擎中已有 self.last_month_amount 等变量，或在初始化时从历史数据预热)
    #     # 这里提供一个简化的 fallback：直接使用对数值，因为 Scaler 会处理分布。
    #     # 严格对齐训练集的话，需要在引擎外部预先计算好每月的 ret 并传入。
    #     # 此处为了代码可运行，使用 log(amount) 和 log(mcap) 作为近似。
        
    #     cols = self.cluster_pipeline['cols']
    #     features = np.array([[vol, turnover, amount, mcap]])
        
    #     # 如果训练时用的是 ret，这里需要做差分。
    #     # 为了严谨，我们修改 _compute_monthly_features 以支持差分：
    #     if 'amount_ret' in cols:
    #         current_amount = df_month['S_DQ_AMOUNT'].sum() if 'S_DQ_AMOUNT' in df_month.columns else 0
    #         amount_ret = (current_amount / self.last_month_amount - 1) if self.last_month_amount > 0 else 0
    #         self.last_month_amount = current_amount
    #         features[0, cols.index('amount_ret')] = amount_ret
            
    #     if 'mcap_ret' in cols:
    #         last_date = df_month['TRADE_DT'].max()
    #         current_mcap = (df_month[df_month['TRADE_DT'] == last_date]['S_DQ_CLOSE'] * df_month[df_month['TRADE_DT'] == last_date]['S_DQ_CAPITAL']).sum()
    #         mcap_ret = (current_mcap / self.last_month_mcap - 1) if self.last_month_mcap > 0 else 0
    #         self.last_month_mcap = current_mcap
    #         features[0, cols.index('mcap_ret')] = mcap_ret
            
    #     return features

    def _get_features_for_model(self, model_name: str) -> List[str]:
        """🔑 特征路由：优先使用专属特征集（支持 _ablation 后缀），否则回退到默认逻辑"""
        # 1. 优先检查是否有为该模型专属配置的特征集（直接匹配 model_name，包含 _ablation 后缀）
        if hasattr(self.cfg, 'FEATURE_SELECTED') and isinstance(self.cfg.FEATURE_SELECTED, dict):
            if model_name in self.cfg.FEATURE_SELECTED:
                return self.cfg.FEATURE_SELECTED[model_name]
        
        # 2. 回退到全量特征集 (用于全量模型、Percentile模型等)
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
        print(f"🚀 启动样本外回测 | 样本数: {len(self.df)} | 模型数: {len(self.cfg.MODELS)} | 标签列: {self.label_col}")
        print(f"Models: {self.cfg.MODELS}")
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

            # 🔑 跨月检测：计算上月特征并预测当前 Cluster
            current_month = date.strftime('%Y-%m')
            
            # 🔑 跨月检测：直接从 JSON 标签中获取当月的 Cluster ID
            if current_month != self.last_processed_month:
                self.last_processed_month = current_month
                if current_month in self.month_to_cluster:
                    new_cluster = self.month_to_cluster[current_month]
                    if new_cluster != self.current_cluster_id:
                        self.current_cluster_id = new_cluster
                        logger.info(f"🔄 市场状态更新 | 月份: {current_month} -> Cluster {self.current_cluster_id}")
                else:
                    logger.warning(f"⚠️ 月份 {current_month} 无聚类标签，保持 Cluster {self.current_cluster_id}")

            # current_month = date.strftime('%Y-%m')
            # if self.cluster_pipeline is not None and current_month != self.last_processed_month:
            #     if self.last_processed_month is not None and len(self.monthly_buffer) > 0:
            #         features = self._compute_monthly_features(self.monthly_buffer)
            #         if features is not None:
            #             scaled_feats = self.cluster_pipeline['scaler'].transform(features)
            #             self.current_cluster_id = self.cluster_pipeline['kmeans'].predict(scaled_feats)[0]
            #             logger.info(f"🔄 市场状态切换 | 月份: {self.last_processed_month} -> Cluster {self.current_cluster_id}")
                
            #     self.last_processed_month = current_month
            #     self.monthly_buffer = []
                
            # 将当日数据加入 buffer (保留原始列用于计算)
            # self.monthly_buffer.append(daily.reset_index())
            
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
                
            # tradable = daily[daily.get('BUY_MASK', 1) == 1].copy()
            # if self.trade_pools is not None:
            #     current_month_str = date.strftime('%Y%m')
            #     allowed_codes = self.trade_pools.get(current_month_str, set())
            #     if allowed_codes:
            #         tradable = tradable[tradable.index.isin(allowed_codes)]
            #     else:
            #         tradable = pd.DataFrame()
            # 🔑 核心修改：获取当日可交易股票，并应用股票池硬约束 (对所有模型全局生效)
            tradable = daily[daily.get('BUY_MASK', 1) == 1].copy()
            if self.trade_pools is not None:
                current_month_str = date.strftime('%Y%m')
                allowed_codes = self.trade_pools.get(current_month_str, set())
                
                # 🔑 最终修复：直接使用 isin 过滤，彻底移除 pd.DataFrame() 的错误分支。
                # 即使 allowed_codes 为空集，Pandas 也会返回一个 0 行但保留原 columns 的 DataFrame，
                # 从而完美保留 label_5 等列信息，避免后续 KeyError。
                tradable = tradable[tradable.index.isin(allowed_codes)]
                    
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

                    # 🔑 新增：DynamicSwitch2 的 IC 计算 (使用当前生效的权重计算加权预测)
                    if m == 'DynamicSwitch2':
                        if self.dynamic_switch2_current_weights is not None and self.dynamic_switch2_valid_models:
                            weighted_preds = pd.Series(0.0, index=valid_tradable.index)
                            for i, base_m in enumerate(self.dynamic_switch2_valid_models):
                                if base_m in self.trainers:
                                    feats = self.aligned_features.get(base_m, self._get_features_for_model(base_m))
                                    try:
                                        preds = self.trainers[base_m].predict(valid_tradable[feats])
                                        weighted_preds += (self.dynamic_switch2_current_weights[i] **2) * pd.Series(preds, index=valid_tradable.index)
                                    except:
                                        continue
                            preds_series = weighted_preds
                        else:
                            continue  # 调仓前无权重，跳过 IC 计算      
                    # elif m == 'DynamicSwitch':
                    #     m_to_calc = self.dynamic_current_model
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

                # 🔑 新增：DynamicSwitch2 权重更新逻辑
                if 'DynamicSwitch2' in self.cfg.MODELS:
                    window = getattr(self.cfg, 'DYNAMIC_SWITCH2_WINDOW', 10)
                    base_models = self.dynamic_switch2_base_models
                    ics = []
                    valid_models = []
                    
                    for m in base_models:
                        if m in self.dynamic_ic_history:
                            hist = self.dynamic_ic_history[m]
                            if len(hist) >= window:
                                ics.append(np.mean(hist[-window:]))
                                valid_models.append(m)
                            elif len(hist) > 0:
                                ics.append(np.mean(hist))
                                valid_models.append(m)
                    
                    if valid_models:
                        ics = np.array(ics)
                        # 🔑 核心：平移 IC 确保权重为正 (处理负 IC 模型)
                        min_ic = np.min(ics)
                        if min_ic <= 0:
                            ics = ics - min_ic + 1e-5
                        else:
                            ics = ics + 1e-5
                        
                        total = np.sum(ics)
                        self.dynamic_switch2_current_weights = ics / total if total > 1e-8 else np.ones(len(ics)) / len(ics)
                        self.dynamic_switch2_valid_models = valid_models
                        
                        # 打印权重分配日志
                        weight_dict = {m: f"{w:.3f}" for m, w in zip(valid_models, self.dynamic_switch2_current_weights)}
                        logger.info(f"⚖️ DynamicSwitch2 更新权重 (Window={window}): {weight_dict}")
                    else:
                        self.dynamic_switch2_current_weights = None                                

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

                # # 🔑 新增：DynamicSwitch_IR 切换逻辑 (基于 IR)
                # if 'DynamicSwitch_IR' in self.cfg.MODELS:
                #     irs = {}
                #     for m in self.dynamic_ic_history:
                #         hist = self.dynamic_ic_history[m]
                #         if len(hist) >= 2:
                #             mean_ic = np.mean(hist)
                #             std_ic = np.std(hist)
                #             irs[m] = mean_ic / (std_ic + 1e-8)
                #         elif len(hist) > 0:
                #             irs[m] = np.mean(hist) / 1e-8  # 只有1天数据，IR极大
                #         else:
                #             irs[m] = -np.inf
                    
                #     if irs:
                #         best_model = max(irs, key=irs.get)
                #         best_ir = irs[best_model]
                        
                #         # 计算当前模型的 IR
                #         current_hist = self.dynamic_ic_history.get(self.dynamic_ir_current_model, [])
                #         if len(current_hist) >= 2:
                #             current_ir = np.mean(current_hist) / (np.std(current_hist) + 1e-8)
                #         elif len(current_hist) > 0:
                #             current_ir = np.mean(current_hist) / 1e-8
                #         else:
                #             current_ir = -np.inf
                            
                #         b_ir = getattr(self.cfg, 'DYNAMIC_SWITCH_IR_B', 1.00)
                #         if best_ir > b_ir * current_ir and best_model != self.dynamic_ir_current_model:
                #             logger.info(f"🔄 DynamicSwitch_IR 切换模型: {self.dynamic_ir_current_model} -> {best_model} (IR: {current_ir:.4f} -> {best_ir:.4f})")
                #             self.dynamic_ir_current_model = best_model

                # # 🔑 新增：DynamicSwitch2_IR 权重计算逻辑 (基于 IR 加权)
                # if 'DynamicSwitch2_IR' in self.cfg.MODELS:
                #     irs_2 = {}
                #     base_models_2 = getattr(self.cfg, 'DYNAMIC_SWITCH2_IR_BASE_MODELS', [])
                #     for m in base_models_2:
                #         if m in self.dynamic_ic_history:
                #             hist = self.dynamic_ic_history[m]
                #             if len(hist) >= 2:
                #                 irs_2[m] = np.mean(hist) / (np.std(hist) + 1e-8)
                #             elif len(hist) > 0:
                #                 irs_2[m] = np.mean(hist) / 1e-8
                #             else:
                #                 irs_2[m] = -np.inf
                    
                #     if irs_2:
                #         # 平移归一化，确保权重为正
                #         min_ir = min(irs_2.values())
                #         if min_ir <= 0:
                #             irs_2 = {k: v - min_ir + 1e-5 for k, v in irs_2.items()}
                #         else:
                #             irs_2 = {k: v + 1e-5 for k, v in irs_2.items()}
                        
                #         total_ir = sum(irs_2.values())
                #         self.dynamic_switch2_ir_weights = {k: v / total_ir for k, v in irs_2.items()}
                #         self.dynamic_switch2_ir_valid_models = list(irs_2.keys())
                        
                #         # 打印权重日志
                #         weight_log = {m: f"{w:.3f}" for m, w in self.dynamic_switch2_ir_weights.items()}
                #         logger.info(f"⚖️ DynamicSwitch2_IR 更新权重 (基于IR): {weight_log}")
                #     else:
                #         self.dynamic_switch2_ir_weights = None

                # 🔑 修复版：DynamicSwitch_IR 切换逻辑 (基于 IR)
                if 'DynamicSwitch_IR' in self.cfg.MODELS:
                    irs = {}
                    base_models_ir = getattr(self.cfg, 'DYNAMIC_SWITCH_IR_BASE_MODELS', [])
                    for m in base_models_ir:
                        if m in self.dynamic_ic_history:
                            hist = self.dynamic_ic_history[m]
                            if len(hist) >= 2:
                                mean_ic = np.mean(hist)
                                # 🔑 核心修复：对标准差设置下限 (1e-3)，防止 IR 爆炸到百万级别
                                std_ic = max(np.std(hist), 1e-3) 
                                irs[m] = mean_ic / std_ic
                            elif len(hist) > 0:
                                irs[m] = np.mean(hist) / 1e-3
                            else:
                                irs[m] = -np.inf # 无数据时给予极小值
                    
                    if irs:
                        # 🔑 核心修复：过滤掉 -inf 和 nan，只比较有效 IR
                        valid_irs = {k: v for k, v in irs.items() if np.isfinite(v)}
                        if valid_irs:
                            best_model = max(valid_irs, key=valid_irs.get)
                            best_ir = valid_irs[best_model]
                            
                            current_hist = self.dynamic_ic_history.get(self.dynamic_ir_current_model, [])
                            if len(current_hist) >= 2:
                                current_ir = np.mean(current_hist) / max(np.std(current_hist), 1e-3)
                            elif len(current_hist) > 0:
                                current_ir = np.mean(current_hist) / 1e-3
                            else:
                                current_ir = -np.inf
                                
                            b_ir = getattr(self.cfg, 'DYNAMIC_SWITCH_IR_B', 1.00)
                            # 只有当当前模型也有有效 IR 时才进行比较
                            if np.isfinite(current_ir) and best_ir > b_ir * current_ir and best_model != self.dynamic_ir_current_model:
                                logger.info(f"🔄 DynamicSwitch_IR 切换模型: {self.dynamic_ir_current_model} -> {best_model} (IR: {current_ir:.4f} -> {best_ir:.4f})")
                                self.dynamic_ir_current_model = best_model

                # 🔑 修复版：DynamicSwitch2_IR 权重计算逻辑 (基于 IR 加权)
                if 'DynamicSwitch2_IR' in self.cfg.MODELS:
                    valid_irs_2 = {}
                    base_models_2_ir = getattr(self.cfg, 'DYNAMIC_SWITCH2_IR_BASE_MODELS', [])
                    for m in base_models_2_ir:
                        if m in self.dynamic_ic_history:
                            hist = self.dynamic_ic_history[m]
                            if len(hist) >= 2:
                                mean_ic = np.mean(hist)
                                std_ic = max(np.std(hist), 1e-3) # 🔑 防止 IR 爆炸
                                valid_irs_2[m] = mean_ic / std_ic
                            elif len(hist) > 0:
                                valid_irs_2[m] = np.mean(hist) / 1e-3
                    
                    # 🔑 核心修复：彻底抛弃 -np.inf，只对有效 IR 进行归一化
                    if valid_irs_2:
                        # 过滤掉可能存在的 nan
                        valid_irs_2 = {k: v for k, v in valid_irs_2.items() if np.isfinite(v)}
                        
                        if valid_irs_2:
                            min_ir = min(valid_irs_2.values())
                            # 平移归一化
                            shifted_irs = {k: v - min_ir + 1e-5 for k, v in valid_irs_2.items()}
                            total_ir = sum(shifted_irs.values())
                            
                            # 生成最终权重字典 (包含所有基础模型，无数据的权重为 0)
                            self.dynamic_switch2_ir_weights = {m: 0.0 for m in base_models_2_ir}
                            for k, v in shifted_irs.items():
                                self.dynamic_switch2_ir_weights[k] = v / total_ir
                                
                            self.dynamic_switch2_ir_valid_models = list(valid_irs_2.keys())
                            
                            # 打印权重日志 (只打印权重 > 0 的模型，避免刷屏)
                            weight_log = {m: f"{w:.3f}" for m, w in self.dynamic_switch2_ir_weights.items() if w > 1e-6}
                            logger.info(f"⚖️ DynamicSwitch2_IR 更新权重 (基于IR): {weight_log}")
                        else:
                            self.dynamic_switch2_ir_weights = None
                    else:
                        self.dynamic_switch2_ir_weights = None

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

                        # if name == 'DynamicSwitch':
                            # ... (原有的 DynamicSwitch 逻辑) ...
                            
                        # 🔑 新增：DynamicSwitch2 专属调仓逻辑
                        elif name == 'DynamicSwitch2':
                            if self.dynamic_switch2_current_weights is not None:
                                weighted_preds = pd.Series(0.0, index=tradable.index)
                                for i, base_m in enumerate(self.dynamic_switch2_valid_models):
                                    if base_m in self.trainers:
                                        feats = self.aligned_features.get(base_m, self._get_features_for_model(base_m))
                                        try:
                                            preds = self.trainers[base_m].predict(tradable[feats])
                                            # weighted_preds += self.dynamic_switch2_current_weights[i] * pd.Series(preds, index=tradable.index)
                                            # weighted_preds += (self.dynamic_switch2_current_weights[i] ** .5)  * pd.Series(preds, index=tradable.index)
                                            weighted_preds += (self.dynamic_switch2_current_weights[i] ** 2)  * pd.Series(preds, index=tradable.index)

                                        except Exception as e:
                                            logger.error(f"❌ DynamicSwitch2 获取 {base_m} 预测失败: {e}")
                                top50 = weighted_preds.nlargest(self.cfg.TOP_K).index.tolist()
                            else:
                                top50 = []
                                
                        # ... [保留原有的 DynamicSwitch, DynamicSwitch2, OptSharpe 逻辑] ...
                        
                        # 🔑 新增：DynamicSwitch_IR 专属调仓逻辑
                        elif name == 'DynamicSwitch_IR':
                            selected_model = self.dynamic_ir_current_model
                            if selected_model == 'OptSharpe':
                                weights = self._calc_opt_sharpe_weights(tradable.index.tolist())
                                top50 = weights.nlargest(self.cfg.TOP_K).index.tolist()
                            elif selected_model in self.trainers:
                                sel_feats = self.aligned_features.get(selected_model, self._get_features_for_model(selected_model))
                                preds = self.trainers[selected_model].predict(tradable[sel_feats])
                                top50 = pd.Series(preds, index=tradable.index).nlargest(self.cfg.TOP_K).index.tolist()
                            else: top50 = []
                            
                        # 🔑 新增：DynamicSwitch2_IR 专属调仓逻辑
                        elif name == 'DynamicSwitch2_IR':
                            if self.dynamic_switch2_ir_weights is not None:
                                weighted_preds = pd.Series(0.0, index=tradable.index)
                                for base_m, weight in self.dynamic_switch2_ir_weights.items():
                                    if base_m in self.trainers:
                                        feats = self.aligned_features.get(base_m, self._get_features_for_model(base_m))
                                        try:
                                            preds = self.trainers[base_m].predict(tradable[feats])
                                            weighted_preds += weight * pd.Series(preds, index=tradable.index)
                                        except Exception as e:
                                            logger.error(f"❌ DynamicSwitch2_IR 获取 {base_m} 预测失败: {e}")
                                top50 = weighted_preds.nlargest(self.cfg.TOP_K).index.tolist()
                            else:
                                top50 = []
                                
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

                        # 🔑 新增：ClusterRegime 专属调仓逻辑
                        elif name == 'ClusterRegime':
                            target_model_name = f'LGBM_c{self.current_cluster_id + 1}'
                            if target_model_name in self.trainers:
                                feats = self.aligned_features.get(target_model_name, self._get_features_for_model(target_model_name))
                                preds = self.trainers[target_model_name].predict(tradable[feats])
                                top50 = pd.Series(preds, index=tradable.index).nlargest(self.cfg.TOP_K).index.tolist()
                            else:
                                logger.warning(f"⚠️ 未找到 Cluster {self.current_cluster_id} 对应的模型 {target_model_name}")
                                top50 = []
                            
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
