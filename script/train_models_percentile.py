# script/train_models_percentile.py
import sys
import os
import json
import logging
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features, compute_derived_factors
from config.Config import Config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def get_volatility_config(cfg: Config):
    """读取月度波动率数据，基于训练期计算 P33/P67 阈值"""
    vol_monthly_path = cfg.OUT_DIR / "monthly_volatility.json"
    vol_pct_path = cfg.OUT_DIR / "volatility_percentiles.json"
    if not vol_monthly_path.exists() or not vol_pct_path.exists():
        raise FileNotFoundError(f"❌ 找不到波动率数据文件，请先运行 calculate_monthly_volatility.py")
    with open(vol_monthly_path, 'r', encoding='utf-8') as f: monthly_vol = json.load(f)
    with open(vol_pct_path, 'r', encoding='utf-8') as f: pct_data = json.load(f)
    all_pct = pct_data.get('All', {})
    p33 = float(all_pct.get('p33'))
    p67 = float(all_pct.get('p67'))
    logging.info(f"📊 [月度] 基于训练期(<=2024-12)计算的波动率阈值 | P33: {p33:.6f} | P67: {p67:.6f}")
    return p33, p67, monthly_vol.get('All', {})

def calculate_weekly_volatility_regime(df: pd.DataFrame):
    """在内存中计算全市场周度波动率，并基于训练期计算 P33/P67 阈值"""
    logging.info("🧮 开始计算全市场【周度】波动率...")
    df_calc = df[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJCLOSE']].copy().sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
    df_calc['daily_ret'] = df_calc.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].pct_change()
    df_calc['YEAR'] = df_calc['TRADE_DT'].dt.isocalendar().year.astype(str)
    df_calc['WEEK'] = df_calc['TRADE_DT'].dt.isocalendar().week.astype(str).str.zfill(2)
    df_calc['YW'] = df_calc['YEAR'] + '-' + df_calc['WEEK']
    df_calc['WEEK_END_DATE'] = df_calc.groupby(['S_INFO_WINDCODE', 'YW'])['TRADE_DT'].transform('max')
    
    stock_weekly_vol = df_calc.groupby(['S_INFO_WINDCODE', 'YW', 'WEEK_END_DATE'])['daily_ret'].std().reset_index()
    # market_weekly_vol = stock_weekly_vol.groupby(['YW', 'WEEK_END_DATE'])['daily_ret'].mean().reset_index()
    # market_weekly_vol.rename(columns={'daily_ret': 'VOLATILITY'}, inplace=True).dropna(subset=['VOLATILITY'], inplace=True)
    
    # 5. 计算全市场每周的平均波动率
    market_weekly_vol = stock_weekly_vol.groupby(['YW', 'WEEK_END_DATE'])['daily_ret'].mean().reset_index()
    
    # 🔑 核心修复：移除 inplace=True，让 rename 返回 DataFrame 以便继续链式调用
    market_weekly_vol = market_weekly_vol.rename(columns={'daily_ret': 'VOLATILITY'}).dropna(subset=['VOLATILITY']).sort_values('WEEK_END_DATE')

    train_weekly = market_weekly_vol[market_weekly_vol['WEEK_END_DATE'] <= '2024-12-31']
    p33_w = float(np.percentile(train_weekly['VOLATILITY'], 33))
    p67_w = float(np.percentile(train_weekly['VOLATILITY'], 67))
    logging.info(f"📊 [周度] 基于训练期(<=2024-12)计算的波动率阈值 | P33_w: {p33_w:.6f} | P67_w: {p67_w:.6f}")
    
    week_to_regime = {}
    for _, row in market_weekly_vol.iterrows():
        vol = row['VOLATILITY']
        if vol <= p33_w: week_to_regime[row['YW']] = 'wLow'
        elif vol <= p67_w: week_to_regime[row['YW']] = 'wMid'
        else: week_to_regime[row['YW']] = 'wHigh'
    return week_to_regime

def train_tree_models(model_name, model_type, X_train, y_train, X_eval=None, y_eval=None):
    eval_set = [(X_eval, y_eval)] if X_eval is not None else None
    if model_type == 'xgb':
        model = xgb.XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0,
                                   early_stopping_rounds=50 if X_eval is not None else None)
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(n_estimators=1000, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)] if X_eval is not None else []
        model.fit(X_train, y_train, eval_X=X_eval, eval_y=y_eval, callbacks=callbacks) if X_eval is not None else model.fit(X_train, y_train)
    return model

def main():
    setup_logging()
    cfg = Config()
    logging.info("📦 加载并预处理面板数据...")
    df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2027)), file_prefix="train", load_train=True, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE') 
    df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-05-01')) & (df['TRADE_DT'] <= pd.to_datetime('2026-03-31'))].copy()
    
    feature_cols = extract_valid_features(df)
    cfg.FEATURE_COLS = feature_cols
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    df = df[(df['FEATURE_MASK'] == 1)].dropna(subset=[label_col] + feature_cols)
    
    p33_m, p67_m, monthly_vol_dict = get_volatility_config(cfg)
    week_to_regime_w = calculate_weekly_volatility_regime(df)
    
    train_end_date, val_start_date, val_end_date = pd.to_datetime('2024-12-31'), pd.to_datetime('2025-01-01'), pd.to_datetime('2026-03-31')
    train_df = df[df['TRADE_DT'] <= train_end_date].copy()
    val_df = df[(df['TRADE_DT'] >= val_start_date) & (df['TRADE_DT'] <= val_end_date)].copy()
    
    for target_df in [train_df, val_df]:
        target_df['YEAR_MONTH'] = target_df['TRADE_DT'].dt.strftime('%Y-%m')
        target_df['YEAR'] = target_df['TRADE_DT'].dt.isocalendar().year.astype(str)
        target_df['WEEK'] = target_df['TRADE_DT'].dt.isocalendar().week.astype(str).str.zfill(2)
        target_df['YW'] = target_df['YEAR'] + '-' + target_df['WEEK']
        
        # 月度标签
        target_df['M_REGIME'] = target_df['YEAR_MONTH'].map(lambda m: 'low' if monthly_vol_dict.get(m, 0) <= p33_m else ('mid' if monthly_vol_dict.get(m, 0) <= p67_m else 'high'))
        target_df['VOL_LOW'] = (target_df['M_REGIME'] == 'low').astype(int)
        target_df['VOL_MID'] = (target_df['M_REGIME'] == 'mid').astype(int)
        target_df['VOL_HIGH'] = (target_df['M_REGIME'] == 'high').astype(int)
        
        # 周度标签
        target_df['W_REGIME'] = target_df['YW'].map(week_to_regime_w)
        target_df['VOL_WLOW'] = (target_df['W_REGIME'] == 'wLow').astype(int)
        target_df['VOL_WMID'] = (target_df['W_REGIME'] == 'wMid').astype(int)
        target_df['VOL_WHIGH'] = (target_df['W_REGIME'] == 'wHigh').astype(int)
        
    train_df.dropna(subset=['M_REGIME', 'W_REGIME'], inplace=True)
    val_df.dropna(subset=['M_REGIME', 'W_REGIME'], inplace=True)
    
    regimes = ['low', 'mid', 'high', 'wLow', 'wMid', 'wHigh']
    trainers = {}
    
    for regime in regimes:
        logging.info(f"🚀 开始训练 [{regime.upper()}] 波动率区间模型...")
        target_col = f'VOL_{regime.upper()}'
        
        train_subset = train_df[train_df[target_col] == 1]
        val_subset = val_df[val_df[target_col] == 1]
        
        X_tr, y_tr = train_subset[feature_cols], train_subset[label_col]
        X_ev, y_ev = None, None
        
        if len(val_subset) >= 500:
            X_ev, y_ev = val_subset[feature_cols], val_subset[label_col]
            logging.info(f"  ✅ [{regime.upper()}] 验证集已启用 | 验证样本数: {len(X_ev)}")
            
        trainers[f"XGB-{regime}"] = train_tree_models(f"XGB-{regime}", 'xgb', X_tr, y_tr, X_ev, y_ev)
        trainers[f"LGBM-{regime}"] = train_tree_models(f"LGBM-{regime}", 'lgbm', X_tr, y_tr, X_ev, y_ev)
        
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    save_path = os.path.join(cfg.MODEL_DIR, "percentile_sklearn_models.pkl")
    joblib.dump(trainers, save_path)
    logging.info(f"✅ 所有波动率区间模型 (月度+周度) 已成功保存至: {save_path}")

# import sys
# import os
# import json
# import logging
# import joblib
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import lightgbm as lgb

# ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT))

# from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features, compute_derived_factors
# from config.Config import Config

# def setup_logging():
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# def get_volatility_config(cfg: Config):
#     """读取月度波动率数据，基于训练期计算 P33/P67 阈值"""
#     vol_monthly_path = cfg.OUT_DIR / "monthly_volatility.json"
#     vol_pct_path = cfg.OUT_DIR / "volatility_percentiles.json"
    
#     if not vol_monthly_path.exists() or not vol_pct_path.exists():
#         raise FileNotFoundError(f"❌ 找不到波动率数据文件: {vol_monthly_path}，请先运行 calculate_monthly_volatility.py")
        
#     with open(vol_monthly_path, 'r', encoding='utf-8') as f:
#         monthly_vol = json.load(f)
#     with open(vol_pct_path, 'r', encoding='utf-8') as f:
#         pct_data = json.load(f)
        
#     all_pct = pct_data.get('All', {})
#     p33 = float(all_pct.get('p33'))
#     p67 = float(all_pct.get('p67'))
    
#     logging.info(f"📊 [月度] 基于训练期(<=2024-12)计算的波动率阈值 | P33: {p33:.6f} | P67: {p67:.6f}")
#     return p33, p67, monthly_vol.get('All', {})

# def calculate_weekly_volatility_regime(df: pd.DataFrame):
#     """
#     在内存中计算全市场周度波动率，并基于训练期计算 P33/P67 阈值，
#     返回周度阈值及 YW(年-周) 到 Regime 的映射字典。
#     """
#     logging.info("🧮 开始计算全市场【周度】波动率...")
#     # 1. 准备计算数据
#     df_calc = df[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJCLOSE']].copy()
#     df_calc = df_calc.sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
    
#     # 2. 计算日收益率
#     df_calc['daily_ret'] = df_calc.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].pct_change()
    
#     # 3. 提取 ISO 周 (Year-Week) 及该周最后一个交易日
#     df_calc['YEAR'] = df_calc['TRADE_DT'].dt.isocalendar().year.astype(str)
#     df_calc['WEEK'] = df_calc['TRADE_DT'].dt.isocalendar().week.astype(str).str.zfill(2)
#     df_calc['YW'] = df_calc['YEAR'] + '-' + df_calc['WEEK']
#     df_calc['WEEK_END_DATE'] = df_calc.groupby(['S_INFO_WINDCODE', 'YW'])['TRADE_DT'].transform('max')
    
#     # 4. 计算每只股票每周的波动率 (标准差)
#     # 使用 std() 自动忽略 NaN，若某周交易天数<2则结果为 NaN
#     stock_weekly_vol = df_calc.groupby(['S_INFO_WINDCODE', 'YW', 'WEEK_END_DATE'])['daily_ret'].std().reset_index()
    
#     # 5. 计算全市场每周的平均波动率
#     market_weekly_vol = stock_weekly_vol.groupby(['YW', 'WEEK_END_DATE'])['daily_ret'].mean().reset_index()
#     market_weekly_vol.rename(columns={'daily_ret': 'VOLATILITY'}, inplace=True)
#     market_weekly_vol = market_weekly_vol.dropna(subset=['VOLATILITY']).sort_values('WEEK_END_DATE')
    
#     # 6. 基于训练期 (<= 2024-12-31) 计算周度 P33 和 P67
#     train_weekly = market_weekly_vol[market_weekly_vol['WEEK_END_DATE'] <= '2024-12-31']
#     if len(train_weekly) < 10:
#         raise ValueError("❌ 训练期周度波动率数据不足！")
        
#     p33_w = float(np.percentile(train_weekly['VOLATILITY'], 33))
#     p67_w = float(np.percentile(train_weekly['VOLATILITY'], 67))
#     logging.info(f"📊 [周度] 基于训练期(<=2024-12)计算的波动率阈值 | P33_w: {p33_w:.6f} | P67_w: {p67_w:.6f}")
    
#     # 7. 生成 YW -> Regime 的映射字典
#     week_to_regime = {}
#     for _, row in market_weekly_vol.iterrows():
#         vol = row['VOLATILITY']
#         if vol <= p33_w:
#             week_to_regime[row['YW']] = 'wLow'
#         elif vol <= p67_w:
#             week_to_regime[row['YW']] = 'wMid'
#         else:
#             week_to_regime[row['YW']] = 'wHigh'
            
#     logging.info(f"✅ 周度波动率计算完成 | 总周数: {len(week_to_regime)} | "
#                  f"wLow: {sum(1 for v in week_to_regime.values() if v == 'wLow')} | "
#                  f"wMid: {sum(1 for v in week_to_regime.values() if v == 'wMid')} | "
#                  f"wHigh: {sum(1 for v in week_to_regime.values() if v == 'wHigh')}")
                 
#     return week_to_regime

# def train_tree_models(model_name, model_type, X_train, y_train, X_eval=None, y_eval=None):
#     """统一封装 XGBoost 和 LightGBM 的训练逻辑，支持验证集早停"""
#     eval_set = [(X_eval, y_eval)] if X_eval is not None else None
    
#     if model_type == 'xgb':
#         model = xgb.XGBRegressor(
#             n_estimators=1000, max_depth=5, learning_rate=0.05, 
#             random_state=42, verbosity=0,
#             early_stopping_rounds=50 if X_eval is not None else None
#         )
#         model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
#     elif model_type == 'lgbm':
#         model = lgb.LGBMRegressor(
#             n_estimators=1000, max_depth=5, learning_rate=0.05, 
#             random_state=42, verbosity=-1
#         )
#         callbacks = []
#         if X_eval is not None:
#             callbacks.extend([lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            
#         if X_eval is not None:
#             model.fit(X_train, y_train, eval_X=X_eval, eval_y=y_eval, callbacks=callbacks)
#         else:
#             model.fit(X_train, y_train)
#     return model

# def main():
#     setup_logging()
#     cfg = Config()
    
#     logging.info("📦 加载并预处理面板数据 (扩展至 2026-03-31)...")
#     df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2027)), file_prefix="train", load_train=True, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
#     df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
#     df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE') 
    
#     # 限制全局时间范围
#     df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-05-01')) & (df['TRADE_DT'] <= pd.to_datetime('2026-03-31'))].copy()
    
#     feature_cols = extract_valid_features(df)
#     cfg.FEATURE_COLS = feature_cols
#     label_col = f'label_{cfg.REBALANCE_DAYS}'
    
#     # 过滤有效样本
#     df = df[(df['FEATURE_MASK'] == 1)].dropna(subset=[label_col] + feature_cols)
#     logging.info(f"📊 全局有效样本数: {len(df)} | 特征数: {len(feature_cols)}")
    
#     # ==========================================
#     # 1. 获取【月度】波动率配置
#     # ==========================================
#     p33_m, p67_m, monthly_vol_dict = get_volatility_config(cfg)
    
#     # ==========================================
#     # 2. 计算【周度】波动率配置
#     # ==========================================
#     week_to_regime_w = calculate_weekly_volatility_regime(df)
    
#     # ==========================================
#     # 3. 划分训练集与验证集，并生成所有 Regime 标签
#     # ==========================================
#     train_end_date = pd.to_datetime('2024-12-31')
#     val_start_date = pd.to_datetime('2025-01-01')
#     val_end_date = pd.to_datetime('2026-03-31')
    
#     train_df = df[df['TRADE_DT'] <= train_end_date].copy()
#     val_df = df[(df['TRADE_DT'] >= val_start_date) & (df['TRADE_DT'] <= val_end_date)].copy()
    
#     logging.info(f"📅 训练集范围: ~ {train_end_date.strftime('%Y-%m-%d')} | 样本数: {len(train_df)}")
#     logging.info(f"📅 验证集范围: {val_start_date.strftime('%Y-%m-%d')} ~ {val_end_date.strftime('%Y-%m-%d')} | 样本数: {len(val_df)}")
    
#     # 提取时间特征
#     for target_df in [train_df, val_df]:
#         target_df['YEAR_MONTH'] = target_df['TRADE_DT'].dt.strftime('%Y-%m')
#         target_df['YEAR'] = target_df['TRADE_DT'].dt.isocalendar().year.astype(str)
#         target_df['WEEK'] = target_df['TRADE_DT'].dt.isocalendar().week.astype(str).str.zfill(2)
#         target_df['YW'] = target_df['YEAR'] + '-' + target_df['WEEK']
        
#     # 🔑 核心修复：直接基于互斥的 Regime 结果生成 0/1 标签列，彻底杜绝区间重叠！
#     # 月度标签
#     train_df['M_REGIME'] = train_df['YEAR_MONTH'].map(lambda m: 'low' if monthly_vol_dict.get(m, 0) <= p33_m else ('mid' if monthly_vol_dict.get(m, 0) <= p67_m else 'high'))
#     train_df['VOL_LOW'] = (train_df['M_REGIME'] == 'low').astype(int)
#     train_df['VOL_MID'] = (train_df['M_REGIME'] == 'mid').astype(int)
#     train_df['VOL_HIGH'] = (train_df['M_REGIME'] == 'high').astype(int)
    
#     val_df['M_REGIME'] = val_df['YEAR_MONTH'].map(lambda m: 'low' if monthly_vol_dict.get(m, 0) <= p33_m else ('mid' if monthly_vol_dict.get(m, 0) <= p67_m else 'high'))
#     val_df['VOL_LOW'] = (val_df['M_REGIME'] == 'low').astype(int)
#     val_df['VOL_MID'] = (val_df['M_REGIME'] == 'mid').astype(int)
#     val_df['VOL_HIGH'] = (val_df['M_REGIME'] == 'high').astype(int)
    
#     # 周度标签
#     train_df['W_REGIME'] = train_df['YW'].map(week_to_regime_w)
#     train_df['VOL_WLOW'] = (train_df['W_REGIME'] == 'wLow').astype(int)
#     train_df['VOL_WMID'] = (train_df['W_REGIME'] == 'wMid').astype(int)
#     train_df['VOL_WHIGH'] = (train_df['W_REGIME'] == 'wHigh').astype(int)
    
#     val_df['W_REGIME'] = val_df['YW'].map(week_to_regime_w)
#     val_df['VOL_WLOW'] = (val_df['W_REGIME'] == 'wLow').astype(int)
#     val_df['VOL_WMID'] = (val_df['W_REGIME'] == 'wMid').astype(int)
#     val_df['VOL_WHIGH'] = (val_df['W_REGIME'] == 'wHigh').astype(int)
    
#     # 剔除无法映射的样本 (极少部分因数据缺失导致)
#     train_df = train_df.dropna(subset=['M_REGIME', 'W_REGIME'])
#     val_df = val_df.dropna(subset=['M_REGIME', 'W_REGIME'])
    
#     # ==========================================
#     # 4. 循环训练 6 种波动率区间的模型
#     # ==========================================
#     regimes = ['low', 'mid', 'high', 'wLow', 'wMid', 'wHigh']
#     trainers = {}
    
#     for regime in regimes:
#         logging.info(f"🚀 开始训练 [{regime.upper()}] 波动率区间模型...")
        
#         target_col = f'VOL_{regime.upper()}'
        
#         # 准备训练集
#         train_subset = train_df[train_df[target_col] == 1]
#         if len(train_subset) < 1000:
#             logging.warning(f"  ⚠️ [{regime.upper()}] 训练集样本不足 ({len(train_subset)} < 1000)，可能导致欠拟合。")
#         X_tr, y_tr = train_subset[feature_cols], train_subset[label_col]
        
#         # 准备验证集
#         val_subset = val_df[val_df[target_col] == 1]
#         X_ev, y_ev = None, None
#         MIN_VAL_SAMPLES = 500 
#         if len(val_subset) >= MIN_VAL_SAMPLES:
#             X_ev, y_ev = val_subset[feature_cols], val_subset[label_col]
#             logging.info(f"  ✅ [{regime.upper()}] 验证集已启用 | 验证样本数: {len(X_ev)}")
#         else:
#             logging.warning(f"  ⚠️ [{regime.upper()}] 验证集样本不足 ({len(val_subset)} < {MIN_VAL_SAMPLES})，不使用早停。")
            
#         logging.info(f"  训练集样本数: {len(X_tr)}")
        
#         # 训练 XGBoost
#         model_name_xgb = f"XGB-{regime}"
#         trainers[model_name_xgb] = train_tree_models(model_name_xgb, 'xgb', X_tr, y_tr, X_ev, y_ev)
        
#         # 训练 LightGBM
#         model_name_lgbm = f"LGBM-{regime}"
#         trainers[model_name_lgbm] = train_tree_models(model_name_lgbm, 'lgbm', X_tr, y_tr, X_ev, y_ev)
        
#     # ==========================================
#     # 5. 保存模型
#     # ==========================================
#     os.makedirs(cfg.MODEL_DIR, exist_ok=True)
#     save_path = os.path.join(cfg.MODEL_DIR, "percentile_sklearn_models.pkl")
#     joblib.dump(trainers, save_path)
#     logging.info(f"✅ 所有波动率区间模型 (月度+周度) 已成功保存至: {save_path}")
#     logging.info(f"📦 保存的模型列表: {list(trainers.keys())}")

if __name__ == "__main__":
    main()