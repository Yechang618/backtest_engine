# /data/cye_temp/workspace/backtest_engine/script/shap_analysis.py
import sys
import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.data_loader import extract_valid_features, compute_ic_ir, compute_real_returns, load_panel_data, compute_derived_factors
from config.Config import Config

try:
    import shap
    import lightgbm as lgb
    import joblib
except ImportError:
    raise ImportError("请确保已安装必要库: pip install shap lightgbm joblib")

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def load_train_data_only(cfg: Config) -> pd.DataFrame:
    logging.info("📦 开始加载面板数据...")
    df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2027)), file_prefix="train", load_train=True, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE')
    # 🔑 统一时间范围至 2024-12-31 (训练集上限)
    df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-05-01')) & (df['TRADE_DT'] <= pd.to_datetime('2024-12-31'))].copy()
    logging.info(f"✅ 训练集加载完成 | 总行数: {len(df):,}")
    return df

def get_or_train_model(cfg: Config, feature_cols: List[str], df_train: pd.DataFrame, model_type: str = 'LightGBM'):
    model_files = ["percentile_sklearn_models.pkl", "sklearn_models.pkl", "ablation_sklearn_models.pkl"]
    for model_file in model_files:
        model_path = os.path.join(cfg.MODEL_DIR, model_file)
        if os.path.exists(model_path):
            try:
                trainers = joblib.load(model_path)
                if model_type in trainers:
                    logging.info(f"✅ 从 {model_file} 加载预训练的 {model_type} 模型")
                    return trainers[model_type]
            except Exception as e:
                logging.warning(f"⚠️ 加载 {model_file} 失败: {e}")
    logging.warning(f"⚠️ 未找到预训练的 {model_type} 模型，将现场训练基准模型...")
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    df_valid = df_train[(df_train['FEATURE_MASK'] == 1)].dropna(subset=[label_col] + feature_cols)
    model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1)
    model.fit(df_valid[feature_cols], df_valid[label_col])
    return model

def run_quarterly_analysis(cfg: Config, model_type: str = 'LightGBM'):
    logging.info("📦 开始加载全量训练集数据...")
    df = load_train_data_only(cfg)
    feature_cols = extract_valid_features(df)
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    
    df_filtered = df.copy()
    
    # ==========================================
    # 1. 处理月度波动率区间模型 (-low, -mid, -high)
    # ==========================================
    if any(model_type.endswith(suffix) for suffix in ['-low', '-mid', '-high', '_low', '_mid', '_high']):
        logging.info(f"🔍 [{model_type}] 正在计算月度波动率标签...")
        vol_monthly_path = cfg.OUT_DIR / "monthly_volatility.json"
        vol_pct_path = cfg.OUT_DIR / "volatility_percentiles.json"
        with open(vol_monthly_path, 'r', encoding='utf-8') as f: monthly_vol = json.load(f)
        with open(vol_pct_path, 'r', encoding='utf-8') as f: pct_data = json.load(f)
        
        all_pct = pct_data.get('All', {})
        p33 = float(all_pct.get('p33'))
        p67 = float(all_pct.get('p67'))
        all_monthly = monthly_vol.get('All', {})
        regime = model_type.split('-')[-1].split('_')[-1]
        
        df_filtered['YEAR_MONTH'] = df_filtered['TRADE_DT'].dt.strftime('%Y-%m')
        def get_monthly_regime(m):
            vol = all_monthly.get(m)
            if vol is None or pd.isna(vol): return None
            if vol <= p33: return 'low'
            elif vol <= p67: return 'mid'
            else: return 'high'
            
        df_filtered['REGIME'] = df_filtered['YEAR_MONTH'].apply(get_monthly_regime)
        df_filtered = df_filtered[df_filtered['REGIME'].notna() & (df_filtered['REGIME'] == regime)]
        logging.info(f"✅ [{model_type}] 已过滤为 {regime} 波动率区间 | 采样数: {len(df_filtered)}")

    # ==========================================
    # 2. 处理周度波动率区间模型 (-wLow, -wMid, -wHigh)
    # ==========================================
    elif any(model_type.endswith(suffix) for suffix in ['-wLow', '-wMid', '-wHigh', '_wLow', '_wMid', '_wHigh']):
        logging.info(f"🧮 [{model_type}] 计算周度波动率标签...")
        df_calc = df_filtered[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJCLOSE']].copy().sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
        df_calc['daily_ret'] = df_calc.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].pct_change()
        df_calc['YEAR'] = df_calc['TRADE_DT'].dt.isocalendar().year.astype(str)
        df_calc['WEEK'] = df_calc['TRADE_DT'].dt.isocalendar().week.astype(str).str.zfill(2)
        df_calc['YW'] = df_calc['YEAR'] + '-' + df_calc['WEEK']
        
        stock_weekly_vol = df_calc.groupby(['S_INFO_WINDCODE', 'YW'])['daily_ret'].std().reset_index()
        market_weekly_vol = stock_weekly_vol.groupby('YW')['daily_ret'].mean()
        
        # 基于训练期 (<= 2024-12) 计算周度 P33/P67
        train_weekly = market_weekly_vol[market_weekly_vol.index <= '2024-52']
        p33_w = float(np.percentile(train_weekly.dropna(), 33))
        p67_w = float(np.percentile(train_weekly.dropna(), 67))
        
        week_to_regime = {}
        for yw, vol in market_weekly_vol.items():
            if pd.isna(vol): continue
            if vol <= p33_w: week_to_regime[yw] = 'wLow'
            elif vol <= p67_w: week_to_regime[yw] = 'wMid'
            else: week_to_regime[yw] = 'wHigh'
            
        regime = 'w' + model_type.split('-')[-1].split('_')[-1].lstrip('w')
        df_filtered['YEAR'] = df_filtered['TRADE_DT'].dt.isocalendar().year.astype(str)
        df_filtered['WEEK'] = df_filtered['TRADE_DT'].dt.isocalendar().week.astype(str).str.zfill(2)
        df_filtered['YW'] = df_filtered['YEAR'] + '-' + df_filtered['WEEK']
        df_filtered['W_REGIME'] = df_filtered['YW'].map(week_to_regime)
        df_filtered = df_filtered[df_filtered['W_REGIME'].notna() & (df_filtered['W_REGIME'] == regime)]
        logging.info(f"✅ [{model_type}] 已过滤为 {regime} 周度区间 | 采样数: {len(df_filtered)}")

    # ==========================================
    # 3. 处理时间窗口模型 (-22, -23, -24)
    # ==========================================
    elif model_type.endswith(('-22', '-23', '-24')):
        end_dates = {'22': '2022-05-31', '23': '2023-05-31', '24': '2024-08-31'}
        suffix = model_type.split('-')[-1]
        df_filtered = df_filtered[df_filtered['TRADE_DT'] <= pd.to_datetime(end_dates[suffix])]
        logging.info(f"✅ [{model_type}] 已过滤为 ≤ {end_dates[suffix]} | 采样数: {len(df_filtered)}")

    target_df = df_filtered
    if 'QUARTER' not in target_df.columns:
        target_df['QUARTER'] = target_df['TRADE_DT'].dt.to_period('Q')
        
    quarters = sorted(target_df['QUARTER'].unique())
    model = get_or_train_model(cfg, feature_cols, target_df, model_type=model_type)
    explainer = shap.TreeExplainer(model)
    
    ic_ir_results = {}
    shap_results = {}
    N_SAMPLING, SAMPLE_SIZE = 20, 5000
    
    for q in quarters:
        q_str = str(q)
        q_df = target_df[target_df['QUARTER'] == q].copy()
        q_df_valid = q_df[q_df['FEATURE_MASK'] == 1]
        
        try:
            ic_ir_results[q_str] = compute_ic_ir(feature_cols, label_col, q_df_valid)
        except Exception as e:
            logging.error(f"⚠️ {q_str} IC/IR 计算失败: {e}")
            continue
            
        q_df_clean = q_df_valid.dropna(subset=feature_cols + [label_col])
        actual_sample_size = min(SAMPLE_SIZE, len(q_df_clean))
        if actual_sample_size < 100:
            logging.warning(f"  ⚠️ {q_str} 有效样本过少 ({actual_sample_size})，跳过 SHAP 计算")
            continue
            
        shap_sum = {col: 0.0 for col in feature_cols}
        for i in range(N_SAMPLING):
            sample_df = q_df_clean.sample(n=actual_sample_size, random_state=42 + i)
            shap_values = explainer.shap_values(sample_df[feature_cols])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            for idx, col in enumerate(feature_cols):
                shap_sum[col] += float(mean_abs_shap[idx])
                
        shap_results[q_str] = {col: float(shap_sum[col] / N_SAMPLING) for col in feature_cols}
        logging.info(f"  ✅ {q_str} SHAP 分析完成")
        
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUT_DIR, f"ic_ir_quarterly_analysis_{model_type}.json"), 'w', encoding='utf-8') as f:
        json.dump(ic_ir_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(cfg.OUT_DIR, f"shap_quarterly_analysis_{model_type}.json"), 'w', encoding='utf-8') as f:
        json.dump(shap_results, f, indent=2, ensure_ascii=False)
    logging.info(f"🎉 季度 SHAP 与 IC/IR 分析({model_type})全部完成！")

def main():
    setup_logging()
    cfg = Config()
    # 🔑 补充周度波动率模型的调用
    for model in ['XGB-wLow', 'XGB-wMid', 'XGB-wHigh', 'LGBM-wLow', 'LGBM-wMid', 'LGBM-wHigh']:
        run_quarterly_analysis(cfg, model_type=model)

# import sys
# import os
# import json
# import logging
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from typing import List, Dict

# # 🔧 配置路径，确保能导入项目内的模块
# ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT))

# # 🔑 修复：引入 compute_real_returns 以生成标签列
# from util.data_loader import extract_valid_features, compute_ic_ir, compute_real_returns, load_panel_data, compute_derived_factors
# from config.Config import Config

# # 尝试导入模型和 SHAP 库
# try:
#     import shap
#     import lightgbm as lgb
#     import joblib
# except ImportError:
#     raise ImportError("请确保已安装必要库: pip install shap lightgbm joblib")

# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO, 
#         format='%(asctime)s | %(levelname)s | %(message)s'
#     )

# def load_train_data_only(cfg: Config) -> pd.DataFrame:
#     """仅加载 /data/train_data 中的全部历史数据，彻底隔离测试集"""
#     dfs = []
#     train_years = list(range(2016, 2024)) 
    
#     for y in train_years:
#         path = os.path.join(cfg.DATA_DIR, f'model_ready_panel_selected_plus_ohlc_train_{y}.parquet')
#         if os.path.exists(path):
#             df_y = pd.read_parquet(path)
#             df_y['TRADE_DT'] = pd.to_datetime(df_y['TRADE_DT'].astype(str), format='%Y%m%d')
#             dfs.append(df_y)
#         else:
#             logging.warning(f"⚠️ 未找到训练数据: {path}")

#     df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2025)), file_prefix="train", load_train=True, load_test=True)
#     # dfs.append(df)
#     # df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
#     # Get df from 2016 to 2024
#     df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-06-01')) & (df['TRADE_DT'] <= pd.to_datetime('2024-07-31'))].copy()
#     dfs.append(df)
#     # feature_cols = extract_valid_features(df)
            
#     if not dfs:
#         raise FileNotFoundError(f"在 {cfg.DATA_DIR} 中未找到任何训练集 parquet 文件")
        
#     df_full = pd.concat(dfs, ignore_index=True)
#     df_full = df_full.sort_values(['TRADE_DT', 'S_INFO_WINDCODE']).reset_index(drop=True)
#     logging.info(f"✅ 训练集加载完成 | 总行数: {len(df_full):,} | 时间范围: {df_full['TRADE_DT'].min()} 至 {df_full['TRADE_DT'].max()}")
#     return df_full

# # def get_or_train_model(cfg: Config, feature_cols: List[str], df_train: pd.DataFrame, model_type: str = 'LightGBM'):
# #     """获取用于 SHAP 分析的基准模型。优先加载已保存的 LightGBM，否则现场训练一个"""
# #     model_path = os.path.join(cfg.MODEL_DIR, "sklearn_models.pkl")
    
# #     if os.path.exists(model_path):
# #         try:
# #             trainers = joblib.load(model_path)
# #             if model_type in trainers:
# #                 logging.info(f"✅ 成功从 saved_models 加载预训练的 {model_type} 模型用于 SHAP 分析")
# #                 return trainers[model_type]
# #         except Exception as e:
# #             logging.warning(f"⚠️ 加载预训练模型失败: {e}，将现场训练一个基准模型")

# #     logging.info(f"🚀 未找到预训练的 {model_type} 模型，正在使用全量训练集现场训练 {model_type} 基准模型...")
# #     label_col = f'label_{cfg.REBALANCE_DAYS}'
# #     valid_mask = (df_train['FEATURE_MASK'] == 1)
# #     df_valid = df_train[valid_mask].dropna(subset=[label_col] + feature_cols)
    
# #     X = df_valid[feature_cols]
# #     y = df_valid[label_col]
    
# #     model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1)
# #     model.fit(X, y)
# #     logging.info(f"✅ 基准模型训练完成 | 训练样本数: {len(X):,}")
# #     return model

# def get_or_train_model(cfg: Config, feature_cols: List[str], df_train: pd.DataFrame, model_type: str = 'LightGBM'):
#     """获取用于 SHAP 分析的模型。遍历所有 pkl 文件查找预训练模型。"""
#     # 🔑 修复：遍历所有模型文件，包括 percentile 和 ablation
#     model_files = [
#         "sklearn_models.pkl",
#         "percentile_sklearn_models.pkl",
#         # "ablation_sklearn_models.pkl"
#     ]
    
#     for model_file in model_files:
#         model_path = os.path.join(cfg.MODEL_DIR, model_file)
#         if os.path.exists(model_path):
#             try:
#                 trainers = joblib.load(model_path)
#                 if model_type in trainers:
#                     logging.info(f"✅ 从 {model_file} 加载预训练的 {model_type} 模型")
#                     return trainers[model_type]
#             except Exception as e:
#                 logging.warning(f"⚠️ 加载 {model_file} 失败: {e}")
    
#     # 回退：现场训练
#     logging.warning(f"⚠️ 未找到预训练的 {model_type} 模型，将现场训练基准模型...")
#     label_col = f'label_{cfg.REBALANCE_DAYS}'
#     valid_mask = (df_train['FEATURE_MASK'] == 1)
#     df_valid = df_train[valid_mask].dropna(subset=[label_col] + feature_cols)
#     X, y = df_valid[feature_cols], df_valid[label_col]
#     model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1)
#     model.fit(X, y)
#     return model

# def run_quarterly_analysis(cfg: Config, model_type: str = 'LightGBM'):
#     """执行按季度的 IC/IR 与 SHAP 值分析"""
#     logging.info("📦 开始加载全量训练集数据...")
#     df = load_train_data_only(cfg)
#     df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
#     df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE')
    
#     feature_cols = extract_valid_features(df)
#     label_col = f'label_{cfg.REBALANCE_DAYS}'
    
#     # 🔑 核心修复：根据模型后缀过滤采样数据，使其与训练集一致
#     df_filtered = df.copy()
    
#     # ==========================================
#     # 1. 处理月度波动率区间模型 (_low, _mid, _high)
#     # ==========================================
#     if any(model_type.endswith(suffix) for suffix in ['-low', '-mid', '-high', '_low', '_mid', '_high']):
#         logging.info(f"🔍 [{model_type}] 正在计算月度波动率标签...")
        
#         # 🔑 自包含：直接读取 JSON，不再依赖外部函数
#         import json
#         vol_monthly_path = cfg.OUT_DIR / "monthly_volatility.json"
#         vol_pct_path = cfg.OUT_DIR / "volatility_percentiles.json"
        
#         with open(vol_monthly_path, 'r', encoding='utf-8') as f:
#             monthly_vol = json.load(f)
#         with open(vol_pct_path, 'r', encoding='utf-8') as f:
#             pct_data = json.load(f)
            
#         all_pct = pct_data.get('All', {})
#         p33 = float(all_pct.get('p33'))
#         p67 = float(all_pct.get('p67'))
#         all_monthly = monthly_vol.get('All', {})
        
#         regime = model_type.split('-')[-1].split('_')[-1]  # 提取 'low'/'mid'/'high'
        
#         df_filtered['YEAR_MONTH'] = df_filtered['TRADE_DT'].dt.strftime('%Y-%m')
        
#         # 🔑 内联 assign_regime 逻辑
#         def get_monthly_regime(m):
#             vol = all_monthly.get(m)
#             if vol is None: return None
#             if vol <= p33: return 'low'
#             elif vol <= p67: return 'mid'
#             else: return 'high'
            
#         df_filtered['VOL_REGIME'] = df_filtered['YEAR_MONTH'].apply(get_monthly_regime)
#         df_filtered = df_filtered[df_filtered['VOL_REGIME'] == regime]
#         logging.info(f"✅ [{model_type}] 已过滤为 {regime} 波动率区间 | 采样数: {len(df_filtered)}")
        
#     elif any(model_type.endswith(suffix) for suffix in ['-wLow', '-wMid', '-wHigh', '_wLow', '_wMid', '_wHigh']):
#         # 周度波动率区间模型：按周过滤
#         logging.info(f"🧮 [{model_type}] 计算周度波动率标签...")
#         df_calc = df_filtered[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJCLOSE']].copy()
#         df_calc = df_calc.sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
#         df_calc['daily_ret'] = df_calc.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].pct_change()
#         df_calc['YEAR'] = df_calc['TRADE_DT'].dt.isocalendar().year.astype(str)
#         df_calc['WEEK'] = df_calc['TRADE_DT'].dt.isocalendar().week.astype(str).str.zfill(2)
#         df_calc['YW'] = df_calc['YEAR'] + '-' + df_calc['WEEK']
        
#         stock_weekly_vol = df_calc.groupby(['S_INFO_WINDCODE', 'YW'])['daily_ret'].std().reset_index()
#         market_weekly_vol = stock_weekly_vol.groupby('YW')['daily_ret'].mean()
        
#         # 基于训练期计算 P33/P67
#         train_weekly = market_weekly_vol[market_weekly_vol.index <= '2024-52']
#         p33_w = float(np.percentile(train_weekly.dropna(), 33))
#         p67_w = float(np.percentile(train_weekly.dropna(), 67))
        
#         week_to_regime = {}
#         for yw, vol in market_weekly_vol.items():
#             if pd.isna(vol): continue
#             if vol <= p33_w: week_to_regime[yw] = 'wLow'
#             elif vol <= p67_w: week_to_regime[yw] = 'wMid'
#             else: week_to_regime[yw] = 'wHigh'
        
#         regime = 'w' + model_type.split('-')[-1].split('_')[-1].lstrip('w')  # 提取 'wLow'/'wMid'/'wHigh'
#         df_filtered['YEAR'] = df_filtered['TRADE_DT'].dt.isocalendar().year.astype(str)
#         df_filtered['WEEK'] = df_filtered['TRADE_DT'].dt.isocalendar().week.astype(str).str.zfill(2)
#         df_filtered['YW'] = df_filtered['YEAR'] + '-' + df_filtered['WEEK']
#         df_filtered['W_REGIME'] = df_filtered['YW'].map(week_to_regime)
#         df_filtered = df_filtered[df_filtered['W_REGIME'] == regime]
#         logging.info(f"🔍 [{model_type}] 已过滤为 {regime} 周度区间 | 采样数: {len(df_filtered)}")
    
#     elif model_type.endswith(('-22', '-23', '-24')):
#         # 时间窗口模型：按截止日期过滤
#         end_dates = {'22': '2022-05-31', '23': '2023-05-31', '24': '2024-08-31'}
#         suffix = model_type.split('-')[-1]
#         df_filtered = df_filtered[df_filtered['TRADE_DT'] <= pd.to_datetime(end_dates[suffix])]
#         logging.info(f"🔍 [{model_type}] 已过滤为 ≤ {end_dates[suffix]} | 采样数: {len(df_filtered)}")
    
#     # 🔑 核心修复：统一确保用于后续分析的 DataFrame 包含 'QUARTER' 列
#     target_df = df_filtered  # ⚠️ 如果你前面定义了 df_filtered，请改为 target_df = df_filtered
    
#     if 'QUARTER' not in target_df.columns:
#         logging.info("🔧 检测到缺少 QUARTER 列，正在自动补充...")
#         target_df['QUARTER'] = target_df['TRADE_DT'].dt.to_period('Q')
        
#     quarters = sorted(target_df['QUARTER'].unique())
#     logging.info(f"📅 检测到 {len(quarters)} 个季度窗口: {quarters[0]} 至 {quarters[-1]}")
    
#     # 获取基准模型 (注意传入 target_df)
#     model = get_or_train_model(cfg, feature_cols, target_df, model_type=model_type)
#     explainer = shap.TreeExplainer(model)
    
#     # 结果存储字典
#     ic_ir_results = {}
#     shap_results = {}
    
#     # 分析参数
#     N_SAMPLING = 20
#     SAMPLE_SIZE = 5000
    
#     for q in quarters:
#         q_str = str(q)
#         logging.info(f"▶️ 正在处理季度: {q_str} ...")
        
#         # 🔑 修复点：必须使用 target_df (或 df_filtered) 来提取季度数据，绝不能用原始的 df
#         q_df = target_df[target_df['QUARTER'] == q].copy()
        
#         # 1. 计算 IC / IR
#         try:
#             # 过滤掉 FEATURE_MASK != 1 的数据，确保与模型训练口径一致
#             q_df_valid = q_df[q_df['FEATURE_MASK'] == 1]
#             ic_ir_q = compute_ic_ir(feature_cols, label_col, q_df_valid)
#             ic_ir_results[q_str] = ic_ir_q
#         except Exception as e:
#             logging.error(f"⚠️ {q_str} IC/IR 计算失败: {e}")
#             continue
            
#         # 2. 计算 SHAP 值 (蒙特卡洛采样)
#         q_df_clean = q_df_valid.dropna(subset=feature_cols + [label_col])
#         actual_sample_size = min(SAMPLE_SIZE, len(q_df_clean))
#         if actual_sample_size < 100:
#             logging.warning(f"  ⚠️ {q_str} 有效样本过少 ({actual_sample_size})，跳过 SHAP 计算")
#             continue
            
#         logging.info(f"  🎲 开始 SHAP 采样计算 (共 {N_SAMPLING} 次, 每次 {actual_sample_size} 样本)...")
        
#         # 累加器：记录每个特征的平均绝对 SHAP 值总和
#         shap_sum = {col: 0.0 for col in feature_cols}
#         for i in range(N_SAMPLING):
#             sample_df = q_df_clean.sample(n=actual_sample_size, random_state=42 + i)
#             sample_X = sample_df[feature_cols]
#             shap_values = explainer.shap_values(sample_X)
#             mean_abs_shap = np.abs(shap_values).mean(axis=0)
#             for idx, col in enumerate(feature_cols):
#                 shap_sum[col] += float(mean_abs_shap[idx])
                
#         shap_results[q_str] = {
#             col: float(shap_sum[col] / N_SAMPLING) for col in feature_cols
#         }
#         logging.info(f"  ✅ {q_str} SHAP 分析完成")
        
#     # ... (后续保存 JSON 和 Parquet 的逻辑保持不变) ...

#     # 3. 保存结果
#     os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
#     ic_ir_path = os.path.join(cfg.OUT_DIR, f"ic_ir_quarterly_analysis_{model_type}.json")
#     with open(ic_ir_path, 'w', encoding='utf-8') as f:
#         json.dump(ic_ir_results, f, indent=2, ensure_ascii=False)
#     logging.info(f"💾 IC/IR 结果已保存至: {ic_ir_path}")
    
#     shap_path = os.path.join(cfg.OUT_DIR, f"shap_quarterly_analysis_{model_type}.json")
#     with open(shap_path, 'w', encoding='utf-8') as f:
#         json.dump(shap_results, f, indent=2, ensure_ascii=False)
#     logging.info(f"💾 SHAP 结果已保存至: {shap_path}")
    
#     shap_df_list = []
#     for q_str, shap_dict in shap_results.items():
#         row = {'QUARTER': q_str}
#         row.update(shap_dict)
#         shap_df_list.append(row)
    
#     shap_df = pd.DataFrame(shap_df_list)
#     shap_df.set_index('QUARTER', inplace=True)
#     parquet_path = os.path.join(cfg.OUT_DIR, f"shap_quarterly_analysis_{model_type}.parquet")
#     shap_df.to_parquet(parquet_path)
#     logging.info(f"💾 SHAP 结果 (Parquet) 已保存至: {parquet_path}")
    
#     logging.info(f"🎉 季度 SHAP 与 IC/IR 分析({model_type})全部完成！")



# def main():
#     setup_logging()
#     cfg = Config()
#     # run_quarterly_analysis(cfg, model_type='XGB-wLow')
#     # run_quarterly_analysis(cfg, model_type='XGB-wMid')
#     # run_quarterly_analysis(cfg, model_type='XGB-wHigh')
#     # run_quarterly_analysis(cfg, model_type='LGBM-wLow')
#     # run_quarterly_analysis(cfg, model_type='LGBM-wMid')
#     # run_quarterly_analysis(cfg, model_type='LGBM-wHigh')
#     # run_quarterly_analysis(cfg, model_type='XGB-22')
#     # run_quarterly_analysis(cfg, model_type='XGB-23')
#     # run_quarterly_analysis(cfg, model_type='XGB-24')
#     # run_quarterly_analysis(cfg, model_type='LGBM-22')
#     # run_quarterly_analysis(cfg, model_type='LGBM-23')
#     # run_quarterly_analysis(cfg, model_type='LGBM-24')
#     run_quarterly_analysis(cfg, model_type='XGB-low')
#     run_quarterly_analysis(cfg, model_type='XGB-mid')
#     run_quarterly_analysis(cfg, model_type='XGB-high')
#     run_quarterly_analysis(cfg, model_type='LGBM-low')
#     run_quarterly_analysis(cfg, model_type='LGBM-mid')
#     run_quarterly_analysis(cfg, model_type='LGBM-high')

if __name__ == "__main__":
    main()