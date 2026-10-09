import sys
import os
import json
import re
import logging
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Run SHAP analysis for specific models.")
    parser.add_argument('--models', nargs='+', default=None, 
                        help='List of models to analyze. Overrides --group if provided.')
    parser.add_argument('--group', type=str, choices=['all', 'base', 'percentile', 'cluster', 'ablation'], default='base',
                        help='Model group to analyze.')
    return parser.parse_args()

def get_target_models(args, cfg):
    """根据命令行参数获取需要分析的模型列表"""
    if args.models:
        return args.models
    
    if args.group == 'base':
        return ['XGB-22', 'XGB-23', 'XGB-24', 'LGBM-22', 'LGBM-23', 'LGBM-24']
    elif args.group == 'percentile':
        return ['XGB-low', 'XGB-mid', 'XGB-high', 'LGBM-low', 'LGBM-mid', 'LGBM-high',
                'XGB-wLow', 'XGB-wMid', 'XGB-wHigh', 'LGBM-wLow', 'LGBM-wMid', 'LGBM-wHigh']
    elif args.group == 'cluster':
        pkl_path = os.path.join(cfg.MODEL_DIR, "cluster_sklearn_models.pkl")
        if os.path.exists(pkl_path):
            trainers = joblib.load(pkl_path)
            return list(trainers.keys())
        else:
            logging.warning(f"⚠️ 未找到 {pkl_path}，请先运行 train_models_cluster.py")
            return []
    elif args.group == 'ablation':
        if os.path.exists(cfg.ABLATION_FEATURE_JSON):
            with open(cfg.ABLATION_FEATURE_JSON, 'r', encoding='utf-8') as f:
                feats = json.load(f)
            return list(feats.keys())
        return []
    elif args.group == 'all':
        base = get_target_models(argparse.Namespace(models=None, group='base'), cfg)
        perc = get_target_models(argparse.Namespace(models=None, group='percentile'), cfg)
        clust = get_target_models(argparse.Namespace(models=None, group='cluster'), cfg)
        abl = get_target_models(argparse.Namespace(models=None, group='ablation'), cfg)
        return base + perc + clust + abl
    return []

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
    """获取用于 SHAP 分析的模型。遍历所有 pkl 文件查找预训练模型。"""
    model_files = [
        "sklearn_models.pkl",
        "percentile_sklearn_models.pkl",
        "cluster_sklearn_models.pkl",  # 🔑 新增对聚类模型的支持
        "ablation_sklearn_models.pkl"
    ]
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
    logging.info(f"📦 开始加载全量训练集数据 (用于 {model_type})...")
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

    # ==========================================
    # 4. 🔑 处理聚类区间模型 (_c1, _c2, _c3...)
    # ==========================================
    elif re.search(r'[_-]c(\d+)$', model_type):
        logging.info(f"🔍 [{model_type}] 正在计算聚类标签...")
        labels_path = cfg.OUT_DIR / "market_regime_labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"❌ 找不到聚类标签文件: {labels_path}，请先运行 cluster_market_regimes.py")
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            regime_labels = json.load(f)
            
        # 提取 cluster ID (例如从 XGB_c1 提取 0，因为 c1 对应 cluster 0)
        match = re.search(r'[_-]c(\d+)$', model_type)
        c_id = int(match.group(1)) - 1  # c1 -> 0, c2 -> 1
        
        df_filtered['YEAR_MONTH'] = df_filtered['TRADE_DT'].dt.strftime('%Y-%m')
        df_filtered['CLUSTER_ID'] = df_filtered['YEAR_MONTH'].map(regime_labels)
        
        # 过滤出对应 cluster 的数据
        df_filtered = df_filtered[df_filtered['CLUSTER_ID'] == c_id]
        logging.info(f"✅ [{model_type}] 已过滤为 Cluster {c_id} | 采样数: {len(df_filtered)}")

    # ==========================================
    # 5. 季度分组与 SHAP 计算
    # ==========================================
    target_df = df_filtered
    if 'QUARTER' not in target_df.columns:
        target_df['QUARTER'] = target_df['TRADE_DT'].dt.to_period('Q')
        
    quarters = sorted(target_df['QUARTER'].unique())
    model = get_or_train_model(cfg, feature_cols, target_df, model_type=model_type)
    explainer = shap.TreeExplainer(model)
    
    ic_ir_results = {}
    shap_results = {}
    N_SAMPLING, SAMPLE_SIZE = 20, 2000
    
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
    args = parse_args()
    
    target_models = get_target_models(args, cfg)
    if not target_models:
        logging.error("❌ 未找到任何需要分析的模型，请检查参数或模型文件。")
        return
        
    logging.info(f"🎯 准备分析以下 {len(target_models)} 个模型: {target_models}")
    
    for model in target_models:
        try:
            run_quarterly_analysis(cfg, model_type=model)
        except Exception as e:
            logging.error(f"❌ 分析模型 {model} 时发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()