# /data/cye_temp/workspace/backtest_engine/util/data_loader.py
import os
import re
import pandas as pd
import numpy as np
import warnings
from typing import List, Dict

# 🔑 修改函数签名，增加 exclude_bj 参数
def load_panel_data(data_root: str, data_test_root: str, years: List[int], file_prefix: str = "train", load_train: bool = True, load_test: bool = True, exclude_bj: bool = False) -> pd.DataFrame:
    dfs = []
    if data_root is not None and load_train:
        print(f"Loading training data from: {data_root} | Years: {years} | File prefix: {file_prefix}")
    if data_test_root is not None and load_test:
        print(f"Loading test data from: {data_test_root}")

    # 1. 加载训练集
    if load_train and data_root:
        for y in years:
            path = os.path.join(data_root, f'model_ready_panel_selected_plus_ohlc_{file_prefix}_{y}.parquet')
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')
                df['DATA_SOURCE'] = 'train'
                dfs.append(df)
            else:
                print(f"⚠️ 警告: 未找到训练数据: {path}")

    # 2. 加载测试集
    if load_test and data_test_root:
        if os.path.isdir(data_test_root):
            test_files = [f for f in os.listdir(data_test_root) if f.endswith('.parquet')]
            for f in test_files:
                path = os.path.join(data_test_root, f)
                df = pd.read_parquet(path)
                df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')
                df['DATA_SOURCE'] = 'test'
                dfs.append(df)
        else:
            if os.path.exists(data_test_root):
                df = pd.read_parquet(data_test_root)
                df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')
                df['DATA_SOURCE'] = 'test'
                dfs.append(df)

    if not dfs: 
        raise FileNotFoundError("未找到任何 parquet 数据")

    df_final = pd.concat(dfs, ignore_index=True).sort_values(['TRADE_DT', 'S_INFO_WINDCODE']).reset_index(drop=True)

    # Print S_INFO_WINDCODE value counts for debugging
    print("S_INFO_WINDCODE value counts (top 10):")
    print(df_final['S_INFO_WINDCODE'].value_counts().head(10))
    # 🔑 新增：北交所数据过滤逻辑
    if exclude_bj:
        original_count = len(df_final)
        # 确保股票代码为字符串类型，并过滤掉以 '_BJ' 结尾的行
        df_final = df_final[~df_final['S_INFO_WINDCODE'].astype(str).str.endswith('.BJ')].reset_index(drop=True)
        print(f"🚫 已排除北交所 (.BJ) 数据 | 移除行数: {original_count - len(df_final):,} | 剩余行数: {len(df_final):,}")

    return df_final

def extract_valid_features(df: pd.DataFrame) -> List[str]:
    cols = df.columns.tolist()
    valid_features = []
    breadth_pattern = re.compile(r'^Breadth_[^_]+$')
    for col in cols:
        if breadth_pattern.match(col):
            valid_features.append(col)
            continue
        if col.endswith('_MKT_Z') or col.endswith('_IND_Z'):
            valid_features.append(col)
            continue
        if any(kw in col for kw in ['MASK', 'FLAG', 'RATE', 'DATA_SOURCE']):
            continue
    return valid_features

def compute_real_returns(raw_panel_path: str, panel: pd.DataFrame, i: int) -> pd.DataFrame:
    target = 'S_DQ_ADJCLOSE'
    raw = pd.read_parquet(raw_panel_path, columns=['S_INFO_WINDCODE', 'TRADE_DT', target])
    print(f"Loaded target: {target} from {raw_panel_path} | shape: {raw.shape}")
    raw['TRADE_DT'] = pd.to_datetime(raw['TRADE_DT'].astype(str), format='%Y%m%d')
    raw = raw.sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
    raw[f'label_{i}'] = raw.groupby('S_INFO_WINDCODE')[target].pct_change().shift(-i)
    ret_df = raw[['S_INFO_WINDCODE', 'TRADE_DT', target, f'label_{i}']].copy()
    ret_df[target] = ret_df.groupby('S_INFO_WINDCODE')[target].ffill()
    merged = panel.merge(ret_df, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')
    print(f"Merged panel with real returns | shape: {merged.shape} | columns: {merged.columns.tolist()}")
    merged[target] = merged[target].ffill()
    return merged

def compute_ic_ir(factor_cols: List[str], label_col: str, df: pd.DataFrame) -> Dict:
    ic_series = {col: [] for col in factor_cols}
    grouped = df.groupby('TRADE_DT')
    total_days = len(grouped)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, (_, day_df) in enumerate(grouped):
            if label_col not in day_df.columns: continue
            label = day_df[label_col]
            if label.isna().all() or len(day_df) < 50: continue
            for col in factor_cols:
                valid = day_df[[col, label_col]].dropna()
                if len(valid) < 3 or valid[col].std() < 1e-10 or valid[label_col].std() < 1e-10:
                    ic_series[col].append(0.0)
                else:
                    ic = valid[col].corr(valid[label_col], method='spearman')
                    ic_series[col].append(ic if not np.isnan(ic) else 0.0)
        ic_df = pd.DataFrame(ic_series)
    return {col: {"mean_ic": float(ic_df[col].mean()), "icir": float(ic_df[col].mean() / (ic_df[col].std() + 1e-8)), "ic_positive_ratio": float((ic_df[col] > 0).mean()), "sample_days": len(ic_df[col])} for col in factor_cols}