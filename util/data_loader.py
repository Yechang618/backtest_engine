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

def compute_derived_factors(df: pd.DataFrame, price_col: str = 'S_DQ_ADJCLOSE') -> pd.DataFrame:
    """
    计算并增加6个衍生因子：过去10/20/30天的平均收益率与夏普比率。
    计算后自动进行截面标准化(_MKT_Z)，以兼容 extract_valid_features 的提取规则。
    """
    print(f"🧮 开始计算衍生因子 (动量与夏普) | 使用价格列: {price_col}...")
    
    # 确保按股票和日期排序，这是计算时间序列滚动指标的前提
    df = df.sort_values(['S_INFO_WINDCODE', 'TRADE_DT']).copy()
    
    # 1. 计算日收益率
    df['_daily_ret'] = df.groupby('S_INFO_WINDCODE')[price_col].pct_change()
    
    raw_cols = []
    # 2. 计算滚动指标 (按股票分组)
    for window in [10, 20, 30]:
        # 过去 N 天的平均收益率
        mean_ret = df.groupby('S_INFO_WINDCODE')['_daily_ret'].transform(
            lambda x: x.rolling(window=window, min_periods=window).mean()
        )
        raw_mean_col = f'RET_MEAN_{window}D_RAW'
        df[raw_mean_col] = mean_ret
        raw_cols.append(raw_mean_col)
        
        # 过去 N 天的夏普比率 (Mean / Std)
        std_ret = df.groupby('S_INFO_WINDCODE')['_daily_ret'].transform(
            lambda x: x.rolling(window=window, min_periods=window).std()
        )
        # 防止除以 0 (例如停牌或连续几天价格不变)
        sharpe = mean_ret / (std_ret + 1e-8)
        raw_sharpe_col = f'SHARPE_{window}D_RAW'
        df[raw_sharpe_col] = sharpe
        raw_cols.append(raw_sharpe_col)
        
    # 3. 截面标准化 (按 TRADE_DT 分组进行 Z-score 标准化)
    print("  🔄 正在进行截面标准化 (Z-score)...")
    for col in raw_cols:
        target_col = col.replace('_RAW', '_MKT_Z')
        
        # 截面去极值与标准化 (使用 transform 保持原 DataFrame 形状)
        grouped = df.groupby('TRADE_DT')[col]
        mean = grouped.transform('mean')
        std = grouped.transform('std')
        
        # 标准化并处理极小标准差的情况
        df[target_col] = (df[col] - mean) / (std + 1e-8)
        
    # 4. 清理中间变量
    df.drop(columns=['_daily_ret'] + raw_cols, inplace=True)
    
    new_factors = [c.replace('_RAW', '_MKT_Z') for c in raw_cols]
    print(f"  ✅ 衍生因子计算完成！新增特征: {new_factors}")
    
    return df

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