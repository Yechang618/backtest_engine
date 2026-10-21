# script/corr_analysis.py
import sys
import os
import json
import glob
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features, compute_derived_factors
from config.Config import Config

# 配置中文字体，确保热力图标签正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def load_trade_pools(pool_dir: str) -> dict:
    """加载股票池，返回 {月份: 股票代码集合}"""
    trade_pools = {}
    if not os.path.exists(pool_dir): return trade_pools
    csv_files = glob.glob(os.path.join(pool_dir, "trade_pool_2026*.csv"))
    for f in csv_files:
        try:
            df_pool = pd.read_csv(f, usecols=['effective_month', 'code'], dtype={'code': str})
            for month, group in df_pool.groupby('effective_month'):
                month_str = str(month).strip()
                if month_str not in trade_pools: trade_pools[month_str] = set()
                trade_pools[month_str].update(set(group['code'].str.strip().tolist()))
        except Exception as e:
            logging.warning(f"⚠️ 读取股票池文件失败 {f}: {e}")
    return trade_pools

def get_model_features(model_name: str, all_feature_cols: list, ablation_features: dict, cfg: Config) -> list:
    """复用回测引擎的特征路由逻辑"""
    # 1. 优先检查是否为消融模型
    if model_name in ablation_features:
        return [f for f in ablation_features[model_name] if f in all_feature_cols]
    
    # 2. 回退到基于前缀的全量特征
    if 'XGB' in model_name or 'XGBoost' in model_name:
        return getattr(cfg, 'FEATURE_COLS_XGB', all_feature_cols)
    elif 'LGBM' in model_name or 'LightGBM' in model_name:
        return getattr(cfg, 'FEATURE_COLS_LGBM', all_feature_cols)
    
    # 3. 默认返回全量特征
    return all_feature_cols

def generate_daily_preds(df: pd.DataFrame, trainers: dict, cfg: Config) -> pd.DataFrame:
    """为给定数据集生成每日各模型的预测值 DataFrame"""
    all_feature_cols = extract_valid_features(df)
    
    # 加载消融特征配置
    ablation_features = {}
    if hasattr(cfg, 'ABLATION_FEATURE_JSON') and os.path.exists(cfg.ABLATION_FEATURE_JSON):
        with open(cfg.ABLATION_FEATURE_JSON, 'r', encoding='utf-8') as f:
            raw_ablation = json.load(f)
        for model_name, feats in raw_ablation.items():
            ablation_features[f"{model_name}_ablation"] = feats

    # 存储每日预测结果
    daily_preds_list = []
    
    # 按日期分组计算，节省内存且符合截面分析逻辑
    for date, group in df.groupby('TRADE_DT'):
        if len(group) < 10:  # 样本太少跳过
            continue
            
        day_preds = {'TRADE_DT': date}
        for model_name, model in trainers.items():
            feats = get_model_features(model_name, all_feature_cols, ablation_features, cfg)
            valid_feats = [f for f in feats if f in group.columns]
            
            try:
                # 预测并填入当日结果
                preds = model.predict(group[valid_feats])
                day_preds[model_name] = np.mean(preds) # 占位，实际我们需要保留每只股票的预测值以计算相关性
                # 修正：我们需要保留每只股票的预测值，所以不在这聚合
            except Exception as e:
                logging.warning(f"模型 {model_name} 在 {date} 预测失败: {e}")
                day_preds[model_name] = np.nan
        
        # 重新构建：直接对整个 group 进行 predict，然后 merge
        # 为了效率，直接对 group 操作
        pass # 见下方优化实现

    # 🔑 优化实现：直接对 DataFrame 进行批量预测，保留 (TRADE_DT, S_INFO_WINDCODE) 索引
    preds_records = []
    for model_name, model in trainers.items():
        feats = get_model_features(model_name, all_feature_cols, ablation_features, cfg)
        valid_feats = [f for f in feats if f in df.columns]
        try:
            preds = model.predict(df[valid_feats])
            # 保存为 Series 以便后续按日期分组
            preds_series = pd.Series(preds, index=df.index, name=model_name)
            preds_records.append(preds_series)
        except Exception as e:
            logging.warning(f"模型 {model_name} 批量预测失败: {e}")
            
    if not preds_records:
        return pd.DataFrame()
        
    preds_df = pd.concat(preds_records, axis=1)
    preds_df['TRADE_DT'] = df['TRADE_DT'].values
    return preds_df

def calc_mean_daily_corr(preds_df: pd.DataFrame) -> pd.DataFrame:
    """计算每日截面相关性矩阵，并返回时间序列均值"""
    daily_corrs = []
    
    for date, group in preds_df.groupby('TRADE_DT'):
        if len(group) >= 10:  # 至少 10 只股票才计算当日相关性
            # 剔除全 NaN 的列
            valid_group = group.drop(columns=['TRADE_DT']).dropna(axis=1, how='all')
            if valid_group.shape[1] >= 2:
                corr_mat = valid_group.corr(method='pearson')
                daily_corrs.append(corr_mat)
                
    if not daily_corrs:
        logging.warning("⚠️ 无有效日期可计算相关性矩阵")
        return pd.DataFrame()
        
    # 对所有日期的相关矩阵求和，然后除以天数 (处理可能缺失的模型列)
    sum_corr = daily_corrs[0]
    for c in daily_corrs[1:]:
        sum_corr = sum_corr.add(c, fill_value=0)
        
    mean_corr_matrix = sum_corr / len(daily_corrs)
    return mean_corr_matrix

def plot_and_save_corr(corr_matrix: pd.DataFrame, dataset_name: str, out_dir: str, fig_dir: str):
    """保存相关性矩阵 CSV 并绘制高可读性热力图"""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. 保存 CSV
    csv_path = os.path.join(out_dir, f"model_corr_matrix_{dataset_name}.csv")
    corr_matrix.to_csv(csv_path)
    logging.info(f"✅ [{dataset_name}] 相关性矩阵已保存至: {csv_path}")
    
    # 2. 绘制热力图
    plt.figure(figsize=(14, 12))  # 增大画布确保标签可读
    
    # 如果模型数量 <= 20，显示数值；否则隐藏数值以防拥挤
    show_annot = len(corr_matrix) <= 20
    
    sns.heatmap(
        corr_matrix, 
        annot=show_annot, 
        fmt=".2f" if show_annot else "", 
        cmap='coolwarm', 
        vmin=-1.0, 
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'}
    )
    
    plt.title(f'Model Prediction Correlation Matrix\n({dataset_name} - Mean Daily Cross-Sectional)', fontsize=14, fontweight='bold')
    
    # 优化坐标轴标签可读性
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f"model_corr_heatmap_{dataset_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ [{dataset_name}] 相关性热力图已保存至: {fig_path}")

def main(use_trade_pool: bool = False):
    setup_logging()
    cfg = Config()
    
    logging.info("📦 1. 加载全量面板数据...")
    try:
        df = load_panel_data(None, cfg.DATA_TEST_DIR, [], load_train=False, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    except TypeError:
        df = load_panel_data(cfg.DATA_DIR, cfg.DATA_TEST_DIR, list(range(2022, 2027)), file_prefix="train", exclude_bj=cfg.EXCLUDE_BJ)
    
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE')
    
    # 确保 TRADE_DT 为 datetime 类型
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
    
    logging.info("📦 2. 加载所有预训练模型...")
    trainers = {}
    model_dir = str(cfg.MODEL_DIR)
    pkl_files = ["sklearn_models.pkl", "percentile_sklearn_models.pkl", "cluster_sklearn_models.pkl", "ablation_sklearn_models.pkl"]
    
    for pkl in pkl_files:
        path = os.path.join(model_dir, pkl)
        if os.path.exists(path):
            trainers.update(joblib.load(path))
            
    if not trainers:
        raise RuntimeError("❌ 未找到任何预训练模型，请先运行训练脚本。")
    logging.info(f"✅ 成功加载 {len(trainers)} 个模型。")
    
    # 加载股票池 (用于测试集过滤)
    trade_pools = load_trade_pools(cfg.TRADE_POOL_DIR) if use_trade_pool else None
    
    # ==========================================
    # 3. 数据集切片与过滤
    # ==========================================
    datasets = {}
    
    # A. 训练集: <= 2024-12-31, 所有股票
    train_df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-05-01')) & (df['TRADE_DT'] <= pd.to_datetime('2024-12-31'))].copy()
    datasets['Train_Set'] = train_df
    
    # B. 验证集: 2025-01-01 至 2026-03-31, 所有股票
    val_df = df[(df['TRADE_DT'] >= pd.to_datetime('2025-01-01')) & (df['TRADE_DT'] <= pd.to_datetime('2026-03-31'))].copy()
    datasets['Val_Set'] = val_df
    
    # C. 测试集: >= 2026-04-01, 根据条件过滤
    test_df = df[df['TRADE_DT'] >= pd.to_datetime('2026-04-01')].copy()
    if use_trade_pool and trade_pools:
        logging.info("🔒 测试集应用股票池硬约束过滤...")
        valid_codes = set()
        for codes in trade_pools.values():
            valid_codes.update(codes)
        test_df = test_df[test_df['S_INFO_WINDCODE'].isin(valid_codes)].copy()
    elif 'BUY_MASK' in test_df.columns:
        logging.info("🔒 测试集应用 BUY_MASK == 1 过滤...")
        test_df = test_df[test_df['BUY_MASK'] == 1].copy()
    else:
        logging.info("ℹ️ 测试集无特殊过滤条件，使用所有股票。")
        
    datasets['Test_Set'] = test_df
    
    # ==========================================
    # 4. 计算并可视化相关性
    # ==========================================
    for ds_name, ds_df in datasets.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"🚀 正在处理数据集: {ds_name} (样本数: {len(ds_df):,})")
        logging.info(f"{'='*60}")
        
        if len(ds_df) == 0:
            logging.warning(f"⚠️ {ds_name} 为空，跳过。")
            continue
            
        # 生成预测值
        preds_df = generate_daily_preds(ds_df, trainers, cfg)
        
        # 计算每日截面相关性均值
        corr_matrix = calc_mean_daily_corr(preds_df)
        
        if not corr_matrix.empty:
            plot_and_save_corr(corr_matrix, ds_name, str(cfg.OUT_DIR), str(cfg.FIG_DIR))
        else:
            logging.warning(f"⚠️ {ds_name} 未能生成有效的相关性矩阵。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析各模型预测值的相关性")
    parser.add_argument('--use_trade_pool', action='store_true', help='在测试集中启用股票池过滤')
    args = parser.parse_args()
    
    main(use_trade_pool=args.use_trade_pool)