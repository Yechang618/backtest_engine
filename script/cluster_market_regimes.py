# script/cluster_market_regimes.py
import sys
import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.data_loader import load_panel_data, compute_real_returns, compute_derived_factors
from config.Config import Config

# 配置中文字体，防止图表标题或标签显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def compute_monthly_market_features(df_month: pd.DataFrame) -> dict:
    """计算单个月份的全市场宏观特征"""
    # 1. 波动率 (Volatility): 每只股票当月日收益率的std，然后求全市场mean
    df_month = df_month.sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
    df_month['daily_ret'] = df_month.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].pct_change()
    stock_vol = df_month.groupby('S_INFO_WINDCODE')['daily_ret'].std()
    vol = stock_vol.mean()
    
    # 2. 换手率 (Turnover): 日均换手率 (S_DQ_VOLUME / S_DQ_CAPITAL) 的全市场均值
    if 'S_DQ_CAPITAL' in df_month.columns and df_month['S_DQ_CAPITAL'].notna().any():
        # 🔑 防御性修复：防止总股本为0导致除以0产生 inf
        safe_capital = df_month['S_DQ_CAPITAL'].replace(0, np.nan)
        df_month['turnover'] = df_month['S_DQ_VOLUME'] / safe_capital
        turnover = df_month['turnover'].mean()
    else:
        turnover = np.nan
        
    # 3. 成交额 (Amount): 当月全市场总成交额
    amount = df_month['S_DQ_AMOUNT'].sum() if 'S_DQ_AMOUNT' in df_month.columns else np.nan
    
    # 4. 流通市值 (Market Cap): 当月最后一个交易日的全市场总流通市值
    last_date = df_month['TRADE_DT'].max()
    df_last_day = df_month[df_month['TRADE_DT'] == last_date]
    if 'S_DQ_CAPITAL' in df_last_day.columns:
        mcap = (df_last_day['S_DQ_CLOSE'] * df_last_day['S_DQ_CAPITAL']).sum()
    else:
        mcap = np.nan
        
    return {'vol': vol, 'turnover': turnover, 'amount': amount, 'mcap': mcap}

def main(n_clusters=3):
    setup_logging()
    cfg = Config()
    
    # 🔑 核心修复：直接从 RAW_PANEL 读取所需列，避免因子面板中缺失原始量价字段导致全 NaN
    logging.info("📦 加载原始行情数据 (RAW_PANEL) 用于计算宏观特征...")
    required_cols = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJCLOSE', 'S_DQ_CLOSE', 
                     'S_DQ_VOLUME', 'S_DQ_AMOUNT', 'S_DQ_CAPITAL']
    
    try:
        df = pd.read_parquet(cfg.RAW_PANEL, columns=required_cols)
    except Exception as e:
        logging.error(f"❌ 读取 RAW_PANEL 失败: {e}")
        return
        
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')
    
    # 限制全局时间范围
    df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-06-01')) & (df['TRADE_DT'] <= pd.to_datetime('2026-03-31'))].copy()
    df['YEAR_MONTH'] = df['TRADE_DT'].dt.strftime('%Y-%m')
    
    logging.info(f"✅ 原始行情数据加载完成 | 形状: {df.shape}")
    logging.info("🧮 开始计算月度全市场宏观特征...")

    monthly_features = []
    for ym, group in df.groupby('YEAR_MONTH'):
        feats = compute_monthly_market_features(group)
        feats['YEAR_MONTH'] = ym
        monthly_features.append(feats)
        
    df_market = pd.DataFrame(monthly_features).sort_values('YEAR_MONTH').reset_index(drop=True)
    
    # 计算环比增长率 (去除趋势影响，使聚类更关注市场状态的变化)
    df_market['amount_ret'] = df_market['amount'].pct_change()
    df_market['mcap_ret'] = df_market['mcap'].pct_change()
    
    # 聚类使用的4个核心特征
    cluster_cols = ['vol', 'turnover', 'amount_ret', 'mcap_ret']
    df_market_clean = df_market.dropna(subset=cluster_cols).reset_index(drop=True)
    
    # 划分训练期与验证期
    train_mask = df_market_clean['YEAR_MONTH'] <= '2024-12'
    val_mask = df_market_clean['YEAR_MONTH'] >= '2025-01'
    
    X_train = df_market_clean.loc[train_mask, cluster_cols].values
    X_val = df_market_clean.loc[val_mask, cluster_cols].values
    
    logging.info(f"🧠 正在训练 KMeans (n_clusters={n_clusters})...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    
    # 预测并生成标签字典
    train_labels = kmeans.predict(X_train_scaled)
    val_labels = kmeans.predict(X_val_scaled)
    
    regime_labels = {}
    for i, ym in enumerate(df_market_clean.loc[train_mask, 'YEAR_MONTH']):
        regime_labels[ym] = int(train_labels[i])
    for i, ym in enumerate(df_market_clean.loc[val_mask, 'YEAR_MONTH']):
        regime_labels[ym] = int(val_labels[i])
        
    # ==========================================
    # 🔑 新增：可视化聚类结果 (2D 散点图)
    # ==========================================
    logging.info("📊 正在生成聚类结果 2D 散点图...")
    
    # 1. 构建用于绘图的完整 DataFrame (包含特征和聚类标签)
    df_plot = df_market_clean.copy()
    df_plot['cluster_id'] = df_plot['YEAR_MONTH'].map(regime_labels)
    
    # 2. 计算每个聚类特征对 cluster_id 的区分度 (使用 ANOVA F-value)
    # F-value 越大，说明该特征在不同簇之间的差异越显著，区分度越高
    X_features = df_plot[cluster_cols].values
    y_labels = df_plot['cluster_id'].values
    f_values, _ = f_classif(X_features, y_labels)
    
    # 3. 选出 F-value 最高的两个特征作为 X 轴和 Y 轴
    feature_scores = list(zip(cluster_cols, f_values))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    x_feature = feature_scores[0][0]
    y_feature = feature_scores[1][0]
    logging.info(f"  ✅ 区分度最高的两个特征: X轴='{x_feature}' (F={f_values[cluster_cols.index(x_feature)]:.2f}), Y轴='{y_feature}' (F={f_values[cluster_cols.index(y_feature)]:.2f})")
    
    # 4. 绘制散点图
    plt.figure(figsize=(10, 7))
    
    # 使用 viridis 颜色映射，确保不同 cluster 颜色区分明显

    cmap = mpl.colormaps['viridis'].resampled(n_clusters)

    # cmap = plt.cm.get_cmap('viridis', n_clusters)
    scatter = plt.scatter(
        df_plot[x_feature], 
        df_plot[y_feature], 
        c=df_plot['cluster_id'], 
        cmap=cmap, 
        alpha=0.7, 
        edgecolors='w', 
        s=80
    )
    
    # 添加图例
    legend_labels = [f'Cluster {i}' for i in range(n_clusters)]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Market Regime", loc='best')
    
    # 设置标题和轴标签 (使用中文字体)
    plt.title(f'Market Regime Clustering (KMeans, K={n_clusters})', fontsize=14, fontweight='bold')
    plt.xlabel(x_feature.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_feature.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 保存图像
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    plot_path = os.path.join(cfg.FIG_DIR, f"market_regime_clustering_{n_clusters}clusters.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"  ✅ 聚类散点图已保存至: {plot_path}")
    # ==========================================
    
    # 保存模型与标签
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(cfg.MODEL_DIR, "cluster_model.pkl")
    joblib.dump({'scaler': scaler, 'kmeans': kmeans, 'cols': cluster_cols}, model_path)
    
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    labels_path = cfg.OUT_DIR / "market_regime_labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(regime_labels, f, indent=4)
        
    logging.info(f"✅ 聚类模型已保存至: {model_path}")
    logging.info(f"✅ 市场状态标签已保存至: {labels_path}")
    
    # 打印聚类结果分布
    for c in range(n_clusters):
        count = list(regime_labels.values()).count(c)
        logging.info(f"  - Cluster {c}: {count} 个月份")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', type=int, default=3, help='聚类数量')
    args = parser.parse_args()
    main(n_clusters=args.n_clusters)