import sys
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.Config import Config

# 配置中文字体，确保图表中的中文（如有）或负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def main():
    setup_logging()
    cfg = Config()
    
    logging.info("📦 加载原始行情数据 (RAW_PANEL)...")
    # 仅加载必要的列以大幅节省内存和加载时间
    cols = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJCLOSE']
    try:
        df = pd.read_parquet(cfg.RAW_PANEL, columns=cols)
    except Exception as e:
        logging.error(f"❌ 读取 RAW_PANEL 失败: {e}")
        return
        
    # 确保日期格式正确并排序
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')
    df = df.sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
    
    # 限制时间范围：2015年至今
    df = df[df['TRADE_DT'].dt.year >= 2015].copy()
    
    # 提取年份用于分组
    df['YEAR'] = df['TRADE_DT'].dt.year
    
    logging.info("🧮 开始计算各股票每年的日收益率...")
    # 计算日收益率
    df['DAILY_RET'] = df.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].pct_change()
    
    logging.info("🧮 开始聚合计算年化预期收益与年化波动率...")
    # 按年份和股票代码分组计算统计量
    grouped = df.groupby(['YEAR', 'S_INFO_WINDCODE'])['DAILY_RET']
    
    stats = grouped.agg(
        mean_ret=('mean'),
        std_ret=('std'),
        count=('count')  # 记录当年的有效交易日数，用于过滤异常数据
    ).reset_index()
    
    # 🔑 数据清洗：过滤掉一年内有效交易日少于 100 天的股票
    # (避免因长期停牌、退市或当年新股导致收益率和波动率计算失真)
    stats = stats[stats['count'] >= 100].copy()
    
    # 计算年化指标 (假设每年 252 个交易日)
    stats['ANNUALIZED_RETURN'] = stats['mean_ret'] * 252
    stats['ANNUALIZED_VOL'] = stats['std_ret'] * np.sqrt(252)
    
    # 确保输出目录存在
    fig_dir = cfg.FIG_DIR
    os.makedirs(fig_dir, exist_ok=True)
    
    years = sorted(stats['YEAR'].unique())
    logging.info(f"✅ 计算完成，共覆盖 {len(years)} 个年份: {years}")
    
    for year in years:
        year_data = stats[stats['YEAR'] == year]
        
        plt.figure(figsize=(10, 8))
        
        # 绘制散点图：X轴为波动率，Y轴为预期收益，每只股票为一个点
        plt.scatter(
            year_data['ANNUALIZED_VOL'], 
            year_data['ANNUALIZED_RETURN'], 
            alpha=0.6, 
            s=15,          # 点的大小
            c='teal',      # 点的颜色
            edgecolors='none'
        )
        
        # 添加标题和轴标签 (标注年份)
        plt.title(f'Efficient Frontier Analysis: Expected Return vs. Volatility ({year})', fontsize=14, fontweight='bold')
        plt.xlabel('Annualized Volatility', fontsize=12)
        plt.ylabel('Annualized Expected Return', fontsize=12)
        
        # 添加参考线和网格
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 保存图片 (图片名标注年份)
        save_path = os.path.join(fig_dir, f'efficient_frontier_{year}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"📊 {year} 年散点图已保存至: {save_path} (有效样本数: {len(year_data)})")
        
    logging.info("🎉 所有年份的有效前沿散点图生成完毕！")

if __name__ == "__main__":
    main()