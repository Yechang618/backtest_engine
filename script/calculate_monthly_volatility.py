# script/calculate_monthly_volatility.py
import sys
import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.Config import Config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def main():
    setup_logging()
    cfg = Config()
    
    logging.info("📦 正在加载原始行情数据...")
    if not os.path.exists(cfg.RAW_PANEL):
        logging.error(f"❌ 找不到 RAW_PANEL 文件: {cfg.RAW_PANEL}")
        return
        
    # 仅加载所需列以节省内存
    df = pd.read_parquet(cfg.RAW_PANEL, columns=['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'SW_L1_CODE'])
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')
    
    # 1. 过滤日期范围: 2015-01-01 到 2024-12-31
    start_date = pd.to_datetime('2015-01-01')
    end_date = pd.to_datetime('2024-12-31')
    df = df[(df['TRADE_DT'] >= start_date) & (df['TRADE_DT'] <= end_date)].copy()
    logging.info(f"✅ 日期过滤完成，剩余记录数: {len(df):,}")
    
    # 2. 计算日收益率
    df = df.sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
    df['RET'] = df.groupby('S_INFO_WINDCODE')['S_DQ_CLOSE'].pct_change()
    
    # 3. 提取年月 (格式: 'YYYY-MM')
    df['YEAR_MONTH'] = df['TRADE_DT'].dt.to_period('M').astype(str)
    
    # 4. 计算每只股票每月的波动率 (日收益率标准差)
    stock_monthly_vol = df.groupby(['S_INFO_WINDCODE', 'SW_L1_CODE', 'YEAR_MONTH'])['RET'].std().reset_index()
    stock_monthly_vol = stock_monthly_vol.dropna(subset=['RET']) # 剔除无效数据
    
    # 5. 计算每个行业每月的平均波动率
    industry_monthly_vol = stock_monthly_vol.groupby(['SW_L1_CODE', 'YEAR_MONTH'])['RET'].mean().reset_index()
    industry_monthly_vol.rename(columns={'RET': 'VOLATILITY'}, inplace=True)
    
    # 6. 计算总体(All)每月的平均波动率
    all_monthly_vol = stock_monthly_vol.groupby('YEAR_MONTH')['RET'].mean().reset_index()
    all_monthly_vol['SW_L1_CODE'] = 'All'
    all_monthly_vol.rename(columns={'RET': 'VOLATILITY'}, inplace=True)
    
    # 7. 合并行业与总体结果
    final_df = pd.concat([industry_monthly_vol, all_monthly_vol], ignore_index=True)
    final_df = final_df.sort_values(['SW_L1_CODE', 'YEAR_MONTH'])
    
    # 8. 转换为嵌套字典格式以便保存为 JSON
    # 格式: {"SW_L1_CODE": {"YYYY-MM": volatility, ...}, ...}
    result_dict = {}
    for code, group in final_df.groupby('SW_L1_CODE'):
        result_dict[str(code)] = {row['YEAR_MONTH']: float(row['VOLATILITY']) for _, row in group.iterrows()}
        
    # 9. 保存到 OUT_DIR
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.OUT_DIR / "monthly_volatility.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
        
    logging.info(f"💾 每月波动率数据已成功保存至: {out_path}")

if __name__ == "__main__":
    main()