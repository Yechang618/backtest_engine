# script/visualize_volatility.py
import sys
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.Config import Config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def main():
    setup_logging()
    cfg = Config()
    
    json_path = cfg.OUT_DIR / "monthly_volatility.json"
    if not os.path.exists(json_path):
        logging.error(f"❌ 找不到波动率数据文件: {json_path}，请先运行 calculate_monthly_volatility.py")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        vol_data = json.load(f)
        
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    percentiles_result = {}
    
    logging.info("📈 开始生成波动率曲线与百分位统计...")
    for sw_code, month_vol_dict in vol_data.items():
        # 转换为 DataFrame 方便绘图和计算
        df = pd.DataFrame(list(month_vol_dict.items()), columns=['YEAR_MONTH', 'VOLATILITY'])
        # 将 'YYYY-MM' 转为 datetime 以便 matplotlib 正确绘制时间轴
        df['DATE'] = pd.to_datetime(df['YEAR_MONTH'] + '-01') 
        df = df.sort_values('DATE')
        
        # 1. 绘制月均波动率曲线
        plt.figure(figsize=(12, 5))
        plt.plot(df['DATE'], df['VOLATILITY'], marker='o', markersize=3, lw=1.5, color='teal')
        plt.title(f'Monthly Volatility - Industry: {sw_code}')
        plt.xlabel('Date')
        plt.ylabel('Volatility (Daily Return Std)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        fig_path = cfg.FIG_DIR / f"volatility_curve_{sw_code}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logging.info(f"  ✅ 曲线已保存: volatility_curve_{sw_code}.png")
        
        # 2. 计算百分位 (10%, 20%, ..., 90%)
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        p_values = np.percentile(df['VOLATILITY'].dropna(), percentiles).tolist()
        
        percentiles_result[sw_code] = {
            f"p{p}": float(v) for p, v in zip(percentiles, p_values)
        }
        
    # 3. 保存百分位结果到 JSON (供后续脚本调用)
    pct_json_path = cfg.OUT_DIR / "volatility_percentiles.json"
    with open(pct_json_path, 'w', encoding='utf-8') as f:
        json.dump(percentiles_result, f, indent=4, ensure_ascii=False)
        
    logging.info(f"💾 波动率百分位数据已成功保存至: {pct_json_path}")

if __name__ == "__main__":
    main()