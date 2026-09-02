# script/analyze_backtest_results.py
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.Config import Config

def main():
    cfg = Config()
    out_dir = cfg.OUT_DIR
    
    summary_path = out_dir / "backtest_summary.json"
    ic_path = out_dir / "daily_ic_results.json"
    resid_ic_path = out_dir / "daily_resid_ic_results.json"
    
    if not summary_path.exists():
        print(f"❌ 找不到回测结果文件: {summary_path}")
        print("请先运行 python script/run_backtest.py")
        return

    # 1. 加载综合绩效数据
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    models_perf = summary_data.get("models", {})
    
    # 2. 加载每日 Rank IC 数据 (如果存在)
    daily_ic_data = {}
    if ic_path.exists():
        with open(ic_path, 'r', encoding='utf-8') as f:
            daily_ic_data = json.load(f)
    else:
        print("⚠️ 未找到 daily_ic_results.json，平均 Rank IC 将显示为 N/A。")
        print("   提示：请在 run_backtest.py 中添加保存 daily_ic_results 的代码并重新运行回测。\n")

    # 3. 加载每日 Residual Rank IC 数据 (如果存在)
    daily_resid_ic_data = {}
    if resid_ic_path.exists():
        with open(resid_ic_path, 'r', encoding='utf-8') as f:
            daily_resid_ic_data = json.load(f)

    # 4. 构建分析 DataFrame
    results = []
    for model_name, perf in models_perf.items():
        # 计算平均 Rank IC
        avg_ic = np.nan
        if model_name in daily_ic_data and len(daily_ic_data[model_name]) > 0:
            ics = [record['IC'] for record in daily_ic_data[model_name]]
            avg_ic = np.mean(ics)
            
        # 计算平均 Residual Rank IC (针对 SensitiveSwitch 等)
        avg_resid_ic = np.nan
        if model_name in daily_resid_ic_data and len(daily_resid_ic_data[model_name]) > 0:
            resid_ics = [record['IC'] for record in daily_resid_ic_data[model_name]]
            avg_resid_ic = np.mean(resid_ics)

        results.append({
            "Model": model_name,
            "Avg Rank IC": avg_ic,
            "Avg Resid IC": avg_resid_ic,
            "Annual Return (PnL)": perf.get("annual_return", 0.0),
            "Total Return": perf.get("total_return", 0.0),
            "Sharpe Ratio": perf.get("sharpe_ratio", 0.0),
            "Max Drawdown": perf.get("max_drawdown", 0.0),
            "Win Rate (%)": perf.get("trade_statistics", {}).get("win_rate_pct", 0.0),
            "Closed Trades": perf.get("trade_statistics", {}).get("total_closed_trades", 0)
        })
    
    df = pd.DataFrame(results)
    
    # 5. 格式化输出
    print("\n" + "="*110)
    print("📊 回测全过程策略表现汇总 (按年化收益降序排列)")
    print("="*110)
    
    # 按年化收益降序排序以便查看
    df = df.sort_values(by="Annual Return (PnL)", ascending=False).reset_index(drop=True)
    
    # 格式化浮点数显示
    pd.set_option('display.float_format', lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    
    # 调整列顺序以便阅读
    display_cols = [
        "Model", "Avg Rank IC", "Avg Resid IC", "Annual Return (PnL)", 
        "Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate (%)", "Closed Trades"
    ]
    
    # 打印表格
    print(df[display_cols].to_string(index=False))
    print("="*110 + "\n")
    
    # 6. 保存为 CSV 方便后续在 Excel 中分析
    csv_path = out_dir / "backtest_results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ 详细汇总数据已保存至 CSV: {csv_path}")

if __name__ == "__main__":
    main()