# /data/cye_temp/workspace/backtest_engine/util/metrics.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional

def compute_rolling_sharpe(values: pd.Series, window: int = 30) -> pd.Series:
    daily_ret = values.pct_change()
    rolling_mean = daily_ret.rolling(window).mean()
    rolling_std = daily_ret.rolling(window).std()
    sharpe = (rolling_mean / (rolling_std + 1e-8)) * np.sqrt(252)
    return sharpe
def evaluate_and_plot(results: Dict[str, pd.DataFrame], output_dir: str, figure_dir: str, ic_ir_dict: Optional[Dict] = None, start_date='', TOP_K=50):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    
    df = None
    for name, hist in results.items():
        col = f'Value_{name}'
        if df is None:
            df = hist.rename(columns={'Value': col})
        else:
            df = df.merge(hist[['TRADE_DT', 'Value']], on='TRADE_DT', how='outer')
            df = df.rename(columns={'Value': col})

    if df is None or df.empty:
        print("⚠️ 无有效回测数据，跳过绘图")
        return {}

    df = df.dropna(subset=['TRADE_DT']).sort_values('TRADE_DT').reset_index(drop=True)
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])

    # 🔑 强化样本外评估：强制只绘制和计算 2025-01-01 之后的绩效 (即测试集阶段)
    # 如果您的测试集起始日不同，请修改此处的日期
    test_start_date = pd.to_datetime(start_date) if start_date else pd.to_datetime('2025-01-01')
    df_test = df[df['TRADE_DT'] >= test_start_date].copy()
    
    if df_test.empty:
        print("⚠️ 测试集时间段内无有效数据，跳过绘图")
        return {}

    palette = plt.cm.tab10.colors
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(results.keys())}
    metrics_summary = {}

    # ─────────────────────────────────────────────────────────────
    # 图 1: 累计净值曲线 (归一化至测试期起点 1.0)
    # ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 6))
    for name in results.keys():
        col = f'Value_{name}'
        if col not in df_test.columns or df_test[col].isna().all():
            continue
            
        init_val = df_test[col].iloc[0] if df_test[col].iloc[0] > 0 else 1.0
        pnl = df_test[col] / init_val
        plt.plot(df_test['TRADE_DT'], pnl, label=name, color=color_map[name], lw=1.5)
        
        rets = df_test[col].pct_change().dropna()
        if len(rets) < 2:
            continue
            
        cum_ret = df_test[col].iloc[-1] / init_val - 1
        days = len(rets)
        ann_ret = (1 + cum_ret) ** (252 / days) - 1 if days > 0 else 0.0
        vol = rets.std() * np.sqrt(252)
        max_dd = ((df_test[col] - df_test[col].cummax()) / df_test[col].cummax()).min()
        sharpe = ann_ret / (vol + 1e-8) if vol > 1e-8 else 0.0
        
        metrics_summary[name] = {
            'total_return': float(cum_ret),
            'annual_return': float(ann_ret),
            'volatility': float(vol),
            'max_drawdown': float(max_dd) if not np.isnan(max_dd) else 0.0,
            'sharpe': float(sharpe)
        }

    plt.title('Out-of-Sample (Test Set) Cumulative PnL (Normalized)')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'pnl_combined_top{TOP_K}_{start_date}.png'), dpi=150)
    plt.close()
    
    # ... (后续 Drawdown 和 Rolling Sharpe 的绘图逻辑同样将 df 替换为 df_test 即可，此处省略以保持简洁，您只需在原代码对应位置将 df 改为 df_test)
    # ─────────────────────────────────────────────────────────────
    # 图 2: 回撤对比图
    # ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 4))
    for name in results.keys():
        col = f'Value_{name}'
        if col not in df_test.columns: continue
        dd = (df_test[col] - df_test[col].cummax()) / df_test[col].cummax()
        plt.fill_between(df_test['TRADE_DT'], dd, 0, color=color_map[name], alpha=0.4, label=f'{name} DD')
    plt.title('Strategy Drawdown Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'drawdown_combined_top{TOP_K}_{start_date}.png'), dpi=150)
    plt.close()

    # ─────────────────────────────────────────────────────────────
    # 图 3: 30 日滚动夏普比率
    # ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 4))
    for name in results.keys():
        col = f'Value_{name}'
        if col not in df_test.columns or len(df_test[col]) < 30: continue
        r_sharpe = compute_rolling_sharpe(df_test[col], window=30)
        valid_mask = r_sharpe.notna()
        if valid_mask.any():
            plt.plot(df_test['TRADE_DT'][valid_mask], r_sharpe[valid_mask], 
                    label=name, color=color_map[name], lw=1.2)
    plt.axhline(0, color='k', linestyle='--', lw=0.8, alpha=0.5)
    plt.title('30-Day Rolling Sharpe Ratio')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'rolling_sharpe_combined_top{TOP_K}_{start_date}.png'), dpi=150)
    plt.close()

    # ─────────────────────────────────────────────────────────────
    # 保存 IC/IR 报告（若提供）
    # ─────────────────────────────────────────────────────────────
    if ic_ir_dict:
        ic_path = os.path.join(output_dir, f'factor_ic_ir_top{TOP_K}_{start_date}.json')
        with open(ic_path, 'w', encoding='utf-8') as f:
            json.dump(ic_ir_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ IC/IR 报告已保存至: {ic_path}")

    return metrics_summary