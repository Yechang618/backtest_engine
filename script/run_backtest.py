# /data/cye_temp/workspace/backtest_engine/src/backtest_engine.py
import sys
import os
import json
import glob
import logging
import datetime
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features, compute_derived_factors
from src.backtest_engine import BacktestEngine
from util.metrics import evaluate_and_plot
from config.Config import Config
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def load_trade_pools(pool_dir: str) -> Dict[str, Set[str]]:
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

def plot_daily_ic(daily_ic_results, dynamic_switch_history, figure_dir, start_date='', reba_wd='', file_suffix=''):
    if not any(daily_ic_results.values()):
        logging.warning("⚠️ 无有效每日 IC 数据，跳过绘图。")
        return
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6)) # 合并模式下不画下方的阶梯图
    for name, records in daily_ic_results.items():
        if not records: continue
        df_ic = pd.DataFrame(records).set_index('TRADE_DT')
        df_ic['IC_MA10'] = df_ic['IC'].rolling(window=10, min_periods=1).mean()
        ax1.plot(df_ic.index, df_ic['IC_MA10'], label=f'{name} (10D MA)', lw=1.5)
    ax1.set_title(f'Daily Cross-Sectional Rank IC (Merged 5 Weekdays) - {start_date}', fontsize=14)
    ax1.set_ylabel('Rank IC', fontsize=12)
    ax1.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize='small')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f'daily_rank_ic_trend_{start_date}_{reba_wd}{file_suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ 合并 Rank IC 趋势图已保存至: {path}")

def plot_daily_resIc(daily_resid_ic_results, sensitive_switch_history, figure_dir, start_date='', reba_wd='', file_suffix=''):
    if not any(daily_resid_ic_results.values()):
        logging.warning("⚠️ 无有效每日 Residual IC 数据，跳过绘图。")
        return
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    for name, records in daily_resid_ic_results.items():
        if not records: continue
        df_ic = pd.DataFrame(records).set_index('TRADE_DT')
        df_ic['IC_MA10'] = df_ic['IC'].rolling(window=10, min_periods=1).mean()
        ax1.plot(df_ic.index, df_ic['IC_MA10'], label=f'{name} (10D MA)', lw=1.5)
    ax1.set_title(f'Daily Cross-Sectional Residual Rank IC (Merged 5 Weekdays) - {start_date}', fontsize=14)
    ax1.set_ylabel('Residual Rank IC', fontsize=12)
    ax1.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize='small')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f'daily_resid_ic_trend_{start_date}_{reba_wd}{file_suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ 合并 Residual Rank IC 趋势图已保存至: {path}")

def main(start_date='2026-04-01', use_trade_pool=False):
    setup_logging()
    cfg = Config()
    total_capital = cfg.INITIAL_CAPITAL  # 记录总本金
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    file_suffix = '_st' if use_trade_pool else '' 
    
    # ==========================================
    # 1. 全局数据与模型加载 (只执行一次，大幅节省时间)
    # ==========================================
    logging.info("📦 加载面板数据...")
    try:
        df = load_panel_data(None, cfg.DATA_TEST_DIR, [], load_train=False, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    except TypeError:
        df = load_panel_data(cfg.DATA_DIR, cfg.DATA_TEST_DIR, list(range(2022, 2027)), file_prefix="train", exclude_bj=cfg.EXCLUDE_BJ)
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE')
    cfg.FEATURE_COLS = extract_valid_features(df)
    
    logging.info("📦 统一加载预训练模型...")
    trainers = {}
    model_dir = str(cfg.MODEL_DIR)
    if os.path.exists(os.path.join(model_dir, "sklearn_models.pkl")):
        trainers.update(joblib.load(os.path.join(model_dir, "sklearn_models.pkl")))
    if os.path.exists(os.path.join(model_dir, "percentile_sklearn_models.pkl")):
        trainers.update(joblib.load(os.path.join(model_dir, "percentile_sklearn_models.pkl")))
    if os.path.exists(os.path.join(model_dir, "ablation_sklearn_models.pkl")):
        trainers.update(joblib.load(os.path.join(model_dir, "ablation_sklearn_models.pkl")))
    if not trainers:
        raise RuntimeError("未加载到任何模型")
        
    tree_models = [name for name in trainers.keys() if 'XGB' in name or 'LGBM' in name]
    # base_strategies = ['ElasticNet', 'OptSharpe', 'DynamicSwitch', 'DynamicSwitch2', 
    #                    'DynamicSwitch_IR', 'DynamicSwitch2_IR', 'SensitiveSwitch', 'BuyAndHoldAll']
    base_strategies = ['ElasticNet', 'OptSharpe', 
                       'DynamicSwitch', 'DynamicSwitch2', 
                       'DynamicSwitch_IR', 'DynamicSwitch2_IR', 
                       'DynamicSwitch3', 'DynamicSwitch4',  # 🔑 新增
                       'SensitiveSwitch', 'BuyAndHoldAll']
    cfg.MODELS = base_strategies + tree_models
    cfg.DYNAMIC_SWITCH_BASE_MODELS = tree_models
    cfg.DYNAMIC_SWITCH2_BASE_MODELS = tree_models
    cfg.DYNAMIC_SWITCH2_IR_BASE_MODELS = tree_models
    cfg.SENSITIVE_SWITCH_BASE_MODELS = tree_models
    cfg.DYNAMIC_SWITCH3_BASE_MODELS = tree_models  # 🔑 新增：将所有树模型纳入 DS3 竞争池
    cfg.DYNAMIC_SWITCH4_BASE_MODELS = tree_models  # 🔑 新增：将所有树模型纳入 DS4 竞争池

    if hasattr(cfg, 'ABLATION_FEATURE_JSON') and os.path.exists(cfg.ABLATION_FEATURE_JSON):
        with open(cfg.ABLATION_FEATURE_JSON, 'r', encoding='utf-8') as f:
            raw_ablation_features = json.load(f)
        cfg.FEATURE_SELECTED = {}
        for model_name, feats in raw_ablation_features.items():
            abl_model_name = f"{model_name}_ablation"
            if abl_model_name in trainers:
                cfg.FEATURE_SELECTED[abl_model_name] = feats
    else:
        cfg.FEATURE_SELECTED = {}
        
    trade_pools = load_trade_pools(cfg.TRADE_POOL_DIR) if use_trade_pool else None
    test_start_date = pd.to_datetime(start_date)
    df_test = df[df['TRADE_DT'] >= test_start_date].copy()
    
    # ==========================================
    # 2. 循环 5 种调仓日进行回测 (每份使用 1/5 本金)
    # ==========================================
    weekdays = [0, 1, 2, 3, 4]
    wd_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五'}
    
    results_by_weekday = {}
    engines_by_weekday = {}
    sub_capital = total_capital / 5.0
    
    for wd in weekdays:
        logging.info(f"\n{'='*50}")
        logging.info(f"🚀 开始回测 | 调仓日: {wd_names[wd]} | 子本金: {sub_capital:,.0f}")
        logging.info(f"{'='*50}")
        
        cfg.REBALANCE_WEEKDAY = wd
        cfg.INITIAL_CAPITAL = sub_capital
        
        engine = BacktestEngine(df_test, cfg, trainers=trainers, label_col=f'label_{cfg.REBALANCE_DAYS}', trade_pools=trade_pools, ablation=False)
        results = engine.run()
        
        results_by_weekday[wd] = results
        engines_by_weekday[wd] = engine
        
    # ==========================================
    # 3. 合并 5 种调仓日的结果 (净值求平均)
    # ==========================================
    logging.info("🔄 正在合并 5 种调仓日的回测结果...")
    all_models = list(results_by_weekday[0].keys())
    merged_results = {}
    
    for model in all_models:
        dfs = []
        for wd in weekdays:
            df_wd = results_by_weekday[wd][model][['TRADE_DT', 'Value']].copy()
            df_wd = df_wd.rename(columns={'Value': f'Value_wd{wd}'})
            dfs.append(df_wd)
            
        # 按日期对齐 (outer join)
        merged_df = dfs[0]
        for i in range(1, len(dfs)):
            merged_df = pd.merge(merged_df, dfs[i], on='TRADE_DT', how='outer')
            
        merged_df = merged_df.sort_values('TRADE_DT').reset_index(drop=True)
        # 计算 5 份资金的平均净值 (因为初始都是1，平均后也是从1开始)
        value_cols = [f'Value_wd{wd}' for wd in weekdays]
        merged_df['Value'] = merged_df[value_cols].mean(axis=1)
        merged_results[model] = merged_df[['TRADE_DT', 'Value']]
        
    # 辅助函数：合并 IC 和 Residual IC 序列
    def merge_ic_results(engines_dict, attr_name):
        merged = {}
        for model in all_models:
            dfs = []
            for wd, engine in engines_dict.items():
                records = getattr(engine, attr_name).get(model, [])
                if records:
                    df = pd.DataFrame(records).set_index('TRADE_DT')
                    df = df.rename(columns={'IC': f'IC_wd{wd}'})
                    dfs.append(df)
            if dfs:
                merged_df = pd.concat(dfs, axis=1).sort_index()
                ic_cols = [c for c in merged_df.columns if c.startswith('IC_')]
                merged_df['IC'] = merged_df[ic_cols].mean(axis=1)
                merged[model] = [{'TRADE_DT': idx, 'IC': row['IC']} for idx, row in merged_df.iterrows()]
            else:
                merged[model] = []
        return merged

    merged_daily_ic = merge_ic_results(engines_by_weekday, 'daily_ic_results')
    merged_daily_resid_ic = merge_ic_results(engines_by_weekday, 'daily_resid_ic_results')
    
    # ==========================================
    # 4. 生成图表与绩效评估
    # ==========================================
    logging.info("📊 生成合并后的图表与绩效评估...")
    cfg.INITIAL_CAPITAL = total_capital  # 恢复总本金用于日志显示
    
    # 绘制 PnL, Drawdown, Rolling Sharpe
    metrics_summary = evaluate_and_plot(merged_results, str(cfg.OUT_DIR), str(cfg.FIG_DIR), 
                                        start_date=start_date, reba_wd='Avg_0-4', 
                                        TOP_K=cfg.TOP_K, suffix='_merged')
                                        
    # 绘制 IC 趋势图 (传入空列表隐藏下方的策略切换阶梯图)
    plot_daily_ic(merged_daily_ic, [], str(cfg.FIG_DIR), start_date=start_date, reba_wd='Avg_0-4', file_suffix='_merged')
    plot_daily_resIc(merged_daily_resid_ic, [], str(cfg.FIG_DIR), start_date=start_date, reba_wd='Avg_0-4', file_suffix='_merged')
    
    # 打印合并后的绩效表格
    print("\n" + "="*110)
    print("🏆 回测综合绩效评估 (5种调仓日合并后的平均表现)")
    print("="*110)
    print(f"{'Model':<25} | {'Total PnL':<10} | {'Annual Ret':<11} | {'Max DD':<10} | {'Sharpe':<8} | {'Win Rate':<10} | {'Closed Trades'}")
    print("-" * 110)
    
    summary_json = {"generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
    for name in all_models:
        stats = metrics_summary.get(name, {})
        total_ret = stats.get('total_return', 0.0)
        ann_ret = stats.get('annual_return', 0.0)
        max_dd = stats.get('max_drawdown', 0.0)
        sharpe = stats.get('sharpe', 0.0)
        
        # 计算 5 个 weekday 的平均交易统计
        total_trades_avg = 0
        win_rate_avg = 0.0
        for wd in weekdays:
            t_stats = engines_by_weekday[wd].portfolios[name].trade_stats
            total_trades_avg += t_stats['total_closed']
            if t_stats['total_closed'] > 0:
                win_rate_avg += (t_stats['wins'] / t_stats['total_closed'])
        total_trades_avg /= 5
        win_rate_avg /= 5
        
        print(f"{name:<25} | {total_ret:>9.2%} | {ann_ret:>10.2%} | {max_dd:>9.2%} | {sharpe:>7.3f} | {win_rate_avg:>9.2f}% | {total_trades_avg:>13.1f}")
        summary_json["models"][name] = {
            "total_return": float(total_ret), "annual_return": float(ann_ret),
            "max_drawdown": float(max_dd), "sharpe_ratio": float(sharpe),
            "trade_statistics": {"win_rate_pct": float(win_rate_avg), "total_closed_trades": float(total_trades_avg)}
        }
        
    print("="*110 + "\n")
    
    json_path = cfg.OUT_DIR / "backtest_summary_merged.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ 综合绩效汇总已保存至 JSON: {json_path}")
    logging.info(f"✅ 实验流程完成！(后缀: {file_suffix if file_suffix else '无'})")

if __name__ == "__main__":
    # 直接调用一次 main，内部会自动循环 5 种 weekday
    main(start_date='2026-04-01', use_trade_pool=True)