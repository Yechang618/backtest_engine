# /data/cye_temp/workspace/backtest_engine/src/backtest_engine.py
import sys
import os
import json
import glob
import logging
import datetime
import joblib
import pandas as pd
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
    cfg = Config()
    # reba_wd = cfg.REBALANCE_WEEKDAY if hasattr(cfg, 'REBALANCE_WEEKDAY') else 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    for name, records in daily_ic_results.items():
        if not records: continue
        df_ic = pd.DataFrame(records).set_index('TRADE_DT')
        df_ic['IC_MA10'] = df_ic['IC'].rolling(window=10, min_periods=1).mean()
        ax1.plot(df_ic.index, df_ic['IC_MA10'], label=f'{name} (10D MA)', lw=1.5)
        
    ax1.set_title('Daily Cross-Sectional Rank IC (10-Day Moving Average)', fontsize=14)
    ax1.set_ylabel('Rank IC', fontsize=12)
    ax1.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    if dynamic_switch_history:
        df_switch = pd.DataFrame(dynamic_switch_history).set_index('TRADE_DT')
        unique_models = sorted(list(df_switch['Model'].unique()))
        model_to_int = {m: i for i, m in enumerate(unique_models)}
        df_switch['Model_Int'] = df_switch['Model'].map(model_to_int)
        ax2.step(df_switch.index, df_switch['Model_Int'], where='post', color='purple', lw=2.5, alpha=0.8)
        ax2.set_yticks(list(model_to_int.values()))
        ax2.set_yticklabels(list(model_to_int.keys()), fontsize=11)
        ax2.set_ylabel('DynamicSwitch\nActive Model', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        for i, model in enumerate(unique_models):
            ax2.axhspan(i - 0.4, i + 0.4, color=plt.cm.tab10(i % 10), alpha=0.15, zorder=0)
    else:
        ax2.text(0.5, 0.5, 'No DynamicSwitch History', ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='gray')
        ax2.set_yticks([])
        
    ax2.set_xlabel('Date', fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f'daily_rank_ic_trend_{start_date}_{reba_wd}{file_suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ 每日 Rank IC 趋势图 (含动态切换) 已保存至: {path}")

def plot_daily_resIc(daily_resid_ic_results, sensitive_switch_history, figure_dir, start_date='', reba_wd='', file_suffix=''):
    if not any(daily_resid_ic_results.values()):
        logging.warning("⚠️ 无有效每日 Residual IC 数据，跳过绘图。")
        return
    cfg = Config()
    # reba_wd = cfg.REBALANCE_WEEKDAY if hasattr(cfg, 'REBALANCE_WEEKDAY') else 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    for name, records in daily_resid_ic_results.items():
        if not records: continue
        df_ic = pd.DataFrame(records).set_index('TRADE_DT')
        df_ic['IC_MA10'] = df_ic['IC'].rolling(window=10, min_periods=1).mean()
        ax1.plot(df_ic.index, df_ic['IC_MA10'], label=f'{name} (10D MA)', lw=1.5)
        
    ax1.set_title('Daily Cross-Sectional Residual Rank IC (10-Day Moving Average)', fontsize=14)
    ax1.set_ylabel('Residual Rank IC', fontsize=12)
    ax1.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    if sensitive_switch_history:
        df_switch = pd.DataFrame(sensitive_switch_history).set_index('TRADE_DT')
        unique_models = sorted(list(df_switch['Model'].unique()))
        model_to_int = {m: i for i, m in enumerate(unique_models)}
        df_switch['Model_Int'] = df_switch['Model'].map(model_to_int)
        ax2.step(df_switch.index, df_switch['Model_Int'], where='post', color='darkorange', lw=2.5, alpha=0.8)
        ax2.set_yticks(list(model_to_int.values()))
        ax2.set_yticklabels(list(model_to_int.keys()), fontsize=11)
        ax2.set_ylabel('SensitiveSwitch\nActive Model', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        for i, model in enumerate(unique_models):
            ax2.axhspan(i - 0.4, i + 0.4, color=plt.cm.tab10(i % 10), alpha=0.15, zorder=0)
    else:
        ax2.text(0.5, 0.5, 'No SensitiveSwitch History', ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='gray')
        ax2.set_yticks([])
        
    ax2.set_xlabel('Date', fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f'daily_resid_ic_trend_{start_date}_{reba_wd}{file_suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ 每日 Residual Rank IC 趋势图 (含 SensitiveSwitch) 已保存至: {path}")

def plot_prediction_error(mse_results, figure_dir, suffix="", start_date='', reba_wd='', file_suffix=''):
    cfg = Config()
    if not any(mse_results.values()):
        logging.warning("⚠️ 无有效预测误差数据，跳过绘图。")
        return
    plt.figure(figsize=(14, 6))
    for name, records in mse_results.items():
        if not records: continue
        df_err = pd.DataFrame(records).set_index('TRADE_DT')
        plt.plot(df_err.index, df_err['MSE'], label=f'{name}', lw=1.5)
    plt.title(f'Realized Prediction Error (MSE Settled at T+i) Over Time {suffix}')
    plt.xlabel('Settlement Date (T+i)')
    plt.ylabel('Mean Squared Error (MSE)')    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f'pred_top{cfg.TOP_K}_mse{suffix}_{start_date}_{reba_wd}{file_suffix}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"✅ 无未来函数预测误差图已保存至: {path}")

def main(start_date='2026-04-01', rebalancing_weekday = None, use_trade_pool=False):
    setup_logging()
    cfg = Config()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    if rebalancing_weekday is not None:
        cfg.REBALANCE_WEEKDAY = rebalancing_weekday
    
    file_suffix = '_st' if use_trade_pool else '' 
    wd_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', None: '禁用(使用天数)'}
    target_wd_str = wd_names.get(getattr(cfg, 'REBALANCE_WEEKDAY', None), '未知')
    logging.info(f"🔧 回测引擎已启动 | 起始日: {start_date} | 股票池约束: {'启用' if use_trade_pool else '禁用'}")
    logging.info(f"⚙️ 回测配置: WARMUP={cfg.WARMUP_DAYS}, REBALANCE={cfg.REBALANCE_DAYS}, WEEKDAY={target_wd_str}")

    # 🔑 1. 统一加载所有模型 (全量 + Percentile + 消融)
    logging.info("📦 统一加载预训练模型...")
    trainers = {}
    model_dir = str(cfg.MODEL_DIR)
    
    # 加载全量模型
    if os.path.exists(os.path.join(model_dir, "sklearn_models.pkl")):
        trainers.update(joblib.load(os.path.join(model_dir, "sklearn_models.pkl")))
    # 加载 Percentile 模型
    if os.path.exists(os.path.join(model_dir, "percentile_sklearn_models.pkl")):
        trainers.update(joblib.load(os.path.join(model_dir, "percentile_sklearn_models.pkl")))
    # 加载消融模型
    if os.path.exists(os.path.join(model_dir, "ablation_sklearn_models.pkl")):
        trainers.update(joblib.load(os.path.join(model_dir, "ablation_sklearn_models.pkl")))
    # 加载 Cluster 模型
    if os.path.exists(os.path.join(model_dir, "cluster_sklearn_models.pkl")):
        trainers.update(joblib.load(os.path.join(model_dir, "cluster_sklearn_models.pkl")))
        
    if not trainers:
        raise RuntimeError("未加载到任何模型，请先运行训练脚本")
        
    logging.info(f"✅ 统一模型加载完成 | 模型总数: {len(trainers)}")
    
    # 🔑 2. 动态更新 Config，将所有树模型纳入路由竞争池
    tree_models = [name for name in trainers.keys() if 'XGB' in name or 'LGBM' in name]
    base_strategies = ['ElasticNet', 'OptSharpe', 
                       'DynamicSwitch', 'DynamicSwitch2', 
                        'DynamicSwitch_IR', 'DynamicSwitch2_IR', 
                       'SensitiveSwitch', 'BuyAndHoldAll']
    
    cfg.MODELS = base_strategies + tree_models
    cfg.DYNAMIC_SWITCH_BASE_MODELS = tree_models
    cfg.DYNAMIC_SWITCH2_BASE_MODELS = tree_models
    cfg.DYNAMIC_SWITCH2_IR_BASE_MODELS = tree_models
    cfg.SENSITIVE_SWITCH_BASE_MODELS = tree_models
    logging.info(f"🔄 动态更新路由配置 | 参与竞争的树模型数: {len(tree_models)}")

    # # 🔑 3. 加载专属特征配置 (供引擎内部按模型名称路由)
    # if hasattr(cfg, 'ABLATION_FEATURE_JSON') and os.path.exists(cfg.ABLATION_FEATURE_JSON):
    #     with open(cfg.ABLATION_FEATURE_JSON, 'r', encoding='utf-8') as f:
    #         cfg.FEATURE_SELECTED = json.load(f)
    #     logging.info(f"✅ 专属特征配置加载完成 | 覆盖模型数: {len(cfg.FEATURE_SELECTED)}")
    # else:
    #     cfg.FEATURE_SELECTED = {}
    # 🔑 3. 加载专属特征配置 (供引擎内部按模型名称路由)
    if hasattr(cfg, 'ABLATION_FEATURE_JSON') and os.path.exists(cfg.ABLATION_FEATURE_JSON):
        with open(cfg.ABLATION_FEATURE_JSON, 'r', encoding='utf-8') as f:
            raw_ablation_features = json.load(f)
        
        # 🔑 核心修复：过滤缺失的消融模型，并为 key 加上 _ablation 后缀以匹配 trainers
        cfg.FEATURE_SELECTED = {}
        skipped_count = 0
        for model_name, feats in raw_ablation_features.items():
            abl_model_name = f"{model_name}_ablation"
            if abl_model_name in trainers:
                cfg.FEATURE_SELECTED[abl_model_name] = feats
            else:
                skipped_count += 1
                
        logging.info(f"✅ 专属特征配置加载完成 | 有效消融模型数: {len(cfg.FEATURE_SELECTED)} | 自动跳过缺失模型: {skipped_count} 个")
    else:
        cfg.FEATURE_SELECTED = {}

    # 4. 加载面板数据
    logging.info("📦 加载面板数据...")
    try:
        df = load_panel_data(None, cfg.DATA_TEST_DIR, [], load_train=False, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    except TypeError:
        df = load_panel_data(cfg.DATA_DIR, cfg.DATA_TEST_DIR, list(range(2022, 2027)), file_prefix="train", exclude_bj=cfg.EXCLUDE_BJ)
        
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE')
    cfg.FEATURE_COLS = extract_valid_features(df)
    
    # 5. 加载股票池
    trade_pools = load_trade_pools(cfg.TRADE_POOL_DIR) if use_trade_pool else None
    
    # 6. 数据隔离
    test_start_date = pd.to_datetime(start_date)
    df_test = df[df['TRADE_DT'] >= test_start_date].copy()
    logging.info(f"🔒 数据隔离完成 | 样本外测试集形状: {df_test.shape} (起始日: {test_start_date})")
    
    # 🔑 7. 运行统一回测 (ablation=False，由引擎内部根据模型名称自动路由特征)
    logging.info("🚀 启动统一回测 (所有模型共同参与路由竞争)...")
    engine = BacktestEngine(df_test, cfg, trainers=trainers, label_col=f'label_{cfg.REBALANCE_DAYS}', trade_pools=trade_pools, ablation=False)
    results = engine.run()
    
    # 8. 保存日志与绘图
    for name, pf in engine.portfolios.items():
        pf.save_logs(name, str(cfg.LOG_DIR))
        
    logging.info("📊 生成图表...")
    metrics_summary = evaluate_and_plot(results, str(cfg.OUT_DIR), str(cfg.FIG_DIR), start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY, TOP_K=cfg.TOP_K, suffix=file_suffix)
    
    plot_daily_ic(engine.daily_ic_results, getattr(engine, 'dynamic_switch_history', []), str(cfg.FIG_DIR), start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY, file_suffix=file_suffix)
    plot_daily_resIc(engine.daily_resid_ic_results, getattr(engine, 'sensitive_switch_history', []), str(cfg.FIG_DIR), start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY, file_suffix=file_suffix)
    plot_prediction_error(engine.mse_results, str(cfg.FIG_DIR), suffix="_unified", start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY, file_suffix=file_suffix)
    
    # 9. 打印绩效表格
    print("\n" + "="*110)
    print("🏆 回测综合绩效评估 (Out-of-Sample / Test Set - 统一路由竞争)")
    print("="*110)
    print(f"{'Model':<25} | {'Total PnL':<10} | {'Annual Ret':<11} | {'Max DD':<10} | {'Sharpe':<8} | {'Win Rate':<10} | {'Closed Trades'}")
    print("-" * 110)
    
    summary_json = {"generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
    for name, pf in engine.portfolios.items():
        stats = metrics_summary.get(name, {})
        total_ret = stats.get('total_return', 0.0)
        ann_ret = stats.get('annual_return', 0.0)
        max_dd = stats.get('max_drawdown', 0.0)
        sharpe = stats.get('sharpe', 0.0)
        t_stats = pf.trade_stats
        total_trades = t_stats['total_closed']
        win_rate = (t_stats['wins'] / total_trades * 100) if total_trades > 0 else 0.0
        
        print(f"{name:<25} | {total_ret:>9.2%} | {ann_ret:>10.2%} | {max_dd:>9.2%} | {sharpe:>7.3f} | {win_rate:>9.2f}% | {total_trades:>13}")
        summary_json["models"][name] = {
            "total_return": float(total_ret), "annual_return": float(ann_ret),
            "max_drawdown": float(max_dd), "sharpe_ratio": float(sharpe),
            "trade_statistics": {"win_rate_pct": float(win_rate), "total_closed_trades": int(total_trades)}
        }
    print("="*110 + "\n")
    
    # json_path = cfg.OUT_DIR / "backtest_summary.json"
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     json.dump(summary_json, f, indent=4, ensure_ascii=False)
    # logging.info(f"✅ 综合绩效汇总已保存至 JSON: {json_path}")
    # logging.info(f"✅ 实验流程完成！(后缀: {file_suffix if file_suffix else '无'})")

    # ... (在 run_backtest.py 的 main 函数中，找到保存 backtest_summary.json 的位置) ...
    
    json_path = cfg.OUT_DIR / f"backtest_summary_{rebalancing_weekday}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ 综合绩效汇总已保存至 JSON: {json_path}")

    # 🔑 新增：保存每日 Rank IC 和 Residual Rank IC 数据，供后续分析脚本使用
    ic_results_path = cfg.OUT_DIR / f"daily_ic_results_{rebalancing_weekday}.json"
    with open(ic_results_path, 'w', encoding='utf-8') as f:
        # 将 Timestamp 转换为字符串以便 JSON 序列化
        serializable_ic = {
            model: [{'TRADE_DT': str(r['TRADE_DT']), 'IC': r['IC']} for r in records]
            for model, records in engine.daily_ic_results.items()
        }
        json.dump(serializable_ic, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ 每日 Rank IC 数据已保存至: {ic_results_path}")

    resid_ic_results_path = cfg.OUT_DIR / f"daily_resid_ic_results_{rebalancing_weekday}.json"
    with open(resid_ic_results_path, 'w', encoding='utf-8') as f:
        serializable_resid_ic = {
            model: [{'TRADE_DT': str(r['TRADE_DT']), 'IC': r['IC']} for r in records]
            for model, records in engine.daily_resid_ic_results.items()
        }
        json.dump(serializable_resid_ic, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ 每日 Residual Rank IC 数据已保存至: {resid_ic_results_path}")

    logging.info(f"✅ 实验流程完成！(后缀: {file_suffix if file_suffix else '无'})")

if __name__ == "__main__":
    main(start_date='2026-04-01', rebalancing_weekday = 0, use_trade_pool=True)

    main(start_date='2026-04-01', rebalancing_weekday = 1, use_trade_pool=True)

    main(start_date='2026-04-01', rebalancing_weekday = 2, use_trade_pool=True)

    main(start_date='2026-04-01', rebalancing_weekday = 3, use_trade_pool=True)

    main(start_date='2026-04-01', rebalancing_weekday = 4, use_trade_pool=True)