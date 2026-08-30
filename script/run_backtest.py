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

def load_pretrained_models(model_dir: str, ablation=False):
    """加载预训练模型，消融模型自动去除 _ablation 后缀以匹配 cfg.MODELS"""
    trainers = {}
    if not ablation:
        sklearn_path = os.path.join(model_dir, "sklearn_models.pkl")
        if os.path.exists(sklearn_path):
            trainers.update(joblib.load(sklearn_path))
    else:
        ablation_sklearn_path = os.path.join(model_dir, "ablation_sklearn_models.pkl")
        if os.path.exists(ablation_sklearn_path):
            raw_trainers = joblib.load(ablation_sklearn_path)
            # 🔑 核心：去除 _ablation 后缀，使模型名称与 cfg.MODELS 对齐
            trainers = {k.replace('_ablation', ''): v for k, v in raw_trainers.items()}
    return trainers

def load_trade_pools(pool_dir: str) -> Dict[str, Set[str]]:
    """加载 TRADE_POOL_DIR 下的 CSV 文件，解析为 {月份: 股票代码集合} 的字典"""
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
    """绘制各模型的每日 Rank IC 随时间变化的曲线 (10日滚动平均)"""
    if not any(daily_ic_results.values()):
        logging.warning("⚠️ 无有效每日 IC 数据，跳过绘图。")
        return
    cfg = Config()
    # start_date = cfg.Date if hasattr(cfg, 'Date') else '20260812'
    reba_wd = cfg.REBALANCE_WEEKDAY if hasattr(cfg, 'REBALANCE_WEEKDAY') else 2

    # 创建上下两个子图，高度比例为 3:1，共享 X 轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # ─────────────────────────────────────────────────────────────
    # 上图：Rank IC 趋势
    # ─────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────
    # 下图：DynamicSwitch 模型切换历史 (阶梯图)
    # ─────────────────────────────────────────────────────────────
    if dynamic_switch_history:
        df_switch = pd.DataFrame(dynamic_switch_history).set_index('TRADE_DT')
        
        # 提取所有出现过的模型并排序，映射为整数以便绘制阶梯图
        unique_models = sorted(list(df_switch['Model'].unique()))
        model_to_int = {m: i for i, m in enumerate(unique_models)}
        
        df_switch['Model_Int'] = df_switch['Model'].map(model_to_int)
        
        # 使用 step 绘制阶梯图，where='post' 表示值在下一个切换点之前保持不变
        ax2.step(df_switch.index, df_switch['Model_Int'], where='post', color='purple', lw=2.5, alpha=0.8)
        
        # 设置 Y 轴刻度为模型名称
        ax2.set_yticks(list(model_to_int.values()))
        ax2.set_yticklabels(list(model_to_int.keys()), fontsize=11)
        ax2.set_ylabel('DynamicSwitch\nActive Model', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 为阶梯线添加轻微的水平填充，增强视觉区分度
        for i, model in enumerate(unique_models):
            ax2.axhspan(i - 0.4, i + 0.4, color=plt.cm.tab10(i % 10), alpha=0.15, zorder=0)
            
    else:
        ax2.text(0.5, 0.5, 'No DynamicSwitch History', ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='gray')
        ax2.set_yticks([])
    
    # ─────────────────────────────────────────────────────────────
    # 全局 X 轴格式化
    # ─────────────────────────────────────────────────────────────
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
    """绘制各模型的每日 Residual Rank IC 随时间变化的曲线，并附带 SensitiveSwitch 的切换记录"""
    if not any(daily_resid_ic_results.values()):
        logging.warning("⚠️ 无有效每日 Residual IC 数据，跳过绘图。")
        return
        
    cfg = Config()
    reba_wd = cfg.REBALANCE_WEEKDAY if hasattr(cfg, 'REBALANCE_WEEKDAY') else 2
    
    # 创建上下两个子图，高度比例为 3:1，共享 X 轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # ─────────────────────────────────────────────────────────────
    # 上图：Residual Rank IC 趋势
    # ─────────────────────────────────────────────────────────────
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
    
    # ─────────────────────────────────────────────────────────────
    # 下图：SensitiveSwitch 模型切换历史 (阶梯图)
    # ─────────────────────────────────────────────────────────────
    if sensitive_switch_history:
        df_switch = pd.DataFrame(sensitive_switch_history).set_index('TRADE_DT')
        unique_models = sorted(list(df_switch['Model'].unique()))
        model_to_int = {m: i for i, m in enumerate(unique_models)}
        df_switch['Model_Int'] = df_switch['Model'].map(model_to_int)
        
        # 🔑 使用深橙色以区分 DynamicSwitch 的紫色
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
        
    # ─────────────────────────────────────────────────────────────
    # 全局 X 轴格式化
    # ─────────────────────────────────────────────────────────────
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
    """绘制模型预测误差 (MSE) 随时间变化的时序图 (无未来函数版)"""
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

def main(start_date='2025-01-01', use_trade_pool=False):
    setup_logging()
    cfg = Config()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    ablation = cfg.SHAP_ABLATION
    file_suffix = '_st' if use_trade_pool else '' 
    
    wd_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', None: '禁用(使用天数)'}
    target_wd_str = wd_names.get(getattr(cfg, 'REBALANCE_WEEKDAY', None), '未知')
    logging.info(f"🔧 回测引擎已启动 | 起始日: {start_date} | 股票池: {'启用' if use_trade_pool else '禁用'} | 消融: {'启用' if ablation else '禁用'}")
    
    # 1. 加载面板数据
    logging.info("📦 加载面板数据...")
    try:
        df = load_panel_data(None, cfg.DATA_TEST_DIR, [], load_train=False, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    except TypeError:
        df = load_panel_data(cfg.DATA_DIR, cfg.DATA_TEST_DIR, list(range(2022, 2027)), file_prefix="train", exclude_bj=cfg.EXCLUDE_BJ)
        
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE')
    cfg.FEATURE_COLS = extract_valid_features(df)
    
    # 2. 加载模型
    logging.info("📦 加载预训练模型...")
    trainers = load_pretrained_models(str(cfg.MODEL_DIR), ablation=False)
    if not trainers: raise RuntimeError("未加载到任何全量模型")
        
    trainers_ablation = {}
    if ablation:
        logging.info("🧪 加载消融实验模型...")
        trainers_ablation = load_pretrained_models(str(cfg.MODEL_DIR), ablation=True)
        if not trainers_ablation:
            logging.warning("⚠️ 未找到消融模型，将跳过消融实验")
            ablation = False
            
    # 3. 加载股票池
    trade_pools = load_trade_pools(cfg.TRADE_POOL_DIR) if use_trade_pool else None
        
    # 4. 数据隔离
    test_start_date = pd.to_datetime(start_date)
    df_test = df[df['TRADE_DT'] >= test_start_date].copy()
    
    # 5. 运行全量回测
    logging.info("🚀 使用全量模型进行回测...")
    engine = BacktestEngine(df_test, cfg, trainers=trainers, label_col=f'label_{cfg.REBALANCE_DAYS}', trade_pools=trade_pools)
    results = engine.run()
    plot_prediction_error(engine.mse_results, str(cfg.FIG_DIR), suffix="_full", start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY, file_suffix=file_suffix)
    
    # 6. 运行消融回测
    if ablation:
        logging.info("🧪 使用消融模型进行回测...")
        engine_ab = BacktestEngine(df_test, cfg, trainers=trainers_ablation, label_col=f'label_{cfg.REBALANCE_DAYS}', ablation=True, trade_pools=trade_pools)
        results_ab = engine_ab.run()
        plot_prediction_error(engine_ab.mse_results, str(cfg.FIG_DIR), suffix="_ablation", start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY, file_suffix=file_suffix)
        
        # 🔑 核心：合并结果与诊断数据，加上 _ablation 后缀以区分
        for name in results_ab.keys():
            results[f"{name}_ablation"] = results_ab[name]
        for name in engine_ab.daily_ic_results.keys():
            engine.daily_ic_results[f"{name}_ablation"] = engine_ab.daily_ic_results[name]
        for name in engine_ab.daily_resid_ic_results.keys():
            engine.daily_resid_ic_results[f"{name}_ablation"] = engine_ab.daily_resid_ic_results[name]

    # 7. 保存日志与绘图
    for name, pf in engine.portfolios.items():
        pf.save_logs(name, str(cfg.LOG_DIR))
        
    logging.info("📊 生成图表...")
    metrics_summary = evaluate_and_plot(results, str(cfg.OUT_DIR), str(cfg.FIG_DIR), start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY, TOP_K=cfg.TOP_K, suffix=file_suffix)
    
    plot_daily_ic(engine.daily_ic_results, getattr(engine, 'dynamic_switch_history', []), str(cfg.FIG_DIR), start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY, file_suffix=file_suffix)
    plot_daily_resIc(engine.daily_resid_ic_results, getattr(engine, 'sensitive_switch_history', []), str(cfg.FIG_DIR), start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY, file_suffix=file_suffix)
    
    # 8. 打印绩效表格
    print("\n" + "="*90)
    print("🏆 回测综合绩效评估 (Out-of-Sample / Test Set)")
    print("="*90)
    print(f"{'Model':<20} | {'Total PnL':<10} | {'Annual Ret':<11} | {'Max DD':<10} | {'Sharpe':<8} | {'Win Rate':<10} | {'Closed Trades'}")
    print("-" * 90)
    
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
        
        print(f"{name:<20} | {total_ret:>9.2%} | {ann_ret:>10.2%} | {max_dd:>9.2%} | {sharpe:>7.3f} | {win_rate:>9.2f}% | {total_trades:>13}")
        summary_json["models"][name] = {
            "total_return": float(total_ret), "annual_return": float(ann_ret),
            "max_drawdown": float(max_dd), "sharpe_ratio": float(sharpe),
            "trade_statistics": {"win_rate_pct": float(win_rate), "total_closed_trades": int(total_trades)}
        }
        
    # 打印消融模型绩效
    if ablation:
        print("-" * 90)
        for name in results.keys():
            if name.endswith('_ablation'):
                stats = metrics_summary.get(name, {})
                total_ret = stats.get('total_return', 0.0)
                ann_ret = stats.get('annual_return', 0.0)
                max_dd = stats.get('max_drawdown', 0.0)
                sharpe = stats.get('sharpe', 0.0)
                print(f"{name:<20} | {total_ret:>9.2%} | {ann_ret:>10.2%} | {max_dd:>9.2%} | {sharpe:>7.3f} | {'N/A':>10} | {'N/A':>13}")
                summary_json["models"][name] = {"total_return": float(total_ret), "annual_return": float(ann_ret), "max_drawdown": float(max_dd), "sharpe_ratio": float(sharpe)}

    print("="*90 + "\n")
    
    json_path = cfg.OUT_DIR / "backtest_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ 综合绩效汇总已保存至 JSON: {json_path}")
    logging.info(f"✅ 实验流程完成！(后缀: {file_suffix if file_suffix else '无'})")

if __name__ == "__main__":
    main(start_date='2026-01-01', use_trade_pool=True)