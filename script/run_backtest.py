# /data/cye_temp/workspace/backtest_engine/src/backtest_engine.py
import sys
import os
import json
import logging
import datetime
import joblib
# import torch
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
print(f"🔧 Backtest Engine Root: {ROOT}")
sys.path.insert(0, str(ROOT))

# from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features
from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features, compute_derived_factors # 🔑 新增导入

from src.backtest_engine import BacktestEngine
from util.metrics import evaluate_and_plot
from config.Config import Config
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from src.transformer_model import PyTorchTabularRegressor, SimpleTabularTransformer

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def load_pretrained_models(model_dir: str, feature_cols: list, ablation=False):
    trainers = {}
    if ablation == False:
        sklearn_path = os.path.join(model_dir, "sklearn_models.pkl")
        if os.path.exists(sklearn_path):
            trainers.update(joblib.load(sklearn_path))
    else:
        ablation_sklearn_path = os.path.join(model_dir, "ablation_sklearn_models.pkl")
        if os.path.exists(ablation_sklearn_path):
            trainers.update(joblib.load(ablation_sklearn_path))
    return trainers


def plot_daily_ic(daily_ic_results, dynamic_switch_history, figure_dir):
    """绘制各模型的每日 Rank IC 随时间变化的曲线 (10日滚动平均)"""
    if not any(daily_ic_results.values()):
        logging.warning("⚠️ 无有效每日 IC 数据，跳过绘图。")
        return
    cfg = Config()
    start_date = cfg.Date if hasattr(cfg, 'Date') else '20260812'
    reba_wd = cfg.REBALANCE_WEEKDAY if hasattr(cfg, 'REBALANCE_WEEKDAY') else 2

    # 创建上下两个子图，高度比例为 3:1，共享 X 轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # plt.figure(figsize=(14, 6))
    
    # for name, records in daily_ic_results.items():
    #     if not records: continue
    #     df_ic = pd.DataFrame(records).set_index('TRADE_DT')
    #     # 计算 10 日滚动平均以平滑噪音，便于观察趋势
    #     df_ic['IC_MA10'] = df_ic['IC'].rolling(window=10, min_periods=1).mean()
    #     plt.plot(df_ic.index, df_ic['IC_MA10'], label=f'{name} (10D MA)', lw=1.5)
        
    # plt.title('Daily Cross-Sectional Rank IC (10-Day Moving Average)')
    # plt.xlabel('Date')
    # plt.ylabel('Rank IC')
    # plt.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    
    # os.makedirs(figure_dir, exist_ok=True)
    # path = os.path.join(figure_dir, 'daily_rank_ic_trend.png')
    # plt.savefig(path, dpi=150)
    # plt.close()
    # logging.info(f"✅ 每日 Rank IC 趋势图已保存至: {path}")
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
    path = os.path.join(figure_dir, f'daily_rank_ic_trend_{start_date}_{reba_wd}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ 每日 Rank IC 趋势图 (含动态切换) 已保存至: {path}")

def plot_prediction_error(mse_results, figure_dir, suffix="", start_date='', reba_wd=''):
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
    path = os.path.join(figure_dir, f'pred_top{cfg.TOP_K}_mse{suffix}_{start_date}_{reba_wd}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"✅ 无未来函数预测误差图已保存至: {path}")

def main(start_date='2025-01-01'):
    setup_logging()
    cfg = Config()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    ablation = cfg.SHAP_ABLATION
    # 🔑 修改：打印 REBALANCE_WEEKDAY 配置
    wd_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', None: '禁用(使用天数)'}
    target_wd_str = wd_names.get(getattr(cfg, 'REBALANCE_WEEKDAY', None), '未知')
    
    logging.info(f"🔧 回测引擎已启动 | 样本外测试集起始日: {start_date} | Top K: {cfg.TOP_K} | 消融实验模式: {'启用' if ablation else '禁用'}")
    logging.info(f"⚙️ 回测配置已调整: WARMUP_DAYS={cfg.WARMUP_DAYS}, REBALANCE_DAYS={cfg.REBALANCE_DAYS}, REBALANCE_WEEKDAY={target_wd_str}")
    
    if ablation:
        logging.info("🧪 消融实验模式已启用 | 仅使用选定特征进行回测")

    logging.info("📦 加载面板数据...")
    try:
        # 🔑 修改：传入 exclude_bj=cfg.EXCLUDE_BJ
        df = load_panel_data(None, cfg.DATA_TEST_DIR, [], load_train=False, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    except TypeError:
        # 🔑 修改：传入 exclude_bj=cfg.EXCLUDE_BJ
        df = load_panel_data(cfg.DATA_DIR, cfg.DATA_TEST_DIR, list(range(2022, 2027)), file_prefix="train", exclude_bj=cfg.EXCLUDE_BJ)
        
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)

    # 🔑 新增：计算衍生因子 (必须与训练集保持完全一致的处理逻辑)
    df = compute_derived_factors(df, price_col='S_DQ_ADJCLOSE')
    
    feature_cols = extract_valid_features(df)
    cfg.FEATURE_COLS = feature_cols  
    logging.info(f"📊 数据加载完成 | 总形状: {df.shape} | 特征数: {len(feature_cols)}")

    logging.info(f"📦 从 {cfg.MODEL_DIR} 加载预训练模型...")
    trainers = load_pretrained_models(str(cfg.MODEL_DIR), cfg.FEATURE_COLS)
    if not trainers:
        raise RuntimeError("未加载到任何模型，请先运行 script/train_models.py")

    logging.info(f"Loading ablation models from {cfg.MODEL_DIR}...")
    trainers_ablation = load_pretrained_models(str(cfg.MODEL_DIR), cfg.FEATURE_SELECTED, ablation=True)
    if ablation and not trainers_ablation:
        raise RuntimeError("未加载到任何消融实验模型，请先运行 script/train_models.py 并启用消融实验")

    # start_date = '2025-01-01'
    # start_date = '2026-01-01'
    test_start_date = pd.to_datetime(start_date) if start_date else pd.to_datetime('2025-01-01')
    df_test = df[df['TRADE_DT'] >= test_start_date].copy()
    logging.info(f"🔒 数据隔离完成 | 样本外测试集形状: {df_test.shape} (起始日: {test_start_date})")

    if ablation:
        logging.info("🧪 使用消融实验模型进行回测...")
        engine_ab = BacktestEngine(df_test, cfg, trainers=trainers_ablation, label_col=f'label_{cfg.REBALANCE_DAYS}', ablation=True)
        results_ab = engine_ab.run()
        plot_prediction_error(engine_ab.mse_results, str(cfg.FIG_DIR), suffix="_ablation", start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY)

    logging.info("🚀 使用预训练模型进行回测...")
    engine = BacktestEngine(df_test, cfg, trainers=trainers, label_col=f'label_{cfg.REBALANCE_DAYS}')
    results = engine.run()
    plot_prediction_error(engine.mse_results, str(cfg.FIG_DIR), suffix="_full", start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY)

    if ablation:
        for name in results_ab.keys():
            if name not in ['ElasticNet', 'BuyAndHoldAll', 'OptSharpe', 'DynamicSwitch']:
                results[name + "_ablation"] = results_ab[name]

    for name, pf in engine.portfolios.items():
        pf.save_logs(name, str(cfg.LOG_DIR))

    logging.info("📊 生成图表...")
    metrics_summary = evaluate_and_plot(results, str(cfg.OUT_DIR), str(cfg.FIG_DIR), start_date=start_date, reba_wd=cfg.REBALANCE_WEEKDAY, TOP_K=cfg.TOP_K)

    # 🔑 新增：绘制每日 Rank IC 趋势图
    # plot_daily_ic(engine.daily_ic_results, str(cfg.FIG_DIR))
    # 🔑 修改：传入 engine.dynamic_switch_history
    plot_daily_ic(engine.daily_ic_results, getattr(engine, 'dynamic_switch_history', []), str(cfg.FIG_DIR))

    print("\n" + "="*90)
    print("🏆 回测综合绩效评估 (Out-of-Sample / Test Set)")
    print("="*90)
    print(f"{'Model':<15} | {'Total PnL':<10} | {'Annual Ret':<11} | {'Max DD':<10} | {'Sharpe':<8} | {'Win Rate':<10} | {'Closed Trades'}")
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
        
        print(f"{name:<15} | {total_ret:>9.2%} | {ann_ret:>10.2%} | {max_dd:>9.2%} | {sharpe:>7.3f} | {win_rate:>9.2f}% | {total_trades:>13}")
        
        summary_json["models"][name] = {
            "total_return": float(total_ret), "annual_return": float(ann_ret),
            "max_drawdown": float(max_dd), "sharpe_ratio": float(sharpe),
            "trade_statistics": {"win_rate_pct": float(win_rate), "total_closed_trades": int(total_trades)}
        }
    print("="*90 + "\n")
    
    json_path = cfg.OUT_DIR / "backtest_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ 综合绩效汇总已保存至 JSON: {json_path}")

    engine.analyze_shap(str(cfg.FIG_DIR), sample_size=cfg.SHAP_SAMPLE_SIZE)
    logging.info("✅ 全部流程完成！")

if __name__ == "__main__":
    main(start_date='2025-01-01')
    main(start_date='2026-01-01')  # 可选：运行第二次回测，起始日为 2026-01-01