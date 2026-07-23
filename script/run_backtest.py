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

def plot_prediction_error(mse_results, figure_dir, suffix="", start_date=''):
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
    path = os.path.join(figure_dir, f'pred_top{cfg.TOP_K}_mse{suffix}_{start_date}.png')
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
    logging.info(f"🔧 回测引擎已启动 | 样本外测试集起始日: {start_date} |Top K: {cfg.TOP_K} | 消融实验模式: {'启用' if ablation else '禁用'}")
    logging.info(f"⚙️ 回测配置已调整: WARMUP_DAYS={cfg.WARMUP_DAYS}, REBALANCE_DAYS={cfg.REBALANCE_DAYS}, EXCLUDE_BJ={cfg.EXCLUDE_BJ}")

    if ablation:
        logging.info("🧪 消融实验模式已启用 | 仅使用选定特征进行回测")

    logging.info("📦 加载面板数据...")
    try:
        # 🔑 修改：传入 exclude_bj=cfg.EXCLUDE_BJ
        df = load_panel_data(None, cfg.DATA_TEST_DIR, [], load_train=False, load_test=True, exclude_bj=cfg.EXCLUDE_BJ)
    except TypeError:
        # 🔑 修改：传入 exclude_bj=cfg.EXCLUDE_BJ
        df = load_panel_data(cfg.DATA_DIR, cfg.DATA_TEST_DIR, list(range(2016, 2027)), file_prefix="train", exclude_bj=cfg.EXCLUDE_BJ)
        
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
        plot_prediction_error(engine_ab.mse_results, str(cfg.FIG_DIR), suffix="_ablation", start_date=start_date)

    logging.info("🚀 使用预训练模型进行回测...")
    engine = BacktestEngine(df_test, cfg, trainers=trainers, label_col=f'label_{cfg.REBALANCE_DAYS}')
    results = engine.run()
    plot_prediction_error(engine.mse_results, str(cfg.FIG_DIR), suffix="_full", start_date=start_date)

    if ablation:
        for name in results_ab.keys():
            if name not in ['ElasticNet', 'BuyAndHoldAll', 'OptSharpe']:
                results[name + "_ablation"] = results_ab[name]

    for name, pf in engine.portfolios.items():
        pf.save_logs(name, str(cfg.LOG_DIR))

    logging.info("📊 生成图表...")
    metrics_summary = evaluate_and_plot(results, str(cfg.OUT_DIR), str(cfg.FIG_DIR), start_date=start_date, TOP_K=cfg.TOP_K)

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