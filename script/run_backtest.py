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

from util.data_loader import load_panel_data, compute_real_returns, extract_valid_features
from src.backtest_engine import BacktestEngine
from util.metrics import evaluate_and_plot
from config.Config import Config
# from src.transformer_model import PyTorchTabularRegressor, SimpleTabularTransformer

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def load_pretrained_models(model_dir: str, feature_cols: list):
    trainers = {}
    sklearn_path = os.path.join(model_dir, "sklearn_models.pkl")
    if os.path.exists(sklearn_path):
        trainers.update(joblib.load(sklearn_path))
        
    for name in ['Transformer']:
        meta_path = os.path.join(model_dir, f"{name}_meta.pkl")
        state_path = os.path.join(model_dir, f"{name}_state.pt")
        if os.path.exists(meta_path) and os.path.exists(state_path):
            meta = joblib.load(meta_path)
            model = PyTorchTabularRegressor(input_dim=meta['input_dim'], hidden_dim=meta['hidden_dim'],
                                            n_heads=meta['n_heads'], n_layers=meta['n_layers'])
            model.feature_names = meta['feature_names']
            model.model = SimpleTabularTransformer(input_dim=meta['input_dim'], hidden_dim=meta['hidden_dim'],
                                                   n_heads=meta['n_heads'], n_layers=meta['n_layers'])
            model.model.load_state_dict(torch.load(state_path, map_location='cpu'))
            model.model.eval()
            trainers[name] = model
            logging.info(f"✅ 成功加载预训练模型: {name}")
    return trainers

def main():
    setup_logging()
    cfg = Config()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 🔑 关键修复：预训练模型无需长预热，强制设为 5 天
    # cfg.WARMUP_DAYS = 5 
    logging.info(f"⚙️ 回测配置已调整: WARMUP_DAYS={cfg.WARMUP_DAYS}, REBALANCE_DAYS={cfg.REBALANCE_DAYS}")

    logging.info("📦 加载面板数据...")
    try:
        df = load_panel_data(None, cfg.DATA_TEST_DIR, [], load_train=False, load_test=True)
    except TypeError:
        df = load_panel_data(cfg.DATA_DIR, cfg.DATA_TEST_DIR, list(range(2016, 2027)), file_prefix="train")
        
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    feature_cols = extract_valid_features(df)
    cfg.FEATURE_COLS = feature_cols  
    logging.info(f"📊 数据加载完成 | 总形状: {df.shape} | 特征数: {len(feature_cols)}")

    logging.info(f"📦 从 {cfg.MODEL_DIR} 加载预训练模型...")
    trainers = load_pretrained_models(str(cfg.MODEL_DIR), cfg.FEATURE_COLS)
    if not trainers:
        raise RuntimeError("未加载到任何模型，请先运行 script/train_models.py")

    test_start_date = pd.to_datetime('2025-01-01')
    df_test = df[df['TRADE_DT'] >= test_start_date].copy()
    logging.info(f"🔒 数据隔离完成 | 样本外测试集形状: {df_test.shape} (起始日: {test_start_date})")

    engine = BacktestEngine(df_test, cfg, trainers=trainers, label_col=f'label_{cfg.REBALANCE_DAYS}')
    results = engine.run()

    for name, pf in engine.portfolios.items():
        pf.save_logs(name, str(cfg.LOG_DIR))

    logging.info("📊 生成图表...")
    metrics_summary = evaluate_and_plot(results, str(cfg.OUT_DIR), str(cfg.FIG_DIR))

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
        summary_json["models"][name] = {"total_return": float(total_ret), "annual_return": float(ann_ret),
            "max_drawdown": float(max_dd), "sharpe_ratio": float(sharpe),
            "trade_statistics": {"win_rate_pct": float(win_rate), "total_closed_trades": int(total_trades)}}
    print("="*90 + "\n")
    
    json_path = cfg.OUT_DIR / "backtest_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ 综合绩效汇总已保存至 JSON: {json_path}")

    # logging.info("🔍 启动 SHAP 因子可解释性分析...")
    # engine.analyze_shap(str(cfg.FIG_DIR), sample_size=cfg.SHAP_SAMPLE_SIZE)
    logging.info("✅ 全部流程完成！")

if __name__ == "__main__":
    main()