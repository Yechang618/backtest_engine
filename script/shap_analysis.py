# /data/cye_temp/workspace/backtest_engine/script/shap_analysis.py
import sys
import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

# 🔧 配置路径，确保能导入项目内的模块
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 🔑 修复：引入 compute_real_returns 以生成标签列
from util.data_loader import extract_valid_features, compute_ic_ir, compute_real_returns, load_panel_data
from config.Config import Config

# 尝试导入模型和 SHAP 库
try:
    import shap
    import lightgbm as lgb
    import joblib
except ImportError:
    raise ImportError("请确保已安装必要库: pip install shap lightgbm joblib")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

def load_train_data_only(cfg: Config) -> pd.DataFrame:
    """仅加载 /data/train_data 中的全部历史数据，彻底隔离测试集"""
    dfs = []
    train_years = list(range(2016, 2024)) 
    
    for y in train_years:
        path = os.path.join(cfg.DATA_DIR, f'model_ready_panel_selected_plus_ohlc_train_{y}.parquet')
        if os.path.exists(path):
            df_y = pd.read_parquet(path)
            df_y['TRADE_DT'] = pd.to_datetime(df_y['TRADE_DT'].astype(str), format='%Y%m%d')
            dfs.append(df_y)
        else:
            logging.warning(f"⚠️ 未找到训练数据: {path}")

    df = load_panel_data(None, cfg.DATA_DIR, list(range(2016, 2025)), file_prefix="train", load_train=True, load_test=True)
    # dfs.append(df)
    # df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    # Get df from 2016 to 2024
    df = df[(df['TRADE_DT'] >= pd.to_datetime('2015-06-01')) & (df['TRADE_DT'] <= pd.to_datetime('2024-07-31'))].copy()
    dfs.append(df)
    # feature_cols = extract_valid_features(df)
            
    if not dfs:
        raise FileNotFoundError(f"在 {cfg.DATA_DIR} 中未找到任何训练集 parquet 文件")
        
    df_full = pd.concat(dfs, ignore_index=True)
    df_full = df_full.sort_values(['TRADE_DT', 'S_INFO_WINDCODE']).reset_index(drop=True)
    logging.info(f"✅ 训练集加载完成 | 总行数: {len(df_full):,} | 时间范围: {df_full['TRADE_DT'].min()} 至 {df_full['TRADE_DT'].max()}")
    return df_full

def get_or_train_model(cfg: Config, feature_cols: List[str], df_train: pd.DataFrame, model_type: str = 'LightGBM'):
    """获取用于 SHAP 分析的基准模型。优先加载已保存的 LightGBM，否则现场训练一个"""
    model_path = os.path.join(cfg.MODEL_DIR, "sklearn_models.pkl")
    
    if os.path.exists(model_path):
        try:
            trainers = joblib.load(model_path)
            if model_type in trainers:
                logging.info(f"✅ 成功从 saved_models 加载预训练的 {model_type} 模型用于 SHAP 分析")
                return trainers[model_type]
        except Exception as e:
            logging.warning(f"⚠️ 加载预训练模型失败: {e}，将现场训练一个基准模型")

    logging.info(f"🚀 未找到预训练的 {model_type} 模型，正在使用全量训练集现场训练 {model_type} 基准模型...")
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    valid_mask = (df_train['FEATURE_MASK'] == 1)
    df_valid = df_train[valid_mask].dropna(subset=[label_col] + feature_cols)
    
    X = df_valid[feature_cols]
    y = df_valid[label_col]
    
    model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1)
    model.fit(X, y)
    logging.info(f"✅ 基准模型训练完成 | 训练样本数: {len(X):,}")
    return model

def run_quarterly_analysis(cfg: Config, model_type: str = 'LightGBM'):
    """执行按季度的 IC/IR 与 SHAP 值分析"""
    logging.info("📦 开始加载全量训练集数据...")
    df = load_train_data_only(cfg)
    print(f"Column names: {df.columns.tolist()}")
    # 🔑 核心修复：计算 T+REBALANCE_DAYS 的真实收益率作为标签
    logging.info(f"🧮 计算真实收益率标签 label_{cfg.REBALANCE_DAYS}...")
    df = compute_real_returns(cfg.RAW_PANEL, df, i=cfg.REBALANCE_DAYS)
    
    logging.info("🔍 提取有效特征列...")
    feature_cols = extract_valid_features(df)
    logging.info(f"✅ 共提取 {len(feature_cols)} 个有效特征")
    
    label_col = f'label_{cfg.REBALANCE_DAYS}'
    
    # 添加季度列用于分组 (例如: 2016Q1)
    df['QUARTER'] = df['TRADE_DT'].dt.to_period('Q')
    quarters = sorted(df['QUARTER'].unique())
    logging.info(f"📅 检测到 {len(quarters)} 个季度窗口: {quarters[0]} 至 {quarters[-1]}")
    
    # 获取基准模型
    model = get_or_train_model(cfg, feature_cols, df, model_type = model_type)
    explainer = shap.TreeExplainer(model)
    
    # 结果存储字典
    ic_ir_results = {}
    shap_results = {}
    
    # 分析参数
    N_SAMPLING = 20
    SAMPLE_SIZE = 5000
    
    for q in quarters:
        q_str = str(q)
        logging.info(f"▶️ 正在处理季度: {q_str} ...")
        
        # 提取该季度数据
        q_df = df[df['QUARTER'] == q].copy()
        
        # 1. 计算 IC / IR
        try:
            # 过滤掉 FEATURE_MASK != 1 的数据，确保与模型训练口径一致
            q_df_valid = q_df[q_df['FEATURE_MASK'] == 1]
            ic_ir_q = compute_ic_ir(feature_cols, label_col, q_df_valid)
            ic_ir_results[q_str] = ic_ir_q
        except Exception as e:
            logging.error(f"⚠️ {q_str} IC/IR 计算失败: {e}")
            continue
            
        # 2. 计算 SHAP 值 (蒙特卡洛采样)
        q_df_clean = q_df_valid.dropna(subset=feature_cols + [label_col])
        actual_sample_size = min(SAMPLE_SIZE, len(q_df_clean))
        
        if actual_sample_size < 100:
            logging.warning(f"  ⚠️ {q_str} 有效样本过少 ({actual_sample_size})，跳过 SHAP 计算")
            continue
            
        logging.info(f"  🎲 开始 SHAP 采样计算 (共 {N_SAMPLING} 次, 每次 {actual_sample_size} 样本)...")
        
        # 累加器：记录每个特征的平均绝对 SHAP 值总和
        shap_sum = {col: 0.0 for col in feature_cols}
        
        for i in range(N_SAMPLING):
            sample_df = q_df_clean.sample(n=actual_sample_size, random_state=42 + i)
            sample_X = sample_df[feature_cols]
            
            shap_values = explainer.shap_values(sample_X)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            for idx, col in enumerate(feature_cols):
                shap_sum[col] += float(mean_abs_shap[idx])
                
        shap_results[q_str] = {
            col: float(shap_sum[col] / N_SAMPLING) for col in feature_cols
        }
        logging.info(f"  ✅ {q_str} SHAP 分析完成")

    # 3. 保存结果
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
    ic_ir_path = os.path.join(cfg.OUT_DIR, f"ic_ir_quarterly_analysis_{model_type}.json")
    with open(ic_ir_path, 'w', encoding='utf-8') as f:
        json.dump(ic_ir_results, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 IC/IR 结果已保存至: {ic_ir_path}")
    
    shap_path = os.path.join(cfg.OUT_DIR, f"shap_quarterly_analysis_{model_type}.json")
    with open(shap_path, 'w', encoding='utf-8') as f:
        json.dump(shap_results, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 SHAP 结果已保存至: {shap_path}")
    
    shap_df_list = []
    for q_str, shap_dict in shap_results.items():
        row = {'QUARTER': q_str}
        row.update(shap_dict)
        shap_df_list.append(row)
    
    shap_df = pd.DataFrame(shap_df_list)
    shap_df.set_index('QUARTER', inplace=True)
    parquet_path = os.path.join(cfg.OUT_DIR, f"shap_quarterly_analysis_{model_type}.parquet")
    shap_df.to_parquet(parquet_path)
    logging.info(f"💾 SHAP 结果 (Parquet) 已保存至: {parquet_path}")
    
    logging.info(f"🎉 季度 SHAP 与 IC/IR 分析({model_type})全部完成！")



def main():
    setup_logging()
    cfg = Config()
    run_quarterly_analysis(cfg, model_type='LightGBM')
    run_quarterly_analysis(cfg, model_type='XGBoost')  # 如果你有 XGBoost 模型，也可以运行

if __name__ == "__main__":
    main()