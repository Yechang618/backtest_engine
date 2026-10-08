import json
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.Config import Config

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SHAP results for specific models.")
    parser.add_argument('--models', nargs='+', default=None, help='List of models to visualize.')
    parser.add_argument('--group', type=str, choices=['all', 'base', 'percentile', 'cluster', 'ablation'], default='base')
    return parser.parse_args()

def get_target_models(args, cfg):
    if args.models:
        return args.models
    
    if args.group == 'base':
        return ['XGB-22', 'XGB-23', 'XGB-24', 'LGBM-22', 'LGBM-23', 'LGBM-24']
    elif args.group == 'percentile':
        return ['XGB-low', 'XGB-mid', 'XGB-high', 'LGBM-low', 'LGBM-mid', 'LGBM-high',
                'XGB-wLow', 'XGB-wMid', 'XGB-wHigh', 'LGBM-wLow', 'LGBM-wMid', 'LGBM-wHigh']
    elif args.group == 'cluster':
        pkl_path = os.path.join(cfg.MODEL_DIR, "cluster_sklearn_models.pkl")
        if os.path.exists(pkl_path):
            trainers = joblib.load(pkl_path)
            return list(trainers.keys())
        return []
    elif args.group == 'ablation':
        if os.path.exists(cfg.ABLATION_FEATURE_JSON):
            with open(cfg.ABLATION_FEATURE_JSON, 'r', encoding='utf-8') as f:
                feats = json.load(f)
            return list(feats.keys())
        return []
    elif args.group == 'all':
        base = get_target_models(argparse.Namespace(models=None, group='base'), cfg)
        perc = get_target_models(argparse.Namespace(models=None, group='percentile'), cfg)
        clust = get_target_models(argparse.Namespace(models=None, group='cluster'), cfg)
        abl = get_target_models(argparse.Namespace(models=None, group='ablation'), cfg)
        return base + perc + clust + abl
    return []

def load_and_preprocess_data(out_dir, model_type='LightGBM'):
    shap_path = os.path.join(out_dir, f"shap_quarterly_analysis_{model_type}.json")
    icir_path = os.path.join(out_dir, f"ic_ir_quarterly_analysis_{model_type}.json")
    if not os.path.exists(shap_path) or not os.path.exists(icir_path):
        raise FileNotFoundError(f"❌ 未找到分析结果文件，请先运行 script/shap_analysis.py\n缺失: {shap_path} 或 {icir_path}")
    with open(shap_path, 'r', encoding='utf-8') as f: shap_data_raw = json.load(f)
    with open(icir_path, 'r', encoding='utf-8') as f: icir_data_raw = json.load(f)
    
    first_key = next(iter(shap_data_raw))
    shap_data = {'Default': shap_data_raw} if first_key.startswith('20') and 'Q' in first_key else shap_data_raw
    
    shap_dfs = {}
    for model, quarters in shap_data.items():
        df = pd.DataFrame(quarters).T
        df.index.name = 'QUARTER'
        shap_dfs[model] = df.sort_index()
        
    ic_dict = {q: {feat: vals['mean_ic'] for feat, vals in features.items()} for q, features in icir_data_raw.items()}
    ic_df = pd.DataFrame(ic_dict).T
    ic_df.index.name = 'QUARTER'
    return shap_dfs, ic_df.sort_index()

def plot_visualizations(shap_dfs, ic_df, fig_dir, model_type='LightGBM'):
    os.makedirs(fig_dir, exist_ok=True)
    for model, shap_df in shap_dfs.items():
        display_name = model_type if model == 'Default' else model
        print(f"\n🎨 正在为模型 [{display_name}] 生成可视化图表...")
        mean_shap = shap_df.mean(axis=0).sort_values(ascending=False)
        top10_shap_feats = mean_shap.head(10).index.tolist()
        
        plt.figure(figsize=(14, 7))
        for feat in top10_shap_feats:
            plt.plot(shap_df.index, shap_df[feat], marker='o', markersize=4, lw=1.5, label=feat)
        plt.title(f'{display_name} - Top 10 Features by Mean SHAP Value', fontsize=14)
        plt.xlabel('Quarter', fontsize=12); plt.ylabel('Mean Absolute SHAP Value', fontsize=12)
        plt.xticks(rotation=45, ha='right'); plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left', fontsize='small', frameon=False)
        plt.savefig(os.path.join(fig_dir, f'shap_top10_trend_{display_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        mean_ic_abs = ic_df.mean(axis=0).abs().sort_values(ascending=False)
        top10_ic_feats = mean_ic_abs.head(10).index.tolist()
        plt.figure(figsize=(14, 7))
        for feat in top10_ic_feats:
            plt.plot(ic_df.index, ic_df[feat], marker='s', markersize=4, lw=1.5, label=feat)
        plt.title(f'{display_name} - Top 10 Features by Mean IC Value', fontsize=14)
        plt.xlabel('Quarter', fontsize=12); plt.ylabel('Mean IC', fontsize=12)
        plt.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
        plt.xticks(rotation=45, ha='right'); plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left', fontsize='small', frameon=False)
        plt.savefig(os.path.join(fig_dir, f'ic_top10_trend_{display_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        top15_shap_feats = mean_shap.head(15).index.tolist()
        vals = mean_shap[top15_shap_feats].sort_values(ascending=True) 
        plt.figure(figsize=(10, 8))
        bars = plt.barh(vals.index, vals.values, color='teal', edgecolor='black', alpha=0.85)
        for bar in bars:
            plt.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.4f}', va='center', fontsize=9)
        plt.title(f'{display_name} - Feature Contribution (Top 15)', fontsize=14)
        plt.xlabel('Mean Absolute SHAP Value', fontsize=12); plt.ylabel('Feature Name', fontsize=12)
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(fig_dir, f'shap_contribution_bar_{display_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

def main():
    cfg = Config()
    args = parse_args()
    out_dir = cfg.OUT_DIR
    fig_dir = out_dir / "figures"
    
    model_types = get_target_models(args, cfg)
    if not model_types:
        print("❌ 未找到任何需要可视化的模型。")
        return
        
    ablation_features = {}
    print(f"🎯 准备可视化以下 {len(model_types)} 个模型: {model_types}")
    
    for m_type in model_types:
        try:
            print(f"\n{'='*40}\n处理模型: {m_type}\n{'='*40}")
            shap_dfs, ic_df = load_and_preprocess_data(str(out_dir), model_type=m_type)
            
            for model, shap_df in shap_dfs.items():
                mean_shap = shap_df.mean(axis=0).sort_values(ascending=False)
                feature_selected = [feature for feature in mean_shap.index if mean_shap[feature] > 0.00001]
                ablation_features[m_type] = feature_selected
                print(f" - {m_type}: 筛选出 {len(feature_selected)} 个消融特征 (SHAP > 0.0001)")
                
            plot_visualizations(shap_dfs, ic_df, str(fig_dir), model_type=m_type)
        except FileNotFoundError as e:
            print(f"⚠️ 跳过 {m_type}: {e}")
            
    json_path = cfg.ABLATION_FEATURE_JSON
    # 如果是增量更新，可以读取已有的 JSON 并合并
    existing_feats = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            existing_feats = json.load(f)
            
    existing_feats.update(ablation_features)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing_feats, f, indent=4, ensure_ascii=False)
    print(f"\n✅ 所有模型的消融特征配置已保存至: {json_path}")
    print("\n🎉 所有可视化图表生成完毕！")

if __name__ == "__main__":
    main()