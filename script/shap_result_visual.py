import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.Config import Config

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(out_dir, model_type='LightGBM'):
    """加载并解析 JSON 数据"""
    shap_path = os.path.join(out_dir, f"shap_quarterly_analysis_{model_type}.json")
    icir_path = os.path.join(out_dir, f"ic_ir_quarterly_analysis_{model_type}.json")
    
    if not os.path.exists(shap_path) or not os.path.exists(icir_path):
        raise FileNotFoundError(f"❌ 未找到分析结果文件，请先运行 script/shap_analysis.py\n缺失: {shap_path} 或 {icir_path}")
        
    with open(shap_path, 'r', encoding='utf-8') as f:
        shap_data_raw = json.load(f)
    with open(icir_path, 'r', encoding='utf-8') as f:
        icir_data_raw = json.load(f)
        
    first_key = next(iter(shap_data_raw))
    if first_key.startswith('20') and 'Q' in first_key:
        shap_data = {'Default': shap_data_raw}
    else:
        shap_data = shap_data_raw
        
    shap_dfs = {}
    for model, quarters in shap_data.items():
        df = pd.DataFrame(quarters).T
        df.index.name = 'QUARTER'
        df = df.sort_index()
        shap_dfs[model] = df
        
    ic_dict = {}
    for q, features in icir_data_raw.items():
        ic_dict[q] = {feat: vals['mean_ic'] for feat, vals in features.items()}
    ic_df = pd.DataFrame(ic_dict).T
    ic_df.index.name = 'QUARTER'
    ic_df = ic_df.sort_index()
    
    return shap_dfs, ic_df

def plot_visualizations(shap_dfs, ic_df, fig_dir, model_type='LightGBM'):
    """针对每个模型生成 3 张核心图表"""
    os.makedirs(fig_dir, exist_ok=True)
    for model, shap_df in shap_dfs.items():
        print(f"\n🎨 正在为模型 [{model}] 生成可视化图表...")
        mean_shap = shap_df.mean(axis=0).sort_values(ascending=False)
        top10_shap_feats = mean_shap.head(10).index.tolist()
        
        # 图 1: Top 10 SHAP 时序
        plt.figure(figsize=(14, 7))
        for feat in top10_shap_feats:
            plt.plot(shap_df.index, shap_df[feat], marker='o', markersize=4, lw=1.5, label=feat)
        plt.title(f'{model} - Top 10 Features by Mean SHAP Value', fontsize=14)
        plt.xlabel('Quarter'); plt.ylabel('Mean Absolute SHAP Value', fontsize=12)
        plt.xticks(rotation=45, ha='right'); plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'shap_top10_trend_{model}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 图 2: Top 10 IC 时序
        mean_ic_abs = ic_df.mean(axis=0).abs().sort_values(ascending=False)
        top10_ic_feats = mean_ic_abs.head(10).index.tolist()
        plt.figure(figsize=(14, 7))
        for feat in top10_ic_feats:
            plt.plot(ic_df.index, ic_df[feat], marker='s', markersize=4, lw=1.5, label=feat)
        plt.title(f'{model} - Top 10 Features by Mean IC Value', fontsize=14)
        plt.xlabel('Quarter'); plt.ylabel('Mean IC', fontsize=12)
        plt.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5)
        plt.xticks(rotation=45, ha='right'); plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'ic_top10_trend_{model}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 图 3: SHAP 贡献度条形图
        top15_shap_feats = mean_shap.head(15).index.tolist()
        vals = mean_shap[top15_shap_feats].sort_values(ascending=True) 
        plt.figure(figsize=(10, 8))
        bars = plt.barh(vals.index, vals.values, color='teal', edgecolor='black', alpha=0.85)
        for bar in bars:
            plt.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.4f}', va='center', fontsize=9)
        plt.title(f'{model} - Feature Contribution (Top 15)', fontsize=14)
        plt.xlabel('Mean Absolute SHAP Value'); plt.ylabel('Feature Name')
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'shap_contribution_bar_{model}.png'), dpi=150, bbox_inches='tight')
        plt.close()

def main():
    cfg = Config()
    out_dir = cfg.OUT_DIR
    fig_dir = out_dir / "figures"
    
    # 🔑 核心修改：定义所有需要提取消融特征的模型列表
    model_types = [
        'LGBM-22', 'LGBM-23', 'LGBM-24', 'XGB-22', 'XGB-23', 'XGB-24',
        'LGBM-low', 'LGBM-mid', 'LGBM-high', 'XGB-low', 'XGB-mid', 'XGB-high'
    ]
    
    ablation_features = {}
    
    print("📦 开始遍历模型，提取消融特征并生成可视化图表...")
    for m_type in model_types:
        try:
            print(f"\n{'='*40}\n处理模型: {m_type}\n{'='*40}")
            shap_dfs, ic_df = load_and_preprocess_data(str(out_dir), model_type=m_type)
            
            # 🔑 提取消融特征 (SHAP > 0.0001)
            # 注意：这里直接使用 m_type 作为 key，确保与后续训练脚本的命名对齐
            for model, shap_df in shap_dfs.items():
                mean_shap = shap_df.mean(axis=0).sort_values(ascending=False)
                feature_selected = [feature for feature in mean_shap.index if mean_shap[feature] > 0.0001]
                ablation_features[m_type] = feature_selected
                print(f" - {m_type}: 筛选出 {len(feature_selected)} 个消融特征 (SHAP > 0.0001)")
                
            # 生成可视化图表
            plot_visualizations(shap_dfs, ic_df, str(fig_dir), model_type=m_type)
            
        except FileNotFoundError as e:
            print(f"⚠️ 跳过 {m_type}: {e}")
            
    # 🔑 保存所有模型的消融特征配置
    json_path = cfg.ABLATION_FEATURE_JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ablation_features, f, indent=4, ensure_ascii=False)
    print(f"\n✅ 所有模型的消融特征配置已保存至: {json_path}")
    print("\n🎉 所有可视化图表生成完毕！")

if __name__ == "__main__":
    main()