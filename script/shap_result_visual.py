# /data/cye_temp/workspace/backtest_engine/script/shap_analysis.py
import json
import os
import sys  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
print(f"🔧 Backtest Engine Root: {ROOT}")
sys.path.insert(0, str(ROOT))
from config.Config import Config

# 🔧 配置中文字体，防止标题或标签显示为方块 (兼容 Linux/Mac/Windows)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(out_dir):
    """加载并解析 JSON 数据，自动兼容旧版单模型格式"""
    shap_path = os.path.join(out_dir, "shap_quarterly_analysis.json")
    icir_path = os.path.join(out_dir, "ic_ir_quarterly_analysis.json")
    
    if not os.path.exists(shap_path) or not os.path.exists(icir_path):
        raise FileNotFoundError(f"❌ 未找到分析结果文件，请先运行 script/shap_analysis.py\n"
                                f"缺失: {shap_path} 或 {icir_path}")
                                
    with open(shap_path, 'r', encoding='utf-8') as f:
        shap_data_raw = json.load(f)
    with open(icir_path, 'r', encoding='utf-8') as f:
        icir_data_raw = json.load(f)
        
    # 1. 兼容旧版 SHAP 数据 (无模型层级，直接是 Quarter -> Feature)
    first_key = next(iter(shap_data_raw))
    if first_key.startswith('20') and 'Q' in first_key:
        print("⚠️ 检测到旧版 SHAP 数据格式 (无模型层级)，默认归类为 'LightGBM' 进行可视化")
        shap_data = {'LightGBM': shap_data_raw}
    else:
        shap_data = shap_data_raw
        
    # 2. 解析 SHAP 数据为 DataFrame (行: Quarter, 列: Feature)
    shap_dfs = {}
    for model, quarters in shap_data.items():
        df = pd.DataFrame(quarters).T
        df.index.name = 'QUARTER'
        df = df.sort_index() # 确保时间序列有序
        shap_dfs[model] = df
        
    # 3. 解析 IC/IR 数据，提取 mean_ic
    ic_dict = {}
    for q, features in icir_data_raw.items():
        ic_dict[q] = {feat: vals['mean_ic'] for feat, vals in features.items()}
    ic_df = pd.DataFrame(ic_dict).T
    ic_df.index.name = 'QUARTER'
    ic_df = ic_df.sort_index()
    
    return shap_dfs, ic_df

def plot_visualizations(shap_dfs, ic_df, fig_dir):
    """针对每个模型生成 3 张核心图表"""
    os.makedirs(fig_dir, exist_ok=True)

    
    for model, shap_df in shap_dfs.items():
        print(f"\n🎨 正在为模型 [{model}] 生成可视化图表...")
        
        # ==========================================
        # 图 1: Top 10 SHAP 因子时序折线图
        # ==========================================
        # 计算全局平均 SHAP 值并排序
        mean_shap = shap_df.mean(axis=0).sort_values(ascending=False)
        top10_shap_feats = mean_shap.head(10).index.tolist()
        
        plt.figure(figsize=(14, 7))
        for feat in top10_shap_feats:
            plt.plot(shap_df.index, shap_df[feat], marker='o', markersize=4, lw=1.5, label=feat)
            
        plt.title(f'{model} - Top 10 Features by Mean SHAP Value (Quarterly Trend)', fontsize=14)
        plt.xlabel('Quarter', fontsize=12)
        plt.ylabel('Mean Absolute SHAP Value', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        # 图例放在图外右侧，防止遮挡折线
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', frameon=False)
        plt.tight_layout()
        path1 = os.path.join(fig_dir, f'shap_top10_trend_{model}.png')
        plt.savefig(path1, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 图1 (SHAP 时序) 已保存: {path1}")
        
        # ==========================================
        # 图 2: Top 10 IC 因子时序折线图
        # ==========================================
        # 按 IC 绝对值排序，选出正向或负向预测能力最强的 10 个因子
        mean_ic_abs = ic_df.mean(axis=0).abs().sort_values(ascending=False)
        top10_ic_feats = mean_ic_abs.head(10).index.tolist()
        
        plt.figure(figsize=(14, 7))
        for feat in top10_ic_feats:
            plt.plot(ic_df.index, ic_df[feat], marker='s', markersize=4, lw=1.5, label=feat)
            
        plt.title(f'{model} - Top 10 Features by Mean IC Value (Quarterly Trend)', fontsize=14)
        plt.xlabel('Quarter', fontsize=12)
        plt.ylabel('Mean IC (Spearman Correlation)', fontsize=12)
        plt.axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.5) # 零轴参考线
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', frameon=False)
        plt.tight_layout()
        path2 = os.path.join(fig_dir, f'ic_top10_trend_{model}.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 图2 (IC 时序) 已保存: {path2}")
        
        # ==========================================
        # 图 3: SHAP 贡献度水平条形图 (Top 15)
        # ==========================================
        top15_shap_feats = mean_shap.head(15).index.tolist()
        # 升序排列，使得条形图从下到上数值递增，视觉更舒适
        vals = mean_shap[top15_shap_feats].sort_values(ascending=True) 
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(vals.index, vals.values, color='teal', edgecolor='black', alpha=0.85)
        
        # 在条形图右侧添加精确数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.0005, bar.get_y() + bar.get_height()/2, 
                     f'{width:.4f}', va='center', fontsize=9, color='darkslategray')
            
        plt.title(f'{model} - Feature Contribution (Top 15 Mean Abs SHAP)', fontsize=14)
        plt.xlabel('Mean Absolute SHAP Value (Global Importance)', fontsize=12)
        plt.ylabel('Feature Name', fontsize=12)
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        path3 = os.path.join(fig_dir, f'shap_contribution_bar_{model}.png')
        plt.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 图3 (SHAP 贡献度) 已保存: {path3}")

def main():
    # 自动推断 output 目录
    # root_dir = Path(__file__).resolve().parents[1]
    cfg = Config()
    out_dir = cfg.OUT_DIR
    fig_dir = out_dir / "figures"
    
    print("📦 加载季度 SHAP 与 IC/IR 分析结果...")
    shap_dfs, ic_df = load_and_preprocess_data(str(out_dir))
    print(f"SHAP df info: {[ (model, df.shape) for model, df in shap_dfs.items() ]}")
    model, shap_df = shap_dfs.popitem()  # 取出一个模型用于打印样例信息
    print(shap_df.head(3))
    print(f"  - {model}: {shap_df.shape[0]} samples")
    mean_shap = shap_df.mean(axis=0).sort_values(ascending=True)
    print(f"  - {model}: Mean SHAP values (Top 5): {mean_shap[-5:]}")
    shap_selected = [(feature, mean_shap[feature]) for feature in mean_shap.index if mean_shap[feature] > 0.0001]
    feature_selected = [feature for feature in mean_shap.index if mean_shap[feature] > 0.0001]
    print(f" - {model}: Selected features (SHAP > 0.0001): {feature_selected}")
    
    print(f"📊 检测到 {len(shap_dfs)} 个模型的 SHAP 数据，共 {len(ic_df)} 个季度的 IC 数据。")
    
    plot_visualizations(shap_dfs, ic_df, str(fig_dir))
    
    print("\n🎉 所有可视化图表生成完毕！请查看 output/figures/ 目录。")

if __name__ == "__main__":
    main()