# /data/cye_temp/workspace/backtest_engine/script/analysis.py
import pandas as pd
import numpy as np
import json
import os
from util.data_loader import compute_ic_ir

def run_factor_analysis(df, start_date, end_date, feature_cols, out_path):
    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    df_sub = df[(df['TRADE_DT'] >= start) & (df['TRADE_DT'] <= end)].copy()
    df_sub = df_sub[df_sub['FEATURE_MASK'] == 1]
    print(f"📊 开始因子分析 | 样本范围: {start_date} ~ {end_date} | 因子数: {len(feature_cols)}")

    stats = df_sub[feature_cols].agg(['mean', 'var', 'max', 'min']).fillna(0.0).to_dict(orient='index')

    # IC/IR 计算
    ic_label1 = compute_ic_ir(feature_cols, 'label_1', df_sub)
    ic_label2 = compute_ic_ir(feature_cols, 'FWD_RET_5D_Z_P01_P99', df_sub)

    res = {}
    for col in feature_cols:
        res[col] = {
            "mean": float(stats['mean'].get(col, 0.0)),
            "variance": float(stats['var'].get(col, 0.0)),
            "max": float(stats['max'].get(col, 0.0)),
            "min": float(stats['min'].get(col, 0.0)),
            "mean_ic_label_1": ic_label1.get(col, {}).get('mean_ic', 0.0),
            "icir_label_1": ic_label1.get(col, {}).get('icir', 0.0),
            "ic_positive_ratio_label_1": ic_label1.get(col, {}).get('ic_positive_ratio', 0.0),
            "mean_ic_label_2": ic_label2.get(col, {}).get('mean_ic', 0.0),
            "icir_label_2": ic_label2.get(col, {}).get('icir', 0.0),
            "ic_positive_ratio_label_2": ic_label2.get(col, {}).get('ic_positive_ratio', 0.0),
            "sample_days": ic_label1.get(col, {}).get('sample_days', 0)
        }
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    print(f"✅ 因子分析完成，已保存至 {out_path}")
    return res