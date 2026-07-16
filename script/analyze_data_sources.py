# script/analyze_data_sources.py
import sys
import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Set

# 尝试导入 pyarrow 以高效读取 parquet 列名，避免内存溢出
try:
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("请安装 pyarrow: pip install pyarrow")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.Config import Config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def get_columns_from_parquet(file_path: str) -> List[str]:
    """安全且高效地获取 Parquet 文件的列名 (不加载数据)"""
    try:
        schema = pq.read_schema(file_path)
        return schema.names
    except Exception as e:
        logging.warning(f"⚠️ PyArrow 读取失败 {file_path}: {e}，回退至 Pandas 读取...")
        df = pd.read_parquet(file_path)
        return df.columns.tolist()

def load_dataset_columns(data_dir: str, data_test_dir: str, raw_panel: str) -> Dict[str, Set[str]]:
    """
    加载三个数据源的列名集合。
    兼容 Config.py 中路径末尾可能存在的空格，以及 DATA_TEST_DIR 可能是文件的情况。
    """
    datasets = {"TRAIN_DATA": set(), "TEST_DATA": set(), "RAW_PANEL": set()}
    
    # # 1. 加载 TRAIN_DATA (遍历目录)
    # data_dir = str(data_dir).strip()
    # if os.path.isdir(data_dir):
    #     files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    #     if not files:
    #         logging.error(f"❌ 训练集目录为空: {data_dir}")
    #     for f in files:
    #         cols = get_columns_from_parquet(os.path.join(data_dir, f))
    #         datasets["TRAIN_DATA"].update(cols)
    #     logging.info(f"✅ TRAIN_DATA 加载完成 | 文件数: {len(files)} | 唯一列数: {len(datasets['TRAIN_DATA'])}")
    # else:
    #     logging.error(f"❌ TRAIN_DATA 路径无效或不是目录: {data_dir}")

    # 1. 加载 TRAIN_DATA (兼容目录或单文件)
    data_dir = str(data_dir).strip()
    if os.path.isdir(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        if not files:
            logging.error(f"❌ 训练集目录为空: {data_dir}")
        for f in files:
            cols = get_columns_from_parquet(os.path.join(data_dir, f))
            datasets["TRAIN_DATA"].update(cols)
        logging.info(f"✅ TRAIN_DATA 加载完成 | 文件数: {len(files)} | 唯一列数: {len(datasets['TRAIN_DATA'])}")
    elif os.path.isfile(data_dir):
        cols = get_columns_from_parquet(data_dir)
        datasets["TRAIN_DATA"].update(cols)
        logging.info(f"✅ TRAIN_DATA (单文件) 加载完成 | 唯一列数: {len(datasets['TRAIN_DATA'])}")
    else:
        logging.error(f"❌ TRAIN_DATA 路径无效或不是目录: {data_dir}")

    # 2. 加载 TEST_DATA (兼容目录或单文件)
    data_test_dir = str(data_test_dir).strip()
    if os.path.isdir(data_test_dir):
        files = [f for f in os.listdir(data_test_dir) if f.endswith('.parquet')]
        for f in files:
            cols = get_columns_from_parquet(os.path.join(data_test_dir, f))
            datasets["TEST_DATA"].update(cols)
    elif os.path.isfile(data_test_dir):
        cols = get_columns_from_parquet(data_test_dir)
        datasets["TEST_DATA"].update(cols)
        logging.info(f"✅ TEST_DATA (单文件) 加载完成 | 唯一列数: {len(datasets['TEST_DATA'])}")
    else:
        logging.error(f"❌ TEST_DATA 路径无效: {data_test_dir}")
        
    if "TEST_DATA" in datasets and datasets["TEST_DATA"]:
        logging.info(f"✅ TEST_DATA 加载完成 | 唯一列数: {len(datasets['TEST_DATA'])}")

    # 3. 加载 RAW_PANEL (单文件)
    raw_panel = str(raw_panel).strip()
    if os.path.isfile(raw_panel):
        cols = get_columns_from_parquet(raw_panel)
        datasets["RAW_PANEL"].update(cols)
        logging.info(f"✅ RAW_PANEL 加载完成 | 唯一列数: {len(datasets['RAW_PANEL'])}")
    else:
        logging.error(f"❌ RAW_PANEL 路径无效: {raw_panel}")

    return datasets

def categorize_columns(columns: Set[str]) -> Dict[str, List[str]]:
    """
    依据 selected_training_fields_dictionary_cn 规范对列进行严格归类
    """
    categories = {
        "metadata": [],
        "price_ohlc": [],
        "label": [],
        "control_mask": [],
        "field_mask": [],
        "factor": [],
        "unknown": []
    }
    
    for col in columns:
        # 1. 元数据 (Metadata)
        if col in ['S_INFO_WINDCODE', 'TRADE_DT', 'SW_L1_CODE', 'DATA_SOURCE']:
            categories["metadata"].append(col)
            continue
            
        # 2. 原始价格/OHLC (Price/OHLC)
        if col.startswith('S_DQ_'):
            categories["price_ohlc"].append(col)
            continue
            
        # 3. 标签 (Label)
        if col.startswith('FWD_RET') or re.match(r'^label_\d+$', col):
            categories["label"].append(col)
            continue
            
        # 4. 控制掩码 (Control Mask)
        if col in ['FEATURE_MASK', 'BUY_MASK', 'SELL_MASK']:
            categories["control_mask"].append(col)
            continue
            
        # 5. 字段掩码 (Field Mask)
        if (col.endswith('__APPLICABLE_MASK') or col.endswith('__MISSING_MASK') or 
            col.endswith('_MISSING_FLAG') or col == 'APPLICABLE_MISSING_RATE'):
            categories["field_mask"].append(col)
            continue
            
        # 6. 因子 (Factor) - 依据字典规则
        if re.match(r'^Breadth_[^_]+$', col) or col.endswith('_MKT_Z') or col.endswith('_IND_Z'):
            categories["factor"].append(col)
            continue
            
        # 7. 未识别列
        categories["unknown"].append(col)
        
    # 排序以便阅读
    for k in categories:
        categories[k].sort()
        
    return categories

def generate_comparison_report(datasets_cols: Dict[str, Set[str]]) -> Dict:
    """比较三个数据集的列，找出交集与差集"""
    train_cols = datasets_cols.get("TRAIN_DATA", set())
    test_cols = datasets_cols.get("TEST_DATA", set())
    raw_cols = datasets_cols.get("RAW_PANEL", set())
    
    # 1. 三者完全一致的列 (交集)
    common_cols = train_cols & test_cols & raw_cols
    
    # 2. 各自独有的列 (差集)
    unique_train = train_cols - test_cols - raw_cols
    unique_test = test_cols - train_cols - raw_cols
    unique_raw = raw_cols - train_cols - test_cols
    
    # 3. 不一致的列 (非三者交集的所有列)
    all_cols = train_cols | test_cols | raw_cols
    inconsistent_cols = all_cols - common_cols
    
    return {
        "common_columns": {"count": len(common_cols), "columns": sorted(list(common_cols))},
        "unique_to_train_data": {"count": len(unique_train), "columns": sorted(list(unique_train))},
        "unique_to_test_data": {"count": len(unique_test), "columns": sorted(list(unique_test))},
        "unique_to_raw_panel": {"count": len(unique_raw), "columns": sorted(list(unique_raw))},
        "inconsistent_columns": {"count": len(inconsistent_cols), "columns": sorted(list(inconsistent_cols))}
    }

def main():
    setup_logging()
    cfg = Config()
    
    out_dir = ROOT / "output" / "data_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("🚀 开始分析数据源基础信息 (使用 PyArrow 零内存加载模式)...")
    
    # 1. 获取列名
    datasets_cols = load_dataset_columns(cfg.DATA_DIR, cfg.DATA_TEST_DIR, cfg.RAW_PANEL)
    
    # 2. 对每个数据集进行归类并生成报告
    for ds_name, cols in datasets_cols.items():
        if not cols:
            logging.warning(f"⚠️ {ds_name} 未获取到任何列，跳过报告生成。")
            continue
            
        logging.info(f"🔍 正在对 [{ds_name}] 的 {len(cols)} 个列进行归类...")
        categorized = categorize_columns(cols)
        
        report = {
            "dataset_name": ds_name,
            "total_columns": len(cols),
            "categories_summary": {k: len(v) for k, v in categorized.items()},
            "categories_detail": categorized
        }
        
        path = out_dir / f"schema_report_{ds_name}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logging.info(f"💾 [{ds_name}] 归类报告已保存: {path}")
        
    # 3. 生成比较报告
    logging.info("🔄 正在比较三个数据集的列一致性...")
    comparison = generate_comparison_report(datasets_cols)
    
    comp_path = out_dir / "schema_comparison_report.json"
    with open(comp_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 数据集比较报告已保存: {comp_path}")
    
    # 4. 终端打印核心摘要
    print("\n" + "="*60)
    print("📊 数据源 Schema 分析摘要")
    print("="*60)
    for ds_name, cols in datasets_cols.items():
        print(f"[{ds_name}] 总列数: {len(cols)}")
    print(f"\n🔗 三个数据集【完全一致】的列数: {comparison['common_columns']['count']}")
    print(f"⚠️ 【不一致】的列数 (并集 - 交集): {comparison['inconsistent_columns']['count']}")
    print(f"   - 仅 TRAIN 独有: {comparison['unique_to_train_data']['count']}")
    print(f"   - 仅 TEST 独有: {comparison['unique_to_test_data']['count']}")
    print(f"   - 仅 RAW 独有: {comparison['unique_to_raw_panel']['count']}")
    print("="*60 + "\n")
    
    logging.info("✅ 数据源分析全部完成！请查看 output/data_analysis/ 目录。")

if __name__ == "__main__":
    main()