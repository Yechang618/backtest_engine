# /data/cye_temp/workspace/backtest_engine/config/Config.py
import os
from pathlib import Path

class Config:
    # 📂 数据路径 (只读)
    Date = "20260827"
    # DATA_RAW_ROOT, DATA_ROOT = r"C:\Users\yecha\workspace\data", r"C:\Users\yecha\workspace\data"
    DATA_RAW_ROOT, DATA_ROOT = "/data/data_process/5.27_update/rebuild", "/data/data_process/5.27_update/rebuild/model_training_step15_selected_panel"
    
    RAW_PANEL = f"{DATA_RAW_ROOT}/market_base_with_industry_20150101_{Date}.parquet"
    DATA_TEST_DIR = f"{DATA_ROOT}/model_ready_panel_selected_fin_ind_pv_adj_masks_rebuild_20150105_{Date}.parquet"
    DATA_DIR = DATA_TEST_DIR
    
    # TRADE_POOL_DIR = f"{DATA_RAW_ROOT}/riva_cn_tech_trade_pools_202601_202608".strip()
    TRADE_POOL_DIR = f"/home/eric/workspace/data/riva_cn_tech_trade_pools_202601_202608".strip()

    # 📂 工程路径 (可写)
    ROOT = Path(__file__).resolve().parents[1]
    ROOT_PRT2 = Path(__file__).resolve().parents[2] / "backtest_engine_output"
    OUT_DIR = ROOT_PRT2 / "output"
    FIG_DIR = OUT_DIR / "figures"
    PROC_DIR = ROOT / "processed_data/backtest"
    LOG_DIR  = ROOT_PRT2 / "log"
    MODEL_DIR = ROOT_PRT2 / "saved_models"

    # 🚫 数据过滤选项
    EXCLUDE_BJ = True  

    # ⚙️ 回测结构参数
    WARMUP_DAYS    = 0      
    REBALANCE_DAYS = 5
    REBALANCE_WEEKDAY = 4   
    TOP_K          = 25
    INITIAL_CAPITAL = 10_000_000.0
    COMMISSION_RATE = 0.0002

    # 💰 交易摩擦
    SLIPPAGE = 0.000
    STAMP_TAX_RATE = 0.0000

    # 🤖 模型配置
    MODELS = [
        'ElasticNet', 
        'XGB-22', 'XGB-23', 'XGB-24', 
        'LGBM-22', 'LGBM-23', 'LGBM-24', 
         'XGB-low', 'XGB-mid', 'XGB-high',
        'LGBM-low', 'LGBM-mid', 'LGBM-high',
        'OptSharpe', 'DynamicSwitch', 'SensitiveSwitch', 
        'BuyAndHoldAll'
    ]    

    # 🔄 动态切换策略配置
    DYNAMIC_SWITCH_INIT_MODEL = 'LGBM-24'
    DYNAMIC_SWITCH_BASE_MODELS = [
        'ElasticNet', 'OptSharpe', 
        'XGB-22', 'XGB-23', 'XGB-24', 
        'LGBM-22', 'LGBM-23', 'LGBM-24',
        'XGB-low', 'XGB-mid', 'XGB-high',
        'LGBM-low', 'LGBM-mid', 'LGBM-high',
        'XGB-22_ablation', 'XGB-23_ablation', 'XGB-24_ablation',
        'LGBM-22_ablation', 'LGBM-23_ablation', 'LGBM-24_ablation',
        'XGB-low_ablation', 'XGB-mid_ablation', 'XGB-high_ablation',
        'LGBM-low_ablation', 'LGBM-mid_ablation', 'LGBM-high_ablation'  
    ]  
    DYNAMIC_SWITCH_B = 1.00  

    # 🔑 SensitiveSwitch 路由配置
    SENSITIVE_SWITCH_INIT_MODEL = 'LGBM-24'
    SENSITIVE_SWITCH_BASE_MODELS = [
        'ElasticNet', 'OptSharpe', 
        'XGB-22', 'XGB-23', 'XGB-24', 
        'LGBM-22', 'LGBM-23', 'LGBM-24',
        'XGB-low', 'XGB-mid', 'XGB-high',
        'LGBM-low', 'LGBM-mid', 'LGBM-high',
        'XGB-22_ablation', 'XGB-23_ablation', 'XGB-24_ablation',
        'LGBM-22_ablation', 'LGBM-23_ablation', 'LGBM-24_ablation',
        'XGB-low_ablation', 'XGB-mid_ablation', 'XGB-high_ablation',
        'LGBM-low_ablation', 'LGBM-mid_ablation', 'LGBM-high_ablation'
    ]  
    SENSITIVE_SWITCH_THRESHOLD = 1.0  

    # 🔍 SHAP 分析配置
    SHAP_SAMPLE_SIZE = 500
    
    # 🔑 消融实验配置 (移除硬编码特征，改为依赖 JSON 文件)
    SHAP_ABLATION = True
    ABLATION_FEATURE_JSON = OUT_DIR / "ablation_feature.json"  # 由 shap_result_visual.py 生成