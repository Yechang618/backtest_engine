# /data/cye_temp/workspace/backtest_engine/config/Config.py
import os
from pathlib import Path

class Config:
    # 📂 数据路径 (只读)
    Date = "20260821"
    DATA_RAW_ROOT, DATA_ROOT = "/data/data_process/5.27_update/rebuild", "/data/data_process/5.27_update/rebuild/model_training_step15_selected_panel"
    # DATA_RAW_ROOT, DATA_ROOT = r"C:\Users\yecha\workspace\data", r"C:\Users\yecha\workspace\data"
    RAW_PANEL = f"{DATA_RAW_ROOT}/market_base_with_industry_20150101_{Date}.parquet"
    DATA_TEST_DIR = f"{DATA_ROOT}/model_ready_panel_selected_fin_ind_pv_adj_masks_rebuild_20150105_{Date}.parquet"
    DATA_DIR = DATA_TEST_DIR

    # 🔑 新增：股票池目录路径 (注意去除可能存在的末尾空格)
    TRADE_POOL_DIR = f"{DATA_RAW_ROOT}/riva_cn_tech_trade_pools_202601_202608".strip()
    # TRADE_POOL_DIR = f"{DATA_RAW_ROOT}/riva_cn_tech_trade_pools_202601_202608"
    
    # 📂 工程路径 (可写)
    ROOT = Path(__file__).resolve().parents[1]
    ROOT_PRT2 = Path(__file__).resolve().parents[2] / "backtest_engine_output"
    OUT_DIR = ROOT_PRT2 / "output"
    FIG_DIR = OUT_DIR / "figures"
    PROC_DIR = ROOT / "processed_data/backtest"
    LOG_DIR  = ROOT_PRT2 / "log"
    MODEL_DIR = ROOT_PRT2 / "saved_models"

    # 🚫 数据过滤选项 (🔑 新增)
    EXCLUDE_BJ = True  # 是否排除北交所数据 (Wind代码以 '_BJ' 结尾)

    # ⚙️ 回测结构参数
    WARMUP_DAYS    = 0      
    REBALANCE_DAYS = 5
    # 🔑 新增：指定每周调仓的星期几 (0=周一, 1=周二, 2=周三, 3=周四, 4=周五)。
    # 若设为 None，则回退使用 REBALANCE_DAYS 的固定天数逻辑。
    REBALANCE_WEEKDAY = 4
    
    TOP_K          = 25
    INITIAL_CAPITAL = 10_000_000.0
    COMMISSION_RATE = 0.0002

    # 💰 交易摩擦 (A股标准)
    SLIPPAGE = 0.000
    STAMP_TAX_RATE = 0.0000

    # 🤖 模型配置 ✅ 细分为 22, 23, 24 三个时间窗口变体
    MODELS = [
        'ElasticNet', 
        'XGB-22', 'XGB-23', 'XGB-24', 
        'LGBM-22', 'LGBM-23', 'LGBM-24', 
        'OptSharpe', 'DynamicSwitch', 'SensitiveSwitch', 
        'BuyAndHoldAll'
    ]    

    # 🔄 动态切换策略配置 (包含所有细分模型及其消融版本)
    # DYNAMIC_SWITCH_INIT_MODEL = 'OptSharpe'
    DYNAMIC_SWITCH_INIT_MODEL =  'LGBM-24'
    DYNAMIC_SWITCH_BASE_MODELS = [
        'ElasticNet', 'OptSharpe', 
        'XGB-22', 'XGB-23', 'XGB-24', 
        'LGBM-22', 'LGBM-23', 'LGBM-24',
        # 'XGB-22_ablation', 'XGB-23_ablation', 'XGB-24_ablation',
        # 'LGBM-22_ablation', 'LGBM-23_ablation', 'LGBM-24_ablation'
    ]  
    DYNAMIC_SWITCH_B = 1.00  


    # 🔑 新增：SensitiveSwitch (基于 Residual 分数) 路由配置
    SENSITIVE_SWITCH_INIT_MODEL = 'LGBM-24'
    SENSITIVE_SWITCH_BASE_MODELS = [
        'ElasticNet', 'OptSharpe', 
        'XGB-22', 'XGB-23', 'XGB-24', 
        'LGBM-22', 'LGBM-23', 'LGBM-24',
        # 'XGB-22_ablation', 'XGB-23_ablation', 'XGB-24_ablation',
        # 'LGBM-22_ablation', 'LGBM-23_ablation', 'LGBM-24_ablation'
    ]  
    SENSITIVE_SWITCH_THRESHOLD = 1.0  # Residual 分数阈值

    # 🔍 SHAP 分析配置
    SHAP_SAMPLE_SIZE = 500

    # 消融实验
    # SHAP_ABLATION = False
    SHAP_ABLATION = True
    # Include BJ
    # FEATURE_SELECTED = ['DAYS_SINCE_UPDATE_IC_MKT_Z', 'PV_CAPITAL_LOG_MKT_Z', 'CRSI_RSI3_MKT_Z', 'TURNOVER_SHARE_OF_MARKET_MKT_Z', 'LESS_BEG_BAL_CASH_EQU_CF_IND_Z', 'CR_MA_10_MKT_Z', 'FIX_ASSETS_DISP_BS_IND_Z', 'CR_MA_5_MKT_Z', 'PLUS_END_BAL_CASH_EQU_CF_IND_Z', 'BOTTOM_BUILD_B_5_MKT_Z', 'PRODUCTIVE_BIO_ASSETS_BS_IND_Z', 'BOTTOM_BUILD_D_10_MKT_Z', 'CRSI_STREAK_RSI2_MKT_Z','DAYS_SINCE_LAST_UP_FRACTAL_MKT_Z', 'DAYS_SINCE_LAST_DOWN_FRACTAL_MKT_Z', 'AROON_DOWN_25_MKT_Z', 'STC_10_23_50_MKT_Z', 'FORCE_LOG_TANH_MKT_Z', 'BORROW_CENTRAL_BANK_BS_IND_Z', 'SMR_12_MKT_Z', 'LOANS_OTH_BANKS_BS_IND_Z', 'PSY_12_MKT_Z', 'Breadth_global']
    # Exclude BJ
    FEATURE_SELECTED =  {
        'LGBM-22': ['BORROW_CENTRAL_BANK_BS_IND_Z', 'BOTTOM_BUILD_B_5_MKT_Z', 'FIX_ASSETS_DISP_BS_IND_Z', 'LESS_BEG_BAL_CASH_EQU_CF_IND_Z', 'Breadth_global'],
        'LGBM-23': ['BORROW_CENTRAL_BANK_BS_IND_Z', 'IFT_RSI_14_MKT_Z', 'BOTTOM_BUILD_B_5_MKT_Z', 'LOANS_OTH_BANKS_BS_IND_Z', 'FIX_ASSETS_DISP_BS_IND_Z', 'DAYS_SINCE_LAST_DOWN_FRACTAL_MKT_Z', 'LESS_BEG_BAL_CASH_EQU_CF_IND_Z', 'STC_10_23_50_MKT_Z', 'Breadth_global'],
        'LGBM-24': ['DAYS_SINCE_UPDATE_CF_MKT_Z', 'PVT_REL_20_MKT_Z', 'DAYS_SINCE_UPDATE_IC_MKT_Z', 'FORCE_EMA_2_MKT_Z', 'Breadth_industry', 'FIX_ASSETS_DISP_BS_IND_Z', 'KAMA_SLOPE_MKT_Z', 'TURNOVER_SHARE_OF_MARKET_MKT_Z', 'DAYS_SINCE_UPDATE_BS_MKT_Z', 'CR_MA_10_MKT_Z', 'BOTTOM_BUILD_B_5_MKT_Z', 'PRODUCTIVE_BIO_ASSETS_BS_IND_Z', 'SMR_12_MKT_Z', 'PV_LIMIT_RET_FROM_PRECLOSE_MKT_Z', 'PSY_12_MKT_Z', 'SHARPE_10D_MKT_Z', 'BOTTOM_BUILD_D_10_MKT_Z', 'AROON_DOWN_25_MKT_Z', 'IFT_RSI_14_MKT_Z', 'CRSI_STREAK_RSI2_MKT_Z', 'DAYS_SINCE_LAST_UP_FRACTAL_MKT_Z', 'FORCE_LOG_TANH_MKT_Z', 'LESS_BEG_BAL_CASH_EQU_CF_IND_Z', 'BORROW_CENTRAL_BANK_BS_IND_Z', 'STC_10_23_50_MKT_Z', 'DAYS_SINCE_LAST_DOWN_FRACTAL_MKT_Z', 'LOANS_OTH_BANKS_BS_IND_Z', 'Breadth_global'],
        'XGB-22': ['PLUS_END_BAL_CASH_EQU_CF_IND_Z', 'CMO_14_MKT_Z', 'NOTES_PAYABLE_BS_IND_Z', 'DAYS_SINCE_UPDATE_CF_MKT_Z', 'AROON_UP_25_MKT_Z', 'PV_STOPPING_RET_FROM_PRECLOSE_MKT_Z', 'DAYS_SINCE_UPDATE_IC_MKT_Z', 'PSY_12_MKT_Z', 'FIX_ASSETS_DISP_BS_IND_Z', 'FORCE_EMA_2_MKT_Z', 'KAMA_SLOPE_MKT_Z', 'CR_MA_5_MKT_Z', 'DAYS_SINCE_UPDATE_BS_MKT_Z', 'SHARPE_10D_MKT_Z', 'CR_MA_10_MKT_Z', 'IFT_RSI_14_MKT_Z', 'PRODUCTIVE_BIO_ASSETS_BS_IND_Z', 'FORCE_LOG_TANH_MKT_Z', 'BOTTOM_BUILD_B_5_MKT_Z', 'CRSI_STREAK_RSI2_MKT_Z', 'AROON_DOWN_25_MKT_Z', 'PV_LIMIT_RET_FROM_PRECLOSE_MKT_Z', 'BOTTOM_BUILD_D_10_MKT_Z', 'LESS_BEG_BAL_CASH_EQU_CF_IND_Z', 'DAYS_SINCE_LAST_UP_FRACTAL_MKT_Z', 'STC_10_23_50_MKT_Z', 'DAYS_SINCE_LAST_DOWN_FRACTAL_MKT_Z', 'BORROW_CENTRAL_BANK_BS_IND_Z', 'LOANS_OTH_BANKS_BS_IND_Z', 'Breadth_global'],
        'XGB-23': ['PSY_12_MKT_Z', 'TURNOVER_SHARE_OF_MARKET_MKT_Z', 'FORCE_EMA_2_MKT_Z', 'CMO_14_MKT_Z', 'CR_MA_5_MKT_Z', 'CR_MA_10_MKT_Z', 'PV_LIMIT_RET_FROM_PRECLOSE_MKT_Z', 'KAMA_SLOPE_MKT_Z', 'PRODUCTIVE_BIO_ASSETS_BS_IND_Z', 'FIX_ASSETS_DISP_BS_IND_Z', 'DAYS_SINCE_UPDATE_BS_MKT_Z', 'CRSI_STREAK_RSI2_M KT Z', 'AROON_DOWN_25_M KT Z', 'IFT_RSI_14 M KT Z', 'BOTTOM_BUILD_D_10 M KT Z ', 	'BOTTOM_BUILD_B_5 M KT Z ', 	'SHARPE _10 D M KT Z ', 	'DAYS _SINCE_LAST_UP_FRACTAL M KT Z ', 	'LESS_BEG_BAL_CASH_EQU_CF_IND _Z ', 	'FORCE_LOG_TANH M KT Z ', 	'DAYS _SINCE_LAST_DOWN_FRACTAL M KT Z ', 	'STC _10 _23 _50 M KT Z ', 	'BORROW_CENTRAL_BANK_BS_IND _Z ', 	'LOANS_OTH_BANKS_BS_IND _Z ', 	'Breadth_global'],
        'XGB-24': ['DAYS_SINCE_UPDATE_IC M KT Z ', 	'PVT_REL _20 M KT Z ', 	'PV_CAPITAL_LOG M KT Z ', 	'Breadth_industry ', 	'PROV_NOM_RISKS_BS_IND _Z ', 	'DAYS_SINCE_UPDATE_BS M KT Z ', 	'FORCE_EMA _2 M KT Z ', 	'FIX_ASSETS_DISP_BS_IND _Z ', 	'SMR _12 M KT Z ', 	'CR_MA _10 M KT Z ', 	'PV_LIMIT_RET_FROM_PRECLOSE M KT Z ', 	'KAMA_SLOPE M KT Z ', 	'TURNOVER_SHARE_OF_MARKET M KT Z '],
                }
    # FEATURE_SELECTED = FEATURE_SELECTED_LGBM['LGBM-22'].copy()  # 默认使用 LGBM 特征集作为消融实验特征集