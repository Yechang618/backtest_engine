# /data/cye_temp/workspace/backtest_engine/config/Config.py
import os
from pathlib import Path

class Config:
    # 📂 数据路径 (只读)
    Date = "20260720"
    DATA_RAW_ROOT, DATA_ROOT = "/data/data_process/5.27_update/rebuild", "/data/data_process/5.27_update/rebuild/model_training_step15_selected_panel"
    RAW_PANEL = f"{DATA_RAW_ROOT}/market_base_with_industry_20150101_{Date}.parquet"
    DATA_TEST_DIR = f"{DATA_ROOT}/model_ready_panel_selected_fin_ind_pv_adj_masks_rebuild_20150105_{Date}.parquet"
    DATA_DIR = DATA_TEST_DIR
    
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
    REBALANCE_DAYS = 10
    TOP_K          = 25
    INITIAL_CAPITAL = 10_000_000.0
    COMMISSION_RATE = 0.0002

    # 💰 交易摩擦 (A股标准)
    SLIPPAGE = 0.000
    STAMP_TAX_RATE = 0.0000

    # 🤖 模型配置
    MODELS = ['ElasticNet', 'XGBoost', 'LightGBM', 'OptSharpe', 'BuyAndHoldAll']    
    
    # 🔍 SHAP 分析配置
    SHAP_SAMPLE_SIZE = 500

    # 消融实验
    SHAP_ABLATION = True
    # Include BJ
    # FEATURE_SELECTED = ['DAYS_SINCE_UPDATE_IC_MKT_Z', 'PV_CAPITAL_LOG_MKT_Z', 'CRSI_RSI3_MKT_Z', 'TURNOVER_SHARE_OF_MARKET_MKT_Z', 'LESS_BEG_BAL_CASH_EQU_CF_IND_Z', 'CR_MA_10_MKT_Z', 'FIX_ASSETS_DISP_BS_IND_Z', 'CR_MA_5_MKT_Z', 'PLUS_END_BAL_CASH_EQU_CF_IND_Z', 'BOTTOM_BUILD_B_5_MKT_Z', 'PRODUCTIVE_BIO_ASSETS_BS_IND_Z', 'BOTTOM_BUILD_D_10_MKT_Z', 'CRSI_STREAK_RSI2_MKT_Z','DAYS_SINCE_LAST_UP_FRACTAL_MKT_Z', 'DAYS_SINCE_LAST_DOWN_FRACTAL_MKT_Z', 'AROON_DOWN_25_MKT_Z', 'STC_10_23_50_MKT_Z', 'FORCE_LOG_TANH_MKT_Z', 'BORROW_CENTRAL_BANK_BS_IND_Z', 'SMR_12_MKT_Z', 'LOANS_OTH_BANKS_BS_IND_Z', 'PSY_12_MKT_Z', 'Breadth_global']
    # Exclude BJ
    FEATURE_SELECTED_LGBM = ['PV_CAPITAL_LOG_MKT_Z', 'BOTTOM_BUILD_B_5_MKT_Z', 'TURNOVER_SHARE_OF_MARKET_MKT_Z', 'FIX_ASSETS_DISP_BS_IND_Z', 'LESS_BEG_BAL_CASH_EQU_CF_IND_Z', 'DAYS_SINCE_UPDATE_IC_MKT_Z', 'PRODUCTIVE_BIO_ASSETS_BS_IND_Z', 'PLUS_END_BAL_CASH_EQU_CF_IND_Z', 'CR_MA_10_MKT_Z', 'CRSI_STREAK_RSI2_MKT_Z', 'CR_MA_5_MKT_Z', 'BOTTOM_BUILD_D_10_MKT_Z', 'SMR_12_MKT_Z', 'DAYS_SINCE_LAST_UP_FRACTAL_MKT_Z', 'PSY_12_MKT_Z', 'AROON_DOWN_25_MKT_Z', 'STC_10_23_50_MKT_Z', 'DAYS_SINCE_LAST_DOWN_FRACTAL_MKT_Z', 'FORCE_LOG_TANH_MKT_Z', 'BORROW_CENTRAL_BANK_BS_IND_Z', 'LOANS_OTH_BANKS_BS_IND_Z', 'Breadth_global']
    FEATURE_SELECTED_XGB =  ['LESS_BEG_BAL_CASH_EQU_CF_IND_Z', 'DAYS_SINCE_UPDATE_IC_MKT_Z', 'PV_CAPITAL_LOG_MKT_Z', 'TURNOVER_SHARE_OF_MARKET_MKT_Z', 'FIX_ASSETS_DISP_BS_IND_Z', 'CR_MA_10_MKT_Z', 'CR_MA_5_MKT_Z', 'PRODUCTIVE_BIO_ASSETS_BS_IND_Z', 'PLUS_END_BAL_CASH_EQU_CF_IND_Z', 'CRSI_STREAK_RSI2_MKT_Z', 'BOTTOM_BUILD_B_5_MKT_Z', 'BOTTOM_BUILD_D_10_MKT_Z', 'DAYS_SINCE_LAST_UP_FRACTAL_MKT_Z', 'SMR_12_MKT_Z', 'PSY_12_MKT_Z', 'AROON_DOWN_25_MKT_Z', 'DAYS_SINCE_LAST_DOWN_FRACTAL_MKT_Z', 'STC_10_23_50_MKT_Z', 'BORROW_CENTRAL_BANK_BS_IND_Z', 'FORCE_LOG_TANH_MKT_Z', 'LOANS_OTH_BANKS_BS_IND_Z', 'Breadth_global']
    FEATURE_SELECTED = FEATURE_SELECTED_LGBM.copy()  # 默认使用 LGBM 特征集作为消融实验特征集