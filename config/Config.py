# /data/cye_temp/workspace/backtest_engine/config/Config.py
import os
from pathlib import Path

import os
from pathlib import Path

class Config:
    # 📂 数据路径 (只读)
    Date = "20261027"
    DATA_RAW_ROOT, DATA_ROOT = "/data/data_process/5.27_update/rebuild", "/data/data_process/5.27_update/rebuild/model_training_step15_selected_panel"
    RAW_PANEL = f"{DATA_RAW_ROOT}/market_base_with_industry_20150101_{Date}.parquet"
    DATA_TEST_DIR = f"{DATA_ROOT}/model_ready_panel_selected_fin_ind_pv_adj_masks_rebuild_20150105_{Date}.parquet"
    DATA_DIR = DATA_TEST_DIR
    TRADE_POOL_DIR = f"/home/eric/workspace/data/riva_cn_tech_trade_pools_202601_202608".strip()

    # 📂 工程基础路径 (可写)
    ROOT = Path(__file__).resolve().parents[1]
    ROOT_PRT2 = Path(__file__).resolve().parents[2] / "backtest_engine_output"

    # 🔑 新增：Label 类型切换配置
    # 可选值: 'return' (使用 label_{i}), 'sharpe' (使用 label_sp{i})
    # LABEL_TYPE = 'sharpe'  # <--- 在这里切换 'return' 或 'sharpe'
    LABEL_TYPE = 'return'

    def __init__(self):
        
        self.REBALANCE_DAYS = 5
        self.LABEL_TYPE == 'return'
        
        # 1. 🔑 核心修复：动态生成 Label 后缀和目标列名，确保与 data_loader.py 完全一致
        if self.LABEL_TYPE == 'sharpe':
            self.LABEL_SUFFIX = '_sp'
            self.TARGET_LABEL_COL = f'label_sp{self.REBALANCE_DAYS}'  # 生成 label_sp5
        else:
            self.LABEL_SUFFIX = ''
            self.TARGET_LABEL_COL = f'label_{self.REBALANCE_DAYS}'    # 生成 label_5
            
        # 2. 动态生成带后缀的输出路径，实现物理隔离
        suffix = self.LABEL_SUFFIX
        self.OUT_DIR = self.ROOT_PRT2 / f"output{suffix}" if suffix else self.ROOT_PRT2 / "output"
        self.FIG_DIR = self.OUT_DIR / "figures"
        self.LOG_DIR = self.ROOT_PRT2 / f"log{suffix}" if suffix else self.ROOT_PRT2 / "log"
        self.MODEL_DIR = self.ROOT_PRT2 / f"saved_models{suffix}" if suffix else self.ROOT_PRT2 / "saved_models"
        self.PROC_DIR = self.ROOT / f"processed_data/backtest{suffix}" if suffix else self.ROOT / "processed_data/backtest"

        # 3. 确保目录存在
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.FIG_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # # 2. 动态生成带后缀的输出路径，实现物理隔离
        # suffix = self.LABEL_SUFFIX
        # self.OUT_DIR = self.ROOT_PRT2 / f"output{suffix}" if suffix else self.ROOT_PRT2 / "output"
        # self.FIG_DIR = self.OUT_DIR / "figures"
        # self.LOG_DIR = self.ROOT_PRT2 / f"log{suffix}" if suffix else self.ROOT_PRT2 / "log"
        # self.MODEL_DIR = self.ROOT_PRT2 / f"saved_models{suffix}" if suffix else self.ROOT_PRT2 / "saved_models"
        # self.PROC_DIR = self.ROOT / f"processed_data/backtest{suffix}" if suffix else self.ROOT / "processed_data/backtest"

        # # 3. 确保目录存在
        # self.OUT_DIR.mkdir(parents=True, exist_ok=True)
        # self.FIG_DIR.mkdir(parents=True, exist_ok=True)
        # self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        # self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # 4. 其他固定配置
        self.EXCLUDE_BJ = True  
        self.WARMUP_DAYS = 0      
        # self.REBALANCE_DAYS = 5
        self.REBALANCE_WEEKDAY = 4   
        self.TOP_K = 25
        self.INITIAL_CAPITAL = 10_000_000.0
        self.COMMISSION_RATE = 0.0002
        self.SLIPPAGE = 0.000
        self.STAMP_TAX_RATE = 0.0000
        
        # ... (保留原有的 MODELS, DYNAMIC_SWITCH 等所有策略配置不变) ...
        self.CLUSTER_MODEL_PKL = self.MODEL_DIR / "cluster_model.pkl"
        self.MARKET_REGIME_LABELS = self.OUT_DIR / "market_regime_labels.json"
        #
        self.MODELS = [
            'ElasticNet', 'XGB-22', 'XGB-23', 'XGB-24', 'LGBM-22', 'LGBM-23', 'LGBM-24', 
            'XGB-low', 'XGB-mid', 'XGB-high', 'LGBM-low', 'LGBM-mid', 'LGBM-high',
            'XGB-wLow', 'XGB-wMid', 'XGB-wHigh', 'LGBM-wLow', 'LGBM-wMid', 'LGBM-wHigh',
            'OptSharpe', 'DynamicSwitch', 'DynamicSwitch2', 'DynamicSwitch_IR', 
            'DynamicSwitch2_IR', 'DynamicSwitch3', 'DynamicSwitch4', 'ClusterRegime', 
            'SensitiveSwitch', 'BuyAndHoldAll'
        ]
        # ... (保留其余所有 DYNAMIC_SWITCH... 等配置) ...
        
        # 🔑 新增：DynamicSwitch3 (Optimal Sharpe 模型加权) 路由配置
        self.DYNAMIC_SWITCH3_BASE_MODELS = []  # 将在 run_backtest.py 中动态赋值为所有树模型
        self.DYNAMIC_SWITCH3_WINDOW = 10       # 计算最优权重的滚动窗口大小 (天数)
        # 🔑 新增：DynamicSwitch4 (个股级 Optimal Sharpe 模型加权) 路由配置
        self.DYNAMIC_SWITCH4_BASE_MODELS = []  # 将在 run_backtest.py 中动态赋值为所有树模型
        self.DYNAMIC_SWITCH4_WINDOW = 10       # 计算个股最优权重的滚动窗口大小 (自然日)
        
        # 🔄 动态切换策略配置
        self.DYNAMIC_SWITCH_INIT_MODEL = 'LGBM-24'
        self.DYNAMIC_SWITCH_BASE_MODELS = [
            'ElasticNet', 'OptSharpe', 
            'XGB-22', 'XGB-23', 'XGB-24', 
            'LGBM-22', 'LGBM-23', 'LGBM-24',
            'XGB-low', 'XGB-mid', 'XGB-high',
            'LGBM-low', 'LGBM-mid', 'LGBM-high',
            'XGB-wLow', 'XGB-wMid', 'XGB-wHigh',  # 🔑 新增周度波动率模型
            'LGBM-wLow', 'LGBM-wMid', 'LGBM-wHigh', # 🔑 新增周度波动率模型
            'XGB-22_ablation', 'XGB-23_ablation', 'XGB-24_ablation',
            'LGBM-22_ablation', 'LGBM-23_ablation', 'LGBM-24_ablation',
            'LGBM_c1', 'LGBM_c2', 'LGBM_c3',  # 🔑 新增
            'LGBM_c4', 'LGBM_c5', 'LGBM_c6',  # 🔑 新增
            'XGB_c1', 'XGB_c2', 'XGB_c3'      # 🔑 新增
            'XGB_c4', 'XGB_c5', 'XGB_c6'      # 🔑 新增
        ]  
        self.DYNAMIC_SWITCH_B = 1.00  

        # 🔑 新增：DynamicSwitch2 (IC 加权集成) 路由配置
        self.DYNAMIC_SWITCH2_BASE_MODELS = self.DYNAMIC_SWITCH_BASE_MODELS
        self.DYNAMIC_SWITCH2_WINDOW = 10  # 计算 IC 权重的滚动窗口大小

        # 🔑 新增：DynamicSwitch_IR (基于 IR 切换) 路由配置
        self.DYNAMIC_SWITCH_IR_INIT_MODEL = 'LGBM-24'
        self.DYNAMIC_SWITCH_IR_BASE_MODELS = self.DYNAMIC_SWITCH_BASE_MODELS
        self.DYNAMIC_SWITCH_IR_B = 1.00  # 切换阈值倍数 (新模型 IR > B * 当前模型 IR 时切换)

        # 🔑 新增：DynamicSwitch2_IR (基于 IR 加权) 路由配置
        self.DYNAMIC_SWITCH2_IR_BASE_MODELS = self.DYNAMIC_SWITCH_BASE_MODELS

        # 🔑 SensitiveSwitch 路由配置
        self.SENSITIVE_SWITCH_INIT_MODEL = 'LGBM-24'
        self.SENSITIVE_SWITCH_BASE_MODELS = self.DYNAMIC_SWITCH_BASE_MODELS
        self.SENSITIVE_SWITCH_THRESHOLD = 1.0  

        # 🔍 SHAP 分析配置
        self.SHAP_SAMPLE_SIZE = 500
        
        self.SHAP_ABLATION = True
        self.ABLATION_FEATURE_JSON = self.OUT_DIR / "ablation_feature.json"

# class Config:
#     # 📂 数据路径 (只读)
#     Date = "20261019"
#     # DATA_RAW_ROOT, DATA_ROOT = r"C:\Users\yecha\workspace\data", r"C:\Users\yecha\workspace\data"
#     DATA_RAW_ROOT, DATA_ROOT = "/data/data_process/5.27_update/rebuild", "/data/data_process/5.27_update/rebuild/model_training_step15_selected_panel"
    
#     RAW_PANEL = f"{DATA_RAW_ROOT}/market_base_with_industry_20150101_{Date}.parquet"
#     DATA_TEST_DIR = f"{DATA_ROOT}/model_ready_panel_selected_fin_ind_pv_adj_masks_rebuild_20150105_{Date}.parquet"
#     DATA_DIR = DATA_TEST_DIR
    
#     # TRADE_POOL_DIR = f"{DATA_RAW_ROOT}/riva_cn_tech_trade_pools_202601_202608".strip()
#     TRADE_POOL_DIR = f"/home/eric/workspace/data/riva_cn_tech_trade_pools_202601_202608".strip()

#     # 📂 工程路径 (可写)
#     ROOT = Path(__file__).resolve().parents[1]
#     ROOT_PRT2 = Path(__file__).resolve().parents[2] / "backtest_engine_output"
#     OUT_DIR = ROOT_PRT2 / "output"
#     FIG_DIR = OUT_DIR / "figures"
#     PROC_DIR = ROOT / "processed_data/backtest"
#     LOG_DIR  = ROOT_PRT2 / "log"
#     MODEL_DIR = ROOT_PRT2 / "saved_models"

#     # 🚫 数据过滤选项
#     EXCLUDE_BJ = True  

#     # ⚙️ 回测结构参数
#     WARMUP_DAYS    = 0      
#     REBALANCE_DAYS = 5
#     REBALANCE_WEEKDAY = 4   
#     TOP_K          = 25
#     INITIAL_CAPITAL = 10_000_000.0
#     COMMISSION_RATE = 0.0002

#     # 💰 交易摩擦
#     SLIPPAGE = 0.000
#     STAMP_TAX_RATE = 0.0000

#     # 🔑 新增：聚类模型与标签路径
#     CLUSTER_MODEL_PKL = MODEL_DIR / "cluster_model.pkl"
#     MARKET_REGIME_LABELS = OUT_DIR / "market_regime_labels.json"

#     # 🤖 模型配置
#     MODELS = [
#         'ElasticNet', 
#         'XGB-22', 'XGB-23', 'XGB-24', 
#         'LGBM-22', 'LGBM-23', 'LGBM-24', 
#         'XGB-low', 'XGB-mid', 'XGB-high',
#         'LGBM-low', 'LGBM-mid', 'LGBM-high',
#         'XGB-wLow', 'XGB-wMid', 'XGB-wHigh',  # 🔑 新增周度波动率模型
#         'LGBM-wLow', 'LGBM-wMid', 'LGBM-wHigh', # 🔑 新增周度波动率模型
#         'OptSharpe', 'DynamicSwitch', 'DynamicSwitch2',
#         'DynamicSwitch_IR', 'DynamicSwitch2_IR',  
#         'DynamicSwitch3', # 🔑 新增
#         'DynamicSwitch4',  # 🔑 新增
#         'ClusterRegime',  # 🔑 新增
#         'SensitiveSwitch', 
#         'BuyAndHoldAll'
#     ]     

#     # 🔑 新增：DynamicSwitch3 (Optimal Sharpe 模型加权) 路由配置
#     DYNAMIC_SWITCH3_BASE_MODELS = []  # 将在 run_backtest.py 中动态赋值为所有树模型
#     DYNAMIC_SWITCH3_WINDOW = 10       # 计算最优权重的滚动窗口大小 (天数)
#     # 🔑 新增：DynamicSwitch4 (个股级 Optimal Sharpe 模型加权) 路由配置
#     DYNAMIC_SWITCH4_BASE_MODELS = []  # 将在 run_backtest.py 中动态赋值为所有树模型
#     DYNAMIC_SWITCH4_WINDOW = 10       # 计算个股最优权重的滚动窗口大小 (自然日)
    
#     # 🔄 动态切换策略配置
#     DYNAMIC_SWITCH_INIT_MODEL = 'LGBM-24'
#     DYNAMIC_SWITCH_BASE_MODELS = [
#         'ElasticNet', 'OptSharpe', 
#         'XGB-22', 'XGB-23', 'XGB-24', 
#         'LGBM-22', 'LGBM-23', 'LGBM-24',
#         'XGB-low', 'XGB-mid', 'XGB-high',
#         'LGBM-low', 'LGBM-mid', 'LGBM-high',
#         'XGB-wLow', 'XGB-wMid', 'XGB-wHigh',  # 🔑 新增周度波动率模型
#         'LGBM-wLow', 'LGBM-wMid', 'LGBM-wHigh', # 🔑 新增周度波动率模型
#         'XGB-22_ablation', 'XGB-23_ablation', 'XGB-24_ablation',
#         'LGBM-22_ablation', 'LGBM-23_ablation', 'LGBM-24_ablation',
#         'LGBM_c1', 'LGBM_c2', 'LGBM_c3',  # 🔑 新增
#         'LGBM_c4', 'LGBM_c5', 'LGBM_c6',  # 🔑 新增
#         'XGB_c1', 'XGB_c2', 'XGB_c3'      # 🔑 新增
#         'XGB_c4', 'XGB_c5', 'XGB_c6'      # 🔑 新增
#     ]  
#     DYNAMIC_SWITCH_B = 1.00  

#     # 🔑 新增：DynamicSwitch2 (IC 加权集成) 路由配置
#     DYNAMIC_SWITCH2_BASE_MODELS = DYNAMIC_SWITCH_BASE_MODELS
#     DYNAMIC_SWITCH2_WINDOW = 10  # 计算 IC 权重的滚动窗口大小

#     # 🔑 新增：DynamicSwitch_IR (基于 IR 切换) 路由配置
#     DYNAMIC_SWITCH_IR_INIT_MODEL = 'LGBM-24'
#     DYNAMIC_SWITCH_IR_BASE_MODELS = DYNAMIC_SWITCH_BASE_MODELS
#     DYNAMIC_SWITCH_IR_B = 1.00  # 切换阈值倍数 (新模型 IR > B * 当前模型 IR 时切换)

#     # 🔑 新增：DynamicSwitch2_IR (基于 IR 加权) 路由配置
#     DYNAMIC_SWITCH2_IR_BASE_MODELS = DYNAMIC_SWITCH_BASE_MODELS

#     # 🔑 SensitiveSwitch 路由配置
#     SENSITIVE_SWITCH_INIT_MODEL = 'LGBM-24'
#     SENSITIVE_SWITCH_BASE_MODELS = DYNAMIC_SWITCH_BASE_MODELS
#     SENSITIVE_SWITCH_THRESHOLD = 1.0  

#     # 🔍 SHAP 分析配置
#     SHAP_SAMPLE_SIZE = 500
    
#     # 🔑 消融实验配置 (移除硬编码特征，改为依赖 JSON 文件)
#     SHAP_ABLATION = True
#     ABLATION_FEATURE_JSON = OUT_DIR / "ablation_feature.json"  # 由 shap_result_visual.py 生成