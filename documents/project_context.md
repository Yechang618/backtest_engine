以下为您定制的 **项目上下文与架构说明文档（Markdown格式）**。您可以直接将此文件保存为 `PROJECT_CONTEXT.md` 并上传至 Gemini、Claude 或其他 AI 平台。该文档详细梳理了项目的数据流、模块职责、核心设计约束以及历史踩坑记录，能确保其他 AI 在辅助您开发时具备全局视角且不会破坏现有架构。

***

```markdown
# A股多因子量化回测与机器学习路由系统 - 项目上下文说明

## 1. 项目概述
本项目是一个端到端的 A 股量化投资研究框架，涵盖了从**面板数据预处理**、**多维度市场状态划分（波动率/宏观聚类）**、**多模型训练（全量/分位数/聚类/消融）**、**SHAP 因子归因**，到**动态路由回测引擎**的完整闭环。
系统的核心亮点在于**统一路由架构**与**无未来函数设计**，支持数十个异构模型（XGBoost, LightGBM, ElasticNet, 动态切换策略, 聚类专家模型等）在同一引擎内公平竞技与特征隔离。

---

## 2. 目录结构
```text
backtest_engine/
├── config/
│   └── Config.py                 # 全局配置（路径、回测参数、模型列表、路由策略超参）
├── src/
│   ├── backtest_core.py          # 回测底层核心（PortfolioManager：交易执行、成本扣除、胜率统计）
│   └── backtest_engine.py        # 回测引擎（特征路由、Residual OLS、动态切换逻辑、误差延迟结算）
├── util/
│   ├── data_loader.py            # 数据加载与预处理（计算T+i收益率、衍生因子、特征掩码提取）
│   └── metrics.py                # 绩效评估与绘图（PnL、回撤、滚动Sharpe、Top10曲线过滤）
├── script/
│   ├── train_models.py           # 训练全量基础模型（XGB/LGBM-22/23/24, ElasticNet）
│   ├── train_models_percentile.py# 训练波动率分位数模型（月度/周度 low/mid/high/wLow/wMid/wHigh）
│   ├── train_models_cluster.py   # 训练宏观聚类分位数模型（KMeans聚类后的 c1/c2/c3）
│   ├── train_models_ablation.py  # 训练消融实验模型（基于SHAP筛选的特征子集）
│   ├── cluster_market_regimes.py # 宏观市场状态聚类（计算波动率、换手率、成交额、市值并KMeans）
│   ├── shap_analysis.py          # 季度级SHAP与IC/IR分析（严格按训练子集过滤数据）
│   ├── shap_result_visual.py     # SHAP结果可视化与消融特征提取（生成ablation_feature.json）
│   ├── calculate_monthly_volatility.py # 计算月度波动率并保存JSON
│   ├── visualize_volatility.py   # 可视化波动率并计算分位数阈值（P33/P67）
│   ├── run_backtest.py           # 回测主控脚本（统一加载模型、动态构建路由池、执行回测）
│   └── analyze_backtest_results.py # 回测结果后处理（读取JSON输出汇总表格）
├── output/                       # 输出目录（图表、JSON、Parquet、消融特征配置）
├── saved_models/                 # 模型存储目录（pkl文件）
└── log/                          # 交易明细日志目录
```

---

## 3. 核心工作流 (Pipeline)

### 3.1 市场状态划分 (Regime Definition)
1. **波动率分位数**：`calculate_monthly_volatility.py` -> `visualize_volatility.py`。计算全市场月度/周度收益率标准差，基于**训练期（<=2024-12）** 计算 **P33/P67** 阈值，生成 `monthly_volatility.json` 和 `volatility_percentiles.json`。
2. **宏观聚类**：`cluster_market_regimes.py`。读取 `RAW_PANEL` 计算4维宏观特征（波动率、换手率、成交额环比、市值环比），使用 KMeans 聚类，生成 `cluster_model.pkl` 和 `market_regime_labels.json`。

### 3.2 模型训练体系 (Model Training)
1. **基础模型**：`train_models.py` (生成 `sklearn_models.pkl`)。
2. **分位数模型**：`train_models_percentile.py` (生成 `percentile_sklearn_models.pkl`)。支持月度与周度双轨划分。
3. **聚类模型**：`train_models_cluster.py` (生成 `cluster_sklearn_models.pkl`)。根据 JSON 标签过滤数据子集训练专家模型。
4. **消融模型**：`train_models_ablation.py` (生成 `ablation_sklearn_models.pkl`)。读取 SHAP 提取的特征字典，按对应子集训练。

### 3.3 因子归因与消融 (SHAP & Ablation)
1. `shap_analysis.py`：根据模型后缀（如 `-low`, `-wMid`, `-c1`）**自动在内存中过滤出与训练时完全一致的数据子集**，进行季度级 SHAP 计算，防止数据分布偏移导致归因失效。
2. `shap_result_visual.py`：提取 SHAP 值 > 0.0001 的特征，增量更新保存至 `output/ablation_feature.json`。

### 3.4 统一回测与路由 (Unified Backtesting)
1. `run_backtest.py`：统一加载所有 `.pkl` 模型，**动态提取所有树模型名称**注入 `cfg.MODELS` 和 `DYNAMIC_SWITCH_BASE_MODELS`。
2. `backtest_engine.py`：执行每日循环，计算 Rank IC 与 Residual IC，触发动态切换逻辑，执行无未来函数的误差结算。

---

## 4. 关键架构设计与防坑约束 (⚠️ 核心上下文)

### 4.1 特征路由与隔离机制 (Feature Routing)
- **痛点**：数十个模型的特征维度各不相同（全量240+，消融5~30，分位数特定子集）。
- **解决方案**：`backtest_engine.py` 中维护了 `self.aligned_features` 字典。在 `_check_feature_alignment` 阶段，为每个模型独立缓存对齐后的特征列，**绝不覆盖全局的 `self.feature_cols`**。
- **路由逻辑**：`_get_features_for_model(name)` 优先查询 `cfg.FEATURE_SELECTED` (消融字典)，若无则回退全量特征。

### 4.2 时间窗口与防未来函数约束
- **训练期上限**：严格限制为 `2024-12-31`。
- **验证/测试期起点**：`2025-01-01` 至 `2026-03-31`。
- **阈值计算防泄漏**：所有分位数（P33/P67）和 KMeans 聚类，**必须且只能**使用 `<= 2024-12-31` 的数据进行 `fit` 或计算，验证集仅做 `transform` 或 `map`。
- **误差延迟结算**：`backtest_engine.py` 中的 MSE/MAE 不在 T 日计算，而是缓存 `prediction_cache`，在 T+i 日真实收益实现后，记录在结算日的 `mse_results` 中。

### 4.3 动态路由策略的数学稳定性
- **IR 计算防爆炸**：`IR = Mean(IC) / max(Std(IC), 1e-3)`。强制设定标准差下限，防止初期 IC 不变时 IR 飙升至百万级别。
- **加权集成防 NaN**：`DynamicSwitch2_IR` 中，若存在负 IR，采用**平移归一化** `shifted = v - min_ir + 1e-5`，确保所有权重为正且和为 1，杜绝 `inf / inf = nan` 的崩溃。
- **Residual OLS 防奇异**：剥离风格/行业因子时，加入 `1e-8` 的正则化项 `np.linalg.solve(XtX + 1e-8 * I, Xty)`，并设定样本数 < 100 或特征数 >= 样本数 - 5 时直接返回 `-1e6` 的熔断机制。

### 4.4 数据源严格隔离
- **因子面板 (`DATA_DIR` / `DATA_TEST_DIR`)**：仅包含标准化后的因子（如 `XXX_MKT_Z`）和复权价格 `S_DQ_ADJCLOSE`。**不包含**原始量价数据。
- **原始面板 (`RAW_PANEL`)**：包含 `S_DQ_CLOSE`, `S_DQ_VOLUME`, `S_DQ_AMOUNT`, `S_DQ_CAPITAL` 等。**计算宏观特征（如聚类、月度波动率）时，必须且只能读取 `RAW_PANEL`**，否则会导致全 `NaN` 崩溃。

### 4.5 区间划分互斥性
- 波动率划分必须使用严格的互斥逻辑：`vol <= P33` (low), `P33 < vol <= P67` (mid), `vol > P67` (high)。
- **禁止**使用类似 `p40 <= vol <= p70` 的交叉判断，否则会导致同一月份数据同时混入 mid 和 high 的训练集，破坏模型学习。

---

## 5. 核心配置说明 (`Config.py`)

```python
# 路由策略超参
DYNAMIC_SWITCH_B = 1.00          # 赢者通吃阈值倍数 (新IC > B * 当前IC 才切换)
DYNAMIC_SWITCH2_WINDOW = 10      # 加权集成的滚动窗口
SENSITIVE_SWITCH_THRESHOLD = 1.0 # Residual 切换阈值

# 交易与回测参数
REBALANCE_DAYS = 5               # 调仓周期
REBALANCE_WEEKDAY = 4            # 按周调仓目标日 (4=周五，支持节假日自动顺延)
TOP_K = 25                       # 选股数量
COMMISSION_RATE = 0.0002         # 手续费

# 路径配置
RAW_PANEL = ".../market_base_...parquet"       # 原始量价数据
DATA_TEST_DIR = ".../model_ready_panel_...parquet" # 因子面板数据
ABLATION_FEATURE_JSON = OUT_DIR / "ablation_feature.json"
```

---

## 6. 给 AI 助手的修改建议指南
如果您（AI）需要为本项目新增功能或修复 Bug，请遵循以下原则：
1. **新增模型类型**：只需在 `train_models_xxx.py` 中训练并保存为 `.pkl`，`run_backtest.py` 会自动识别带有 `XGB` 或 `LGBM` 关键字的模型并将其加入路由池，**无需手动修改 `Config.py` 的 `MODELS` 列表**。
2. **处理 Pandas 链式调用**：避免使用 `inplace=True` 后继续链式调用（如 `df.replace(..., inplace=True).dropna()`），这会返回 `None` 导致崩溃。请改为 `df = df.replace(...).dropna()`。
3. **处理 Matplotlib 兼容性**：项目运行环境可能为 Matplotlib >= 3.9，**禁止使用** `plt.cm.get_cmap()`，请替换为 `mpl.colormaps['viridis'].resampled(N)`。
4. **股票池空集处理**：在 `backtest_engine.py` 中，若股票池为空，**禁止**使用 `tradable = pd.DataFrame()`（会丢失列信息导致 `KeyError: label_5`），必须使用 `tradable = tradable[tradable.index.isin(allowed_codes)]` 以保留 0 行但保留列结构。
```

***

### 💡 使用建议：
1. 将此 Markdown 内容保存为 `PROJECT_CONTEXT.md`。
2. 在与其他 AI（如 Gemini 1.5 Pro / Claude 3.5 Sonnet）对话时，将其作为 **System Prompt** 或 **Context File** 上传。
3. 当您需要 AI 帮您写新脚本（例如新增一种基于宏观因子的路由策略）时，AI 会自动遵循“特征隔离”、“防未来函数”、“数据源隔离”等核心约束，写出与现有架构完美兼容的代码，避免反复调试维度报错或数据泄漏问题。