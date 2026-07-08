以下为您生成的详细 Markdown 说明文档。该文档全面整合了项目的架构、核心优化逻辑（严格防泄漏、增量调仓、胜率追踪、SHAP 分析）以及运行规范，可直接作为团队协作、代码维护或项目交付的基准文档。

---

# 📊 A股量化多因子回测系统 (A-Share Backtest Engine)

## 1. 项目概述
本项目是一套面向 A 股市场的事件驱动型多因子量化回测引擎。系统支持传统机器学习模型（ElasticNet）与树模型（XGBoost, LightGBM）的信号生成，并集成基于历史收益的均值方差优化策略（OptSharpe）及买入持有基线（BuyAndHoldAll）。

引擎严格遵循 **`T日特征预测 → T日收盘调仓 → T+1日收益实现`** 的因果链条，内置 A 股真实交易摩擦模型（整手限制、佣金），并提供完整的因子分析、回测执行、绩效评估、可解释性分析（SHAP）与可视化流水线。

---

## 2. 核心特性与优化亮点
1. **物理级数据隔离（防前视偏差）**：
   - 数据加载时强制注入 `DATA_SOURCE` (`'train'` / `'test'`) 标签。
   - 模型训练阶段强制增加 `DATA_SOURCE == 'train'` 过滤条件，杜绝测试集特征/标签泄漏。
   - **模型冻结机制**：2025-01-01 回测开始后，模型使用全量历史训练集进行最后一次训练并永久冻结，样本外测试期间绝对不进行任何重训。
2. **极简增量调仓逻辑（最小化换手率）**：
   - 摒弃传统的全量权重再平衡。采用集合运算：`卖出集合 = 当前持仓 - 新Top50`，`买入集合 = 新Top50 - 当前持仓`。
   - 留在 Top50 内的标的**零摩擦持有**，释放的现金与新进标的等权分配，大幅降低不必要的交易成本。
3. **精准持仓成本与胜率追踪**：
   - 引入 `cost_basis` 动态维护每只股票的**加权平均买入成本**。
   - 在平仓（SELL）时，对比卖出价与成本价，精确统计**单笔交易胜率 (Win Rate)**。
4. **模型可解释性分析 (SHAP)**：
   - 针对 XGBoost/LightGBM 自动生成 SHAP 蜂群图与 Top 20 因子重要性 JSON。
   - SHAP 背景数据严格限定在 2025-01-01 之前的纯训练集中随机采样，兼顾计算效率与严谨性。
5. **结构化绩效输出**：
   - 终端打印对齐的综合绩效表格，并自动导出机器可读的 `backtest_summary.json`，包含收益、回撤、夏普及详细交易统计。

---

## 3. 目录结构与职责划分
```text
/data/cye_temp/workspace/backtest_engine/
├── config/
│   └── Config.py              # 全局配置中心（路径、回测参数、交易成本、模型列表）
├── util/
│   ├── data_loader.py         # 数据加载、特征动态提取、截面 IC/IR 评估
│   └── metrics.py             # 绩效计算、归一化 PnL 绘图、回撤与滚动夏普可视化
├── model/
│   └── baseline_model.py      # 模型注册表（训练、预测、OptSharpe 权重优化）
├── src/
│   ├── backtest_engine.py     # 回测核心调度器（日频循环、模型冻结、调仓路由）
│   └── backtest_core.py       # 投资组合管理器（持仓、现金、增量调仓、成本追踪）
├── script/
│   ├── run_backtest.py        # 回测主入口脚本（集成 SHAP 与 JSON 汇总）
│   └── analysis.py            # 独立因子分析脚本（按时间段导出统计与 IC 报告）
├── output/                    # 📂 产出目录（自动创建）
│   ├── figures/               # 策略对比图 (PnL, Drawdown, Rolling Sharpe) + SHAP 图表
│   ├── backtest_summary.json  # 综合绩效与交易胜率汇总 (JSON)
│   └── factor_ic_ir.json      # 全因子 IC/IR 评估报告
├── log/                       # 📂 交易明细日志（按模型分 CSV）
└── processed_data/            # 📂 允许写入的中间数据缓存
```

---

## 4. 数据规范与流向
| 数据类别 | 路径 | 权限 | 关键字段说明 |
| :--- | :--- | :---: | :--- |
| **训练集** | `/data/train_data/model_ready_panel_..._{YEAR}.parquet` | 🔒 只读 | YEAR∈[2016,2023]，含 `_MKT_Z`/`_IND_Z` 标准化因子、`FEATURE_MASK`、`BUY_MASK` |
| **测试集** | `/data/test_data/model_ready_panel_..._test_*.parquet` | 🔒 只读 | YEAR∈[2024,2026]，结构同训练集，用于样本外验证 |
| **原始行情** | `/data/data_process/4.15_revision/raw_daily_panel.parquet` | 🔒 只读 | 用于计算真实日收益率 `label_i` (`S_DQ_ADJCLOSE.pct_change().shift(-i)`) |
| **加工数据** | `/data/cye_temp/workspace/backtest_engine/processed_data/` | ✏️ 可读写 | 缓存对齐后的面板数据、特征掩码、回测中间状态 |

> ⚠️ **数据对齐原则**：所有因子与标签均基于 `TRADE_DT` 与 `S_INFO_WINDCODE` 联合主键进行 `left join`。缺失的 `label_i` 默认保留 `NaN`，实际训练中通过 `dropna` 过滤无效样本，防止前视偏差。

---

## 5. 核心架构与执行流程
回测主循环 (`BacktestEngine.run`) 按日遍历，执行以下状态机：

```mermaid
graph TD
    A[加载年度 Parquet 并拼接] --> B[计算 T+i 日真实收益率 label_i]
    B --> C[动态提取合法特征列]
    C --> D{回测主循环: 按 TRADE_DT 遍历}
    
    D --> E{当前日期 < 2025-01-01 ?}
    E -- 是 (训练期) --> F[按 RETRAIN_DAYS 周期滚动训练<br/>严格过滤 DATA_SOURCE == 'train']
    E -- 否 (测试期) --> G{是否首次进入测试期?}
    G -- 是 --> H[使用全量历史训练集进行最终训练<br/>设置 model_frozen = True]
    G -- 否 --> I[跳过训练，模型永久冻结]
    
    F --> J{到达 REBALANCE_DAYS ?}
    I --> J
    H --> J
    
    J -- 是 --> K[模型预测 / OptSharpe 计算<br/>筛选 Top K 标的]
    K --> L[PortfolioManager.rebalance<br/>1. 卖出跌出标的<br/>2. 现金汇总等权买入新进标的]
    J -- 否 --> M[跳过调仓，保持持仓]
    
    L --> N[update_daily: 计算当日净值 NAV]
    M --> N
    N --> O[更新收益历史 (保留近80天)]
    O --> P{遍历结束?}
    P -- 否 --> D
    P -- 是 --> Q[生成交易日志 + 绩效图表 + SHAP 分析 + JSON 汇总]
```

---

## 6. 模块详细设计

### 🔹 `config/Config.py`
集中管理路径、回测超参、交易成本与模型白名单。
- `WARMUP_DAYS = 2200`: 预热期天数，期间仅记录净值，不触发调仓。
- `RETRAIN_DAYS = 1800000`: 模型重训周期（当前设为极大值，实际由 2025-01-01 冻结逻辑接管）。
- `REBALANCE_DAYS = 10`: 仓位调整周期。
- `TOP_K = 50`: 每次调仓入选股票数。
- `COMMISSION_RATE = 0.0002`: 双边交易佣金。

### 🔹 `util/data_loader.py`
- `load_panel_data`: 按年读取 parquet，解析日期，纵向拼接，并**强制注入 `DATA_SOURCE` 标签**。
- `extract_valid_features`: 依据字典规则，使用正则表达式动态提取合法特征（保留 `^Breadth_[^_]+$` 及 `_MKT_Z`/`_IND_Z` 结尾列，排除所有含 `MASK`, `FLAG`, `RATE` 的列）。
- `compute_real_returns`: 计算 `label_i`，对价格进行防御性 `ffill`，并与因子面板 merge。

### 🔹 `src/backtest_engine.py`
回测调度中枢。
- `run()`: 主循环。包含 `model_frozen` 状态机，确保 2025-01-01 后模型绝对静止。
- `analyze_shap()`: 提取 2025-01-01 之前的纯训练集数据（默认采样 500 条），使用 `shap.TreeExplainer` 计算 XGBoost/LightGBM 的特征重要性，输出蜂群图与 JSON。

### 🔹 `src/backtest_core.py`
投资组合状态机 (`PortfolioManager`)。
- `update_daily`: 实时聚合持仓市值与现金，停牌股使用 `last_known_prices` 估值。
- `rebalance`: **增量调仓核心**。通过集合运算 `sell_codes = current - target` 和 `buy_codes = target - current`，先执行卖出释放现金，再将可用现金等权分配给 `buy_codes`（向下取整至 100 股）。
- `cost_basis` 追踪：买入时动态更新加权平均成本，卖出时对比成本价以统计 `wins` / `losses`。

### 🔹 `script/run_backtest.py`
端到端流水线入口。完成数据加载、引擎初始化、回测执行，并在末尾触发 `metrics.evaluate_and_plot`、`engine.analyze_shap`，最终打印并保存 `backtest_summary.json`。

---

## 7. 运行指南

### 7.1 环境准备
确保已安装核心依赖，特别是用于可解释性分析的 `shap` 库：
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib shap
```

### 7.2 执行回测
在项目根目录下运行主脚本：
```bash
cd /data/cye_temp/workspace/backtest_engine
python script/run_backtest.py
```

### 7.3 预期控制台输出
```text
📦 加载训练面板...
📊 数据加载完成 | 形状: (1245890, 460) | 时间范围: 2016-01-04 - 2026-02-21
📊 特征提取完成 | 数量: 452 | 包含模型: ['ElasticNet', 'XGBoost', 'LightGBM', 'OptSharpe', 'BuyAndHoldAll']
🚀 启动回测 | 交易日: 2450 | 模型: [...]
🔧 ML模型滚动重训完成 | 纯训练集样本: 45210 | 日期: 2020-03-15
🔒 触发样本外测试冻结机制 | 日期: 2025-01-02
🔧 样本外模型已永久冻结 | 最终纯训练集样本: 850000
📊 进度: 2025-06-01 | 现金(EN): 8,452,300
...
==========================================================================================
🏆 回测综合绩效评估 (Out-of-Sample / Test Set >= 2025-01-01)
==========================================================================================
Model           | Total PnL  | Annual Ret  | Max DD     | Sharpe   | Win Rate   | Closed Trades
------------------------------------------------------------------------------------------
ElasticNet      |    15.32% |     18.45% |    -8.21% |   1.452 |     58.33% |           120
XGBoost         |    22.15% |     27.10% |    -6.54% |   1.821 |     62.50% |           104
LightGBM        |    19.80% |     24.05% |    -7.12% |   1.655 |     60.19% |           108
OptSharpe       |     8.45% |     10.12% |   -11.34% |   0.850 |     51.22% |            82
BuyAndHoldAll   |     5.10% |      6.15% |   -15.20% |   0.420 |      0.00% |             0
==========================================================================================
✅ 综合绩效汇总已保存至 JSON: /data/cye_temp/workspace/backtest_engine/output/backtest_summary.json
🔍 启动 SHAP 因子可解释性分析...
  背景数据采样完成: 500 行 × 452 特征
  正在计算 XGBoost 的 SHAP 值...
  ✅ XGBoost SHAP 分析完成！
✅ 全部流程完成！
```

---

## 8. 输出与产物说明
| 路径 | 内容 | 用途 |
| :--- | :--- | :--- |
| `output/figures/pnl_combined.png` | 多策略归一化累计收益曲线 (2025-01-01 起) | 直观对比样本外收益表现 |
| `output/figures/drawdown_combined.png` | 多策略历史最大回撤对比 | 评估策略下行风险 |
| `output/figures/rolling_sharpe_combined.png` | 30日滚动夏普比率时序 | 观察策略稳定性 |
| `output/figures/shap_summary_*.png` | SHAP 蜂群图 (Top 20 特征) | 验证模型逻辑，排查过拟合/泄漏 |
| `output/figures/shap_importance_*.json` | SHAP 平均绝对贡献值排名 | 量化报告因子重要性依据 |
| `output/backtest_summary.json` | 综合绩效与交易胜率结构化数据 | 自动化报告生成、超参搜索记录 |
| `log/trades_{MODEL}.csv` | 完整买卖记录（日期、代码、方向、股数、价格、费用） | 交易明细审计、滑点分析 |

---

## 9. 风险控制与工程规范
1. **严格防前视偏差 (Look-ahead Bias)**：
   - 训练集时间窗口硬编码截止至 `T` 日，且强制校验 `DATA_SOURCE == 'train'`。
   - `returns_history` 仅记录已实现价格变动，不参与当日信号生成。
2. **数值稳定性保障**：
   - OptSharpe 协方差矩阵加入对角收缩正则项 `1e-4 * trace / N`，防止高维矩阵奇异。
   - 所有除法操作附加 `1e-8` 保护，防止零波动率或零持仓导致 `NaN` 或 `ZeroDivisionError`。
   - 价格 `< 1e-6` 的标的被视为停牌或异常，自动跳过交易。
3. **内存与性能优化**：
   - `returns_history` 限制最大长度 80 天，避免全量历史累积导致 OOM。
   - SHAP 分析采用随机下采样 (`sample_size=500`)，在保持特征重要性排序稳定性的同时，将计算时间控制在秒级。
4. **数据只读原则**：
   - 所有 `/data/` 下的原始 parquet 文件均为只读挂载，严禁在代码中修改。加工数据请统一存放至 `processed_data/` 目录。

---

## 10. 扩展与优化建议 (Roadmap)
| 模块 | 当前状态 | 建议优化方向 |
| :--- | :--- | :--- |
| `backtest_core.py` | 增量调仓，新进标的等权分配 | 升级为 **全量权重再平衡** (Sell All → Rebuy TopK)，彻底消除长期持仓的权重漂移 |
| `data_loader.py` | 价格缺失使用 `ffill` | 引入 `LABEL_VALID` 掩码列，训练前严格 `dropna`，保留原始标签分布，避免填充扭曲 |
| `backtest_engine.py` | 固定 `label_i` 训练 | 切换至 **滚动截面标准化** 的 `label_i`，或使用 Rank IC 作为优化目标，适应市场风格切换 |
| 特征工程 | 全局 Z-Score 标准化 | 改为 **滚动截面标准化**（如过去 60 天均值/标准差），防止未来分布泄漏 |

> 📌 **维护提示**：新增因子需同步更新 `util/data_loader.py` 中的 `extract_valid_features` 正则匹配逻辑，并确保其已包含在 `selected_training_fields_dictionary_cn.md` 的白名单中。