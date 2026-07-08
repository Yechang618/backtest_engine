以下是基于当前代码库深度优化后的 `project.md` 项目说明文档。文档已按照量化系统工程标准进行结构化重构，补充了数据流向、模块接口、核心算法逻辑、防泄漏设计及运行规范，可直接作为团队协作与代码维护的基准文档。

---

# 📊 A股量化回测系统 (A-Share Backtest Engine)

## 1. 项目概述
本项目是一套面向 A 股市场的**事件驱动型多因子量化回测引擎**。系统支持传统机器学习模型（ElasticNet）与树模型（XGBoost、LightGBM）的信号生成，并集成基于历史收益的均值方差优化策略（OptSharpe）。引擎严格遵循 `T日特征预测 → T日收盘调仓 → T+1日收益实现` 的因果链条，内置 A 股真实交易摩擦模型，提供完整的因子分析、回测执行、绩效评估与可视化流水线。

---

## 2. 目录结构与职责划分
```text
/data/cye_temp/workspace/backtest_engine/
├── config/
│   └── Config.py              # 全局配置中心（路径、回测参数、交易成本、模型列表）
├── util/
│   ├── data_loader.py         # 数据加载、收益率计算、截面 IC/IR 评估
│   └── metrics.py             # 绩效计算、归一化 PnL 绘图、回撤与滚动夏普可视化
├── model/
│   └── baseline_model.py      # 模型注册表（训练、预测、OptSharpe 权重优化）
├── src/
│   ├── backtest_engine.py     # 回测核心调度器（日频循环、重训触发、调仓路由）
│   └── backtest_core.py       # 投资组合管理器（持仓、现金、调仓执行、交易记录）
├── script/
│   ├── run_backtest.py        # 回测主入口脚本
│   └── analysis.py            # 独立因子分析脚本（按时间段导出统计与 IC 报告）
├── output/                    # 📂 产出目录（自动创建）
│   ├── figures/               # 3 张策略对比图（PnL、Drawdown、Rolling Sharpe）
│   ├── factor_ic_ir.json      # 全因子 IC/IR 评估报告
│   └── res_factor_stat.json   # 因子统计与双标签 IC 报告（analysis.py 输出）
├── log/                       # 📂 交易明细日志（按模型分 CSV）
└── processed_data/            # 📂 允许写入的中间数据缓存
```

---

## 3. 数据规范与流向

| 数据类别 | 路径 | 权限 | 关键字段说明 |
|:---|:---|:---|:---|
| **训练集** | `/data/train_data/model_ready_panel_selected_plus_ohlc_train_{YEAR}.parquet` | 🔒 只读 | `YEAR∈[2016,2023]`，含 `_MKT_Z`/`_IND_Z` 标准化因子、`FWD_RET_5D_Z_P01_P99`、掩码列 |
| **测试集** | `/data/train_data/model_ready_panel_selected_plus_ohlc_test_{YEAR}.parquet` | 🔒 只读 | `YEAR∈[2024,2026]`，结构同训练集，用于样本外验证 |
| **原始行情** | `/data/data_process/4.15_revision/raw_daily_panel.parquet` | 🔒 只读 | 用于计算真实日收益率 `label_1` (`S_DQ_ADJCLOSE.pct_change().shift(-1)`) |
| **加工数据** | `/data/cye_temp/processed_data/backtest/` | ✏️ 可读写 | 缓存对齐后的面板数据、特征掩码、回测中间状态 |

> ⚠️ **数据对齐原则**：所有因子与标签均基于 `TRADE_DT` 与 `S_INFO_WINDCODE` 联合主键进行 `left join`。缺失的 `label_1` 默认填充 `0.0`，实际训练中通过 `dropna` 过滤无效样本。

---

## 4. 核心架构与执行流程

```mermaid
graph LR
A[加载年度Parquet] --> B[计算T+1日收益率label_1]
B --> C[特征过滤与掩码应用]
C --> D{回测主循环 run()}
D --> E[预热期: 仅记录现金净值]
D --> F[到达重训周期? → 取[T-5, T-5-MAX_LOOKBACK]训练防泄漏]
D --> G[到达调仓周期? → 模型预测/优化 → 筛选Top50]
D --> H[组合管理器: 卖出掉出标的 → 等权买入新进标的]
D --> I[更新当日净值(NAV)与收益历史]
I --> J[循环结束 → 生成交易日志]
J --> K[metrics模块: 计算年化/夏普/回撤 → 输出3张对比图]
```

---

## 5. 模块详细设计

### 🔹 `config/Config.py`
- **职责**：集中管理路径、回测超参、交易成本与模型白名单。
- **关键参数**：
  - `WARMUP_DAYS=180`：模型预热期，期间不调仓、不重训。
  - `RETRAIN_DAYS=120`：模型重训触发间隔。
  - `REBALANCE_DAYS=5`：仓位调整周期（每周调仓）。
  - `COMMISSION_RATE=0.0002`：双边交易佣金。
  - `MODELS`：支持的策略列表，引擎按此列表并行维护独立组合。

### 🔹 `util/data_loader.py`
| 函数 | 输入 | 输出 | 核心逻辑 |
|:---|:---|:---|:---|
| `load_panel_data()` | `data_root`, `years` | `pd.DataFrame` | 按年读取 parquet，解析日期，纵向拼接并排序 |
| `compute_real_returns()` | `raw_panel_path`, `panel` | `DataFrame` | 计算 `label_1`，与因子面板 merge，缺失值防御性填充 |
| `compute_ic_ir()` | `factor_cols`, `label_col`, `df` | `Dict` | 按交易日分组计算截面 Spearman 相关系数，输出均值、IR、正 IC 比例 |

### 🔹 `model/baseline_model.py`
- **`ModelRegistry` 类**：统一管理多模型生命周期。
  - `train_models()`：自动过滤含 NaN 的行，样本量 `<500` 时跳过训练，防止欠拟合。
  - `predict_rank()`：对单日因子进行推理，返回预测值 Series 供排序。
  - `_calc_opt_sharpe_weights()`：基于过去 30~120 天收益构建协方差矩阵，加入对角正则化 `1e-6` 防奇异，求解长-only 权重，退化时切换为均值方差比启发式分配。

### 🔹 `src/backtest_engine.py`
- **`BacktestEngine` 类**：回测调度中枢。
  - `run()` 主循环按日遍历：
    1. **收益历史更新**：记录 `label_1`（保留最近 80 天，OptSharpe 截取后 30 天）。
    2. **防泄漏训练**：训练窗口严格截止至 `T-5`，避免未来数据混入。
    3. **信号生成**：ML 模型使用当日因子；OptSharpe 使用历史收益序列。
    4. **组合交互**：将 Top50 标的列表传入 `PortfolioManager.rebalance()`。
    5. **净值记录**：每日收盘后计算 `cash + ∑(shares * price)`。

### 🔹 `src/backtest_core.py`
- **`PortfolioManager` 类**：资产状态机。
  - `update_daily()`：实时聚合持仓市值与现金，返回组合净值。
  - `rebalance()`：**增量调仓逻辑**
    - 清仓：遍历当前持仓，卖出不在 Top50 的标的，扣除佣金。
    - 买入：仅对**新进** Top50 标的分配现金，按 `现金 / 新进数量` 等权计算预算，向上取整至 100 股（A 股整手规则），扣除佣金后买入。
  - `save_logs()`：导出结构化交易明细至 `log/`。

### 🔹 `script/run_backtest.py` & `analysis.py`
- **`run_backtest.py`**：端到端流水线入口。完成数据加载、特征提取、引擎初始化、回测执行、日志保存与可视化触发。
- **`analysis.py`**：独立因子诊断工具。支持自定义时间切片，输出因子描述性统计与双标签（1日/5日）IC/IR 报告至 JSON。

### 🔹 `util/metrics.py`
- **`evaluate_and_plot()`**：绩效归因与可视化。
  - 动态合并多模型净值序列，统一归一化至起点 `1.0`。
  - 计算指标：累计收益、年化收益 `((1+cum)^(252/days)-1)`、年化波动率、最大回撤、夏普比率。
  - 输出三图：`pnl_combined.png`、`drawdown_combined.png`、`rolling_sharpe_combined.png`（30日滚动窗口）。
  - 若传入 `ic_ir_dict`，同步保存因子评估报告。

---

## 6. 关键参数配置说明

| 参数 | 默认值 | 作用与调优建议 |
|:---|:---|:---|
| `MIN_LOOKBACK` | 30 | 模型训练最小样本窗口。过小易导致树模型过拟合。 |
| `MAX_LOOKBACK` | 365 | 训练数据回溯上限。建议与因子衰减周期匹配。 |
| `TOP_K` | 50 | 每次调仓入选股票数。过多增加摩擦成本，过少分散度不足。 |
| `SLIPPAGE` / `STAMP_TAX_RATE` | 0.001 / 0.0005 | 真实交易摩擦。回测负收益常因未计入滑点与印花税导致虚高实盘亏损。 |
| `RETRAIN_DAYS` | 120 | 模型重训频率。高频重训易过拟合噪声，低频易滞后市场风格切换。 |

---

## 7. 运行指南与输出规范

### 🚀 执行回测
```bash
cd /data/cye_temp/workspace/backtest_engine
python script/run_backtest.py
```
**控制台日志流**：
```
📦 加载训练面板 (2016-2023)...
✅ 真实收益率与收盘价已对齐 | 有效记录: 1,245,890 | 缺失已填 0.0
📊 有效特征数: 142 | 包含模型: ['ElasticNet', 'XGBoost', 'LightGBM', 'OptSharpe']
🚀 启动回测引擎...
🔧 ML模型重训完成 | 样本: 45,210
📈 买入 000001.SZ | 股数: 200 | 价格: 12.45 | 费用: 0.498
📊 进度: 2020-03-15 | 现金(EN): 8,452,300
...
✅ 回测流程全部完成
```

### 📈 预期输出
| 路径 | 内容 |
|:---|:---|
| `output/figures/pnl_combined.png` | 多策略归一化累计收益曲线 |
| `output/figures/drawdown_combined.png` | 多策略历史最大回撤对比 |
| `output/figures/rolling_sharpe_combined.png` | 30日滚动夏普比率时序 |
| `log/trades_{MODEL}.csv` | 完整买卖记录（日期、代码、方向、股数、价格、费用） |
| `output/factor_ic_ir.json` | 全因子 IC 评估（若启用） |
| `output/res_factor_stat.json` | `analysis.py` 输出的因子统计与双标签 IC 报告 |

---

## 8. 风险控制与工程规范

1. **严格防前视偏差**：
   - 训练集时间窗口硬编码截止至 `T-5`，确保 `label_1`（T→T+1）在预测时未泄漏。
   - `returns_history` 仅记录已实现价格变动，不参与当日信号生成。
2. **交易摩擦真实化**：
   - 买入价叠加滑点，卖出价扣除印花税，双边收取佣金。避免回测收益“纸上富贵”。
3. **数值稳定性保障**：
   - 协方差矩阵加入对角收缩正则项，防止高维矩阵奇异。
   - 所有除法操作附加 `1e-8` 保护，防止零波动率导致 `NaN`。
4. **内存与性能优化**：
   - `returns_history` 限制最大长度 80 天，避免全量历史累积 OOM。
   - 使用 `pd.concat` 预分配索引，避免循环内 `append` 导致性能退化。

---

## 9. 扩展与优化建议（Roadmap）

| 模块 | 当前状态 | 建议优化方向 |
|:---|:---|:---|
| `backtest_core.py` | 仅对新进标的等权分配 | 升级为**全量权重再平衡**（Sell All → Rebuy TopK），消除持仓权重漂移 |
| `data_loader.py` | `fillna(0.0)` 扭曲标签分布 | 引入 `LABEL_VALID` 掩码列，训练前 `dropna`，保留原始分布 |
| `backtest_engine.py` | 使用 `FWD_RET_5D` 训练 | 切换至 `label_1` 训练，匹配日频调仓周期；或改为滚动截面标准化 |
| `metrics.py` | 短周期年化公式放大误差 | 改用 `scipy.stats.sharpe_ratio` 或 `ret.mean()/ret.std()*sqrt(252)` |
| 特征工程 | 全局 Z-Score 标准化 | 改为 **滚动截面标准化**（如过去 60 天均值/标准差），防止未来分布泄漏 |

---
> 📌 **维护提示**：所有数据源路径均为只读挂载，严禁在代码中写入原始 parquet。加工数据请统一存放至 `processed_data/` 目录。新增因子需同步更新 `analysis.py` 的字段过滤逻辑与 `Config.FEATURE_COLS` 动态注入机制。