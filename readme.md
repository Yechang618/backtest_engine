```markdown
# 📈 A-Share ML Quant Backtest Engine

基于滚动窗口机器学习模型（ElasticNet / XGBoost / LightGBM）的A股截面多因子选股与回测系统。严格遵循时间序列防泄漏原则，支持动态仓位管理、整手交易规则与多维绩效评估。

## 📁 项目目录结构
```text
/data/cye_temp/workspace/backtest_engine/
├── script/
│   └── run_backtest.py              # 🚀 主入口：调度数据加载、引擎运行、评估输出
├── src/
│   ├── backtest_core.py             # 💼 组合管理：现金/持仓跟踪、调仓逻辑、整手向上取整、手续费
│   └── backtest_engine.py           # ⚙️ 回测引擎：滚动训练、截面预测排序、策略调度
├── util/
│   ├── data_loader.py               # 📦 数据IO：Parquet分年读取、真实日收益计算、IC/IR统计
│   └── metrics.py                   # 📊 评估模块：PnL/回撤/滚动夏普绘图、JSON指标导出
├── output/                          # 📤 最终输出目录
│   └── figures/                     # 生成的可视化图表 (PNG)
├── processed_data/                  # 🗃️ 允许写入：每日净值明细、中间缓存
├── model/                           # 🤖 允许写入：保存训练后的模型权重（可选）
├── log/                             # 📝 允许写入：运行日志
└── README.md
```

## ⚙️ 核心功能与规则
| 模块 | 实现说明 |
|:---|:---|
| **数据流** | 训练集读取 `2016-2023` 年份面板。严格使用 `FEATURE_MASK==1` 过滤低质量样本。回测收益由原始面板 `S_DQ_ADJCLOSE` 独立计算，**杜绝标签污染与未来函数**。 |
| **模型训练** | 滚动窗口 `train_window`（默认60交易日≈3个月）。到达窗口边界时，用最新截面数据重训 `ElasticNet`, `XGBoost`, `LightGBM`。单模型覆盖全市场股票。 |
| **调仓规则** | 每 **5个交易日** 触发调仓。按模型预测值截面排序，买入 **Top 50**。掉出前50的持仓立即清仓。 |
| **仓位计算** | 单票目标预算 `~20万`。严格遵循A股交易规则：`ceil(预算 / 当日收盘价 / 100) * 100` 向上取整至整手。双边手续费率默认 `0.02%`。 |
| **掩码控制** | `FEATURE_MASK` 控制训练样本准入；`BUY_MASK`/`SELL_MASK` 控制调仓日可交易池。掩码字段仅用于流程过滤，不混入模型特征。 |
| **评估输出** | 自动计算累计收益、年化、波动率、最大回撤、静态夏普。生成 **30日滚动夏普曲线**。所有模型结果合并至同一图表（共3张）。因子IC/IR导出为JSON。 |

## 🛠️ 环境依赖
```bash
# 推荐 Python 3.9+，PyArrow 12+ (用于高效 Parquet 读写)
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib pyarrow
```

## 🚀 快速运行
1. **确认路径**：确保数据已挂载至指定只读目录，或修改 `script/run_backtest.py` 顶部的路径配置。
2. **执行回测**：
   ```bash
   cd /data/cye_temp/workspace/backtest_engine
   python script/run_backtest.py
   ```
3. **查看输出**：
   - `output/figures/pnl_combined.png` → 多模型累计净值对比
   - `output/figures/drawdown_combined.png` → 策略回撤对比
   - `output/figures/rolling_sharpe_combined.png` → 30日滚动夏普比率
   - `output/factor_ic_ir.json` → 全量因子截面IC/IR有效性报告
   - `processed_data/backtest_*.csv` → 每日组合净值明细

## 🔧 核心参数调优指南
在 `script/run_backtest.py` 中修改 `BacktestEngine` 初始化参数：

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `rebalance_days` | `int` | `5` | 调仓频率（交易日） |
| `train_window` | `int` | `60` | 滚动训练窗口大小（交易日） |
| `budget_per_stock` | `float` | `200_000.0` | 单票目标建仓预算（元） |
| `initial_capital` | `float` | `10_000_000.0` | 初始总资金（元） |
| `commission_rate` | `float` | `0.0002` | 双边交易手续费率 |
| `models` | `List[str]` | `['ElasticNet','XGBoost','LightGBM']` | 启用的模型列表 |
| `feature_cols` | `List[str]` | 自动过滤 | 模型输入特征列（自动排除元数据/掩码/标签） |

## 📐 数据口径与字段规范
本项目严格遵循 `selected_training_fields_dictionary_cn.md` 定义：
- **总字段数**：`459`（模型输入候选 `452`）
- **标签字段**：`FWD_RET_5D_Z_P01_P99`（仅用于监督学习，回测PnL不使用）
- **元数据**：`S_INFO_WINDCODE`, `TRADE_DT`, `SW_L1_CODE`
- **控制掩码**：`FEATURE_MASK`（训练质量）, `BUY_MASK`（买入许可）, `SELL_MASK`（卖出许可）
- **特征类型**：`financial_or_fundamental` (152), `technical_factor` (104), `mask` (196)
- **口径隔离**：训练使用缩尾+Z-Score化的远期标签；回测使用 `log1p` 或原始 `S_DQ_ADJCLOSE` 计算的真实日收益率 `shift(-1)`。

## 📤 输出文件说明
| 路径 | 格式 | 内容 |
|:---|:---|:---|
| `output/figures/pnl_combined.png` | PNG | 多策略累计净值归一化曲线（基准=1.0） |
| `output/figures/drawdown_combined.png` | PNG | 各策略历史最大回撤填充图 |
| `output/figures/rolling_sharpe_combined.png` | PNG | 30日窗口滚动年化夏普比率时序 |
| `output/factor_ic_ir.json` | JSON | `{因子名: {mean_ic, icir, ic_positive_ratio, sample_days}}` |
| `processed_data/backtest_*.csv` | CSV | 每日 `TRADE_DT` 与 `Value` 明细，支持下游进一步分析 |

## ⚠️ 注意事项与最佳实践
1. **防泄漏设计**：训练窗口严格截止于调仓日前一日；标签与回测收益计算完全解耦。严禁在特征工程中引入未来数据。
2. **整手逻辑**：A股买入必须为100股整数倍。代码已实现 `np.ceil(budget / price / 100) * 100`，确保预算覆盖率与交易合规性。
3. **内存优化**：大数据集按年分块读取 + `groupby` 流式处理。若遇 OOM，建议在 `data_loader.py` 中启用 `polars` 或分片迭代。
4. **掩码处理**：`*_APPLICABLE_MASK` 与 `*_MISSING_MASK` 已按业务规范保留，但默认不作为模型特征输入。如需启用，请在 `run_backtest.py` 中手动追加至 `feature_cols`。
5. **扩展接口**：继承 `BacktestEngine` 并覆盖 `_predict()` 即可快速接入自定义信号或集成学习架构。

---
*版本: v2.1 | 更新日期: 2026-05-27 | 维护: Quant Eng Team*
```