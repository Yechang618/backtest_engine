以下为您基于最新代码库与您的明确指示（**全量权重再平衡**）深度重构的 **A股多因子量化回测系统架构与执行规范文档**。本文档严格遵循量化工程标准，可作为团队协作、代码审计与实盘对接的基准合同。

---

# A股多因子量化回测系统执行合同与规范 (Execution Contract)

## 1. 用途与适用范围
本文件规定 A股多因子量化回测系统（A-Share Backtest Engine）的执行口径、交易逻辑与评价标准。本系统旨在评估不同机器学习模型、特征子集（消融实验）、主题股票池约束以及双重动态路由策略下的样本外表现。

**适用策略与实验组：**
| 实验组 | 候选池 | 目标持仓数 (Top K) | 排除北交所 (`.BJ`/`_BJ`) | 核心逻辑 |
| :--- | :--- | :--- | :--- | :--- |
| **Full-Market** | 全市场有效标的 (`BUY_MASK=1`) | 25 | 是 | 全截面多因子打分排序 |
| **Tech-25 (Pool)** | 当月生效的 `trade_pool_YYYYMM.csv` | 25 | 是 | 主题股票池硬约束 + 模型打分 |
| **Ablation** | 全市场有效标的 | 25 | 是 | 使用 SHAP 筛选的 Top 因子子集 |
| **DynamicSwitch** | 全市场 / 股票池 | 25 | 是 | 基于 Raw Rank IC 的动态模型路由 |
| **SensitiveSwitch** | 全市场 / 股票池 | 25 | 是 | 基于 Residual Rank IC 的动态模型路由 |

**模型输入要求：**
*   每行唯一键为 `S_INFO_WINDCODE + TRADE_DT`。
*   特征必须经过严格的截面标准化（`_MKT_Z` 全市场标准化，`_IND_Z` 行业标准化）。
*   严格遵循物理级数据隔离：训练集与测试集绝对隔离，样本外测试期间模型参数永久冻结，杜绝任何数据泄漏。

## 2. Residual 分数处理 (SensitiveSwitch 核心)
为剥离风格与行业暴露，系统每日对模型原始预测值 `PRED_Z_RAW` 执行 OLS 回归。

### 2.1 处理顺序
每个交易日独立执行以下步骤：
1. 读取模型原始分数 `PRED_Z_RAW`。
2. 先应用策略候选池过滤（如启用股票池硬约束）。
3. 在过滤后的当日横截面计算风格暴露。
4. 以 `PRED_Z_RAW` 对风格变量和申万一级行业哑变量做无权最小二乘回归（OLS）。
5. 回归残差记为 `RESID_SCORE`，分数越大，表示在控制当日风格与行业后模型给出的相对高分越强。
*注：Residual 只用于排序和路由比较，不改写模型原始输出文件。*

### 2.2 回归形式与风格变量
对股票 `i`、交易日 `t`：
```text
PRED_Z_RAW(i,t) = alpha(t) + beta_size * z(log_mcap) + beta_turnover * z(turnover_proxy) 
                + beta_vol20 * z(vol20) + beta_vol60 * z(vol60) 
                + beta_mom20 * z(mom20) + beta_mom60 * z(mom60) 
                + SW_L1 industry dummies + residual(i,t)
```
| 变量 | 当前代码定义 |
| :--- | :--- |
| `log_mcap` | `log1p(S_DQ_CLOSE * S_DQ_CAPITAL / 10000)` (缺失回退 `PV_CAPITAL_LOG_MKT_Z`) |
| `turnover_proxy` | `S_DQ_VOLUME / S_DQ_CAPITAL` (缺失回退 `TURNOVER_SHARE_OF_MARKET_MKT_Z`) |
| `vol20` / `vol60` | `S_DQ_ADJCLOSE` 日收益率的 20/60 日标准差 (`ddof=0`)，最少 10/20 个观察值 |
| `mom20` / `mom60` | `S_DQ_ADJCLOSE` 的 20/60 日涨跌幅 |

### 2.3 有效性与缺失处理
*   连续风格变量在当日有效横截面中以总体标准差 `ddof=0` 做 z-score。
*   行业使用 `SW_L1_NAME`，缺失时回退 `SW_L1_CODE`，再缺失标记为 `missing`；行业哑变量保留截距并去掉一个基准行业 (`drop_first=True`)。
*   当日有效样本 `< 100` 只，或设计矩阵列数 `>= 样本数 - 5` 时，不回归。
*   Top 25 排序前，残差缺失值降为极低分 `-1e6`；若有效残差不足 Top 25，则降级使用包含 `-1e6` 的原始残差补齐。

## 3. 回测时间与调仓框架
| 项目 | 固定口径 |
| :--- | :--- |
| **交易日历** | A股标准交易日历（自动处理节假日顺延） |
| **调仓频率** | **按周调仓**：由 `REBALANCE_WEEKDAY` 控制（当前配置为 `0` 即周一）。提取每周中 `weekday >= 目标星期` 的交易日，取**最小值**。若目标日放假，自动顺延至该周第一个满足条件的交易日。 |
| **执行滞后** | `execution_lag_days = 0`（当日收盘信号，当日收盘成交假设） |
| **排序时点** | 调仓日当日横截面 |
| **成交价/估值** | `S_DQ_ADJCLOSE` (后复权收盘价) |
| **模型冻结** | 样本外测试期间，模型参数永久锁定，仅执行 `predict`。 |

## 4. 调仓与资金机制 (全量权重再平衡)
本系统采用**全量权重再平衡 (Full Weight Rebalancing)** 逻辑，每次调仓日强制将组合内所有持仓的权重拉回目标等权状态，以严格控制单只股票的集中度风险。

### 4.1 初始建仓
*   **初始资金**：10,000,000 元。
*   **BuyAndHoldAll 基线**：首个交易日全量等权买入所有 `BUY_MASK=1` 标的，后续永不换仓，不受股票池约束。
*   **ML 策略建仓**：首个调仓日，按模型预测分数排序选取 Top 25，可用现金在 Top 25 间尽量等额分配。

### 4.2 后续全量再平衡 (核心逻辑)
调仓日按下列顺序执行：
1.  **计算总净值**：`Total_NAV = Cash + Σ(Shares * S_DQ_ADJCLOSE)`。
2.  **计算目标权重**：单只目标股票的目标市值 `Target_Value = Total_NAV / Top_K`。
3.  **计算目标股数**：按 `S_DQ_ADJCLOSE` 向下取整至 100 股（A股整手规则），`Target_Shares = floor(Target_Value / Price / 100) * 100`。
4.  **生成调仓差异 (Delta)**：`Delta = Target_Shares - Current_Shares`。
    *   **跌出 Top 25**：`Current_Shares` 全部作为负 Delta，生成 SELL 指令。
    *   **留在 Top 25 内**：若权重因市场漂移偏离目标，`Delta` 可正可负，生成 BUY (加仓) 或 SELL (减仓) 指令。
    *   **新进 Top 25**：`Current_Shares` 为 0，`Delta` 为正，生成 BUY 指令。
5.  **执行顺序**：严格按 **SELL → BUY** 顺序执行。优先清仓和减仓释放现金，确保后续买入指令有足够的资金支持。
6.  **成本扣除**：买入扣除单边佣金，卖出扣除单边佣金。若现金不足以买入 1 手，则跳过该标的。

## 5. 成本、交易约束与不可成交处理
### 5.1 成本
| 项目 | 当前值 |
| :--- | :--- |
| **单边佣金** | 2 bps (`COMMISSION_RATE = 0.0002`) |
| **印花税/滑点** | 0 bps (当前配置未计入，预留接口) |
| **买入成本** | `成交股数 * 价格 * (1 + 0.0002)` |
| **卖出到账** | `成交股数 * 价格 * (1 - 0.0002)` |

### 5.2 买卖约束条件
*   **买入条件**：`FEATURE_MASK = 1` 且 `BUY_MASK = 1` 且 价格有效 (`> 1e-6`) 且 属于当月股票池（若启用 `trade_pools` 硬约束）。
*   **卖出条件**：标的跌出 Top 25 或 需减仓，且 价格有效。
*   **停牌处理**：若股票当日无有效价格 (`<= 1e-6`)，则使用 `last_known_prices` 进行净值估值，且跳过该股票的买卖交易。

## 6. 双重动态路由状态机
引擎内置两种元策略，在**调仓日**评估并切换底层打分模型：

| 路由策略 | 评估指标 | 历史窗口 | 切换条件 | Top 25 生成逻辑 |
| :--- | :--- | :--- | :--- | :--- |
| **DynamicSwitch** | 每日截面 **Raw Rank IC** | 过去 10 天 | `Best_IC > B * Current_IC` (默认 B=1.00)，且当前模型历史 IC 记录已满 10 天（防抖） | 使用选中模型的 **Raw 预测分数** 降序取 Top 25 |
| **SensitiveSwitch** | 每日截面 **Residual Rank IC** | 过去 10 天 | 选取平均 Residual IC 最高的模型 | 使用选中模型的 **Residual 残差** 降序取 Top 25 (优先过滤 `> -1e5` 的有效残差) |

## 7. 无未来函数误差结算 (Delayed Realized Error)
为严谨评估模型预测能力，MSE/MAE 采用延迟结算机制：
*   **T 日 (预测日)**：模型生成预测分数后，**仅缓存**至 `prediction_cache[T]`，绝不访问 `label_i`。
*   **T+i 日 (结算日)**：当时间推进到 `T+i` 日（即 `REBALANCE_DAYS` 天后），真实收益已客观实现。系统回溯读取 T 日缓存，与已实现的 `label_i` 计算 MSE/MAE。
*   **记录归属**：误差记录在 **T+i 日（结算日）** 的时间序列上，真实反映“事后验尸”的客观规律，结算后释放内存。

## 8. 结果评价与交付要求
### 8.1 核心评价指标
| 指标 | 定义 / 计算口径 |
| :--- | :--- |
| **Total PnL / Annual Ret** | 测试期累计收益率 / 年化收益率 `((1+cum)^(252/days)-1)` |
| **Max DD / Sharpe** | 历史最大回撤 / 年化夏普比率 `AnnRet / (Vol + 1e-8)` |
| **Win Rate** | **单笔平仓胜率** (`wins / total_closed_trades`，基于加权平均持仓成本对比) |

### 8.2 交付物清单
每个实验组必须输出以下文件至 `output/` 目录：
1.  **绩效图表**：`pnl_combined.png` (归一化净值), `drawdown_combined.png` (回撤), `rolling_sharpe_combined.png` (30日滚动夏普)。
2.  **诊断图表**：
    *   `daily_rank_ic_trend.png`：各模型 10日 MA Raw Rank IC 趋势，**下方附带 DynamicSwitch 切换阶梯图**。
    *   `daily_resid_ic_trend.png`：各模型 10日 MA Residual Rank IC 趋势，**下方附带 SensitiveSwitch 切换阶梯图**。
    *   `pred_top*_mse_*.png`：无未来函数延迟结算的 MSE 时序图。
3.  **可解释性**：`shap_summary_*.png` (蜂群图), `shap_importance_*.json`。
4.  **审计日志**：`log/trades_{MODEL}.csv` (完整买卖明细), `output/backtest_summary.json` (结构化绩效汇总)。

## 9. 验收清单 (Checklist)
- [ ] **数据隔离**：确认 2025-01-01 后无模型重训日志，`prediction_cache` 仅在 T+i 日读取 T 日缓存。
- [ ] **特征路由**：确认 `XGB-*` 使用了 `feature_cols_xgb`，`LGBM-*` 使用了 `feature_cols_lgbm`，且自动对齐生效。
- [ ] **全量再平衡**：检查 `log/trades_*.csv`，确认调仓日对留在 Top 25 内但权重发生漂移的标的产生了加仓/减仓指令（非零摩擦持有）。
- [ ] **股票池约束**：确认 Tech-25 实验的持仓标的 100% 属于当月 `trade_pool` 名单。
- [ ] **节假日顺延**：确认按周调仓逻辑在遇到长假时，正确顺延至节后首个 `weekday >= target_wd` 的交易日。
- [ ] **双重路由**：确认 `DynamicSwitch` 使用 Raw IC (10天窗口, B=1.00)，`SensitiveSwitch` 使用 Residual IC (5天窗口)，且仅在调仓日评估。
- [ ] **残差降级**：确认 `SensitiveSwitch` 在有效残差不足 Top 25 时，能正确降级使用包含 `-1e6` 的原始残差补齐。
- [ ] **胜率统计**：确认 `backtest_summary.json` 中的 `win_rate_pct` 基于真实的平仓成本对比。

## 10. 代码依据
*   **配置中心**：`config/Config.py` (路径、调仓周期、模型白名单、特征子集、股票池路径)
*   **数据加载与隔离**：`util/data_loader.py` (Parquet 读取、北交所过滤、衍生因子计算、截面 IC/IR 评估)
*   **回测调度与路由**：`src/backtest_engine.py` (主循环、特征路由、按周顺延、Residual OLS 剥离、双重动态切换、延迟误差结算)
*   **组合管理与全量再平衡**：`src/backtest_core.py` (目标等权计算、Delta 生成、先卖后买执行、成本追踪、胜率统计)
*   **主入口与实验编排**：`script/run_backtest.py` (多相位/多实验组调用、股票池加载、图表生成、JSON 汇总)