Backtest Engine for A Share

*** 主结构 ***
Workspace: /data/cye_temp/workspace/.
** 数据源 **
（只允许读取， 不允许编辑或写入）
* 训练集 *
/data/train_data/model_ready_panel_selected_plus_ohlc_train_{YEAR}.parquet
YEAR = [2016, 2017, …, 2023]
* 测试集 *
/data/train_data/model_ready_panel_selected_plus_ohlc_test_{YEAR}.parquet
YEAR = [2024, 2025, 2026]
* 原始数据 （用 'S_DQ_ADJCLOSE' 来计算真实收益）*
/data/data_process/4.15_revision/raw_daily_panel.parquet

** 处理后的数据 **
（允许编辑和写入）
/data/cye_temp/processed_data/backtest/.
** 辅助脚本 **
/data/cye_temp/workspace/backtest_engine/util/.
** 脚本 **
/data/cye_temp/workspace/backtest_engine/script/.
** 源代码 **
/data/cye_temp/workspace/backtest_engine/src/.
** 结构参数 **
/data/cye_temp/workspace/backtest_engine/config/Config.py
** 模型参数 **
/data/cye_temp/workspace/backtest_engine/model/.
** 日志 **
/data/cye_temp/workspace/backtest_engine/log/.
** 输出 **
/data/cye_temp/workspace/backtest_engine/output/.
/data/cye_temp/workspace/backtest_engine/output/figures/.

*** 功能1：因子分析 ***
# 输入数据

# 输出：
** 代码 **
/data/cye_temp/workspace/backtest_engine/script/analysis.py (需要新建)
** 输入 **
时间段： 比如 ['2016-01-01', '2020-12-31']
** 数据预处理 **
依据时间段， 从原始数据计算日收益率 `S_DQ_ADJCLOSE` 为label_1
依据时间段， 从训练集或测试集提取因子信息(其中`FWD_RET_5D_Z_P01_P99`为label_2)
** 因子分析 **
1. 分别计算所有因子相对于label_1和label_2的IC和IR值
2. 计算所有因子的最大值、最小值、方差、均值
** 输出 **
把结果保存为json文件到output里， 格式样本：
"LESS_FIN_EXP_IND_Z": {
    "mean": 1e-5,
    "variance": 1.001,
    "max": 2.1242213,
    "min": -2.323232,
    "mean_ic_label_1": 0.00306144159876406,
    "icir_label_1": 0.08368726759021877,
    "ic_positive_ratio_label_1": 0.533515731874145,
    "mean_ic_label_2": 0.00306144159876406,
    "icir_label_2": 0.08368726759021877,
    "ic_positive_ratio_label_2": 0.533515731874145,    
    "sample_days": 731
    }
文件名：/data/cye_temp/workspace/backtest_engine/output/res_factor_stat.json

*** 模块1： 信号生成 ***
** 代码 **
/data/cye_temp/workspace/backtest_engine/model/baseline_model.py (需要新建)
** class **
* ElasticNet *
1.训练
输入： 训练集（因子 + label）
输出： 训练好的模型
2.生成信号
输入： 单日的因子
预测：用输入的因子评估label
生成排名：依照label的评估进行排名
* XGBoost *
1.训练
输入： 训练集（因子 + label）
输出： 训练好的模型
2.生成信号
输入： 单日的因子
预测：用输入的因子评估label
生成排名：依照label的评估进行排名
* LightGBM *
1.训练
输入： 训练集（因子 + label）
输出： 训练好的模型
2.生成信号
输入： 单日的因子
预测：用输入的因子评估label
生成排名：依照label的评估进行排名
* OptSharpe *
1.生成信号
输入： 过去30天的单日收益
权重优化：依照30天的历史单日收益， 评估Sharpe ratio最大化的权重
生成排名：依照权重的评估进行排名
参照代码：
def _calc_opt_sharpe_weights(self, valid_codes: List[str]) -> pd.Series:
    print("⚡ 计算 OptSharpe 权重...")
    print(f"  🔍 可用股票数: {len(valid_codes)} | 预热历史长度: {len(self.returns_history[valid_codes[0]]) if valid_codes else 0}天")
    """Baseline: 基于历史收益率的均值方差优化权重计算（严格保留原始逻辑）"""
    eligible = [c for c in valid_codes if len(self.returns_history.get(c, [])) >= 30]
    if len(eligible) < 10:
        return pd.Series(0.0, index=valid_codes)

    try:
        max_len = min(120, max(len(self.returns_history[c]) for c in eligible))
        aligned_data = {}
        for c in eligible:
            hist = self.returns_history[c][-max_len:]
            aligned_data[c] = [np.nan] * (max_len - len(hist)) + hist if len(hist) < max_len else hist
        
        ret_df = pd.DataFrame(aligned_data)
        mu = ret_df.mean()
        cov_matrix = ret_df.cov()
        reg = np.eye(len(eligible)) * 1e-6
        raw_w = np.linalg.solve(cov_matrix.values + reg, mu.values)
        raw_w = np.maximum(raw_w, 0)
        weights = raw_w / raw_w.sum() if raw_w.sum() > 1e-8 else np.zeros_like(raw_w)
    except Exception:
        ret_data = {c: self.returns_history[c] for c in eligible}
        ret_df = pd.DataFrame(ret_data)
        raw_w = np.maximum(ret_df.mean().values / (ret_df.var().values + 1e-8), 0)
        weights = raw_w / (raw_w.sum() + 1e-8)

    full_scores = pd.Series(0.0, index=valid_codes)
    full_scores[eligible] = weights
    print(f"(weights, codes) | {list(zip(weights, eligible))[:5]} ...")
    return full_scores

*** 功能2：回测 ***
** 代码 **
/data/cye_temp/workspace/backtest_engine/script/run_backtest.py
/data/cye_temp/workspace/backtest_engine/src/backtest_engine.py
/data/cye_temp/workspace/backtest_engine/src/backtest_core.py
** 输入 **
预热时间段： 从 '2016-01-01' 到 '2017-12-31'
回测时间段： 比如 ['2018-01-01', '2022-12-31']
最少回顾天数： 比如30天
最大回顾天数： 比如365天
仓位调整周期：比如5天
模型重新训练周期：比如120天
初始本金：1000万元
回测模型： 比如 ['ElasticNet', 'XGBoost', 'LightGBM', 'OptSharpe']
** 数据预处理 **
依据时间段， 从原始数据计算日收益率，`S_DQ_ADJCLOSE`
依据时间段， 从训练集或测试集提取因子信息
** 回测流程 **
1.每个模型重新训练周期， 对需要训练的模型进行重新训练，训练集的时间窗口为从过往第5天开始，回溯最大回顾天数的天数（若可回顾天数小于最大回顾天数， 则采用所有历史数据）。要求：注意避免数据泄漏。每个模型训练时，均使用当前训练集内的所有股票数据， 即使用一个模型来处理所有股票。
2.每个仓位调整周期，依照不同模型（参照‘模块1：信号生成’），对股票进行排序。 对于 ['ElasticNet', 'XGBoost', 'LightGBM']， 输入当天的单天因子； 对于 ['OptSharpe']， 输入过去30天的`S_DQ_ADJCLOSE`。 选择排名前50名的股票，对于依然保留在前50名的股票，不进行仓位变动；对于掉出前50名的持有仓位的股票则清仓卖出；对于新进前50名的股票，使用当前现金等权买入。
3.每个交易日开始时， 用`S_DQ_ADJCLOSE`计算当前净值。并把交易记录保存到/data/cye_temp/workspace/backtest_engine/log/.
** 结果评估 **
生成并保存PnL曲线、回撤、30天窗口的Sharpe ratio。 要求把所有模型的结果保存到同一张图里， 即合计保存3张图。

*** 已有项目结构 ***
/data/cye_temp/workspace/backtest_engine/script/run_backtest.py
/data/cye_temp/workspace/backtest_engine/util/data_loader.py
/data/cye_temp/workspace/backtest_engine/util/metrics.py
/data/cye_temp/workspace/backtest_engine/src/backtest_engine.py
/data/cye_temp/workspace/backtest_engine/src/backtest_core.py
/data/cye_temp/workspace/backtest_engine/config/Config.py

*** 其他资料 ***
数据处理的说明文档：
selected_training_fields_dictionary_cn(1).md