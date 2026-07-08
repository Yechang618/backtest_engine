#!/bin/bash
# ==============================================================================
# 🚀 A股量化回测离线运行脚本 (Server-Ready)
# 适用于 Linux 服务器，支持 Matplotlib 无头渲染、实时日志、异常捕获
# ==============================================================================

# ------------------------- 🔧 配置区 -------------------------
# 项目根目录 (严格对应您的 workspace 结构)
PROJECT_ROOT="/data/cye_temp/workspace/backtest_engine"

# Python 执行器路径 (若使用 Conda，请替换为绝对路径，例如: /opt/conda/envs/quant/bin/python)
PYTHON_BIN="python3"

# 主入口脚本
MAIN_SCRIPT="${PROJECT_ROOT}/script/run_backtest.py"

# 日志配置
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STDOUT_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"
STDERR_LOG="${LOG_DIR}/err_${TIMESTAMP}.log"

# ------------------------- 🛠️ 环境初始化 -------------------------
# 1. 创建日志目录
mkdir -p "$LOG_DIR"

# 2. 切换至项目根目录 (确保相对路径与 sys.path 注入正确)
cd "$PROJECT_ROOT" || { echo "❌ 无法进入项目目录: $PROJECT_ROOT"; exit 1; }

# 3. ⚠️ 关键：设置 Matplotlib 为 Agg 后端，解决服务器无图形界面的 TclError
export MPLBACKEND=Agg

# 4. ⚠️ 关键：禁用 Python 标准输出缓冲，保证日志实时写入磁盘
export PYTHONUNBUFFERED=1

# 5. (可选) 若使用 Conda 环境，请取消注释并修改下方路径
# source /opt/conda/etc/profile.d/conda.sh
# conda activate your_quant_env

# ------------------------- 📝 执行回测 -------------------------
echo "========================================================" | tee "$STDOUT_LOG"
echo "🚀 开始执行回测任务 | $(date)" | tee -a "$STDOUT_LOG"
echo "📂 工作目录: $(pwd)" | tee -a "$STDOUT_LOG"
echo "🐍 执行环境: $($PYTHON_BIN --version 2>&1)" | tee -a "$STDOUT_LOG"
echo "========================================================" | tee -a "$STDOUT_LOG"

# 执行主脚本：标准输出追加至日志，错误单独记录
$PYTHON_BIN "$MAIN_SCRIPT" >> "$STDOUT_LOG" 2>> "$STDERR_LOG"
EXIT_CODE=$?

# ------------------------- ✅ 结果汇总 -------------------------
echo "" | tee -a "$STDOUT_LOG"
echo "========================================================" | tee -a "$STDOUT_LOG"
echo "🏁 回测任务结束 | $(date)" | tee -a "$STDOUT_LOG"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 执行成功，退出码: $EXIT_CODE" | tee -a "$STDOUT_LOG"
else
    echo "❌ 执行异常，退出码: $EXIT_CODE" | tee -a "$STDOUT_LOG"
    echo "📉 错误详情请查看: $STDERR_LOG" | tee -a "$STDOUT_LOG"
fi
echo "📄 完整运行日志: $STDOUT_LOG" | tee -a "$STDOUT_LOG"
echo "========================================================" | tee -a "$STDOUT_LOG"

exit $EXIT_CODE