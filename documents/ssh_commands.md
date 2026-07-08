以下是专为 Linux 服务器环境设计的离线运行脚本 `run_backtest.sh`。该脚本已针对**无头环境（Headless）、长时间运行、实时日志追踪、依赖隔离**进行优化，可直接部署使用。

### 📄 `run_backtest.sh`
```bash
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
```

---

### 🛠️ 部署与使用说明

#### 1. 赋予执行权限
```bash
chmod +x /data/cye_temp/workspace/backtest_engine/run_backtest.sh
```

#### 2. 服务器后台运行（防断连）
推荐使用 `nohup` 或 `tmux`/`screen`：
```bash
# 方式 A：nohup (简单后台)
nohup ./run_backtest.sh > /dev/null 2>&1 &

# 方式 B：tmux (推荐，可随时 attach 查看进度)
tmux new -s backtest
./run_backtest.sh
# 按 Ctrl+B 然后按 D 脱离会话，后续用 tmux attach -t backtest 恢复
```

#### 3. 日志查看技巧
```bash
# 实时滚动查看日志
tail -f /data/cye_temp/workspace/backtest_engine/logs/run_20260601_*.log

# 查看错误堆栈（若运行失败）
cat /data/cye_temp/workspace/backtest_engine/logs/err_20260601_*.log
```

---

### 💡 核心设计说明
| 配置项 | 作用 | 为什么服务器必须加？ |
|:---|:---|:---|
| `export MPLBACKEND=Agg` | 强制 Matplotlib 使用非交互式后端 | 服务器无 X11/GUI，默认 `TkAgg` 会直接抛出 `TclError: no display name` 导致中断 |
| `export PYTHONUNBUFFERED=1` | 禁用 stdout/stderr 缓冲 | Python 默认 4KB 缓冲，长任务日志会“卡住”不刷新，设置后 `print` 实时落盘 |
| 标准输出/错误分离 | `>> run.log` 与 `>> err.log` | 便于快速定位是业务逻辑异常还是环境/依赖报错，避免日志混杂 |
| `cd $PROJECT_ROOT` | 强制工作目录对齐 | 脚本中 `sys.path.insert` 依赖 `__file__` 相对位置，固定 cwd 可避免 `ModuleNotFoundError` |

如需对接 **SLURM/PBS 集群调度** 或添加 **企业微信/钉钉告警**，可提供调度脚本模板。祝回测顺利！


在 Linux 服务器上检查后台回测任务是否中断，可分为 **状态确认**、**原因定位** 和 **预防监控** 三步。以下是针对量化回测场景（长耗时、高内存、多进程）的实操指南：

---
### 🔍 一、快速确认进程是否还在运行

| 命令 | 说明 | 输出解读 |
|:---|:---|:---|
| `pgrep -af run_backtest` | 精确匹配脚本名 | 返回 PID 表示存活；无输出表示已终止 |
| `ps aux \| grep backtest \| grep -v grep` | 传统进程查看 | 状态列：`R`/`S`=运行中，`Z`=僵尸进程，`T`=已停止 |
| `ls -l --time=modify /data/cye_temp/workspace/backtest_engine/logs/run_*.log` | 检查日志最后修改时间 | 若时间停留在数小时前，大概率已中断或卡死 |

💡 **实时盯盘推荐**：
```bash
watch -n 5 "pgrep -af run_backtest || echo '❌ 进程已退出'"
```

---
### 📂 二、根据启动方式精准判断

#### 1. 若使用 `nohup ./run_backtest.sh &`
```bash
# 查看标准输出/错误日志末尾
tail -n 50 logs/run_$(date +%Y%m%d)_*.log
tail -n 50 logs/err_$(date +%Y%m%d)_*.log

# 检查是否被系统杀死（无报错但进程消失）
grep -i "killed" logs/err_*.log || echo "未发现 kill 记录"
```
✅ **正常结束标志**：日志末尾出现 `🏁 回测任务结束 | ✅ 执行成功，退出码: 0`  
❌ **异常中断标志**：日志突然截断、出现 `Traceback`、或 `err_*.log` 有报错但主日志未打印结束标记。

#### 2. 若使用 `tmux` / `screen`
```bash
tmux ls          # 状态为 (attached)/(detached) 表示存活，(dead) 表示已中断
screen -ls       # 类似 tmux
```
恢复会话查看现场：
```bash
tmux attach -t backtest   # 或 screen -r backtest
```

#### 3. 若在当前终端直接 `&` 后台运行
```bash
jobs -l   # 仅在同一 shell 有效。显示 `+ Running` 或 `- Stopped`
```

---
### 🔎 三、定位中断根本原因（量化回测高频问题）

| 现象 | 排查命令 | 典型原因 |
|:---|:---|:---|
| **进程消失，日志无报错** | `dmesg -T \| grep -i oom \| tail -n 5` | 🚨 **OOM Killer 触发**（ML 训练/全量特征加载内存溢出） |
| **日志卡在某一日/某模型不动** | `strace -p <PID> 2>&1 \| tail -n 20` | 磁盘 IO 阻塞、网络挂载超时、或死锁 |
| `err_*.log` 报 `MemoryError` / `Killed` | `free -h` 查看剩余内存 | 未限制 Python 内存，触发系统保护 |
| 日志报 `No space left on device` | `df -h /data/cye_temp/` | 输出目录分区写满（parquet/日志/图片膨胀） |
| SSH 断开后任务消失 | `ps aux \| grep python \| grep -v grep` | 未使用 `nohup`/`tmux`，或标准输出未重定向导致挂起 |

🔧 **针对性修复**：
- **防 OOM**：在脚本开头添加 `export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"`（若用 GPU），或限制进程内存：
  ```bash
  # 启动时限制最大虚拟内存 16GB
  ulimit -v 16777216
  ./run_backtest.sh &
  ```
- **防卡死**：在 `backtest_engine.py` 循环内加超时保护或定期 `sys.stdout.flush()`。

---
### 🛡️ 四、生产环境防中断建议

#### 1. 添加进程守护（Systemd 示例）
创建 `/etc/systemd/system/backtest.service`：
```ini
[Unit]
Description=A-Share Backtest Engine
After=network.target

[Service]
Type=oneshot
User=your_user
WorkingDirectory=/data/cye_temp/workspace/backtest_engine
ExecStart=/bin/bash run_backtest.sh
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```
启用并运行：`sudo systemctl enable backtest && sudo systemctl start backtest`

#### 2. 企业微信/钉钉告警（轻量级）
在 `run_backtest.sh` 末尾 `exit $EXIT_CODE` 前添加：
```bash
if [ $EXIT_CODE -ne 0 ]; then
    curl -s "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY" \
      -H "Content-Type: application/json" \
      -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"⚠️ 回测异常中断！\\n退出码: $EXIT_CODE\\n日志: $STDERR_LOG\"}}"
fi
```

#### 3. 定期心跳检查（Cron）
```bash
# 每 10 分钟检查一次，若进程退出且日志未标记成功，则发通知
*/10 * * * * pgrep -f run_backtest >/dev/null 2>&1 || \
  grep -q "✅ 执行成功" /data/cye_temp/workspace/backtest_engine/logs/run_$(date +%Y%m%d)*.log 2>/dev/null || \
  echo "回测任务疑似中断" | mail -s "Backtest Alert" your@email.com
```

---
### 💡 附：一键诊断脚本 (`check_backtest.sh`)
```bash
#!/bin/bash
echo "🔍 回测任务状态检查 | $(date)"
echo "────────────────────────────────────"

PID=$(pgrep -f run_backtest)
if [ -z "$PID" ]; then
    echo "❌ 进程已不存在"
else
    echo "✅ 进程存活 | PID: $PID"
    ps -p $PID -o pid,stat,etime,cmd
fi

echo ""
echo "📂 最新日志状态:"
LOG=$(ls -t /data/cye_temp/workspace/backtest_engine/logs/run_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
    MOD_TIME=$(stat -c %y "$LOG" | cut -d. -f1)
    SIZE=$(du -h "$LOG" | cut -f1)
    echo "📄 文件: $LOG"
    echo "⏱️ 最后更新: $MOD_TIME | 📦 大小: $SIZE"
    echo "🔚 末尾 3 行:"
    tail -n 3 "$LOG" | sed 's/^/   /'
else
    echo "⚠️ 未找到运行日志"
fi

echo ""
echo "🚨 OOM 检查:"
dmesg -T 2>/dev/null | grep -i "oom" | tail -n 3 || echo "   未发现 OOM 记录"
```
赋予权限后执行：`chmod +x check_backtest.sh && ./check_backtest.sh`

按上述流程，可 100% 覆盖后台中断的检测、归因与防御。如需针对 **内存泄漏调试** 或 **断点续跑机制** 提供代码实现，请告知当前中断的具体表现（日志末尾/退出码/系统状态）。