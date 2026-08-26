#!/bin/bash
# bootstrap.sh — 一次安装，任意新实例复用（container-like control plane, §04）
# 用法（在实例内 /workspace 下执行一次）:  bash bootstrap.sh
# 幂等: 重复执行安全（runner 先杀后起, stamp 覆盖）
set -e
cd /workspace

echo "[bootstrap] 1/5 control-plane: runner.py"
pkill -f "python3 /workspace/runner.py" 2>/dev/null || true
# 保证 runner.py 与 bootstrap.sh 同目录即用（被 push 上传到 /workspace）
[ -f /workspace/runner.py ] || cp "$(dirname "$0")/runner.py" /workspace/runner.py
nohup python3 /workspace/runner.py > /workspace/runner.log 2>&1 &
echo "[bootstrap] runner pid=$!"

echo "[bootstrap] 2/5 env fingerprint"
{
  echo "stamp: $(hostname) $(date -u +%FT%TZ)"
  echo "python: $(python3 --version 2>&1)"
  echo "rocm: $(ls /opt/rocm 2>/dev/null | head -1 || echo n/a)"
  echo "gpus: $(rocm-smi --showmeminfo vram 2>/dev/null | grep -c 'GPU\[[0-9]' || echo 0)"
  echo "torch: $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
} > /workspace/bootstrap.stamp

echo "[bootstrap] 3/5 dirs"
mkdir -p /workspace/app /workspace/data /workspace/out /workspace/logs

echo "[bootstrap] 4/5 receipt"
printf '\n$ %s bootstrap done (host=%s, gpus=%s)\n' "$(date -u +%FT%TZ)" "$(hostname)" \
  "$(rocm-smi --showmeminfo vram 2>/dev/null | grep -c 'GPU\[[0-9]' || echo 0)" >> /workspace/qout

echo "[bootstrap] 5/5 done"
cat /workspace/bootstrap.stamp
