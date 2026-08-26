#!/usr/bin/env bash
# setup_week0.sh — 在 DSW-AMD 实例重建 v1 周 0 环境（幂等，可重复跑）
# 用法: nohup bash /mnt/workspace/v1/setup_week0.sh > /tmp/setup0.log 2>&1 &
set -u
W=/mnt/workspace/v1
mkdir -p "$W"/models "$W"/data "$W"/runs/eval_sets "$W"/services

echo "[1/7] Lean 4.28.0-rc1 prebuilt toolchain"
if [ ! -x /usr/local/bin/lake ]; then
  cd /tmp
  [ -f lean.tar.gz ] || curl -sL -o lean.tar.gz "https://gh-proxy.com/https://github.com/leanprover/lean4/releases/download/v4.28.0-rc1/lean-4.28.0-rc1-linux.tar.gz"
  mkdir -p /opt/lean4
  tar xzf lean.tar.gz -C /opt/lean4
  D=$(ls -d /opt/lean4/lean-* 2>/dev/null | head -1)
  [ -n "$D" ] && ln -sf "$D"/bin/lean "$D"/bin/lake "$D"/bin/leanc /usr/local/bin/
fi
echo "lean: $(lean --version 2>/dev/null || echo MISSING)"

echo "[2/7] reap repo (shallow, gh-proxy)"
[ -d "$W/reap/.git" ] || git clone --depth 1 "https://gh-proxy.com/https://github.com/IQuestLab/reap.git" "$W/reap"

echo "[3/7] mathlib4 src (tag v4.28.0-rc1)"
[ -d "$W/mathlib-src/Mathlib" ] || git clone --depth 1 --branch v4.28.0-rc1 "https://gh-proxy.com/https://github.com/leanprover-community/mathlib4.git" "$W/mathlib-src"

echo "[4/7] REAL-Prover 7B weights (~15GB, hf-mirror)"
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub
[ -d "$W/models/REAL-Prover" ] || python3 -c "from huggingface_hub import snapshot_download; snapshot_download('FrenzyMath/REAL-Prover', local_dir='$W/models/REAL-Prover')"

echo "[5/7] state_tactic_pairs dataset (55MB)"
[ -d "$W/data/state_tactic_pairs" ] || python3 -c "from huggingface_hub import snapshot_download; snapshot_download('FrenzyMath/state_tactic_pairs', repo_type='dataset', local_dir='$W/data/state_tactic_pairs')"

echo "[6/7] real-prover repo (FATE-M jsonl)"
[ -d "$W/real-prover/.git" ] || git clone --depth 1 "https://gh-proxy.com/https://github.com/frenzymath/REAL-Prover.git" "$W/real-prover"

echo "[7/7] eval sets (fatem-100 / holdout-30, 已随脚本上传)"
ls -la "$W/runs/eval_sets/" 2>/dev/null

echo "SETUP_DONE $(date -u +%FT%TZ)"
