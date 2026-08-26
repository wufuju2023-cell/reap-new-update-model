#!/bin/bash
# archive.sh — 每天收尾：状态归档 → ModelScope + hash 清单进 git
# 在执行实例内运行; 幂等（重复运行只追加新增文件）。
set -euo pipefail
OUT=/workspace/out; STAMP=/workspace/bootstrap.stamp
TS=$(date -u +%FT%TZ)
TAR=/workspace/out/state-${TS}.tar.zst
[ -d "$OUT" ] || { echo "[archive] no out dir"; exit 1; }
echo "[archive] tar: $TAR"
tar -cf "$TAR" checkpoints/ value_head.pt ttt_buffer.jsonl rttt_metrics.jsonl 2>/dev/null \
  || tar -cf "$TAR" -C "$OUT" checkpoints value_head.pt ttt_buffer.jsonl rttt_metrics.jsonl 2>/dev/null || true
# NOTE(骨架): 之后替换为 modelscope 上传 (私有 repo reap/rl-v1) + 失败重试 2 次
echo "[archive] (skeleton) ← 实现: modelscope 上传 + 失败重试"
# hash 清单（产物指纹）
{ echo "# archive $TS"; sha256sum checkpoints/* value_head.pt ttt_buffer.jsonl 2>/dev/null | sort; } > "$OUT/hash.txt"
echo "[archive] hash at $OUT/hash.txt"
