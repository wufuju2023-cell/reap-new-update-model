#!/bin/bash
# restore.sh — 从 ModelScope 私有 repo 恢复状态（断点续训用）
# 用法: bash restore.sh --from modelscope:reap/rl-v1 --to /workspace/out
# 说明: 平台实例 destroy 后 /workspace 丢失；模型/花状态一律从归档拉回。
set -euo pipefail
FROM=${1:-modelscope:reap/rl-v1}; TO=${2:-/workspace/out}
echo "[restore] from=$FROM to=$TO"
mkdir -p "$TO"
# NOTE(骨架): 具体改用 modelscope API 下载私有 repo 文件:
#   1) 读 out/hash.txt 与远端清单 diff → 只拉缺失/改动文件
#   2) 大文件（权重）按 4 块并行 + sha256 校验, 临时文件 + mv
echo "[restore] (skeleton) 拉取 ckpt/adapters + value_head.pt + ttt_buffer.jsonl"
