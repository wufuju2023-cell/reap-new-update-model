#!/bin/bash
# scan-secrets.sh — 敏感指纹扫描（pre-commit / CI gate 用；只读）
# 用法: bash tools/scan-secrets.sh [dir]
cd "${1:-<local-f-drive>/projects/reap-new-update-model}"

declare -a PATTERNS=(
  'rc-[0-9a-f]{60,}'          # Radeon Cloud Model API Key
  'tskey-auth-'               # Tailscale auth key
  'ghp_[A-Za-z0-9]{30,}'      # GitHub PAT
  'BEGIN [A-Z ]*PRIVATE KEY'  # 私钥
  'gho_[A-Za-z0-9]{30,}'
  'sk-[A-Za-z0-9]{30,}'
  'Bearer [A-Za-z0-9._-]{20,}'
  '<email>'
)
hits=0
for p in "${PATTERNS[@]}"; do
  res=$(grep -rInE "$p" --include='*' . 2>/dev/null | grep -v '^\./\.git/' | head -10)
  if [ -n "$res" ]; then echo "=== [$p] ==="; echo "$res"; hits=$((hits+1)); fi
done
if [ "$hits" -eq 0 ]; then echo "SCAN OK: no secret patterns"; else echo "SCAN FAIL: $hits pattern groups hit"; exit 1; fi
