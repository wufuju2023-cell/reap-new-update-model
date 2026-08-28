#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-/workspace/out/cpu-smoke}"
mkdir -p "$out_dir"

python3 -m cpu_runtime.mock_services --port 18080 >"$out_dir/mock.log" 2>&1 &
mock_pid=$!
trap 'kill "$mock_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 50); do
  if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=1)"; then
    break
  fi
  sleep 0.1
done

python3 -m cpu_runtime.batch_solver \
  --manifest /opt/reap-tools/cpu_runtime/examples/smoke_sessions.jsonl \
  --project-dir /opt/reap-runtime \
  --output-dir "$out_dir/sessions" \
  --concurrency 2 \
  --timeout-seconds 120

python3 -m cpu_runtime.normalize_rollout \
  --sessions-dir "$out_dir/sessions" \
  --output "$out_dir/trajectories.jsonl"

python3 - "$out_dir" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summaries = [json.loads(line) for line in (root / "sessions" / "summary.jsonl").read_text().splitlines()]
assert len(summaries) == 2, summaries
assert all(item["solved"] for item in summaries), summaries
assert (root / "trajectories.jsonl").stat().st_size > 0
print(json.dumps({"ok": True, "sessions": len(summaries), "output": str(root)}))
PY

