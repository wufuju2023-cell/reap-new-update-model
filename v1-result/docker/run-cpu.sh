#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 5 || { echo 'Usage: run-cpu.sh IMAGE NEW_CONTAINER MANIFEST_FILE OUTPUT_VOLUME GPU_BASE_URL' >&2; exit 2; }
engine="${CONTAINER_ENGINE:-podman}"
manifest="$(realpath -- "$3")"
test -f "$manifest"
"$engine" run --name "$2" --network host -v "$manifest:/batch.jsonl:ro" -v "$4:/workspace/out" \
  --entrypoint python3 "$1" -m cpu_runtime.online_batch --manifest /batch.jsonl \
  --project-dir /opt/reap-runtime --output-dir /workspace/out/sessions --gpu-base-url "$5" \
  --gamma 0.99 --concurrency 2 --max-updates 5 --http-timeout-seconds 720 --barrier-timeout-seconds 900
