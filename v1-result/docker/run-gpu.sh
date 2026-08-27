#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 3 || { echo 'Usage: run-gpu.sh IMAGE NEW_CONTAINER NEW_OUTPUT_VOLUME' >&2; exit 2; }
test "${AUTHORIZED_GPU_RUNTIME:-}" = yes || { echo 'Requires user-authorized AMD container devices/runtime.' >&2; exit 2; }
engine="${CONTAINER_ENGINE:-podman}"
"$engine" run --name "$2" --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 8g -p 127.0.0.1:8760:8760 -v "$3:/workspace/out" "$1"
