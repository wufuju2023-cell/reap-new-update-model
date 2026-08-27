#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
engine="${CONTAINER_ENGINE:-podman}"
case "${1:-}" in
  cpu)
    test "$#" -eq 2 || { echo 'Usage: build.sh cpu IMAGE' >&2; exit 2; }
    "$engine" build -f "$root/v1-result/docker/cpu.Dockerfile" -t "$2" "$root"
    ;;
  gpu-existing)
    test "$#" -eq 6 || { echo 'Usage: build.sh gpu-existing IMAGE MODEL_DIR MANIFEST MANIFEST_SHA256 NEW_CONTEXT' >&2; exit 2; }
    test "${AUTHORIZED_REMOTE_BUILDER:-}" = yes || { echo 'Only run on a user-authorized remote builder; no local model copies.' >&2; exit 2; }
    python3 "$root/v1-result/docker/prepare_context.py" --authorized-remote-builder \
      --repo-root "$root" --model-dir "$3" --manifest "$4" --manifest-sha256 "$5" --output "$6"
    "$engine" build --target release-existing --build-arg "REAL_PROVER_MANIFEST_SHA256=$5" \
      -f "$6/containers/gpu/Containerfile" -t "$2" "$6"
    ;;
  *) echo 'Choose cpu or gpu-existing; see docker/README.md.' >&2; exit 2 ;;
esac
