#!/usr/bin/env bash
set -euo pipefail
: "${RUN:?}" "${CPU_IMAGE:?}" "${GPU_URL:?}"
sid=${1:?session ID required}; input=${2:?input filename required}; shift 2
[[ "$sid" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$ ]]
[[ "$input" =~ ^0[1-5]-[A-Za-z]+\.lean$ ]]
test -f "$RUN/inputs/$input"
test ! -e "$RUN/outputs/$sid"
name="reap-online-$sid"
set +e
podman container exists "$name"
exists=$?
set -e
test "$exists" -eq 1
mkdir "$RUN/launches/$sid"
set +e
podman run --name "$name" --pull never --network host \
  --userns keep-id:uid=10001,gid=10001 --user 10001:10001 \
  --volume "$RUN/source:/repro/source:ro" \
  --volume "$RUN/inputs:/repro/inputs:ro" \
  --volume "$RUN/outputs:/workspace/out:rw" \
  --env PYTHONPATH=/repro/source --env PYTHONDONTWRITEBYTECODE=1 \
  --workdir /opt/reap-runtime --entrypoint python3 "$CPU_IMAGE" \
  -B -m cpu_runtime.online_ttt \
  --session-id "$sid" --project-dir /opt/reap-runtime \
  --theorem-file "/repro/inputs/$input" --output-dir /workspace/out \
  --gpu-base-url "$GPU_URL" --gamma 0.99 --max-updates 5 \
  --experience-candidate --http-timeout-seconds 1140 \
  --barrier-timeout-seconds 1200 "$@" \
  > "$RUN/launches/$sid/stdout.log" 2> "$RUN/launches/$sid/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$RUN/launches/$sid/returncode.txt"
podman inspect "$name" > "$RUN/launches/$sid/container-inspect.json"
inspect_rc=$?
set -e
test "$inspect_rc" -eq 0 || exit "$inspect_rc"
exit "$rc"
