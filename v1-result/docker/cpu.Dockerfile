# syntax=docker/dockerfile:1.7
ARG LEAN_BASE_IMAGE=ghcr.io/wufuju2023-cell/reap-lean@sha256:6be053f9c1395890ca28385da56888f0735f785e6e56f54a8b96692cace8e353
FROM ${LEAN_BASE_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG REAP_REPOSITORY=https://github.com/IQuestLab/reap.git
ARG REAP_COMMIT=0090d73c5f739e4d74000e053b00fd0148ff46aa

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl git python3 zstd \
    && rm -rf /var/lib/apt/lists/*

# Reuse the verified official release binaries in the digest-pinned GHCR image.
# Only Reap and the V1 overlay are compiled; Lean itself is never built here.
ENV PATH=/opt/lean/bin:${PATH}
RUN lean --version | grep -F 'version 4.28.0-rc1,'

WORKDIR /opt/reap
RUN git init \
    && git remote add origin "${REAP_REPOSITORY}" \
    && git fetch --depth 1 origin "${REAP_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${REAP_COMMIT}"

COPY containers/cpu/reap-overlay/Reap/Training.lean Reap/Training.lean
COPY containers/cpu/reap-overlay/Reap/Training Reap/Training
COPY containers/cpu/patches/0001-training-endpoints-and-value.patch /tmp/reap-v1.patch
COPY containers/cpu/patches/0002-training-observer.patch /tmp/reap-observer.patch
COPY containers/cpu/patches/0003-strict-value-errors.patch /tmp/reap-strict-value.patch
RUN sed -i 's/\r$//' /tmp/reap-v1.patch \
    && git apply --check /tmp/reap-v1.patch \
    && git apply /tmp/reap-v1.patch \
    && sed -i 's/\r$//' /tmp/reap-observer.patch \
    && git apply --check /tmp/reap-observer.patch \
    && git apply /tmp/reap-observer.patch \
    && sed -i 's/\r$//' /tmp/reap-strict-value.patch \
    && git apply --check /tmp/reap-strict-value.patch \
    && git apply /tmp/reap-strict-value.patch \
    && lake build \
    && lake build Reap.Training.RolloutSink \
    && lake build Reap.Training

# Runtime project pins mathlib to the same Lean release and bakes its binary cache.
WORKDIR /opt/reap-runtime
COPY containers/cpu/runtime/lean-toolchain ./lean-toolchain
COPY containers/cpu/runtime/lakefile.toml ./lakefile.toml
COPY containers/cpu/runtime/ReapRuntime.lean ./ReapRuntime.lean
COPY containers/cpu/runtime/Smoke.lean ./Smoke.lean
RUN lake update \
    && lake exe cache get \
    && lake build

FROM ubuntu:24.04 AS runtime
ARG DEBIAN_FRONTEND=noninteractive
# Ubuntu 24.04 already reserves UID 1000 for its ubuntu user.
# Keep the application non-root without changing that existing account.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 zstd \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 reap

ENV PATH=/opt/lean/bin:${PATH} \
    PYTHONPATH=/opt/reap-tools \
    REAP_PROJECT_DIR=/opt/reap-runtime \
    REAP_OUTPUT_DIR=/workspace/out

COPY --from=build /opt/lean /opt/lean
COPY --from=build --chown=reap:reap /opt/reap /opt/reap
COPY --from=build --chown=reap:reap /opt/reap-runtime /opt/reap-runtime
COPY --chown=reap:reap cpu_runtime /opt/reap-tools/cpu_runtime
COPY --chown=reap:reap containers/cpu/tests/fixtures/real/RecurrenceSquare.lean /opt/reap-runtime/real-fixtures/RecurrenceSquare.lean
COPY --chown=reap:reap containers/cpu/tests/fixtures/real/RecurrenceGeometric.lean /opt/reap-runtime/real-fixtures/RecurrenceGeometric.lean
COPY --chown=reap:reap containers/cpu/smoke_cpu.sh /usr/local/bin/reap-cpu-smoke

# Reap's Requests library shells out to curl for policy/value HTTP calls.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/\r$//' /usr/local/bin/reap-cpu-smoke \
    && chmod 0755 /usr/local/bin/reap-cpu-smoke \
    && mkdir -p /workspace/out && chown -R reap:reap /workspace
USER reap
WORKDIR /opt/reap-runtime

ENTRYPOINT ["python3", "-m", "cpu_runtime.batch_solver"]
CMD ["--help"]
