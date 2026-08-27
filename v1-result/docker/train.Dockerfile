# syntax=docker/dockerfile:1.7
ARG GPU_BASE_IMAGE=rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e
FROM ${GPU_BASE_IMAGE} AS runtime-base

ARG REAL_PROVER_REPOSITORY=FrenzyMath/REAL-Prover
ARG REAL_PROVER_REVISION=fe76f68d9a88f342cb7b546307c20292fea9cced
ARG REAL_PROVER_HIDDEN_SIZE=3584
ARG REAL_PROVER_MANIFEST_SHA256=b20526a1d3a08365893fe937dfdec4b1d953b8c63f26cc89e922c962c11f3915

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/reap-python:/opt/reap-gpu \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    REAP_MODEL_PATH=/opt/models/REAL-Prover \
    REAP_SNAPSHOT_ROOT=/workspace/out/snapshots

# Install only the reviewed/hash-pinned wheels in an image-local overlay. The
# ROCm base's torch remains untouched; no resolver may fetch another torch.
COPY containers/gpu/requirements-gpu-hashed.lock /tmp/requirements-gpu-hashed.lock
RUN <<'SHELL'
set -eu
python - <<'PY'
import json, re
from pathlib import Path
import torch
pins = {}
for line in Path('/tmp/requirements-gpu-hashed.lock').read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    match = re.fullmatch(r'([a-z0-9-]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})', line)
    if match is None:
        raise RuntimeError('GPU dependencies must be exact wheel hashes')
    name = match.group(1)
    if name in pins or name in {'torch', 'torchvision', 'torchaudio', 'triton', 'pytorch-triton', 'pytorch-triton-rocm'} or name.startswith(('torch-', 'nvidia-', 'rocm-')):
        raise RuntimeError('Duplicate or forbidden host-GPU package in lock')
    pins[name] = match.group(2)
if len(pins) != 22:
    raise RuntimeError('Expected the reviewed 22-package GPU dependency lock')
Path('/opt/reap-base-torch.json').write_text(json.dumps({'version': str(torch.__version__), 'hip': torch.version.hip, 'module': str(Path(torch.__file__).resolve())}))
PY
python -m pip --isolated --disable-pip-version-check --no-cache-dir install \
    --target /opt/reap-python --require-hashes --no-deps \
    -r /tmp/requirements-gpu-hashed.lock
python - <<'PY'
from importlib import metadata
import json
from pathlib import Path
from packaging.requirements import Requirement
from packaging.version import Version
import torch
before = json.loads(Path('/opt/reap-base-torch.json').read_text())
after = {'version': str(torch.__version__), 'hip': torch.version.hip, 'module': str(Path(torch.__file__).resolve())}
if before != after:
    raise RuntimeError('The digest-pinned base torch changed unexpectedly')
packages = {}
for line in Path('/tmp/requirements-gpu-hashed.lock').read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    name, expected = line.split()[0].split('==')
    distribution = metadata.distribution(name)
    if distribution.version != expected or not Path(distribution.locate_file('')).resolve().is_relative_to(Path('/opt/reap-python')):
        raise RuntimeError(f'GPU dependency overlay/version mismatch: {name}')
    packages[name] = distribution.version
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        installed = metadata.version(requirement.name)
        if requirement.specifier and Version(installed) not in requirement.specifier:
            raise RuntimeError(f'Unsatisfied GPU dependency: {name} -> {requirement.name}')
import transformers, peft, accelerate, huggingface_hub, safetensors
Path('/opt/reap-python-environment.json').write_text(json.dumps({'torch': after, 'packages': packages, 'core_imports_passed': True, 'GPU_runtime_tested': False}, sort_keys=True))
PY
SHELL

COPY gpu_runtime /opt/reap-gpu/gpu_runtime
COPY containers/gpu/smoke_gpu.py /usr/local/bin/reap-gpu-smoke

RUN useradd --create-home --uid 10001 reap \
    && mkdir -p /workspace/out /opt/models /opt/huggingface \
    && chown -R reap:reap /workspace /opt/models /opt/huggingface

# Useful for control-plane/toy validation without downloading the 7B weights.
FROM runtime-base AS dev-no-weights
USER reap
WORKDIR /workspace
ENTRYPOINT ["python", "-m", "gpu_runtime.server"]
CMD ["--backend", "toy", "--snapshot-root", "/workspace/out/snapshots"]

# Build only on an authorized builder with access to the verified remote model.
# The context must come from prepare_gpu_build_context.py; no model bind mount
# is needed by the resulting image. Ownership is set during COPY, not in a
# later chown layer that would duplicate the large model data.
FROM runtime-base AS release-existing
COPY --chown=10001:10001 model/ /opt/models/REAL-Prover/
COPY model-manifest.json /opt/reap-official-model-manifest.json
COPY build-context-manifest.json /opt/reap-build-context-manifest.json
COPY containers/gpu/download_model.py /tmp/download_model.py
RUN python /tmp/download_model.py \
      --repo "${REAL_PROVER_REPOSITORY}" \
      --revision "${REAL_PROVER_REVISION}" \
      --output /opt/models/REAL-Prover \
      --expected-hidden-size "${REAL_PROVER_HIDDEN_SIZE}" \
      --manifest /opt/reap-official-model-manifest.json \
      --manifest-sha256 "${REAL_PROVER_MANIFEST_SHA256}" \
      --verify-only
USER reap
WORKDIR /workspace
EXPOSE 8760
ENTRYPOINT ["python", "-m", "gpu_runtime.server"]
CMD ["--backend", "real-search", "--gamma", "0.99", "--model-path", "/opt/models/REAL-Prover", "--snapshot-root", "/workspace/out/snapshots"]

# This online stage follows release-existing so an older sequential builder
# stopping at release-existing cannot accidentally download the model first.
FROM runtime-base AS model-download
ENV TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0
COPY containers/gpu/download_model.py /tmp/download_model.py
RUN python /tmp/download_model.py \
      --repo "${REAL_PROVER_REPOSITORY}" \
      --revision "${REAL_PROVER_REVISION}" \
      --output /opt/models/REAL-Prover \
      --expected-hidden-size "${REAL_PROVER_HIDDEN_SIZE}" \
    && chown -R reap:reap /opt/models/REAL-Prover

# Keep the original default/explicit release online behavior compatible.
FROM model-download AS release
ENV TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
USER reap
WORKDIR /workspace
EXPOSE 8760
ENTRYPOINT ["python", "-m", "gpu_runtime.server"]
CMD ["--backend", "real-search", "--gamma", "0.99", "--model-path", "/opt/models/REAL-Prover", "--snapshot-root", "/workspace/out/snapshots"]
