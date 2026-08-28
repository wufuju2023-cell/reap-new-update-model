#!/bin/sh
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
    --target /opt/reap-python --require-hashes --no-deps --no-index --find-links /opt/reap-wheels \
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
