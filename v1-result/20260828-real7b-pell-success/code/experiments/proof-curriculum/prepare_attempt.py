"""Prepare a pinned hard-problem attempt locally; never contact a GPU.

This prepares source, not mathematical admission or curriculum membership.
Compile preflight.lean in the pinned CPU environment before running run_online.sh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cpu_runtime import closed_problem as closed

HERE = Path(__file__).resolve().parent


def profile(name: str, budget_file: Path = HERE / 'budgets.json') -> dict:
    config = json.loads(budget_file.read_text(encoding='utf8'))
    if config.get('schema_version') not in ('reap.proof-curriculum.budgets.v1', 'reap.proof-curriculum.budgets.v2') or config.get('gamma') != 0.99:
        raise ValueError('unsupported budget contract')
    result = config['profiles'][name]
    fields = {'search_steps', 'num_samples', 'max_tokens', 'max_updates', 'min_checkpoint_interval'}
    if config['schema_version'] == 'reap.proof-curriculum.budgets.v2':
        fields.add('max_nodes')
    if set(result) != fields or any(type(v) is not int or v <= 0 for v in result.values()):
        raise ValueError('budget fields must be explicit positive integers')
    if result['max_updates'] * result['min_checkpoint_interval'] > result['search_steps']:
        raise ValueError('update cap exceeds scheduled search checkpoints')
    if 'max_nodes' in result and not result['search_steps'] <= result['max_nodes'] <= 1000000:
        raise ValueError('node budget must cover the step budget and be at most 1000000')
    return result


RUNNER = '''#!/usr/bin/env bash
set -euo pipefail
: "${SESSION_ID:?unique session ID required}"
: "${PROJECT_DIR:?pinned Lean project required}"
: "${OUTPUT_DIR:?new experiment output root required}"
: "${GPU_BASE_URL:?local bridge or authorized GPU endpoint required}"
: "${TTT_SOURCE_ROOT:?current project source root required}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$TTT_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 "$HERE/verify_inputs.py"
args=(--session-id "$SESSION_ID" --project-dir "$PROJECT_DIR"
  --theorem-file "$HERE/attempt.lean" --output-dir "$OUTPUT_DIR"
  --gpu-base-url "$GPU_BASE_URL" --gamma 0.99
  --max-updates __UPDATES__ --min-checkpoint-interval __INTERVAL__
  --max-request-body-bytes __BODY__
  --http-timeout-seconds __HTTP__ --barrier-timeout-seconds __BARRIER__
  --experience-candidate)
if [[ -n "${EXPERIENCE_ID:-}" ]]; then
  : "${EXPERIENCE_WEIGHTS_SHA256:?inherited weights pin required}"
  : "${EXPERIENCE_SNAPSHOT_SHA256:?source snapshot pin required}"
  args+=(--experience-id "$EXPERIENCE_ID"
    --experience-weights-sha256 "$EXPERIENCE_WEIGHTS_SHA256"
    --experience-snapshot-sha256 "$EXPERIENCE_SNAPSHOT_SHA256")
elif [[ -n "${EXPERIENCE_WEIGHTS_SHA256:-}${EXPERIENCE_SNAPSHOT_SHA256:-}" ]]; then
  echo 'Incomplete experience source; no work started' >&2
  exit 2
fi
exec python3 -B -m cpu_runtime.online_ttt "${args[@]}"
'''

# A receipt is an operator-owned local assertion, not authentication of remote files.
VERIFY = '''from pathlib import Path
import hashlib
import json
root = Path(__file__).resolve().parent
plan = json.loads((root/'attempt-plan.json').read_bytes())
for name, pin in plan['files_sha256'].items():
    path = root/name
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != pin:
        raise SystemExit('Prepared file changed: '+name)
receipt = json.loads((root/'preflight-receipt.json').read_bytes())
if not (receipt.get('schema_version') == 'reap.proof-curriculum.preflight.v1'
        and type(receipt.get('returncode')) is int and receipt['returncode'] == 0
        and receipt.get('scope') == 'statement_compilation_only'
        and receipt.get('plan_sha256') == hashlib.sha256((root/'attempt-plan.json').read_bytes()).hexdigest()
        and receipt.get('environment_sha256') == plan['environment_sha256']):
    raise SystemExit('Exact successful preflight receipt required')
for name in ('stdout.log', 'stderr.log'):
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != receipt['streams_sha256'][name]:
        raise SystemExit('Preflight stream changed: '+name)
'''


def prepare(*, declarations: Path, prop: str, environment_sha256: str, budget: str,
            output: Path, unfold_definitions: list[str] | None = None) -> dict:
    selected = profile(budget)
    transport = json.loads((HERE/'budgets.json').read_text(encoding='utf8'))['transport']
    if transport != {'max_request_body_bytes': 262144, 'http_timeout_seconds': 1800,
                      'barrier_timeout_seconds': 1860, 'policy_scoring': 'tokenwise'}:
        raise ValueError('unreviewed transport/scoring configuration')
    raw = declarations.read_bytes()
    # These are operator-reviewed source files, not a sandbox for untrusted Lean.
    if re.search(r'\b(sorry|admit|axiom)\b', raw.decode('utf8')):
        raise ValueError('unproved declarations are not permitted in student input')
    prepared = closed.prepare(environment_sha256=environment_sha256, declarations=raw,
        closed_prop_name=prop, polarity='prove', budget_steps=selected['search_steps'],
        num_samples=selected['num_samples'], max_tokens=selected['max_tokens'],
        max_nodes=selected.get('max_nodes'),
        unfold_closed_prop=True, unfold_definitions=unfold_definitions)
    files = {
        'declarations.lean': raw,
        'preflight.lean': prepared['preflight_source'],
        'attempt.lean': prepared['execution_source'],
        'run_online.sh': RUNNER.replace('__UPDATES__', str(selected['max_updates'])).replace(
            '__INTERVAL__', str(selected['min_checkpoint_interval'])).replace(
            '__BODY__', str(transport['max_request_body_bytes'])).replace(
            '__HTTP__', str(transport['http_timeout_seconds'])).replace(
            '__BARRIER__', str(transport['barrier_timeout_seconds'])).encode(),
        'verify_inputs.py': VERIFY.encode(),
    }
    plan = {'schema_version': 'reap.proof-curriculum.attempt-plan.v1',
        'environment_sha256': environment_sha256, 'budget_name': budget, 'budget': selected,
        'transport': transport,
        'problem': prepared['problem'], 'theorem': prepared['theorem'],
        'mode': 'online-search-visit-backup-v1', 'gamma': 0.99,
        'mathematical_admission': 'requires_separate_teacher_review',
        'proof_verified': False, 'gpu_started': False,
        'files_sha256': {name: closed.digest(data) for name, data in files.items()}}
    if unfold_definitions:
        plan['unfold_definitions'] = list(prepared['unfold_definitions'])
    output = Path(output).absolute()
    for path in (output, *output.parents):
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise ValueError('output links refused')
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        with (output/name).open('xb') as stream:
            stream.write(data)
    with (output/'attempt-plan.json').open('xb') as stream:
        stream.write(closed.canonical(plan))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declarations', required=True, type=Path)
    parser.add_argument('--prop', required=True)
    parser.add_argument('--environment-sha256', required=True)
    parser.add_argument('--budget', choices=('probe', 'main', 'deep'), default='main')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--unfold-definition', action='append', default=None,
        help='Explicit qualified nonrecursive definition to expose; repeat in dependency order.')
    args = parser.parse_args()
    plan = prepare(declarations=args.declarations, prop=args.prop,
        environment_sha256=args.environment_sha256, budget=args.budget, output=args.output,
        unfold_definitions=args.unfold_definition)
    print(json.dumps({'output': str(args.output), 'problem_sha256': plan['problem']['problem_sha256'],
        'preflight_required': True, 'gpu_started': False}))


if __name__ == '__main__':
    main()
