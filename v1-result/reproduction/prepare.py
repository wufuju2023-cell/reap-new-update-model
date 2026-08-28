"""Prepare verified code and answer-free inputs offline in a new directory.

No network, model loading, training, container startup, or existing-output reuse.
Run --check to verify a prepared workspace without executing any experiment.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SPEC = importlib.util.spec_from_file_location('_reproduction_package_check', PACKAGE / 'source/current/check_package.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
INPUT_NAMES = (
    '01-CoupledOddSquare.lean', '02-AffineAccumulator.lean',
    '03-ScaledTriangular.lean', '04-DifferenceInvariant.lean',
    '05-CubeAccumulator.lean',
)


def payload():
    """Verify both readable entry copies against the existing frozen archives."""
    code = HERE / 'code'
    original = PACKAGE / 'source/current'
    for name in ('manifest.json', 'source.zip'):
        if (code / name).read_bytes() != (original / name).read_bytes():
            raise ValueError('reproduction code differs from the frozen source: ' + name)
    manifest = CHECK.strict_json((code / 'manifest.json').read_bytes())
    source = CHECK.archive_values(manifest, (code / 'source.zip').read_bytes(), 'zip')
    readable = {p.relative_to(code / 'src').as_posix(): p for p in (code / 'src').rglob('*') if p.is_file()}
    expected_readable = {n: b for n, b in source.items() if not n.endswith('.md')}
    if set(readable) != set(expected_readable):
        raise ValueError('readable code file set differs from frozen source')
    for name, path in readable.items():
        CHECK.VERIFY.no_links(path)
        if path.read_bytes() != expected_readable[name]:
            raise ValueError('readable code differs from frozen source: ' + name)
    historical = PACKAGE / 'evidence/multiround'
    input_manifest = CHECK.strict_json((historical / 'raw-evidence-manifest.json').read_bytes())
    old = CHECK.archive_values(input_manifest, (historical / 'raw-evidence.tar.gz').read_bytes(), 'tar')
    inputs = {name: old['inputs/' + name] for name in INPUT_NAMES}
    for name, raw in inputs.items():
        if (HERE / 'inputs' / name).read_bytes() != raw:
            raise ValueError('input differs from the original search input: ' + name)
    return {'source/' + n: b for n, b in source.items()} | {'inputs/' + n: b for n, b in inputs.items()}


def inventory(values):
    return {n: {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()} for n, b in sorted(values.items())}


def check(output):
    output = CHECK.VERIFY.no_links(output)
    expected = inventory(payload())
    actual = {}
    for folder in ('source', 'inputs'):
        for path in (output / folder).rglob('*'):
            CHECK.VERIFY.no_links(path)
            if path.is_file():
                raw = path.read_bytes()
                actual[path.relative_to(output).as_posix()] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if actual != expected:
        raise ValueError('prepared code/input content or file set differs')
    receipt = CHECK.strict_json((output / 'prepared.json').read_bytes())
    if receipt.get('files') != expected or receipt.get('experiment_executed') is not False:
        raise ValueError('preparation receipt differs')
    return {'ok': True, 'source_files': len(expected) - len(INPUT_NAMES), 'input_files': len(INPUT_NAMES),
            'experiment_executed': False, 'output': str(output)}


def prepare(output):
    output = CHECK.VERIFY.no_links(output)
    if output.exists():
        raise FileExistsError('use a new output directory; --check inspects an existing preparation')
    if not output.parent.is_dir():
        raise ValueError('create the parent directory first')
    if output.is_relative_to(PACKAGE):
        raise ValueError('prepare outside the report directory')
    values = payload()  # All archives/inputs are checked before creating output.
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '-prepare-', dir=output.parent))
    try:
        for name, raw in values.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
        with (stage / 'prepared.json').open('x', encoding='utf8') as stream:
            json.dump({'schema': 'reap.reproduction-preparation.v1', 'experiment_executed': False,
                       'files': inventory(values)}, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        check(stage)
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    except BaseException:
        # Preserve this owned temporary directory for inspection; no final output is committed.
        print(json.dumps({'incomplete_preparation': str(stage), 'experiment_executed': False}), flush=True)
        raise
    return check(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = check(args.output) if args.check else prepare(args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
