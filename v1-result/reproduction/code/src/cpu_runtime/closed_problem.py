"""Pinned closed-Prop wrappers for proof/disproof curriculum attempts.

Operator-owned declarations must define a qualified name of type Prop. Lean
checks that closedness in a separate preflight, without admitting a proof.
Disproof wraps that entire name in Not; it never rewrites a theorem conclusion.
Preflight is not proof verification and never authorizes training by itself.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

WRAPPER_VERSION = 'reap.closed-prop-wrapper.v1'
SHA = re.compile(r'[0-9a-f]{64}\Z')
NAME = re.compile(r'[A-Za-z_][A-Za-z_0-9\']*(?:\.[A-Za-z_][A-Za-z_0-9\']*)+\Z')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(*, environment_sha256, declarations: bytes, closed_prop_name: str):
    require(isinstance(environment_sha256, str) and SHA.fullmatch(environment_sha256), 'pinned environment required')
    require(isinstance(closed_prop_name, str) and NAME.fullmatch(closed_prop_name), 'qualified closed Prop name required')
    require(isinstance(declarations, bytes) and declarations.startswith(b'import ReapRuntime\n'),
            'declarations must start with the pinned ReapRuntime import using LF')
    text = declarations.decode('utf8')
    require('reapTrainingMCTS' not in text and 'ReapCurriculumWrapper' not in text,
            'declarations must not contain the reserved attempt marker/namespace')
    value = {'schema_version': 'reap.closed-prop.identity.v1', 'environment_sha256': environment_sha256,
             'declarations_sha256': digest(declarations), 'closed_prop_name': closed_prop_name}
    return {**value, 'problem_sha256': digest(value)}


def attempted_identity(problem_sha256: str, polarity: str):
    require(isinstance(problem_sha256, str) and SHA.fullmatch(problem_sha256), 'problem pin required')
    require(polarity in ('prove', 'disprove'), 'polarity must be prove or disprove')
    value = {'schema_version': 'reap.closed-prop.attempt.v1', 'problem_sha256': problem_sha256,
             'polarity': polarity, 'wrapper_version': WRAPPER_VERSION}
    return {**value, 'attempted_prop_sha256': digest(value)}


def prepare(*, environment_sha256: str, declarations: bytes, closed_prop_name: str,
            polarity: str, budget_steps: int, num_samples: int = 2, max_tokens: int = 128,
            unfold_closed_prop: bool = False):
    """Deterministic source bytes; budgets never alter mathematical identity."""
    problem = identity(environment_sha256=environment_sha256, declarations=declarations, closed_prop_name=closed_prop_name)
    attempted = attempted_identity(problem['problem_sha256'], polarity)
    require(type(unfold_closed_prop) is bool, 'unfold_closed_prop must be bool')
    for value, limit, label in ((budget_steps, 1000000, 'steps'), (num_samples, 256, 'samples'), (max_tokens, 4096, 'tokens')):
        require(type(value) is int and 1 <= value <= limit, 'invalid '+label)
    rooted = '_root_.'+closed_prop_name
    proposition = rooted if polarity == 'prove' else '_root_.Not ('+rooted+')'
    lean_name = 'Lean.Name.anonymous'
    for component in closed_prop_name.split('.'):
        lean_name = 'Lean.Name.str ('+lean_name+') '+json.dumps(component)
    prefix = declarations.decode('utf8').rstrip() + '\n\n'
    # An expected `: Prop` alone would accept Lean's Bool-to-Prop coercion.
    # Inspect the exact global declaration type, before any expected-type cast.
    prefix += ('open Lean in\nrun_cmd do\n'
        '  unless (← getCurrNamespace) == Name.anonymous do\n'
        '    throwError "root namespace required"\n'
        '  unless (← Lean.Elab.Command.getScope).varDecls.isEmpty do\n'
        '    throwError "section variables forbidden"\n'
        f'  let name : Lean.Name := {lean_name}\n'
        '  let some info := (← getEnv).find? name | throwError "missing declaration"\n'
        '  unless info.levelParams.isEmpty && info.type == Lean.mkSort Lean.Level.zero do\n'
        '    throwError "closed Prop declaration required"\n\n')
    prefix += ('set_option reap.num_premises 0\n'
               f'set_option reap.num_samples {num_samples}\nset_option reap.max_tokens {max_tokens}\n'
               f'set_option reap.max_steps {budget_steps}\nset_option reap.max_goals 64\n'
               'set_option reap.visit_discount 990\n\n')
    preflight = (prefix + f'def ReapCurriculumWrapper.closed : Prop := {rooted}\n'
                 f'def ReapCurriculumWrapper.attempted : Prop := {proposition}\n'
                 '#check ReapCurriculumWrapper.closed\n#check ReapCurriculumWrapper.attempted\n').encode()
    expected = 'Lean.mkConst target'
    if polarity == 'disprove':
        expected = 'Lean.mkApp (Lean.mkConst (Lean.Name.str Lean.Name.anonymous "Not")) (Lean.mkConst target)'
    postcondition = ('\nopen Lean in\nrun_cmd do\n'
        f'  let target : Lean.Name := {lean_name}\n'
        '  let theoremName := Lean.Name.str (Lean.Name.str Lean.Name.anonymous "ReapCurriculumWrapper") "attempt"\n'
        '  let info ← getConstInfo theoremName\n'
        f'  let expected := {expected}\n'
        '  unless info.levelParams.isEmpty && info.type == expected do\n'
        '    throwError "attempt theorem type differs from the complete closed proposition"\n')
    # Expose the statement to the policy without supplying any proof tactic.
    # The theorem type and complete-Not postcondition remain unchanged.
    prelude = f'  unfold {rooted}\n' if unfold_closed_prop else ''
    execution = (prefix + f'theorem ReapCurriculumWrapper.attempt : {proposition} := by\n'+prelude+'  reapTrainingMCTS\n'+postcondition).encode()
    result = {'problem': problem, 'attempted': attempted, 'budget_steps': budget_steps,
            'declarations': declarations, 'num_samples': num_samples, 'max_tokens': max_tokens,
            'preflight_source': preflight, 'execution_source': execution,
            'preflight_source_sha256': digest(preflight), 'execution_source_sha256': digest(execution),
            'theorem': 'ReapCurriculumWrapper.attempt'}
    if unfold_closed_prop:
        result['unfold_closed_prop'] = True
    return result


def compile_preflight(prepared, *, project_dir: Path, output_dir: Path, lean_bin='lake'):
    """Compile only closed propositions/options; no search, proof or admission.

    The caller must run in the independently pinned environment. Its declared
    environment identity is recorded, not magically measured by this function.
    Output is exclusive; failures are retained and never turned into admission.
    """
    expected = prepare(environment_sha256=prepared['problem']['environment_sha256'],
        declarations=prepared['declarations'], closed_prop_name=prepared['problem']['closed_prop_name'],
        polarity=prepared['attempted']['polarity'], budget_steps=prepared['budget_steps'],
        num_samples=prepared['num_samples'], max_tokens=prepared['max_tokens'],
        unfold_closed_prop=prepared.get('unfold_closed_prop', False))
    require(prepared == expected, 'prepared wrapper/source/identity changed')
    project_dir, output_dir = Path(project_dir).resolve(), Path(output_dir).absolute()
    require(not any(p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction())
                    for p in (output_dir, *output_dir.parents)), 'preflight output links refused')
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, field in (('preflight.lean', 'preflight_source'), ('attempt.lean', 'execution_source')):
        with (output_dir/name).open('xb') as stream:
            stream.write(prepared[field])
    command = [lean_bin, 'env', 'lean', str(output_dir/'preflight.lean')]
    with (output_dir/'stdout.log').open('xb') as stdout, (output_dir/'stderr.log').open('xb') as stderr:
        process = subprocess.run(command, cwd=project_dir, stdout=stdout, stderr=stderr)
    require((output_dir/'preflight.lean').read_bytes() == prepared['preflight_source']
            and (output_dir/'attempt.lean').read_bytes() == prepared['execution_source'],
            'source changed during preflight; no admission receipt')
    receipt = {'schema_version': 'reap.closed-prop.preflight.v1', 'returncode': process.returncode,
        'scope': 'closed_propositions_and_search_options_only', 'proof_verified': False,
        'environment_sha256': prepared['problem']['environment_sha256'],
        'problem_sha256': prepared['problem']['problem_sha256'],
        'attempted_prop_sha256': prepared['attempted']['attempted_prop_sha256'],
        'execution_source_sha256': prepared['execution_source_sha256'],
        'preflight_source_sha256': prepared['preflight_source_sha256'],
        'command': command, 'stdout_sha256': digest((output_dir/'stdout.log').read_bytes()),
        'stderr_sha256': digest((output_dir/'stderr.log').read_bytes())}
    with (output_dir/'receipt.json').open('xb') as stream:
        stream.write(canonical(receipt))
    require(process.returncode == 0, 'closed proposition preflight failed; receipt retained')
    return {'execution_source_sha256': prepared['execution_source_sha256'],
        'compilation_receipt_sha256': digest(receipt),
        'attempted_prop_sha256': prepared['attempted']['attempted_prop_sha256'],
        'budget_steps': prepared['budget_steps']}
