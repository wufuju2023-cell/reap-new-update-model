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

from cpu_runtime import premise_hints as fixed_hints
from cpu_runtime import student_proof_library

WRAPPER_VERSION = 'reap.closed-prop-wrapper.student-local-facts.v1'
DEFINITION_VISIBILITY_VERSION = 'reap.closed-prop.definition-visibility.v1'
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


def _lean_name(name: str) -> str:
    value = 'Lean.Name.anonymous'
    for component in name.split('.'):
        value = 'Lean.Name.str ('+value+') '+json.dumps(component)
    return value


def _definition_visibility(target: str, definitions: list[str]) -> str:
    """A deterministic Meta transformation, not a supplied mathematical proof.

    Only safe nonrecursive definitions are expanded. The same function runs in
    the compilation preflight and immediately before the student's search.
    """
    names = '#['+', '.join('('+_lean_name(name)+')' for name in definitions)+']'
    return (f'def ReapCurriculumWrapper.expandDefinitions (original : Lean.Expr) : Lean.MetaM Lean.Expr := do\n'
        f'  let target : Lean.Name := {_lean_name(target)}\n'
        f'  let names : Array Lean.Name := {names}\n'
        '  for name in names do\n'
        '    let some (.defnInfo info) := (← Lean.getEnv).find? name\n'
        '      | throwError "visibility entry must be a definition: {name}"\n'
        '    unless info.safety == .safe do\n'
        '      throwError "unsafe visibility definition forbidden: {name}"\n'
        '    if (← Lean.Meta.isRecursiveDefinition name) then\n'
        '      throwError "recursive visibility definition forbidden: {name}"\n'
        '    if (← Lean.Meta.isProp info.type) then\n'
        '      throwError "proof-valued visibility definition forbidden: {name}"\n'
        '  let mut expanded ← Lean.Meta.deltaExpand original (· == target)\n'
        '  for name in names do\n'
        '    unless expanded.getUsedConstants.contains name do\n'
        '      throwError "visibility definition absent at its ordered expansion step: {name}"\n'
        '    expanded ← Lean.Meta.deltaExpand expanded (· == name)\n'
        '  for name in expanded.getUsedConstants do\n'
        '    if ((← Lean.getEnv).getModuleIdxFor? name).isNone then\n'
        '      if let some (.defnInfo _) := (← Lean.getEnv).find? name then\n'
        '        throwError "unexpanded source-local definition remains: {name}"\n'
        '  unless (← Lean.Meta.isDefEq original expanded) do\n'
        '    throwError "visibility transformation changed the complete proposition"\n'
        '  unless (← Lean.Meta.inferType expanded) == Lean.mkSort Lean.Level.zero do\n'
        '    throwError "visibility result must have exact type Prop"\n'
        '  return expanded\n\n')


def prepare(*, environment_sha256: str, declarations: bytes, closed_prop_name: str,
            polarity: str, budget_steps: int, num_samples: int = 2, max_tokens: int = 128,
            max_nodes: int | None = None,
            unfold_closed_prop: bool = False, unfold_definitions: list[str] | None = None,
            premise_hints: list[dict] | None = None):
    """Deterministic source bytes; budgets never alter mathematical identity."""
    problem = identity(environment_sha256=environment_sha256, declarations=declarations, closed_prop_name=closed_prop_name)
    attempted = attempted_identity(problem['problem_sha256'], polarity)
    require(type(unfold_closed_prop) is bool, 'unfold_closed_prop must be bool')
    definitions = [] if unfold_definitions is None else unfold_definitions
    require(type(definitions) is list and len(definitions) <= 16,
            'unfold_definitions must be a list of at most 16 qualified names')
    require(all(isinstance(name, str) and NAME.fullmatch(name) for name in definitions),
            'qualified definition names required; tactic text forbidden')
    require(len(set(definitions)) == len(definitions), 'duplicate unfold definition names forbidden')
    require(closed_prop_name not in definitions, 'closed target is already expanded separately')
    require(not definitions or unfold_closed_prop, 'definition visibility requires unfold_closed_prop=True')
    definitions = list(definitions)
    student_mode = bool(premise_hints) and any(type(h) is dict and h.get('kind') == student_proof_library.KIND for h in premise_hints)
    hints = (student_proof_library.validate(premise_hints, closed_prop_name, declarations)
             if student_mode else fixed_hints.validate(premise_hints, closed_prop_name))
    for value, limit, label in ((budget_steps, 1000000, 'steps'), (num_samples, 256, 'samples'), (max_tokens, 4096, 'tokens')):
        require(type(value) is int and 1 <= value <= limit, 'invalid '+label)
    require(max_nodes is None or (type(max_nodes) is int and 1 <= max_nodes <= 1000000),
            'max_nodes must be an integer from 1 to 1000000')
    # Omission preserves the historical wrapper bytes. Reap checks tree size
    # before each MCTS step, so one expansion may cross this threshold.
    node_limit = 64 if max_nodes is None else max_nodes
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
    prefix += (f'set_option reap.num_premises {len(hints)}\n'
               f'set_option reap.num_samples {num_samples}\nset_option reap.max_tokens {max_tokens}\n'
               f'set_option reap.max_steps {budget_steps}\nset_option reap.max_goals {node_limit}\n'
               'set_option reap.visit_discount 990\n\n')
    if definitions:
        prefix += _definition_visibility(closed_prop_name, definitions)
    if hints:
        prefix += (student_proof_library if student_mode else fixed_hints).lean_prefix(hints, closed_prop_name, _lean_name)
    # CPU15 opt-in is isolated by this frozen source tree and wrapper identity.
    # Introduce only the complete, already validated theorem constants. Do not
    # instantiate arguments, destruct existentials, choose witnesses, or supply
    # proof-search actions. Verification/replay replace only reapTrainingMCTS,
    # so this exact environment prefix is retained in all three paths.
    local_facts = ''.join(
        f'  have student_fact_{i} := @_root_.{hint["name"]}\n'
        for i, hint in enumerate(hints, 1)) if student_mode else ''
    preflight = (prefix + f'def ReapCurriculumWrapper.closed : Prop := {rooted}\n'
                 f'def ReapCurriculumWrapper.attempted : Prop := {proposition}\n'
                 '#check ReapCurriculumWrapper.closed\n#check ReapCurriculumWrapper.attempted\n').encode()
    if local_facts:
        # Type-check the loader without attempting or admitting the real goal.
        preflight += ('\nexample : True := by\n'+local_facts+'  trivial\n').encode()
    expected = 'Lean.mkConst target'
    if polarity == 'disprove':
        expected = 'Lean.mkApp (Lean.mkConst (Lean.Name.str Lean.Name.anonymous "Not")) (Lean.mkConst target)'
    if definitions:
        preflight += ('\nopen Lean in\nrun_cmd Lean.Elab.Command.liftTermElabM do\n'
            f'  let target : Lean.Name := {lean_name}\n'
            f'  let expanded ← ReapCurriculumWrapper.expandDefinitions ({expected})\n'
            '  logInfo m!"REAP_DEFINITION_VISIBILITY_GOAL\\n{← Lean.Meta.ppExpr expanded}"\n').encode()
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
    if definitions:
        prelude = ('  run_tac\n'
            '    let goal ← Lean.Elab.Tactic.getMainGoal\n'
            '    let expanded ← goal.withContext do\n'
            '      ReapCurriculumWrapper.expandDefinitions (← goal.getType)\n'
            '    let next ← goal.change expanded\n'
            '    Lean.Elab.Tactic.replaceMainGoal [next]\n')
    execution = (prefix + f'theorem ReapCurriculumWrapper.attempt : {proposition} := by\n'+prelude+local_facts+'  reapTrainingMCTS\n'+postcondition).encode()
    result = {'problem': problem, 'attempted': attempted, 'budget_steps': budget_steps,
            'declarations': declarations, 'num_samples': num_samples, 'max_tokens': max_tokens,
            'preflight_source': preflight, 'execution_source': execution,
            'preflight_source_sha256': digest(preflight), 'execution_source_sha256': digest(execution),
            'theorem': 'ReapCurriculumWrapper.attempt'}
    if max_nodes is not None:
        result['max_nodes'] = max_nodes
    if unfold_closed_prop:
        result['unfold_closed_prop'] = True
    if definitions:
        result['unfold_definitions'] = definitions
        result['definition_visibility_version'] = DEFINITION_VISIBILITY_VERSION
    if hints:
        result['premise_hints'] = hints
        result['premise_hints_version'] = student_proof_library.VERSION if student_mode else fixed_hints.VERSION
    if local_facts:
        result['student_local_facts'] = {'version': WRAPPER_VERSION,
            'names': [h['name'] for h in hints], 'prelude_sha256': digest(local_facts.encode()),
            'uninstantiated': True, 'model_actions': False, 'training_targets': False}
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
        max_nodes=prepared.get('max_nodes'),
        unfold_closed_prop=prepared.get('unfold_closed_prop', False),
        unfold_definitions=prepared.get('unfold_definitions'),
        premise_hints=prepared.get('premise_hints'))
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
    if 'max_nodes' in prepared:
        receipt['max_nodes'] = prepared['max_nodes']
    if prepared.get('premise_hints_version') == student_proof_library.VERSION:
        receipt['student_proof_library'] = {'version': student_proof_library.VERSION,
            'hints': prepared['premise_hints'], 'automatic_retrieval': False,
            'scope': 'exact accepted student actions; full local theorem proof recompiled; source/type/provenance pins and axiom whitelist'}
        if prepared.get('student_local_facts'):
            receipt['student_local_facts'] = prepared['student_local_facts']
    elif prepared.get('premise_hints'):
        receipt['premise_hints'] = {'version': fixed_hints.VERSION,
            'hints': prepared['premise_hints'], 'automatic_retrieval': False,
            'scope': 'module_and_pretty_printed_type_pin_and_definitional_target_equality_check',
            'manual_non_equivalence_audit_required': True}
    if prepared.get('unfold_definitions'):
        receipt['definition_visibility'] = {'version': DEFINITION_VISIBILITY_VERSION,
            'unfold_definitions': prepared['unfold_definitions'], 'supplied_proof': False,
            'scope': 'exact_definitional_expansion_and_remaining_source_local_definition_check'}
    with (output_dir/'receipt.json').open('xb') as stream:
        stream.write(canonical(receipt))
    require(process.returncode == 0, 'closed proposition preflight failed; receipt retained')
    return {'execution_source_sha256': prepared['execution_source_sha256'],
        'compilation_receipt_sha256': digest(receipt),
        'attempted_prop_sha256': prepared['attempted']['attempted_prop_sha256'],
        'budget_steps': prepared['budget_steps']}
