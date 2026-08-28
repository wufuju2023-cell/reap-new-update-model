"""Deterministic, restricted target transformations checked by generated Lean.

Generating bytes does not verify a relationship or prove a proposition. Compile
these declarations with the pinned environment (e.g. closed_problem preflight)
to check the Expr-derived relationship. Proof/disproof still requires a separate
kernel-checked theorem and source-bound receipt. No model, search, or I/O here.
"""
from __future__ import annotations

from copy import deepcopy
import json

from .closed_problem import identity, digest, require

SCHEMA_VERSION = 'reap.target-variant.v1'
GENERATOR_VERSION = 'reap.target-variant-generator.v1'
RELATION_THEOREM = 'ReapTargetVariantRelation.witness'
MAX_NAT_INSTANCE = 1_000_000
MAX_DECLARATION_BYTES = 4*1024*1024


def _lean_name(name):
    value = 'Lean.Name.anonymous'
    for component in name.split('.'):
        value = 'Lean.Name.str ('+value+') '+json.dumps(component)
    return value


def _transform(value):
    require(type(value) is dict and type(value.get('kind')) is str, 'explicit transform object required')
    if value['kind'] == 'nat_forall_instance':
        require(set(value) == {'kind', 'value'} and type(value['value']) is int
                and 0 <= value['value'] <= MAX_NAT_INSTANCE, 'bounded nonnegative Nat instance required')
    else:
        require(value['kind'] in ('and_left', 'and_right') and set(value) == {'kind'},
                'unsupported transform or extra fields')
    return deepcopy(value)


def generate_variant(*, environment_sha256: str, declarations: bytes,
                     target_name: str, variant_name: str, transform: dict) -> dict:
    """Append a checked Expr-derived Prop definition, preserving input bytes.

    Inputs are operator-owned Lean declarations, not a sandbox. Root namespace,
    section variables, exact declaration type, and supported Expr shape are
    checked by Lean when the resulting file is compiled. Never silently rewrite
    unsupported targets. Changing a target, environment, name, or transform
    changes the corresponding identity/relationship pins.
    """
    require(type(declarations) is bytes and len(declarations) <= MAX_DECLARATION_BYTES,
            'bounded declaration bytes required')
    target = identity(environment_sha256=environment_sha256, declarations=declarations, closed_prop_name=target_name)
    # Reuse the same qualified-name checks without guessing Lean syntax by text.
    identity(environment_sha256=environment_sha256, declarations=declarations, closed_prop_name=variant_name)
    require(target_name != variant_name, 'variant must use a distinct qualified name')
    rule = _transform(transform)
    if rule['kind'] == 'nat_forall_instance':
        derive = (
            '    let .forallE _ domain body _ := normalized\n'
            '      | throwError "target variant requires a first-layer forall Nat"\n'
            '    unless (← Lean.Meta.whnf domain) == Lean.mkConst ``Nat do\n'
            '      throwError "target variant forall domain must be Nat"\n'
            f'    let derived := body.instantiate1 (Lean.mkNatLit {rule["value"]})\n')
    else:
        index = 0 if rule['kind'] == 'and_left' else 1
        derive = (
            '    unless normalized.isAppOfArity ``And 2 do\n'
            '      throwError "target variant requires a first-layer And"\n'
            f'    let derived := normalized.getAppArgs[{index}]!\n')
    recipe = {'generator_version': GENERATOR_VERSION, 'target_problem_sha256': target['problem_sha256'],
              'variant_name': variant_name, 'transform': rule}
    # Original bytes remain an exact prefix. No source text substitution, Bool
    # coercion, parsing of printed goals, or hand-written family membership.
    addition = ('\n\n-- '+json.dumps(recipe, ensure_ascii=True, sort_keys=True, separators=(',', ':'))+'\n'
        'open Lean in\nrun_cmd do\n'
        '  unless (← getCurrNamespace) == Name.anonymous do\n'
        '    throwError "root namespace required for target variant"\n'
        '  unless (← Lean.Elab.Command.getScope).varDecls.isEmpty do\n'
        '    throwError "section variables forbidden for target variant"\n'
        f'  let targetName : Lean.Name := {_lean_name(target_name)}\n'
        f'  let variantName : Lean.Name := {_lean_name(variant_name)}\n'
        '  let some info := (← getEnv).find? targetName\n'
        '    | throwError "target variant source declaration missing"\n'
        '  unless info.levelParams.isEmpty && info.type == Lean.mkSort Lean.Level.zero do\n'
        '    throwError "target variant requires an exact closed Prop declaration"\n'
        '  if (← getEnv).contains variantName then\n'
        '    throwError "target variant name already exists"\n'
        '  Lean.Elab.Command.liftTermElabM do\n'
        '    let normalized ← Lean.Meta.whnf (Lean.mkConst targetName)\n'+derive+
        '    unless !derived.hasFVar && !derived.hasMVar && !derived.hasLooseBVars do\n'
        '      throwError "target variant derived expression is not closed"\n'
        '    unless (← Lean.Meta.inferType derived) == Lean.mkSort Lean.Level.zero do\n'
        '      throwError "target variant derived expression must have exact type Prop"\n'
        '    Lean.addAndCompile (.defnDecl {\n'
        '      name := variantName\n'
        '      levelParams := []\n'
        '      type := Lean.mkSort Lean.Level.zero\n'
        '      value := derived\n'
        '      hints := .abbrev\n'
        '      safety := .safe })\n')
    generated = declarations + addition.encode('utf8')
    variant = identity(environment_sha256=environment_sha256, declarations=generated, closed_prop_name=variant_name)
    relation = {'schema_version': 'reap.target-variant-relation.v1', 'generator_version': GENERATOR_VERSION,
                'target_problem_sha256': target['problem_sha256'], 'variant_problem_sha256': variant['problem_sha256'],
                'transform': rule}
    proof = ('exact @h '+str(rule['value']) if rule['kind'] == 'nat_forall_instance' else
             'exact h.1' if rule['kind'] == 'and_left' else 'exact h.2')
    relation_source = generated + (
        f'\ntheorem {RELATION_THEOREM} : _root_.{target_name} → _root_.{variant_name} := by\n'
        '  intro h\n  '+proof+'\n'
        f'#print axioms {RELATION_THEOREM}\n').encode('utf8')
    return {'schema_version': SCHEMA_VERSION, 'generator_version': GENERATOR_VERSION,
            'target_identity': target, 'variant_identity': variant, 'transform': rule,
            'transformation_sha256': digest(relation), 'declarations': generated,
            'relation_source': relation_source, 'relation_theorem': RELATION_THEOREM,
            'relation_source_sha256': digest(relation_source)}
