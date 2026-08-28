"""Explicit, audited library hints, never automatic premise retrieval.

Pins use SHA256 of UTF-8 ``str (Lean.Meta.ppExpr info.type)`` in the
wrapper's root command context. A matching hash is not a mathematical audit:
operators must still reject specialized or propositionally equivalent answers.
"""
from __future__ import annotations

import json
import re

VERSION = 'reap.closed-prop.fixed-premise-hints.v1'
NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9']*(?:\.[A-Za-z_][A-Za-z_0-9']*)*\Z")
SHA = re.compile(r'[0-9a-f]{64}\Z')
LIBRARY_MODULE_PREFIXES = ('Mathlib.', 'Init.')


def validate(hints, target):
    if hints is None:
        return []
    if type(hints) is not list or len(hints) > 6:
        raise ValueError('premise_hints must be a list of at most 6 fixed library hints')
    result = []
    for hint in hints:
        if type(hint) is not dict or set(hint) != {'name', 'module', 'type_sha256'}:
            raise ValueError('premise hint requires exactly name, module, type_sha256')
        name, module, type_sha = hint['name'], hint['module'], hint['type_sha256']
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError('premise hint must be a declaration name, not tactic text')
        if name == target or name.split('.')[0] in ('Archive', 'ReapCurriculumWrapper'):
            raise ValueError('target, Archive, and wrapper premise hints forbidden')
        if (not isinstance(module, str) or not NAME.fullmatch(module)
                or not module.startswith(LIBRARY_MODULE_PREFIXES) or 'Archive' in module.split('.')):
            raise ValueError('premise hint requires a pinned Mathlib or Init library module')
        if not isinstance(type_sha, str) or not SHA.fullmatch(type_sha):
            raise ValueError('premise hint type_sha256 must be a lowercase SHA256')
        result.append(dict(hint))
    if len({hint['name'] for hint in result}) != len(result):
        raise ValueError('duplicate premise hints forbidden')
    return result


def lean_prefix(hints, target, lean_name):
    """Shared preflight/execution checks and a fixed Lean Selector.

    Python is already a required CPU runtime dependency; use its SHA256 instead
    of a noncryptographic Lean hash. No statement text is accepted from a caller.
    """
    entries = '#[' + ', '.join(
        '(' + lean_name(hint['name']) + ', ' + json.dumps(hint['module']) + ', '
        + json.dumps(hint['type_sha256']) + ')' for hint in hints) + ']'
    names = '#[' + ', '.join('(' + lean_name(hint['name']) + ')' for hint in hints) + ']'
    return (
        'open Lean in\nrun_cmd Lean.Elab.Command.liftTermElabM do\n'
        f'  let target : Lean.Name := {lean_name(target)}\n'
        '  let closed ← Lean.Meta.deltaExpand (Lean.mkConst target) (· == target)\n'
        '  let negative := Lean.mkApp (Lean.mkConst ``Not) closed\n'
        f'  let hints : Array (Lean.Name × String × String) := {entries}\n'
        '  for (name, expectedModule, expectedHash) in hints do\n'
        '    let info ← Lean.getConstInfo name\n'
        '    unless (match info with | .thmInfo _ => true | _ => false) do\n'
        '      throwError "premise hint must be a library theorem: {name}"\n'
        '    let env ← Lean.getEnv\n'
        '    let some moduleIdx := env.getModuleIdxFor? name\n'
        '      | throwError "source-local premise hint forbidden: {name}"\n'
        '    let actualModule := toString env.header.moduleNames[moduleIdx.toNat]!\n'
        '    unless actualModule == expectedModule do\n'
        '      throwError "premise hint module mismatch: {name}: {actualModule}"\n'
        '    if (← Lean.Meta.isDefEq info.type closed) || (← Lean.Meta.isDefEq info.type negative) then\n'
        '      throwError "target-equivalent premise hint forbidden (definitional equality): {name}"\n'
        '    let statement := toString (← Lean.Meta.ppExpr info.type)\n'
        '    let hashed ← IO.Process.output { cmd := "python3", args := #["-c",\n'
        '      "import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())"] } (some statement)\n'
        '    unless hashed.exitCode == 0 && hashed.stdout.trimAscii.toString == expectedHash do\n'
        '      throwError "premise hint type hash mismatch: {name}: {hashed.stdout.trimAscii.toString}"\n'
        '    logInfo m!"REAP_FIXED_PREMISE_HINT {name} {actualModule} {expectedHash}\\n{statement}"\n\n'
        'def ReapCurriculumWrapper.fixedPremiseSelector : Lean.LibrarySuggestions.Selector := fun _ config => do\n'
        f'  let names : Array Lean.Name := {names}\n'
        '  let suggestions : Array Lean.LibrarySuggestions.Suggestion := names.map fun name =>\n'
        '    { name := name, score := 1.0 }\n'
        '  let suggestions ← suggestions.filterM fun suggestion => config.filter suggestion.name\n'
        '  return suggestions.take config.maxSuggestions\n\n'
        'set_library_suggestions ReapCurriculumWrapper.fixedPremiseSelector\n\n')
