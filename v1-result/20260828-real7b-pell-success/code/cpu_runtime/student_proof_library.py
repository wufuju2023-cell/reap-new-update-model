"""Explicit verified student proofs, separate from basic imported-library hints."""
import hashlib
import json
from pathlib import Path
import re

VERSION = 'reap.closed-prop.student-proof-library.v1'
KIND = 'verified_student_proof'
NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9']*(?:\.[A-Za-z_][A-Za-z_0-9']*)+\Z")
SHA = re.compile(r'[0-9a-f]{64}\Z')
AXIOMS = {'propext', 'Classical.choice', 'Quot.sound'}
FIELDS = {'kind', 'name', 'proved_declaration', 'type_sha256', 'declarations_sha256',
          'acceptance_path', 'acceptance_sha256', 'proof_path', 'proof_sha256',
          'session_id', 'action_body_sha256'}

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def validate(hints, target, declarations):
    require(type(hints) is list and 1 <= len(hints) <= 4, 'bounded explicit student proof list required')
    result = []
    for item in hints:
        require(type(item) is dict and set(item) == FIELDS and item['kind'] == KIND, 'student proof metadata schema mismatch')
        for field in ('name', 'proved_declaration'):
            require(isinstance(item[field], str) and NAME.fullmatch(item[field]), 'qualified student proof name required')
            require(item[field] != target and not item[field].startswith(('Archive.', 'ReapCurriculumWrapper.')), 'target/reserved student premise forbidden')
        for field in ('type_sha256', 'declarations_sha256', 'acceptance_sha256', 'proof_sha256', 'action_body_sha256'):
            require(isinstance(item[field], str) and SHA.fullmatch(item[field]), 'student proof SHA required')
        require(digest(declarations) == item['declarations_sha256'], 'student declaration source changed')
        acceptance_path, proof_path = Path(item['acceptance_path']), Path(item['proof_path'])
        require(acceptance_path.is_absolute() and proof_path.is_absolute(), 'absolute proof provenance paths required')
        require(acceptance_path.name == 'accepted.json' and proof_path.name == 'proof.lean' and acceptance_path.parent == proof_path.parent, 'exact accepted proof pair required')
        require(not acceptance_path.is_symlink() and not proof_path.is_symlink(), 'proof source symlinks refused')
        accepted_bytes, proof = acceptance_path.read_bytes(), proof_path.read_bytes()
        require(digest(accepted_bytes) == item['acceptance_sha256'] and digest(proof) == item['proof_sha256'], 'accepted proof file changed')
        accepted = json.loads(accepted_bytes)
        require(accepted.get('schema_version') == 'reap.reproduction.independent-lean.v1'
                and accepted.get('passed') is True and accepted.get('lean_returncode') == 0
                and accepted.get('session_id') == item['session_id']
                and accepted.get('proof_sha256') == item['proof_sha256']
                and set(accepted.get('axioms', [])) <= AXIOMS, 'genuine independent student acceptance required')
        header = ('theorem ReapCurriculumWrapper.attempt : _root_.' + item['proved_declaration'] + ' := by\n').encode()
        require(proof.count(header) == 1, 'accepted theorem binding missing')
        body = proof.split(header, 1)[1].split(b'\nopen Lean in\nrun_cmd do\n', 1)[0]
        first, actions = body.split(b'\n', 1)
        require(first == ('  unfold _root_.' + item['proved_declaration']).encode(), 'only deterministic closed-goal unfold may be omitted')
        require(digest(actions) == item['action_body_sha256'], 'student action bytes changed')
        begin = ('-- STUDENT_VERIFIED_BEGIN ' + item['name'] + '\n').encode()
        end = ('-- STUDENT_VERIFIED_END ' + item['name'] + '\n').encode()
        require(declarations.count(begin) == declarations.count(end) == 1, 'unique student proof source markers required')
        block = declarations.split(begin, 1)[1].split(end, 1)[0]
        require(block.startswith(('theorem ' + item['name'] + ' :\n').encode()), 'student theorem source binding mismatch')
        require(block.count(b' := by\n') == 1 and block.split(b' := by\n', 1)[1] == actions, 'only exact verified student proof actions are permitted')
        result.append(dict(item))
    require(len({item['name'] for item in result}) == len(result), 'duplicate student proofs')
    return result

def lean_prefix(hints, target, lean_name):
    entries = '#[' + ', '.join('(' + lean_name(h['name']) + ', ' + lean_name(h['proved_declaration']) + ', ' + json.dumps(h['type_sha256']) + ')' for h in hints) + ']'
    names = '#[' + ', '.join('(' + lean_name(h['name']) + ')' for h in hints) + ']'
    return (
        'open Lean in\nrun_cmd Lean.Elab.Command.liftTermElabM do\n'
        f'  let target : Lean.Name := {lean_name(target)}\n'
        '  let closed ← Lean.Meta.deltaExpand (Lean.mkConst target) (· == target)\n'
        '  let negative := Lean.mkApp (Lean.mkConst ``Not) closed\n'
        f'  let hints : Array (Lean.Name × Lean.Name × String) := {entries}\n'
        '  for (name, proved, expectedHash) in hints do\n'
        '    let info ← Lean.getConstInfo name\n'
        '    unless (match info with | .thmInfo _ => true | _ => false) do\n'
        '      throwError "student premise must contain a theorem proof: {name}"\n'
        '    unless info.levelParams.isEmpty && ((← Lean.getEnv).getModuleIdxFor? name).isNone do\n'
        '      throwError "student premise must be closed and source-local: {name}"\n'
        '    unless (← Lean.Meta.isDefEq info.type (Lean.mkConst proved)) do\n'
        '      throwError "student proof proposition differs from independently accepted source: {name}"\n'
        '    for ax in (← Lean.collectAxioms name) do\n'
        '      unless #[``propext, ``Classical.choice, ``Quot.sound].contains ax do\n'
        '        throwError "student proof has forbidden axiom: {ax}"\n'
        '    if (← Lean.Meta.isDefEq info.type closed) || (← Lean.Meta.isDefEq info.type negative) then\n'
        '      throwError "whole-target-equivalent student hint forbidden: {name}"\n'
        '    let statement := toString (← Lean.Meta.ppExpr info.type)\n'
        '    let hashed ← IO.Process.output { cmd := "python3", args := #["-c",\n'
        '      "import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())"] } (some statement)\n'
        '    unless hashed.exitCode == 0 && hashed.stdout.trimAscii.toString == expectedHash do\n'
        '      throwError "student premise type hash mismatch: {name}"\n'
        '    logInfo m!"REAP_STUDENT_PROOF_PREMISE {name} {expectedHash}\\n{statement}"\n\n'
        'def ReapCurriculumWrapper.studentProofSelector : Lean.LibrarySuggestions.Selector := fun _ config => do\n'
        f'  let names : Array Lean.Name := {names}\n'
        '  let suggestions : Array Lean.LibrarySuggestions.Suggestion := names.map fun name =>\n'
        '    { name := name, score := 1.0 }\n'
        '  let suggestions ← suggestions.filterM fun suggestion => config.filter suggestion.name\n'
        '  return suggestions.take config.maxSuggestions\n\n'
        'set_library_suggestions ReapCurriculumWrapper.studentProofSelector\n\n')
