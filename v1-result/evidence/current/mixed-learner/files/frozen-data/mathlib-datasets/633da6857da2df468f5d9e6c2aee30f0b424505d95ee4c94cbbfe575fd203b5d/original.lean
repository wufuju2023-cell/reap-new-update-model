import ReapRuntime
import Reap.Tactic.TreeSearch

open Lean Meta Elab Tactic Reap.TreeSearch

namespace Reap.MathlibSFT

structure Plan where
  schema_version : String
  declaration : String
  actions : Array String
deriving FromJson

meta def states : TacticM (Array String) := do
  return (← (← getUnsolvedGoals).mapM fun goal => do
    return toString (← Meta.ppGoal goal)).toArray

meta partial def references (target : Name) : Expr → Bool
  | .const name _ => name == target
  | .app fn arg => references target fn || references target arg
  | .lam _ type body _ => references target type || references target body
  | .forallE _ type body _ => references target type || references target body
  | .letE _ type value body _ => references target type || references target value || references target body
  | .mdata _ body => references target body
  | .proj _ _ body => references target body
  | _ => false

elab "#mathlibSFTNoSelf " original:ident candidate:ident : command => do
  let some info := (← getEnv).find? candidate.getId
    | throwError "mathlib SFT missing replay declaration"
  let some value := info.value?
    | throwError "mathlib SFT replay has no proof value"
  if references original.getId value then
    throwError "mathlib SFT proof references its original theorem"

elab "mathlibSFTReplay " planPath:str tracePath:str expectedPath:str : tactic => do
  let raw ← IO.FS.readFile planPath.getString
  let parsed ← match Json.parse raw with
    | .ok value => pure value
    | .error error => throwError "mathlib SFT invalid JSON: {error}"
  let plan ← match (fromJson? parsed : Except String Plan) with
    | .ok value => pure value
    | .error error => throwError "mathlib SFT invalid plan: {error}"
  unless plan.schema_version == "reap.mathlib-sft.linear-plan.v1" && !plan.actions.isEmpty do
    throwError "mathlib SFT unsupported or empty plan"
  let some declaration ← Lean.Elab.Term.getDeclName?
    | throwError "mathlib SFT requires a named theorem"
  unless declaration.toString == "MathlibSFTGate." ++ plan.declaration do
    throwError "mathlib SFT declaration mismatch"
  let ctx ← mkProofCheckContext
  let mut rows := #[]
  for index in [:plan.actions.size] do
    let before ← states
    unless before.size == 1 do
      throwError "mathlib SFT linear profile requires exactly one unsolved goal"
    let tactic := plan.actions[index]!
    match ← evalTacticStrNoFinalCheck ctx tactic (reap.heartbeats.get (← getOptions)) with
    | .error error => throwError "mathlib SFT tactic failed: {toJson error}"
    | .ok _ => pure ()
    let after ← states
    let expectedCount := if index + 1 == plan.actions.size then 0 else 1
    unless after.size == expectedCount do
      throwError "mathlib SFT unsupported branch, early completion, or unfinished proof"
    let value := -(Int.ofNat (plan.actions.size - index))
    rows := rows.push (json%{
      row: $(index), state: $(before), tactic: $(tactic), next_state: $(after),
      prompt: $(TacticGenerator.mkPrompt before[0]! #[]), "return": $(value)
    })
  match ← checkProof ctx with
  | .error error => throwError "mathlib SFT final proof check failed: {toJson error}"
  | .ok _ => pure ()
  let trace := json%{
    schema_version: "reap.mathlib-sft.linear-trace.v1", complete: true,
    "theorem": $(declaration.toString), original_declaration: $(plan.declaration),
    root_return: $(-(Int.ofNat plan.actions.size)), rows: $(rows)
  }
  unless expectedPath.getString.isEmpty do
    let expectedRaw ← IO.FS.readFile expectedPath.getString
    let expected ← match Json.parse expectedRaw with
      | .ok value => pure value
      | .error error => throwError "mathlib SFT invalid saved trace: {error}"
    unless trace == expected do throwError "mathlib SFT saved state/action/return trace mismatch"
  IO.FS.writeFile tracePath.getString trace.pretty

end Reap.MathlibSFT

namespace MathlibSFTGate
variable {α : Sort*}
theorem ExistsUnique.elim₂ {p : α → Sort*} [∀ x, Subsingleton (p x)]
    {q : ∀ (x) (_ : p x), Prop} {b : Prop} (h₂ : ∃! x, ∃! h : p x, q x h)
    (h₁ : ∀ (x) (h : p x), q x h → (∀ (y) (hy : p y), q y hy → y = x) → b) : b := by
  simp only [existsUnique_iff_exists] at h₂
  apply h₂.elim
  exact fun x ⟨hxp, hxq⟩ H ↦ h₁ x hxp hxq fun y hyp hyq ↦ H y ⟨hyp, hyq⟩
end MathlibSFTGate
#mathlibSFTNoSelf ExistsUnique.elim₂ MathlibSFTGate.ExistsUnique.elim₂
example : @ExistsUnique.elim₂ = @MathlibSFTGate.ExistsUnique.elim₂ := by rfl
#print axioms MathlibSFTGate.ExistsUnique.elim₂
