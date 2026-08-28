import ReapRuntime
import Mathlib
import Reap.Tactic.TreeSearch

/- Independent, opt-in replay of an already accepted proof path. This module is
   embedded into a standalone theorem file; it never starts search or learning. -/
open Lean Meta Elab Tactic Reap.TreeSearch

namespace Reap.VerifiedReplay

structure ReplayNode where
  node_index : Nat
  state : Array String
  kind : String
  tactic : String
  children : Array Nat
deriving FromJson

structure ReplayPlan where
  schema_version : String
  nodes : Array ReplayNode
deriving FromJson

meta def actualState : TacticM (Array String) := do
  return (← (← getUnsolvedGoals).mapM fun goal => do
    return toString (← Meta.ppGoal goal)).toArray

meta partial def replay (ctx : ProofCheckContext) (nodes : Array ReplayNode)
    (index : Nat) (fuel : Nat) : TacticM (Int × Array Json) := do
  if fuel == 0 then throwError "verified replay exhausted structural bound"
  let some node := nodes[index]? | throwError "verified replay missing node"
  let before ← actualState
  unless before == node.state do
    throwError "verified replay state mismatch at node {node.node_index}: expected {toJson node.state}, actual {toJson before}"
  if node.kind == "terminal" then
    unless before.isEmpty && node.children.isEmpty && node.tactic.isEmpty do
      throwError "verified replay nonterminal leaf"
    return (0, #[])
  else if node.kind == "OR" then
    unless before.size == 1 && node.children.size == 1 && !node.tactic.isEmpty do
      throwError "verified replay invalid OR shape"
    let heartbeats := reap.heartbeats.get (← getOptions)
    match ← evalTacticStrNoFinalCheck ctx node.tactic heartbeats with
    | .error err => throwError "verified replay tactic failed at node {node.node_index}: {toJson err}"
    | .ok _ => pure ()
    let after ← actualState
    let (rest, rows) ← replay ctx nodes node.children[0]! (fuel - 1)
    let value := rest - 1
    let row := json%{
      node_index: $(node.node_index), state: $(before), tactic: $(node.tactic),
      next_state: $(after), "return": $(value)
    }
    return (value, #[row] ++ rows)
  else if node.kind == "AND" then
    unless before.size >= 2 && node.children.size == before.size && node.tactic.isEmpty do
      throwError "verified replay invalid AND shape"
    let goals := (← getUnsolvedGoals).toArray
    if ← goals.toList.anyM goalContainsExprMVar then
      throwError "verified replay dependent/metavariable AND goals are unsupported"
    let mut worst : Option Int := none
    let mut rows := #[]
    for child in node.children, goal in goals do
      if ← goal.isAssigned then
        throwError "verified replay shared-metavariable branch was solved by a sibling; labels unsupported"
      setGoals [goal]
      let (value, branchRows) ← replay ctx nodes child (fuel - 1)
      unless (← getUnsolvedGoals).isEmpty do throwError "verified replay unfinished AND branch"
      worst := some (worst.map (min value) |>.getD value)
      rows := rows ++ branchRows
    setGoals goals.toList
    pruneSolvedGoals
    unless (← getUnsolvedGoals).isEmpty do throwError "verified replay unfinished conjunction"
    return (worst.getD 0, rows)
  else throwError "verified replay unknown node kind"

elab "reapVerifiedReplay " planPath:str tracePath:str : tactic => do
  let raw ← IO.FS.readFile planPath.getString
  let parsed ← match Json.parse raw with
    | .ok value => pure value
    | .error error => throwError "verified replay invalid JSON: {error}"
  let plan ← match (fromJson? parsed : Except String ReplayPlan) with
    | .ok value => pure value
    | .error error => throwError "verified replay invalid plan: {error}"
  unless plan.schema_version == "reap.verified-replay.plan.v1" do
    throwError "verified replay unsupported schema"
  let ctx ← mkProofCheckContext
  let some declaration ← Lean.Elab.Term.getDeclName?
    | throwError "verified replay requires a named theorem declaration"
  let (value, rows) ← replay ctx plan.nodes 0 (plan.nodes.size + 1)
  unless (← getUnsolvedGoals).isEmpty do throwError "verified replay left goals"
  match ← checkProof ctx with
  | .error error => throwError "verified replay final proof check: {toJson error}"
  | .ok _ => pure ()
  IO.FS.writeFile tracePath.getString (json%{
    schema_version: "reap.verified-replay.trace.v1", complete: true,
    "theorem": $(declaration.toString), root_return: $(value), rows: $(rows)
  }).pretty

end Reap.VerifiedReplay


/-!
Replacement collection with five complete targets. Original declarations
for the retained exercises are byte-identical to the first collection.
The teacher has read the fixed Mathlib Archive proof of IMO 1988 Q6.
This student file does not import Archive or teacher proof files.
-/

namespace CodexMathFive.FunctionalEquation

def Satisfies (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (x + f y) = f x + y

def Target : Prop :=
  ∀ f : ℤ → ℤ, Satisfies f ↔
    ((∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x))

def CourseInvolution : Prop :=
  ∀ f : ℤ → ℤ, Satisfies f → f 0 = 0 ∧ ∀ y : ℤ, f (f y) = y

def CourseAdditivity : Prop :=
  ∀ f : ℤ → ℤ, Satisfies f → ∀ x y : ℤ, f (x + y) = f x + f y

def CourseIntegerLinearity : Prop :=
  ∀ f : ℤ → ℤ, (∀ x y : ℤ, f (x + y) = f x + f y) →
    ∀ n : ℤ, f n = n * f 1

end CodexMathFive.FunctionalEquation

namespace CodexMathFive.PrimeSquareSum

def Target : Prop :=
  ∀ p : ℕ, Nat.Prime p → p % 4 = 3 →
    ∀ x y : ℤ, (p : ℤ) ∣ x ^ 2 + y ^ 2 → (p : ℤ) ∣ x ∧ (p : ℤ) ∣ y

def CourseMinusOne : Prop :=
  ∀ p : ℕ, Nat.Prime p → p % 4 = 3 →
    ¬ ∃ z : ZMod p, z ^ 2 = -1

def CourseResiduePair : Prop :=
  ∀ p : ℕ, Nat.Prime p → p % 4 = 3 →
    ∀ x y : ZMod p, x ^ 2 + y ^ 2 = 0 → x = 0 ∧ y = 0

end CodexMathFive.PrimeSquareSum

namespace CodexMathFive.Pell

def step (p : ℕ × ℕ) : ℕ × ℕ :=
  (3 * p.1 + 4 * p.2, 2 * p.1 + 3 * p.2)

def pellPair : ℕ → ℕ × ℕ
  | 0 => (3, 2)
  | k + 1 => step (pellPair k)

def Target : Prop :=
  ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1

def CourseInvariant : Prop :=
  ∀ x y : ℕ, x ^ 2 = 2 * y ^ 2 + 1 →
    (step (x, y)).1 ^ 2 = 2 * (step (x, y)).2 ^ 2 + 1

def CoursePositiveGrowth : Prop :=
  ∀ x y : ℕ, 0 < x → 0 < y → x < (step (x, y)).1 ∧ y < (step (x, y)).2

def CourseIterates : Prop :=
  ∀ k : ℕ, k < (pellPair k).1 ∧ k < (pellPair k).2 ∧
    (pellPair k).1 ^ 2 = 2 * (pellPair k).2 ^ 2 + 1

end CodexMathFive.Pell

namespace CodexMathFive.Imo1988Q6

def Equation (a b q : ℕ) : Prop := a ^ 2 + b ^ 2 = (a * b + 1) * q

def Target : Prop :=
  ∀ a b : ℕ, 0 < a → 0 < b → (a * b + 1) ∣ (a ^ 2 + b ^ 2) →
    ∃ k : ℕ, a ^ 2 + b ^ 2 = (a * b + 1) * k ^ 2

def CoursePositiveQuotient : Prop :=
  ∀ a b : ℕ, 0 < a → 0 < b → (a * b + 1) ∣ (a ^ 2 + b ^ 2) →
    ∃ q : ℕ, 0 < q ∧ Equation a b q

def CourseOtherRoot : Prop :=
  ∀ a b q : ℤ, a ^ 2 + b ^ 2 = (a * b + 1) * q →
    (q * a - b) ^ 2 + a ^ 2 = ((q * a - b) * a + 1) * q ∧
    (q * a - b) * b = a ^ 2 - q

def CourseDescent : Prop :=
  ∀ a b q : ℕ, 0 < a → a < b → Equation a b q →
    ∃ c : ℕ, c ≤ a ∧ Equation c a q

def CourseBoundary : Prop :=
  ∀ a q : ℕ, (Equation a 0 q ∨ Equation a a q) → ∃ k : ℕ, q = k ^ 2

end CodexMathFive.Imo1988Q6

namespace CodexMathFive.Schur

noncomputable def schurPolynomial (n : ℕ) : Polynomial ℤ :=
  (∏ i ∈ Finset.Icc 1 n, (Polynomial.X - Polynomial.C (i : ℤ))) - 1

def Target : Prop :=
  ∀ n : ℕ, 1 ≤ n → Irreducible (schurPolynomial n)

def CourseMonicDegree : Prop :=
  ∀ n : ℕ, 1 ≤ n → (schurPolynomial n).Monic ∧ (schurPolynomial n).natDegree = n

def CourseIntegerValues : Prop :=
  ∀ n i : ℕ, 1 ≤ i → i ≤ n → (schurPolynomial n).eval (i : ℤ) = -1

def CourseProperFactorDegrees : Prop :=
  ∀ n : ℕ, 1 ≤ n → ∀ g h : Polynomial ℤ,
    schurPolynomial n = g * h → ¬ IsUnit g → ¬ IsUnit h →
      g.natDegree < n ∧ h.natDegree < n

def CourseOppositeFactors : Prop :=
  ∀ n : ℕ, 1 ≤ n → ∀ g h : Polynomial ℤ,
    schurPolynomial n = g * h → g.natDegree < n → h.natDegree < n →
      g + h = 0

def CourseMonicNotNegativeSquare : Prop :=
  ∀ f : Polynomial ℤ, f.Monic → ∀ g : Polynomial ℤ, f ≠ -(g ^ 2)

end CodexMathFive.Schur

namespace CodexMathFive.Pell

def CourseRecurrence : Prop :=
  ∀ s : ℕ → ℕ × ℕ, s 0 = (3, 2) →
    (∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2)) →
    ∀ k : ℕ, k < (s k).1 ∧ k < (s k).2 ∧
      (s k).1 ^ 2 = 2 * (s k).2 ^ 2 + 1

end CodexMathFive.Pell


namespace CodexMathFive.Pell

/-- Construct a sequence from the prescribed initial pair and recurrence. -/
def CourseSequenceExists : Prop :=
  ∃ s : ℕ → ℕ × ℕ, s 0 = (3, 2) ∧
    ∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2)

/-- Use a recurrence sequence to produce witnesses above an arbitrary bound. -/
def CourseUnboundedFromRecurrence : Prop :=
  ∀ s : ℕ → ℕ × ℕ, s 0 = (3, 2) →
    (∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2)) →
    ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1

end CodexMathFive.Pell

namespace CodexMathFive.Pell

def CourseIndexedWitness : Prop :=
  ∀ s : ℕ → ℕ × ℕ,
    (∀ k : ℕ, k < (s k).1 ∧ k < (s k).2 ∧
      (s k).1 ^ 2 = 2 * (s k).2 ^ 2 + 1) →
    ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1

end CodexMathFive.Pell

set_option reap.temperature 99
set_option reap.c_init 1500
set_option reap.progressive_sampling_c 500

-- STUDENT_VERIFIED_BEGIN CodexMathFive.Pell.StudentLibraryRecurrence
theorem CodexMathFive.Pell.StudentLibraryRecurrence :
  ∀ s : ℕ → ℕ × ℕ, s 0 = (3, 2) →
    (∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2)) →
    ∀ k : ℕ, k < (s k).1 ∧ k < (s k).2 ∧
      (s k).1 ^ 2 = 2 * (s k).2 ^ 2 + 1 := by
  intro s h₀ h₁ k
  induction k <;> simp_all
  ring_nf at h₀ h₁ ⊢
  simp_all
  omega
-- STUDENT_VERIFIED_END CodexMathFive.Pell.StudentLibraryRecurrence

#print axioms CodexMathFive.Pell.StudentLibraryRecurrence

-- STUDENT_VERIFIED_BEGIN CodexMathFive.Pell.StudentLibraryIndexedWitness
theorem CodexMathFive.Pell.StudentLibraryIndexedWitness :
  ∀ s : ℕ → ℕ × ℕ,
    (∀ k : ℕ, k < (s k).1 ∧ k < (s k).2 ∧
      (s k).1 ^ 2 = 2 * (s k).2 ^ 2 + 1) →
    ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1 := by
  intro s hs B
  specialize hs B
  suffices s B = s B by rw [this] at hs; aesop
  simp only [Prod.mk.injEq, and_self]
-- STUDENT_VERIFIED_END CodexMathFive.Pell.StudentLibraryIndexedWitness

#print axioms CodexMathFive.Pell.StudentLibraryIndexedWitness

-- STUDENT_VERIFIED_BEGIN CodexMathFive.Pell.StudentLibrarySequenceExists
theorem CodexMathFive.Pell.StudentLibrarySequenceExists :
  ∃ s : ℕ → ℕ × ℕ, s 0 = (3, 2) ∧
    ∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2) := by
  exact ⟨fun k ↦ Nat.recOn k (3, 2) fun k s ↦ (3 * s.1 + 4 * s.2, 2 * s.1 + 3 * s.2), by simp⟩
-- STUDENT_VERIFIED_END CodexMathFive.Pell.StudentLibrarySequenceExists

#print axioms CodexMathFive.Pell.StudentLibrarySequenceExists

-- STUDENT_VERIFIED_BEGIN CodexMathFive.Pell.StudentLibraryUnboundedFromRecurrence
theorem CodexMathFive.Pell.StudentLibraryUnboundedFromRecurrence :
  ∀ s : ℕ → ℕ × ℕ, s 0 = (3, 2) →
    (∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2)) →
    ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1 := by
  have student_fact_1 := @_root_.CodexMathFive.Pell.StudentLibraryRecurrence
  have student_fact_2 := @_root_.CodexMathFive.Pell.StudentLibraryIndexedWitness
  exact fun s ↦ by tauto
-- STUDENT_VERIFIED_END CodexMathFive.Pell.StudentLibraryUnboundedFromRecurrence

#print axioms CodexMathFive.Pell.StudentLibraryUnboundedFromRecurrence

open Lean in
run_cmd do
  unless (← getCurrNamespace) == Name.anonymous do
    throwError "root namespace required"
  unless (← Lean.Elab.Command.getScope).varDecls.isEmpty do
    throwError "section variables forbidden"
  let name : Lean.Name := Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "Target"
  let some info := (← getEnv).find? name | throwError "missing declaration"
  unless info.levelParams.isEmpty && info.type == Lean.mkSort Lean.Level.zero do
    throwError "closed Prop declaration required"

set_option reap.num_premises 2
set_option reap.num_samples 8
set_option reap.max_tokens 256
set_option reap.max_steps 32
set_option reap.max_goals 2048
set_option reap.visit_discount 990

open Lean in
run_cmd Lean.Elab.Command.liftTermElabM do
  let target : Lean.Name := Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "Target"
  let closed ← Lean.Meta.deltaExpand (Lean.mkConst target) (· == target)
  let negative := Lean.mkApp (Lean.mkConst ``Not) closed
  let hints : Array (Lean.Name × Lean.Name × String) := #[(Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "StudentLibrarySequenceExists", Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "CourseSequenceExists", "49459def3ee98d3ccb6eedfe42bfbb3c9b10e31ec084ff71d795807cf8c73522"), (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "StudentLibraryUnboundedFromRecurrence", Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "CourseUnboundedFromRecurrence", "3c2045f615d28e4f8d9f83655415526e952422448a516b1d3ece2d636a6c36a1")]
  for (name, proved, expectedHash) in hints do
    let info ← Lean.getConstInfo name
    unless (match info with | .thmInfo _ => true | _ => false) do
      throwError "student premise must contain a theorem proof: {name}"
    unless info.levelParams.isEmpty && ((← Lean.getEnv).getModuleIdxFor? name).isNone do
      throwError "student premise must be closed and source-local: {name}"
    unless (← Lean.Meta.isDefEq info.type (Lean.mkConst proved)) do
      throwError "student proof proposition differs from independently accepted source: {name}"
    for ax in (← Lean.collectAxioms name) do
      unless #[``propext, ``Classical.choice, ``Quot.sound].contains ax do
        throwError "student proof has forbidden axiom: {ax}"
    if (← Lean.Meta.isDefEq info.type closed) || (← Lean.Meta.isDefEq info.type negative) then
      throwError "whole-target-equivalent student hint forbidden: {name}"
    let statement := toString (← Lean.Meta.ppExpr info.type)
    let hashed ← IO.Process.output { cmd := "python3", args := #["-c",
      "import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())"] } (some statement)
    unless hashed.exitCode == 0 && hashed.stdout.trimAscii.toString == expectedHash do
      throwError "student premise type hash mismatch: {name}"
    logInfo m!"REAP_STUDENT_PROOF_PREMISE {name} {expectedHash}\n{statement}"

def ReapCurriculumWrapper.studentProofSelector : Lean.LibrarySuggestions.Selector := fun _ config => do
  let names : Array Lean.Name := #[(Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "StudentLibrarySequenceExists"), (Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "StudentLibraryUnboundedFromRecurrence")]
  let suggestions : Array Lean.LibrarySuggestions.Suggestion := names.map fun name =>
    { name := name, score := 1.0 }
  let suggestions ← suggestions.filterM fun suggestion => config.filter suggestion.name
  return suggestions.take config.maxSuggestions

set_library_suggestions ReapCurriculumWrapper.studentProofSelector

theorem ReapCurriculumWrapper.attempt : _root_.CodexMathFive.Pell.Target := by
  unfold _root_.CodexMathFive.Pell.Target
  have student_fact_1 := @_root_.CodexMathFive.Pell.StudentLibrarySequenceExists
  have student_fact_2 := @_root_.CodexMathFive.Pell.StudentLibraryUnboundedFromRecurrence
  reapVerifiedReplay "${HISTORICAL_RUNS}/radeon-pell-target-after-unbounded-cpu15-20260828-native/replay-01/plan.json" "${HISTORICAL_RUNS}/radeon-pell-target-after-unbounded-cpu15-20260828-native/replay-01/trace.json"

open Lean in
run_cmd do
  let target : Lean.Name := Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "Target"
  let theoremName := Lean.Name.str (Lean.Name.str Lean.Name.anonymous "ReapCurriculumWrapper") "attempt"
  let info ← getConstInfo theoremName
  let expected := Lean.mkConst target
  unless info.levelParams.isEmpty && info.type == expected do
    throwError "attempt theorem type differs from the complete closed proposition"

#print axioms ReapCurriculumWrapper.attempt
