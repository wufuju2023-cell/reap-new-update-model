import ReapRuntime
import Mathlib

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

namespace CodexMathFive.Pell

/-- Advance any positive Pell witness beyond a prescribed natural bound. -/
def CourseBoundStep : Prop :=
  ∀ (B x y : ℕ), B < x → B < y → x ^ 2 = 2 * y ^ 2 + 1 →
    ∃ u v : ℕ, B + 1 < u ∧ B + 1 < v ∧ u ^ 2 = 2 * v ^ 2 + 1

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

/-- Combine bound advancement and the Pell equation for an explicit successor. -/
def CourseExplicitSuccessor : Prop :=
  ∀ (B x y : ℕ), B < x → B < y → x ^ 2 = 2 * y ^ 2 + 1 →
    let u := 3 * x + 4 * y
    let v := 2 * x + 3 * y
    B + 1 < u ∧ B + 1 < v ∧ u ^ 2 = 2 * v ^ 2 + 1

end CodexMathFive.Pell

set_option reap.temperature 150

set_option reap.c_init 1500

open Lean in
run_cmd do
  unless (← getCurrNamespace) == Name.anonymous do
    throwError "root namespace required"
  unless (← Lean.Elab.Command.getScope).varDecls.isEmpty do
    throwError "section variables forbidden"
  let name : Lean.Name := Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "CourseExplicitSuccessor"
  let some info := (← getEnv).find? name | throwError "missing declaration"
  unless info.levelParams.isEmpty && info.type == Lean.mkSort Lean.Level.zero do
    throwError "closed Prop declaration required"

set_option reap.num_premises 2
set_option reap.num_samples 8
set_option reap.max_tokens 256
set_option reap.max_steps 32
set_option reap.max_goals 1024
set_option reap.visit_discount 990

open Lean in
run_cmd Lean.Elab.Command.liftTermElabM do
  let target : Lean.Name := Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "CourseExplicitSuccessor"
  let closed ← Lean.Meta.deltaExpand (Lean.mkConst target) (· == target)
  let negative := Lean.mkApp (Lean.mkConst ``Not) closed
  let hints : Array (Lean.Name × String × String) := #[(Lean.Name.str (Lean.Name.anonymous) "and_iff_left", "Init.PropLemmas", "b6878ec5bff6fb097aa6bfef3083f898416ce0d27dfdb5aeb25eed3f85155ec5"), (Lean.Name.str (Lean.Name.anonymous) "and_iff_right", "Init.PropLemmas", "0ef93d0640d9b4111e9655bac8d43da01b64146532969c26fbd94d49b1bb64a6")]
  for (name, expectedModule, expectedHash) in hints do
    let info ← Lean.getConstInfo name
    unless (match info with | .thmInfo _ => true | _ => false) do
      throwError "premise hint must be a library theorem: {name}"
    let env ← Lean.getEnv
    let some moduleIdx := env.getModuleIdxFor? name
      | throwError "source-local premise hint forbidden: {name}"
    let actualModule := toString env.header.moduleNames[moduleIdx.toNat]!
    unless actualModule == expectedModule do
      throwError "premise hint module mismatch: {name}: {actualModule}"
    if (← Lean.Meta.isDefEq info.type closed) || (← Lean.Meta.isDefEq info.type negative) then
      throwError "target-equivalent premise hint forbidden (definitional equality): {name}"
    let statement := toString (← Lean.Meta.ppExpr info.type)
    let hashed ← IO.Process.output { cmd := "python3", args := #["-c",
      "import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())"] } (some statement)
    unless hashed.exitCode == 0 && hashed.stdout.trimAscii.toString == expectedHash do
      throwError "premise hint type hash mismatch: {name}: {hashed.stdout.trimAscii.toString}"
    logInfo m!"REAP_FIXED_PREMISE_HINT {name} {actualModule} {expectedHash}\n{statement}"

def ReapCurriculumWrapper.fixedPremiseSelector : Lean.LibrarySuggestions.Selector := fun _ config => do
  let names : Array Lean.Name := #[(Lean.Name.str (Lean.Name.anonymous) "and_iff_left"), (Lean.Name.str (Lean.Name.anonymous) "and_iff_right")]
  let suggestions : Array Lean.LibrarySuggestions.Suggestion := names.map fun name =>
    { name := name, score := 1.0 }
  let suggestions ← suggestions.filterM fun suggestion => config.filter suggestion.name
  return suggestions.take config.maxSuggestions

set_library_suggestions ReapCurriculumWrapper.fixedPremiseSelector

theorem ReapCurriculumWrapper.attempt : _root_.CodexMathFive.Pell.CourseExplicitSuccessor := by
  unfold _root_.CodexMathFive.Pell.CourseExplicitSuccessor
  reapTrainingMCTS

open Lean in
run_cmd do
  let target : Lean.Name := Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "CodexMathFive") "Pell") "CourseExplicitSuccessor"
  let theoremName := Lean.Name.str (Lean.Name.str Lean.Name.anonymous "ReapCurriculumWrapper") "attempt"
  let info ← getConstInfo theoremName
  let expected := Lean.mkConst target
  unless info.levelParams.isEmpty && info.type == expected do
    throwError "attempt theorem type differs from the complete closed proposition"
