import ReapRuntime
import Mathlib

set_option reap.c_init 1500
set_option reap.progressive_sampling_c 500

namespace ChallengeV1.H1Curriculum

/-- Initial lesson: rehearse the complete successor/IH/algebra composition at lower degree. -/
def CourseSquareTelescope : Prop :=
  ∀ n : ℕ,
    (∑ i ∈ Finset.range n, (((i : ℤ) + 1) ^ 2 - (i : ℤ) ^ 2)) = (n : ℤ) ^ 2

/-- Lesson 2: isolate exactly one cubic induction step, with the IH supplied. -/
def CourseCubeSuccessorFromIH : Prop :=
  ∀ n : ℕ,
    (∑ i ∈ Finset.range n, (((i : ℤ) + 1) ^ 3 - (i : ℤ) ^ 3)) = (n : ℤ) ^ 3 →
      (∑ i ∈ Finset.range (n + 1), (((i : ℤ) + 1) ^ 3 - (i : ℤ) ^ 3)) =
        ((n + 1 : ℕ) : ℤ) ^ 3

end ChallengeV1.H1Curriculum

/-- The original target, retained without mathematical rewriting. -/
def ChallengeV1.H1 : Prop :=
  ∀ n : ℕ,
    (∑ i ∈ Finset.range n, (((i : ℤ) + 1) ^ 3 - (i : ℤ) ^ 3)) = (n : ℤ) ^ 3

open Lean in
run_cmd do
  unless (← getCurrNamespace) == Name.anonymous do
    throwError "root namespace required"
  unless (← Lean.Elab.Command.getScope).varDecls.isEmpty do
    throwError "section variables forbidden"
  let name : Lean.Name := Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "ChallengeV1") "H1Curriculum") "CourseSquareTelescope"
  let some info := (← getEnv).find? name | throwError "missing declaration"
  unless info.levelParams.isEmpty && info.type == Lean.mkSort Lean.Level.zero do
    throwError "closed Prop declaration required"

set_option reap.num_premises 0
set_option reap.num_samples 8
set_option reap.max_tokens 512
set_option reap.max_steps 32
set_option reap.max_goals 128
set_option reap.visit_discount 990

theorem ReapCurriculumWrapper.attempt : _root_.ChallengeV1.H1Curriculum.CourseSquareTelescope := by
  unfold _root_.ChallengeV1.H1Curriculum.CourseSquareTelescope
  intro n
  simp [Finset.sum_range_succ, Finset.sum_range_zero]
  simp [Finset.sum_range_succ, Finset.sum_range_zero, pow_two, Nat.cast_add_one, Nat.cast_one]
  induction n <;> simp [Finset.sum_range_succ, *]
  linarith

open Lean in
run_cmd do
  let target : Lean.Name := Lean.Name.str (Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "ChallengeV1") "H1Curriculum") "CourseSquareTelescope"
  let theoremName := Lean.Name.str (Lean.Name.str Lean.Name.anonymous "ReapCurriculumWrapper") "attempt"
  let info ← getConstInfo theoremName
  let expected := Lean.mkConst target
  unless info.levelParams.isEmpty && info.type == expected do
    throwError "attempt theorem type differs from the complete closed proposition"

#print axioms ReapCurriculumWrapper.attempt
