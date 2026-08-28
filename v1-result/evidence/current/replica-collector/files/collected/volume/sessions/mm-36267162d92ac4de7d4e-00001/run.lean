import Reap.VerifiedCollector
import ReapRuntime
def ClosedFixture.trueClaim : Prop := forall n : Nat, n = n
def ClosedFixture.falseClaim : Prop := forall n : Nat, n = 0
def ClosedFixture.parameterized (n : Nat) : Prop := n = n
def ClosedFixture.boolTarget : Bool := true

open Lean in
run_cmd do
  unless (← getCurrNamespace) == Name.anonymous do
    throwError "root namespace required"
  unless (← Lean.Elab.Command.getScope).varDecls.isEmpty do
    throwError "section variables forbidden"
  let name : Lean.Name := Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "ClosedFixture") "falseClaim"
  let some info := (← getEnv).find? name | throwError "missing declaration"
  unless info.levelParams.isEmpty && info.type == Lean.mkSort Lean.Level.zero do
    throwError "closed Prop declaration required"

set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 8
set_option reap.max_goals 64
set_option reap.visit_discount 990

theorem ReapCurriculumWrapper.attempt : _root_.Not (_root_.ClosedFixture.falseClaim) := by
  unfold _root_.ClosedFixture.falseClaim
  set_option reap.visit_discount 990 in
    reapVerifiedCollectorMCTS

open Lean in
run_cmd do
  let target : Lean.Name := Lean.Name.str (Lean.Name.str (Lean.Name.anonymous) "ClosedFixture") "falseClaim"
  let theoremName := Lean.Name.str (Lean.Name.str Lean.Name.anonymous "ReapCurriculumWrapper") "attempt"
  let info ← getConstInfo theoremName
  let expected := Lean.mkApp (Lean.mkConst (Lean.Name.str Lean.Name.anonymous "Not")) (Lean.mkConst target)
  unless info.levelParams.isEmpty && info.type == expected do
    throwError "attempt theorem type differs from the complete closed proposition"
