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

def ReapCurriculumWrapper.closed : Prop := _root_.ClosedFixture.falseClaim
def ReapCurriculumWrapper.attempted : Prop := _root_.Not (_root_.ClosedFixture.falseClaim)
#check ReapCurriculumWrapper.closed
#check ReapCurriculumWrapper.attempted
