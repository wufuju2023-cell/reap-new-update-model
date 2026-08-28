module

public meta import Reap.Training.Observer
public meta import Reap.Tactic.Generator

open Lean Elab Tactic

public meta section

namespace Reap.Training

/-- Value-only inference, never policy generation or learning. Strict observer
mode makes the existing value transport throw instead of returning a sentinel. -/
def generateSelectionValue (goals : List MVarId) : MetaM Float := do
  let opts ← getOptions
  let generator ← TacticGenerator.getClient
  let state := toString (← TacticGenerator.Meta.ppProofState goals)
  let premises ← TacticGenerator.getRelatedTheorems goals state opts
  TacticGenerator.generateValueFromPrompt generator opts state premises
    (TacticGenerator.mkPrompt state premises)

/-- Defaults to the unchanged observer. Enabling requires a real ACK barrier;
there is no clock polling, global model switch, or automatic fallback. -/
def makeSelectionValueTrainingObserver (sessionId : String)
    (valueEval : List MVarId → MetaM Float := generateSelectionValue) :
    IO (Option Reap.TreeSearch.MCTSObserver × Option Reap.TreeSearch.SelectionValueRefresh) := do
  let mode := (← IO.getEnv "REAP_SELECTION_VALUE_REFRESH").getD ""
  if mode.isEmpty then
    return (← makeTrainingObserver sessionId, none)
  unless mode == "selection-value-refresh" do
    throw <| IO.userError "REAP_SELECTION_VALUE_REFRESH must be empty or selection-value-refresh"
  unless !((← IO.getEnv "REAP_OBSERVER_PATH").getD "").isEmpty &&
         !((← IO.getEnv "REAP_CHECKPOINT_DIR").getD "").isEmpty do
    throw <| IO.userError "selection-value-refresh requires observer output and checkpoint ACK directory"
  let version ← IO.mkRef (0 : Nat)
  let observer? ← makeTrainingObserver sessionId (policyVersionRef? := some version)
  let control : Reap.TreeSearch.SelectionValueRefresh := {
    currentVersion := liftM (show IO Nat from version.get)
    evalValue := valueEval
  }
  return (observer?, some control)

end Reap.Training
