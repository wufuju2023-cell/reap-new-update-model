import Reap.Training.SelectionValueRefresh

open Lean Elab Tactic

set_option reap.progressive_sampling_c 0

meta def refreshMockGenerator : Reap.TreeSearch.PolicyValueEval := fun goals => do
  let text := toString (← TacticGenerator.Meta.ppProofState goals)
  let tactic := if text.contains '∧' then "constructor" else "assumption"
  return (-2.0, #[("invalid_observer_tactic", #[], -4.0), (tactic, #[], -1.0)])

meta def refreshMockValue : List MVarId → MetaM Float := fun goals => do
  if goals.isEmpty then throwError "proof leaf was queried"
  if (← IO.getEnv "OBSERVER_TEST_VALUE_ERROR").getD "" == "1" then
    throwError "deterministic value refresh failure"
  return -7.0

elab "selectionValueTestMCTS" : tactic => do
  let session := (← IO.getEnv "REAP_SESSION_ID").getD "selection-value-test"
  let output := System.FilePath.mk ((← IO.getEnv "OBSERVER_TEST_OUT").getD "/workspace/out/selection-value-test")
  IO.FS.createDirAll output
  let (observer?, control?) ← Reap.Training.makeSelectionValueTrainingObserver session
    (valueEval := refreshMockValue)
  let guarded? := observer?.map fun observer => fun record => do
    observer record
    -- The core must restore even deliberately poisoned observer goals.
    setGoals []
  let saved ← saveState
  let result ← Reap.TreeSearch.runMCTS refreshMockGenerator
    (maxNodes := 32) (maxSteps := 8) (observer? := guarded?) (selectionValueRefresh? := control?)
  saved.restore
  Reap.TreeSearch.writeRawTree (output / "tree.json").toString result.info
  let some solution := result.solution? | throwError "Refresh fixture was not solved"
  let .ok script := Reap.TreeSearch.MCTS.proofScriptForSolvedNode result.nodes solution
    | throwError "Refresh fixture could not extract proof"
  let .ok _ ← Reap.TreeSearch.checkProofScript result.ctx script
    | throwError "Refresh fixture final proof failed"
  let some root := result.nodes[0]? | throwError "Missing root"
  root.data.state.restore
  Reap.TreeSearch.MCTS.replaySolvedNode result.ctx 1000000 result.nodes 0
  let .ok _ ← Reap.TreeSearch.checkProof result.ctx | throwError "Refresh replay failed"
  IO.FS.writeFile (output / "result.json") (json%{solved: true, proof: $script}).compress

theorem refresh_fixture (P Q : Prop) (hP : P) (hQ : Q) : P ∧ Q := by
  selectionValueTestMCTS

#print axioms refresh_fixture
