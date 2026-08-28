import Reap.Training.Observer

open Lean Elab Tactic

set_option reap.progressive_sampling_c 0
set_option reap.max_steps 8
set_option reap.max_goals 32

meta def observerMockGenerator : Reap.TreeSearch.PolicyValueEval := fun goals => do
  let text := toString (← TacticGenerator.Meta.ppProofState goals)
  let tactic := if text.contains '∧' then "constructor" else "assumption"
  return (-2.0, #[("invalid_observer_tactic", #[], -4.0), (tactic, #[], -1.0)])

elab "observerTestMCTS" : tactic => do
  let session := (← IO.getEnv "REAP_SESSION_ID").getD "observer-test"
  let output := System.FilePath.mk ((← IO.getEnv "OBSERVER_TEST_OUT").getD "/workspace/out/observer-test")
  IO.FS.createDirAll output
  let observer? ← Reap.Training.makeTrainingObserver session
  let poison := (← IO.getEnv "OBSERVER_TEST_POISON").getD "0" == "1"
  let guarded? := observer?.map fun observer => fun record => do
    observer record
    if poison then setGoals []
  let saved ← saveState
  let result ← Reap.TreeSearch.runMCTS observerMockGenerator
    (maxNodes := 32) (maxSteps := 8) (observer? := guarded?)
  saved.restore
  Reap.TreeSearch.writeRawTree (output / "tree.json").toString result.info
  let some solution := result.solution? | throwError "Observer fixture was not solved"
  let .ok script := Reap.TreeSearch.MCTS.proofScriptForSolvedNode result.nodes solution
    | throwError "Observer fixture could not extract proof"
  IO.println s!"FIXTURE_SCRIPT: {script}"
  let checked ← Reap.TreeSearch.checkProofScript result.ctx script
  match checked with
  | .ok _ => pure ()
  | .error error => throwError "Observer fixture final proof failed: {(toJson error).compress}"
  let some root := result.nodes[0]? | throwError "Missing root"
  root.data.state.restore
  Reap.TreeSearch.MCTS.replaySolvedNode result.ctx 1000000 result.nodes 0
  let .ok _ ← Reap.TreeSearch.checkProof result.ctx | throwError "Replay failed"
  IO.FS.writeFile (output / "result.json") (json%{solved: true, proof: $script}).compress

theorem observer_fixture (P Q : Prop) (hP : P) (hQ : Q) : P ∧ Q := by
  observerTestMCTS
