module

public meta import Reap.Options
public meta import Reap.Tactic.Generator
public meta import Reap.Tactic.TreeSearch
public meta import Reap.Tactic.WallClock
public meta import Reap.Training.Observer

open Lean Elab Tactic

public meta section

namespace Reap.Training

def envOr (name fallback : String) : IO String := do
  return (← IO.getEnv name).getD fallback

def sessionOutputDir : IO System.FilePath := do
  let root ← envOr "REAP_SESSION_DIR" "/workspace/out/default"
  let path := System.FilePath.mk root
  IO.FS.createDirAll path
  return path

def appendJsonl (path : System.FilePath) (record : Json) : IO Unit := do
  IO.FS.withFile path .append fun h => do
    h.putStrLn record.compress
    h.flush

def writeJson (path : System.FilePath) (record : Json) : IO Unit := do
  IO.FS.withFile path .write fun h => h.putStrLn record.compress

def progressJson (sessionId : String) (elapsedNs : Nat)
    (progress : Reap.TreeSearch.MCTSProgress) : Json :=
  json%{
    schema_version: "reap.training.progress.v1",
    session_id: $sessionId,
    elapsed_ns: $elapsedNs,
    visited_nodes: $progress.visitedNodes,
    max_nodes: $progress.maxNodes,
    step: $progress.step,
    max_steps: $progress.maxSteps,
    goal_type: $progress.goalType,
    done: $progress.done,
    solved: $progress.solved,
    status: $progress.status
  }

def resultJson (sessionId : String) (elapsedNs : Nat) (solved : Bool)
    (status : String) (script? : Option String) (error? : Option String) : Json :=
  json%{
    schema_version: "reap.training.result.v1",
    session_id: $sessionId,
    elapsed_ns: $elapsedNs,
    solved: $solved,
    status: $status,
    proof_script: $script?,
    error: $error?
  }

elab "reapTrainingMCTS" : tactic => do
  let opts ← Lean.getOptions
  let maxNodes := reap.max_goals.get opts
  let maxSteps := reap.max_steps.get opts
  let sessionId ← envOr "REAP_SESSION_ID" "default"
  let outputDir ← sessionOutputDir
  let progressPath := outputDir / "progress.jsonl"
  let resultPath := outputDir / "result.json"
  let treePath := outputDir / "raw_tree.json"
  let wallClockPath := outputDir / "wall_clock.jsonl"
  let start ← IO.monoNanosNow
  Reap.WallClock.openLogFile wallClockPath
  let observer? ← makeTrainingObserver sessionId
  let reportProgress (progress : Reap.TreeSearch.MCTSProgress) : TacticM Unit := do
    let now ← IO.monoNanosNow
    appendJsonl progressPath (progressJson sessionId (now - start) progress)
  try
    let saved ← saveState
    let result ← Reap.TreeSearch.runMCTS
      TacticGenerator.generatePolicyValue
      (maxNodes := maxNodes) (maxSteps := maxSteps) (progress? := some reportProgress)
      (observer? := observer?)
    saved.restore
    Reap.TreeSearch.writeRawTree treePath.toString result.info
    let now ← IO.monoNanosNow
    match result.solution? with
    | none =>
        writeJson resultPath <| resultJson sessionId (now - start) false "exhausted" none none
        throwError "REAP training search exhausted without a verified proof"
    | some nodeIdx =>
        match Reap.TreeSearch.MCTS.proofScriptForSolvedNode result.nodes nodeIdx with
        | .error err =>
            writeJson resultPath <| resultJson sessionId (now - start) false
              "script_extraction_failed" none (some err)
            throwError "REAP proof script extraction failed: {err}"
        | .ok script =>
            match ← Reap.TreeSearch.checkProofScript result.ctx script with
            | .error err =>
                let message := (toJson err).compress
                writeJson resultPath <| resultJson sessionId (now - start) false
                  "final_check_failed" (some script) (some message)
                throwError "REAP final proof check failed: {message}"
            | .ok _ =>
                let some root := result.nodes[0]?
                  | throwError "REAP MCTS result has no root node"
                root.data.state.restore
                Reap.TreeSearch.MCTS.replaySolvedNode result.ctx
                  (reap.heartbeats.get opts) result.nodes 0
                match ← Reap.TreeSearch.checkProof result.ctx with
                | .error err =>
                    let message := (toJson err).compress
                    writeJson resultPath <| resultJson sessionId (now - start) false
                      "replay_check_failed" (some script) (some message)
                    throwError "REAP replay check failed: {message}"
                | .ok _ =>
                    writeJson resultPath <| resultJson sessionId (now - start) true
                      "solved" (some script) none
  catch e =>
    if !(← resultPath.pathExists) then
      let now ← IO.monoNanosNow
      let message ← e.toMessageData.toString
      writeJson resultPath <| resultJson sessionId (now - start) false
        "exception" none (some message)
    throw e

end Reap.Training
