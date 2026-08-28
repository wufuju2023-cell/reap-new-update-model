module

public meta import Reap.Tactic.TreeSearch

open Lean Elab Tactic

public meta section

namespace Reap.Training

private def requiredNat (value label : String) : IO Nat := do
  match value.toNat? with
  | some n => return n
  | none => throw <| IO.userError s!"{label} must be a nonnegative integer"

private def field {α : Type} [FromJson α] (record : Json) (name : String) : IO α := do
  match record.getObjValAs? α name with
  | .ok value => return value
  | .error error => throw <| IO.userError s!"Invalid checkpoint ACK {name}: {error}"

private def paddedStep (step : Nat) : String :=
  let text := toString step
  String.ofList (List.replicate (6 - text.length) '0') ++ text

private def appendEvent (path : System.FilePath) (record : Json) : IO Unit :=
  IO.FS.withFile path .append fun handle => do
    handle.putStrLn record.compress
    handle.flush

/-- Wait only at a completed iteration. The caller must publish ACK files atomically.
No timeout or malformed/error ACK permits the search to continue. -/
private def awaitCheckpoint (directory : System.FilePath) (sessionId : String)
    (step version timeoutSeconds : Nat) : IO Nat := do
  let path := directory / s!"checkpoint-{paddedStep step}.ack.json"
  let deadline := (← IO.monoMsNow) + timeoutSeconds * 1000
  while !(← path.pathExists) do
    if (← IO.monoMsNow) >= deadline then
      throw <| IO.userError s!"Checkpoint {step} ACK timed out (barrier protection, not total TTT deadline)"
    IO.sleep 20
  let record ← match Json.parse (← IO.FS.readFile path) with
    | .ok record => pure record
    | .error error => throw <| IO.userError s!"Invalid checkpoint ACK JSON: {error}"
  let ackSession : String ← field record "session_id"
  let ackStep : Nat ← field record "step"
  let status : String ← field record "status"
  let nextVersion : Nat ← field record "policy_version"
  unless ackSession == sessionId && ackStep == step do
    throw <| IO.userError "Checkpoint ACK session/step mismatch"
  unless nextVersion >= version do
    throw <| IO.userError "Checkpoint ACK policy_version decreased"
  if status == "error" then
    let message := (record.getObjValAs? String "error").toOption.getD "coordinator reported an error"
    throw <| IO.userError s!"Checkpoint coordinator failed: {message}"
  unless status == "continue" do
    throw <| IO.userError "Checkpoint ACK status must be continue or error"
  return nextVersion

/-- Default-off stream. Only JSON leaves the core; checkpoint ACK changes the
external policy version, never Lean goals or the in-memory MCTS tree. -/
def makeTrainingObserver (sessionId : String) : IO (Option Reap.TreeSearch.MCTSObserver) := do
  let pathText := (← IO.getEnv "REAP_OBSERVER_PATH").getD ""
  if pathText.isEmpty then
    return none
  let path := System.FilePath.mk pathText
  if let some parent := path.parent then IO.FS.createDirAll parent
  let treeId := (← IO.getEnv "REAP_TREE_ID").getD sessionId
  let initialVersion ← requiredNat ((← IO.getEnv "REAP_POLICY_VERSION").getD "0") "REAP_POLICY_VERSION"
  let timeout ← requiredNat ((← IO.getEnv "REAP_CHECKPOINT_TIMEOUT_SECONDS").getD "900") "REAP_CHECKPOINT_TIMEOUT_SECONDS"
  if timeout == 0 then throw <| IO.userError "Checkpoint timeout must be positive"
  let checkpointDir := (← IO.getEnv "REAP_CHECKPOINT_DIR").filter (!·.isEmpty)
  if let some directory := checkpointDir then IO.FS.createDirAll (.mk directory)
  let sequence ← IO.mkRef (0 : Nat)
  let policyVersion ← IO.mkRef initialVersion
  let emit (record : Json) : IO Unit := do
    let index ← sequence.modifyGet fun n => (n, n + 1)
    let version ← policyVersion.get
    let wrapped := record
      |>.setObjVal! "schema_version" (toJson "reap.training.observer.v1")
      |>.setObjVal! "session_id" (toJson sessionId)
      |>.setObjVal! "tree_id" (toJson treeId)
      |>.setObjVal! "policy_version" (toJson version)
      |>.setObjVal! "sequence" (toJson index)
      |>.setObjVal! "monotonic_ns" (toJson (← IO.monoNanosNow))
    appendEvent path wrapped
  return some fun record => do
    liftM <| emit record
    if (record.getObjValAs? String "kind").toOption == some "checkpoint" then
      if let some directory := checkpointDir then
        let step : Nat ← liftM <| field record "step"
        let nextVersion ← liftM <| awaitCheckpoint (.mk directory) sessionId step
          (← policyVersion.get) timeout
        policyVersion.set nextVersion
        liftM <| emit <| json%{ kind: "checkpoint_ack", step: $step }

end Reap.Training
