module

public meta import Reap.Training.RolloutSink

open Lean Elab Tactic

public meta section

namespace Reap.VerifiedCollector

private def requiredNat (name : String) : IO Nat := do
  let value := (← IO.getEnv name).getD ""
  let some result := value.toNat? | throw <| IO.userError s!"Invalid {name}"
  return result

private def readJson (path : System.FilePath) : IO Json := do
  match Json.parse (← IO.FS.readFile path) with
  | .ok value => return value
  | .error error => throw <| IO.userError s!"Invalid collector ACK: {error}"

private def waitReady (path : System.FilePath) (session tree release : String) : IO Unit := do
  let timeout ← requiredNat "REAP_CHECKPOINT_TIMEOUT_SECONDS"
  let deadline := (← IO.monoMsNow) + timeout * 1000
  while !(← path.pathExists) do
    if (← IO.monoMsNow) >= deadline then throw <| IO.userError "Collector ready ACK timeout"
    IO.sleep 20
  let ack ← readJson path
  unless (ack.getObjValAs? String "session_id").toOption == some session &&
      (ack.getObjValAs? String "tree_id").toOption == some tree &&
      (ack.getObjValAs? String "model_release_sha256").toOption == some release &&
      (ack.getObjValAs? String "status").toOption == some "continue" &&
      (ack.getObjValAs? Nat "policy_version").toOption == some 0 do
    throw <| IO.userError "Collector ready ACK identity/version mismatch"

/-- The existing Generator converts the HTTP positive distance to negative V.
This gate does not negate again and does not perform sigmoid/gamma conversion. -/
def checkedValue (maximum : Nat) (value : Float) : MetaM Float := do
  if value.isNaN || value.isInf || value > -1.0 || value < -(maximum.toFloat) then
    throwError "Collector value is not a finite negative distance in support"
  return value

/-- Fixed-release observer. In addition to the base observer's checks, reject
any nonzero ACK, including the final solved checkpoint before root replay. -/
def makeObserver (session : String) : TacticM (Reap.TreeSearch.MCTSObserver × Nat) := do
  let release := (← IO.getEnv "REAP_COLLECTOR_RELEASE_SHA256").getD ""
  unless release.length == 64 && release.toList.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) do
    throwError "Collector requires explicit lowercase release SHA256"
  unless (← IO.getEnv "REAP_POLICY_VERSION") == some "0" &&
      (← IO.getEnv "REAP_COLLECTOR_RETURN_DISCOUNT") == some "1" &&
      ((← IO.getEnv "REAP_SELECTION_VALUE_REFRESH").getD "").isEmpty do
    throwError "Collector requires fixed version zero, undiscounted returns and no refresh"
  let option ← requiredNat "REAP_COLLECTOR_PUCT_GAMMA_OPTION"
  let maximum ← requiredNat "REAP_COLLECTOR_MAX_DISTANCE"
  unless option > 0 && option < 1000 && maximum ≥ 2 && maximum ≤ 4096 &&
      reap.visit_discount.get (← getOptions) == option do
    throwError "Collector PUCT gamma or distance support mismatch"
  let directory := (← IO.getEnv "REAP_CHECKPOINT_DIR").getD ""
  let tree := (← IO.getEnv "REAP_TREE_ID").getD ""
  unless !directory.isEmpty && !tree.isEmpty do throwError "Collector requires identity and ACK directory"
  let some observer ← Reap.Training.makeTrainingObserver session
    | throwError "Collector requires an observer"
  observer <| json%{kind: "collector_contract", profile: "verified-release-collector-v1",
    model_release_sha256: $release, return_discount: 1,
    puct_value_gamma: $(option.toFloat / 1000.0), max_distance: $maximum,
    value_adapter: "positive-distance-negated-once", training_enabled: false}
  waitReady ((System.FilePath.mk directory) / "collector-ready.ack.json") session tree release
  return ((fun event => do
    observer event
    if (event.getObjValAs? String "kind").toOption == some "checkpoint" then
      let .ok step := event.getObjValAs? Nat "step" | throwError "Collector checkpoint missing step"
      let text := toString step
      let padded := String.ofList (List.replicate (6 - text.length) '0') ++ text
      let ack ← readJson ((System.FilePath.mk directory) / s!"checkpoint-{padded}.ack.json")
      unless (ack.getObjValAs? Nat "policy_version").toOption == some 0 do
        throwError "Fixed release collector cannot change policy version"), maximum)

/-- Shared only by this explicit entry and local deterministic fixtures.
Search/OR/AND backup and proof reconstruction remain the existing Reap code. -/
def runCollector (generate : Reap.TreeSearch.PolicyValueEval) : TacticM Unit := do
  let opts ← getOptions
  let session ← Reap.Training.envOr "REAP_SESSION_ID" "default"
  let output ← Reap.Training.sessionOutputDir
  let resultPath := output / "result.json"
  let start ← IO.monoNanosNow
  Reap.WallClock.openLogFile (output / "wall_clock.jsonl")
  try
    let (observer, maximum) ← makeObserver session
    let evaluate : Reap.TreeSearch.PolicyValueEval := fun goals => do
      if goals.isEmpty then throwError "Collector must not evaluate a proved terminal"
      let (value, candidates) ← generate goals
      let value ← checkedValue maximum value
      -- The core generation event records this exact value as search_value.
      return (value, candidates)
    let progress (p : Reap.TreeSearch.MCTSProgress) : TacticM Unit := do
      Reap.Training.appendJsonl (output / "progress.jsonl")
        (Reap.Training.progressJson session ((← IO.monoNanosNow) - start) p)
    let saved ← saveState
    let result ← Reap.TreeSearch.runMCTS evaluate
      (maxNodes := reap.max_goals.get opts) (maxSteps := reap.max_steps.get opts)
      (progress? := some progress) (observer? := some observer)
    saved.restore
    Reap.TreeSearch.writeRawTree (output / "raw_tree.json").toString result.info
    let elapsed := (← IO.monoNanosNow) - start
    let some index := result.solution? | do
      Reap.Training.writeJson resultPath <|
        Reap.Training.resultJson session elapsed false "exhausted" none none
      throwError "REAP collector search exhausted without a verified proof"
    let .ok script := Reap.TreeSearch.MCTS.proofScriptForSolvedNode result.nodes index
      | throwError "Collector proof script extraction failed"
    let .ok _ ← Reap.TreeSearch.checkProofScript result.ctx script
      | throwError "Collector extracted proof failed check"
    let some root := result.nodes[0]? | throwError "Collector result has no root"
    root.data.state.restore
    Reap.TreeSearch.MCTS.replaySolvedNode result.ctx (reap.heartbeats.get opts) result.nodes 0
    let .ok _ ← Reap.TreeSearch.checkProof result.ctx | throwError "Collector root replay failed"
    Reap.Training.writeJson resultPath <|
      Reap.Training.resultJson session elapsed true "solved" (some script) none
  catch error =>
    if !(← resultPath.pathExists) then
      Reap.Training.writeJson resultPath <| Reap.Training.resultJson session
        ((← IO.monoNanosNow) - start) false "exception" none (some (← error.toMessageData.toString))
    throw error

elab "reapVerifiedCollectorMCTS" : tactic => do
  runCollector TacticGenerator.generatePolicyValue

end Reap.VerifiedCollector
