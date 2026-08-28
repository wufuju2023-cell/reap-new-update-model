import ReapRuntime
import Reap.Tactic.TreeSearch

/- Independent, opt-in replay of an already accepted proof path. This module is
   embedded into a standalone theorem file; it never starts search or learning. -/
open Lean Meta Elab Tactic Reap.TreeSearch

namespace Reap.VerifiedReplay

structure ReplayNode where
  node_index : Nat
  state : Array String
  kind : String
  tactic : String
  children : Array Nat
deriving FromJson

structure ReplayPlan where
  schema_version : String
  nodes : Array ReplayNode
deriving FromJson

meta def actualState : TacticM (Array String) := do
  return (← (← getUnsolvedGoals).mapM fun goal => do
    return toString (← Meta.ppGoal goal)).toArray

meta partial def replay (ctx : ProofCheckContext) (nodes : Array ReplayNode)
    (index : Nat) (fuel : Nat) : TacticM (Int × Array Json) := do
  if fuel == 0 then throwError "verified replay exhausted structural bound"
  let some node := nodes[index]? | throwError "verified replay missing node"
  let before ← actualState
  unless before == node.state do
    throwError "verified replay state mismatch at node {node.node_index}: expected {toJson node.state}, actual {toJson before}"
  if node.kind == "terminal" then
    unless before.isEmpty && node.children.isEmpty && node.tactic.isEmpty do
      throwError "verified replay nonterminal leaf"
    return (0, #[])
  else if node.kind == "OR" then
    unless before.size == 1 && node.children.size == 1 && !node.tactic.isEmpty do
      throwError "verified replay invalid OR shape"
    let heartbeats := reap.heartbeats.get (← getOptions)
    match ← evalTacticStrNoFinalCheck ctx node.tactic heartbeats with
    | .error err => throwError "verified replay tactic failed at node {node.node_index}: {toJson err}"
    | .ok _ => pure ()
    let after ← actualState
    let (rest, rows) ← replay ctx nodes node.children[0]! (fuel - 1)
    let value := rest - 1
    let row := json%{
      node_index: $(node.node_index), state: $(before), tactic: $(node.tactic),
      next_state: $(after), "return": $(value)
    }
    return (value, #[row] ++ rows)
  else if node.kind == "AND" then
    unless before.size >= 2 && node.children.size == before.size && node.tactic.isEmpty do
      throwError "verified replay invalid AND shape"
    let goals := (← getUnsolvedGoals).toArray
    if ← goals.toList.anyM goalContainsExprMVar then
      throwError "verified replay dependent/metavariable AND goals are unsupported"
    let mut worst : Option Int := none
    let mut rows := #[]
    for child in node.children, goal in goals do
      if ← goal.isAssigned then
        throwError "verified replay shared-metavariable branch was solved by a sibling; labels unsupported"
      setGoals [goal]
      let (value, branchRows) ← replay ctx nodes child (fuel - 1)
      unless (← getUnsolvedGoals).isEmpty do throwError "verified replay unfinished AND branch"
      worst := some (worst.map (min value) |>.getD value)
      rows := rows ++ branchRows
    setGoals goals.toList
    pruneSolvedGoals
    unless (← getUnsolvedGoals).isEmpty do throwError "verified replay unfinished conjunction"
    return (worst.getD 0, rows)
  else throwError "verified replay unknown node kind"

elab "reapVerifiedReplay " planPath:str tracePath:str : tactic => do
  let raw ← IO.FS.readFile planPath.getString
  let parsed ← match Json.parse raw with
    | .ok value => pure value
    | .error error => throwError "verified replay invalid JSON: {error}"
  let plan ← match (fromJson? parsed : Except String ReplayPlan) with
    | .ok value => pure value
    | .error error => throwError "verified replay invalid plan: {error}"
  unless plan.schema_version == "reap.verified-replay.plan.v1" do
    throwError "verified replay unsupported schema"
  let ctx ← mkProofCheckContext
  let some declaration ← Lean.Elab.Term.getDeclName?
    | throwError "verified replay requires a named theorem declaration"
  let (value, rows) ← replay ctx plan.nodes 0 (plan.nodes.size + 1)
  unless (← getUnsolvedGoals).isEmpty do throwError "verified replay left goals"
  match ← checkProof ctx with
  | .error error => throwError "verified replay final proof check: {toJson error}"
  | .ok _ => pure ()
  IO.FS.writeFile tracePath.getString (json%{
    schema_version: "reap.verified-replay.trace.v1", complete: true,
    "theorem": $(declaration.toString), root_return: $(value), rows: $(rows)
  }).pretty

end Reap.VerifiedReplay


-- Every recurrence assumption is visible in the local goal.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace MultiroundCandidates.CubeAccumulator

theorem cube_accumulator (f : Nat → Nat)
    (hzero : f 0 = 0)
    (hstep : ∀ n, f (n + 1) = f n + (n + 1) ^ 3) :
    ∀ n, 4 * f n = (n * (n + 1)) ^ 2 := by
  reapVerifiedReplay "/replay-out/05/plan.json" "/replay-out/05/trace.json"

end MultiroundCandidates.CubeAccumulator

#print axioms MultiroundCandidates.CubeAccumulator.cube_accumulator
