import Reap.VerifiedCollector

open Lean Meta Elab Tactic Reap.TreeSearch TreeSearch Reap.TreeSearch.MCTS

example : True := by
  run_tac do
    let original ← NodeData.fromState
    let edge : EdgeData := {tacticStr := "fixture", premise := #[], probability := 1.0, value := 0, numVisit := 1}
    unless edgeStepCost edge == 1 && edgeStepCost {edge with isFocus := true} == 0 do
      throwError "collector action/focus step costs changed"
    let child (value : Float) (solved := false) : Node NodeData (EdgeData × Nat) :=
      {data := {original with numVisit := 1, valueSum := value, isSolved := solved}}
    let nodes := #[
      {data := {original with numVisit := 2}, children := #[(edge, 1), (edge, 2)]},
      child 0 true, child (-2)]
    let some rootRef := nodes[0]? | throwError "missing OR root"
    let (root, _) ← StateT.run (s := nodes) (resolve rootRef)
    for gamma in [0.9, 0.99] do
      let params := {SearchHyperparameters.fromOptions (← getOptions) with visitDiscount := gamma}
      let scores := computePUCTScores params root
      unless scores[0]!.2.Q == 1 && Float.abs (scores[1]!.2.Q - gamma.pow 2) < 0.000000001 do
        throwError "collector PUCT distance mapping changed"
      unless backupValueForParent root (-3) == -3 do throwError "OR return was discounted"
    let andNodes := #[
      {data := {original with toPlay := .andNode, numVisit := 2},
        children := #[({edge with isFocus := true}, 1), ({edge with isFocus := true}, 2),
                       ({edge with isFocus := true}, 3)]},
      child (-2), child (-4), child 0 true]
    let some andRef := andNodes[0]? | throwError "missing AND root"
    let (andRoot, _) ← StateT.run (s := andNodes) (resolve andRef)
    unless backupValueForParent andRoot (-100) == -4 do throwError "AND longest branch changed"
    for gamma in [0.9, 0.99] do
      let params := {SearchHyperparameters.fromOptions (← getOptions) with visitDiscount := gamma}
      let scores := computePUCTScores params andRoot
      unless scores[1]!.2.Q > scores[0]!.2.Q && scores[2]!.2.Q == -Float.inf do
        throwError "AND order/proved child exclusion changed"
    for value in [(-1.0 : Float), -3.0, -64.0] do
      unless (← Reap.VerifiedCollector.checkedValue 64 value) == value do
        throwError "collector negated a value twice"
    for value in [(0.0 : Float), 1.0, -65.0, Float.inf, (0.0 / 0.0)] do
      let rejected ← try
        discard <| Reap.VerifiedCollector.checkedValue 64 value
        pure false
      catch _ => pure true
      unless rejected do throwError "invalid collector value accepted"
  trivial

#eval IO.println "VERIFIED_COLLECTOR_UNIT_OK"
