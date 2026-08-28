import Reap.Training.SelectionValueRefresh

open Lean Meta Elab Tactic Reap.TreeSearch TreeSearch
open Reap.TreeSearch.MCTS

set_option reap.progressive_sampling_c 0

meta def guardHistory (before after : Array (Node NodeData (EdgeData × Nat))) : TacticM Unit := do
  unless before.size == after.size do throwError "tree size changed"
  for (old, fresh) in before.zip after do
    unless old.data.key == fresh.data.key && old.data.toPlay == fresh.data.toPlay &&
           old.data.isSolved == fresh.data.isSolved && old.data.numVisit == fresh.data.numVisit &&
           old.data.numEvaluations == fresh.data.numEvaluations && old.data.valueSum == fresh.data.valueSum do
      throwError "historical node/proof/visit data changed"
    unless (toJson old.children).compress == (toJson fresh.children).compress do
      throwError "historical edges changed"
    let beforeJson ← ppNodeData old.data
    let afterJson ← ppNodeData {fresh.data with selectionValue? := none}
    unless beforeJson.compress == afterJson.compress do throwError "saved Lean state changed"

meta def fixtureTree : TacticM (Array (Node NodeData (EdgeData × Nat))) := do
  let original ← NodeData.fromState
  let data := {original with numVisit := 11, numEvaluations := 1, valueSum := -110.0}
  let edge : EdgeData := {tacticStr := "fixture", premise := #[], probability := 1.0,
                          value := -31.0, numVisit := 4}
  return #[
    {data, children := #[(edge, 1), (edge, 4), (edge, 5)]},
    {data := {data with toPlay := .andNode, numVisit := 6, valueSum := -66.0},
      children := #[({edge with isFocus := true}, 2), ({edge with isFocus := true}, 3)]},
    {data := {data with numVisit := 3, valueSum := -30.0}},
    {data := {data with isSolved := true, numVisit := 2, valueSum := 0.0}},
    {data := {data with numVisit := 4, valueSum := -80.0}},
    {data := {original with numVisit := 0, valueSum := 0.0}}
  ]

example : True := by
  run_tac do
    let saved ← saveState
    let original ← fixtureTree
    let calls ← IO.mkRef (0 : Nat)
    let control : SelectionValueRefresh := {
      currentVersion := pure 1
      evalValue := fun _ => do
        let index ← calls.modifyGet fun n => (n, n + 1)
        return (#[(-2.0 : Float), -7.0, -4.0])[index]!
    }
    let (_, refreshed) ← StateT.run (s := original) (refreshSelectionValues control 1 0)
    guardHistory original refreshed
    unless (← calls.get) == 3 do throwError "queried a proof/unvisited/AND node"
    let some andNode := refreshed[1]? | throwError "missing AND"
    let some proved := refreshed[3]? | throwError "missing proof leaf"
    let some unvisited := refreshed[5]? | throwError "missing unvisited"
    unless andNode.data.selectionValue == -7.0 do throwError "AND min mismatch"
    unless proved.data.selectionValue?.isNone && unvisited.data.selectionValue?.isNone do
      throwError "proof or unvisited node cache changed"
    let params := SearchHyperparameters.fromOptions (← getOptions)
    let some oldRef := original[0]? | throwError "missing old root"
    let some newRef := refreshed[0]? | throwError "missing new root"
    let (oldRoot, _) ← StateT.run (s := original) (resolve oldRef)
    let (newRoot, _) ← StateT.run (s := refreshed) (resolve newRef)
    unless selectChild params oldRoot == some 0 && selectChild params newRoot == some 1 do
      throwError "new selection cache did not affect PUCT"
    unless backpropValueTowardsMin oldRoot == backpropValueTowardsMin newRoot do
      throwError "historical training backup was changed"
    saved.restore
  trivial

example : True := by
  run_tac do
    let saved ← saveState
    for failure in ["error", "nan", "positive", "observer"] do
      let original ← fixtureTree
      let calls ← IO.mkRef (0 : Nat)
      let control : SelectionValueRefresh := {
        currentVersion := pure 1
        evalValue := fun goals => do
          let index ← calls.modifyGet fun n => (n, n + 1)
          if index == 1 then
            if failure == "error" then throwError "fixture value error"
            if failure == "nan" then return 0.0 / 0.0
            if failure == "positive" then return 1.0
          if let some goal := goals.head? then goal.assign (mkConst ``True.intro)
          return -4.0
      }
      let observer? : Option MCTSObserver := if failure == "observer" then
        some (fun _ => throwError "fixture observer error") else none
      let (caught, unchanged) ← StateT.run (s := original) do
        try
          refreshSelectionValues control 1 0 observer?
          return false
        catch _ => return true
      unless caught do throwError "failed refresh continued"
      unless unchanged.all (·.data.selectionValue?.isNone) do throwError "partial cache committed"
      guardHistory original unchanged
      saved.restore
      unless !(← getUnsolvedGoals).isEmpty do throwError "failed evaluator mutated caller goals"
  trivial

example : True := by
  run_tac do
    let saved ← saveState
    for nextVersion in [0, 1, 2] do
      let version ← IO.mkRef (0 : Nat)
      let queries ← IO.mkRef (0 : Nat)
      let steps ← IO.mkRef (0 : Nat)
      let control : SelectionValueRefresh := {
        currentVersion := liftM (show IO Nat from version.get)
        evalValue := fun _ => do queries.modify (· + 1); pure (-3.0)
      }
      let observer : MCTSObserver := fun record => do
        if (record.getObjValAs? String "kind").toOption == some "checkpoint" then
          steps.modify (· + 1)
          version.set nextVersion
      let generator : PolicyValueEval := fun _ => pure (-2.0, #[("skip", #[], -1.0)])
      let mut failed := false
      try
        discard <| runMCTS generator (maxSteps := 3) (observer? := some observer)
          (selectionValueRefresh? := some control)
      catch _ => failed := true
      if nextVersion == 2 then
        unless failed && (← steps.get) == 1 && (← queries.get) == 0 do
          throwError "jumped version continued or queried"
      else
        unless !failed && (← steps.get) == 3 && (← queries.get) == nextVersion do
          throwError "same version refreshed twice or skipped valid update"
      saved.restore
  trivial

#eval IO.println "SELECTION_VALUE_REFRESH_UNIT_OK"
