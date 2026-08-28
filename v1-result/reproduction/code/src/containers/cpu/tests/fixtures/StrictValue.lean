import Reap.Tactic.Generator

open Lean Elab Tactic

-- Direct real HTTP client regression with a local test server, not GPU training.
elab "strictValueProbe" : tactic => do
  let generator ← TacticGenerator.getClient
  let value ← TacticGenerator.generateValueFromPrompt generator (← getOptions)
    "⊢ True" #[] "strict-value-test"
  IO.println s!"VALUE_PROBE: {value}"
  evalTactic (← `(tactic| trivial))

theorem value_probe : True := by
  strictValueProbe
