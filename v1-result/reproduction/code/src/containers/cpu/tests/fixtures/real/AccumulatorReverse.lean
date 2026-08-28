import ReapRuntime

-- Real policy/value service only; the model chooses all tactics.
-- This normally invites induction with a generalized accumulator, not a forced trace.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace StrictV1Fixtures.AccumulatorReverse

def reverseInto {α : Type} : List α → List α → List α
  | [], acc => acc
  | x :: xs, acc => reverseInto xs (x :: acc)

theorem reverseInto_append {α : Type} (xs ys acc : List α) :
    reverseInto (xs ++ ys) acc = reverseInto ys (reverseInto xs acc) := by
  reapTrainingMCTS

end StrictV1Fixtures.AccumulatorReverse
