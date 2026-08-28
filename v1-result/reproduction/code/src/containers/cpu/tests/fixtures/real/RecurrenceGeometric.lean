import ReapRuntime

-- No hidden definition or prescribed tactic; the model receives both hypotheses.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace StrictV1Fixtures.RecurrenceGeometric

theorem recurrence_geometric (f : Nat → Nat) (hzero : f 0 = 1)
    (hstep : ∀ n, f (n + 1) = 2 * f n) :
    ∀ n, f n = 2 ^ n := by
  reapTrainingMCTS

end StrictV1Fixtures.RecurrenceGeometric
