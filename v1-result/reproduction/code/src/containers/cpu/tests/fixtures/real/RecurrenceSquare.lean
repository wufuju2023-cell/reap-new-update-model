import ReapRuntime

-- All recurrence information is visible in the goal's local hypotheses.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace StrictV1Fixtures.RecurrenceSquare

theorem recurrence_square (f : Nat → Nat) (hzero : f 0 = 0)
    (hstep : ∀ n, f (n + 1) = f n + (2 * n + 1)) :
    ∀ n, f n = n * n := by
  reapTrainingMCTS

end StrictV1Fixtures.RecurrenceSquare
