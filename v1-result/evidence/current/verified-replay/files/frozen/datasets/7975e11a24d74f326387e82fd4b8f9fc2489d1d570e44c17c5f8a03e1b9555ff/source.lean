import ReapRuntime

-- Every recurrence assumption is visible in the local goal.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace MultiroundCandidates.ScaledTriangular

theorem scaled_triangular (f : Nat → Nat) (a b : Nat)
    (hzero : f 0 = a)
    (hstep : ∀ n, f (n + 1) = f n + b * (n + 1)) :
    ∀ n, 2 * f n = 2 * a + b * n * (n + 1) := by
  reapTrainingMCTS

end MultiroundCandidates.ScaledTriangular
