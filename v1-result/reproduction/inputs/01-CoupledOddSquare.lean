import ReapRuntime

-- Every recurrence assumption is visible in the local goal.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace MultiroundCandidates.CoupledOddSquare

theorem coupled_odd_square (f g : Nat → Nat)
    (hfzero : f 0 = 1) (hgzero : g 0 = 0)
    (hfstep : ∀ n, f (n + 1) = f n + 2)
    (hgstep : ∀ n, g (n + 1) = g n + f n) :
    ∀ n, f n = 2 * n + 1 ∧ g n = n * n := by
  reapTrainingMCTS

end MultiroundCandidates.CoupledOddSquare
