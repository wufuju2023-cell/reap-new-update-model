import ReapRuntime

-- Every recurrence assumption is visible in the local goal.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace MultiroundCandidates.DifferenceInvariant

theorem difference_invariant (f g : Nat → Nat) (a b : Nat)
    (hfzero : f 0 = a) (hgzero : g 0 = b)
    (hfstep : ∀ n, f (n + 1) = f n + (3 * n + 1))
    (hgstep : ∀ n, g (n + 1) = g n + (3 * n + 2)) :
    ∀ n, f n + n + b = g n + a := by
  reapTrainingMCTS

end MultiroundCandidates.DifferenceInvariant
