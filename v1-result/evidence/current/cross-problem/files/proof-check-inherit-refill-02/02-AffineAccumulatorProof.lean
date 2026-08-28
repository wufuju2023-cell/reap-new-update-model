import ReapRuntime

-- Every recurrence assumption is visible in the local goal.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace MultiroundCandidates.AffineAccumulator

theorem affine_accumulator (f g : Nat → Nat) (a b : Nat)
    (hfzero : f 0 = a) (hgzero : g 0 = 0)
    (hfstep : ∀ n, f (n + 1) = f n + b)
    (hgstep : ∀ n, g (n + 1) = g n + f n) :
    ∀ n, f n = a + b * n ∧
      2 * g n + b * n = 2 * a * n + b * n * n := by
  intro n
  induction n <;> simp_all [Nat.mul_add, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
  nlinarith

end MultiroundCandidates.AffineAccumulator

#print axioms MultiroundCandidates.AffineAccumulator.affine_accumulator
