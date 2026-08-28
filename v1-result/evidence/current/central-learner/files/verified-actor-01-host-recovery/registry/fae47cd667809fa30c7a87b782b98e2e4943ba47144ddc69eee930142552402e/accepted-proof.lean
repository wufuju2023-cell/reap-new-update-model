import ReapRuntime

set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 16
set_option reap.max_goals 64
set_option reap.visit_discount 990

theorem LearnerLoop.AffineStrideThree (f : Nat → Nat)
    (h0 : f 0 = 2) (hs : ∀ n, f (n + 1) = f n + 3) :
    ∀ n, f n = 2 + 3 * n := by
  intro n
  induction n <;> simp_all [hs, pow_succ, mul_add, mul_one, add_assoc, add_comm, add_left_comm]

#print axioms LearnerLoop.AffineStrideThree
