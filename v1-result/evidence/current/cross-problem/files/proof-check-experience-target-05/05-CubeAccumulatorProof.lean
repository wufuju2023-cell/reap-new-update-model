import ReapRuntime

-- Every recurrence assumption is visible in the local goal.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace MultiroundCandidates.CubeAccumulator

theorem cube_accumulator (f : Nat → Nat)
    (hzero : f 0 = 0)
    (hstep : ∀ n, f (n + 1) = f n + (n + 1) ^ 3) :
    ∀ n, 4 * f n = (n * (n + 1)) ^ 2 := by
  intro n
  induction n <;> simp_all [hzero, hstep]
  ring
  nlinarith [hzero, hstep]

end MultiroundCandidates.CubeAccumulator

#print axioms MultiroundCandidates.CubeAccumulator.cube_accumulator
