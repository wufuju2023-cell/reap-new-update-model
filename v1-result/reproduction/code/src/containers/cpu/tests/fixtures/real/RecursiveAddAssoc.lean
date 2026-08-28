import ReapRuntime

-- Real policy/value service only. No prescribed candidate or successful proof.
-- A proof found before an update is valid, but does not cover online training.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace StrictV1Fixtures.RecursiveAddAssoc

def foldAdd (a : Nat) : Nat → Nat
  | 0 => a
  | n + 1 => Nat.succ (foldAdd a n)

theorem foldAdd_assoc (a b c : Nat) :
    foldAdd (foldAdd a b) c = foldAdd a (foldAdd b c) := by
  reapTrainingMCTS

end StrictV1Fixtures.RecursiveAddAssoc
