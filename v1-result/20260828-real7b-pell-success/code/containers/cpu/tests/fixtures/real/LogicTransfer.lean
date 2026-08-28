import ReapRuntime

-- Real service only. All binders remain in the initial goal; no prescribed proof.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace StrictV1Fixtures.LogicTransfer

theorem logic_transfer :
    ∀ (P Q R S : Prop), (P → Q) → (R → S) → P ∧ R → Q ∧ S := by
  reapTrainingMCTS

end StrictV1Fixtures.LogicTransfer
