import ReapRuntime

-- Real policy/value service only; no mock responses, filtered proofs or preset solution.
set_option reap.model "REAL-Prover"
set_option reap.num_premises 0
set_option reap.num_samples 2
set_option reap.max_tokens 128
set_option reap.max_steps 32
set_option reap.max_goals 64
set_option reap.visit_discount 990

namespace StrictV1Fixtures.TreeMirrorLeaves

inductive ForkTree where
  | leaf : Nat → ForkTree
  | fork : ForkTree → ForkTree → ForkTree

def mirror : ForkTree → ForkTree
  | .leaf n => .leaf n
  | .fork left right => .fork (mirror right) (mirror left)

def leaves : ForkTree → List Nat
  | .leaf n => [n]
  | .fork left right => leaves left ++ leaves right

theorem mirror_leaves (tree : ForkTree) :
    leaves (mirror tree) = (leaves tree).reverse := by
  reapTrainingMCTS

end StrictV1Fixtures.TreeMirrorLeaves
