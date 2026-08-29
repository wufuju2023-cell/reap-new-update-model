import ReapRuntime
import Mathlib

set_option reap.c_init 1500
set_option reap.progressive_sampling_c 500

namespace ChallengeV1.H1Curriculum

/-- Initial lesson: rehearse the complete successor/IH/algebra composition at lower degree. -/
def CourseSquareTelescope : Prop :=
  ∀ n : ℕ,
    (∑ i ∈ Finset.range n, (((i : ℤ) + 1) ^ 2 - (i : ℤ) ^ 2)) = (n : ℤ) ^ 2

/-- Lesson 2: isolate exactly one cubic induction step, with the IH supplied. -/
def CourseCubeSuccessorFromIH : Prop :=
  ∀ n : ℕ,
    (∑ i ∈ Finset.range n, (((i : ℤ) + 1) ^ 3 - (i : ℤ) ^ 3)) = (n : ℤ) ^ 3 →
      (∑ i ∈ Finset.range (n + 1), (((i : ℤ) + 1) ^ 3 - (i : ℤ) ^ 3)) =
        ((n + 1 : ℕ) : ℤ) ^ 3

end ChallengeV1.H1Curriculum

/-- The original target, retained without mathematical rewriting. -/
def ChallengeV1.H1 : Prop :=
  ∀ n : ℕ,
    (∑ i ∈ Finset.range n, (((i : ℤ) + 1) ^ 3 - (i : ℤ) ^ 3)) = (n : ℤ) ^ 3
