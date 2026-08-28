import ReapRuntime
import Mathlib

/-!
Replacement collection with five complete targets. Original declarations
for the retained exercises are byte-identical to the first collection.
The teacher has read the fixed Mathlib Archive proof of IMO 1988 Q6.
This student file does not import Archive or teacher proof files.
-/

namespace CodexMathFive.FunctionalEquation

def Satisfies (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (x + f y) = f x + y

def Target : Prop :=
  ∀ f : ℤ → ℤ, Satisfies f ↔
    ((∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x))

def CourseInvolution : Prop :=
  ∀ f : ℤ → ℤ, Satisfies f → f 0 = 0 ∧ ∀ y : ℤ, f (f y) = y

def CourseAdditivity : Prop :=
  ∀ f : ℤ → ℤ, Satisfies f → ∀ x y : ℤ, f (x + y) = f x + f y

def CourseIntegerLinearity : Prop :=
  ∀ f : ℤ → ℤ, (∀ x y : ℤ, f (x + y) = f x + f y) →
    ∀ n : ℤ, f n = n * f 1

end CodexMathFive.FunctionalEquation

namespace CodexMathFive.PrimeSquareSum

def Target : Prop :=
  ∀ p : ℕ, Nat.Prime p → p % 4 = 3 →
    ∀ x y : ℤ, (p : ℤ) ∣ x ^ 2 + y ^ 2 → (p : ℤ) ∣ x ∧ (p : ℤ) ∣ y

def CourseMinusOne : Prop :=
  ∀ p : ℕ, Nat.Prime p → p % 4 = 3 →
    ¬ ∃ z : ZMod p, z ^ 2 = -1

def CourseResiduePair : Prop :=
  ∀ p : ℕ, Nat.Prime p → p % 4 = 3 →
    ∀ x y : ZMod p, x ^ 2 + y ^ 2 = 0 → x = 0 ∧ y = 0

end CodexMathFive.PrimeSquareSum

namespace CodexMathFive.Pell

def step (p : ℕ × ℕ) : ℕ × ℕ :=
  (3 * p.1 + 4 * p.2, 2 * p.1 + 3 * p.2)

def pellPair : ℕ → ℕ × ℕ
  | 0 => (3, 2)
  | k + 1 => step (pellPair k)

def Target : Prop :=
  ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1

def CourseInvariant : Prop :=
  ∀ x y : ℕ, x ^ 2 = 2 * y ^ 2 + 1 →
    (step (x, y)).1 ^ 2 = 2 * (step (x, y)).2 ^ 2 + 1

def CoursePositiveGrowth : Prop :=
  ∀ x y : ℕ, 0 < x → 0 < y → x < (step (x, y)).1 ∧ y < (step (x, y)).2

def CourseIterates : Prop :=
  ∀ k : ℕ, k < (pellPair k).1 ∧ k < (pellPair k).2 ∧
    (pellPair k).1 ^ 2 = 2 * (pellPair k).2 ^ 2 + 1

end CodexMathFive.Pell

namespace CodexMathFive.Imo1988Q6

def Equation (a b q : ℕ) : Prop := a ^ 2 + b ^ 2 = (a * b + 1) * q

def Target : Prop :=
  ∀ a b : ℕ, 0 < a → 0 < b → (a * b + 1) ∣ (a ^ 2 + b ^ 2) →
    ∃ k : ℕ, a ^ 2 + b ^ 2 = (a * b + 1) * k ^ 2

def CoursePositiveQuotient : Prop :=
  ∀ a b : ℕ, 0 < a → 0 < b → (a * b + 1) ∣ (a ^ 2 + b ^ 2) →
    ∃ q : ℕ, 0 < q ∧ Equation a b q

def CourseOtherRoot : Prop :=
  ∀ a b q : ℤ, a ^ 2 + b ^ 2 = (a * b + 1) * q →
    (q * a - b) ^ 2 + a ^ 2 = ((q * a - b) * a + 1) * q ∧
    (q * a - b) * b = a ^ 2 - q

def CourseDescent : Prop :=
  ∀ a b q : ℕ, 0 < a → a < b → Equation a b q →
    ∃ c : ℕ, c ≤ a ∧ Equation c a q

def CourseBoundary : Prop :=
  ∀ a q : ℕ, (Equation a 0 q ∨ Equation a a q) → ∃ k : ℕ, q = k ^ 2

end CodexMathFive.Imo1988Q6

namespace CodexMathFive.Schur

noncomputable def schurPolynomial (n : ℕ) : Polynomial ℤ :=
  (∏ i ∈ Finset.Icc 1 n, (Polynomial.X - Polynomial.C (i : ℤ))) - 1

def Target : Prop :=
  ∀ n : ℕ, 1 ≤ n → Irreducible (schurPolynomial n)

def CourseMonicDegree : Prop :=
  ∀ n : ℕ, 1 ≤ n → (schurPolynomial n).Monic ∧ (schurPolynomial n).natDegree = n

def CourseIntegerValues : Prop :=
  ∀ n i : ℕ, 1 ≤ i → i ≤ n → (schurPolynomial n).eval (i : ℤ) = -1

def CourseProperFactorDegrees : Prop :=
  ∀ n : ℕ, 1 ≤ n → ∀ g h : Polynomial ℤ,
    schurPolynomial n = g * h → ¬ IsUnit g → ¬ IsUnit h →
      g.natDegree < n ∧ h.natDegree < n

def CourseOppositeFactors : Prop :=
  ∀ n : ℕ, 1 ≤ n → ∀ g h : Polynomial ℤ,
    schurPolynomial n = g * h → g.natDegree < n → h.natDegree < n →
      g + h = 0

def CourseMonicNotNegativeSquare : Prop :=
  ∀ f : Polynomial ℤ, f.Monic → ∀ g : Polynomial ℤ, f ≠ -(g ^ 2)

end CodexMathFive.Schur

namespace CodexMathFive.Pell

def CourseRecurrence : Prop :=
  ∀ s : ℕ → ℕ × ℕ, s 0 = (3, 2) →
    (∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2)) →
    ∀ k : ℕ, k < (s k).1 ∧ k < (s k).2 ∧
      (s k).1 ^ 2 = 2 * (s k).2 ^ 2 + 1

end CodexMathFive.Pell


namespace CodexMathFive.Pell

/-- Construct a sequence from the prescribed initial pair and recurrence. -/
def CourseSequenceExists : Prop :=
  ∃ s : ℕ → ℕ × ℕ, s 0 = (3, 2) ∧
    ∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2)

/-- Use a recurrence sequence to produce witnesses above an arbitrary bound. -/
def CourseUnboundedFromRecurrence : Prop :=
  ∀ s : ℕ → ℕ × ℕ, s 0 = (3, 2) →
    (∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2)) →
    ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1

end CodexMathFive.Pell

namespace CodexMathFive.Pell

def CourseIndexedWitness : Prop :=
  ∀ s : ℕ → ℕ × ℕ,
    (∀ k : ℕ, k < (s k).1 ∧ k < (s k).2 ∧
      (s k).1 ^ 2 = 2 * (s k).2 ^ 2 + 1) →
    ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1

end CodexMathFive.Pell

set_option reap.temperature 99
set_option reap.c_init 1500
set_option reap.progressive_sampling_c 500

-- STUDENT_VERIFIED_BEGIN CodexMathFive.Pell.StudentLibraryRecurrence
theorem CodexMathFive.Pell.StudentLibraryRecurrence :
  ∀ s : ℕ → ℕ × ℕ, s 0 = (3, 2) →
    (∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2)) →
    ∀ k : ℕ, k < (s k).1 ∧ k < (s k).2 ∧
      (s k).1 ^ 2 = 2 * (s k).2 ^ 2 + 1 := by
  intro s h₀ h₁ k
  induction k <;> simp_all
  ring_nf at h₀ h₁ ⊢
  simp_all
  omega
-- STUDENT_VERIFIED_END CodexMathFive.Pell.StudentLibraryRecurrence

#print axioms CodexMathFive.Pell.StudentLibraryRecurrence

-- STUDENT_VERIFIED_BEGIN CodexMathFive.Pell.StudentLibraryIndexedWitness
theorem CodexMathFive.Pell.StudentLibraryIndexedWitness :
  ∀ s : ℕ → ℕ × ℕ,
    (∀ k : ℕ, k < (s k).1 ∧ k < (s k).2 ∧
      (s k).1 ^ 2 = 2 * (s k).2 ^ 2 + 1) →
    ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1 := by
  intro s hs B
  specialize hs B
  suffices s B = s B by rw [this] at hs; aesop
  simp only [Prod.mk.injEq, and_self]
-- STUDENT_VERIFIED_END CodexMathFive.Pell.StudentLibraryIndexedWitness

#print axioms CodexMathFive.Pell.StudentLibraryIndexedWitness

-- STUDENT_VERIFIED_BEGIN CodexMathFive.Pell.StudentLibrarySequenceExists
theorem CodexMathFive.Pell.StudentLibrarySequenceExists :
  ∃ s : ℕ → ℕ × ℕ, s 0 = (3, 2) ∧
    ∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2) := by
  exact ⟨fun k ↦ Nat.recOn k (3, 2) fun k s ↦ (3 * s.1 + 4 * s.2, 2 * s.1 + 3 * s.2), by simp⟩
-- STUDENT_VERIFIED_END CodexMathFive.Pell.StudentLibrarySequenceExists

#print axioms CodexMathFive.Pell.StudentLibrarySequenceExists

-- STUDENT_VERIFIED_BEGIN CodexMathFive.Pell.StudentLibraryUnboundedFromRecurrence
theorem CodexMathFive.Pell.StudentLibraryUnboundedFromRecurrence :
  ∀ s : ℕ → ℕ × ℕ, s 0 = (3, 2) →
    (∀ k : ℕ, s (k + 1) =
      (3 * (s k).1 + 4 * (s k).2, 2 * (s k).1 + 3 * (s k).2)) →
    ∀ B : ℕ, ∃ x y : ℕ, B < x ∧ B < y ∧ x ^ 2 = 2 * y ^ 2 + 1 := by
  have student_fact_1 := @_root_.CodexMathFive.Pell.StudentLibraryRecurrence
  have student_fact_2 := @_root_.CodexMathFive.Pell.StudentLibraryIndexedWitness
  exact fun s ↦ by tauto
-- STUDENT_VERIFIED_END CodexMathFive.Pell.StudentLibraryUnboundedFromRecurrence

#print axioms CodexMathFive.Pell.StudentLibraryUnboundedFromRecurrence
