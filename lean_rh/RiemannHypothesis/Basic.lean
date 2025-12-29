/-
Copyright (c) 2024 Riemann Hypothesis Formalization Project. All rights reserved.
Released under All Rights Reserved - Viewing and Study Only License as described in the file LICENSE.
Authors: RH Formalization Team
-/
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Complex.Log
import Mathlib.Topology.Basic

/-!
# Basic Definitions for the Riemann Hypothesis

This file contains the foundational definitions needed for the
Riemann Hypothesis formalization.

## Main definitions

* `CriticalStrip`: The region 0 < Re(s) < 1
* `CriticalLine`: The line Re(s) = 1/2
* `IsInCriticalStrip`: Predicate for a complex number being in the critical strip
* `IsOnCriticalLine`: Predicate for a complex number being on the critical line

## References

* [E.C. Titchmarsh, *The Theory of the Riemann Zeta-Function*][titchmarsh1986]
* [H.M. Edwards, *Riemann's Zeta Function*][edwards1974]
-/

namespace RiemannHypothesis

open Complex

/-! ## Critical Strip and Critical Line -/

/-- A complex number is in the critical strip if 0 < Re(s) < 1 -/
def IsInCriticalStrip (s : ℂ) : Prop :=
  0 < s.re ∧ s.re < 1

/-- A complex number is on the critical line if Re(s) = 1/2 -/
def IsOnCriticalLine (s : ℂ) : Prop :=
  s.re = 1/2

/-- The critical line is contained in the critical strip -/
theorem criticalLine_subset_criticalStrip (s : ℂ) (h : IsOnCriticalLine s) :
    IsInCriticalStrip s := by
  unfold IsOnCriticalLine at h
  unfold IsInCriticalStrip
  constructor
  · simp [h]
  · simp [h]

/-! ## Symmetry about the Critical Line -/

/-- Reflection about the critical line: s ↦ 1 - s̄ -/
def reflectAboutCriticalLine (s : ℂ) : ℂ :=
  1 - conj s

/-- Reflection is an involution: applying it twice gives back the original -/
theorem reflect_involution (s : ℂ) :
    reflectAboutCriticalLine (reflectAboutCriticalLine s) = s := by
  unfold reflectAboutCriticalLine
  simp [conj_conj]
  ring

/-- A point is on the critical line iff it equals its reflection -/
theorem onCriticalLine_iff_fixed (s : ℂ) :
    IsOnCriticalLine s ↔ reflectAboutCriticalLine s = s := by
  unfold IsOnCriticalLine reflectAboutCriticalLine
  constructor
  · intro h
    ext
    · simp [h]; ring
    · simp
  · intro h
    have h_re : (1 - conj s).re = s.re := by rw [h]
    simp at h_re
    linarith

/-! ## Trivial Zeros -/

/-- The trivial zeros of ζ are at s = -2, -4, -6, ... -/
def IsTrivialZero (s : ℂ) : Prop :=
  ∃ n : ℕ, n ≥ 1 ∧ s = -2 * n

/-- Trivial zeros are not in the critical strip -/
theorem trivialZeros_not_in_criticalStrip (s : ℂ) (h : IsTrivialZero s) :
    ¬IsInCriticalStrip s := by
  unfold IsTrivialZero at h
  unfold IsInCriticalStrip
  obtain ⟨n, hn, hs⟩ := h
  push_neg
  intro _
  simp [hs]
  have : (n : ℝ) ≥ 1 := by exact Nat.one_le_cast.mpr hn
  linarith

end RiemannHypothesis

