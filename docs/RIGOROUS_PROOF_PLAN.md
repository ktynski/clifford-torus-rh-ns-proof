# Rigorous Computer-Assisted Proof Plan

## Status: IN PROGRESS

This document outlines the test-driven plan to convert our numerical evidence into a rigorous computer-assisted proof in the style of Hales/Flyspeck.

---

## The Gap Analysis

| What We Have | What We Need | Gap |
|--------------|--------------|-----|
| `mpmath.zeta(s)` (floats) | Certified interval evaluation | **ARB library** |
| Numerical differentiation of E | Symbolic E'' + interval eval | **Derivation** |
| "Asymptotic" hand-wave | Explicit T₀(ε) | **Effective bounds** |
| Zero list from computation | Unconditional bounds | **Circularity audit** |
| Point sampling | Interval covering | **True intervals** |

---

## Phase 1: Certified Zeta Evaluation

### Goal
Replace `mpmath.zeta` with ARB-style evaluation that provides **guaranteed error bounds**.

### Approach
Use Fredrik Johansson's `arb` library (via `python-flint` or direct C binding) which provides:
- Ball arithmetic: value ± certified error
- Rigorous evaluation of ζ(s), Γ(s), etc.
- Proven truncation bounds

### Tests

```
TEST 1.1: ARB Installation
  Input: Import arb/flint
  Expected: No errors
  
TEST 1.2: Known Values
  Input: ζ(2) via ARB
  Expected: Interval containing π²/6 = 1.6449340668...
  Verification: Check interval width < 10⁻³⁰
  
TEST 1.3: Critical Strip
  Input: ξ(0.5 + 14.134725i) via ARB
  Expected: Interval containing 0 (first zero)
  Verification: 0 ∈ [lower, upper]
  
TEST 1.4: Error Propagation
  Input: |ξ(0.3 + 20i)|² via ARB
  Expected: Interval [a, b] with certified bounds
  Verification: b - a < 10⁻²⁰ × |midpoint|
```

### Deliverable
`src/symbolic/arb_zeta_evaluator.py` - Certified evaluator with tests

---

## Phase 2: Symbolic E'' Derivation

### Goal
Derive an **exact symbolic formula** for E''(σ,t) that can be evaluated with intervals.

### The Formula

Starting from E(σ,t) = |ξ(σ+it)|², we have:

```
E = ξ · ξ̄  (where ξ̄ is complex conjugate)

E' = ξ'·ξ̄ + ξ·ξ̄'
   = 2·Re(ξ'·ξ̄)  (since ∂/∂σ and conjugate commute)

E'' = 2·Re(ξ''·ξ̄) + 2·|ξ'|²
```

**Key insight**: E'' = 2|ξ'|² + 2·Re(ξ''·ξ̄)

For convexity, we need E'' > 0, which is guaranteed if:
- |ξ'|² > |Re(ξ''·ξ̄)|, OR
- Re(ξ''·ξ̄) ≥ 0

### Tests

```
TEST 2.1: Formula Verification
  Input: Symbolic E'' formula
  Expected: Matches numerical differentiation to 10⁻¹⁵
  Method: Compare at 100 random points
  
TEST 2.2: Interval Evaluation
  Input: E''(0.3, 20) via symbolic formula + ARB
  Expected: Interval [a, b] with a > 0
  Verification: Lower bound is positive
  
TEST 2.3: Near-Zero Behavior
  Input: E''(0.5, 14.134725) (at first zero)
  Expected: E'' > 0 (Speiser: ξ'(ρ) ≠ 0)
  Verification: Lower bound of interval > 0
  
TEST 2.4: Boundary Behavior
  Input: E''(0.01, t) and E''(0.99, t) for various t
  Expected: E'' > 0 (away from critical line)
  Verification: All lower bounds positive
```

### Deliverable
`src/symbolic/symbolic_E_derivatives.py` - Symbolic formula + interval evaluator

---

## Phase 3: Explicit T₀ from Effective Bounds

### Goal
Compute an **explicit, computable T₀** such that for t ≥ T₀, asymptotic dominance is proven.

### Required Bounds (from literature)

**Trudgian (2014)**: For the S(T) error in zero counting:
```
|S(T)| < 0.137 log(T) + 0.443 log(log(T)) + 4.350  for T ≥ e
```

**Zero density**: N(T) = (T/2π)log(T/2πe) + 7/8 + S(T)

**Gap bounds**: Consecutive zeros γₙ, γₙ₊₁ satisfy:
```
γₙ₊₁ - γₙ ≥ c/log(γₙ)  for explicit c
```

### The Computation

For the anchoring term A(s) and curvature bound |K|:

```
A(s) ≥ c₁(ε) · log³(t)     (from zero density + gap bounds)
|K| ≤ c₂ · log²(t)         (from Voronin + growth bounds)

Ratio: A/|K| ≥ (c₁/c₂) · log(t)

For A > |K|, need: log(t) > c₂/c₁(ε)
Therefore: T₀(ε) = exp(c₂/c₁(ε))
```

### Tests

```
TEST 3.1: Trudgian Bound Verification
  Input: N(1000) via Riemann-von Mangoldt
  Expected: |N(1000) - 649| ≤ S_bound(1000)
  Verification: Known N(1000) = 649 is within bounds
  
TEST 3.2: Explicit c₁(ε) Computation
  Input: ε = 0.1 (distance from critical line)
  Expected: Computable lower bound on A(s)
  Verification: Formula with explicit constants
  
TEST 3.3: Explicit c₂ Computation
  Input: Growth bounds on |ξ''|/|ξ|
  Expected: Computable upper bound on |K|
  Verification: Formula with explicit constants
  
TEST 3.4: T₀ Computation
  Input: ε = 0.1
  Expected: Explicit T₀ (e.g., T₀ = 10⁶)
  Verification: For t > T₀, A(s) > |K| provably
  
TEST 3.5: Finite Window Coverage
  Input: Interval verification for t ∈ [1, T₀]
  Expected: E'' > 0 on grid (now with ARB)
  Verification: All intervals have positive lower bound
```

### Deliverable
`src/symbolic/explicit_T0_computation.py` - Computes T₀ with proven bounds

---

## Phase 4: Circularity Audit

### Goal
Ensure **no step assumes the conclusion** (that zeros are on the line).

### Audit Categories

Each inequality/bound must be tagged as:

| Category | Description | Allowed? |
|----------|-------------|----------|
| **A** | Pure analysis (no zero knowledge) | ✅ Yes |
| **B** | Unconditional zero-counting (R-vM) | ✅ Yes |
| **C** | Computed zeros + remainder bound | ⚠️ If remainder proven |
| **D** | Assumes zeros on line | ❌ No |

### Tests

```
TEST 4.1: Anchoring Term Audit
  Input: Definition of A(s)
  Question: Does it depend on zero locations?
  Expected: Category B (uses N(T) bounds, not specific zeros)
  
TEST 4.2: Curvature Bound Audit
  Input: Definition of |K| bound
  Question: Does it assume RH?
  Expected: Category A (analytic bound on derivatives)
  
TEST 4.3: Hadamard Pairing Audit
  Input: Pairing argument (ρ, 1-ρ)
  Question: Does symmetry argument assume σ = 1/2?
  Expected: Category A (functional equation is unconditional)
  
TEST 4.4: Full Dependency Graph
  Input: All lemmas in proof
  Expected: No path contains Category D
  Verification: Automated graph traversal
```

### Deliverable
`src/symbolic/circularity_audit.py` - Dependency checker with categories

---

## Phase 5: Navier-Stokes Formalization

### Goal
Address the critique that "Beltrami regularity ≠ general data regularity".

### The Gap

The Clay problem requires regularity for **all** smooth divergence-free data, not just Beltrami.

### Approach Options

**Option A: Density Argument (if valid)**
- Prove: Beltrami flows are dense in relevant topology
- Prove: Regularity is continuous in that topology
- Conclude: Regularity extends to all data

**Option B: Decomposition + Stability**
- Prove: Any flow = Beltrami + remainder
- Prove: Remainder stays small under evolution
- Prove: Small remainder + Beltrami regularity → full regularity

**Option C: Scope Reduction**
- Acknowledge: Proof is for Beltrami class only
- Clarify: This is still interesting but not Millennium solution

### Tests

```
TEST 5.1: Density in Which Topology?
  Input: Beltrami flows
  Expected: Dense in H^k for which k?
  Verification: Literature reference or proof

TEST 5.2: Continuity of Regularity
  Input: Definition of "regularity"
  Question: Is it continuous in the topology where Beltrami is dense?
  Expected: Either proof or counterexample
  
TEST 5.3: Remainder Bound
  Input: u = u_B + u_⊥ decomposition
  Expected: ||u_⊥(t)|| ≤ f(||u_⊥(0)||, t) with explicit f
  Verification: Numerical + analytic bound
  
TEST 5.4: Coupling Control
  Input: Interaction term ⟨u_⊥, nonlinear(u_B)⟩
  Expected: Bounded by g(||u_⊥||, ||u_B||, Ω)
  Verification: Explicit bound with proof
```

### Deliverable
`src/symbolic/ns_formal_gap_analysis.py` - Identifies exactly what's proven vs. claimed

---

## Implementation Timeline

| Phase | Duration | Deliverable | Tests |
|-------|----------|-------------|-------|
| 1 | 3-5 days | ARB evaluator | 4 |
| 2 | 3-5 days | Symbolic E'' | 4 |
| 3 | 5-7 days | Explicit T₀ | 5 |
| 4 | 2-3 days | Circularity audit | 4 |
| 5 | 7-14 days | NS formalization | 4 |

**Total: 3-5 weeks for rigorous RH, additional 2 weeks for NS**

---

## Success Criteria

### For RH (Computer-Assisted Proof)

1. ✅ ARB-certified ξ(s) evaluation with error < 10⁻³⁰
2. ✅ Symbolic E'' formula verified against ARB evaluation
3. ✅ Explicit T₀ computed with proven constants
4. ✅ Interval verification of E'' > 0 on [ε, 0.5-ε] × [1, T₀]
5. ✅ No circularity (all dependencies audited as Category A or B)

### For NS (Either full proof or honest scope)

1. Either: Prove density + continuity argument rigorously
2. Or: Prove decomposition + remainder bounds rigorously  
3. Or: Clearly state scope is Beltrami class only

---

## Next Steps

1. **Install ARB/flint** and create basic evaluator
2. **Write TEST 1.1-1.4** before any implementation
3. **Run tests** - they should fail initially
4. **Implement** until tests pass
5. **Repeat** for each phase

This is the Flyspeck/Kepler-conjecture approach: define success criteria first, then implement.
