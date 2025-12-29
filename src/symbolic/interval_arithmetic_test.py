#!/usr/bin/env python3
"""
Interval Arithmetic Verification for RH Convexity

This script tests whether E''(σ,t) > 0 can be rigorously verified
using interval arithmetic on the finite window [0, T₀].

The critique correctly notes that "validated numerics" requires
actual interval arithmetic, not point sampling.

Key question: Can we RIGOROUSLY certify E_σσ > 0 for all (σ,t) 
in the finite window?
"""

from mpmath import mp, mpf, mpc, zeta, gamma, pi, log, exp, diff, fabs
import numpy as np

# Set high precision
mp.dps = 50

def xi(s):
    """Completed zeta function ξ(s) = (1/2)s(s-1)π^(-s/2)Γ(s/2)ζ(s)"""
    if s == 0 or s == 1:
        return mpf(0)
    return mpf('0.5') * s * (s - 1) * (pi ** (-s/2)) * gamma(s/2) * zeta(s)

def E(sigma, t):
    """Energy functional E(σ,t) = |ξ(σ+it)|²"""
    s = mpc(sigma, t)
    xi_val = xi(s)
    return fabs(xi_val) ** 2

def numerical_second_derivative(sigma, t, h=1e-6):
    """
    Compute ∂²E/∂σ² numerically using central difference.
    
    For RIGOROUS interval arithmetic, this would need to be replaced
    with proper interval bounds.
    """
    E_plus = E(sigma + h, t)
    E_center = E(sigma, t)
    E_minus = E(sigma - h, t)
    return (E_plus - 2*E_center + E_minus) / (h**2)

def test_convexity_point(sigma, t):
    """Test if E''(σ,t) > 0 at a single point."""
    E_ss = numerical_second_derivative(sigma, t)
    return float(E_ss) > 0, float(E_ss)

def test_finite_window(T0=100, sigma_points=20, t_points=50):
    """
    Test convexity on finite window [0, T₀].
    
    NOTE: This is POINT SAMPLING, not interval arithmetic!
    The critique correctly identifies this as insufficient for a proof.
    """
    print(f"Testing convexity on finite window t ∈ [1, {T0}]")
    print("="*60)
    print("WARNING: This is point sampling, NOT rigorous interval arithmetic!")
    print("="*60)
    
    sigma_vals = np.linspace(0.1, 0.45, sigma_points)  # Left half-strip
    t_vals = np.linspace(1, T0, t_points)
    
    min_E_ss = float('inf')
    min_location = None
    violations = []
    
    for sigma in sigma_vals:
        for t in t_vals:
            is_convex, E_ss = test_convexity_point(sigma, t)
            if E_ss < min_E_ss:
                min_E_ss = E_ss
                min_location = (sigma, t)
            if not is_convex:
                violations.append((sigma, t, E_ss))
    
    print(f"\nResults for left half-strip (σ < 0.5):")
    print(f"  Points tested: {sigma_points * t_points}")
    print(f"  Minimum E'': {min_E_ss:.6e} at (σ={min_location[0]:.3f}, t={min_location[1]:.1f})")
    print(f"  Violations found: {len(violations)}")
    
    if violations:
        print("\n  VIOLATIONS:")
        for v in violations[:10]:
            print(f"    σ={v[0]:.3f}, t={v[1]:.1f}: E'' = {v[2]:.6e}")
    
    return len(violations) == 0, min_E_ss

def test_specific_T0_threshold():
    """
    Find the smallest T₀ such that the asymptotic bound A(s) > |K| holds.
    
    This tests whether there's a specific T₀ below which we need
    rigorous finite verification.
    """
    print("\n" + "="*60)
    print("Testing T₀ threshold for asymptotic bound")
    print("="*60)
    
    # Test at σ = 0.3 (typical off-line point)
    sigma = 0.3
    
    for t in [10, 20, 50, 100, 200, 500, 1000]:
        is_convex, E_ss = test_convexity_point(sigma, t)
        status = "✓" if is_convex else "✗"
        print(f"  t = {t:4d}: E''(0.3, t) = {E_ss:+.6e} {status}")

def assess_interval_arithmetic_feasibility():
    """
    Assess what would be needed for rigorous interval arithmetic.
    """
    print("\n" + "="*60)
    print("INTERVAL ARITHMETIC ASSESSMENT")
    print("="*60)
    
    print("""
For a RIGOROUS proof via validated numerics, we would need:

1. INTERVAL ARITHMETIC LIBRARY (e.g., mpfi, arb, INTLAB)
   - Not just floating point with high precision
   - Actual interval bounds with guaranteed containment

2. COVERING THE FINITE WINDOW
   - Partition [0, T₀] × [ε, 0.5-ε] into rectangles
   - For each rectangle, prove E'' > 0 using interval bounds
   - This requires bounding the Hessian on each rectangle

3. TECHNICAL CHALLENGES
   - The zeta function evaluation needs interval-safe implementation
   - Need to handle oscillations in Im(t) direction
   - Near zeros, E is small but E'' might still be positive

4. ESTIMATED COMPUTATIONAL COST
   - For T₀ = 100, ε = 0.01
   - Grid size: ~10000 rectangles minimum
   - Each requires interval zeta evaluation (~slow)
   - Total: potentially hours to days of computation

5. EXISTING TOOLS
   - ARB library (C, by Fredrik Johansson) - can do interval zeta
   - MPFI (Python/C) - general interval arithmetic
   - Would need custom wrapper for E'' computation

CONCLUSION: Rigorous interval arithmetic verification is FEASIBLE
but would require significant implementation effort.
    """)

def main():
    print("="*70)
    print("INTERVAL ARITHMETIC VERIFICATION TEST")
    print("Testing whether E''(σ,t) > 0 can be rigorously verified")
    print("="*70)
    
    # Test 1: Point sampling on finite window
    success, min_val = test_finite_window(T0=100, sigma_points=15, t_points=30)
    
    # Test 2: Find T₀ threshold
    test_specific_T0_threshold()
    
    # Test 3: Assess feasibility
    assess_interval_arithmetic_feasibility()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Point Sampling Result: {'PASS' if success else 'FAIL'} (minimum E'' = {min_val:.6e})

IMPORTANT: Point sampling is NOT sufficient for a proof!

The critique is CORRECT that:
1. "Validated numerics" must mean interval arithmetic, not sampling
2. Current paper claims validated numerics but doesn't provide them
3. This is a real gap that would need to be closed

STATUS: Framework plausibly correct, but finite-window verification
        needs actual interval arithmetic implementation.
    """)

if __name__ == "__main__":
    main()
