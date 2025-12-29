#!/usr/bin/env python3
"""
Test: Does convexity hold at large t where Voronin universality applies?

Voronin's theorem says ζ(s) can locally approximate any non-vanishing function
in the strip 1/2 < σ < 1. Does this create local concavity in |ξ|²?
"""

from mpmath import mp, mpf, zeta, gamma, pi, exp, log, sqrt, cos, sin, fabs
import numpy as np

# High precision for large t
mp.dps = 50

def xi(s):
    """Completed zeta function: ξ(s) = (1/2)s(s-1)π^{-s/2}Γ(s/2)ζ(s)"""
    if s == 1:
        return mp.mpf('inf')
    try:
        return mpf('0.5') * s * (s - 1) * (pi ** (-s/2)) * gamma(s/2) * zeta(s)
    except:
        return mp.mpf('nan')

def E(sigma, t):
    """Energy functional E(σ,t) = |ξ(σ+it)|²"""
    s = mp.mpc(sigma, t)
    xi_val = xi(s)
    return fabs(xi_val) ** 2

def d2E_dsigma2(sigma, t, h=1e-6):
    """Second derivative ∂²E/∂σ² via central differences."""
    E_plus = E(sigma + h, t)
    E_minus = E(sigma - h, t)
    E_center = E(sigma, t)
    return (E_plus - 2*E_center + E_minus) / (h**2)

def test_convexity_at_t(t_val, sigma_range=(0.1, 0.9), n_points=20):
    """Test convexity at a specific t value."""
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_points)
    
    results = []
    min_d2E = float('inf')
    min_sigma = None
    
    for sigma in sigmas:
        try:
            d2E = float(d2E_dsigma2(sigma, t_val))
            results.append((sigma, d2E))
            if d2E < min_d2E:
                min_d2E = d2E
                min_sigma = sigma
        except Exception as e:
            results.append((sigma, None))
    
    return results, min_d2E, min_sigma


def test_voronin_regime():
    """
    Test convexity in the Voronin universality regime (large t).
    
    If Voronin allows local concavity, we should find ∂²E/∂σ² < 0 somewhere.
    """
    print("=" * 70)
    print("TEST: Convexity in Voronin Universality Regime")
    print("=" * 70)
    print("\nVoronin's theorem applies for large t in the strip 1/2 < σ < 1.")
    print("Testing whether local concavity (∂²E/∂σ² < 0) appears...\n")
    
    # Test at progressively larger t
    test_heights = [100, 500, 1000, 5000, 10000, 50000]
    
    all_positive = True
    
    for t in test_heights:
        print(f"\nTesting at t = {t}...")
        mp.dps = max(50, int(t / 100))  # More precision for larger t
        
        results, min_d2E, min_sigma = test_convexity_at_t(t, sigma_range=(0.51, 0.99), n_points=30)
        
        if min_d2E < 0:
            print(f"   ⚠ FOUND NEGATIVE: min(∂²E/∂σ²) = {min_d2E:.4e} at σ = {min_sigma:.3f}")
            all_positive = False
        else:
            print(f"   ✓ All positive: min(∂²E/∂σ²) = {min_d2E:.4e} at σ = {min_sigma:.3f}")
    
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if all_positive:
        print("\n   ✓ Convexity HOLDS at all tested t values (up to 50,000)")
        print("     → Voronin universality does NOT create local concavity in |ξ|²")
        print("     → The Hadamard pairing structure appears to dominate")
    else:
        print("\n   ✗ Convexity VIOLATED at some t values")
        print("     → Voronin universality MAY create local concavity")
        print("     → The RH proof has a gap")
    
    return all_positive


def test_near_zeros():
    """
    Test convexity near actual zeros where things are most constrained.
    """
    print("\n" + "=" * 70)
    print("TEST: Convexity Near Known Zeros")
    print("=" * 70)
    
    # First few non-trivial zeros (imaginary parts)
    zero_t_values = [
        14.134725,
        21.022040,
        25.010858,
        30.424876,
        32.935062,
        37.586178,
        40.918720,
        43.327073,
        48.005151,
        49.773832
    ]
    
    mp.dps = 100
    
    print("\nTesting ∂²E/∂σ² at zeros (should be ~2|ξ'(ρ)|² > 0):\n")
    
    for t0 in zero_t_values[:5]:
        # Test at the zero
        sigma = 0.5
        try:
            d2E = float(d2E_dsigma2(sigma, t0))
            print(f"   t = {t0:.4f}: ∂²E/∂σ² = {d2E:.4e}", end="")
            if d2E > 0:
                print(" ✓")
            else:
                print(" ✗ NEGATIVE!")
        except Exception as e:
            print(f"   t = {t0:.4f}: Error - {e}")


def test_off_line_strip():
    """
    Test specifically in the Voronin strip (1/2 < σ < 1).
    """
    print("\n" + "=" * 70)
    print("TEST: Voronin Strip (1/2 < σ < 1) at Various Heights")
    print("=" * 70)
    
    mp.dps = 50
    
    heights = [100, 1000, 10000]
    sigma_in_strip = [0.55, 0.6, 0.7, 0.8, 0.9]
    
    for t in heights:
        print(f"\n   t = {t}:")
        for sigma in sigma_in_strip:
            try:
                d2E = float(d2E_dsigma2(sigma, t))
                status = "✓" if d2E > 0 else "✗"
                print(f"      σ = {sigma}: ∂²E/∂σ² = {d2E:+.4e} {status}")
            except Exception as e:
                print(f"      σ = {sigma}: Error")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VORONIN COMPATIBILITY INVESTIGATION")
    print("=" * 70)
    print("\nQuestion: Does Voronin universality contradict global convexity?\n")
    
    # Test 1: Near zeros
    test_near_zeros()
    
    # Test 2: Off-line strip
    test_off_line_strip()
    
    # Test 3: Large t regime
    result = test_voronin_regime()
    
    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)
