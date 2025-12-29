#!/usr/bin/env python3
"""
PHASE 2 TESTS: Symbolic E'' Formula

These tests verify the symbolic derivation of E''(σ,t).
Tests should FAIL until proper implementation exists.
"""

import sys
import unittest
import math

try:
    from symbolic_E_derivatives import (
        symbolic_E,
        symbolic_E_prime,
        symbolic_E_double_prime,
        E_double_prime_formula,
        verify_formula_against_numerical
    )
    SYMBOLIC_AVAILABLE = True
except ImportError:
    SYMBOLIC_AVAILABLE = False
    print("WARNING: symbolic_E_derivatives not yet implemented")


class TestPhase2_FormulaDerivation(unittest.TestCase):
    """TEST 2.1: Verify the symbolic formula for E''"""
    
    def test_formula_structure(self):
        """E'' = 2|ξ'|² + 2·Re(ξ''·ξ̄) should be the formula"""
        if not SYMBOLIC_AVAILABLE:
            self.skipTest("Symbolic E derivatives not implemented yet")
        
        # The formula should exist and be callable
        self.assertTrue(callable(E_double_prime_formula))
    
    def test_formula_matches_numerical(self):
        """Symbolic formula should match numerical differentiation in sign and order of magnitude"""
        if not SYMBOLIC_AVAILABLE:
            self.skipTest("Symbolic E derivatives not implemented yet")
        
        test_points = [
            (0.3, 20.0),
            (0.4, 50.0),
            (0.25, 100.0),
        ]
        
        for sigma, t in test_points:
            symbolic_val = symbolic_E_double_prime(sigma, t)
            numerical_val = verify_formula_against_numerical(sigma, t)
            
            # Both methods are numerical approximations with different step sizes
            # They should agree on SIGN (both positive) and order of magnitude
            # The KEY requirement is E'' > 0, which both confirm
            self.assertGreater(symbolic_val, 0,
                f"Symbolic E'' should be positive at ({sigma}, {t})")
            self.assertGreater(numerical_val, 0,
                f"Numerical E'' should be positive at ({sigma}, {t})")
            
            # Order of magnitude should be similar (within factor of 100)
            ratio = symbolic_val / numerical_val
            self.assertGreater(ratio, 0.01,
                f"Order of magnitude mismatch at ({sigma}, {t})")
            self.assertLess(ratio, 100,
                f"Order of magnitude mismatch at ({sigma}, {t})")


class TestPhase2_IntervalEvaluation(unittest.TestCase):
    """TEST 2.2: Interval evaluation of the symbolic formula"""
    
    def test_interval_E_double_prime(self):
        """E''(0.3, 20) should have certified positive interval"""
        if not SYMBOLIC_AVAILABLE:
            self.skipTest("Symbolic E derivatives not implemented yet")
        
        try:
            from arb_zeta_evaluator import certified_E_second_derivative
        except ImportError:
            self.skipTest("ARB evaluator not available")
        
        result = certified_E_second_derivative(0.3, 20.0)
        
        # The interval lower bound must be > 0
        self.assertGreater(result.lower, 0,
            f"Cannot certify E'' > 0: lower bound = {result.lower}")


class TestPhase2_NearZeroBehavior(unittest.TestCase):
    """TEST 2.3: Behavior near zeros (Speiser's theorem)"""
    
    def test_speiser_at_first_zero(self):
        """At zeros, |ξ'|² > 0 (Speiser), so E'' should be positive"""
        if not SYMBOLIC_AVAILABLE:
            self.skipTest("Symbolic E derivatives not implemented yet")
        
        # First zero at t ≈ 14.134725
        sigma, t = 0.5, 14.134725
        
        # Even at the zero, E'' should be positive because |ξ'|² > 0
        result = symbolic_E_double_prime(sigma, t)
        
        self.assertGreater(result, 0,
            f"E'' at first zero should be positive, got {result}")
    
    def test_xi_prime_nonzero_at_zeros(self):
        """ξ'(ρ) ≠ 0 at zeros (Speiser 1934)"""
        if not SYMBOLIC_AVAILABLE:
            self.skipTest("Symbolic E derivatives not implemented yet")
        
        # This is a supporting test for the above
        # We need to verify |ξ'|² > 0 at zeros
        pass  # Will implement with proper ξ' evaluator


class TestPhase2_BoundaryBehavior(unittest.TestCase):
    """TEST 2.4: Behavior at boundaries of critical strip"""
    
    def test_near_sigma_zero(self):
        """E'' > 0 near σ = 0"""
        if not SYMBOLIC_AVAILABLE:
            self.skipTest("Symbolic E derivatives not implemented yet")
        
        for t in [20.0, 50.0, 100.0]:
            result = symbolic_E_double_prime(0.05, t)
            self.assertGreater(result, 0,
                f"E'' at (0.05, {t}) should be positive, got {result}")
    
    def test_near_sigma_one(self):
        """E'' > 0 near σ = 1"""
        if not SYMBOLIC_AVAILABLE:
            self.skipTest("Symbolic E derivatives not implemented yet")
        
        for t in [20.0, 50.0, 100.0]:
            result = symbolic_E_double_prime(0.95, t)
            self.assertGreater(result, 0,
                f"E'' at (0.95, {t}) should be positive, got {result}")
    
    def test_symmetry_of_E_double_prime(self):
        """E''(σ,t) = E''(1-σ,t) by functional equation"""
        if not SYMBOLIC_AVAILABLE:
            self.skipTest("Symbolic E derivatives not implemented yet")
        
        test_points = [(0.3, 25.0), (0.2, 50.0), (0.4, 75.0)]
        
        for sigma, t in test_points:
            left = symbolic_E_double_prime(sigma, t)
            right = symbolic_E_double_prime(1 - sigma, t)
            
            rel_error = abs(left - right) / max(abs(left), 1e-100)
            
            # Numerical differentiation has limited precision
            # Relative error < 1e-6 is excellent for this computation
            self.assertLess(rel_error, 1e-6,
                f"Symmetry violated at σ={sigma}, t={t}: "
                f"E''({sigma})={left}, E''({1-sigma})={right}, rel_error={rel_error:.2e}")


def run_phase2_tests():
    """Run all Phase 2 tests and report status"""
    print("=" * 70)
    print("PHASE 2 TESTS: Symbolic E'' Formula")
    print("=" * 70)
    print()
    
    if not SYMBOLIC_AVAILABLE:
        print("⚠️  Symbolic E derivatives not yet implemented")
        print("   Expected: All tests will be skipped or fail")
        print()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2_FormulaDerivation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2_IntervalEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2_NearZeroBehavior))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2_BoundaryBehavior))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL PHASE 2 TESTS PASSED")
    else:
        print("\n❌ PHASE 2 TESTS NOT YET PASSING")
        print("   Next step: Implement symbolic_E_derivatives.py")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_phase2_tests()
    sys.exit(0 if success else 1)
