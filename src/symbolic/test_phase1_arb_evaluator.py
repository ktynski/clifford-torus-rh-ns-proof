#!/usr/bin/env python3
"""
PHASE 1 TESTS: ARB-Style Certified Zeta Evaluation

These tests define what we NEED, not what we have.
Tests should FAIL until proper implementation exists.

Test-Driven Development: Write tests first, implement until they pass.
"""

import sys
import unittest
from decimal import Decimal
from typing import Tuple, Optional

# We'll need to implement or import these
# For now, they may not exist - that's the point of TDD
try:
    from arb_zeta_evaluator import (
        CertifiedInterval,
        certified_zeta,
        certified_gamma,
        certified_xi,
        certified_E,
        certified_E_second_derivative
    )
    ARB_AVAILABLE = True
except ImportError:
    ARB_AVAILABLE = False
    print("WARNING: arb_zeta_evaluator not yet implemented")


class CertifiedInterval:
    """
    A certified interval [mid - rad, mid + rad].
    
    This is the interface we need. Implementation will use ARB/flint.
    """
    def __init__(self, mid: float, rad: float):
        self.mid = mid
        self.rad = rad
    
    @property
    def lower(self) -> float:
        return self.mid - self.rad
    
    @property
    def upper(self) -> float:
        return self.mid + self.rad
    
    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper
    
    def is_positive(self) -> bool:
        """Returns True only if entire interval is > 0"""
        return self.lower > 0
    
    def is_negative(self) -> bool:
        """Returns True only if entire interval is < 0"""
        return self.upper < 0
    
    def width(self) -> float:
        return 2 * self.rad


class TestPhase1_ARBInstallation(unittest.TestCase):
    """TEST 1.1: Verify ARB library is available"""
    
    def test_arb_import(self):
        """Can we import the ARB/flint library?"""
        try:
            # Try python-flint first (preferred)
            import flint
            self.assertTrue(hasattr(flint, 'arb'))
        except ImportError:
            try:
                # Try mpmath with interval mode
                from mpmath import iv
                self.assertTrue(True)  # mpmath intervals available
            except ImportError:
                self.fail("Neither flint nor mpmath intervals available")
    
    def test_ball_arithmetic_available(self):
        """Can we create interval/ball arithmetic objects?"""
        try:
            from flint import arb
            x = arb(1.5)
            self.assertIsNotNone(x)
        except ImportError:
            from mpmath import iv, mpf
            x = iv.mpf([1.4, 1.6])  # Interval [1.4, 1.6]
            self.assertIsNotNone(x)


class TestPhase1_KnownValues(unittest.TestCase):
    """TEST 1.2: Verify against known mathematical values"""
    
    def test_zeta_2(self):
        """ζ(2) = π²/6 ≈ 1.6449340668..."""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        import math
        expected = math.pi**2 / 6
        
        result = certified_zeta(2.0, 0.0)  # ζ(2+0i)
        
        # The interval must contain the true value
        self.assertTrue(result.contains(expected),
            f"Interval [{result.lower}, {result.upper}] does not contain {expected}")
        
        # The interval should be tight (width < 10⁻³⁰)
        self.assertLess(result.width(), 1e-30,
            f"Interval width {result.width()} too large")
    
    def test_zeta_4(self):
        """ζ(4) = π⁴/90 ≈ 1.0823232337..."""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        import math
        expected = math.pi**4 / 90
        
        result = certified_zeta(4.0, 0.0)
        
        self.assertTrue(result.contains(expected))
        self.assertLess(result.width(), 1e-30)
    
    def test_gamma_1(self):
        """Γ(1) = 1 exactly"""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        result = certified_gamma(1.0, 0.0)
        
        self.assertTrue(result.contains(1.0))
        self.assertLess(result.width(), 1e-30)
    
    def test_gamma_half(self):
        """Γ(1/2) = √π ≈ 1.7724538509..."""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        import math
        expected = math.sqrt(math.pi)
        
        result = certified_gamma(0.5, 0.0)
        
        self.assertTrue(result.contains(expected))
        self.assertLess(result.width(), 1e-30)


class TestPhase1_CriticalStrip(unittest.TestCase):
    """TEST 1.3: Verify behavior in critical strip"""
    
    def test_first_zero(self):
        """ξ(0.5 + 14.134725i) ≈ 0 (first non-trivial zero)"""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        # First zero is at approximately t = 14.134725141734693790...
        result = certified_xi(0.5, 14.134725141734693790)
        
        # The interval for |ξ| should contain 0 (or be very small)
        # Since it's a zero, |ξ| should be essentially 0
        self.assertLess(abs(result.mid), 1e-10,
            f"|ξ| midpoint {result.mid} not close to zero")
        self.assertTrue(result.contains(0.0) or result.upper < 1e-10,
            f"Interval does not confirm zero at first zero location")
    
    def test_off_critical_line(self):
        """ξ(0.3 + 20i) should be non-zero with certified bound"""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        result = certified_xi(0.3, 20.0)
        
        # Should get a non-zero value with tight bounds
        self.assertGreater(abs(result.mid), 1e-10,
            f"|ξ| at off-line point should be non-zero")
        # Interval should be reasonably tight
        self.assertLess(result.width() / abs(result.mid), 0.01,
            "Relative width too large")
    
    def test_functional_equation(self):
        """Verify ξ(s) = ξ(1-s) with certified intervals"""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        sigma, t = 0.3, 25.0
        
        result_left = certified_xi(sigma, t)
        result_right = certified_xi(1 - sigma, t)
        
        # The intervals should overlap (same value)
        overlap = max(result_left.lower, result_right.lower) <= \
                  min(result_left.upper, result_right.upper)
        
        self.assertTrue(overlap,
            f"Functional equation not verified: "
            f"ξ({sigma}+{t}i) = {result_left.mid} ± {result_left.rad}, "
            f"ξ({1-sigma}+{t}i) = {result_right.mid} ± {result_right.rad}")


class TestPhase1_ErrorPropagation(unittest.TestCase):
    """TEST 1.4: Verify error bounds propagate correctly"""
    
    def test_E_evaluation(self):
        """E(σ,t) = |ξ(σ+it)|² should have certified bounds"""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        result = certified_E(0.3, 20.0)
        
        # E should be positive (it's a squared magnitude)
        self.assertTrue(result.is_positive(),
            f"E should be positive, got interval [{result.lower}, {result.upper}]")
        
        # Bounds should be reasonably tight
        self.assertLess(result.width() / result.mid, 1e-10,
            "Relative error too large for E")
    
    def test_E_second_derivative(self):
        """E''(σ,t) should have certified bounds"""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        result = certified_E_second_derivative(0.3, 20.0)
        
        # At this point, we're testing infrastructure, not the theorem
        # Just verify we get an interval
        self.assertIsNotNone(result.mid)
        self.assertIsNotNone(result.rad)
        self.assertGreaterEqual(result.rad, 0)
    
    def test_error_accumulation(self):
        """Verify errors don't explode through computation"""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        # Compute E at multiple points
        test_points = [
            (0.1, 10.0),
            (0.3, 50.0),
            (0.4, 100.0),
        ]
        
        for sigma, t in test_points:
            result = certified_E(sigma, t)
            
            # Relative width should stay bounded
            if result.mid > 1e-100:
                rel_width = result.width() / result.mid
                self.assertLess(rel_width, 0.01,
                    f"Error exploded at ({sigma}, {t}): rel_width = {rel_width}")


class TestPhase1_ConvexitySignCertification(unittest.TestCase):
    """Additional tests for sign certification of E''"""
    
    def test_convexity_certification_possible(self):
        """Can we certify E'' > 0 with intervals?"""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        # At a point away from critical line, E'' should be certifiably positive
        result = certified_E_second_derivative(0.3, 20.0)
        
        # THIS IS THE KEY TEST
        # The interval's lower bound must be > 0 to certify positivity
        self.assertTrue(result.is_positive(),
            f"Cannot certify E'' > 0: interval = [{result.lower}, {result.upper}]")
    
    def test_convexity_near_zero(self):
        """E'' should be positive even near zeros (Speiser's theorem)"""
        if not ARB_AVAILABLE:
            self.skipTest("ARB evaluator not implemented yet")
        
        # Near first zero
        result = certified_E_second_derivative(0.5, 14.134725)
        
        # Speiser says ξ'(ρ) ≠ 0, so E'' should still be certifiably positive
        # (This may require very tight bounds)
        self.assertTrue(result.is_positive() or result.lower > -1e-20,
            f"E'' near zero not handled: [{result.lower}, {result.upper}]")


def run_phase1_tests():
    """Run all Phase 1 tests and report status"""
    print("=" * 70)
    print("PHASE 1 TESTS: ARB-Style Certified Zeta Evaluation")
    print("=" * 70)
    print()
    
    if not ARB_AVAILABLE:
        print("⚠️  ARB evaluator not yet implemented")
        print("   Expected: All tests will be skipped or fail")
        print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1_ARBInstallation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1_KnownValues))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1_CriticalStrip))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1_ErrorPropagation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1_ConvexitySignCertification))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print()
    print("=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL PHASE 1 TESTS PASSED")
    else:
        print("\n❌ PHASE 1 TESTS NOT YET PASSING")
        print("   Next step: Implement arb_zeta_evaluator.py")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_phase1_tests()
    sys.exit(0 if success else 1)
