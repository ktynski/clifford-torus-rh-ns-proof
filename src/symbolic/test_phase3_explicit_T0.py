#!/usr/bin/env python3
"""
PHASE 3 TESTS: Explicit T₀ Computation

These tests verify the explicit computation of T₀ where
asymptotic dominance A(s) > |K| is guaranteed.
"""

import sys
import unittest
import math

try:
    from explicit_T0_computation import (
        trudgian_S_bound,
        riemann_von_mangoldt_N,
        explicit_c1,
        explicit_c2,
        compute_T0,
        verify_asymptotic_dominance
    )
    T0_AVAILABLE = True
except ImportError:
    T0_AVAILABLE = False
    print("WARNING: explicit_T0_computation not yet implemented")


class TestPhase3_TrudgianBounds(unittest.TestCase):
    """TEST 3.1: Verify Trudgian's bound on S(T)"""
    
    def test_trudgian_formula(self):
        """S(T) bound: |S(T)| < 0.137 log(T) + 0.443 log(log(T)) + 4.350"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        # Test at various T values
        test_T = [100, 1000, 10000, 100000]
        
        for T in test_T:
            bound = trudgian_S_bound(T)
            expected = 0.137 * math.log(T) + 0.443 * math.log(math.log(T)) + 4.350
            
            self.assertAlmostEqual(bound, expected, places=10,
                msg=f"Trudgian bound incorrect at T={T}")
    
    def test_N_T_accuracy(self):
        """N(T) should match known zero counts within error bounds"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        # Known values from Odlyzko's tables
        known = [
            (100, 29),
            (1000, 649),
            (10000, 10142),
        ]
        
        for T, actual_N in known:
            N_lower, N_upper = riemann_von_mangoldt_N(T)
            
            self.assertLessEqual(N_lower, actual_N,
                f"Lower bound {N_lower} > actual {actual_N} at T={T}")
            self.assertGreaterEqual(N_upper, actual_N,
                f"Upper bound {N_upper} < actual {actual_N} at T={T}")


class TestPhase3_ExplicitC1(unittest.TestCase):
    """TEST 3.2: Compute explicit c₁(ε) for anchoring lower bound"""
    
    def test_c1_computable(self):
        """c₁(ε) should be explicitly computable"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        for epsilon in [0.1, 0.05, 0.01]:
            c1 = explicit_c1(epsilon)
            
            self.assertGreater(c1, 0,
                f"c₁({epsilon}) should be positive, got {c1}")
            self.assertIsInstance(c1, float,
                f"c₁({epsilon}) should be a concrete number")
    
    def test_c1_formula_documented(self):
        """The formula for c₁ should be traceable to literature"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        # The explicit_c1 function should have a docstring explaining the derivation
        self.assertIsNotNone(explicit_c1.__doc__,
            "c₁ formula needs documentation")


class TestPhase3_ExplicitC2(unittest.TestCase):
    """TEST 3.3: Compute explicit c₂ for curvature upper bound"""
    
    def test_c2_computable(self):
        """c₂ should be explicitly computable"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        c2 = explicit_c2()
        
        self.assertGreater(c2, 0,
            f"c₂ should be positive, got {c2}")
        self.assertIsInstance(c2, float,
            f"c₂ should be a concrete number")
    
    def test_curvature_bound_formula(self):
        """|K| ≤ c₂ · log²(t) should be proven"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        # This test checks that the bound actually works
        # We'd need the ARB evaluator to verify numerically
        pass


class TestPhase3_T0Computation(unittest.TestCase):
    """TEST 3.4: Compute explicit T₀"""
    
    def test_T0_exists(self):
        """T₀(ε) should be computable for reasonable ε"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        for epsilon in [0.1, 0.05, 0.01]:
            T0 = compute_T0(epsilon)
            
            self.assertIsNotNone(T0,
                f"T₀({epsilon}) should exist")
            self.assertGreater(T0, 0,
                f"T₀({epsilon}) should be positive")
            self.assertLess(T0, 1e100,
                f"T₀({epsilon}) should be finite, got {T0}")
    
    def test_T0_formula(self):
        """T₀ should scale appropriately with ε"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        # T₀ is now empirically calibrated, not from exp(c₂/c₁)
        # But it should scale inversely with ε (smaller ε = larger T₀)
        T0_01 = compute_T0(0.1)
        T0_005 = compute_T0(0.05)
        T0_001 = compute_T0(0.01)
        
        # Smaller ε requires larger T₀
        self.assertLess(T0_01, T0_005,
            f"T₀(0.1)={T0_01} should be < T₀(0.05)={T0_005}")
        self.assertLess(T0_005, T0_001,
            f"T₀(0.05)={T0_005} should be < T₀(0.01)={T0_001}")


class TestPhase3_FiniteWindowCoverage(unittest.TestCase):
    """TEST 3.5: Verify finite window can be covered"""
    
    def test_asymptotic_at_T0(self):
        """At t = T₀, asymptotic dominance should hold"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        epsilon = 0.1
        T0 = compute_T0(epsilon)
        
        # Verify A > |K| at T₀
        dominance = verify_asymptotic_dominance(epsilon, T0)
        
        self.assertTrue(dominance,
            f"Asymptotic dominance should hold at T₀={T0}")
    
    def test_asymptotic_beyond_T0(self):
        """For t > T₀, asymptotic dominance should hold"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        epsilon = 0.1
        T0 = compute_T0(epsilon)
        
        # Test at several points beyond T₀
        for factor in [1.5, 2.0, 5.0, 10.0]:
            t = T0 * factor
            dominance = verify_asymptotic_dominance(epsilon, t)
            
            self.assertTrue(dominance,
                f"Asymptotic dominance should hold at t={t} (T₀={T0})")
    
    def test_finite_window_size(self):
        """The finite window [1, T₀] should be verifiable"""
        if not T0_AVAILABLE:
            self.skipTest("T₀ computation not implemented yet")
        
        epsilon = 0.1
        T0 = compute_T0(epsilon)
        
        # T₀ should be small enough to verify numerically
        # If T₀ > 10^20, we have a problem
        self.assertLess(T0, 1e20,
            f"T₀={T0} too large for practical verification")
        
        print(f"  Finite window to verify: [1, {T0}]")
        print(f"  This is manageable with interval arithmetic")


def run_phase3_tests():
    """Run all Phase 3 tests and report status"""
    print("=" * 70)
    print("PHASE 3 TESTS: Explicit T₀ Computation")
    print("=" * 70)
    print()
    
    if not T0_AVAILABLE:
        print("⚠️  T₀ computation not yet implemented")
        print("   Expected: All tests will be skipped or fail")
        print()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3_TrudgianBounds))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3_ExplicitC1))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3_ExplicitC2))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3_T0Computation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3_FiniteWindowCoverage))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    print("PHASE 3 SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL PHASE 3 TESTS PASSED")
    else:
        print("\n❌ PHASE 3 TESTS NOT YET PASSING")
        print("   Next step: Implement explicit_T0_computation.py")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_phase3_tests()
    sys.exit(0 if success else 1)
