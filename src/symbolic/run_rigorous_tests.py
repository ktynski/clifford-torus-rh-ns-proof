#!/usr/bin/env python3
"""
RIGOROUS PROOF TEST RUNNER

Run all phase tests in sequence and report overall status.
This is the entry point for the test-driven development process.

Usage:
    python3 run_rigorous_tests.py           # Run all phases
    python3 run_rigorous_tests.py --phase 1 # Run only phase 1
"""

import sys
import argparse

def run_phase_tests(test_module, phase_name):
    """Run tests and return (passed, skipped_count, total_count)"""
    import unittest
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_module)
    
    # Run silently first to get counts
    from io import StringIO
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=0)
    result = runner.run(suite)
    
    total = result.testsRun
    skipped = len(result.skipped)
    failed = len(result.failures) + len(result.errors)
    passed = total - skipped - failed
    
    return passed, skipped, failed, total

def run_phase1():
    """Phase 1: ARB-style certified zeta evaluation"""
    print("\n" + "=" * 70)
    print("PHASE 1: ARB-Style Certified Zeta Evaluation")
    print("=" * 70)
    try:
        import test_phase1_arb_evaluator as test_module
        passed, skipped, failed, total = run_phase_tests(test_module, "Phase 1")
        
        print(f"  Tests: {total} total, {passed} passed, {skipped} skipped, {failed} failed")
        
        if skipped > 0:
            print(f"  ⚠️  {skipped} tests skipped - implementation needed")
            return False  # Skipped tests mean incomplete
        elif failed > 0:
            print(f"  ❌ {failed} tests failed")
            return False
        else:
            print(f"  ✅ All {passed} tests passed!")
            return True
    except Exception as e:
        print(f"Error running Phase 1: {e}")
        return False

def run_phase2():
    """Phase 2: Symbolic E'' derivation"""
    print("\n" + "=" * 70)
    print("PHASE 2: Symbolic E'' Formula")
    print("=" * 70)
    try:
        import test_phase2_symbolic_E as test_module
        passed, skipped, failed, total = run_phase_tests(test_module, "Phase 2")
        
        print(f"  Tests: {total} total, {passed} passed, {skipped} skipped, {failed} failed")
        
        if skipped > 0:
            print(f"  ⚠️  {skipped} tests skipped - implementation needed")
            return False
        elif failed > 0:
            print(f"  ❌ {failed} tests failed")
            return False
        else:
            print(f"  ✅ All {passed} tests passed!")
            return True
    except Exception as e:
        print(f"Error running Phase 2: {e}")
        return False

def run_phase3():
    """Phase 3: Explicit T₀ computation"""
    print("\n" + "=" * 70)
    print("PHASE 3: Explicit T₀ Computation")
    print("=" * 70)
    try:
        import test_phase3_explicit_T0 as test_module
        passed, skipped, failed, total = run_phase_tests(test_module, "Phase 3")
        
        print(f"  Tests: {total} total, {passed} passed, {skipped} skipped, {failed} failed")
        
        if skipped > 0:
            print(f"  ⚠️  {skipped} tests skipped - implementation needed")
            return False
        elif failed > 0:
            print(f"  ❌ {failed} tests failed")
            return False
        else:
            print(f"  ✅ All {passed} tests passed!")
            return True
    except Exception as e:
        print(f"Error running Phase 3: {e}")
        return False

def run_phase4():
    """Phase 4: Circularity audit"""
    print("\n" + "=" * 70)
    print("PHASE 4: Circularity Audit")
    print("=" * 70)
    try:
        import test_phase4_circularity as test_module
        passed, skipped, failed, total = run_phase_tests(test_module, "Phase 4")
        
        print(f"  Tests: {total} total, {passed} passed, {skipped} skipped, {failed} failed")
        
        if skipped > 0:
            print(f"  ⚠️  {skipped} tests skipped - implementation needed")
            return False
        elif failed > 0:
            print(f"  ❌ {failed} tests failed")
            return False
        else:
            print(f"  ✅ All {passed} tests passed!")
            return True
    except Exception as e:
        print(f"Error running Phase 4: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run rigorous proof tests")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                        help="Run only the specified phase")
    args = parser.parse_args()
    
    print("=" * 70)
    print("RIGOROUS COMPUTER-ASSISTED PROOF: TEST SUITE")
    print("=" * 70)
    print()
    print("This test suite verifies the requirements for a rigorous proof.")
    print("Tests are written FIRST - implementations come later (TDD).")
    print()
    print("Status Legend:")
    print("  ✅ PASS  - Implementation complete and verified")
    print("  ❌ FAIL  - Implementation missing or incorrect")
    print("  ⏭️  SKIP  - Dependency not available")
    
    results = {}
    
    if args.phase is None or args.phase == 1:
        results['Phase 1 (ARB Evaluator)'] = run_phase1()
    
    if args.phase is None or args.phase == 2:
        results['Phase 2 (Symbolic E\'\')'] = run_phase2()
    
    if args.phase is None or args.phase == 3:
        results['Phase 3 (Explicit T₀)'] = run_phase3()
    
    if args.phase is None or args.phase == 4:
        results['Phase 4 (Circularity)'] = run_phase4()
    
    # Final summary
    print("\n" + "=" * 70)
    print("OVERALL STATUS")
    print("=" * 70)
    
    all_passed = True
    for phase, passed in results.items():
        status = "✅ PASS" if passed else "❌ NOT READY"
        print(f"  {phase}: {status}")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("=" * 70)
        print("🎉 ALL PHASES COMPLETE - PROOF IS RIGOROUS")
        print("=" * 70)
    else:
        print("=" * 70)
        print("⚠️  WORK REMAINING - See failing tests above")
        print("=" * 70)
        print()
        print("Next steps:")
        for phase, passed in results.items():
            if not passed:
                print(f"  • Implement {phase}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
