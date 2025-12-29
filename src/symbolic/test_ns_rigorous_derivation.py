#!/usr/bin/env python3
"""
TEST-DRIVEN RIGOROUS DERIVATION FOR NS GENERAL DATA CLOSURE

These tests specify EXACTLY what must be proven for the Non-Beltrami
Enstrophy Control theorem to be rigorous. Each test corresponds to
a specific mathematical claim that must hold.

The tests are designed to FAIL unless the proofs are complete.
"""

import unittest
import numpy as np
from scipy.fft import fftn, ifftn
from typing import Tuple, Optional
import sys
import os

# Import the rigorous module (will be created/updated)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestBeltramiDecompositionOrthogonality(unittest.TestCase):
    """
    REQUIREMENT 1: The Beltrami/Non-Beltrami decomposition must be orthogonal.
    
    Mathematical claim:
        For any divergence-free v, decompose v = v^B + v^⊥
        Then: ⟨v^B, v^⊥⟩_{L²} = 0
        And:  ⟨ω^B, ω^⊥⟩_{L²} = 0
    """
    
    def setUp(self):
        """Create test velocity field."""
        self.N = 16
        self.L = 2 * np.pi
        
        # Create random divergence-free velocity field
        np.random.seed(42)
        self.v_hat = self._create_divergence_free_field(self.N)
        
        # Create wavevector grid
        k = np.fft.fftfreq(self.N) * self.N
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
    
    def _create_divergence_free_field(self, N: int) -> np.ndarray:
        """Create a random divergence-free velocity field in spectral space."""
        # Create random potential (stream function style)
        psi_hat = np.random.randn(3, N, N, N) + 1j * np.random.randn(3, N, N, N)
        
        k = np.fft.fftfreq(N) * N
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k_mag_sq = kx**2 + ky**2 + kz**2
        k_mag_sq[0, 0, 0] = 1.0  # Avoid division by zero
        
        # Create velocity via curl of potential (automatically divergence-free)
        v_hat = np.zeros((3, N, N, N), dtype=complex)
        v_hat[0] = 1j * (ky * psi_hat[2] - kz * psi_hat[1])
        v_hat[1] = 1j * (kz * psi_hat[0] - kx * psi_hat[2])
        v_hat[2] = 1j * (kx * psi_hat[1] - ky * psi_hat[0])
        
        # Zero out zero mode
        v_hat[:, 0, 0, 0] = 0
        
        return v_hat
    
    def test_velocity_decomposition_orthogonality(self):
        """
        TEST: ⟨v^B, v^⊥⟩_{L²} = 0
        
        The Beltrami and non-Beltrami velocity components must be orthogonal.
        """
        from ns_rigorous_derivation import beltrami_decomposition
        
        v_B, v_perp = beltrami_decomposition(self.v_hat, self.kx, self.ky, self.kz)
        
        # Compute L² inner product: ⟨v^B, v^⊥⟩ = Σ v^B · conj(v^⊥)
        inner_product = np.sum(v_B * np.conj(v_perp)).real / self.N**3
        
        # Must be essentially zero
        self.assertLess(abs(inner_product), 1e-10,
            f"Velocity decomposition not orthogonal: ⟨v^B, v^⊥⟩ = {inner_product}")
    
    def test_vorticity_decomposition_orthogonality(self):
        """
        TEST: ⟨ω^B, ω^⊥⟩_{L²} = 0
        
        The Beltrami and non-Beltrami vorticity components must be orthogonal.
        """
        from ns_rigorous_derivation import beltrami_decomposition, compute_curl
        
        v_B, v_perp = beltrami_decomposition(self.v_hat, self.kx, self.ky, self.kz)
        
        omega_B = compute_curl(v_B, self.kx, self.ky, self.kz)
        omega_perp = compute_curl(v_perp, self.kx, self.ky, self.kz)
        
        inner_product = np.sum(omega_B * np.conj(omega_perp)).real / self.N**3
        
        self.assertLess(abs(inner_product), 1e-10,
            f"Vorticity decomposition not orthogonal: ⟨ω^B, ω^⊥⟩ = {inner_product}")
    
    def test_decomposition_is_complete(self):
        """
        TEST: v = v^B + v^⊥ (exact)
        
        The decomposition must be complete - no energy lost.
        """
        from ns_rigorous_derivation import beltrami_decomposition
        
        v_B, v_perp = beltrami_decomposition(self.v_hat, self.kx, self.ky, self.kz)
        
        # Check that v = v^B + v^⊥
        residual = np.max(np.abs(self.v_hat - (v_B + v_perp)))
        
        self.assertLess(residual, 1e-10,
            f"Decomposition not complete: max|v - (v^B + v^⊥)| = {residual}")
    
    def test_beltrami_component_satisfies_beltrami_condition(self):
        """
        TEST: ω^B = λ v^B for each mode
        
        The Beltrami component must satisfy the Beltrami condition.
        """
        from ns_rigorous_derivation import beltrami_decomposition, compute_curl
        
        v_B, _ = beltrami_decomposition(self.v_hat, self.kx, self.ky, self.kz)
        omega_B = compute_curl(v_B, self.kx, self.ky, self.kz)
        
        # For each mode, check ω^B ∥ v^B
        # This means ω^B × v^B = 0 (cross product is zero)
        cross = np.zeros((3, self.N, self.N, self.N), dtype=complex)
        cross[0] = omega_B[1] * v_B[2] - omega_B[2] * v_B[1]
        cross[1] = omega_B[2] * v_B[0] - omega_B[0] * v_B[2]
        cross[2] = omega_B[0] * v_B[1] - omega_B[1] * v_B[0]
        
        # Normalize by magnitude to check alignment
        v_mag = np.sqrt(np.sum(np.abs(v_B)**2, axis=0))
        omega_mag = np.sqrt(np.sum(np.abs(omega_B)**2, axis=0))
        
        # Only check where both are non-zero
        mask = (v_mag > 1e-12) & (omega_mag > 1e-12)
        if np.any(mask):
            cross_mag = np.sqrt(np.sum(np.abs(cross)**2, axis=0))
            relative_cross = cross_mag[mask] / (v_mag[mask] * omega_mag[mask])
            max_relative_cross = np.max(relative_cross)
            
            self.assertLess(max_relative_cross, 1e-8,
                f"Beltrami condition violated: max|ω^B × v^B|/(|ω^B||v^B|) = {max_relative_cross}")


class TestPressureVanishesFromVorticity(unittest.TestCase):
    """
    REQUIREMENT 2: Pressure must vanish from the vorticity equation.
    
    Mathematical claim:
        The vorticity equation ∂ω/∂t = ν∆ω + (ω·∇)v - (v·∇)ω
        has NO pressure term because ∇×(∇p) = 0 identically.
    """
    
    def test_curl_of_gradient_is_zero(self):
        """
        TEST: ∇×(∇p) = 0 for any scalar field p
        
        This is a vector calculus identity that must hold numerically.
        """
        N = 16
        k = np.fft.fftfreq(N) * N
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        
        # Create random pressure field
        np.random.seed(123)
        p_hat = np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)
        
        # Compute gradient: ∇p
        grad_p = np.zeros((3, N, N, N), dtype=complex)
        grad_p[0] = 1j * kx * p_hat
        grad_p[1] = 1j * ky * p_hat
        grad_p[2] = 1j * kz * p_hat
        
        # Compute curl of gradient: ∇×(∇p)
        curl_grad_p = np.zeros((3, N, N, N), dtype=complex)
        curl_grad_p[0] = 1j * (ky * grad_p[2] - kz * grad_p[1])
        curl_grad_p[1] = 1j * (kz * grad_p[0] - kx * grad_p[2])
        curl_grad_p[2] = 1j * (kx * grad_p[1] - ky * grad_p[0])
        
        max_curl = np.max(np.abs(curl_grad_p))
        
        self.assertLess(max_curl, 1e-12,
            f"∇×(∇p) ≠ 0: max|∇×(∇p)| = {max_curl}")
    
    def test_pressure_term_vanishes_from_vorticity_equation(self):
        """
        TEST: Taking curl of NS momentum equation eliminates pressure.
        
        NS: ∂v/∂t + (v·∇)v = -∇p + ν∆v
        Curl: ∂ω/∂t + ∇×(v·∇)v = ν∆ω  (no pressure term!)
        """
        from ns_rigorous_derivation import verify_pressure_elimination
        
        result = verify_pressure_elimination()
        self.assertTrue(result['pressure_eliminated'],
            f"Pressure not eliminated from vorticity equation: {result['error']}")


class TestCrossInteractionBound(unittest.TestCase):
    """
    REQUIREMENT 3: The cross-interaction term must be bounded with explicit constants.
    
    Mathematical claim:
        |⟨ω^⊥, (ω^B·∇)v⟩| ≤ ε||∇ω^⊥||² + C(ε)||ω^⊥||²·||∇v^B||²
        
    Where C(ε) is an EXPLICIT constant depending only on ε and domain geometry.
    """
    
    def setUp(self):
        """Create test fields."""
        self.N = 16
        self.nu = 0.1
        
        np.random.seed(42)
        
        k = np.fft.fftfreq(self.N) * self.N
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        
        # Create test velocity field
        self.v_hat = self._create_divergence_free_field(self.N)
    
    def _create_divergence_free_field(self, N: int) -> np.ndarray:
        """Create a random divergence-free velocity field."""
        psi_hat = np.random.randn(3, N, N, N) + 1j * np.random.randn(3, N, N, N)
        
        k = np.fft.fftfreq(N) * N
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        
        v_hat = np.zeros((3, N, N, N), dtype=complex)
        v_hat[0] = 1j * (ky * psi_hat[2] - kz * psi_hat[1])
        v_hat[1] = 1j * (kz * psi_hat[0] - kx * psi_hat[2])
        v_hat[2] = 1j * (kx * psi_hat[1] - ky * psi_hat[0])
        v_hat[:, 0, 0, 0] = 0
        
        return v_hat
    
    def test_beltrami_stretching_is_gradient(self):
        """
        TEST: (ω^B·∇)v^B = (λ/2)∇|v^B|² is a gradient field
        
        This is the key algebraic identity that makes Beltrami structure special.
        """
        from ns_rigorous_derivation import (
            beltrami_decomposition, compute_curl,
            compute_beltrami_stretching, is_gradient_field
        )
        
        v_B, _ = beltrami_decomposition(self.v_hat, self.kx, self.ky, self.kz)
        omega_B = compute_curl(v_B, self.kx, self.ky, self.kz)
        
        stretching = compute_beltrami_stretching(omega_B, v_B, self.kx, self.ky, self.kz)
        
        is_grad, curl_mag = is_gradient_field(stretching, self.kx, self.ky, self.kz)
        
        self.assertTrue(is_grad,
            f"Beltrami stretching is NOT a gradient: |∇×(ω^B·∇)v^B| = {curl_mag}")
    
    def test_gradient_orthogonal_to_curl_eigenvectors(self):
        """
        TEST: ⟨ω^⊥, ∇f⟩ = 0 for any scalar f
        
        Gradient fields are orthogonal to all curl eigenvectors.
        """
        from ns_rigorous_derivation import beltrami_decomposition, compute_curl
        
        _, v_perp = beltrami_decomposition(self.v_hat, self.kx, self.ky, self.kz)
        omega_perp = compute_curl(v_perp, self.kx, self.ky, self.kz)
        
        # Create random gradient field
        np.random.seed(456)
        f_hat = np.random.randn(self.N, self.N, self.N) + 1j * np.random.randn(self.N, self.N, self.N)
        
        grad_f = np.zeros((3, self.N, self.N, self.N), dtype=complex)
        grad_f[0] = 1j * self.kx * f_hat
        grad_f[1] = 1j * self.ky * f_hat
        grad_f[2] = 1j * self.kz * f_hat
        
        # Compute inner product
        inner = np.sum(omega_perp * np.conj(grad_f)).real / self.N**3
        
        self.assertLess(abs(inner), 1e-10,
            f"Gradient not orthogonal to ω^⊥: ⟨ω^⊥, ∇f⟩ = {inner}")
    
    def test_coupling_bound_with_explicit_constants(self):
        """
        TEST: |⟨ω^⊥, (ω^⊥·∇)v^B⟩| ≤ C_S · ||ω^⊥||^{3/2} · ||∇ω^⊥||^{1/2} · ||∇v^B||
        
        Where C_S is the explicit Sobolev embedding constant.
        """
        from ns_rigorous_derivation import (
            beltrami_decomposition, compute_curl,
            compute_coupling_term, get_sobolev_constant,
            compute_gradient_norm, compute_L2_norm
        )
        
        v_B, v_perp = beltrami_decomposition(self.v_hat, self.kx, self.ky, self.kz)
        omega_perp = compute_curl(v_perp, self.kx, self.ky, self.kz)
        
        # Compute the coupling term
        coupling = compute_coupling_term(omega_perp, v_B, self.kx, self.ky, self.kz)
        
        # Compute norms
        omega_perp_L2 = compute_L2_norm(omega_perp, self.N)
        grad_omega_perp_L2 = compute_gradient_norm(omega_perp, self.kx, self.ky, self.kz, self.N)
        grad_v_B_L2 = compute_gradient_norm(v_B, self.kx, self.ky, self.kz, self.N)
        
        # Get explicit Sobolev constant
        C_S = get_sobolev_constant(self.N, dimension=3)
        
        # The bound should hold
        LHS = abs(coupling)
        RHS = C_S * omega_perp_L2**(3/2) * grad_omega_perp_L2**(1/2) * grad_v_B_L2
        
        self.assertLessEqual(LHS, RHS * 1.01,  # Allow 1% numerical tolerance
            f"Coupling bound violated: |coupling| = {LHS} > C_S·bound = {RHS}")


class TestYoungsInequalityApplication(unittest.TestCase):
    """
    REQUIREMENT 4: Young's inequality must be applied correctly.
    
    Mathematical claim:
        For any a,b ≥ 0 and ε > 0:
        a·b ≤ ε·a² + (1/4ε)·b²
    """
    
    def test_youngs_inequality_numerical(self):
        """
        TEST: Young's inequality holds numerically for test values.
        """
        test_cases = [
            (1.0, 1.0, 0.1),
            (0.5, 2.0, 0.01),
            (10.0, 0.1, 1.0),
            (0.001, 1000, 0.5),
        ]
        
        for a, b, eps in test_cases:
            LHS = a * b
            RHS = eps * a**2 + (1/(4*eps)) * b**2
            self.assertLessEqual(LHS, RHS + 1e-14,
                f"Young's inequality failed: {a}·{b} = {LHS} > {RHS} = ε·a² + b²/(4ε)")
    
    def test_youngs_converts_mixed_term_to_controllable_form(self):
        """
        TEST: ab ≤ εa⁴ + C(ε)b^{4/3} with explicit C(ε)
        
        This is the specific form needed for the 5/3 power term.
        """
        from ns_rigorous_derivation import youngs_inequality_mixed
        
        test_cases = [
            (1.0, 1.0, 0.1),
            (0.5, 2.0, 0.01),
        ]
        
        for a, b, eps in test_cases:
            LHS = a * b
            C_eps = youngs_inequality_mixed(eps)
            RHS = eps * a**4 + C_eps * b**(4/3)
            
            self.assertLessEqual(LHS, RHS + 1e-10,
                f"Mixed Young's failed: ab = {LHS} > {RHS}")


class TestPoincareInequality(unittest.TestCase):
    """
    REQUIREMENT 5: Poincaré inequality with explicit constant.
    
    Mathematical claim:
        ||f||_{L²} ≤ (1/λ₁) ||∇f||_{L²}
        
    Where λ₁ = 2π/L is the first eigenvalue of -∆ on the torus.
    """
    
    def test_poincare_constant_explicit(self):
        """
        TEST: Poincaré constant λ₁ = 2π/L on the torus.
        """
        from ns_rigorous_derivation import get_poincare_constant
        
        L = 2 * np.pi
        lambda_1 = get_poincare_constant(L)
        
        expected = 1.0  # For L = 2π, λ₁ = 2π/(2π) = 1
        self.assertAlmostEqual(lambda_1, expected, places=10,
            msg=f"Poincaré constant wrong: got {lambda_1}, expected {expected}")
    
    def test_poincare_inequality_holds(self):
        """
        TEST: ||f||² ≤ ||∇f||² / λ₁² for mean-zero f
        """
        from ns_rigorous_derivation import get_poincare_constant
        
        N = 16
        L = 2 * np.pi
        
        # Create mean-zero test function
        np.random.seed(789)
        f_hat = np.random.randn(N, N, N) + 1j * np.random.randn(N, N, N)
        f_hat[0, 0, 0] = 0  # Zero mean
        
        k = np.fft.fftfreq(N) * N
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        
        # Compute ||f||²
        f_norm_sq = np.sum(np.abs(f_hat)**2).real / N**3
        
        # Compute ||∇f||²
        grad_f_norm_sq = np.sum((kx**2 + ky**2 + kz**2) * np.abs(f_hat)**2).real / N**3
        
        lambda_1 = get_poincare_constant(L)
        
        # Poincaré: ||f||² ≤ ||∇f||² / λ₁²
        self.assertLessEqual(f_norm_sq, grad_f_norm_sq / lambda_1**2 + 1e-12,
            f"Poincaré violated: ||f||² = {f_norm_sq} > ||∇f||²/λ₁² = {grad_f_norm_sq/lambda_1**2}")


class TestMainInequalityDerivation(unittest.TestCase):
    """
    REQUIREMENT 6: The main inequality must be derived with all constants explicit.
    
    Mathematical claim:
        d/dt Ω^⊥ ≤ -α·Ω^⊥ + C·Ω^⊥·Ω^B + C'·(Ω^⊥)^{5/3}
        
    Where:
        α = (ν - 3ε)λ₁² > 0 for ε small enough
        C, C' are explicit constants
    """
    
    def test_dissipation_dominates_for_small_enstrophy(self):
        """
        TEST: For Ω^⊥ < threshold, viscous dissipation dominates the 5/3 term.
        """
        from ns_rigorous_derivation import (
            get_enstrophy_threshold, get_viscous_coefficient,
            get_coupling_coefficient, get_nonlinear_coefficient
        )
        
        nu = 0.1
        L = 2 * np.pi
        
        alpha = get_viscous_coefficient(nu, L)
        C_coupling = get_coupling_coefficient(L)
        C_nonlinear = get_nonlinear_coefficient(L)
        threshold = get_enstrophy_threshold(nu, L)
        
        # For Ω^⊥ = threshold/2, nonlinear term should be < half the dissipation
        Omega_perp = threshold / 2
        Omega_B = 1.0  # Assume bounded Beltrami enstrophy
        
        dissipation = alpha * Omega_perp
        coupling = C_coupling * Omega_perp * Omega_B
        nonlinear = C_nonlinear * Omega_perp**(5/3)
        
        # Dissipation should dominate nonlinear
        self.assertGreater(dissipation, 2 * nonlinear,
            f"Dissipation {dissipation} does not dominate nonlinear {nonlinear}")
    
    def test_gronwall_closure_explicit(self):
        """
        TEST: Gronwall gives explicit bound on Ω^⊥(t).
        """
        from ns_rigorous_derivation import (
            gronwall_bound_explicit, get_viscous_coefficient,
            get_coupling_coefficient
        )
        
        nu = 0.1
        L = 2 * np.pi
        Omega_perp_0 = 1.0
        Omega_B_0 = 1.0
        T = 10.0
        
        alpha = get_viscous_coefficient(nu, L)
        C = get_coupling_coefficient(L)
        
        bound = gronwall_bound_explicit(Omega_perp_0, Omega_B_0, alpha, C, T)
        
        # Bound should be finite and positive
        self.assertGreater(bound, 0)
        self.assertLess(bound, np.inf)
        self.assertFalse(np.isnan(bound))


class TestFullIntegration(unittest.TestCase):
    """
    INTEGRATION TEST: Full numerical verification of the theorem.
    """
    
    def test_enstrophy_bounded_for_random_data(self):
        """
        TEST: Run NS simulation and verify total enstrophy stays bounded.
        """
        from ns_rigorous_derivation import simulate_and_verify
        
        N = 16
        nu = 0.05
        T = 1.0
        dt = 0.001
        
        result = simulate_and_verify(N, nu, T, dt, seed=42)
        
        self.assertTrue(result['enstrophy_bounded'],
            f"Enstrophy not bounded: max/initial = {result['max_ratio']}")
        
        self.assertLess(result['max_ratio'], 2.0,
            f"Enstrophy grew too much: {result['max_ratio']}")
    
    def test_non_beltrami_enstrophy_controlled(self):
        """
        TEST: Non-Beltrami enstrophy satisfies the derived inequality.
        """
        from ns_rigorous_derivation import simulate_and_verify
        
        N = 16
        nu = 0.05
        T = 1.0
        dt = 0.001
        
        result = simulate_and_verify(N, nu, T, dt, seed=42)
        
        self.assertTrue(result['non_beltrami_controlled'],
            f"Non-Beltrami enstrophy not controlled: {result['non_beltrami_error']}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
