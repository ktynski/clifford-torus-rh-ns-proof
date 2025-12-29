#!/usr/bin/env python3
"""
Test: Does φ-Beltrami structure survive under Navier-Stokes evolution?

This is the critical test. If the answer is NO, then the density argument fails.
"""

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.integrate import solve_ivp

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

def create_phi_beltrami_field(N=32):
    """
    Create a φ-Beltrami initial condition.
    
    Beltrami: ∇×v = λv (curl equals scaled velocity)
    φ-structure: wavenumbers related by golden ratio
    """
    # Grid
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # φ-Beltrami mode: k = (1, 1/φ, 1/φ²) normalized
    k1, k2, k3 = 1.0, 1.0/PHI, 1.0/PHI**2
    k_mag = np.sqrt(k1**2 + k2**2 + k3**2)
    
    # Beltrami field: v parallel to k × (k × e_z) + i k × e_z
    # Simplified: ABC flow-like structure
    A, B, C = 1.0, 1.0/PHI, 1.0/PHI**2  # φ-structured amplitudes
    
    u = A * np.sin(k1*Z) + C * np.cos(k3*Y)
    v = B * np.sin(k2*X) + A * np.cos(k1*Z)
    w = C * np.sin(k3*Y) + B * np.cos(k2*X)
    
    return np.stack([u, v, w], axis=-1)


def compute_curl(v, dx=2*np.pi/32):
    """Compute curl of velocity field using spectral method."""
    N = v.shape[0]
    
    # Fourier transform each component
    v_hat = np.stack([fftn(v[..., i]) for i in range(3)], axis=-1)
    
    # Wavenumbers
    k = np.fft.fftfreq(N, d=dx/(2*np.pi)) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    
    # Curl in Fourier space: ω_hat = i k × v_hat
    omega_hat = np.zeros_like(v_hat, dtype=complex)
    omega_hat[..., 0] = 1j * (ky * v_hat[..., 2] - kz * v_hat[..., 1])
    omega_hat[..., 1] = 1j * (kz * v_hat[..., 0] - kx * v_hat[..., 2])
    omega_hat[..., 2] = 1j * (kx * v_hat[..., 1] - ky * v_hat[..., 0])
    
    # Inverse transform
    omega = np.stack([np.real(ifftn(omega_hat[..., i])) for i in range(3)], axis=-1)
    return omega


def compute_divergence(v, dx=2*np.pi/32):
    """Compute divergence (should be ~0 for incompressible)."""
    N = v.shape[0]
    v_hat = np.stack([fftn(v[..., i]) for i in range(3)], axis=-1)
    
    k = np.fft.fftfreq(N, d=dx/(2*np.pi)) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    
    div_hat = 1j * (kx * v_hat[..., 0] + ky * v_hat[..., 1] + kz * v_hat[..., 2])
    return np.real(ifftn(div_hat))


def measure_beltrami_deviation(v):
    """
    Measure how far v is from being Beltrami.
    
    For Beltrami: ω = λv for some λ
    Deviation = ||ω - λ_opt v|| / ||ω||
    where λ_opt minimizes the ratio
    """
    omega = compute_curl(v)
    
    # Optimal λ via least squares: λ = (ω · v) / (v · v)
    omega_flat = omega.reshape(-1)
    v_flat = v.reshape(-1)
    
    lambda_opt = np.dot(omega_flat, v_flat) / (np.dot(v_flat, v_flat) + 1e-10)
    
    # Deviation
    residual = omega - lambda_opt * v
    deviation = np.linalg.norm(residual) / (np.linalg.norm(omega) + 1e-10)
    
    return deviation, lambda_opt


def ns_rhs_spectral(t, v_flat, N, nu, dx):
    """
    Right-hand side of NS in Fourier space (simplified).
    
    ∂v/∂t = -v·∇v - ∇p + ν∇²v
    """
    v = v_flat.reshape(N, N, N, 3)
    v_hat = np.stack([fftn(v[..., i]) for i in range(3)], axis=-1)
    
    k = np.fft.fftfreq(N, d=dx/(2*np.pi)) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1  # Avoid division by zero
    
    # Nonlinear term: (v·∇)v in Fourier space (pseudo-spectral)
    # Compute in physical space, transform back
    dvdx = np.stack([np.real(ifftn(1j * kx * v_hat[..., i])) for i in range(3)], axis=-1)
    dvdy = np.stack([np.real(ifftn(1j * ky * v_hat[..., i])) for i in range(3)], axis=-1)
    dvdz = np.stack([np.real(ifftn(1j * kz * v_hat[..., i])) for i in range(3)], axis=-1)
    
    nonlin = v[..., 0:1] * dvdx + v[..., 1:2] * dvdy + v[..., 2:3] * dvdz
    nonlin_hat = np.stack([fftn(nonlin[..., i]) for i in range(3)], axis=-1)
    
    # Pressure projection (remove divergent part)
    div_nonlin = kx * nonlin_hat[..., 0] + ky * nonlin_hat[..., 1] + kz * nonlin_hat[..., 2]
    p_hat = div_nonlin / k_sq
    p_hat[0, 0, 0] = 0
    
    grad_p_hat = np.stack([1j * kx * p_hat, 1j * ky * p_hat, 1j * kz * p_hat], axis=-1)
    
    # Viscous term: ν∇²v
    visc_hat = -nu * k_sq[..., None] * v_hat
    
    # Total RHS
    rhs_hat = -nonlin_hat - grad_p_hat + visc_hat
    
    # Transform back
    rhs = np.stack([np.real(ifftn(rhs_hat[..., i])) for i in range(3)], axis=-1)
    
    return rhs.flatten()


def evolve_ns(v0, t_final, nu=0.01, N=32):
    """Evolve NS equations from initial condition v0."""
    dx = 2*np.pi / N
    
    def rhs(t, y):
        return ns_rhs_spectral(t, y, N, nu, dx)
    
    # Use implicit method for stability
    sol = solve_ivp(
        rhs, 
        [0, t_final], 
        v0.flatten(),
        method='RK45',
        max_step=0.01,
        rtol=1e-6,
        atol=1e-8
    )
    
    return sol.y[:, -1].reshape(N, N, N, 3), sol.t


def compute_enstrophy(v):
    """Compute enstrophy Ω = ∫|ω|² dV."""
    omega = compute_curl(v)
    return np.mean(omega**2)


def test_beltrami_preservation():
    """
    Main test: Does φ-Beltrami structure survive NS evolution?
    """
    print("=" * 70)
    print("TEST: Beltrami Structure Preservation Under NS Evolution")
    print("=" * 70)
    
    N = 32
    nu = 0.01
    
    # Create initial φ-Beltrami field
    print("\n1. Creating φ-Beltrami initial condition...")
    v0 = create_phi_beltrami_field(N)
    
    # Check initial properties
    dev0, lambda0 = measure_beltrami_deviation(v0)
    div0 = np.max(np.abs(compute_divergence(v0)))
    enstrophy0 = compute_enstrophy(v0)
    
    print(f"   Initial Beltrami deviation: {dev0:.6f}")
    print(f"   Initial λ (eigenvalue): {lambda0:.6f}")
    print(f"   Initial divergence (max): {div0:.2e}")
    print(f"   Initial enstrophy: {enstrophy0:.6f}")
    
    # Evolve
    print("\n2. Evolving under Navier-Stokes (ν = {})...".format(nu))
    
    times = [0.1, 0.5, 1.0]
    results = []
    
    v_current = v0
    t_current = 0
    
    for t_target in times:
        print(f"\n   Evolving to t = {t_target}...")
        try:
            v_evolved, t_history = evolve_ns(v_current, t_target - t_current, nu=nu, N=N)
            
            dev, lam = measure_beltrami_deviation(v_evolved)
            div = np.max(np.abs(compute_divergence(v_evolved)))
            enstrophy = compute_enstrophy(v_evolved)
            
            results.append({
                't': t_target,
                'deviation': dev,
                'lambda': lam,
                'divergence': div,
                'enstrophy': enstrophy,
                'enstrophy_ratio': enstrophy / enstrophy0
            })
            
            print(f"   t = {t_target}: deviation = {dev:.4f}, λ = {lam:.4f}, Ω/Ω₀ = {enstrophy/enstrophy0:.4f}")
            
            v_current = v_evolved
            t_current = t_target
            
        except Exception as e:
            print(f"   Error at t = {t_target}: {e}")
            break
    
    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n   t     | Beltrami Dev | λ (eigenvalue) | Ω/Ω₀  | Divergence")
    print("   " + "-" * 60)
    print(f"   0.00  | {dev0:.4f}       | {lambda0:.4f}         | 1.0000 | {div0:.2e}")
    for r in results:
        print(f"   {r['t']:.2f}  | {r['deviation']:.4f}       | {r['lambda']:.4f}         | {r['enstrophy_ratio']:.4f} | {r['divergence']:.2e}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if len(results) > 0:
        final_dev = results[-1]['deviation']
        final_enstrophy_ratio = results[-1]['enstrophy_ratio']
        
        if final_dev < 0.1:
            print("\n   ✓ Beltrami structure APPROXIMATELY PRESERVED (deviation < 10%)")
            print("     → The density argument may be salvageable")
        elif final_dev < 0.5:
            print("\n   ⚠ Beltrami structure PARTIALLY PRESERVED (deviation < 50%)")
            print("     → Need careful analysis of error growth")
        else:
            print("\n   ✗ Beltrami structure NOT PRESERVED (deviation ≥ 50%)")
            print("     → The density argument has a serious gap")
            print("     → Need to either:")
            print("       (a) Use spectral Galerkin with uniform estimates, or")
            print("       (b) Weaken the claim to 'φ-Beltrami class has regularity'")
        
        if final_enstrophy_ratio <= 1.01:
            print(f"\n   ✓ Enstrophy bound HOLDS: Ω(t)/Ω(0) = {final_enstrophy_ratio:.4f} ≤ 1")
        else:
            print(f"\n   ✗ Enstrophy bound VIOLATED: Ω(t)/Ω(0) = {final_enstrophy_ratio:.4f} > 1")
    
    return results


def test_mode_coupling():
    """
    Test: Does the nonlinear term couple φ-Beltrami modes?
    
    If v = v₁ + v₂ where v₁, v₂ are Beltrami with different λ,
    then (v·∇)v generates non-Beltrami components.
    """
    print("\n" + "=" * 70)
    print("TEST: Mode Coupling in Nonlinear Term")
    print("=" * 70)
    
    N = 32
    
    # Create two Beltrami modes with different eigenvalues
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    z = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Mode 1: k₁ = (1, 0, 0)
    v1 = np.zeros((N, N, N, 3))
    v1[..., 1] = np.cos(X)
    v1[..., 2] = np.sin(X)
    
    # Mode 2: k₂ = (0, 1, 0)  
    v2 = np.zeros((N, N, N, 3))
    v2[..., 0] = np.cos(Y)
    v2[..., 2] = np.sin(Y)
    
    # Each is Beltrami individually
    dev1, lam1 = measure_beltrami_deviation(v1)
    dev2, lam2 = measure_beltrami_deviation(v2)
    
    print(f"\n   Mode 1: Beltrami deviation = {dev1:.6f}, λ = {lam1:.4f}")
    print(f"   Mode 2: Beltrami deviation = {dev2:.6f}, λ = {lam2:.4f}")
    
    # Sum is NOT Beltrami
    v_sum = v1 + v2
    dev_sum, lam_sum = measure_beltrami_deviation(v_sum)
    
    print(f"\n   Sum v₁+v₂: Beltrami deviation = {dev_sum:.4f}")
    
    if dev_sum > 0.1:
        print("\n   ✓ CONFIRMED: Sum of Beltrami modes is NOT Beltrami")
        print("     → The nonlinear term WILL couple modes")
        print("     → φ-Beltrami structure is NOT automatically preserved")
    else:
        print("\n   Unexpected: Sum appears Beltrami-like")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BELTRAMI EVOLUTION INVESTIGATION")
    print("=" * 70)
    print("\nThis test determines whether the NS density argument is valid.\n")
    
    # Test 1: Mode coupling
    test_mode_coupling()
    
    # Test 2: Evolution preservation
    results = test_beltrami_preservation()
    
    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)
