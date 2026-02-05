"""
Three-Layered Interphase Model for Particle-Reinforced Composites (3D/CSA)

Python port of Three_Layered_Interphase_3D.m
Computes K and G of particle-reinforced composite with 3-layer interphase

METHOD: CSA (Composite Sphere Assemblage) + Mori-Tanaka Homogenization

Author: Sumanth Reddy Settipalli
"""

import numpy as np
from typing import Tuple, Dict, Optional


def compute_3layer_properties(
    kappa_inc: float,
    mu_inc: float,
    kappa_mat: float,
    mu_mat: float,
    phi_1: float = 0.2,
    phi_2: float = 0.3,
    phi_3: float = 0.4,
    f_particle: float = 0.3,
    interp_weights: Optional[Tuple[float, float, float]] = None
) -> Tuple[float, float]:
    """
    Compute effective bulk and shear moduli using 3-Layer Interphase model.

    Args:
        kappa_inc: Inclusion bulk modulus
        mu_inc: Inclusion shear modulus
        kappa_mat: Matrix bulk modulus
        mu_mat: Matrix shear modulus
        phi_1: Volume ratio for layer 1 (particle/layer1 interface)
        phi_2: Volume ratio for layer 2 (layer1/layer2 interface)
        phi_3: Volume ratio for layer 3 (layer2/layer3 interface)
        f_particle: Volume fraction of particles in composite
        interp_weights: Optional tuple (w1, w2, w3) for interpolation weights
                       Default: (0.75, 0.50, 0.25) - linear interpolation

    Returns:
        K_composite: Effective bulk modulus
        G_composite: Effective shear modulus
    """
    # Default interpolation weights (from particle to matrix)
    if interp_weights is None:
        interp_weights = (0.75, 0.50, 0.25)

    w1, w2, w3 = interp_weights

    # Interphase layer properties (interpolated)
    coat1_kappa = w1 * kappa_inc + (1 - w1) * kappa_mat
    coat1_mu = w1 * mu_inc + (1 - w1) * mu_mat

    coat2_kappa = w2 * kappa_inc + (1 - w2) * kappa_mat
    coat2_mu = w2 * mu_inc + (1 - w2) * mu_mat

    coat3_kappa = w3 * kappa_inc + (1 - w3) * kappa_mat
    coat3_mu = w3 * mu_inc + (1 - w3) * mu_mat

    # Solve Volumetric BVP (8x8 system)
    T1_b, H1_b = solve_volumetric_bvp_3layer(
        kappa_inc, mu_inc,
        coat1_kappa, coat1_mu,
        coat2_kappa, coat2_mu,
        coat3_kappa, coat3_mu,
        kappa_mat, mu_mat,
        phi_1, phi_2, phi_3
    )

    # Solve Deviatoric BVP (16x16 system)
    T1_s, H1_s = solve_deviatoric_bvp_3layer(
        kappa_inc, mu_inc,
        coat1_kappa, coat1_mu,
        coat2_kappa, coat2_mu,
        coat3_kappa, coat3_mu,
        kappa_mat, mu_mat,
        phi_1, phi_2, phi_3
    )

    # Volume fractions for each phase within equivalent particle
    c_particle = phi_1 * phi_2 * phi_3
    c_coat1 = phi_2 * phi_3 - c_particle
    c_coat2 = phi_3 - phi_2 * phi_3
    c_coat3 = 1 - phi_3

    # Equivalent particle tensors (scalar representations)
    T_eq_b = c_particle * T1_b['particle'] + c_coat1 * T1_b['coat1'] + \
             c_coat2 * T1_b['coat2'] + c_coat3 * T1_b['coat3']
    T_eq_s = c_particle * T1_s['particle'] + c_coat1 * T1_s['coat1'] + \
             c_coat2 * T1_s['coat2'] + c_coat3 * T1_s['coat3']

    H_eq_b = c_particle * H1_b['particle'] + c_coat1 * H1_b['coat1'] + \
             c_coat2 * H1_b['coat2'] + c_coat3 * H1_b['coat3']
    H_eq_s = c_particle * H1_s['particle'] + c_coat1 * H1_s['coat1'] + \
             c_coat2 * H1_s['coat2'] + c_coat3 * H1_s['coat3']

    # Mori-Tanaka homogenization
    # For isotropic case, reduces to scalar operations
    A_b = 1.0 / (f_particle * T_eq_b + (1 - f_particle))
    A_s = 1.0 / (f_particle * T_eq_s + (1 - f_particle))

    K_composite = ((1 - f_particle) * kappa_mat + f_particle * H_eq_b / 3) * A_b
    G_composite = ((1 - f_particle) * mu_mat + f_particle * H_eq_s / 2) * A_s

    return K_composite, G_composite


def solve_volumetric_bvp_3layer(
    particle_kappa, particle_mu,
    coat1_kappa, coat1_mu,
    coat2_kappa, coat2_mu,
    coat3_kappa, coat3_mu,
    matrix_kappa, matrix_mu,
    phi_1, phi_2, phi_3
) -> Tuple[Dict, Dict]:
    """
    Solve the 8x8 volumetric BVP for the 3-layer interphase model.

    Returns:
        T1_b: Dict with bulk strain concentration for each phase
        H1_b: Dict with bulk stress concentration for each phase
    """
    A = np.zeros((8, 8))
    b = np.zeros(8)

    # Interface 1: Particle to Coating 1
    A[0, 0] = 1
    A[0, 1] = -1
    A[0, 2] = -1
    A[1, 0] = 3 * particle_kappa
    A[1, 1] = -3 * coat1_kappa
    A[1, 2] = 4 * coat1_mu

    # Interface 2: Coating 1 to Coating 2
    A[2, 1] = 1
    A[2, 2] = phi_1
    A[2, 3] = -1
    A[2, 4] = -1
    A[3, 1] = 3 * coat1_kappa
    A[3, 2] = -4 * coat1_mu * phi_1
    A[3, 3] = -3 * coat2_kappa
    A[3, 4] = 4 * coat2_mu

    # Interface 3: Coating 2 to Coating 3
    A[4, 3] = 1
    A[4, 4] = phi_2
    A[4, 5] = -1
    A[4, 6] = -1
    A[5, 3] = 3 * coat2_kappa
    A[5, 4] = -4 * coat2_mu * phi_2
    A[5, 5] = -3 * coat3_kappa
    A[5, 6] = 4 * coat3_mu

    # Interface 4: Coating 3 to Matrix
    A[6, 5] = 1
    A[6, 6] = phi_3
    A[6, 7] = -1
    A[7, 5] = 3 * coat3_kappa
    A[7, 6] = -4 * coat3_mu * phi_3
    A[7, 7] = 4 * matrix_mu

    b[6] = 1
    b[7] = 3 * matrix_kappa

    Xi = np.linalg.solve(A, b)

    T1_b = {
        'particle': Xi[0],
        'coat1': Xi[1],
        'coat2': Xi[3],
        'coat3': Xi[5]
    }

    H1_b = {
        'particle': 3 * particle_kappa * T1_b['particle'],
        'coat1': 3 * coat1_kappa * T1_b['coat1'],
        'coat2': 3 * coat2_kappa * T1_b['coat2'],
        'coat3': 3 * coat3_kappa * T1_b['coat3']
    }

    return T1_b, H1_b


def solve_deviatoric_bvp_3layer(
    particle_kappa, particle_mu,
    coat1_kappa, coat1_mu,
    coat2_kappa, coat2_mu,
    coat3_kappa, coat3_mu,
    matrix_kappa, matrix_mu,
    phi_1, phi_2, phi_3
) -> Tuple[Dict, Dict]:
    """
    Solve the 16x16 deviatoric BVP for the 3-layer interphase model.

    Returns:
        T1_s: Dict with shear strain concentration for each phase
        H1_s: Dict with shear stress concentration for each phase
    """
    # X coefficients for each material
    def get_X_coeffs(kappa, mu):
        X1 = 1
        X2 = (9 * kappa - 6 * mu) / (15 * kappa + 11 * mu)
        X3 = -3/2
        X4 = (3 * kappa + 3 * mu) / (2 * mu)
        return X1, X2, X3, X4

    X_p = get_X_coeffs(particle_kappa, particle_mu)
    X_c1 = get_X_coeffs(coat1_kappa, coat1_mu)
    X_c2 = get_X_coeffs(coat2_kappa, coat2_mu)
    X_c3 = get_X_coeffs(coat3_kappa, coat3_mu)
    X_m = get_X_coeffs(matrix_kappa, matrix_mu)

    # Stress coefficient functions
    def compute_stress_coeffs(kappa, mu, eigenvalues):
        X1, X2, X3, X4 = get_X_coeffs(kappa, mu)
        X_map = {1: X1, 3: X2, -4: X3, -2: X4}

        SRR = []
        SRT = []
        for xi in eigenvalues:
            Xi = X_map[xi]
            srr = (3 * kappa * (Xi * xi + 2 * Xi - 3) + 2 * mu * (2 * Xi * xi - 2 * Xi + 3)) / 3
            srt = mu * (2 * Xi + xi - 1)
            SRR.append(srr)
            SRT.append(srt)
        return SRR, SRT

    SRR_p, SRT_p = compute_stress_coeffs(particle_kappa, particle_mu, [1, 3])
    SRR_c1, SRT_c1 = compute_stress_coeffs(coat1_kappa, coat1_mu, [1, 3, -4, -2])
    SRR_c2, SRT_c2 = compute_stress_coeffs(coat2_kappa, coat2_mu, [1, 3, -4, -2])
    SRR_c3, SRT_c3 = compute_stress_coeffs(coat3_kappa, coat3_mu, [1, 3, -4, -2])
    SRR_m, SRT_m = compute_stress_coeffs(matrix_kappa, matrix_mu, [-4, -2])

    # TERM2 factors
    def compute_term2(kappa, mu, phi=None):
        base = (63 * kappa + 21 * mu) / (75 * kappa + 55 * mu)
        if phi is not None:
            base *= (phi ** (-2/3) - phi) / (1 - phi)
        return base

    TERM2_particle = compute_term2(particle_kappa, particle_mu)
    TERM2_coat1 = compute_term2(coat1_kappa, coat1_mu, phi_1)
    TERM2_coat2 = compute_term2(coat2_kappa, coat2_mu, phi_2)
    TERM2_coat3 = compute_term2(coat3_kappa, coat3_mu, phi_3)

    # Build 16x16 system
    A = np.zeros((16, 16))
    b = np.zeros(16)

    # Interface 1: Particle to Coating 1 (rows 0-3)
    A[0, 0] = X_p[0]; A[0, 1] = X_p[1]
    A[0, 2] = -X_c1[0]; A[0, 3] = -X_c1[1]; A[0, 4] = -X_c1[2]; A[0, 5] = -X_c1[3]

    A[1, 0] = 1; A[1, 1] = 1
    A[1, 2] = -1; A[1, 3] = -1; A[1, 4] = -1; A[1, 5] = -1

    A[2, 0] = SRR_p[0]; A[2, 1] = SRR_p[1]
    A[2, 2] = -SRR_c1[0]; A[2, 3] = -SRR_c1[1]; A[2, 4] = -SRR_c1[2]; A[2, 5] = -SRR_c1[3]

    A[3, 0] = SRT_p[0]; A[3, 1] = SRT_p[1]
    A[3, 2] = -SRT_c1[0]; A[3, 3] = -SRT_c1[1]; A[3, 4] = -SRT_c1[2]; A[3, 5] = -SRT_c1[3]

    # Interface 2: Coating 1 to Coating 2 (rows 4-7)
    pow2 = phi_1 ** ((3-1)/3)
    pow3 = phi_1 ** ((-4-1)/3)
    pow4 = phi_1 ** ((-2-1)/3)

    A[4, 2] = X_c1[0]; A[4, 3] = X_c1[1] * pow2; A[4, 4] = X_c1[2] * pow3; A[4, 5] = X_c1[3] * pow4
    A[4, 6] = -X_c2[0]; A[4, 7] = -X_c2[1]; A[4, 8] = -X_c2[2]; A[4, 9] = -X_c2[3]

    A[5, 2] = 1; A[5, 3] = pow2; A[5, 4] = pow3; A[5, 5] = pow4
    A[5, 6] = -1; A[5, 7] = -1; A[5, 8] = -1; A[5, 9] = -1

    A[6, 2] = SRR_c1[0]; A[6, 3] = SRR_c1[1] * pow2; A[6, 4] = SRR_c1[2] * pow3; A[6, 5] = SRR_c1[3] * pow4
    A[6, 6] = -SRR_c2[0]; A[6, 7] = -SRR_c2[1]; A[6, 8] = -SRR_c2[2]; A[6, 9] = -SRR_c2[3]

    A[7, 2] = SRT_c1[0]; A[7, 3] = SRT_c1[1] * pow2; A[7, 4] = SRT_c1[2] * pow3; A[7, 5] = SRT_c1[3] * pow4
    A[7, 6] = -SRT_c2[0]; A[7, 7] = -SRT_c2[1]; A[7, 8] = -SRT_c2[2]; A[7, 9] = -SRT_c2[3]

    # Interface 3: Coating 2 to Coating 3 (rows 8-11)
    pow2 = phi_2 ** ((3-1)/3)
    pow3 = phi_2 ** ((-4-1)/3)
    pow4 = phi_2 ** ((-2-1)/3)

    A[8, 6] = X_c2[0]; A[8, 7] = X_c2[1] * pow2; A[8, 8] = X_c2[2] * pow3; A[8, 9] = X_c2[3] * pow4
    A[8, 10] = -X_c3[0]; A[8, 11] = -X_c3[1]; A[8, 12] = -X_c3[2]; A[8, 13] = -X_c3[3]

    A[9, 6] = 1; A[9, 7] = pow2; A[9, 8] = pow3; A[9, 9] = pow4
    A[9, 10] = -1; A[9, 11] = -1; A[9, 12] = -1; A[9, 13] = -1

    A[10, 6] = SRR_c2[0]; A[10, 7] = SRR_c2[1] * pow2; A[10, 8] = SRR_c2[2] * pow3; A[10, 9] = SRR_c2[3] * pow4
    A[10, 10] = -SRR_c3[0]; A[10, 11] = -SRR_c3[1]; A[10, 12] = -SRR_c3[2]; A[10, 13] = -SRR_c3[3]

    A[11, 6] = SRT_c2[0]; A[11, 7] = SRT_c2[1] * pow2; A[11, 8] = SRT_c2[2] * pow3; A[11, 9] = SRT_c2[3] * pow4
    A[11, 10] = -SRT_c3[0]; A[11, 11] = -SRT_c3[1]; A[11, 12] = -SRT_c3[2]; A[11, 13] = -SRT_c3[3]

    # Interface 4: Coating 3 to Matrix (rows 12-15)
    pow2 = phi_3 ** ((3-1)/3)
    pow3 = phi_3 ** ((-4-1)/3)
    pow4 = phi_3 ** ((-2-1)/3)

    A[12, 10] = X_c3[0]; A[12, 11] = X_c3[1] * pow2; A[12, 12] = X_c3[2] * pow3; A[12, 13] = X_c3[3] * pow4
    A[12, 14] = -X_m[2]; A[12, 15] = -X_m[3]

    A[13, 10] = 1; A[13, 11] = pow2; A[13, 12] = pow3; A[13, 13] = pow4
    A[13, 14] = -1; A[13, 15] = -1

    A[14, 10] = SRR_c3[0]; A[14, 11] = SRR_c3[1] * pow2; A[14, 12] = SRR_c3[2] * pow3; A[14, 13] = SRR_c3[3] * pow4
    A[14, 14] = -SRR_m[0]; A[14, 15] = -SRR_m[1]

    A[15, 10] = SRT_c3[0]; A[15, 11] = SRT_c3[1] * pow2; A[15, 12] = SRT_c3[2] * pow3; A[15, 13] = SRT_c3[3] * pow4
    A[15, 14] = -SRT_m[0]; A[15, 15] = -SRT_m[1]

    b[12] = 1
    b[13] = 1
    b[14] = 2 * matrix_mu
    b[15] = 2 * matrix_mu

    Xi = np.linalg.solve(A, b)

    T1_s = {
        'particle': Xi[0] + TERM2_particle * Xi[1],
        'coat1': Xi[2] + TERM2_coat1 * Xi[3],
        'coat2': Xi[6] + TERM2_coat2 * Xi[7],
        'coat3': Xi[10] + TERM2_coat3 * Xi[11]
    }

    H1_s = {
        'particle': 2 * particle_mu * T1_s['particle'],
        'coat1': 2 * coat1_mu * T1_s['coat1'],
        'coat2': 2 * coat2_mu * T1_s['coat2'],
        'coat3': 2 * coat3_mu * T1_s['coat3']
    }

    return T1_s, H1_s


def validate_against_matlab():
    """
    Validate Python implementation against MATLAB reference values.
    Uses the same parameters as Model_Results.md
    """
    # Reference parameters
    kappa_inc = 1.2 + (2/3) * 0.8  # = 1.7333
    mu_inc = 0.8
    kappa_mat = 12 + (2/3) * 8     # = 17.3333
    mu_mat = 8.0

    phi_1 = 0.2
    phi_2 = 0.3
    phi_3 = 0.4
    f_particle = 0.3

    K, G = compute_3layer_properties(
        kappa_inc, mu_inc, kappa_mat, mu_mat,
        phi_1, phi_2, phi_3, f_particle
    )

    print("3-Layer Interphase Model Validation")
    print("=" * 50)
    print(f"Inputs:")
    print(f"  Particle: kappa={kappa_inc:.4f}, mu={mu_inc:.4f}")
    print(f"  Matrix: kappa={kappa_mat:.4f}, mu={mu_mat:.4f}")
    print(f"  Phi values: {phi_1}, {phi_2}, {phi_3}")
    print(f"  Volume fraction: f={f_particle}")
    print()
    print(f"Results:")
    print(f"  K_composite = {K:.6f}")
    print(f"  G_composite = {G:.6f}")
    print()
    print("Expected (from MATLAB/Model_Results.md):")
    print("  K_composite ~ 14.91")
    print("  G_composite ~ 8.02")

    return K, G


if __name__ == "__main__":
    validate_against_matlab()
