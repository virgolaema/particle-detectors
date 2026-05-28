"""Electromagnetic and hadronic shower physics library."""

import numpy as np
from scipy.special import gamma as gamma_func

# ===================================================================
# SHOWER MATERIAL DATABASE
# ===================================================================
# Properties: Z, A, density (g/cm³), X0 radiation length (cm),
#             lambda_I nuclear interaction length (cm), Ec critical energy (MeV)

MATERIALS = {
    'lead': {
        'Z': 82, 'A': 207.2, 'density': 11.34,
        'X0': 0.56, 'lambda_I': 17.1, 'Ec': 7.43,
        'color': '#C73E1D',
    },
    'iron': {
        'Z': 26, 'A': 55.845, 'density': 7.87,
        'X0': 1.76, 'lambda_I': 16.8, 'Ec': 21.8,
        'color': '#F18F01',
    },
    'aluminum': {
        'Z': 13, 'A': 26.98, 'density': 2.70,
        'X0': 8.90, 'lambda_I': 38.4, 'Ec': 42.7,
        'color': '#A23B72',
    },
    'copper': {
        'Z': 29, 'A': 63.55, 'density': 8.96,
        'X0': 1.43, 'lambda_I': 15.1, 'Ec': 19.6,
        'color': '#E67E22',
    },
    'plastic': {
        'Z': 6, 'A': 12.01, 'density': 1.03,
        'X0': 42.7, 'lambda_I': 79.5, 'Ec': 81.2,
        'color': '#2E86AB',
    },
    'tungsten': {
        'Z': 74, 'A': 183.8, 'density': 19.3,
        'X0': 0.35, 'lambda_I': 9.6, 'Ec': 7.97,
        'color': '#8E44AD',
    },
    'concrete': {
        'Z': 11, 'A': 22, 'density': 2.3,
        'X0': 10.7, 'lambda_I': 39.3, 'Ec': 49.2,
        'color': '#27AE60',
    },
}

# ===================================================================
# ELECTROMAGNETIC SHOWER FUNCTIONS
# ===================================================================

def calculate_critical_energy(Z):
    """
    Critical energy where radiation losses equal ionization losses.
    Empirical: Ec ≈ 610 MeV / (Z + 1.24)
    """
    return 610.0 / (Z + 1.24)


def calculate_moliere_radius(X0, Z, A=None, density=None):
    """
    Molière radius for transverse containment of EM showers.
    RM = X0 * Es / Ec  where Es ≈ 21.2 MeV
    """
    Es = 21.2  # MeV (scale energy)
    Ec = calculate_critical_energy(Z)
    return X0 * (Es / Ec)


def em_shower_maximum(E0, Ec):
    """
    Depth of shower maximum in radiation lengths.

    Returns
    -------
    tmax_electron, tmax_photon : float
        Shower max depth for electron- and photon-initiated showers
    """
    if E0 <= Ec:
        return 0.5, 0.5
    tmax_electron = np.log(E0 / Ec) - 0.5
    tmax_photon = np.log(E0 / Ec) + 0.5
    return tmax_electron, tmax_photon


def em_longitudinal_profile(t, tmax, s=0.5):
    """
    Longitudinal shower profile N(t): number of particles vs depth.

    Uses gamma-function parametrization.

    Parameters
    ----------
    t : array-like
        Depth in radiation lengths
    tmax : float
        Shower maximum depth (radiation lengths)
    s : float
        Shower width parameter
    """
    tmax = max(tmax, 1)
    a = tmax / s
    t = np.asarray(t, dtype=float)
    profile = (t / s) ** (a - 1) * np.exp(-t / s)
    peak = np.max(profile)
    if peak > 0:
        profile /= peak
    return profile


def em_lateral_profile(r, RM):
    """
    Lateral shower profile (radial distance from axis).

    Parameters
    ----------
    r : array-like
        Radial distance in cm
    RM : float
        Molière radius in cm
    """
    r = np.asarray(r, dtype=float)
    return np.exp(-(r / RM) ** 2 / 2) * (1 + (r / RM) ** 2 / 8)


def em_total_particles(E0, Ec):
    """Estimate peak particle count: Nmax ≈ E0 / Ec."""
    if E0 <= Ec:
        return 1
    return E0 / Ec


# ===================================================================
# HADRONIC SHOWER FUNCTIONS
# ===================================================================

def hadronic_shower_development(E0, lambda_I, f_em=0.3):
    """
    Hadronic shower decomposition into EM and hadronic components.

    Parameters
    ----------
    E0 : float
        Initial hadron energy (MeV)
    lambda_I : float
        Nuclear interaction length (cm)
    f_em : float
        EM fraction from π⁰ → γγ (typically ~30%)

    Returns
    -------
    E_em, E_had, tmax_had : float
    """
    E_em = f_em * E0
    E_had = (1 - f_em) * E0
    E0_GeV = E0 / 1000.0
    tmax_had = max(0.2 * np.log(E0_GeV) + 0.7, 1.0) if E0_GeV >= 1 else 1.0
    return E_em, E_had, tmax_had


def hadronic_longitudinal_profile(t, tmax_had, sigma_had=0.8):
    """
    Longitudinal profile for hadronic showers (broader than EM).

    Parameters
    ----------
    t : array-like
        Depth in interaction lengths
    tmax_had : float
        Hadronic shower maximum depth
    sigma_had : float
        Shower width (larger than EM)
    """
    t = np.asarray(t, dtype=float)
    profile = np.exp(-0.5 * ((t - tmax_had) / sigma_had) ** 2)
    tail = 0.1 * np.exp(-t / (3 * tmax_had))
    total = profile + tail
    peak = np.max(total)
    if peak > 0:
        total /= peak
    return total


def hadronic_lateral_profile(r, lambda_I):
    """
    Lateral profile for hadronic showers. Broader than EM.

    Parameters
    ----------
    r : array-like
        Radial distance (cm)
    lambda_I : float
        Nuclear interaction length (cm)
    """
    R_had = lambda_I / 3.0
    r = np.asarray(r, dtype=float)
    return np.exp(-(r / R_had) ** 1.5 / 2)


def shower_fluctuations(N_particles, shower_type='em'):
    """
    Relative shower fluctuation σ/N.

    Parameters
    ----------
    N_particles : float
        Average number of particles at shower max
    shower_type : str
        'em' (Poisson) or 'hadronic' (larger, from π⁰ fraction)
    """
    if shower_type == 'em':
        return 1.0 / np.sqrt(N_particles)
    else:
        return 0.5 / np.sqrt(N_particles)


print("✓ lib_shower loaded:", list(MATERIALS.keys()))
