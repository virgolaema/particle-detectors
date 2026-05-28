"""Alpha particle stopping power and quenching helpers."""

import numpy as np

# Physical constants
M_ALPHA_MEV = 3727.379  # MeV/c^2 (alpha mass ~ 4 amu)
M_E_MEV = 0.511  # MeV/c^2 (electron mass)

# Material database for alpha calculations
MATERIALS = {
    "plastic_scintillator": {
        "name": "Plastic Scintillator (BC-408)",
        "formula": "C9H10 (typical)",
        "Z": 5.74,
        "A": 11.2,
        "density": 1.032,
        "I": 64.7,
        "kB": 0.0004,
        "color": "#2E86AB",
        "description": "Organic scintillator with quenching",
    },
    "air": {
        "name": "Air",
        "formula": "N2+O2",
        "Z": 7.22,
        "A": 14.4,
        "density": 0.001205,
        "I": 85.7,
        "kB": 0.0,
        "color": "#A0C4E8",
        "description": "Standard air at STP",
    },
    "silicon": {
        "name": "Silicon",
        "formula": "Si",
        "Z": 14,
        "A": 28.09,
        "density": 2.33,
        "I": 173.0,
        "kB": 0.0,
        "color": "#F18F01",
        "description": "Silicon detector",
    },
    "water": {
        "name": "Water",
        "formula": "H2O",
        "Z": 7.42,
        "A": 18.0,
        "density": 1.0,
        "I": 75.0,
        "kB": 0.0,
        "color": "#66B3FF",
        "description": "Liquid water",
    },
    "aluminum": {
        "name": "Aluminum",
        "formula": "Al",
        "Z": 13,
        "A": 26.98,
        "density": 2.70,
        "I": 166.0,
        "kB": 0.0,
        "color": "#C0C0C0",
        "description": "Aluminum metal",
    },
}


def bethe_bloch_stopping_power(E_MeV, Z_particle, material_key):
    """
    Calculate stopping power dE/dx for heavy particles.

    Uses semi-empirical formula calibrated to NIST PSTAR data.
    For alphas (Z=2) at 1-10 MeV in plastics/organics.
    """
    mat = MATERIALS[material_key]

    # Ensure array
    E = np.atleast_1d(E_MeV)

    # Calculate beta and gamma
    gamma = 1.0 + E / M_ALPHA_MEV
    beta = np.sqrt(1.0 - 1.0 / (gamma**2))
    beta = np.maximum(beta, 1e-4)

    # Material properties
    Z_mat = mat["Z"]
    A_mat = mat["A"]
    rho = mat["density"]
    I = mat["I"] * 1e-6  # Convert eV to MeV

    # Bethe-Bloch constant
    K = 0.307075  # MeV*cm^2*g^-1

    beta2 = beta**2
    gamma2 = gamma**2

    # Maximum energy transfer
    T_max = 2 * M_E_MEV * beta2 * gamma2

    # Logarithmic term
    ln_term = 0.5 * np.log(2 * M_E_MEV * beta2 * gamma2 * T_max / (I**2))

    # Bethe-Bloch formula
    dEdx_base = K * (Z_particle**2) * (Z_mat / A_mat) * rho * (ln_term - beta2) / beta2

    # Empirical scaling factor to match NIST PSTAR data
    scale_factor = 0.2
    dEdx = dEdx_base * scale_factor

    # Return scalar if input was scalar
    if np.isscalar(E_MeV):
        return dEdx[0]

    return dEdx


def calculate_range(E_initial_MeV, Z_particle, material_key, num_steps=1000):
    """Calculate range by integrating stopping power: R = integral dE/(dE/dx)."""
    energies = np.linspace(E_initial_MeV, 0.01, num_steps)

    dEdx_arr = bethe_bloch_stopping_power(energies, Z_particle, material_key)
    dEdx_arr = np.maximum(dEdx_arr, 1e-10)

    dE = np.abs(np.diff(energies))
    dx = dE / dEdx_arr[:-1]

    depths = np.concatenate([[0], np.cumsum(dx)])
    total_range = depths[-1]

    return total_range, energies, depths


def calculate_visible_energy_birks(E_initial_MeV, material_key, num_steps=1000):
    """Calculate visible energy in scintillator accounting for Birks quenching."""
    mat = MATERIALS[material_key]
    kB = mat["kB"]
    rho = mat["density"]

    total_range, energies, depths = calculate_range(
        E_initial_MeV,
        Z_particle=2,
        material_key=material_key,
        num_steps=num_steps,
    )

    dEdx_arr = bethe_bloch_stopping_power(energies, Z_particle=2, material_key=material_key)

    dE = np.abs(np.diff(energies))

    dEdx_mass = dEdx_arr[:-1] / rho  # MeV/(g/cm^2)
    birks_factor = 1.0 / (1.0 + kB * dEdx_mass)
    dE_visible = dE * birks_factor

    E_visible = np.sum(dE_visible)
    E_deposited = E_initial_MeV

    quenching_factor = E_visible / E_deposited if E_deposited > 0 else 0

    return E_visible, E_deposited, quenching_factor, energies, depths, dEdx_arr
