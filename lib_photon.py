"""Photon attenuation library using NIST XCOM evaluated data."""

import numpy as np

# ===================================================================
# NIST XCOM DATA TABLES
# ===================================================================
# Mass attenuation coefficients μ/ρ in cm²/g
# Format: [Energy(MeV), Total, Photoelectric, Compton, Pair]
# Source: https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html

NIST_POLYETHYLENE = np.array([
    [0.01,    4.071,    0.0448,   3.991,    0.0],
    [0.015,   1.696,    0.0131,   1.673,    0.0],
    [0.02,    0.868,    0.00504,  0.860,    0.0],
    [0.03,    0.346,    0.00126,  0.344,    0.0],
    [0.04,    0.206,    0.000473, 0.205,    0.0],
    [0.05,    0.149,    0.000238, 0.149,    0.0],
    [0.06,    0.119,    0.000137, 0.119,    0.0],
    [0.08,    0.0850,   5.66e-05, 0.0849,   0.0],
    [0.10,    0.0685,   2.90e-05, 0.0684,   0.0],
    [0.15,    0.0488,   8.47e-06, 0.0487,   0.0],
    [0.20,    0.0396,   3.81e-06, 0.0396,   0.0],
    [0.30,    0.0300,   1.15e-06, 0.0300,   0.0],
    [0.40,    0.0254,   5.16e-07, 0.0254,   0.0],
    [0.50,    0.0227,   2.67e-07, 0.0227,   0.0],
    [0.60,    0.0211,   1.59e-07, 0.0210,   0.0],
    [0.80,    0.0194,   7.13e-08, 0.0193,   6e-06],
    [1.00,    0.0186,   3.78e-08, 0.0182,   2.8e-05],
    [1.50,    0.0175,   1.14e-08, 0.0165,   9.5e-05],
    [2.00,    0.0173,   4.99e-09, 0.0154,   1.86e-04],
    [3.00,    0.0180,   1.51e-09, 0.0139,   4.02e-04],
    [4.00,    0.0193,   6.42e-10, 0.0130,   6.27e-04],
    [5.00,    0.0209,   3.32e-10, 0.0123,   8.50e-04],
    [6.00,    0.0227,   1.94e-10, 0.0117,   1.066e-03],
    [8.00,    0.0267,   8.29e-11, 0.0108,   1.486e-03],
    [10.00,   0.0309,   4.13e-11, 0.0102,   1.891e-03],
    [15.00,   0.0421,   1.25e-11, 0.00907,  2.856e-03],
    [20.00,   0.0541,   5.47e-12, 0.00837,  3.775e-03],
])

NIST_CARBON = np.array([
    [0.01,    8.779,    0.226,    8.473,    0.0],
    [0.015,   3.315,    0.0631,   3.222,    0.0],
    [0.02,    1.602,    0.0230,   1.568,    0.0],
    [0.03,    0.588,    0.00565,  0.579,    0.0],
    [0.04,    0.331,    0.00218,  0.327,    0.0],
    [0.05,    0.230,    0.00109,  0.228,    0.0],
    [0.06,    0.178,    0.000622, 0.177,    0.0],
    [0.08,    0.124,    0.000258, 0.124,    0.0],
    [0.10,    0.0989,   0.000132, 0.0987,   0.0],
    [0.15,    0.0695,   3.86e-05, 0.0694,   0.0],
    [0.20,    0.0562,   1.73e-05, 0.0562,   0.0],
    [0.30,    0.0424,   5.21e-06, 0.0424,   0.0],
    [0.40,    0.0358,   2.34e-06, 0.0358,   0.0],
    [0.50,    0.0320,   1.21e-06, 0.0320,   0.0],
    [0.60,    0.0296,   7.22e-07, 0.0296,   0.0],
    [0.80,    0.0270,   3.24e-07, 0.0270,   1e-05],
    [1.00,    0.0257,   1.72e-07, 0.0255,   5e-05],
    [1.50,    0.0243,   5.17e-08, 0.0238,   1.7e-04],
    [2.00,    0.0241,   2.27e-08, 0.0222,   3.3e-04],
    [3.00,    0.0251,   6.84e-09, 0.0200,   7.2e-04],
    [4.00,    0.0270,   2.91e-09, 0.0186,   1.13e-03],
    [5.00,    0.0292,   1.51e-09, 0.0175,   1.53e-03],
    [6.00,    0.0317,   8.80e-10, 0.0167,   1.92e-03],
    [8.00,    0.0374,   3.76e-10, 0.0154,   2.67e-03],
    [10.00,   0.0435,   1.87e-10, 0.0145,   3.40e-03],
    [15.00,   0.0593,   5.65e-11, 0.0130,   5.14e-03],
    [20.00,   0.0761,   2.48e-11, 0.0119,   6.79e-03],
])

NIST_LEAD = np.array([
    [0.01,    136.5,    133.4,    3.045,    0.0],
    [0.015,   66.10,    63.05,    3.011,    0.0],
    [0.02,    33.60,    30.66,    2.904,    0.0],
    [0.03,    11.47,    8.828,    2.610,    0.0],
    [0.04,    5.549,    3.424,    2.097,    0.0],
    [0.05,    3.582,    1.768,    1.792,    0.0],
    [0.06,    2.735,    1.120,    1.598,    0.0],
    [0.08,    1.927,    0.549,    1.363,    0.0],
    [0.10,    1.568,    0.305,    1.248,    0.0],
    [0.15,    0.999,    0.0885,   0.898,    0.0],
    [0.20,    0.719,    0.0371,   0.672,    0.0],
    [0.30,    0.435,    0.0108,   0.420,    0.00002],
    [0.40,    0.323,    0.00457,  0.315,    0.00032],
    [0.50,    0.266,    0.00238,  0.258,    0.00127],
    [0.60,    0.235,    0.00142,  0.227,    0.00301],
    [0.80,    0.206,    6.35e-04, 0.196,    0.00793],
    [1.00,    0.196,    3.36e-04, 0.180,    0.0141],
    [1.50,    0.194,    1.01e-04, 0.162,    0.0302],
    [2.00,    0.209,    4.43e-05, 0.152,    0.0473],
    [3.00,    0.251,    1.33e-05, 0.141,    0.0831],
    [4.00,    0.299,    5.67e-06, 0.134,    0.1189],
    [5.00,    0.350,    2.93e-06, 0.129,    0.1541],
    [6.00,    0.401,    1.71e-06, 0.126,    0.1886],
    [8.00,    0.506,    7.30e-07, 0.121,    0.2564],
    [10.00,   0.613,    3.63e-07, 0.118,    0.3228],
    [15.00,   0.905,    1.10e-07, 0.113,    0.4902],
    [20.00,   1.209,    4.81e-08, 0.110,    0.6521],
])

# ===================================================================
# MATERIAL DATABASE
# ===================================================================

MATERIALS = {
    'hdpe': {
        'name': 'Polyethylene (HDPE)',
        'formula': 'CH₂',
        'density': 0.94,       # g/cm³
        'composition': 'H: 14.37%, C: 85.63% by weight',
        'data': NIST_POLYETHYLENE,
        'color': '#2E86AB',
    },
    'carbon': {
        'name': 'Carbon (Graphite)',
        'formula': 'C',
        'density': 2.0,        # g/cm³
        'composition': 'Pure carbon',
        'data': NIST_CARBON,
        'color': '#A23B72',
    },
    'lead': {
        'name': 'Lead',
        'formula': 'Pb',
        'density': 11.35,      # g/cm³
        'composition': 'Z=82',
        'data': NIST_LEAD,
        'color': '#F18F01',
    },
}

# ===================================================================
# CALCULATION FUNCTIONS
# ===================================================================

def calculate_attenuation_coefficients(material_key, energy_mev):
    """
    Calculate photon attenuation coefficients using NIST XCOM data.

    Parameters
    ----------
    material_key : str
        Material key: 'hdpe', 'carbon', or 'lead'
    energy_mev : float or array-like
        Photon energy in MeV

    Returns
    -------
    mu_total, mu_photo, mu_compton, mu_pair : float or ndarray
        Linear attenuation coefficients in cm⁻¹
    """
    mat = MATERIALS[material_key]
    data = mat['data']
    rho = mat['density']

    E_data = data[:, 0]
    mu_rho_total = data[:, 1]
    mu_rho_photo = data[:, 2]
    mu_rho_compton = data[:, 3]
    mu_rho_pair = data[:, 4]

    energy_arr = np.atleast_1d(np.asarray(energy_mev, dtype=float))

    mu_total = np.interp(energy_arr, E_data, mu_rho_total) * rho
    mu_photo = np.interp(energy_arr, E_data, mu_rho_photo) * rho
    mu_compton = np.interp(energy_arr, E_data, mu_rho_compton) * rho
    mu_pair = np.interp(energy_arr, E_data, mu_rho_pair) * rho

    if np.isscalar(energy_mev):
        return mu_total[0], mu_photo[0], mu_compton[0], mu_pair[0]
    return mu_total, mu_photo, mu_compton, mu_pair


def photon_transmission(material_key, energy_mev, thickness_cm):
    """
    Calculate photon transmission through a slab.

    Returns
    -------
    transmission : float or ndarray
        Fraction of photons transmitted (Beer-Lambert law)
    """
    mu_total, _, _, _ = calculate_attenuation_coefficients(material_key, energy_mev)
    return np.exp(-mu_total * thickness_cm)


print("✓ lib_photon loaded: NIST XCOM data for", list(MATERIALS.keys()))
