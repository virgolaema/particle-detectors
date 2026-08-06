"""
Neutron physics library: plotting, AmBe source, moderation, ToF, and ENDF data.

Complements lib.py (NuclearDatabase, capture probability) with higher-level
analysis and simulation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator

from lib import (
    NuclearDatabase,
    calculate_capture_probability,
    nuclear_db as _default_nuclear_db,
)

# ===================================================================
# DEFAULT MATERIAL DENSITIES
# ===================================================================

DEFAULT_DENSITIES = {
    'B-10': 2.34,       # g/cm³ (boron carbide)
    'Cd-113': 8.65,     # g/cm³ (metallic cadmium)
    'Gd-155': 7.90,     # g/cm³ (metallic gadolinium)
    'Gd-157': 7.90,
    'Li-6': 0.534,      # g/cm³ (metallic lithium)
    'He-3': 0.000178,   # g/cm³ (gas at STP)
    'U-235': 19.05,     # g/cm³ (metallic uranium)
    'H-1': 1.0,         # g/cm³ (water)
    'C-12': 2.267,      # g/cm³ (graphite)
    'Fe-56': 7.87,      # g/cm³ (iron)
}

# ===================================================================
# NEUTRON PLOTTING FUNCTIONS
# ===================================================================

def plot_cross_sections(materials_list, energies_kev, temperature_k=300,
                        nuclear_db=None):
    """Plot neutron capture cross-sections vs energy for selected isotopes."""
    if nuclear_db is None:
        nuclear_db = _default_nuclear_db

    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(materials_list)))

    for isotope, color in zip(materials_list, colors):
        try:
            energies_ev = energies_kev * 1000
            cross_section = nuclear_db.calculate_capture_cross_section(
                isotope, energies_ev, temperature_k)
            valid = cross_section > 0
            if np.any(valid):
                plt.loglog(energies_kev[valid], cross_section[valid],
                           label=isotope, color=color, linewidth=2.5)
        except Exception as exc:
            print(f"Warning: could not plot {isotope}: {exc}")

    plt.xlabel('Neutron Energy (keV)', fontsize=14)
    plt.ylabel('Capture Cross-Section (barns)', fontsize=14)
    plt.title('Neutron Capture Cross-Sections vs Energy', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_capture_probabilities(materials_list, energies_kev,
                               thickness_cm=1.0, temperature_k=300,
                               material_densities=None, nuclear_db=None):
    """Plot capture probabilities vs energy for different materials."""
    if nuclear_db is None:
        nuclear_db = _default_nuclear_db
    if material_densities is None:
        material_densities = DEFAULT_DENSITIES

    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(materials_list)))

    for isotope, color in zip(materials_list, colors):
        try:
            density = material_densities.get(isotope, 1.0)
            energies_ev = energies_kev * 1000
            prob, xs, macro_xs = calculate_capture_probability(
                energies_ev, isotope, nuclear_db, density,
                thickness_cm, temperature_k)
            plt.loglog(energies_kev, prob,
                       label=f'{isotope} ({thickness_cm} cm)',
                       color=color, linewidth=2.5)
        except Exception as exc:
            print(f"Warning: could not plot {isotope}: {exc}")

    plt.xlabel('Neutron Energy (keV)', fontsize=14)
    plt.ylabel('Capture Probability', fontsize=14)
    plt.title(f'Neutron Capture Probability vs Energy (thickness = {thickness_cm} cm)',
              fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(1e-6, 1)
    plt.tight_layout()
    plt.show()


def plot_thickness_dependence(isotope, energies_kev_to_plot,
                              thickness_range=None, temperature_k=300,
                              material_densities=None, nuclear_db=None):
    """Plot capture probability vs thickness for different neutron energies."""
    if nuclear_db is None:
        nuclear_db = _default_nuclear_db
    if material_densities is None:
        material_densities = DEFAULT_DENSITIES
    if thickness_range is None:
        thickness_range = np.logspace(-2, 2, 100)

    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(energies_kev_to_plot)))
    density = material_densities.get(isotope, 1.0)

    for energy_kev, color in zip(energies_kev_to_plot, colors):
        try:
            prob_vs_t = []
            energy_ev = energy_kev * 1000
            for thickness in thickness_range:
                prob, xs, macro_xs = calculate_capture_probability(
                    [energy_ev], isotope, nuclear_db, density,
                    thickness, temperature_k)
                prob_vs_t.append(prob[0])
            plt.semilogx(thickness_range, prob_vs_t,
                         label=f'{energy_kev} keV', color=color, linewidth=2.5)
        except Exception as exc:
            print(f"Warning: could not plot {energy_kev} keV: {exc}")

    plt.xlabel('Thickness (cm)', fontsize=14)
    plt.ylabel('Capture Probability', fontsize=14)
    plt.title(f'Capture Probability vs Thickness — {isotope}', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_temperature_dependence(isotope, energies_kev,
                                temperatures=None, thickness_cm=1.0,
                                material_densities=None, nuclear_db=None):
    """Plot how temperature affects capture probability."""
    if nuclear_db is None:
        nuclear_db = _default_nuclear_db
    if material_densities is None:
        material_densities = DEFAULT_DENSITIES
    if temperatures is None:
        temperatures = [77, 300, 600, 1000]

    plt.figure(figsize=(12, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(temperatures)))
    density = material_densities.get(isotope, 1.0)

    for temp_k, color in zip(temperatures, colors):
        try:
            energies_ev = energies_kev * 1000
            prob, xs, macro_xs = calculate_capture_probability(
                energies_ev, isotope, nuclear_db, density,
                thickness_cm, temp_k)
            plt.loglog(energies_kev, prob,
                       label=f'{temp_k} K', color=color, linewidth=2.5)
        except Exception as exc:
            print(f"Warning: could not plot T={temp_k} K: {exc}")

    plt.xlabel('Neutron Energy (keV)', fontsize=14)
    plt.ylabel('Capture Probability', fontsize=14)
    plt.title(f'Temperature Dependence — {isotope} ({thickness_cm} cm)', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ===================================================================
# AmBe NEUTRON SOURCE SPECTRUM
# ===================================================================
# Experimental spectrum extracted from calibrated AmBe source measurements.
# Energy in MeV, flux normalized (relative units).

_AMBE_ENERGY_MEV = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
    4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
    5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0,
    6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0,
    8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0,
    10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0,
])

_AMBE_FLUX_NORMALIZED = np.array([
    3.30, 3.20, 3.10, 3.02, 2.95, 2.85, 2.70, 2.55, 2.35, 2.15,
    1.95, 1.80, 1.70, 1.65, 1.62, 1.63, 1.68, 1.80, 1.88, 1.92,
    1.95, 1.98, 2.05, 2.15, 2.30, 2.50, 2.70, 2.90, 3.02, 3.08,
    3.05, 2.95, 2.80, 2.55, 2.35, 2.32, 2.35, 2.42, 2.50, 2.55,
    2.58, 2.60, 2.62, 2.62, 2.60, 2.55, 2.50, 2.45, 2.38, 2.30,
    2.25, 2.20, 2.10, 1.95, 1.80, 1.65, 1.55, 1.48, 1.42, 1.38,
    1.32, 1.28, 1.25, 1.20, 1.12, 1.05, 0.98, 0.92, 0.88, 0.85,
    0.75, 0.60, 0.48, 0.38, 0.32, 0.28, 0.30, 0.35, 0.42, 0.45,
    0.40, 0.32, 0.22, 0.12, 0.06, 0.02, 0.01, 0.005, 0.002, 0.001,
])


def get_ambe_spectrum(energy_mev=None, num_points=1000):
    """
    AmBe neutron spectrum from experimental data (9Be(α,n)12C).

    Parameters
    ----------
    energy_mev : array-like or None
        Energy values in MeV; if None uses default 0.1–12 MeV range
    num_points : int
        Points to generate when energy_mev is None

    Returns
    -------
    energies_mev : ndarray
    spectrum : ndarray
        Intensity normalized so that peak = 1.0
    """
    if energy_mev is None:
        energies_mev = np.linspace(0.1, 12.0, num_points)
    else:
        energies_mev = np.asarray(energy_mev, dtype=float)

    spectrum = np.interp(energies_mev, _AMBE_ENERGY_MEV, _AMBE_FLUX_NORMALIZED,
                         left=0.0, right=0.0)
    max_val = np.max(spectrum)
    if max_val > 0:
        spectrum = spectrum / max_val
    return energies_mev, spectrum


# ===================================================================
# NEUTRON MODERATION FUNCTIONS
# ===================================================================

def calculate_moderated_spectrum(initial_energies_kev, initial_spectrum,
                                 hdpe_thickness_cm=10.0):
    """
    Neutron spectrum after moderation in HDPE (simplified analytical model).

    Parameters
    ----------
    initial_energies_kev : ndarray
        Initial neutron energies (keV)
    initial_spectrum : ndarray
        Initial spectrum intensity
    hdpe_thickness_cm : float
        HDPE thickness (cm)

    Returns
    -------
    moderated_energies_kev, moderated_spectrum : ndarray
    """
    hdpe_density = 0.95      # g/cm³
    hydrogen_density = hdpe_density * 2 / 14
    carbon_density = hdpe_density * 12 / 14

    sigma_s_H = 20.0         # barns
    sigma_a_H = 0.33         # barns
    sigma_s_C = 4.8          # barns

    N_A = 6.022e23
    n_H = hydrogen_density * N_A / 1.0
    n_C = carbon_density * N_A / 12.0

    sigma_total = n_H * (sigma_s_H + sigma_a_H) * 1e-24 + n_C * sigma_s_C * 1e-24
    mean_free_path = 1.0 / sigma_total
    n_collisions = hdpe_thickness_cm / mean_free_path

    avg_energy_loss = 0.8 * 0.5 + 0.2 * (2 * 12 / (12 + 1) ** 2)

    moderated_energies_kev = np.logspace(-5, 5, 1000)
    moderated_spectrum = np.zeros_like(moderated_energies_kev)

    for i, E_initial_kev in enumerate(initial_energies_kev):
        if initial_spectrum[i] > 0.01:
            survival_prob = np.exp(-n_collisions * sigma_a_H / sigma_s_H * 0.1)

            if E_initial_kev > 100:
                E_final_avg_kev = max(
                    E_initial_kev * (1 - avg_energy_loss) ** n_collisions, 1e-5)
                sigma_mod = E_final_avg_kev * 0.5
                moderated_spectrum += (
                    survival_prob * initial_spectrum[i]
                    * np.exp(-0.5 * ((moderated_energies_kev - E_final_avg_kev)
                                     / sigma_mod) ** 2))
            else:
                moderated_spectrum += (
                    survival_prob * 0.7 * initial_spectrum[i]
                    * np.exp(-0.5 * ((moderated_energies_kev - E_initial_kev)
                                     / (0.1 * E_initial_kev)) ** 2))

    thermal_peak = 0.3 * np.exp(
        -0.5 * ((moderated_energies_kev - 0.000025) / 0.000015) ** 2)
    moderated_spectrum += thermal_peak

    peak = np.max(moderated_spectrum)
    if peak > 0:
        moderated_spectrum /= peak

    return moderated_energies_kev, moderated_spectrum


def calculate_flux_fractions(energies_kev, spectrum):
    """
    Fraction of neutron flux in each energy region.

    Returns a dict with keys 'cold', 'thermal', 'epithermal', 'fast'.
    """
    total_flux = np.trapezoid(spectrum, energies_kev)
    if total_flux == 0:
        return {'cold': 0.0, 'thermal': 0.0, 'epithermal': 0.0, 'fast': 1.0}

    cold_max = 0.0001       # keV (0.1 eV)
    thermal_max = 0.1       # keV (100 eV)
    epithermal_max = 500    # keV (0.5 MeV)

    def _integrate(mask):
        if not np.any(mask):
            return 0.0
        return np.trapezoid(spectrum[mask], energies_kev[mask]) / total_flux

    return {
        'cold': _integrate(energies_kev < cold_max),
        'thermal': _integrate((energies_kev >= cold_max) & (energies_kev < thermal_max)),
        'epithermal': _integrate((energies_kev >= thermal_max) & (energies_kev < epithermal_max)),
        'fast': _integrate(energies_kev >= epithermal_max),
    }


def calculate_air_moderated_spectrum(initial_energies_kev, initial_spectrum,
                                     air_thickness_cm=10.0):
    """
    Neutron spectrum after passing through an air gap (minimal moderation).

    Parameters
    ----------
    initial_energies_kev : ndarray
        Input neutron energies (keV), typically from HDPE moderation output
    initial_spectrum : ndarray
        Input spectrum intensity
    air_thickness_cm : float
        Air gap thickness (cm)

    Returns
    -------
    air_moderated_energies_kev, air_moderated_spectrum : ndarray
    """
    air_density = 0.00125       # g/cm³

    nitrogen_fraction = 0.755
    oxygen_fraction = 0.232
    argon_fraction = 0.013

    sigma_s_N, sigma_a_N = 11.5, 1.9
    sigma_s_O, sigma_a_O = 4.2, 0.00019
    sigma_s_Ar, sigma_a_Ar = 0.66, 0.675

    N_A = 6.022e23
    n_N = air_density * nitrogen_fraction * N_A / 14.0
    n_O = air_density * oxygen_fraction * N_A / 16.0
    n_Ar = air_density * argon_fraction * N_A / 40.0

    sigma_total = (
        n_N * (sigma_s_N + sigma_a_N)
        + n_O * (sigma_s_O + sigma_a_O)
        + n_Ar * (sigma_s_Ar + sigma_a_Ar)
    ) * 1e-24

    mean_free_path = 1.0 / sigma_total if sigma_total > 0 else 1e6
    n_collisions = air_thickness_cm / mean_free_path

    mass_weighted_loss = (
        nitrogen_fraction * 2 * 14 / (14 + 1) ** 2
        + oxygen_fraction * 2 * 16 / (16 + 1) ** 2
        + argon_fraction * 2 * 40 / (40 + 1) ** 2
    )

    air_moderated_energies_kev = initial_energies_kev.copy()
    air_moderated_spectrum = np.zeros_like(air_moderated_energies_kev)

    sigma_abs = (n_N * sigma_a_N + n_O * sigma_a_O + n_Ar * sigma_a_Ar) * 1e-24
    absorption_prob = 1 - np.exp(-sigma_abs * air_thickness_cm)
    survival_prob = 1 - absorption_prob

    for i, E_initial_kev in enumerate(initial_energies_kev):
        if initial_spectrum[i] > 0.001:
            if n_collisions > 0.01:
                E_final_avg_kev = E_initial_kev * (1 - mass_weighted_loss) ** n_collisions
            else:
                E_final_avg_kev = E_initial_kev

            energy_idx = np.argmin(np.abs(air_moderated_energies_kev - E_final_avg_kev))

            if E_initial_kev > 1.0:
                air_moderated_spectrum[energy_idx] += (
                    survival_prob * initial_spectrum[i] * 0.95)
                if E_final_avg_kev != E_initial_kev:
                    final_idx = np.argmin(np.abs(air_moderated_energies_kev - E_final_avg_kev))
                    air_moderated_spectrum[final_idx] += (
                        survival_prob * initial_spectrum[i] * 0.05)
            else:
                air_moderated_spectrum[energy_idx] += survival_prob * initial_spectrum[i]

    output_total = np.trapezoid(air_moderated_spectrum, air_moderated_energies_kev)
    input_total = np.trapezoid(initial_spectrum, initial_energies_kev)
    if output_total > 0:
        air_moderated_spectrum *= (input_total * 0.98) / output_total

    return air_moderated_energies_kev, air_moderated_spectrum


def calculate_region_fractions(energies_kev, spectrum_norm):
    """
    Fraction of flux in cold / thermal / epithermal / fast regions.

    Uses slightly different boundaries than calculate_flux_fractions for
    consistency with the region-comparison plots.

    Returns a list [cold, thermal, epithermal, fast].
    """
    cold_cutoff = 0.00001       # keV (0.01 eV)
    epithermal_cutoff = 0.1     # keV (100 eV)
    thermal_cutoff = 500        # keV (0.5 MeV)

    def _integrate(mask):
        if not np.any(mask):
            return 0.0
        return np.trapezoid(spectrum_norm[mask], energies_kev[mask])

    cold = _integrate(energies_kev < cold_cutoff)
    thermal = _integrate((energies_kev >= cold_cutoff) & (energies_kev < epithermal_cutoff))
    epithermal = _integrate((energies_kev >= epithermal_cutoff) & (energies_kev < thermal_cutoff))
    fast = _integrate(energies_kev >= thermal_cutoff)

    total = cold + thermal + epithermal + fast
    if total > 0:
        return [cold / total, thermal / total, epithermal / total, fast / total]
    return [0, 0, 0, 1]


# ===================================================================
# TIME OF FLIGHT (ToF) FUNCTIONS
# ===================================================================

# Physical constants
_C_LIGHT = 2.998e10   # cm/s
_M_NEUTRON = 939.565e6  # eV/c²


def neutron_velocity(E_eV):
    """
    Non-relativistic neutron velocity from kinetic energy.

    v = c × √(2E / m_n c²)  [cm/s]
    Valid for E << 939 MeV.
    """
    return _C_LIGHT * np.sqrt(2 * np.asarray(E_eV, dtype=float) / _M_NEUTRON)


def tof_constant_energy(E_eV, distance_cm):
    """ToF assuming constant energy (no moderation): t = d / v  [seconds]."""
    return distance_cm / neutron_velocity(E_eV)


def tof_with_moderation(E_initial_eV, hdpe_thickness_cm, air_gap_cm=10.0,
                        mean_free_path_cm=0.5, energy_loss_per_collision=0.5):
    """
    Estimate ToF including energy loss during moderation in HDPE.

    Parameters
    ----------
    E_initial_eV : float
        Initial neutron energy (eV)
    hdpe_thickness_cm : float
        HDPE moderator thickness (cm)
    air_gap_cm : float
        Air gap after moderator (cm)
    mean_free_path_cm : float
        Mean free path in HDPE (cm)
    energy_loss_per_collision : float
        Average fractional energy loss per collision

    Returns
    -------
    tof_total : float
        Total time of flight in seconds
    tof_hdpe : float
        Time spent in HDPE
    tof_air : float
        Time spent in air gap
    E_final_eV : float
        Energy after moderation
    """
    n_collisions = hdpe_thickness_cm / mean_free_path_cm
    path_length_hdpe = hdpe_thickness_cm * 1.5  # random walk factor

    tof_hdpe = 0.0
    E_current = E_initial_eV
    step_length = mean_free_path_cm * 1.5

    for _ in range(int(n_collisions)):
        v = neutron_velocity(E_current)
        tof_hdpe += step_length / v
        E_current *= (1 - energy_loss_per_collision)
        E_current = max(E_current, 0.001)  # don't go below 1 meV

    E_final_eV = E_current
    v_final = neutron_velocity(E_final_eV)
    tof_air = air_gap_cm / v_final

    return tof_hdpe + tof_air, tof_hdpe, tof_air, E_final_eV


# ===================================================================
# ENDF-BASED CROSS-SECTION DATA FOR HDPE
# ===================================================================
# Sampled from ENDF/B-VIII.0 (NNDC). Log-log interpolation gives smooth
# curves without artificial discontinuities.
# ⚠️ Fine resonance structure is smoothed over; accuracy ~10-20%.

ENDF_H1_DATA = np.array([
    [1.0e-3,  20.50], [1.0e-2,  20.50], [1.0e-1,  20.50], [1.0,    20.47],
    [10.0,    20.44], [100.0,   20.35], [1.0e3,   20.10], [1.0e4,  19.30],
    [5.0e4,   15.90], [1.0e5,   12.90], [2.0e5,    9.40], [5.0e5,   6.24],
    [1.0e6,    4.26], [2.0e6,    2.87], [2.45e6,   2.56], [3.0e6,   2.30],
    [5.0e6,    1.61], [8.0e6,    1.14], [1.0e7,    0.95], [1.2e7,   0.83],
])

# Bound-atom enhancement for H chemically bound in a CH2 / PVT lattice.
# As E -> 0 the neutron sees the whole molecule, so sigma -> ((A+1)/A)^2 = 4x
# the free-atom value (thermal: 4 x 20.5 b = 82 b, the standard value for H in
# polyethylene).  The enhancement dies away once E exceeds molecular binding
# energies (~1 eV), above which the free-atom cross section applies.
H_BOUND_MULTIPLIER = np.array([
    [1.0e-3, 4.30], [1.0e-2, 4.15], [2.53e-2, 4.00], [1.0e-1, 2.70],
    [1.0,    1.45], [5.0,    1.10], [2.0e1,   1.00], [1.2e7,  1.00],
])

ENDF_C12_DATA = np.array([
    [1.0e-3,   4.75], [1.0e-2,   4.75], [1.0e-1,  4.75], [1.0,    4.75],
    [10.0,     4.75], [100.0,    4.75], [1.0e3,   4.75], [1.0e4,  4.74],
    [5.0e4,    4.69], [1.0e5,    4.55], [2.0e5,   4.20], [5.0e5,  3.20],
    [1.0e6,    2.61], [2.0e6,    2.20], [2.45e6,  2.07], [3.0e6,  1.95],
    [5.0e6,    1.65], [8.0e6,    1.42], [1.0e7,   1.35], [1.2e7,  1.30],
])

_H_INTERP = PchipInterpolator(np.log10(ENDF_H1_DATA[:, 0]),
                              np.log10(ENDF_H1_DATA[:, 1]))
_C_INTERP = PchipInterpolator(np.log10(ENDF_C12_DATA[:, 0]),
                              np.log10(ENDF_C12_DATA[:, 1]))
_H_BOUND_INTERP = PchipInterpolator(np.log10(H_BOUND_MULTIPLIER[:, 0]),
                                    H_BOUND_MULTIPLIER[:, 1])


def get_endf_cross_sections(E_eV, bound_H=True):
    """
    H-1 and C-12 elastic scattering cross-sections from ENDF/B-VIII.0.

    Interpolation is PCHIP (shape-preserving) in log-log space.  A cubic
    spline is *not* used here: on a sparse knot set it rings, and the previous
    version of this function undershot to 5.2 b at 1 eV where the true H
    elastic cross section is ~20.5 b.

    Parameters
    ----------
    E_eV : float or array-like
        Neutron energy in eV.
    bound_H : bool, default True
        Apply the bound-atom enhancement to hydrogen (see
        H_BOUND_MULTIPLIER).  Correct for H bound in a molecular solid --
        polyethylene, PVT scintillator, wood -- which is every material in
        this package.  Pass False for free/gaseous hydrogen.

    Returns
    -------
    sigma_H, sigma_C : ndarray
        Elastic scattering cross-sections in barns.
    """
    E_eV = np.atleast_1d(np.asarray(E_eV, dtype=float))
    log_E = np.log10(np.clip(E_eV, 1.0e-3, 1.2e7))

    sigma_H = 10 ** _H_INTERP(log_E)
    if bound_H:
        sigma_H = sigma_H * _H_BOUND_INTERP(log_E)
    sigma_C = 10 ** _C_INTERP(log_E)
    return sigma_H, sigma_C


def mean_free_path_hdpe_endf(E_eV):
    """
    Mean free path in HDPE using ENDF cross-sections.

    λ = 1 / (n_H σ_H + n_C σ_C)

    Parameters
    ----------
    E_eV : float or array-like
        Neutron energy in eV

    Returns
    -------
    lambda_mfp : ndarray  [cm]
    n_H, n_C   : float    [nuclei/cm³]
    sigma_H, sigma_C : ndarray  [barns]
    """
    rho_hdpe = 0.95     # g/cm³
    N_A = 6.022e23
    M_CH2 = 14.0        # g/mol

    n_CH2 = (rho_hdpe / M_CH2) * N_A
    n_H = 2 * n_CH2
    n_C = 1 * n_CH2

    sigma_H, sigma_C = get_endf_cross_sections(E_eV)
    Sigma = n_H * sigma_H * 1e-24 + n_C * sigma_C * 1e-24
    lambda_mfp = 1.0 / Sigma

    return lambda_mfp, n_H, n_C, sigma_H, sigma_C


def get_mean_free_path(E_eV):
    """
    Mean free path in HDPE as a scalar (convenience wrapper for MC loops).

    Parameters
    ----------
    E_eV : float
        Neutron energy in eV

    Returns
    -------
    float : mean free path in cm
    """
    mfp, _, _, _, _ = mean_free_path_hdpe_endf(E_eV)
    return float(np.atleast_1d(mfp)[0])


def calculate_energy_fractions(energies_eV):
    """
    Fraction of neutrons (by count) in each energy region.

    Works on an array of individual neutron energies (e.g. from MC output).
    Returns a dict with 'cold', 'thermal', 'epithermal', 'fast',
    'mean_eV', and 'median_eV'.
    """
    energies_eV = np.asarray(energies_eV, dtype=float)
    total = len(energies_eV)
    if total == 0:
        return {'cold': 0, 'thermal': 0, 'epithermal': 0, 'fast': 1,
                'mean_eV': 0, 'median_eV': 0}

    cold = np.sum(energies_eV < 0.1)
    thermal = np.sum((energies_eV >= 0.1) & (energies_eV < 100))
    epithermal = np.sum((energies_eV >= 100) & (energies_eV < 500e3))
    fast = np.sum(energies_eV >= 500e3)

    return {
        'cold': cold / total,
        'thermal': thermal / total,
        'epithermal': epithermal / total,
        'fast': fast / total,
        'mean_eV': np.mean(energies_eV),
        'median_eV': np.median(energies_eV),
    }


print("✓ lib_neutrons loaded: plotting, AmBe spectrum, moderation, ToF, ENDF data")
