"""
Particle Detectors Physics Library
=================================

This module contains all the heavy physics calculations, material databases,
and nuclear data for particle detector analysis.

Classes:
- MaterialDatabase: Bethe-Bloch material properties and element lookup
- NuclearDatabase: Neutron capture cross-sections and nuclear data

Functions:
- Bethe-Bloch energy loss calculations
- Neutron capture probability calculations
- Photon attenuation calculations
"""

import numpy as np
from scipy.constants import N_A, pi
import warnings
warnings.filterwarnings('ignore')

# Define physical constants with fallbacks
try:
    # Test if scipy constants are available
    test_na = N_A
except:
    # Fallback definitions
    N_A = 6.02214076e23  # mol⁻¹
    pi = 3.141592653589793

# Try to import mendeleev for automatic element lookup
try:
    from mendeleev import element
    HAS_MENDELEEV = True
except ImportError:
    HAS_MENDELEEV = False

# ===================================================================
# MATERIAL DATABASE FOR BETHE-BLOCH CALCULATIONS
# ===================================================================

class MaterialDatabase:
    """Database of material properties for Bethe-Bloch calculations."""

    def __init__(self):
        # Manual database for compounds and special materials
        # Format: {name: (Z_eff, A_eff, I_eV, density_g_cm3)}
        self.compound_database = {
            'Water': (10, 18.015, 75.0, 1.0),
            'Air': (7.22, 14.4, 85.7, 0.001225),
            'Scintillator': (5.57, 11.9, 64.7, 1.032),  # Organic scintillator
            'PMMA': (6.56, 13.25, 74.0, 1.19),
            'Polyethylene': (5.37, 11.17, 57.4, 0.94),
            'Steel': (26, 55.8, 286.0, 7.87),  # Approximate
        }

        # Mean excitation energies for elements (eV)
        self.mean_excitation_energies = {
            'H': 19.2, 'He': 41.8, 'Li': 40.0, 'Be': 63.7, 'B': 76.0, 'C': 78.0,
            'N': 82.0, 'O': 95.0, 'F': 115.0, 'Ne': 137.0, 'Na': 149.0, 'Mg': 156.0,
            'Al': 166.0, 'Si': 173.0, 'P': 173.0, 'S': 180.0, 'Cl': 174.0, 'Ar': 188.0,
            'K': 190.0, 'Ca': 191.0, 'Sc': 216.0, 'Ti': 233.0, 'V': 245.0, 'Cr': 257.0,
            'Mn': 272.0, 'Fe': 286.0, 'Co': 297.0, 'Ni': 311.0, 'Cu': 322.0, 'Zn': 330.0,
            'Ga': 334.0, 'Ge': 350.0, 'As': 347.0, 'Se': 348.0, 'Br': 343.0, 'Kr': 352.0,
            'Rb': 363.0, 'Sr': 366.0, 'Y': 379.0, 'Zr': 393.0, 'Nb': 417.0, 'Mo': 424.0,
            'Tc': 428.0, 'Ru': 441.0, 'Rh': 449.0, 'Pd': 470.0, 'Ag': 470.0, 'Cd': 469.0,
            'In': 488.0, 'Sn': 488.0, 'Sb': 487.0, 'Te': 485.0, 'I': 491.0, 'Xe': 482.0,
            'Cs': 488.0, 'Ba': 491.0, 'La': 501.0, 'Ce': 523.0, 'Pr': 535.0, 'Nd': 546.0,
            'Pm': 560.0, 'Sm': 574.0, 'Eu': 580.0, 'Gd': 591.0, 'Tb': 614.0, 'Dy': 628.0,
            'Ho': 650.0, 'Er': 658.0, 'Tm': 674.0, 'Yb': 684.0, 'Lu': 694.0, 'Hf': 705.0,
            'Ta': 718.0, 'W': 727.0, 'Re': 736.0, 'Os': 746.0, 'Ir': 757.0, 'Pt': 790.0,
            'Au': 790.0, 'Hg': 800.0, 'Tl': 810.0, 'Pb': 823.0, 'Bi': 823.0, 'Po': 830.0,
            'At': 825.0, 'Rn': 794.0, 'Fr': 827.0, 'Ra': 826.0, 'Ac': 841.0, 'Th': 847.0,
            'Pa': 878.0, 'U': 890.0
        }

    def get_element_properties(self, symbol):
        """Get properties for a chemical element using mendeleev or built-in data."""
        if HAS_MENDELEEV:
            try:
                elem = element(symbol)
                Z = elem.atomic_number
                A = elem.atomic_weight

                # Try to get density (solid at STP)
                density = getattr(elem, 'density', None)
                if density is None:
                    # Fallback densities for common elements
                    density_fallback = {
                        'H': 0.0000899, 'He': 0.0001785, 'C': 2.267, 'N': 0.001251,
                        'O': 0.001429, 'Al': 2.702, 'Si': 2.329, 'Ar': 0.0017837,
                        'Fe': 7.874, 'Cu': 8.96, 'Ge': 5.323, 'Pb': 11.342
                    }
                    density = density_fallback.get(symbol, 1.0)  # Default to 1 g/cm³

                # Get mean excitation energy
                I = self.mean_excitation_energies.get(symbol, 10.0 * Z)  # Rough approximation

                return Z, A, I, density

            except Exception as e:
                print(f"Warning: Could not lookup {symbol} with mendeleev: {e}")

        # Fallback to manual lookup for common elements
        manual_data = {
            'H': (1, 1.008, 19.2, 0.0000899),
            'He': (2, 4.003, 41.8, 0.0001785),
            'C': (6, 12.011, 78.0, 2.267),
            'Al': (13, 26.982, 166.0, 2.702),
            'Si': (14, 28.085, 173.0, 2.329),
            'Ar': (18, 39.948, 188.0, 0.0017837),
            'Fe': (26, 55.845, 286.0, 7.874),
            'Cu': (29, 63.546, 322.0, 8.96),
            'Ge': (32, 72.630, 350.0, 5.323),
            'Pb': (82, 207.2, 823.0, 11.342)
        }

        if symbol in manual_data:
            return manual_data[symbol]
        else:
            raise ValueError(f"Element {symbol} not found in database")

    def get_material_properties(self, material_name):
        """Get material properties (Z, A, I, density) for any material."""
        # First check if it's a compound/special material
        if material_name in self.compound_database:
            return self.compound_database[material_name]

        # Otherwise assume it's an element symbol
        return self.get_element_properties(material_name)

    def list_available_materials(self):
        """List all available materials."""
        elements = list(self.mean_excitation_energies.keys())[:10]  # Show first 10
        compounds = list(self.compound_database.keys())

        print("Available elements (first 10):", elements + ["... and more"])
        print("Available compounds:", compounds)

        if HAS_MENDELEEV:
            print("⚠ Full periodic table available via mendeleev library")

# ===================================================================
# NUCLEAR DATABASE FOR NEUTRON CALCULATIONS
# ===================================================================

class NuclearDatabase:
    """Database of neutron capture cross-sections and nuclear properties."""

    def __init__(self):
        # Nuclear data: {isotope: (mass, thermal_xs, abundance, resonances, fast_xs)}
        # mass: atomic mass (amu)
        # thermal_xs: thermal neutron capture cross-section (barns)
        # abundance: isotopic abundance (fraction)
        # resonances: [(E_res, Γ_n, Γ_γ, J), ...] - resonance parameters
        # fast_xs: fast neutron cross-section (barns)

        self.nuclear_data = {
            # Isotope: (mass, σ_thermal, abundance, resonances, σ_fast)
            'H-1': (1.008, 0.332, 0.99985, [], 0.0003),
            'He-3': (3.016, 5333, 0.000137, [], 0.0001),
            'Li-6': (6.015, 940, 0.0759, [], 0.0009),
            'B-10': (10.013, 3835, 0.199, [(0.0025, 0.0001, 0.0024, 2)], 0.0005),
            'Cd-113': (112.904, 20600, 0.122, [(0.178, 0.00011, 0.116, 1)], 0.001),
            'Gd-155': (154.923, 60900, 0.148, [(0.0268, 0.00004, 0.108, 2)], 0.002),
            'Gd-157': (156.924, 254000, 0.157, [(0.0314, 0.00006, 0.106, 2)], 0.002),
            'U-235': (235.044, 680.9, 0.0072, [(0.29, 0.0015, 0.037, 3)], 0.0045),

            # Additional common isotopes
            'C-12': (12.000, 0.00353, 0.9893, [], 0.0001),
            'Al-27': (26.982, 0.231, 1.0, [], 0.0009),
            'Fe-56': (55.845, 2.59, 0.9175, [], 0.0011),
            'Ag-107': (106.905, 37.6, 0.5184, [], 0.0063),
            'In-115': (114.904, 202, 0.957, [], 0.0012),
        }

        # Physical constants
        self.barn = 1e-24  # cm²
        self.avogadro = N_A

    def get_isotope_data(self, isotope):
        """Get nuclear data for a specific isotope."""
        if isotope not in self.nuclear_data:
            available = list(self.nuclear_data.keys())
            raise ValueError(f"Isotope {isotope} not found. Available: {available}")

        return self.nuclear_data[isotope]

    def calculate_number_density(self, isotope, density_g_cm3, abundance_override=None):
        """Calculate number density of nuclei in atoms/cm³."""
        mass, thermal_xs, abundance, resonances, fast_xs = self.get_isotope_data(isotope)

        if abundance_override is not None:
            abundance = abundance_override

        # Number density = ρ * N_A * abundance / A
        number_density = density_g_cm3 * self.avogadro * abundance / mass
        return number_density

    def thermal_cross_section(self, isotope, energy_ev, temperature_k=300):
        """Calculate thermal neutron cross-section using 1/v law."""
        mass, thermal_xs, abundance, resonances, fast_xs = self.get_isotope_data(isotope)

        # Thermal energy at given temperature
        k_b = 8.617333e-5  # eV/K
        thermal_energy_ref = 0.0253  # eV at 20°C

        # 1/v law: σ(E) = σ_th * sqrt(E_th/E) * sqrt(T_ref/T)
        thermal_factor = np.sqrt(thermal_energy_ref / energy_ev)
        temperature_factor = np.sqrt(300 / temperature_k)

        return thermal_xs * thermal_factor * temperature_factor

    def resonance_cross_section(self, isotope, energy_ev):
        """Calculate resonance contribution to cross-section."""
        mass, thermal_xs, abundance, resonances, fast_xs = self.get_isotope_data(isotope)

        if not resonances:
            return np.zeros_like(energy_ev)

        total_resonance = np.zeros_like(energy_ev)

        for E_res, Gamma_n, Gamma_gamma, J in resonances:
            # Breit-Wigner formula (simplified)
            Gamma_total = Gamma_n + Gamma_gamma

            # Statistical factor
            g = (2*J + 1) / 4  # Assuming spin 1/2 neutrons, spin 0 target for simplicity

            # Breit-Wigner peak
            sigma_res = g * (Gamma_n * Gamma_gamma) / ((energy_ev - E_res)**2 + (Gamma_total/2)**2)
            sigma_res *= 4 * pi / (2 * 0.0253)  # Normalization factor

            total_resonance += sigma_res

        return total_resonance

    def calculate_capture_cross_section(self, isotope, energy_ev, temperature_k=300):
        """Calculate total capture cross-section vs energy."""
        mass, thermal_xs, abundance, resonances, fast_xs = self.get_isotope_data(isotope)

        # Convert energy to array if needed
        energy_ev = np.array(energy_ev, dtype=float)

        # Initialize cross-section array
        sigma_total = np.zeros_like(energy_ev)

        # Thermal region (E < 0.5 eV)
        thermal_mask = energy_ev < 0.5
        if np.any(thermal_mask):
            sigma_total[thermal_mask] = self.thermal_cross_section(
                isotope, energy_ev[thermal_mask], temperature_k)

        # Resonance region (0.5 eV < E < 10 keV)
        resonance_mask = (energy_ev >= 0.5) & (energy_ev < 1e4)
        if np.any(resonance_mask):
            # Smooth transition from thermal + resonances
            thermal_part = self.thermal_cross_section(
                isotope, energy_ev[resonance_mask], temperature_k) * 0.1  # Reduced thermal
            resonance_part = self.resonance_cross_section(
                isotope, energy_ev[resonance_mask])
            sigma_total[resonance_mask] = thermal_part + resonance_part

        # Fast neutron region (E > 10 keV)
        fast_mask = energy_ev >= 1e4
        if np.any(fast_mask):
            # Transition to constant fast cross-section
            transition_energy = 1e4
            transition_xs = self.thermal_cross_section(isotope, transition_energy, temperature_k) * 0.1

            # Linear interpolation to fast cross-section
            log_energy_fast = np.log10(energy_ev[fast_mask])
            log_transition = np.log10(transition_energy)
            log_fast_energy = np.log10(1e6)  # 1 MeV

            interp_factor = np.clip((log_energy_fast - log_transition) / (log_fast_energy - log_transition), 0, 1)
            sigma_total[fast_mask] = transition_xs * (1 - interp_factor) + fast_xs * interp_factor

        return sigma_total

    def list_available_isotopes(self):
        """List all available isotopes with their thermal cross-sections."""
        print("Available isotopes and thermal capture cross-sections (barns):")
        print("-" * 60)
        for isotope, (mass, thermal_xs, abundance, resonances, fast_xs) in self.nuclear_data.items():
            print(f"{isotope:8s} | σ_th = {thermal_xs:8.1f} b | abundance = {abundance:6.4f}")

# ===================================================================
# BETHE-BLOCH CALCULATION FUNCTIONS
# ===================================================================

def calculate_bethe_bloch(kinetic_energy, particle, material_name, material_db):
    """
    Calculate Bethe-Bloch energy loss -dE/dx.

    Parameters:
    -----------
    kinetic_energy : array-like
        Kinetic energy in MeV
    particle : dict
        Particle properties with 'charge', 'mass' (MeV/c²), 'name'
    material_name : str
        Material name or chemical symbol
    material_db : MaterialDatabase
        Material database instance

    Returns:
    --------
    dEdx : array
        Energy loss -dE/dx in MeV·cm²/g
    """

    # Physical constants
    K = 0.307075  # MeV·cm²/g (4πNₐrₑ²mₑc²)
    m_e = 0.511   # MeV/c² (electron mass)

    # Get material properties
    Z, A, I_eV, density = material_db.get_material_properties(material_name)
    I = I_eV * 1e-6  # Convert eV to MeV

    # Particle properties
    z = particle['charge']
    mass = particle['mass']

    # Convert to arrays
    T = np.array(kinetic_energy, dtype=float)

    # Calculate relativistic parameters
    gamma = (T + mass) / mass
    beta_squared = 1.0 - 1.0 / (gamma**2)
    beta = np.sqrt(np.maximum(beta_squared, 0))

    # Avoid issues at very low energies
    valid_mask = (beta > 0.01) & (T > 0.1)

    # Calculate maximum energy transfer Tₘₐₓ
    numerator = 2 * m_e * beta_squared * (gamma**2)
    denominator = 1 + 2*gamma*m_e/mass + (m_e/mass)**2
    T_max = numerator / denominator

    # Bethe-Bloch formula
    factor1 = K * (z**2) * (Z / A) / beta_squared

    # Logarithmic term
    log_arg = (2 * m_e * beta_squared * (gamma**2) * T_max) / (I**2)
    log_arg = np.maximum(log_arg, 1e-10)  # Avoid log(0)
    log_term = np.log(log_arg)

    # Full formula (without density correction for simplicity)
    dEdx = factor1 * (log_term - beta_squared)

    # Set invalid regions to NaN
    dEdx[~valid_mask] = np.nan

    return dEdx

def calculate_specific_bethe_bloch_case(particle_name, material_name, energy_mev, particles_list, material_db):
    """Calculate and print Bethe-Bloch for a specific case."""

    # Find particle
    particle = None
    for p in particles_list:
        if p['name'].lower() == particle_name.lower():
            particle = p
            break

    if particle is None:
        print(f"Particle {particle_name} not found in particles list")
        return None

    try:
        dEdx = calculate_bethe_bloch([energy_mev], particle, material_name, material_db)

        print(f"\n{'='*50}")
        print(f"BETHE-BLOCH CALCULATION")
        print(f"{'='*50}")
        print(f"Particle: {particle['name']} (mass = {particle['mass']} MeV/c²)")
        print(f"Material: {material_name}")
        print(f"Kinetic Energy: {energy_mev} MeV")
        print(f"Energy Loss: {dEdx[0]:.3f} MeV·cm²/g")

        # Get material properties for context
        Z, A, I, rho = material_db.get_material_properties(material_name)
        print(f"Material properties:")
        print(f"  Z = {Z}, A = {A:.1f} g/mol")
        print(f"  I = {I:.1f} eV, ρ = {rho:.3f} g/cm³")
        print(f"{'='*50}")

        return dEdx[0]

    except Exception as e:
        print(f"Error in calculation: {e}")
        return None

# ===================================================================
# NEUTRON CAPTURE CALCULATION FUNCTIONS
# ===================================================================

def calculate_capture_probability(energies_ev, isotope, nuclear_db, density_g_cm3=None,
                                thickness_cm=1.0, temperature_k=300,
                                abundance_override=None):
    """
    Calculate neutron capture probability as function of energy.

    Parameters:
    -----------
    energies_ev : array-like
        Neutron energies in eV
    isotope : str
        Isotope name (e.g., 'B-10', 'Gd-157')
    nuclear_db : NuclearDatabase
        Nuclear database instance
    density_g_cm3 : float
        Material density in g/cm³ (uses default if None)
    thickness_cm : float
        Material thickness in cm
    temperature_k : float
        Temperature in Kelvin
    abundance_override : float
        Override natural abundance (0-1)

    Returns:
    --------
    probability : array
        Capture probability (0-1)
    cross_section : array
        Microscopic cross-section in barns
    macro_cross_section : array
        Macroscopic cross-section in cm⁻¹
    """

    # Use default density if not specified
    if density_g_cm3 is None:
        # Default densities for common materials
        default_densities = {
            'B-10': 2.34, 'Cd-113': 8.65, 'Gd-155': 7.90, 'Gd-157': 7.90,
            'Li-6': 0.534, 'He-3': 0.000178, 'U-235': 19.05, 'H-1': 1.0,
        }
        density_g_cm3 = default_densities.get(isotope, 1.0)
        if isotope not in default_densities:
            print(f"Warning: Using default density 1.0 g/cm³ for {isotope}")

    # Convert energies to array
    energies_ev = np.array(energies_ev, dtype=float)

    # Calculate microscopic cross-section
    cross_section = nuclear_db.calculate_capture_cross_section(
        isotope, energies_ev, temperature_k)

    # Calculate number density
    number_density = nuclear_db.calculate_number_density(
        isotope, density_g_cm3, abundance_override)

    # Macroscopic cross-section (cm⁻¹)
    macro_cross_section = number_density * cross_section * nuclear_db.barn

    # Capture probability
    probability = 1.0 - np.exp(-macro_cross_section * thickness_cm)

    return probability, cross_section, macro_cross_section

def calculate_transmission(energies_ev, isotope, nuclear_db, density_g_cm3=None,
                         thickness_cm=1.0, temperature_k=300):
    """Calculate neutron transmission probability (1 - capture probability)."""
    prob, xs, macro_xs = calculate_capture_probability(
        energies_ev, isotope, nuclear_db, density_g_cm3, thickness_cm, temperature_k)
    return 1.0 - prob, xs, macro_xs

def calculate_specific_neutron_case(isotope, energy_ev, nuclear_db, density_g_cm3=None,
                          thickness_cm=1.0, temperature_k=300):
    """Calculate and display capture probability for a specific case."""

    try:
        prob, xs, macro_xs = calculate_capture_probability(
            [energy_ev], isotope, nuclear_db, density_g_cm3, thickness_cm, temperature_k)

        print(f"\n{'='*60}")
        print(f"NEUTRON CAPTURE CALCULATION")
        print(f"{'='*60}")
        print(f"Isotope: {isotope}")
        print(f"Neutron Energy: {energy_ev:.6f} eV")
        print(f"Material Density: {density_g_cm3:.3f} g/cm³")
        print(f"Thickness: {thickness_cm:.2f} cm")
        print(f"Temperature: {temperature_k:.1f} K")
        print(f"")
        print(f"Results:")
        print(f"  Microscopic σ: {xs[0]:.3f} barns")
        print(f"  Macroscopic Σ: {macro_xs[0]:.6f} cm⁻¹")
        print(f"  Capture Probability: {prob[0]:.6f} ({prob[0]*100:.4f}%)")
        print(f"  Transmission: {(1-prob[0]):.6f} ({(1-prob[0])*100:.4f}%)")

        # Material info
        mass, thermal_xs, abundance, resonances, fast_xs = nuclear_db.get_isotope_data(isotope)
        number_density = nuclear_db.calculate_number_density(isotope, density_g_cm3 or 1.0)

        print(f"")
        print(f"Material Properties:")
        print(f"  Atomic Mass: {mass:.3f} amu")
        print(f"  Natural Abundance: {abundance:.4f}")
        print(f"  Thermal σ (0.0253 eV): {thermal_xs:.1f} barns")
        print(f"  Number Density: {number_density:.3e} atoms/cm³")
        print(f"{'='*60}")

        return prob[0], xs[0], macro_xs[0]

    except Exception as e:
        print(f"Error in calculation: {e}")
        return None, None, None

def compare_materials_at_energy(energy_ev, materials_list, nuclear_db, thickness_cm=1.0):
    """Compare capture probabilities for different materials at fixed energy."""

    print(f"\nMaterial Comparison at {energy_ev} eV (thickness = {thickness_cm} cm)")
    print("-" * 80)
    print(f"{'Isotope':<12} {'Density':<10} {'σ (barns)':<12} {'Σ (cm⁻¹)':<12} {'P_capture':<12}")
    print("-" * 80)

    results = []

    default_densities = {
        'B-10': 2.34, 'Cd-113': 8.65, 'Gd-155': 7.90, 'Gd-157': 7.90,
        'Li-6': 0.534, 'He-3': 0.000178, 'U-235': 19.05, 'H-1': 1.0,
    }

    for isotope in materials_list:
        try:
            density = default_densities.get(isotope, 1.0)
            prob, xs, macro_xs = calculate_capture_probability(
                [energy_ev], isotope, nuclear_db, density, thickness_cm)

            print(f"{isotope:<12} {density:<10.3f} {xs[0]:<12.3f} {macro_xs[0]:<12.6f} {prob[0]:<12.6f}")
            results.append((isotope, prob[0], xs[0], macro_xs[0]))

        except Exception as e:
            print(f"{isotope:<12} Error: {str(e)[:50]}")

    print("-" * 80)
    return results

# ===================================================================
# PHOTON ATTENUATION CALCULATIONS
# ===================================================================

def calculate_photoelectric_cross_section(energy_kev, Z):
    """Calculate photoelectric cross-section (simplified model)."""
    # Simplified approximation: σ_pe ∝ Z^n / E^m
    # More accurate models would use detailed tabulated data
    if energy_kev < 1:
        energy_kev = 1  # Avoid division by zero

    n = 4.5  # Atomic number dependence
    m = 3.0  # Energy dependence

    # Normalization constant (roughly fitted to experimental data)
    K_pe = 1e-3  # Adjust this constant based on actual data

    sigma_pe = K_pe * (Z**n) / (energy_kev**m)
    return sigma_pe

def calculate_compton_cross_section(energy_kev):
    """Calculate Compton scattering cross-section using Klein-Nishina formula."""
    # Klein-Nishina formula (per electron)
    m_e_kev = 511  # Electron rest mass in keV
    alpha = energy_kev / m_e_kev

    if alpha < 0.001:
        # Low energy approximation (Thomson scattering)
        sigma_kn = 0.665  # Thomson scattering cross-section in barns
    else:
        # Full Klein-Nishina formula
        term1 = 1 + alpha
        term2 = 2*(1 + alpha)/(1 + 2*alpha) - np.log(1 + 2*alpha)/alpha
        term3 = np.log(1 + 2*alpha)/(2*alpha) - (1 + 3*alpha)/((1 + 2*alpha)**2)

        sigma_kn = 0.665 * (term2 + term3) / alpha  # barns per electron

    return sigma_kn

def calculate_pair_production_cross_section(energy_kev, Z):
    """Calculate pair production cross-section."""
    # Threshold energy is 2*m_e*c² = 1022 keV
    if energy_kev < 1022:
        return 0.0

    # Simplified approximation above threshold
    # Real calculations would use more sophisticated models
    K_pp = 1e-3  # Normalization constant
    sigma_pp = K_pp * Z**2 * np.log(energy_kev / 1022)

    return max(sigma_pp, 0.0)

def calculate_total_attenuation_coefficient(energy_kev, Z, density_g_cm3):
    """Calculate total mass attenuation coefficient."""
    # Calculate individual cross-sections per atom
    sigma_pe = calculate_photoelectric_cross_section(energy_kev, Z)
    sigma_cs = calculate_compton_cross_section(energy_kev) * Z  # Per atom
    sigma_pp = calculate_pair_production_cross_section(energy_kev, Z)

    # Total cross-section per atom
    sigma_total = sigma_pe + sigma_cs + sigma_pp

    # Convert to mass attenuation coefficient (cm²/g)
    # μ/ρ = σ * N_A / A, where A is atomic mass
    A = Z * 1.0  # Approximate atomic mass (better to use actual values)
    mu_over_rho = sigma_total * N_A * 1e-24 / A  # Convert barns to cm²

    return mu_over_rho

def calculate_photon_transmission(energy_kev, Z, density_g_cm3, thickness_cm):
    """Calculate photon transmission probability using Beer-Lambert law."""
    mu_over_rho = calculate_total_attenuation_coefficient(energy_kev, Z, density_g_cm3)
    mu = mu_over_rho * density_g_cm3  # Linear attenuation coefficient

    transmission = np.exp(-mu * thickness_cm)
    absorption_probability = 1.0 - transmission

    return transmission, absorption_probability, mu

# ===================================================================
# INITIALIZATION
# ===================================================================

# Create singleton instances that can be imported
material_db = MaterialDatabase()
nuclear_db = NuclearDatabase()

print("✓ Particle Detectors Physics Library loaded successfully!")
print(f"  - MaterialDatabase: {len(material_db.mean_excitation_energies)} elements + {len(material_db.compound_database)} compounds")
print(f"  - NuclearDatabase: {len(nuclear_db.nuclear_data)} isotopes")
print("  - Bethe-Bloch, neutron capture, and photon attenuation functions available")