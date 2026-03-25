# Particle Detectors - Physics Calculators

A comprehensive collection of physics calculators for particle detector design and analysis, including Bethe-Bloch energy loss calculations and neutron capture probability analysis.

## Features

### 🎯 Bethe-Bloch Energy Loss Calculator
- **Automatic material lookup** using chemical symbols (Si, Pb, Au, etc.)
- **Multiple particle types** (muons, pions, protons, electrons)
- **Energy-dependent calculations** with relativistic corrections
- **Comprehensive plotting tools** for materials and particles comparison
- **Built-in material database** with 15+ materials and compounds

### ⚛️ Neutron Capture Probability Calculator
- **Nuclear database** with capture cross-sections for 13+ isotopes
- **Energy-dependent modeling** (1/v law, Breit-Wigner resonances, fast neutrons)
- **Temperature effects** and material properties
- **Interactive analysis tools** for detector design
- **Comprehensive physics modeling** from ultra-cold to fast neutrons

### 📸 Photon Interactions Calculator
- **Photoelectric effect, Compton scattering, and pair production**
- **Energy-dependent cross-sections** for different materials
- **Attenuation coefficient calculations**
- **Beer-Lambert law implementation**

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/emanuele-villa/particle-detectors.git
   cd particle-detectors
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv energy-deposits-env

   # Activate virtual environment
   # On macOS/Linux:
   source energy-deposits-env/bin/activate

   # On Windows:
   energy-deposits-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install Jupyter kernel (for automatic notebook environment selection)**
   ```bash
   python -m ipykernel install --user --name energy-deposits-env --display-name "Energy Deposits (Physics)"
   ```

5. **Start Jupyter Lab**
   ```bash
   jupyter lab
   ```

6. **Open the notebooks**
   - `bethe-bloch.ipynb` - Bethe-Bloch energy loss calculations
   - `neutrons.ipynb` - Neutron capture probability analysis
   - `photon.ipynb` - Photon interaction analysis

   **Note**: The notebooks will automatically use the "Energy Deposits (Physics)" kernel. If not, select it manually from the kernel menu in Jupyter Lab.

## Environment & Kernel Setup

The project uses a dedicated Python environment called `energy-deposits-env` with a custom Jupyter kernel:

- **Environment name**: `energy-deposits-env`
- **Jupyter kernel**: "Energy Deposits (Physics)"
- **Auto-selection**: Notebooks are configured to use this kernel automatically
- **SciPy included**: All required physics libraries including SciPy are pre-installed
- **Modular architecture**: Heavy calculations moved to `lib.py` for clean notebook interface

## Usage Examples

### Bethe-Bloch Calculator

```python
# Calculate energy loss for muons in silicon
# Modify SELECTED_MATERIALS in the notebook:
SELECTED_MATERIALS = [
    'Si',   # Silicon
    'Ge',   # Germanium
    'Ar',   # Argon
    'Al',   # Aluminum
]

# Run the notebook cells to generate interactive plots
```

### Neutron Capture Calculator

```python
# Configure materials and parameters in the notebook:
SELECTED_MATERIALS = [
    'B-10',     # Boron-10 (high capture cross-section)
    'Gd-157',   # Gadolinium-157 (highest thermal cross-section)
    'He-3',     # Helium-3 (neutron detector gas)
    'Li-6'      # Lithium-6 (neutron converter)
]

# Adjust physical parameters
TEMPERATURE = 300        # Kelvin
DEFAULT_THICKNESS = 1.0  # cm
ENERGY_RANGE = [1e-9, 1e7]  # eV
```

## Configuration

All notebooks feature easy configuration sections at the top where you can:

- **Select materials** using chemical symbols or compound names
- **Configure energy ranges** for different physics regimes
- **Adjust physical parameters** (temperature, density, thickness)
- **Choose particle types** and their properties
- **Customize plotting options** and output formats

## Included Physics

### Bethe-Bloch Formula
```
-dE/dx = K·z²·(Z/A)·(1/β²)·[ln(2mₑc²β²γ²Tₘₐₓ/I²) - β²]
```
- **Relativistic corrections** (β, γ)
- **Material properties** (Z, A, I, ρ)
- **Temperature dependence** for accurate calculations

### Neutron Capture Physics
```
P = 1 - exp(-Σt)  where  Σ = n·σ(E)
```
- **1/v law** for thermal neutrons
- **Breit-Wigner resonances** for epithermal region
- **Fast neutron cross-sections** for high energies
- **Number density calculations**: n = ρ·N_A·abundance/A

### Photon Interactions
- **Photoelectric effect**: Energy-dependent cross-sections
- **Compton scattering**: Klein-Nishina formula
- **Pair production**: High-energy photon interactions
- **Total attenuation**: μ = μ_pe + μ_cs + μ_pp

## File Structure

```
particle-detectors/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── lib.py                 # Physics library with calculation functions
├── energy-deposits-env/   # Virtual environment (created during setup)
├── bethe-bloch.ipynb     # Interactive Bethe-Bloch calculator
├── neutrons.ipynb        # Neutron capture analysis
├── photon.ipynb          # Photon interaction analysis
└── .gitignore            # Git ignore file
```

## Dependencies

Core scientific libraries:
- **NumPy** ≥2.4.3 - Numerical computations
- **SciPy** ≥1.17.1 - Scientific computing and physical constants
- **Matplotlib** ≥3.10.8 - Plotting and visualization
- **Pandas** ≥3.0.1 - Data handling and analysis
- **Jupyter** ≥1.1.1 - Interactive notebook environment
- **Mendeleev** ≥0.6.1 - Automatic element property lookup

See `requirements.txt` for complete list with exact versions.

## Applications

### ✨ Detector Physics
- **Silicon detectors** - Energy loss calculations for charged particles
- **Gaseous detectors** - Particle identification and energy measurement
- **Calorimeter design** - Energy deposition modeling
- **Photon detectors** - X-ray and gamma-ray absorption analysis

### ⚛️ Nuclear Engineering
- **Neutron detectors** - Efficiency calculations for thermal and fast neutrons
- **Radiation shielding** - Attenuation analysis for different materials
- **Reactor physics** - Control rod effectiveness and neutron absorption
- **Medical physics** - Radiation therapy and imaging applications

### 🔬 Research Applications
- **Particle accelerators** - Beam diagnostics and detector optimization
- **Space applications** - Radiation environment modeling
- **Nuclear security** - Detection system design
- **Materials science** - Radiation damage studies

## Interactive Features

All notebooks include:
- **Real-time plotting** with parameter adjustment
- **Material comparison tools** at specific energies
- **Temperature and thickness dependence** analysis
- **Export capabilities** for data and plots
- **Built-in physics validation** and cross-checks

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs or request features via [GitHub Issues](https://github.com/emanuele-villa/particle-detectors/issues)
- Submit pull requests for improvements
- Add new materials or isotopes to the databases
- Improve documentation or add examples
- Contribute new physics calculators

## License

This project is open source. See repository for license details.

## Citation

If you use these calculators in your research, please consider citing:
```
Particle Detectors Physics Calculators
GitHub: https://github.com/emanuele-villa/particle-detectors
```

## Support

For questions or support:
- Check the [documentation](https://github.com/emanuele-villa/particle-detectors)
- Open an [issue](https://github.com/emanuele-villa/particle-detectors/issues)
- Review the notebook examples and inline documentation

---

**Made for the physics community** 🔬 **Built for detector design** ⚛️ **Optimized for research** 📊