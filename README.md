# LISAeccentric
**Xuan et al. (2026)**

## Overview

**LISAeccentric** is a Python toolkit for simulating eccentric compact binary populations and analyzing their gravitational-wave signals in the LISA band. It provides tools for population modeling, waveform generation, and signal analysis, including:

#### BBH Population Modeling
* **Galactic Nuclei (GN)**: SMBH-perturbed mergers (steady-state & starburst), based on *[e.g., Naoz et al., Your GN Ref]*.
* **Globular Clusters (GC)**: Dynamically formed BBHs, including in-cluster and ejected mergers, based on *Kremer et al. (2020)* and *Zevin et al. (2020)*.
* **Galactic Field**: Fly-by–induced mergers in Milky Way–like and elliptical galaxies, based on *Michaely & Perets (2019)* and *Xuan et al. (2024)*.

#### Gravitational Wave Modeling
* PN-based, time-domain inspiral waveforms for eccentric binaries.
* Orbital parameter evolution during inspiral.

#### Signal Analysis
* LISA detector response calculation (TDI).
* Signal-to-Noise Ratio (SNR) and inner-product computation.
* Characteristic strain ($h_c$) and stochastic background evaluation.

---

## Installation

To install the package locally, download and extract the source code. From the root directory (containing `setup.py`), run:
```bash
pip install .
```
You can also install the latest version directly from GitHub using pip:
```
pip install git+[https://github.com/zeyuanxuanphy/LISAeccentric.git](https://github.com/zeyuanxuanphy/LISAeccentric.git)
```
Or if you prefer to clone manually:
```
git clone [https://github.com/zeyuanxuanphy/LISAeccentric.git](https://github.com/zeyuanxuanphy/LISAeccentric.git)
cd LISAeccentric
pip install .
```

## Features & Usage Examples

The following examples demonstrate the core workflows derived from the official tutorial.1. Galactic Nucleus (GN)Model SMBH-perturbed mergers and starburst scenarios using the Kozai-Lidov mechanism.Sample Eccentricities (LIGO Band)Analyze the eccentricity distribution as binaries enter the 10Hz frequency band.Pythonimport LISAeccentric
import matplotlib.pyplot as plt

### Sample 5000 systems
```
gn_e_samples = LISAeccentric.GN.sample_eccentricities(
    n_samples=5000,
    max_bh_mass=50.0,
    plot=True  # Generates CDF plot
)
```
Output Example:
<p align="left">
  <img src="./images/GNecc_LIGO.png" width="500">
</p>

### Population Snapshot (LISA Band)
Simulate the current population of Black Hole Binaries (BBHs) in the nucleus.Pythongn_snapshot = LISAeccentric.GN.get_snapshot(
    rate_gn=2.0,       # Formation rate [systems/Myr]
    age_ync=6.0e6,     # Age of Young Nuclear Cluster
    n_ync_sys=100,
    plot=True
)
2. Globular Clusters (GC)Analyze dynamical mergers, distinguishing between "In-cluster" retained binaries and "Ejected" populations.Python# Compare populations
gc_e_samples = LISAeccentric.GC.sample_eccentricities(
    n=5000,
    channel_name='Incluster',  # Options: 'Incluster', 'Ejected'
    plot=True
)

# Generate a full population realization (e.g., 10 realizations)
gc_data = LISAeccentric.GC.get_snapshot(mode='10_realizations')
Output Example:3. Galactic FieldSimulate fly-by induced mergers in Milky Way-like and Elliptical galaxies.Python# Milky Way Field Simulation
mw_field = LISAeccentric.Field.simulate_mw_field(
    n_systems=1000,
    plot=True
)
4. Waveform AnalysisCompute Signal-to-Noise Ratio (SNR), orbital evolution, and Characteristic Strain ($h_c$).Characteristic Strain ($h_c$)Calculate and plot the characteristic strain against the LISA sensitivity curve.Python# Select a target system from the snapshot
target_sys = gn_snapshot[0]

# Compute strain for a 4-year observation period
LISAeccentric.Waveform.compute_characteristic_strain_single(
    system=target_sys,
    tobs_years=4.0,
    plot=True
)
Output Example:Orbital EvolutionCalculate time to merger and evolve system parameters.Python# Compute time until merger
t_merge = LISAeccentric.Waveform.compute_merger_time(system=target_sys)

# Evolve the orbit to a future time (e.g., halfway to merger)
if t_merge != float('inf'):
    LISAeccentric.Waveform.evolve_orbit(
        system=target_sys,
        delta_t_years=t_merge / 2.0
    )
DependenciesThe package relies on the following Python libraries:numpyscipymatplotlibpandasnumbaLicense[Insert License Information Here]
