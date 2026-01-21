# LISAeccentric
**Xuan et al. (2026)**

## Overview

**LISAeccentric** is a Python toolkit for generating eccentric compact binary populations and computing their gravitational wave signals in the LISA band. It supports population synthesis, waveform computation, and signal analysis, including:
#### BBH Population Catalogs
* **Galactic Nuclei (GN)**: SMBH-perturbed mergers (steady-state & starburst), based on [e.g., Naoz et al., ....].
* **Globular Clusters (GC)**: Dynamically formed BBHs, including in-cluster and ejected mergers, based on Kremer et al. (2020) and Zevin et al. (2020).
* **Galactic Field**: Fly-by–induced mergers in Milky Way–like and elliptical galaxies, based on Michaely & Perets (2019) and Xuan et al. (2024).

#### Waveform & Signal Analysis
* Generate PN-based, time-domain waveforms for eccentric binaries.
* Evolve orbital parameters throughout the inspiral.
* Compute LISA detector response (TDI).
* Calculate signal-to-noise ratio (SNR) and waveform inner products.
* Evaluate characteristic strain ($h_c$) and stochastic background.

---

## Installation

You can install `LISAeccentric` directly from GitHub without manually downloading or unzipping files. Please choose the method that matches your environment.

#### Method 1: Jupyter Notebook / Google Colab (Recommended)
If you are working in a notebook (Jupyter, Colab, Kaggle), run the following command in a code cell. 
```python
!pip install https://github.com/zeyuanxuanphy/LISAeccentric/archive/refs/heads/main.zip
```
#### Method 2: Terminal / Command Line
If you are using a standard terminal, run the command without the !
```python
pip install https://github.com/zeyuanxuanphy/LISAeccentric/archive/refs/heads/main.zip
```
**Note:** for Mac/Linux: If pip command is not found or defaults to Python 2, try using pip3 instead: 
```
pip3 install https://github.com/zeyuanxuanphy/LISAeccentric/archive/refs/heads/main.zip
```
#### Method 3: University Clusters / HPC
If you are running jobs on a cluster using existing Python modules (like `module load python/3.9.6`), **load the same module before installing.**

Step 1: Load the Python module you intend to use in your job script
```bash
# Example: If your submission script uses python/3.9.6, load it now:
module load python/3.9.6
```
Step 2: Install the package with --user. This installs the package into your local directory specific to that Python version (e.g., ~/.local/lib/python3.9/site-packages).
```
python3 -m pip install --user https://github.com/zeyuanxuanphy/LISAeccentric/archive/refs/heads/main.zip
```
Step 3: Run your job
```
# In your job script (.sh/.pbs):
module load python/3.9.6
python your_script.py
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
