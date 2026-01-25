# LISAeccentric
**Xuan et al. (2026)**

## Overview

**LISAeccentric** is a Python toolkit for generating eccentric compact binary populations and computing their gravitational wave signals in the LISA band. It supports population synthesis, waveform computation, and signal analysis, including:
#### BBH Population Catalogs
* **Galactic Nuclei (GN)**: SMBH-perturbed mergers (steady-state & starburst), based on [e.g., Naoz et al., ....].
* **Globular Clusters (GC)**: Dynamically formed BBHs, including in-cluster and ejected mergers, based on Kremer et al. (2020) and Zevin et al. (2020).
* **Galactic Field**: Fly-by–induced mergers in Milky Way–like and elliptical galaxies, based on Michaely & Perets (2019) ...

#### Waveform & Signal Analysis
* Generate PN-based, time-domain waveforms for eccentric binaries.
* Evolve orbital parameters throughout the inspiral stage.
* Compute LISA detector response (Michelson response).
* Evaluate characteristic strain ($h_c$) and stochastic backgrounds.
* Calculate signal-to-noise ratio (SNR) and noise-weighted inner products for time-domain waveforms.


---

## Installation

You can install `LISAeccentric` directly from GitHub without manually downloading or unzipping files. Please choose the method that matches your environment.

#### Method 1: Jupyter Notebook / Google Colab (Recommended)
If you are working in a notebook (Jupyter, Colab, Kaggle), run the following command in a code cell. 
```
!pip install https://github.com/zeyuanxuan/LISAeccentric/archive/refs/heads/main.zip
```
#### Method 2: Terminal / Command Line
If you are using a standard terminal, run the command without the !
```
pip install https://github.com/zeyuanxuan/LISAeccentric/archive/refs/heads/main.zip
```
**Note:** for Mac/Linux: If pip command is not found or defaults to Python 2, try using pip3 instead: 
```
pip3 install https://github.com/zeyuanxuan/LISAeccentric/archive/refs/heads/main.zip
```
#### Method 3: University Clusters / HPC
If you are running jobs on a cluster using existing Python modules (like `module load python/3.9.6`), **load the same module before installing.**

Step 1: Load the Python module you intend to use in your job script
```
# Example: If your submission script uses python/3.9.6, load it now:
module load python/3.9.6
```
Step 2: Install the package with --user. This installs the package into your local directory specific to that Python version (e.g., ~/.local/lib/python3.9/site-packages).
```
python3 -m pip install --user https://github.com/zeyuanxuan/LISAeccentric/archive/refs/heads/main.zip
```
Step 3: Import LISAeccentric in your code and run your job
```
# In your job script (.sh/.pbs):
module load python/3.9.6
python your_script.py
```

## Features & Usage Examples

### 1. Global Configuration

#### `LISAeccentric.set_output_control`
Sets the global verbosity and warning suppression levels.
* **Input**: 
    * `verbose` (bool): If `False`, disables internal library printing.
    * `show_warnings` (bool): If `False`, suppresses warnings.
* **Output**: `None`.

**Example:**
```python
# Set verbose=False to disable internal library printing.
LISAeccentric.set_output_control(verbose=False, show_warnings=False)
```

### 2. CompactBinary Class
The fundamental unit of the package. This class handles the physics, evolution, and I/O for a single binary system.

#### `LISAeccentric.CompactBinary()`
To create a binary system object:
* **Input**:
    * `m1`, `m2` (float): Masses. [m_sun]
    * `a` (float): Semi-major axis. [au]
    * `e` (float): Eccentricity.
    * `Dl` (float): Luminosity distance. [kpc]
    * `label` (str): Identifier.
    * `extra` (dict, optional): Dictionary for storing extended parameters (e.g., SNR, inclination **[rad]**, spin, lifetime).
* **Output**: `CompactBinary` object.

**Example:**
```python
my_binary = LISAeccentric.CompactBinary(
    m1=10.0, m2=10.0, a=0.26, e=0.985, Dl=8.0, 
    label="Tutorial_Core_Obj",
    extra={
        'inclination': 0.7854,  # [rad] (~45 degrees)
    }
)
print(f"   Output Object: {my_binary}")
print(f"   Type Inspection: {type(my_binary)}")
# You can also access extra data directly
print(f"   Inclination: {my_binary.extra['inclination']:.4f} rad")
```
* **Output**:
  ```
   Output Object: <CompactBinary [Tutorial_Core_Obj]: M=10.0+10.0 m_sun, a=2.600e-01AU, e=0.9850, Dl=8.0kpc | inclination=0.785>
   Type Inspection: <class 'LISAeccentric.core.CompactBinary'>
   Inclination: 0.7854 rad
  ```

#### `.compute_merger_time()`
Calculates the remaining time until the merger due to gravitational wave emission.
* **Input**: None (uses object attributes).
* **Output**:
    * `t_merge_yr` (float): Time to merger in years.

**Example:**
```python
t_merge_yr = my_binary.compute_merger_time(verbose=False)
print(f"      Return Value: {t_merge_yr:.4e} [years] (Type: float)")
```
* **Output**:
  ```
         Return Value: 4.8407e+06 [years] (Type: float)
  ```

#### `.compute_snr_analytical()`
Computes the sky-averaged Signal-to-Noise Ratio (SNR) for the LISA detector. This method supports two calculation modes: full integration over harmonics (default) or a fast approximation.
* **Input**:
    * `tobs_yr` (float): Observation duration in years.
    * `quick_analytical` (bool, optional):
        * If `False` (default): Uses full integration (summing over harmonics via `PN_waveform.SNR`).
        * If `True`: Uses a fast geometric approximation based on peak frequency and amplitude, suitable for high-e systems.
    * `verbose` (bool, optional): Controls standard output printing. Default is `True`.
* **Output**:
    * `snr_val` (float): The calculated SNR value.
* **Note:** The calculation assumes the binary's evolution is negilible during the observation.
  
**Example:**
```python
snr_val = my_binary.compute_snr_analytical(tobs_yr=4.0, verbose=False, quick_analytical=False)
print(f"      Return Value: {snr_val:.4f} (Type: float)")
```
* **Output**:
  ```
      Return Value: 10.9644 (Type: float)
  ```

#### `.compute_fpeak()`
Calculates the peak gravitational wave frequency ($f_{\rm peak}$) for the eccentric binary using the Wen (2003) approximation. For highly eccentric systems, the GW power peaks at a frequency significantly higher than the orbital frequency: $f_{\rm peak} \approx f_{\rm orb} \frac{(1+e)^{1.1954}}{(1-e)^{1.5}}$.

* **Input**:
    * `verbose` (bool, optional): Controls standard output printing. Default is `True`.
* **Output**:
    * `f_peak` (float): Peak GW frequency [Hz].
    * **Note**: The result is also stored in `self.extra['f_peak_Hz']`.

**Example:**
```python
f_peak = my_binary.compute_fpeak(verbose=False)
print(f"      Return Value: {f_peak:.4e} [Hz] (Type: float)")
```
* **Output**:
  ```
       Return Value: 1.3205e-03 [Hz] (Type: float)
  ```

#### `.evolve_orbit()`
Predicts the future state of the binary system by evolving its orbital parameters forward in time due to gravitational wave emission (Peters64 formula).
* **Input**:
    * `delta_t_yr` (float): Time duration to evolve the system in years.
    * `update_self` (bool, optional):
        * If `True`: Updates the `a` and `e` attributes of the `CompactBinary` object itself.
        * If `False` (default): Returns the new values without modifying the object.
    * `verbose` (bool, optional): Controls standard output printing.
* **Output**:
    * `a_new` (float): The evolved semi-major axis [au].
    * `e_new` (float): The evolved eccentricity.

**Example:**
```python
a_new, e_new = my_binary.evolve_orbit(delta_t_yr=1000.0, update_self=False, verbose=False)
print(f"      Return Tuple: a={a_new} au, e={e_new}")
```
* **Output**:
  ```
        Return Tuple: a=0.25991616861323 au, e=0.9849951873952284
  ```
  
#### `.compute_waveform()`
A convenience method to compute the Gravitational Wave (GW) waveform specifically for the initialized binary system. It automatically utilizes the object's internal physical attributes ($m_1, m_2, a, e, D_L$) and supports adaptive time sampling.
* **Input**:
    * Observation:
        * `tobs_yr` (float): Observation duration in years.
        * `initial_orbital_phase` (float, optional): Initial mean anomaly $l_0$ [rad]. Default is 0.
    * Source Geometry:
        * `theta` (float, optional): Line-of-sight inclination angle in source frame $\theta$ [rad]. Default is $\pi/4$.
        * `phi` (float, optional): Line-of-sight azimuthal angle in source frame $\phi$ [rad]. Default is $\pi/4$.
    * Physics Model:
        * `PN_orbit` (int, optional): PN order for conservative orbital dynamics (0, 1, 2, 3). Default is 3.
        * `PN_reaction` (int, optional): PN order for radiation reaction (0, 1, 2). Default is 2.
    * Computational Control:
        * `ts` (float, optional): Fixed sampling time step [s]. If `None` (default), uses adaptive sampling.
        * `points_per_peak` (int, optional): Resolution for adaptive sampling (points per periastron passage). Default is 50.
        * `max_memory_GB` (float, optional): Safety limit for array size in GB. Default is 16.0.
    * Output Control:
        * `plot` (bool, optional): If `True`, plots the $h_+$ waveform.
        * `verbose` (bool, optional): Controls standard output printing.
* **Output**:
    * A list of three NumPy arrays: `[time_vector, h_plus, h_cross]`.
    * Returns `None` if calculation fails.
* **Note:** If the merger time is shorter than tobs, the code will truncate the waveform before reaching the ISCO.
  
**Example:**
```python
wf_data_obj = my_binary.compute_waveform(
    tobs_yr=1.0, points_per_peak=50, verbose=False, plot=True
)
```
* **Output**:
<p align="left">
  <img src="./images/waveformeg.png" width="500">
</p>

#### `.compute_characteristic_strain()`
Computes the characteristic strain spectrum ($h_c$) for the binary system by decomposing the signal into orbital harmonics.
* **Input**:
    * `tobs_yr` (float): Integration time in years.
    * `plot` (bool, optional): If `True`, generates a spectrum plot.
* **Output**:
    * A list of 4 NumPy arrays: `[freq, hc_spectrum, harmonics, snr_contrib]`.
        * `[0] freq`: Frequency List [Hz].
        * `[1] hc_spectrum`: Time-integrated Spectrum Amplitude ($h_{c,\rm avg}$), representing the accumulated signal over $T_{\rm obs}$.
        * `[2] harmonics`: Instantaneous characteristic strain ($h_{c,n}$) for each harmonic.
        * `[3] snr_contrib`: Contribution to noise power spectral density ($S_n(f)$) at harmonic frequencies.
* **Note:** The calculation assumes the binary's evolution is slow during the observation.
  
**Example:**
```python
strain_res_list = my_binary.compute_characteristic_strain(tobs_yr=4.0, plot=True)
```
* **Output**:
<p align="left">
  <img src="./images/characteristic.png" width="500">
</p>

#### `.to_list()` `.from_list()`
Methods to convert `CompactBinary` objects to and from list formats, facilitating data storage (e.g., to CSV/NumPy files) and retrieval.
* **.to_list()**: 
    * **Input**: `schema` (str) - formatting standard (default: snapshot_std, i.e., `['label', 'Dl', 'a', 'e', 'm1', 'm2', 'snr']`).
    * **Output**: A list representing the system's data.
* **.from_list()**: 
    * **Input**: `data_list` (list) - raw data values; `schema` (str).
    * **Output**: A new `CompactBinary` object instantiated from the list.

**Example:**
```python
    # Export
    print("   A. to_list(schema='snapshot_std')")
    data_row = my_binary.to_list(schema='snapshot_std')
    print(f"      Output: {data_row} (Type: List)")
    # Import
    print("   B. from_list(data_list=..., schema='snapshot_std')")
    raw_in = ["Imp_Sys", 16.8, 0.5, 0.9, 50.0, 50.0, 0.0]
    new_obj = LISAeccentric.CompactBinary.from_list(data_list=raw_in, schema='snapshot_std')
    print(f"      Output: {new_obj}")
  ```
* **Output**:
    ```
    A. to_list(schema='snapshot_std')
      Output: ['Tutorial_Core_Obj', 8.0, 0.26, 0.985, 10.0, 10.0, 0.0] (Type: List)
   B. from_list(data_list=..., schema='snapshot_std')
      Output: <CompactBinary [Imp_Sys]: M=50.0+50.0 m_sun, a=0.50AU, e=0.9000, Dl=16.8kpc, SNR=0.00>
    ```
### 3. Population analysis
#### 3.1 Galactic Nuclei (GN)
This module models Binary Black Holes formed dynamically in the Milky Way galactic nuclei (due to the perturbation of the central supermassive black hole). It is based on detailed three-body simulations.
#### ` LISAeccentric.GN.sample_eccentricities()`
Randomly samples $N$ merger eccentricities for BBHs formed in Galactic Nuclei, defined at the LIGO frequency band (10Hz).
* **Input**:
    * `n_samples` (int): Number of eccentricity samples to generate.
    * `max_bh_mass` (float, optional): Maximum Black Hole mass to consider for the population filter [$M_\odot$]. Default is 50.
    * `plot` (bool, optional): If `True`, plots the Cumulative Distribution Function (CDF) of $\log_{10}(e)$.
* **Output**:
    * `gn_e_samples` (NumPy Array): A 1D array containing the sampled eccentricity values at 10Hz.

**Example:**
```python
gn_e_samples = LISAeccentric.GN.sample_eccentricities(
    n_samples=5000, max_bh_mass=50.0, plot=True
)
print(f"   Output Shape: {np.shape(gn_e_samples)}")
print(f"   Mean Eccentricity: {np.mean(gn_e_samples)}")
```
* **Output**:
    ```
    Output Shape: (5000,)
    Mean Eccentricity: 3.791297808628803e-05
    ```
<p align="left">
<img src="./images/GNecc_LIGO.png" width="500">
</p>

#### `LISAeccentric.GN.get_progenitor()`
Retrieves the properties of the binary progenitors (initial states) from the underlying population catalog (BBH in GN, orbiting around a SMBH with M = 4e6 msun). These are the systems *before* they evolve to merger.
* **Input**:
    * `n_inspect` (int, optional): Number of random systems to retrieve for inspection. Default is 3.
* **Output**:
    * A list of `CompactBinary` objects representing the progenitor systems.
    * **Note**: The objects contain detailed GN evolutionary parameters in their `extra` attributes (e.g., outer orbit SMA `a2`, eccentricity `e2`, initial mutal orbit inclination `i`, and total `lifetime_yr`).

**Example:**
```python
gn_progenitors = LISAeccentric.GN.get_progenitor(n_inspect=3)
print(f"   Output List Length: {len(gn_progenitors)}")
print(f"   Sample Item: {gn_progenitors[0]}")
```
* **Output**:
    ```
   Output List Length: 3
   Sample Item: <CompactBinary [GN_Progenitor]: M=50.6+25.7 m_sun, a=3.395e-01AU, e=0.9278, Dl=8.0kpc | e2_init=0.505, i_init_rad=2.167, a2_init=1.57e+04, a_final=1.45e-05, e_final=3.04e-06, lifetime_yr=1.03e+08>
    ```
#### `LISAeccentric.GN.get_snapshot()`
Generates a snapshot of the BBH population currently in the GN. This includes systems from both the steady-state formation channel and a recent starburst event (Young Nuclear Cluster, YNC). The results can be changed by adjusting the BBH formation rate in the steady-state population and the age and total BBH number in the YNC population.
* **Input**:
    * `rate_gn` (float, optional): Merger rate for the steady-state channel [Myr$^{-1}$]. Default is 2.0.
    * `age_ync` (float, optional): Age of the Young Nuclear Cluster [yr]. Default is 6.0e6.
    * `n_ync_sys` (int, optional): Number of systems to simulate for the YNC channel. Default is 100.
    * `max_bh_mass` (float, optional): Maximum Black Hole mass to consider [$M_\odot$]. Default is 50.
    * `plot` (bool, optional): If `True`, plots the snapshot population ($1-e$ vs. $a$, color-coded by SNR).
* **Output**:
    * A list of `CompactBinary` objects representing the BBHs in the Milky Way center, typically sorted by SNR.

**Example:**
```python
gn_snapshot = LISAeccentric.GN.get_snapshot(
    rate_gn=2.0, age_ync=6.0e6, n_ync_sys=100, max_bh_mass=50.0, plot=True
)
print(f"   Output List Length: {len(gn_snapshot)} systems")
```
* **Output**:
    ```
   Output List Length: 1806 systems
    ```
<p align="left">
<img src="./images/GNsnapshot0.png" width="500">
</p>

### 3.2 Globular Clusters (GC)
This module models Binary Black Holes formed dynamically in Milky Way globular clusters. It supports sampling from specific dynamic formation channels (e.g., Kozai-Lidov triples, binary-single captures) based on detailed Monte Carlo N-body simulations.

#### `LISAeccentric.GC.sample_eccentricities()`
Randomly samples $N$ merger eccentricities for GC BBHs at the LIGO frequency band (10Hz).
* **Input**:
    * `n` (int): Number of eccentricity samples to generate.
    * `channel_name` (str): Specific formation channel.
        * `'Incluster'`: Weighted average of all in-cluster channels (default).
        * `'Ejected'`: Mergers occurring after ejection.
        * **Sub-channels**: Supports specific dynamical channels such as `'KL Triple'`, `'Non-KL Triple'`, `'Single Capture'`, `'Fewbody Capture'`.
    * `plot` (bool, optional): If `True`, plots the CDF of $\log_{10}(e)$.
* **Output**:
    * `gc_e_samples` (NumPy Array): A 1D array containing the sampled eccentricity values at 10Hz.

**Example:**
```python
gc_e_samples = LISAeccentric.GC.sample_eccentricities(
    n=5000, channel_name='KL Triple', plot=True
)
print(f"   Output Shape: {np.shape(gc_e_samples)}")
```
* **Output**:
    ```
   Output Shape: (5000,)
    ```
<p align="left">
<img src="./images/GCecc_LIGO.png" width="500">
</p>

#### `LISAeccentric.GC.get_snapshot()`
Retrieves a snapshot of the GC BBH population in the Milky Way. This method supports three retrieval modes to allow for different scales of analysis (full ensemble vs. single galaxy realization).
* **Input**:
    * `mode` (str): Data selection mode.
        * `'10_realizations'`: Returns the full catalog from 10 MW realizations (~2300 systems).
        * `'single'`: Returns data from a single MW realization (randomly selected subset, ~230 systems).
        * `'random'`: Returns a specific number of randomly selected systems.
    * `n_random` (int, optional): Number of systems to retrieve (only used if `mode='random'`). Default is 500.
    * `plot` (bool, optional): If `True`, plots the snapshot ($1-e$ vs $a$).
* **Output**:
    * A list of `CompactBinary` objects.
    * **Warning**: The underlying catalog represents a finite set of simulations (~230 systems per realization). If `n_random` exceeds the size of a single realization, the returned sample will inevitably mix systems from different stochastic realizations. Due to the small sample size of the MC N-body source catalog, these samples may not be strictly statistically independent.

**Example:**
```python
gc_data_full = LISAeccentric.GC.get_snapshot(mode='10_realizations', plot=True)
print(f"   Output List Length: {len(gc_data_full)}")
```
* **Output**:
    ```
   Output List Length: 2325
    ```
<p align="left">
<img src="./images/GCsnapshot.png" width="500">
</p>

### 3.2 Galactic Field (Field)
This module models Binary Black Holes mergers formed via dynamic fly-by interactions in galactic field environments. It supports simulations for both Milky Way-like (disk) galaxies and Elliptical galaxies.

#### `LISAeccentric.Field.run_simulation()`
Executes a Monte Carlo simulation to generate a population of fly-by mergers based on specific galactic structure and physical parameters. The results are saved to disk for subsequent analysis (sampling/snapshotting).

* **Input**:
    * `galaxy_type` (str, optional): Target environment `'MW'` (Milky Way) or `'Elliptical'`. Default: `'MW'`.
    * **Physics Parameters**:
        * `m1`, `m2`, `mp` (float, optional): Masses of the binary components and perturber [$M_\odot$]. Default: `10`, `10`, `0.6`.
        * `fbh` (float, optional): Fraction of stars that are wide binary black holes. Default: `7.5e-4`.
        * `fgw` (float, optional): Gravitational wave frequency for getting the eccentricity distribution (default 10Hz, LIGO band). Default: `10`.
        * `formation_mod` (str, optional): Star formation history model (e.g., `'starburst'`, `'continuous'`). Default: `'starburst'`.
    * **Structure Parameters (MW)**:
        * `n0` (float, optional): Stellar number density normalization in the solar neighborhood [pc$^{-3}$]. Default: `0.1`.
        * `rsun` (float, optional): Distance of the solar system to the Galactic center [pc]. Default: `8000.0` (8e3).
        * `Rl`, `h` (float, optional): Galactic scale lengths (Radial length, Vertical height) [pc]. Default: `2600.0`, `1000.0`.
        * `sigmav` (float, optional): Velocity dispersion [m/s]. Default: `50000.0` (50e3).
        * `age_mw` (float, optional): Age of the MW galaxy [years]. Default: `10e9`.
    * **Structure Parameters (Elliptical)**:
        * `M_gal` (float, optional): Total mass of the galaxy [$M_\odot$]. Default: `1.0e12`.
        * `Re` (float, optional): Effective radius (half-light radius) [pc]. Default: `8000.0`.
        * `distance_Mpc` (float, optional): Distance to the galaxy [Mpc]. Default: `16.8`.
        * `age_ell` (float, optional): Age of the elliptical galaxy [years]. Default: `13e9`.
    * **Control**:
        * `arange_log` (list, optional): Range of BBH semi-major axis $\log_{10}(a)$ to sample [min, max] in au. Default: `[2, 4.5]`.
        * **For MW**:
            * `n_sim_samples` (int, optional): Total number of MC trials. Default: `200000`.
            * `target_N` (int, optional): Target number of successful mergers to accumulate. Default: `100000`.
            * `rrange_mw` (list, optional): Radial range for simulation [min, max] in kpc. Default: `[0.5, 15]`.
        * **For Elliptical**:
            * `ell_n_sim` (int, optional): Total number of MC trials. Default: `100000`.
            * `ell_target_N` (int, optional): Target number of successful mergers to accumulate. Default: `50000`.
            * `rrange_ell` (list, optional): Radial range for simulation [min, max] in kpc. Default: `[0.05, 100]`.
* **Output**:
    * `None`. (Results are saved internally to `data/` directory).

**Example:**
```python
LISAeccentric.Field.run_simulation(
    galaxy_type='MW',
    # Physics (Optional overrides)
    m1=10.0, m2=10.0, mp=0.6, 
    fbh=7.5e-4, fgw=10.0,
    formation_mod='starburst',
    # Structure (Optional overrides)
    n0=0.1, rsun=8000.0, Rl=2600.0, h=1000.0, sigmav=50000.0,
    # Control (Optional overrides)
    n_sim_samples=200000, target_N=100000, rrange_mw=[0.5, 15]
)
print("   Status: Simulation completed and saved.")
```

#### `LISAeccentric.Field.get_progenitor()`
Retrieves the properties of the binary progenitors (initial states at formation) from the simulated library. These represent the system parameters right after being perturbed to high eccentricity and start evolving via GW emission. 

* **Input**:
    * `galaxy_type` (str, optional): Target environment `'MW'` (Milky Way) or `'Elliptical'`. Default: `'MW'`.
    * `plot` (bool, optional): If `True`, plots the initial semi-major axis distribution and the merger lifetime CDF. Default: `True`.
* **Output**:
    * A list of `CompactBinary` objects representing the progenitor systems. The length of the list equals `target_N` when running the galaxy simulation.
    * **Note**: These objects contain detailed simulation statistics in their `extra` attributes, including:
        * `merger_rate`: Effective merger rate weight for this system.
        * `lifetime_yr`: Total time from formation to merger.
        * `e_final_LIGO`: Eccentricity when entering the LIGO band.

**Example:**
```python
field_progs = LISAeccentric.Field.get_progenitor(galaxy_type='MW', plot=True)
print(f"   Output List Length: {len(field_progs)}")
```
* **Output**:
    ```
   Output List Length: 50000
    ```
<p align="left">
<img src="./images/Field_sma.png" width="500">
</p>
<p align="left">
<img src="./images/Field_lifetime.png" width="500">
</p>
