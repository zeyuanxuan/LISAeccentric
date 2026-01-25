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

### 2. CompactBinary Core (Object-Oriented API)
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
    * **Note**: The objects contain detailed GN evolutionary parameters in their `extra` attributes (e.g., outer orbit SMA `a2`, eccentricity `e2`, initial inclination `i`, and total `lifetime_yr`).

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
Generates a snapshot of the BBH population currently in the GN. This includes systems from both the steady-state formation channel and a recent starburst event (Young Nuclear Cluster, YNC).
* **Input**:
    * `rate_gn` (float, optional): Merger rate for the steady-state channel [Myr$^{-1}$]. Default is 2.0.
    * `age_ync` (float, optional): Age of the Young Nuclear Cluster [yr]. Default is 6.0e6.
    * `n_ync_sys` (int, optional): Number of systems to simulate for the YNC channel. Default is 100.
    * `max_bh_mass` (float, optional): Maximum Black Hole mass to consider [$M_\odot$]. Default is 50.
    * `plot` (bool, optional): If `True`, plots the snapshot population ($1-e$ vs. $a$, color-coded by SNR).
* **Output**:
    * A list of `CompactBinary` objects representing the surviving LISA sources, typically sorted by SNR.

**Example:**
```python
gn_snapshot = LISAeccentric.GN.get_snapshot(
    rate_gn=2.0, age_ync=6.0e6, n_ync_sys=100, max_bh_mass=50.0, plot=True
)
```
    <p align="left">
  <img src="./images/GNsnapshot.png" width="500">
   </p>

