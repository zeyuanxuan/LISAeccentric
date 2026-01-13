# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import scipy.interpolate as sci_interpolate
import scipy.constants as sciconsts
from scipy.optimize import brentq
import copy
import random
import os

# ==========================================
# 1. Physical Constants Definition (Strictly following Natural/Geometric Units)
# ==========================================
# In this system, mass, length, and time are unified as "seconds (s)"
# G = c = 1

pi = sciconsts.pi
gama = 0.577215664901532860606512090082402431042159335  # Euler's constant

# Convert physical quantities to seconds
m_sun_sec = 1.9891e30 * sciconsts.G / np.power(sciconsts.c, 3.0)  # Solar mass (s)
AU_sec = sciconsts.au / sciconsts.c  # Astronomical Unit (s)
pc_sec = 3.261 * sciconsts.light_year / sciconsts.c  # Parsec (s)
year_sec = 365 * 24 * 3600.0  # Year (s)
day_sec = 24 * 3600.0  # Day (s)


# ==========================================
# 2. Evolutionary Physics Functions (Keep original simulation units: Msun, AU, Year)
# ==========================================
# These functions are used to process orbital evolution in NPY files, kept as is.

def GWtime(m1, m2, a1, e1):
    """Calculate merger time due to GW emission (Peters 1964)."""
    if e1 >= 1.0 or a1 <= 0: return 0.0
    factor = 1.6e13  # Coeff derived from constants
    return factor * (2 / m1 / m2 / (m1 + m2)) * np.power(a1 / 0.1, 4.0) * np.power(1 - e1 * e1, 7 / 2)


def peters_factor_func(e):
    """Auxiliary function for Peters equation evolution."""
    if e < 1e-10: return 0.0
    term1 = np.power(e, 12.0 / 19.0)
    term2 = 1.0 - e * e
    term3 = np.power(1.0 + (121.0 / 304.0) * e * e, 870.0 / 2299.0)
    return (term1 / term2) * term3


def solve_ae_after_time(m1, m2, a0, e0, dt):
    """
    Evolve semi-major axis (a) and eccentricity (e) by time dt using GW emission.
    """
    current_life = GWtime(m1, m2, a0, e0)

    # If dt is longer than remaining life, the system has merged.
    if dt >= current_life:
        return 0.0, 0.0  # Merged

    t_rem_target = current_life - dt

    # Solving for new eccentricity e_curr
    c0 = a0 / peters_factor_func(e0)
    try:
        # e decreases with time, so e_curr must be between 0 and e0
        e_curr = brentq(lambda e: GWtime(m1, m2, c0 * peters_factor_func(e), e) - t_rem_target,
                        1e-50, e0, xtol=1e-12, maxiter=50)
    except:
        # Fallback if solver fails (should be rare)
        e_curr = e0

    a_curr = c0 * peters_factor_func(e_curr)
    return a_curr, e_curr


"!!"


# ==========================================
# 3. SNR Calculation Functions (Strictly following Geometric Units)
# ==========================================

def S_gal_N2A5(f):  # LISA noise curve (Galactic Foreground)
    if f >= 1.0e-5 and f < 1.0e-3:
        return np.power(f, -2.3) * np.power(10, -44.62) * 20.0 / 3.0
    if f >= 1.0e-3 and f < np.power(10, -2.7):
        return np.power(f, -4.4) * np.power(10, -50.92) * 20.0 / 3.0
    if f >= np.power(10, -2.7) and f < np.power(10, -2.4):
        return np.power(f, -8.8) * np.power(10, -62.8) * 20.0 / 3.0
    if f >= np.power(10, -2.4) and f <= 0.01:
        return np.power(f, -20.0) * np.power(10, -89.68) * 20.0 / 3.0
    if f > 0.01 or f < 1.0e-5:
        return 0
    return 0  # Safety return


def S_n_lisa(f):  # Calculate Sn, calling the previous function
    L_param = 5.0e9  # Meter based constant from original code
    f_transfer = sciconsts.c * 0.41 / L_param / 2.0  # Transfer frequency

    # Optical Metrology & Acceleration Noise constants
    P_oms = 2.96e-23
    P_acc = 2.65e-23

    term_acc = 4.0 * (9.0e-30 / np.power(2 * pi * f, 4.0) * (1 + 1.0e-4 / f))
    term_resp = (1 + np.power(f / f_transfer, 2.0))

    Sn = 20.0 / 3.0 * term_resp * (term_acc + P_oms + P_acc) / np.power(L_param, 2.0)

    return Sn + S_gal_N2A5(f)


def SNR_analytical_geo(m1_sol, m2_sol, a_au, e, tobs_yr, Dl_kpc):
    """
    Calculate SNR using natural units.
    Input parameters are in astronomical units, converted to seconds internally.
    """
    if a_au <= 0 or e >= 1.0: return 0.0

    # 1. Unit conversion (All converted to seconds)
    m1_s = m1_sol * m_sun_sec
    m2_s = m2_sol * m_sun_sec
    a_s = a_au * AU_sec
    Dl_s = Dl_kpc * 1000.0 * pc_sec
    tobs_s = tobs_yr * year_sec

    # 2. Calculate peak frequency (Hz)
    # f0max = 2 * sqrt( (m1+m2) / (4*pi^2 * [a(1-e)]^3) )
    # In geometric units, G=1, sum masses directly
    rp_s = a_s * (1 - e)
    term_f = (m1_s + m2_s) / (4 * pi * pi * np.power(rp_s, 3.0))
    f0max = 2 * np.sqrt(term_f)

    # 4. Calculate amplitude h0max (dimensionless strain)
    # h0max = sqrt(32/5) * m1*m2 / (Dl * a * (1-e))
    # Unit check: s*s / (s*s*1) = dimensionless -> Correct
    h0max = np.sqrt(32 / 5) * m1_s * m2_s / (Dl_s * a_s * (1 - e))

    # 5. Calculate noise Sn
    Sn_val = S_n_lisa(f0max)
    if Sn_val <= 0: return 0.0

    sqrtsnf = np.sqrt(Sn_val)

    # 6. Calculate final SNR
    # snr = h0 / sqrt(Sn) * sqrt(Tobs * (1-e)^1.5)
    # Unit check: 1 / sqrt(s) * sqrt(s) = 1 -> Correct
    snrcur = h0max / sqrtsnf * np.sqrt(tobs_s * np.power(1 - e, 3 / 2))

    return snrcur


# ==========================================
# 4. Data Management Class (Integrating original logic and new snapshot functionality)
# ==========================================

class _GNBBHInternalManager:
    def __init__(self, filename_gn="evolution_history.npy", filename_ync="evolution_history_YNC.npy"):
        # Use __file__ to locate the 'data' folder relative to this script,
        # ensuring it works regardless of the current working directory.
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load GN Data (Updated to read from 'data' folder relative to this file)
        self.file_path_gn = os.path.join(current_script_dir, 'data', filename_gn)
        self.raw_data_gn = []
        self._load_data(self.file_path_gn, is_ync=False)

        # Load YNC Data (Updated to read from 'data' folder relative to this file)
        self.file_path_ync = os.path.join(current_script_dir, 'data', filename_ync)
        self.raw_data_ync = []
        self._load_data(self.file_path_ync, is_ync=True)

        self.efinal_cdf_func = None
        self.merged_indices = []

        # Build stats from GN data (as per original logic, usually statistics are derived from the main population)
        self._build_merger_statistics()

    def _load_data(self, path, is_ync=False):
        """Load the NPY file."""
        label = "YNC" if is_ync else "GN"
        if os.path.exists(path):
            print(f"[{label}_BBH] Loading data from {path}...")
            data = np.load(path, allow_pickle=True)
            if is_ync:
                self.raw_data_ync = data
            else:
                self.raw_data_gn = data
            print(f"[{label}_BBH] Loaded {len(data)} systems.")
        else:
            print(f"[Warning] File {path} not found.")

    def _build_merger_statistics(self):
        """Pre-calculate CDF for e_final of merged systems (Using GN data)."""
        if len(self.raw_data_gn) == 0: return

        # Extract e_final from systems.
        e_vals = []
        indices = []

        for i, sys_data in enumerate(self.raw_data_gn):
            # sys_data[9] is a_final, sys_data[10] is e_final
            a_fin = sys_data[9]
            e_fin = sys_data[10]

            # Filter: "merged" systems if a_final is small
            if a_fin < 1e-2:
                e_vals.append(e_fin)
                indices.append(i)

        self.merged_indices = indices

        if len(e_vals) > 0:
            e_arr = np.sort(np.array(e_vals))
            y_vals = np.arange(1, len(e_arr) + 1) / len(e_arr)

            # Create inverse CDF: U[0,1] -> e
            self.efinal_inv_cdf = sci_interpolate.interp1d(
                y_vals, e_arr, kind='linear', bounds_error=False, fill_value=(e_arr[0], e_arr[-1])
            )
            self.sorted_efinal_for_plot = e_arr
        else:
            print("[Warning] No merger systems found for statistics.")

    def get_random_mergers(self, n):
        """Return N random merged samples (full data)."""
        if len(self.merged_indices) == 0: return []
        chosen_inds = random.choices(self.merged_indices, k=n)
        return [self.raw_data_gn[i] for i in chosen_inds]

    def generate_ecc_from_cdf(self, n):
        """Generate N eccentricity samples from the interpolated CDF."""
        if self.efinal_cdf_func is None and not hasattr(self, 'efinal_inv_cdf'):
            return np.zeros(n)

        u = np.random.uniform(0, 1, n)
        return self.efinal_inv_cdf(u)

    def generate_snapshot(self, Gamma_rep, ync_age=None, ync_count=0):
        """
        Generate a snapshot of the Milky Way Galactic Nucleus.

        Args:
            Gamma_rep: GN Formation rate (systems per Myr).
            ync_age: (T) Age of the Young Nuclear Cluster (years). Default None (no YNC).
            ync_count: (N) Number of YNC systems to simulate. Default 0.

        Returns:
            a_list, e_list, m1_list, m2_list
        """
        snapshot_a = []
        snapshot_e = []
        snapshot_m1 = []
        snapshot_m2 = []

        # -----------------------------------------------
        # Part 1: Standard GN Population (Random Birth Time)
        # -----------------------------------------------
        if len(self.raw_data_gn) > 0 and Gamma_rep > 0:
            # 1. Find max lifetime to set the integration window
            lifetimes = np.array([row[8] for row in self.raw_data_gn])
            t_final_max = np.max(lifetimes)  # in years

            # 2. Total simulation window
            window_yr = t_final_max
            window_myr = window_yr / 1e6

            # 3. Number of systems to generate
            total_systems_to_gen = int(window_myr * Gamma_rep)

            print(f"[Snapshot-GN] Simulating {window_myr:.2f} Myr history with rate {Gamma_rep}/Myr.")
            print(f"[Snapshot-GN] Generating ~{total_systems_to_gen} systems...")

            # 4. Generate birth times
            birth_times = np.random.uniform(-t_final_max, 0, total_systems_to_gen)
            template_indices = np.random.randint(0, len(self.raw_data_gn), total_systems_to_gen)

            count_alive_gn = 0

            # Loop through generated population
            for i in range(total_systems_to_gen):
                sys_idx = template_indices[i]
                t_start = birth_times[i]  # negative value

                system = self.raw_data_gn[sys_idx]

                # Logic to get state
                params = self._get_system_state_at_time(system, -t_start)

                if params is not None:
                    count_alive_gn += 1
                    snapshot_a.append(params[0])
                    snapshot_e.append(params[1])
                    snapshot_m1.append(params[2])
                    snapshot_m2.append(params[3])

            print(f"[Snapshot-GN] Found {count_alive_gn} alive systems at t=0.")

        # -----------------------------------------------
        # Part 2: YNC Population (Fixed Age T, Random N Samples)
        # -----------------------------------------------
        if len(self.raw_data_ync) > 0 and ync_count > 0 and ync_age is not None:
            print(f"[Snapshot-YNC] Sampling {ync_count} systems at Age T={ync_age / 1e6:.2f} Myr...")

            # Randomly choose N indices from YNC data
            # Use random.choices allows replacement, sample does not.
            # Assuming population size allows sampling, using choices is safer if N > data_len
            ync_indices = np.random.randint(0, len(self.raw_data_ync), int(ync_count))

            count_alive_ync = 0

            for sys_idx in ync_indices:
                system = self.raw_data_ync[sys_idx]

                # Use T as the current age directly
                params = self._get_system_state_at_time(system, ync_age)

                if params is not None:
                    count_alive_ync += 1
                    snapshot_a.append(params[0])
                    snapshot_e.append(params[1])
                    snapshot_m1.append(params[2])
                    snapshot_m2.append(params[3])

            print(f"[Snapshot-YNC] Found {count_alive_ync} alive systems at Age T={ync_age}.")

        return snapshot_a, snapshot_e, snapshot_m1, snapshot_m2

    def _get_system_state_at_time(self, system, current_age):
        """
        Helper to calculate a, e for a single system at a specific age.
        Returns (a, e, m1, m2) or None if merged/invalid.
        """
        # Structure: [0:ind, 1:m1, 2:m2, 3:a1i, 4:e1i, 5:a2, 6:ii, 7:tf_pre, 8:tf_act, 9:a_fin, 10:e_fin, 11:snaps]
        m1 = system[1]
        m2 = system[2]
        tf_actual = system[8]
        snapshots = system[11]

        # Check if age exceeds lifetime (merged)
        # Note: In GN logic, age is derived from t_start (negative) -> current_age = -t_start
        # If current_age > tf_actual, it has merged.
        if current_age > tf_actual:
            return None

        # Logic to find state at current_age
        a_curr, e_curr = -1, -1

        # Case A: Snapshots exist
        if len(snapshots) > 0:
            snaps_arr = np.array(snapshots)
            times = snaps_arr[:, 0]

            # Use nearest snapshot logic or interpolation?
            # Original code logic: find closest snapshot if current_age <= last_snapshot_time
            # "Directly find the snapshot closest to T" -> implies finding closest index in time array

            if current_age <= times[-1]:
                # Find index closest to current_age
                idx = (np.abs(times - current_age)).argmin()
                a_curr = snaps_arr[idx, 1]
                e_curr = snaps_arr[idx, 2]
            else:
                # Case B: Analytical Extension (if age is between last snapshot and merger time)
                t_last = times[-1]
                a_last = snaps_arr[-1, 1]
                e_last = snaps_arr[-1, 2]
                dt = current_age - t_last
                # Ensure dt is positive and valid
                if dt > 0:
                    a_curr, e_curr = solve_ae_after_time(m1, m2, a_last, e_last, dt)
                else:
                    # Fallback to last snapshot
                    a_curr, e_curr = a_last, e_last

        # Case C: No snapshots (initial only)
        else:
            # If current_age is small relative to evolution, use initial
            # Or evolve analytically from initial
            # Original code simply returned initial if no snapshots and t_start+tf > 0.
            # But here we have explicit age. Let's stick to original logic:
            # If no snapshots, assume very slow evolution or just return initial
            a_curr = system[3]
            e_curr = system[4]

            # Refined: if age > 0, maybe evolve from initial?
            # Keeping strictly to "snapshot" logic requested:
            # "Original population follows original code... This part belongs to YNC population calculated separately"
            # For YNC: "Directly find the snapshot closest to T"
            # If no snapshot array, the only snapshot is the initial state.

        if a_curr > 0:
            return (a_curr, e_curr, m1, m2)
        return None


# ==========================================
# 5. Public API Functions (Restore all original functions)
# ==========================================

_manager = _GNBBHInternalManager()


def generate_random_merger_eccentricities(n=1000):
    """
    Function 1: Generate N random eccentricities for mergers based on the data CDF.
    """
    return _manager.generate_ecc_from_cdf(n)


def plot_ecc_cdf_log(e_list=None):
    """
    Function 2: Plot the CDF of log10(e).
    """
    if e_list is None:
        if not hasattr(_manager, 'sorted_efinal_for_plot'):
            print("No data available.")
            return
        data = _manager.sorted_efinal_for_plot
        label = "GN Mergers Samples"
    else:
        data = np.array(e_list)
        label = "Sample"

    # Filter out 0 or negative for log plot
    valid_mask = data > 1e-10
    if np.sum(valid_mask) == 0:
        print("No valid eccentricity > 0.")
        return

    e_valid = data[valid_mask]
    log_e = np.log10(e_valid)
    sorted_log_e = np.sort(log_e)
    cdf = np.arange(1, len(sorted_log_e) + 1) / len(sorted_log_e)

    plt.figure(figsize=(7, 6))
    plt.step(sorted_log_e, cdf, where='post', label=f"{label} (N={len(e_valid)})", lw=2)
    plt.xlabel(r"$\log_{10}(e)$ @10Hz", fontsize=16)
    plt.ylabel("CDF", fontsize=16)
    plt.title("Eccentricity of Merging BBHs in LIGO band", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()  # Display restored


def get_random_merger_systems(n=10):
    """
    Function 3: Randomly extract N merger systems with their parameters.
    """
    raw_sys = _manager.get_random_mergers(n)
    result = []
    for s in raw_sys:
        sys_dict = {
            "id": s[0],
            "m1": s[1], "m2": s[2],
            "a_initial": s[3], "e_initial": s[4],
            "a2_initial": s[5], "i_initial": s[6],
            "a_final": s[9], "e_final": s[10],
            "t_final_actual": s[8]
        }
        result.append(sys_dict)
    return result


def generate_and_plot_snapshot(Gamma_rep=3.0, ync_age=None, ync_count=0, title="MW Galactic Nucleus BBH Snapshot"):
    """
    Function 4 & 5: Generate Milky Way GN snapshot and plot population with SNR Map.
    Updated to accept YNC parameters.
    """
    # 1. Generate data
    a_list, e_list, m1_list, m2_list = _manager.generate_snapshot(Gamma_rep, ync_age, ync_count)

    if len(a_list) == 0:
        print("No systems found for snapshot.")
        return

    a_arr = np.array(a_list)
    e_arr = np.array(e_list)
    m1_arr = np.array(m1_list)
    m2_arr = np.array(m2_list)

    # 2. Compute SNR (Using Geometric Units)
    # Obs Settings
    Dl_kpc = 8.0
    Tobs_yr = 10.0

    snr_vals = []
    for i in range(len(a_arr)):
        # Pass Msun, AU, Year, Kpc -> Function converts to Seconds
        s = SNR_analytical_geo(m1_arr[i], m2_arr[i], a_arr[i], e_arr[i], Tobs_yr, Dl_kpc)
        snr_vals.append(s)

    snr_arr = np.array(snr_vals)
    ome_arr = 1.0 - e_arr

    # 3. Plotting Setup
    # Sort: Low SNR first, High SNR last (top)
    idx = np.argsort(snr_arr)

    a_p = a_arr[idx]
    ome_p = ome_arr[idx]
    snr_p = snr_arr[idx]

    # Custom Colormap
    my_cmap = copy.copy(mpl.colormaps['jet'])
    my_cmap.set_over('red')
    my_cmap.set_under(my_cmap(0))

    plt.figure(figsize=(8, 6))

    # Scatter
    # SNR < 1e-4 shown as base color (under), > 100 shown as red (over)
    # Size scaled by sqrt(SNR)
    sc = plt.scatter(a_p, ome_p,
                     s=np.clip(np.sqrt(snr_p) * 30, 10, 400),
                     c=snr_p,
                     cmap=my_cmap,
                     norm=mcolors.LogNorm(vmin=0.1, vmax=100),
                     alpha=1, edgecolors='k', linewidths=0.3)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r"Semi-major Axis [au]", fontsize=14)
    plt.ylabel(r"$1 - e$", fontsize=14)
    plt.title(f"{title}\nRate={Gamma_rep}/Myr, YNC_N={ync_count}, Total={len(a_arr)}", pad=15, fontsize=14)

    # Reference limits
    # plt.xlim(1e-4, 1e2)
    # plt.ylim(1e-5, 1.0)
    plt.grid(True, which="both", alpha=0.15)

    # Colorbar
    cbar = plt.colorbar(sc, extend='both', aspect=30)
    cbar.set_label(r'SNR (10yr LISA)', fontsize=12, labelpad=10)

    plt.tight_layout()
    plt.show()  # Display restored


# --- Example Usage ---
if __name__ == "__main__":

    # 1. Random Eccentricities
    print("Random Ecc:", generate_random_merger_eccentricities(3))

    # 2. Plot CDF
    # plot_ecc_cdf_log() # Uncomment to view

    # 3. Get Systems
    # print("Random System:", get_random_merger_systems(1))

    # 4. Snapshot & SNR Plot
    # Example: Add YNC data, assuming YNC age is 5 Myr (5e6 years), containing 200 systems
    # generate_and_plot_snapshot(Gamma_rep=3.0, ync_age=5.0e6, ync_count=0) # Main demo runs but won't trigger if imported
    pass