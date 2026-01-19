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
# 1. Physical Constants Definition
# ==========================================
pi = sciconsts.pi
m_sun_sec = 1.9891e30 * sciconsts.G / np.power(sciconsts.c, 3.0)
AU_sec = sciconsts.au / sciconsts.c
pc_sec = 3.261 * sciconsts.light_year / sciconsts.c
year_sec = 365 * 24 * 3600.0
day_sec = 24 * 3600.0


# ==========================================
# 2. Evolutionary Physics Functions
# ==========================================

def GWtime(m1, m2, a1, e1):
    if e1 >= 1.0 or a1 <= 0: return 0.0
    factor = 1.6e13
    return factor * (2 / m1 / m2 / (m1 + m2)) * np.power(a1 / 0.1, 4.0) * np.power(1 - e1 * e1, 7 / 2)


def peters_factor_func(e):
    if e < 1e-10: return 0.0
    term1 = np.power(e, 12.0 / 19.0)
    term2 = 1.0 - e * e
    term3 = np.power(1.0 + (121.0 / 304.0) * e * e, 870.0 / 2299.0)
    return (term1 / term2) * term3


def solve_ae_after_time(m1, m2, a0, e0, dt):
    current_life = GWtime(m1, m2, a0, e0)
    if dt >= current_life:
        return 0.0, 0.0
    t_rem_target = current_life - dt
    c0 = a0 / peters_factor_func(e0)
    try:
        e_curr = brentq(lambda e: GWtime(m1, m2, c0 * peters_factor_func(e), e) - t_rem_target,
                        1e-50, e0, xtol=1e-12, maxiter=50)
    except:
        e_curr = e0
    a_curr = c0 * peters_factor_func(e_curr)
    return a_curr, e_curr


# ==========================================
# 3. SNR Calculation Functions
# ==========================================
def S_gal_N2A5(f):
    if f >= 1.0e-5 and f < 1.0e-3: return np.power(f, -2.3) * np.power(10, -44.62) * 20.0 / 3.0
    if f >= 1.0e-3 and f < np.power(10, -2.7): return np.power(f, -4.4) * np.power(10, -50.92) * 20.0 / 3.0
    if f >= np.power(10, -2.7) and f < np.power(10, -2.4): return np.power(f, -8.8) * np.power(10, -62.8) * 20.0 / 3.0
    if f >= np.power(10, -2.4) and f <= 0.01: return np.power(f, -20.0) * np.power(10, -89.68) * 20.0 / 3.0
    return 0


def S_n_lisa(f):
    L_param = 5.0e9
    f_transfer = sciconsts.c * 0.41 / L_param / 2.0
    P_oms = 2.96e-23
    P_acc = 2.65e-23
    term_acc = 4.0 * (9.0e-30 / np.power(2 * pi * f, 4.0) * (1 + 1.0e-4 / f))
    term_resp = (1 + np.power(f / f_transfer, 2.0))
    Sn = 20.0 / 3.0 * term_resp * (term_acc + P_oms + P_acc) / np.power(L_param, 2.0)
    return Sn + S_gal_N2A5(f)


def SNR_analytical_geo(m1_sol, m2_sol, a_au, e, tobs_yr, Dl_kpc):
    if a_au <= 0 or e >= 1.0: return 0.0
    m1_s = m1_sol * m_sun_sec
    m2_s = m2_sol * m_sun_sec
    a_s = a_au * AU_sec
    Dl_s = Dl_kpc * 1000.0 * pc_sec
    tobs_s = tobs_yr * year_sec
    rp_s = a_s * (1 - e)
    term_f = (m1_s + m2_s) / (4 * pi * pi * np.power(rp_s, 3.0))
    f0max = 2 * np.sqrt(term_f)
    h0max = np.sqrt(32 / 5) * m1_s * m2_s / (Dl_s * a_s * (1 - e))
    Sn_val = S_n_lisa(f0max)
    if Sn_val <= 0: return 0.0
    sqrtsnf = np.sqrt(Sn_val)
    snrcur = h0max / sqrtsnf * np.sqrt(tobs_s * np.power(1 - e, 3 / 2))
    return snrcur


# ==========================================
# 4. Data Management Class (Lazy Loading & Filtering)
# ==========================================

class _GNBBHInternalManager:
    def __init__(self, filename_gn="evolution_history.npy", filename_ync="evolution_history_YNC.npy"):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path_gn = os.path.join(current_script_dir, 'data', filename_gn)
        self.file_path_ync = os.path.join(current_script_dir, 'data', filename_ync)

        # Lazy Loading: Data is None initially
        self.raw_data_gn = None
        self.raw_data_ync = None

        # Filtering State
        self.current_max_mass = 100.0  # Default Threshold

        self.efinal_inv_cdf = None
        self.sorted_efinal_for_plot = None
        self.merged_indices = []

    def check_and_update_threshold(self, new_max_mass):
        """
        Public functions call this first.
        If the requested mass threshold differs from what is currently loaded,
        clear the cache to force a reload and re-filter.
        """
        if abs(self.current_max_mass - new_max_mass) > 1e-6:
            # print(f"[Manager] Threshold changed ({self.current_max_mass} -> {new_max_mass}). Reloading data...")
            self.current_max_mass = new_max_mass
            self.raw_data_gn = None
            self.raw_data_ync = None
            self.merged_indices = []
            self.efinal_inv_cdf = None

    def _ensure_gn_loaded(self):
        if self.raw_data_gn is None:
            self._load_data(self.file_path_gn, is_ync=False)
            self._build_merger_statistics()

    def _ensure_ync_loaded(self):
        if self.raw_data_ync is None:
            self._load_data(self.file_path_ync, is_ync=True)

    def _load_data(self, path, is_ync=False):
        """
        Loads data and IMMEDIATELY filters out systems where m1 or m2 > self.current_max_mass.
        """
        label = "YNC" if is_ync else "GN"
        if os.path.exists(path):
            # 1. Load raw data
            data = np.load(path, allow_pickle=True)
            original_len = len(data)

            # 2. Filter data based on current_max_mass
            # Columns: 1 is m1, 2 is m2
            if len(data) > 0:
                mask = (data[:, 1] <= self.current_max_mass) & (data[:, 2] <= self.current_max_mass)
                data = data[mask]

            # print(f"[{label}_BBH] Loaded and filtered (Max M={self.current_max_mass}): {len(data)}/{original_len} systems.")

            if is_ync:
                self.raw_data_ync = data
            else:
                self.raw_data_gn = data
        else:
            print(f"[Warning] File {path} not found.")
            if is_ync:
                self.raw_data_ync = []
            else:
                self.raw_data_gn = []

    def _build_merger_statistics(self):
        # Operates on already filtered self.raw_data_gn
        if len(self.raw_data_gn) == 0: return
        e_vals = []
        indices = []
        for i, sys_data in enumerate(self.raw_data_gn):
            a_fin = sys_data[9]
            e_fin = sys_data[10]
            if a_fin < 1e-2:
                e_vals.append(e_fin)
                indices.append(i)
        self.merged_indices = indices
        if len(e_vals) > 0:
            e_arr = np.sort(np.array(e_vals))
            y_vals = np.arange(1, len(e_arr) + 1) / len(e_arr)
            self.efinal_inv_cdf = sci_interpolate.interp1d(
                y_vals, e_arr, kind='linear', bounds_error=False, fill_value=(e_arr[0], e_arr[-1])
            )
            self.sorted_efinal_for_plot = e_arr

    def get_random_mergers(self, n):
        self._ensure_gn_loaded()
        if len(self.merged_indices) == 0: return []
        chosen_inds = random.choices(self.merged_indices, k=n)
        return [self.raw_data_gn[i] for i in chosen_inds]

    def generate_ecc_from_cdf(self, n):
        self._ensure_gn_loaded()
        if self.efinal_inv_cdf is None: return np.zeros(n)
        u = np.random.uniform(0, 1, n)
        return self.efinal_inv_cdf(u)

    def generate_snapshot_objects(self, Gamma_rep, ync_age=None, ync_count=0):
        """
        Logic remains identical to original source code.
        Data is already filtered at load time.
        """
        # Ensure data is loaded (filtered by current threshold)
        if Gamma_rep > 0: self._ensure_gn_loaded()
        if ync_count > 0: self._ensure_ync_loaded()

        mwGNsnapshot = []
        dist_kpc = 8.0
        Tobs_yr = 10.0

        # --- Helper to process system ---
        def process_and_add(system, current_age, label):
            res = self._get_system_state_at_time(system, current_age)
            if res is not None:
                # res is (a, e, m1, m2)
                a, e, m1, m2 = res
                snr = SNR_analytical_geo(m1, m2, a, e, Tobs_yr, dist_kpc)
                mwGNsnapshot.append([label, dist_kpc, a, e, m1, m2, snr])

        # --- 1. GN Steady State ---
        if self.raw_data_gn is not None and len(self.raw_data_gn) > 0 and Gamma_rep > 0:
            lifetimes = np.array([row[8] for row in self.raw_data_gn])
            t_final_max = np.max(lifetimes)
            window_myr = t_final_max / 1e6
            total_systems_to_gen = int(window_myr * Gamma_rep)

            birth_times = np.random.uniform(-t_final_max, 0, total_systems_to_gen)
            template_indices = np.random.randint(0, len(self.raw_data_gn), total_systems_to_gen)

            for i in range(total_systems_to_gen):
                sys_idx = template_indices[i]
                t_start = birth_times[i]
                process_and_add(self.raw_data_gn[sys_idx], -t_start, 'GN_Steadystate')

        # --- 2. YNC ---
        if self.raw_data_ync is not None and len(self.raw_data_ync) > 0 and ync_count > 0 and ync_age is not None:
            ync_indices = np.random.randint(0, len(self.raw_data_ync), int(ync_count))
            for sys_idx in ync_indices:
                process_and_add(self.raw_data_ync[sys_idx], ync_age, 'GN_YNC')

        return mwGNsnapshot

    def _get_system_state_at_time(self, system, current_age):
        m1, m2, tf_actual = system[1], system[2], system[8]
        snapshots = system[11]

        if current_age > tf_actual: return None

        a_curr, e_curr = -1, -1
        if len(snapshots) > 0:
            snaps_arr = np.array(snapshots)
            times = snaps_arr[:, 0]
            if current_age <= times[-1]:
                idx = (np.abs(times - current_age)).argmin()
                a_curr, e_curr = snaps_arr[idx, 1], snaps_arr[idx, 2]
            else:
                t_last, a_last, e_last = times[-1], snaps_arr[-1, 1], snaps_arr[-1, 2]
                dt = current_age - t_last
                if dt > 0:

                    a_curr, e_curr = solve_ae_after_time(m1, m2, a_last, e_last, dt)
                else:
                    a_curr, e_curr = a_last, e_last
        else:
            a_curr, e_curr = system[3], system[4]

        if a_curr > 0: return (a_curr, e_curr, m1, m2)
        return None


_manager = _GNBBHInternalManager()


# ==========================================
# 5. Public API Functions
# ==========================================

def generate_random_merger_eccentricities(n=1000, max_bh_mass=100.0):
    """
    Returns random merged eccentricities.
    Filter: m1, m2 <= max_bh_mass (default 100).
    """
    _manager.check_and_update_threshold(max_bh_mass)
    return _manager.generate_ecc_from_cdf(n)


def plot_ecc_cdf_log(e_list=None, max_bh_mass=100.0):
    """
    Plots CDF of log(e).
    If e_list is None, loads data using max_bh_mass filter.
    """
    if e_list is None:
        _manager.check_and_update_threshold(max_bh_mass)
        _manager._ensure_gn_loaded()
        if not hasattr(_manager, 'sorted_efinal_for_plot'): return
        data = _manager.sorted_efinal_for_plot
        label = f"GN Mergers (M<{max_bh_mass})"
    else:
        data = np.array(e_list)
        label = "Sample"

    valid_mask = data > 1e-50
    if np.sum(valid_mask) == 0: return
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
    plt.show()


def get_random_merger_systems(n=10, max_bh_mass=100.0):
    """
    Returns N random merger systems as a list of parameters.
    Filter: m1, m2 <= max_bh_mass (default 100).
    Format: [m1, m2, ai, ei, i_i, a2, afinal, efinal, t_final]
    """
    _manager.check_and_update_threshold(max_bh_mass)
    raw_sys = _manager.get_random_mergers(n)
    result = []
    for s in raw_sys:
        # s indices: 1:m1, 2:m2, 3:ai, 4:ei, 6:ii, 5:a2, 9:afin, 10:efin, 8:t_final
        sys_list = [s[1], s[2], s[3], s[4], s[6], s[5], s[9], s[10], s[8]]
        result.append(sys_list)
    return result


def generate_snapshot_population(Gamma_rep=3.0, ync_age=None, ync_count=0, max_bh_mass=100.0):
    """
    Generates the snapshot object list.
    Filter: m1, m2 <= max_bh_mass (default 100) applied at load time.
    Returns: list of [label, dist, a, e, m1, m2, snr]
    """
    _manager.check_and_update_threshold(max_bh_mass)
    return _manager.generate_snapshot_objects(Gamma_rep, ync_age, ync_count)


def plot_snapshot_population(mwGNsnapshot, title="MW Galactic Nucleus BBH Snapshot"):
    """
    Plots the snapshot population list.
    mwGNsnapshot: list of [label, dist, a, e, m1, m2, snr]
    """
    if not mwGNsnapshot:
        print("Empty snapshot.")
        return

    # Extract data columns
    data = np.array(mwGNsnapshot, dtype=object)
    # col indices: 2:a, 3:e, 6:snr
    a_arr = data[:, 2].astype(float)
    e_arr = data[:, 3].astype(float)
    snr_arr = data[:, 6].astype(float)
    ome_arr = 1.0 - e_arr

    # Sort for plotting (High SNR on top)
    idx = np.argsort(snr_arr)
    a_p = a_arr[idx]
    ome_p = ome_arr[idx]
    snr_p = snr_arr[idx]

    my_cmap = copy.copy(mpl.colormaps['jet'])
    my_cmap.set_over('red')
    my_cmap.set_under(my_cmap(0))

    plt.figure(figsize=(8, 6))
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
    plt.title(f"{title}\nTotal Systems: {len(mwGNsnapshot)}", pad=15, fontsize=14)
    plt.grid(True, which="both", alpha=0.15)

    cbar = plt.colorbar(sc, extend='both', aspect=30)
    cbar.set_label(r'SNR (10yr LISA)', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.show()


def generate_and_plot_snapshot(Gamma_rep=3.0, ync_age=None, ync_count=0, max_bh_mass=100.0, title="MW Snapshot"):
    """Wrapper to generate and plot in one go."""
    snap = generate_snapshot_population(Gamma_rep, ync_age, ync_count, max_bh_mass=max_bh_mass)
    plot_snapshot_population(snap, title)