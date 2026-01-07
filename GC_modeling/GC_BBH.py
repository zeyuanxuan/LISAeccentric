# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import scipy.interpolate as sci_interpolate
import scipy.constants as sciconsts
import copy
import random
import os


class _GCBBHInternalManager:
    def __init__(self, ecc_name="GC_10Hz_eccentricity.csv", pop_name="Mock_GC_BBHs.csv"):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.ecc_file = os.path.join(current_dir, "data", ecc_name)
        self.pop_file = os.path.join(current_dir, "data", pop_name)

        # Data storage
        self.channels = []
        self.efuncdict = {}
        self.colmap = {}
        self.weight_dict = {}
        self.loge_grid = np.linspace(-10, 0, 1000)
        self.merged_incluster_cdf = None
        self.full_population_data = []  # Stores all 10 realizations

        # Trigger automatic loading
        self._load_all_data()

    def _load_all_data(self):
        """Load all CSV data upon module import."""
        # 1. Load 10Hz Eccentricity CDFs (LIGO Band)
        if os.path.exists(self.ecc_file):
            raw_ecc = pd.read_csv(self.ecc_file, header=None)
            channel_row = raw_ecc.iloc[0].tolist()
            self.raw_ecc_df = raw_ecc.iloc[2:].reset_index(drop=True)

            i = 0
            while i < len(channel_row) - 1:
                ch_name = str(channel_row[i]).strip()
                if ch_name != "" and ch_name != "nan":
                    self.channels.append(ch_name)
                    self.colmap[ch_name] = {"log(e)": i, "CDF": i + 1}
                    try:
                        self.weight_dict[ch_name] = float(str(channel_row[i + 1]).split('=')[1])
                    except:
                        self.weight_dict[ch_name] = 0.0
                    i += 2
                else:
                    i += 1

            for ch in self.channels:
                le = pd.to_numeric(self.raw_ecc_df.iloc[:, self.colmap[ch]["log(e)"]], errors='coerce').to_numpy()
                ce = pd.to_numeric(self.raw_ecc_df.iloc[:, self.colmap[ch]["CDF"]], errors='coerce').to_numpy()
                mask = np.isfinite(le) & np.isfinite(ce)
                if np.any(mask):
                    idx = np.argsort(le[mask])
                    self.efuncdict[ch] = sci_interpolate.interp1d(
                        le[mask][idx], ce[mask][idx], kind="linear", bounds_error=False, fill_value=(0.0, 1.0)
                    )

            # Weighted merge for In-cluster channels
            in_names = [c for c in self.channels if c.lower() != "ejected"]
            total_n = sum(self.weight_dict[c] for c in in_names)
            self.merged_incluster_cdf = np.zeros_like(self.loge_grid)
            for c in in_names:
                if c in self.efuncdict:
                    self.merged_incluster_cdf += (self.weight_dict[c] / total_n) * self.efuncdict[c](self.loge_grid)

        # 2. Load MW Globular Cluster BBH snapshot
        if os.path.exists(self.pop_file):
            # Read CSV and drop potential empty rows to keep data clean
            self.full_population_data = pd.read_csv(self.pop_file).dropna().values.tolist()

    def generate_ecc_samples_10Hz(self, channel_name, size):
        """Generate random samples from the 10Hz eccentricity distributions."""
        is_in = (channel_name.lower() == 'incluster')

        if not is_in and channel_name not in self.efuncdict:
            main_cats = ['Incluster', 'Ejected']
            sub_cats = [c for c in self.channels if c != 'Ejected']
            m_str = ", ".join([f"'{c}'" for c in main_cats])
            s_str = ", ".join([f"'{c}'" for c in sub_cats])
            raise ValueError(
                f"\n[Error] Invalid channel name: '{channel_name}'\n"
                f"Main Categories: {m_str}\n"
                f"Sub-categories:  {s_str}"
            )

        u = np.random.uniform(0, 1, size)
        if is_in:
            interp = sci_interpolate.interp1d(self.merged_incluster_cdf, self.loge_grid,
                                              kind='linear', bounds_error=False, fill_value=(-10, 0))
        else:
            le_raw = pd.to_numeric(self.raw_ecc_df.iloc[:, self.colmap[channel_name]["log(e)"]],
                                   errors='coerce').to_numpy()
            ce_raw = pd.to_numeric(self.raw_ecc_df.iloc[:, self.colmap[channel_name]["CDF"]],
                                   errors='coerce').to_numpy()
            mask = np.isfinite(le_raw) & np.isfinite(ce_raw)
            interp = sci_interpolate.interp1d(ce_raw[mask], le_raw[mask], kind='linear',
                                              bounds_error=False, fill_value="extrapolate")
        return np.power(10.0, interp(u))


# --- Public API Functions ---

def generate_ecc_samples_10Hz(channel_name, size=1000):
    return _manager.generate_ecc_samples_10Hz(channel_name, size)


def plot_ecc_cdf(e_list, label="Sample"):
    e_arr = np.array(e_list, dtype=float)
    valid = e_arr[e_arr > 0]
    log_e_sorted = np.sort(np.log10(valid))
    cdf = np.arange(1, len(log_e_sorted) + 1) / len(log_e_sorted)
    plt.step(log_e_sorted, cdf, where='post', label=f"{label} (N={len(valid)})")
    plt.xlabel(r"$\log_{10}(e)$ @ 10Hz");
    plt.ylabel("CDF")
    plt.xlim(-10, 0);
    plt.ylim(0, 1.05);
    plt.title(f"Eccentricity of merging BBHs in LIGO band", pad=15, fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

def get_full_10_realizations():
    return _manager.full_population_data


def get_single_mw_realization():
    n = len(_manager.full_population_data) // 10
    return random.sample(_manager.full_population_data, n)


def get_random_systems(n):
    if n <= len(_manager.full_population_data):
        return random.sample(_manager.full_population_data, n)
    else:
        return random.choices(_manager.full_population_data, k=n)


def plot_mw_gc_bbh_snapshot(systems=None, title="MW Globular Cluster BBH Snapshot"):
    """
    Scatter plot with SNR color mapping. Skips the first string column for numeric safety.
    """
    if systems is None:
        systems = get_single_mw_realization()

    # Skip the first column (location) to ensure the rest is treated as a float array
    # This prevents the 'UFuncNoLoopError' caused by mixed types
    data_np = np.array(systems)[:, 1:].astype(float)

    # After skipping col 0, new indices are:
    # 0: distance(kpc), 1: SMA(au), 2: eccentricity, 5: SNR(10yr)
    a = data_np[:, 1]
    e = data_np[:, 2]
    snr = data_np[:, 5]

    # Z-ordering by SNR (low SNR points on top)
    idx = np.argsort(snr)[::-1]
    a_p, ome_p, snr_p = a[idx], 1.0 - e[idx], snr[idx]

    # Visual Setup
    my_cmap = copy.copy(mpl.colormaps['jet'])
    my_cmap.set_over('red')
    my_cmap.set_under(my_cmap(0))
    my_cmap.set_bad(my_cmap(0))

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(a_p, ome_p, s=np.clip(np.sqrt(snr_p) * 50, 10, 600),
                     c=np.clip(snr_p, 1e-3, None), cmap=my_cmap,
                     norm=mcolors.LogNorm(vmin=0.1, vmax=200),
                     alpha=1, edgecolors='black', linewidths=0.5)

    plt.xscale('log');
    plt.yscale('log')
    plt.xlabel(r"Semi-major Axis $a$ [au]", fontsize=13)
    plt.ylabel(r"$1-e$", fontsize=13)
    plt.xlim(0.001, 4e4);
    plt.ylim(0.0008, 1)

    cbar = plt.colorbar(sc, extend='both', aspect=25)
    if hasattr(cbar, 'solids'): cbar.solids.set_alpha(1)
    cbar.set_label('SNR (10yr LISA)', fontsize=11, labelpad=10)

    plt.title(f"{title} ($N_{{plot}}$={len(systems)})", pad=15, fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.tight_layout()
    plt.show()


# Instantiate globally for auto-loading
_manager = _GCBBHInternalManager()