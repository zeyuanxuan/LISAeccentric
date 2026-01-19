# coding:utf-8
import numpy as np
import scipy.constants as sciconsts
import random
import math
import os
import copy
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

# ==========================================
# Import Helpers
# ==========================================
try:
    from .Field_BBH import (
        m_sun, years, pc, AU, pi,
        forb, tmerger, peters_factor_func, calculate_snr
    )
except ImportError:
    import sys

    sys.path.append(os.path.dirname(__file__))
    from Field_BBH import (
        m_sun, years, pc, AU, pi,
        forb, tmerger, peters_factor_func, calculate_snr
    )


# ==========================================
# Engine Class: Elliptical Galaxy
# ==========================================
class _Elliptical_Field_BBH_Engine:
    def __init__(self,
                 M_gal=1e11 * m_sun, Re=4.0 * 1000 * pc, distance_Mpc=10.0,
                 m1=10 * m_sun, m2=10 * m_sun,
                 formation_mod='starburst',
                 age=10e9 * years,
                 fbh=7.5e-4, mp=0.6 * m_sun, fgw=10,
                 n_sim_samples=100000, target_N=50000,
                 rrange_kpc=[0.05, 20], arange_log=[2, 4.5], blocknum=50,
                 data_dir=None, load_default=True):

        self.m1, self.m2 = m1, m2
        self.formation_mod, self.age = formation_mod, age
        self.avgage = 1e9 * years
        self.M_gal = M_gal
        self.Re = Re
        self.a_scale = self.Re / 1.8153
        self.fbh = fbh
        self.mp = mp
        self.fgw = fgw
        self.distance_Mpc = distance_Mpc
        self.Dl_fixed = distance_Mpc * 1e6 * pc
        self.rrange = [x * 1000 * pc for x in rrange_kpc]
        self.arange = arange_log
        self.blocknum, self.target_N = int(blocknum), int(target_N)

        # Sampling logic (Linear per block, no sqrt)
        samples_per_block = float(n_sim_samples) / self.blocknum
        self.radnum = max(1, int(samples_per_block))

        self.systemlist = []
        self.totalrate = 0.0
        self.is_simulated = False

        if data_dir is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(module_dir, 'data')
        else:
            self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
            except:
                pass

        self.pop_file = os.path.join(self.data_dir, 'elliptical_field_bbh_population.npy')
        self.meta_file = os.path.join(self.data_dir, 'elliptical_field_bbh_meta.npy')

        if load_default: self.load_data()

    def n(self, r):
        rho = (self.M_gal / (2 * pi)) * (self.a_scale / r) * (1 / (r + self.a_scale) ** 3)
        return rho / self.mp

    def sigma_r(self, r):
        v_disp_sq = 0.5 * self.M_gal * r / ((r + self.a_scale) ** 2)
        return np.sqrt(v_disp_sq)

    def run_simulation(self):
        total_iters = self.blocknum * self.radnum
        print(f"Running MC simulation (Elliptical Galaxy @ {self.distance_Mpc} Mpc)...")
        print(f"   -> Grid: {self.blocknum} radial bins (Log-spaced)")
        print(f"   -> Samples per bin: {self.radnum}")
        print(f"   -> TOTAL ITERATIONS: {total_iters}")

        raw_systemlist = []
        self.totalrate = 0.0

        r_edges = np.geomspace(self.rrange[0], self.rrange[1], self.blocknum + 1)
        beta = 85 / 3 * self.m1 * self.m2 * (self.m1 + self.m2)

        for i in range(self.blocknum):
            r_inner, r_outer = r_edges[i], r_edges[i + 1]
            deltar = r_outer - r_inner
            ravg = (r_inner + r_outer) / 2
            ncur = self.n(ravg)
            nbh = ncur * self.fbh * 4 * pi * ravg ** 2 * deltar
            vcur = self.sigma_r(ravg)

            submerger = 0

            for j in range(self.radnum):
                acur = np.power(10, random.random() * (self.arange[1] - self.arange[0]) + self.arange[0]) * AU
                tau = 2.33e-3 * vcur / (self.mp * ncur * acur) / 0.69315

                b = min(0.1 * vcur / forb(self.m1, self.m2, acur),
                        np.sqrt(1 / vcur * np.power(
                            27 / 4 * np.power(acur, 29 / 7) * self.mp ** 2 / (self.m1 + self.m2) * np.power(
                                ncur * pi / beta, 2 / 7), 7 / 12)))

                T = min(1 / (ncur * pi * b * b * vcur), self.age)
                acrit = np.power(4 / 27 * (self.m1 + self.m2) * np.power(beta, 2 / 7) * np.power(T, -12 / 7) / (
                        self.mp ** 2 * pi ** 2 * ncur ** 2), 7 / 29)

                try:
                    exp_term = math.exp(-self.age / tau)
                except OverflowError:
                    exp_term = 0.0

                if acur < acrit:
                    rate = ncur * self.mp * np.power(acur, 13 / 14) * np.sqrt(
                        27 / 4 * np.power(beta * T, 2 / 7) / (self.m1 + self.m2)) * exp_term
                    rate1 = tau * rate / (exp_term + 1e-50) * (1 - exp_term) * self.avgage / self.age / self.avgage
                else:
                    rate = np.power(acur, -8 / 7) * np.power(T, -5 / 7) * np.power(beta, 2 / 7) * exp_term
                    rate1 = tau * rate / (exp_term + 1e-50) * (1 - exp_term) * self.avgage / self.age / self.avgage

                final_rate = rate if self.formation_mod == 'starburst' else rate1
                submerger += final_rate * nbh / self.radnum * 1e6 * years

                ecrit = np.sqrt(max(0, 1 - np.power(beta * T / np.power(acur, 4), 2 / 7)))
                if np.isnan(ecrit): ecrit = 0
                e_initial = random.random() * (1 - ecrit) + ecrit

                a_final = np.power((self.m1 + self.m2) * np.power(2.0 / self.fgw, 2) / (4 * pi ** 2), 1.0 / 3.0)
                efinal = 0.0
                if e_initial > 1e-8:
                    val_initial = peters_factor_func(e_initial)
                    c0 = acur / val_initial
                    if a_final < acur:
                        try:
                            efinal = brentq(lambda e: c0 * peters_factor_func(e) - a_final, 1e-16, e_initial,
                                            xtol=1e-12, maxiter=100)
                        except:
                            efinal = 0.0

                Dl = self.Dl_fixed

                if final_rate > 1e-50:
                    raw_systemlist.append(
                        [acur, e_initial, efinal, Dl, final_rate, tmerger(self.m1, self.m2, acur, e_initial), tau])

            self.totalrate += submerger

        print(f"Resampling population (Total Rate: {self.totalrate:.5f}/Myr)...")
        print(f"Raw Candidates found: {len(raw_systemlist)}")

        if len(raw_systemlist) > 0:
            data = np.array(raw_systemlist)
            weights = data[:, 4]
            if np.sum(weights) > 0:
                probs = weights / np.sum(weights)
                self.systemlist = data[np.random.choice(len(data), size=self.target_N, replace=True, p=probs)]
                self.is_simulated = True
                print(f"Simulation Done. Sample Size: {len(self.systemlist)}")
            else:
                print("Warning: Sum of rates is zero.")
                self.systemlist = []
        else:
            print("Error: No systems generated (Check physics parameters).")

    def save_data(self):
        np.save(self.pop_file, self.systemlist)
        np.save(self.meta_file, {'totalrate': self.totalrate, 'distance_Mpc': self.distance_Mpc})
        print(f"Data saved to {self.data_dir}")

    def load_data(self):
        if os.path.exists(self.pop_file) and os.path.exists(self.meta_file):
            print(f"Loading Elliptical data from: {self.data_dir}")
            self.systemlist = np.load(self.pop_file, allow_pickle=True)
            meta = np.load(self.meta_file, allow_pickle=True).item()
            self.totalrate = meta.get('totalrate', 0.0)
            self.distance_Mpc = meta.get('distance_Mpc', 10.0)
            self.Dl_fixed = self.distance_Mpc * 1e6 * pc
            self.is_simulated = True
            print(f"Loaded. Rate={self.totalrate:.5f}/Myr, Dist={self.distance_Mpc} Mpc")
        else:
            print(f"No pre-generated Elliptical data found.")

    def _process_candidates(self, events, t_window_Gyr, tobs_yr):
        n_cand = len(events)
        t_future = np.random.uniform(0, t_window_Gyr * 1e9, size=n_cand) * years
        surv_prob = np.exp(-t_future / events[:, 6])
        surv_mask = np.random.random(n_cand) < surv_prob
        accepted_events = events[surv_mask]
        accepted_times = t_future[surv_mask]
        current_ages = accepted_events[:, 5] - accepted_times
        valid_mask = current_ages > 0
        final_events = accepted_events[valid_mask]
        final_ages = current_ages[valid_mask]
        output_list = []
        for i, row in enumerate(final_events):
            a0, e0 = row[0], row[1]
            dl = row[3]
            t_rem = row[5] - final_ages[i]
            if e0 < 1e-8:
                a_curr, e_curr = a0 * np.power(t_rem / row[5], 0.25), e0
            else:
                c0 = a0 / peters_factor_func(e0)
                try:
                    e_curr = brentq(lambda e: tmerger(self.m1, self.m2, c0 * peters_factor_func(e), e) - t_rem, 1e-16,
                                    e0, xtol=1e-12, maxiter=50)
                except:
                    e_curr = 0.0
                a_curr = c0 * peters_factor_func(e_curr)
            snr = calculate_snr(self.m1, self.m2, a_curr, e_curr, dl, tobs_yr * years)
            output_list.append(['Elliptical', dl / 1000 / pc, a_curr / AU, e_curr, 10.0, 10.0, snr])
        return output_list


# ==========================================
# Public Interface
# ==========================================
_GLOBAL_MODEL = None


def _get_model():
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = _Elliptical_Field_BBH_Engine(load_default=True)
    return _GLOBAL_MODEL


def simulate_and_save_default_population(m1=10 * m_sun, m2=10 * m_sun, distance_Mpc=10.0, n_sim_samples=200000,
                                         target_N=50000, **kwargs):
    global _GLOBAL_MODEL
    print("Initializing Elliptical Galaxy Simulation (Log Bins)...")
    model = _Elliptical_Field_BBH_Engine(load_default=False, m1=m1, m2=m2, distance_Mpc=distance_Mpc,
                                         n_sim_samples=n_sim_samples, target_N=target_N, **kwargs)
    model.run_simulation()
    model.save_data()
    _GLOBAL_MODEL = model
    return model


def generate_eccentricity_samples(size=10000):
    model = _get_model()
    if len(model.systemlist) == 0: return np.array([])
    indices = np.random.choice(len(model.systemlist), size=size, replace=True)
    return model.systemlist[indices, 2]


def get_merger_progenitor_population():
    model = _get_model()
    return model.systemlist


# --- REALIZATION FUNCTIONS (Matched to MW API) ---

def get_single_realization(t_window_Gyr=10.0, tobs_yr=10.0):
    """Generates one realization (Poisson sampled) of the galaxy."""
    model = _get_model()
    if len(model.systemlist) == 0: return []

    rate = model.totalrate * 1e3  # Convert rate from /Myr to /Gyr
    num = np.random.poisson(rate * t_window_Gyr)

    if num == 0: return []

    indices = np.random.choice(len(model.systemlist), size=num, replace=True)
    return model._process_candidates(model.systemlist[indices], t_window_Gyr, tobs_yr)


def get_multi_realizations(n_realizations=10, t_window_Gyr=10.0, tobs_yr=10.0):
    """Generates N stacked realizations."""
    model = _get_model()
    if len(model.systemlist) == 0: return []

    rate = model.totalrate * 1e3 * n_realizations
    num = np.random.poisson(rate * t_window_Gyr)

    if num == 0: return []

    indices = np.random.choice(len(model.systemlist), size=num, replace=True)
    return model._process_candidates(model.systemlist[indices], t_window_Gyr, tobs_yr)


def get_random_systems(n_systems=500, t_window_Gyr=10.0, tobs_yr=10.0):
    """Generates a fixed number of systems (non-Poisson)."""
    model = _get_model()
    if len(model.systemlist) == 0: return []
    output_list = []
    batch_size = n_systems * 2
    attempts = 0
    while len(output_list) < n_systems and attempts < 100:
        indices = np.random.choice(len(model.systemlist), size=batch_size, replace=True)
        batch_res = model._process_candidates(model.systemlist[indices], t_window_Gyr, tobs_yr)
        output_list.extend(batch_res)
        attempts += 1
    return output_list[:n_systems]


# --- PLOTTING FUNCTIONS ---

def plot_eccentricity_cdf(e_samples, label=None):
    if e_samples is None or len(e_samples) == 0:
        print("Error: No samples provided.")
        return
    plt.figure(figsize=(6, 5))
    sorted_e = np.sort(e_samples)
    y_vals = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
    lbl = label if label else f'Sampled (N={len(e_samples)})'
    plt.plot(np.log10(sorted_e + 1e-20), y_vals, drawstyle='steps-post', linewidth=2.0, color='#e74c3c', label=lbl)
    plt.xlabel(r"$\log_{10}(e)$", fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.title(f"Elliptical Merger Eccentricities", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_progenitor_sma_distribution(bins=50):
    model = _get_model()
    if len(model.systemlist) == 0: return
    sma_au = model.systemlist[:, 0] / AU
    plt.figure(figsize=(6, 5))
    log_bins = np.logspace(np.log10(min(sma_au)), np.log10(max(sma_au)), bins)
    plt.hist(sma_au, bins=log_bins, color='#3498db', alpha=0.7, edgecolor='black', label='Elliptical Progenitors')
    plt.xscale('log')
    plt.xlabel('Semi-Major Axis [au]', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(f'Initial SMA (N={len(model.systemlist)})', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_lifetime_cdf():
    model = _get_model()
    if len(model.systemlist) == 0: return
    lifetimes = model.systemlist[:, 5]
    lifetimes_Gyr = lifetimes / years / 1e9
    sorted_lifetimes = np.sort(lifetimes_Gyr)
    y_vals = np.arange(1, len(sorted_lifetimes) + 1) / len(sorted_lifetimes)
    plt.figure(figsize=(6, 5))
    plt.plot(sorted_lifetimes, y_vals, linewidth=2.0, color='#2ecc71', label='Elliptical Lifetime')
    plt.xscale('log')
    plt.xlabel('Merger Time (Gyr)', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.title(f'Lifetime CDF', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_snapshot(systems, title="Snapshot (Elliptical)", tobs_yr=10.0):
    if not systems:
        print("No systems to plot.")
        return
    data = np.array(systems)[:, 1:].astype(float)
    a = data[:, 1]
    e = data[:, 2]
    snr = data[:, 5]

    if len(a) == 0: return

    idx = np.argsort(snr)[::-1]
    a_p, ome_p, snr_p = a[idx], 1.0 - e[idx], snr[idx]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(a_p, ome_p,
                     s=np.clip(np.sqrt(snr_p) * 20, 5, 200),
                     c=np.clip(snr_p, 1e-3, None),
                     cmap=copy.copy(mpl.colormaps['jet']),
                     norm=mcolors.LogNorm(vmin=0.1, vmax=200),
                     edgecolors='k', linewidths=0.5)

    model = _get_model()
    x_min_val = min(0.001, np.min(a_p))
    x_max_val = max(4e4, np.max(a_p))

    a_grid = np.logspace(np.log10(x_min_val), np.log10(x_max_val), 500) * AU
    K = (768 / 425) / (4 * (64 / 5 * model.m1 * model.m2 * (model.m1 + model.m2)))

    added_legend = False
    for tyr, lbl in zip([1e10, 1e8, 1e6, 1e4], ['10Gyr', '0.1Gyr', '1Myr', '10kyr']):
        val = np.power(tyr * years / (K * a_grid ** 4), 2 / 7)
        if np.any(val <= 1.0):
            lbl_text = "Merger Timescale" if not added_legend else "_nolegend_"
            valid_mask = val <= 1.0
            x_line = a_grid[valid_mask] / AU
            y_line = 1 - np.sqrt(1 - val[valid_mask])
            plt.plot(x_line, y_line, '--', color='gray', alpha=0.5, label=lbl_text)

            plt.text(x_line[-1], y_line[-1], lbl, fontsize=10, color='dimgray', ha='left', va='center')
            added_legend = True

    plt.xscale('log')
    plt.yscale('log')

    # --- MODIFIED: Extend X-axis by 10% for labels ---
    log_min = np.log10(x_min_val)
    log_max = np.log10(x_max_val)
    log_span = log_max - log_min
    new_log_max = log_max + (log_span * 0.10)  # Add 10% buffer
    plt.xlim(10 ** log_min, 10 ** new_log_max)

    plt.xlabel('Semi-major Axis [au]', fontsize=16)
    plt.ylabel('1 - e', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    cbar = plt.colorbar(sc)
    cbar.set_label(f'SNR ({tobs_yr}yr LISA)', rotation=270, labelpad=20, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.title(f"{title} (N={len(systems)})", fontsize=13)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, which='both', ls='-', alpha=0.15)
    plt.tight_layout()
    plt.show()