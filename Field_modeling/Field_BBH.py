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
# Global Constants & Helpers
# ==========================================
m_sun = 1.9891e30 * sciconsts.G / np.power(sciconsts.c, 3.0)
pi = sciconsts.pi
years = 365 * 24 * 3600.0
pc = 3.261 * sciconsts.light_year / sciconsts.c
AU = sciconsts.au / sciconsts.c


def forb(m1, m2, a):
    return 1 / 2 / pi * np.sqrt(m1 + m2) * np.power(a, -3.0 / 2.0)


def tmerger(m1, m2, a, e):
    beta = 64 / 5 * m1 * m2 * (m1 + m2)
    tc = np.power(a, 4) / (4 * beta)
    t = 768 / 425 * tc * np.power(1 - e * e, 7 / 2)
    return t


def peters_factor_func(e):
    if e <= 1e-16: return 0.0
    if e >= 1.0: return float('inf')
    term1 = np.power(e, 12.0 / 19.0)
    term2 = np.power(1 + (121.0 / 304.0) * e * e, 870.0 / 2299.0)
    term3 = 1 - e * e
    return term1 * term2 / term3


# --- SNR Functions ---
def S_gal_N2A5(f):
    if f >= 1.0e-5 and f < 1.0e-3: return np.power(f, -2.3) * np.power(10, -44.62) * 20.0 / 3.0
    if f >= 1.0e-3 and f < np.power(10, -2.7): return np.power(f, -4.4) * np.power(10, -50.92) * 20.0 / 3.0
    if f >= np.power(10, -2.7) and f < np.power(10, -2.4): return np.power(f, -8.8) * np.power(10, -62.8) * 20.0 / 3.0
    if f >= np.power(10, -2.4) and f <= 0.01: return np.power(f, -20.0) * np.power(10, -89.68) * 20.0 / 3.0
    return 0


def S_n_lisa(f):
    m1 = 5.0e9
    m2 = sciconsts.c * 0.41 / m1 / 2.0
    return 20.0 / 3.0 * (1 + np.power(f / m2, 2.0)) * (4.0 * (
            9.0e-30 / np.power(2 * sciconsts.pi * f, 4.0) * (1 + 1.0e-4 / f)) + 2.96e-23 + 2.65e-23) / np.power(m1,
                                                                                                                2.0) + S_gal_N2A5(
        f)


def calculate_snr(m1, m2, a, e, Dl, tobs):
    h0max = np.sqrt(32 / 5) * m1 * m2 / (Dl * a * (1 - e))
    f0max = 2 * np.sqrt((m1 + m2) / (4 * pi * pi * np.power(a * (1 - e), 3.0)))
    if f0max <= 1e-6 or f0max > 1.0: return 0.0
    sqrtsnf = np.sqrt(S_n_lisa(f0max))
    treal = tobs#max(tobs, 1 / forb(m1, m2, a))
    return h0max / sqrtsnf * np.sqrt(treal * np.power(1 - e, 3 / 2))


# ==========================================
# Internal Class (Hidden Implementation)
# ==========================================
class _MW_Field_BBH_Engine:
    def __init__(self, m1=10 * m_sun, m2=10 * m_sun, formation_mod='starburst',
                 age=10e9 * years, n0=0.1 / (np.power(pc, 3)), rsun=8e3 * pc,
                 Rl=2.6e3 * pc, h=1e3 * pc, sigmav=50e3 / sciconsts.c, fbh=7.5e-4,
                 n_sim_samples=100000, target_N=50000,
                 rrange_kpc=[0.5, 15], arange_log=[2, 4.5], blocknum=29,
                 data_dir=None, load_default=True):

        self.m1, self.m2 = m1, m2
        self.formation_mod, self.age = formation_mod, age
        self.avgage = 1e9 * years
        self.n0, self.rsun, self.Rl, self.h = n0, rsun, Rl, h
        self.sigmav, self.fbh = sigmav, fbh
        self.mp = 0.6 * m_sun
        self.fgw = 10
        self.rrange = [x * 1000 * pc for x in rrange_kpc]
        self.arange = arange_log
        self.blocknum, self.target_N = int(blocknum), int(target_N)

        samples_per_block = n_sim_samples / self.blocknum
        self.radnum = max(1, int(np.sqrt(samples_per_block)))
        self.radnum1 = self.radnum2 = self.radnum

        self.systemlist = []
        self.totalrate = 0.0
        self.is_simulated = False

        # --- PATH HANDLING ---
        if data_dir is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(module_dir, 'data')
        else:
            self.data_dir = data_dir

        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
                print(f"Created data directory at: {self.data_dir}")
            except OSError as e:
                print(f"Error creating data directory: {e}")

        self.pop_file = os.path.join(self.data_dir, 'mw_field_bbh_population.npy')
        self.meta_file = os.path.join(self.data_dir, 'mw_field_bbh_meta.npy')

        if load_default: self.load_data()

    def n(self, r):
        return self.n0 * math.exp(-1 * (r - self.rsun) / self.Rl)

    def run_simulation(self):
        print(f"Running simulation ({self.blocknum} blocks x {self.radnum}^2)...")
        raw_systemlist = []
        self.totalrate = 0.0
        deltar = (self.rrange[1] - self.rrange[0]) / self.blocknum
        beta = 85 / 3 * self.m1 * self.m2 * (self.m1 + self.m2)

        for i in range(self.blocknum):
            r1 = self.rrange[0] + i * deltar
            ravg = (r1 + r1 + deltar) / 2
            ncur = self.n(ravg)
            nbh = ncur * self.fbh * 2 * pi * ravg * self.h * deltar

            submerger, submerger1 = 0, 0
            for j in range(self.radnum):
                acur = np.power(10, random.random() * (self.arange[1] - self.arange[0]) + self.arange[0]) * AU
                tau = 2.33e-3 * self.sigmav / (self.mp * ncur * acur) / 0.69315
                b = min(0.1 * self.sigmav / forb(self.m1, self.m2, acur),
                        np.sqrt(1 / self.sigmav * np.power(
                            27 / 4 * np.power(acur, 29 / 7) * self.mp ** 2 / (self.m1 + self.m2) * np.power(
                                ncur * pi / beta, 2 / 7), 7 / 12)))
                T = min(1 / (ncur * pi * b * b * self.sigmav), self.age)
                acrit = np.power(4 / 27 * (self.m1 + self.m2) * np.power(beta, 2 / 7) * np.power(T, -12 / 7) / (
                            self.mp ** 2 * pi ** 2 * ncur ** 2), 7 / 29)

                if acur < acrit:
                    rate = ncur * self.mp * np.power(acur, 13 / 14) * np.sqrt(
                        27 / 4 * np.power(beta * T, 2 / 7) / (self.m1 + self.m2)) * math.exp(-self.age / tau)
                    rate1 = tau * rate / math.exp(-self.age / tau) * (
                                1 - math.exp(-self.age / tau)) * self.avgage / self.age / self.avgage
                else:
                    rate = np.power(acur, -8 / 7) * np.power(T, -5 / 7) * np.power(beta, 2 / 7) * math.exp(
                        -self.age / tau)
                    rate1 = tau * rate / math.exp(-self.age / tau) * (
                                1 - math.exp(-self.age / tau)) * self.avgage / self.age / self.avgage

                submerger += rate * nbh / self.radnum * 1e6 * years
                submerger1 += rate1 * nbh / self.radnum * 1e6 * years

                ecrit = np.sqrt(max(0, 1 - np.power(beta * T / np.power(acur, 4), 2 / 7)))
                if np.isnan(ecrit): ecrit = 0

                for k in range(self.radnum):
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
                            except ValueError:
                                efinal = 0.0

                    Rcur, phi, cosi = ravg / 1000 / pc, 2 * pi * random.random(), 0
                    Dl = np.sqrt((Rcur * np.sqrt(1 - cosi ** 2) * np.sin(phi)) ** 2 + (Rcur * cosi) ** 2 + (
                                Rcur * np.sqrt(1 - cosi ** 2) * np.cos(phi) - 8) ** 2) * 1000 * pc
                    lifetime = tmerger(self.m1, self.m2, acur, e_initial)

                    final_rate = rate if self.formation_mod == 'starburst' else rate1
                    raw_systemlist.append([acur, e_initial, efinal, Dl, final_rate, lifetime, tau])

            self.totalrate += submerger if self.formation_mod == 'starburst' else submerger1

        print("Resampling population based on merger rates...")
        if len(raw_systemlist) > 0:
            data = np.array(raw_systemlist)
            weights = data[:, 4]
            probs = weights / np.sum(weights)
            self.systemlist = data[np.random.choice(len(data), size=self.target_N, replace=True, p=probs)]
            self.is_simulated = True
            print(f"Simulation Done. Rate: {self.totalrate:.5f} /Myr. Population: {len(self.systemlist)}")
        else:
            print("Error: No systems generated.")

    def save_data(self):
        np.save(self.pop_file, self.systemlist)
        np.save(self.meta_file, {'totalrate': self.totalrate})
        print(f"Data saved to {self.data_dir}")

    def load_data(self):
        if os.path.exists(self.pop_file) and os.path.exists(self.meta_file):
            print(f"Loading data from: {self.data_dir}")
            self.systemlist = np.load(self.pop_file, allow_pickle=True)
            self.totalrate = np.load(self.meta_file, allow_pickle=True).item()['totalrate']
            self.is_simulated = True
            print(f"Loaded default data. Population N={len(self.systemlist)}, Rate={self.totalrate:.5f}/Myr")
        else:
            print(f"No pre-generated data found in {self.data_dir}.")


# ==========================================
# Module Level Interface (The API)
# ==========================================
_GLOBAL_MODEL = None


def _get_model():
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = _MW_Field_BBH_Engine(load_default=True)
        if not _GLOBAL_MODEL.is_simulated:
            raise RuntimeError("Default data not found. Please run 'simulate_and_save_default_population()' first.")
    return _GLOBAL_MODEL


def simulate_and_save_default_population(n_sim_samples=100000, target_N=50000, **kwargs):
    global _GLOBAL_MODEL
    print("Initializing fresh simulation...")
    model = _MW_Field_BBH_Engine(load_default=False, n_sim_samples=n_sim_samples, target_N=target_N, **kwargs)
    model.run_simulation()
    model.save_data()
    _GLOBAL_MODEL = model
    return model


def generate_eccentricity_samples(size=10000):
    model = _get_model()
    if len(model.systemlist) == 0: return np.array([])
    indices = np.random.choice(len(model.systemlist), size=size, replace=True)
    return model.systemlist[indices, 2]  # Index 2 is e_final


def plot_eccentricity_cdf(e_samples=None, label=None):
    model = _get_model()
    plt.figure(figsize=(8, 6))

    pop_e = model.systemlist[:, 2]
    pop_e = np.sort(pop_e[pop_e > 1e-20])
    y_pop = np.arange(1, len(pop_e) + 1) / len(pop_e)
    plt.plot(np.log10(pop_e), y_pop, color='gray', alpha=0.5, linewidth=4, label='Underlying Population')

    if e_samples is not None:
        sorted_e = np.sort(e_samples)
        y_vals = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        lbl = label if label else f'Sampled (N={len(e_samples)})'
        plt.plot(np.log10(sorted_e + 1e-20), y_vals, drawstyle='steps-post', linewidth=2.0, color='#e74c3c', label=lbl)

    plt.xlabel(r'$\log_{10}(e_{final})$', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Field BBH Eccentricity CDF', fontsize=14)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.6)
    plt.show()


def plot_lifetime_cdf():
    model = _get_model()
    lifetimes = model.systemlist[:, 5]
    lifetimes_Gyr = lifetimes / years / 1e9
    sorted_lifetimes = np.sort(lifetimes_Gyr)
    y_vals = np.arange(1, len(sorted_lifetimes) + 1) / len(sorted_lifetimes)
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_lifetimes, y_vals, linewidth=2.0, color='#2ecc71', label='Lifetime CDF')
    age_Gyr = model.age / years / 1e9
    plt.axvline(x=age_Gyr, color='k', linestyle='--', alpha=0.5, label=f'Universe Age ({age_Gyr:.1f} Gyr)')
    plt.xscale('log')
    plt.xlabel('Merger Time / Lifetime (Gyr)', fontsize=12)
    plt.ylabel('CDF (Probability)', fontsize=12)
    plt.title(f'Field BBH Lifetime CDF (N={len(model.systemlist)})', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def generate_snapshot(t_window_Gyr=10.0, num_realizations=20, tobs_yr=10.0):
    """
    Generates a snapshot of BBHs merging within t_window_Gyr.
    Returns array with appended Num_Realizations.
    Columns: [0:a_ini, 1:e_ini, 2:e_fin, 3:Dl, 4:rate, 5:lifetime, 6:tau, 7:curr_age, 8:a_curr, 9:e_curr, 10:SNR, 11:N_real]
    """
    model = _get_model()
    rate_factor = model.totalrate * 1e3 * num_realizations
    num_candidates = np.random.poisson(rate_factor * t_window_Gyr)

    if num_candidates == 0 or len(model.systemlist) == 0:
        return np.array([])

    indices = np.random.choice(len(model.systemlist), size=num_candidates, replace=True)
    events = model.systemlist[indices]

    t_future = np.random.uniform(0, t_window_Gyr * 1e9, size=num_candidates) * years
    surv_prob = np.exp(-t_future / events[:, 6])

    accept_mask = np.random.random(num_candidates) < surv_prob
    accepted = events[accept_mask]
    t_accepted = t_future[accept_mask]

    age_curr = accepted[:, 5] - t_accepted
    valid = age_curr > 0
    final_events = np.column_stack((accepted[valid], age_curr[valid]))

    if len(final_events) == 0: return np.array([])

    res_list = []
    for row in final_events:
        a0, e0, life_tot, age_now = row[0], row[1], row[5], row[7]
        t_rem = life_tot - age_now

        if e0 < 1e-8:
            a_curr, e_curr = a0 * np.power(t_rem / life_tot, 0.25), e0
        else:
            c0 = a0 / peters_factor_func(e0)
            try:
                e_curr = brentq(lambda e: tmerger(model.m1, model.m2, c0 * peters_factor_func(e), e) - t_rem, 1e-16, e0,
                                xtol=1e-12, maxiter=50)
            except:
                e_curr = 0.0
            a_curr = c0 * peters_factor_func(e_curr)

        snr = calculate_snr(model.m1, model.m2, a_curr, e_curr, row[3], tobs_yr * years)
        res_list.append([a_curr, e_curr, snr])

    # === NEW: Append num_realizations column ===
    # Create a column vector where every element is num_realizations
    n_real_col = np.full((len(final_events), 1), num_realizations)

    # Combine: [Existing Data] + [a_curr, e_curr, snr] + [num_realizations]
    evolvelist_final = np.column_stack((final_events, np.array(res_list), n_real_col))

    return evolvelist_final


def plot_snapshot(snapshot_data, title="MW Field BBH Snapshot"):
    if len(snapshot_data) == 0: return

    # Cols: 8=a_curr, 9=e_curr, 10=SNR, 11=N_realizations
    a, e, snr = snapshot_data[:, 8] / AU, snapshot_data[:, 9], snapshot_data[:, 10]

    # Extract N_realizations
    if snapshot_data.shape[1] > 11:
        n_real = int(snapshot_data[0, 11])
        title_str = f"{title}\n($N_{{systems}}$={len(snapshot_data)}, $N_{{realizations}}$={n_real})"
    else:
        title_str = f"{title}\n($N_{{systems}}$={len(snapshot_data)})"

    idx = np.argsort(snr)[::-1]

    plt.figure(figsize=(10, 8))
    # Scatter plot
    sc = plt.scatter(a[idx], 1.0 - e[idx], s=np.clip(np.sqrt(snr[idx]) * 20, 1, 200),
                     c=np.clip(snr[idx], 1e-3, None), cmap=copy.copy(mpl.colormaps['jet']),
                     norm=mcolors.LogNorm(vmin=0.1, vmax=200), edgecolors='k', linewidths=0.5)

    # Contours logic
    model = _get_model()
    a_grid = np.logspace(np.log10(min(0.001, a.min())), np.log10(max(4e4, a.max())), 500) * AU
    K = (768 / 425) / (4 * (64 / 5 * model.m1 * model.m2 * (model.m1 + model.m2)))

    # Flag to ensure we only add the label to the legend once
    added_legend = False

    for tyr, lbl in zip([1e10, 1e8, 1e6, 1e4], ['10Gyr', '0.1Gyr', '1Myr', '10kyr']):
        val = np.power(tyr * years / (K * a_grid ** 4), 2 / 7)
        valid = val <= 1.0
        if np.any(valid):
            # Only label the first curve for the legend
            label_text = "Merger Timescale" if not added_legend else "_nolegend_"

            plt.plot(a_grid[valid] / AU, 1 - np.sqrt(1 - val[valid]), '--', color='gray', alpha=0.5, label=label_text)
            plt.text(a_grid[valid][-1] / AU, 1 - np.sqrt(1 - val[valid][-1]), lbl, fontsize=9, color='dimgray',
                     ha='left')

            added_legend = True

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Semi-major Axis $a$ [au]", fontsize=13)
    plt.ylabel(r"$1-e$", fontsize=13)
    #plt.xlim(0.001, 4e4)  # Keep limits consistent for better view
    #plt.ylim(0.0008, 1)

    cbar = plt.colorbar(sc, label='SNR (10yr LISA)')
    plt.title(title_str, fontsize=14)
    plt.grid(True, which='both', ls='-', alpha=0.15)

    # Add legend to show what the dashed lines mean
    plt.legend(loc='lower left', fontsize=10, frameon=True)

    plt.tight_layout()
    plt.show()  #