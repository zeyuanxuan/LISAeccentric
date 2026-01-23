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

# --- Global Noise Data Storage ---
_LISA_NOISE_DATA = None


def _try_load_lisa_noise():
    """
    尝试从程序所在文件夹的上一级目录加载 LISA_noise_ASD.csv
    如果成功，将数据存储在全局变量 _LISA_NOISE_DATA 中
    改动：预计算 Log-Log 数据以加速插值，并计算低频延拓斜率
    """
    global _LISA_NOISE_DATA
    try:
        # 获取当前脚本所在目录的上一级目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'LISA_noise_ASD.csv')

        if os.path.exists(file_path):
            # 尝试读取
            try:
                data = np.loadtxt(file_path, delimiter=',')
            except ValueError:
                data = np.loadtxt(file_path, delimiter=',', skiprows=1)

            # 按第一列（频率）排序
            sort_idx = np.argsort(data[:, 0])
            sorted_data = data[sort_idx]

            # 提取频率和ASD
            f_data = sorted_data[:, 0]
            asd_data = sorted_data[:, 1]

            # [关键修改] 预计算 Log10 数据，用于 Log-Log 插值
            # 加上极小值防止 log(0) 报错（虽然物理上 f 和 asd 应该都 > 0）
            log_f = np.log10(f_data + 1e-30)
            log_asd = np.log10(asd_data + 1e-30)

            # [关键修改] 计算低频端的斜率 (Slope)，用于 Power-law 延拓
            # 使用最左边两个点来确定延拓趋势： slope = (y2-y1)/(x2-x1)
            # y = log(ASD), x = log(f)
            low_f_slope = (log_asd[1] - log_asd[0]) / (log_f[1] - log_f[0])

            _LISA_NOISE_DATA = {
                'f_min': f_data[0],
                'f_max': f_data[-1],
                'log_f': log_f,  # 存储 log(f)
                'log_asd': log_asd,  # 存储 log(ASD)
                'low_f_slope': low_f_slope,  # 低频斜率
                'log_f_0': log_f[0],  # 第一个点的 log(f)
                'log_asd_0': log_asd[0]  # 第一个点的 log(ASD)
            }
            print(f"[Info] Successfully loaded LISA noise file: {file_path}")
        else:
            print(f"[Info] LISA noise file not found at {file_path}. Using default analytical model.")
    except Exception as e:
        print(f"[Warning] Failed to load LISA noise file ({e}). Using default analytical model.")
        _LISA_NOISE_DATA = None


# 初始化加载
_try_load_lisa_noise()


# --- SNR Functions ---
def _S_gal_N2A5_scalar(f):
    if f >= 1.0e-5 and f < 1.0e-3: return np.power(f, -2.3) * np.power(10, -44.62) * 20.0 / 3.0
    if f >= 1.0e-3 and f < np.power(10, -2.7): return np.power(f, -4.4) * np.power(10, -50.92) * 20.0 / 3.0
    if f >= np.power(10, -2.7) and f < np.power(10, -2.4): return np.power(f, -8.8) * np.power(10, -62.8) * 20.0 / 3.0
    if f >= np.power(10, -2.4) and f <= 0.01: return np.power(f, -20.0) * np.power(10, -89.68) * 20.0 / 3.0
    return 0.0


S_gal_N2A5 = np.vectorize(_S_gal_N2A5_scalar)


def _S_n_lisa_original(f):
    """原有程序的 Snf 计算方法（作为 fallback）"""
    m1 = 5.0e9
    m2 = sciconsts.c * 0.41 / m1 / 2.0
    return 20.0 / 3.0 * (1 + np.power(f / m2, 2.0)) * (4.0 * (
            9.0e-30 / np.power(2 * sciconsts.pi * f, 4.0) * (1 + 1.0e-4 / f)) + 2.96e-23 + 2.65e-23) / np.power(m1,
                                                                                                                2.0) + S_gal_N2A5(
        f)


def S_n_lisa(f):
    """
    修改后的 Snf 计算方法 (向量化 + Log-Log 插值 + 智能延拓)：
    1. 输入 f 转换为 log10(f)
    2. 在 Log-Log 空间进行线性插值 (对应物理空间的 Power-law 插值)
    3. 低频 (f < f_min): 按 log-log 斜率直线延拓
    4. 高频 (f > f_max): ASD 设为 1.0 (Snf = 1.0)
    """
    if _LISA_NOISE_DATA is not None:
        # 确保输入是数组，方便处理向量化逻辑
        f_arr = np.atleast_1d(f)
        # 转换为 log10(f)，防止 f=0 报错加一个极小值（虽然物理上不应有0）
        log_f_in = np.log10(np.maximum(f_arr, 1e-30))

        # 1. Log-Log 插值
        # left=NaN: 暂时不处理低频，留给后面单独处理
        # right=0.0: 对应 ASD=1.0 (log10(1)=0)，满足高频置1的需求
        log_asd_out = np.interp(
            log_f_in,
            _LISA_NOISE_DATA['log_f'],
            _LISA_NOISE_DATA['log_asd'],
            left=np.nan,
            right=0.0
        )

        # 2. 低频 Power-law 延拓处理
        # 找到超出左边界的索引
        # 使用 np.isnan 来定位，因为上面 interp left 设置为了 NaN
        mask_low = np.isnan(log_asd_out)

        if np.any(mask_low):
            # 公式: y = y0 + slope * (x - x0)
            log_asd_out[mask_low] = _LISA_NOISE_DATA['log_asd_0'] + \
                                    _LISA_NOISE_DATA['low_f_slope'] * \
                                    (log_f_in[mask_low] - _LISA_NOISE_DATA['log_f_0'])

        # 3. 还原回线性空间 ASD = 10^(log_asd)
        asd_out = np.power(10.0, log_asd_out)

        # 4. 计算 Sn(f) = ASD^2
        res = asd_out * asd_out

        # 如果输入是标量，返回标量；如果是数组，返回数组
        if np.isscalar(f):
            return res[0]
        return res
    else:
        # Fallback 到原程序方法
        return _S_n_lisa_original(f)


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


def calculate_snr(m1, m2, a, e, Dl, tobs):
    h0max = np.sqrt(32 / 5) * m1 * m2 / (Dl * a * (1 - e))
    f0max = 2 * np.sqrt((m1 + m2) / (4 * pi * pi * np.power(a * (1 - e), 3.0)))
    if f0max <= 1e-6 or f0max > 1.0: return 0.0
    sqrtsnf = np.sqrt(S_n_lisa(f0max))
    treal = tobs
    return h0max / sqrtsnf * np.sqrt(treal * np.power(1 - e, 3 / 2))
# ==========================================
# Internal Engine Class
# ==========================================
class _MW_Field_BBH_Engine:
    def __init__(self, m1=10 * m_sun, m2=10 * m_sun, formation_mod='starburst',
                 age=10e9 * years, n0=0.1 / (np.power(pc, 3)), rsun=8e3 * pc,
                 Rl=2.6e3 * pc, h=1e3 * pc, sigmav=50e3 / sciconsts.c, fbh=7.5e-4,
                 mp=0.6 * m_sun, fgw=10,
                 n_sim_samples=100000, target_N=50000,
                 rrange_kpc=[0.5, 15], arange_log=[2, 4.5], blocknum=29,
                 data_dir=None, load_default=True):

        self.m1, self.m2 = m1, m2
        self.formation_mod, self.age = formation_mod, age
        self.avgage = 1e9 * years
        self.n0, self.rsun, self.Rl, self.h = n0, rsun, Rl, h
        self.sigmav, self.fbh = sigmav, fbh

        self.mp = mp
        self.fgw = fgw

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
        print(
            f"Running MC simulation ...")
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
                # randomly generating BBH with log uniform seimajor axis
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
                    #for mergers with e>ecrit, eccentricities follows f(e)=2e, here all the cases are e->1
                    e_initial = random.random() * (1 - ecrit) + ecrit
                    #SMA when entering LIGO band (fgw)
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

                    Rcur, phi, cosi = ravg / 1000 / pc, 2 * pi * random.random(), 0
                    Dl = np.sqrt((Rcur * np.sqrt(1 - cosi ** 2) * np.sin(phi)) ** 2 + (Rcur * cosi) ** 2 + (
                            Rcur * np.sqrt(1 - cosi ** 2) * np.cos(phi) - 8) ** 2) * 1000 * pc
                    lifetime = tmerger(self.m1, self.m2, acur, e_initial)

                    final_rate = rate if self.formation_mod == 'starburst' else rate1
                    raw_systemlist.append([acur, e_initial, efinal, Dl, final_rate, lifetime, tau])

            self.totalrate += submerger if self.formation_mod == 'starburst' else submerger1

        print("Resampling population based on different fly-by induced BBH merger rates at different initial SMA...")
        if len(raw_systemlist) > 0:
            data = np.array(raw_systemlist)
            weights = data[:, 4]
            probs = weights / np.sum(weights)
            self.systemlist = data[np.random.choice(len(data), size=self.target_N, replace=True, p=probs)]
            self.is_simulated = True
            print(
                f"Simulation Done. Merger Rate in the galaxy: {self.totalrate:.5f} /Myr. Merger Sample Size: {len(self.systemlist)}")
        else:
            print("Error: No systems generated.")

    def save_data(self):
        np.save(self.pop_file, self.systemlist)
        # [修改点] 将 m1, m2, mp 等物理参数也存入 meta 文件
        meta_data = {
            'totalrate': self.totalrate,
            'm1': self.m1,
            'm2': self.m2,
            'mp': self.mp
        }
        np.save(self.meta_file, meta_data)
        print(
            f"Data saved to {self.data_dir} (Rate={self.totalrate:.4f}, M={self.m1 / m_sun:.1f}+{self.m2 / m_sun:.1f} Msun)")

    def load_data(self):
        if os.path.exists(self.pop_file) and os.path.exists(self.meta_file):
            print(f"Loading data from: {self.data_dir}")
            self.systemlist = np.load(self.pop_file, allow_pickle=True)

            # [修改点] 读取 meta 并更新类属性
            meta_data = np.load(self.meta_file, allow_pickle=True).item()
            self.totalrate = meta_data['totalrate']

            # 尝试读取质量（兼容旧版数据，如果读不到就保持默认）
            if 'm1' in meta_data: self.m1 = meta_data['m1']
            if 'm2' in meta_data: self.m2 = meta_data['m2']
            if 'mp' in meta_data: self.mp = meta_data['mp']

            self.is_simulated = True
            m1_solar = self.m1 / m_sun
            m2_solar = self.m2 / m_sun
            print(
                f"Loaded simulation data. N={len(self.systemlist)}, Rate={self.totalrate:.5f}/Myr, Masses={m1_solar:.1f}+{m2_solar:.1f}")
        else:
            print(f"No pre-generated data found in {self.data_dir}.")

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
            a0, e0, life_tot = row[0], row[1], row[5]
            age_now = final_ages[i]
            dl = row[3]
            t_rem = life_tot - age_now

            if e0 < 1e-8:
                a_curr, e_curr = a0 * np.power(t_rem / life_tot, 0.25), e0
            else:
                c0 = a0 / peters_factor_func(e0)
                try:
                    e_curr = brentq(lambda e: tmerger(self.m1, self.m2, c0 * peters_factor_func(e), e) - t_rem, 1e-16,
                                    e0, xtol=1e-12, maxiter=50)
                except:
                    e_curr = 0.0
                a_curr = c0 * peters_factor_func(e_curr)

            snr = calculate_snr(self.m1, self.m2, a_curr, e_curr, dl, tobs_yr * years)

            output_list.append([
                'Field',
                dl / 1000 / pc,
                a_curr / AU,
                e_curr,
                10.0, 10.0,
                snr
            ])

        return output_list


# ==========================================
# Public Interface (API)
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


# --- MODIFIED: Plotting Features ---

def generate_eccentricity_samples(size=10000):
    model = _get_model()
    if len(model.systemlist) == 0: return np.array([])
    indices = np.random.choice(len(model.systemlist), size=size, replace=True)
    return model.systemlist[indices, 2]


def plot_eccentricity_cdf(e_samples=None, label=None):
    model = _get_model()
    # MODIFICATION: Smaller figure size
    plt.figure(figsize=(6, 5))

    pop_e = model.systemlist[:, 2]
    pop_e = np.sort(pop_e[pop_e > 1e-20])
    y_pop = np.arange(1, len(pop_e) + 1) / len(pop_e)

    if e_samples is not None:
        sorted_e = np.sort(e_samples)
        y_vals = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        lbl = label if label else f'Sampled (N={len(e_samples)})'
        plt.plot(np.log10(sorted_e + 1e-20), y_vals, drawstyle='steps-post', linewidth=2.0, color='#e74c3c', label=lbl)

    # MODIFICATION: Larger labels and ticks
    plt.xlabel(r"$\log_{10}(e)$", fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.title(f"Merger Eccentricities (Default at 10Hz)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.legend(fontsize=11)
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def get_merger_progenitor_population():
    model = _get_model()
    return model.systemlist


def plot_progenitor_sma_distribution(bins=50):
    model = _get_model()
    if len(model.systemlist) == 0:
        print("No data to plot.")
        return

    sma_au = model.systemlist[:, 0] / AU

    # MODIFICATION: Smaller figure size
    plt.figure(figsize=(6, 5))
    log_bins = np.logspace(np.log10(min(sma_au)), np.log10(max(sma_au)), bins)

    plt.hist(sma_au, bins=log_bins, color='#3498db', alpha=0.7, edgecolor='black', label='Fly-by Induced BBH Merger Progenitors')
    plt.xscale('log')

    # MODIFICATION: Larger labels and ticks
    plt.xlabel('Semi-Major Axis [au]', fontsize=14)
    plt.ylabel('Count (Merger Rate Weighted)', fontsize=14)
    plt.title(f'Initial SMA of Merger Progenitors (N={len(model.systemlist)})', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_lifetime_cdf():
    model = _get_model()
    lifetimes = model.systemlist[:, 5]
    lifetimes_Gyr = lifetimes / years / 1e9
    sorted_lifetimes = np.sort(lifetimes_Gyr)
    y_vals = np.arange(1, len(sorted_lifetimes) + 1) / len(sorted_lifetimes)

    # MODIFICATION: Smaller figure size
    plt.figure(figsize=(6, 5))
    plt.plot(sorted_lifetimes, y_vals, linewidth=2.0, color='#2ecc71', label='Fly-by Induced BBH Merger Progenitors')
    age_Gyr = model.age / years / 1e9
    plt.axvline(x=age_Gyr, color='k', linestyle='--', alpha=0.5, label=f'Age ({age_Gyr:.1f} Gyr)')

    plt.xscale('log')
    # MODIFICATION: Larger labels and ticks
    plt.xlabel('Merger Time (Gyr)', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.title(f'Lifetime CDF', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()


def get_single_mw_realization(t_window_Gyr=10.0, tobs_yr=10.0):
    model = _get_model()
    rate = model.totalrate * 1e3
    num = np.random.poisson(rate * t_window_Gyr)
    if num == 0: return []
    indices = np.random.choice(len(model.systemlist), size=num, replace=True)
    return model._process_candidates(model.systemlist[indices], t_window_Gyr, tobs_yr)


def get_multi_mw_realizations(n_realizations=10, t_window_Gyr=10.0, tobs_yr=10.0):
    model = _get_model()
    rate = model.totalrate * 1e3 * n_realizations
    num = np.random.poisson(rate * t_window_Gyr)
    if num == 0: return []
    indices = np.random.choice(len(model.systemlist), size=num, replace=True)
    return model._process_candidates(model.systemlist[indices], t_window_Gyr, tobs_yr)


def get_random_systems(n_systems=500, t_window_Gyr=10.0, tobs_yr=10.0):
    model = _get_model()
    output_list = []
    batch_size = n_systems * 2
    attempts = 0
    while len(output_list) < n_systems and attempts < 100:
        indices = np.random.choice(len(model.systemlist), size=batch_size, replace=True)
        batch_res = model._process_candidates(model.systemlist[indices], t_window_Gyr, tobs_yr)
        output_list.extend(batch_res)
        attempts += 1
    return output_list[:n_systems]


def plot_mw_field_bbh_snapshot(systems, title="Snapshot of BBH Merger Progenitors in the Galactic Field", tobs_yr=10.0):
    if not systems:
        print("No systems to plot.")
        return

    data = np.array(systems)[:, 1:].astype(float)
    a = data[:, 1]
    e = data[:, 2]
    snr = data[:, 5]

    idx = np.argsort(snr)[::-1]
    a_p, ome_p, snr_p = a[idx], 1.0 - e[idx], snr[idx]

    # MODIFICATION: Figure size
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

        valid = (val <= 1.0)

        if np.any(valid):
            lbl_text = "Merger Timescale" if not added_legend else "_nolegend_"
            plt.plot(a_grid[valid] / AU, 1 - np.sqrt(1 - val[valid]), '--', color='gray', alpha=0.5, label=lbl_text)

            plt.text(a_grid[valid][-1] / AU, 1 - np.sqrt(1 - val[valid][-1]), lbl,
                     fontsize=10, color='dimgray', ha='left', va='center')
            added_legend = True

    plt.xscale('log')
    plt.yscale('log')

    log_min = np.log10(x_min_val)
    log_max = np.log10(x_max_val)
    log_span = log_max - log_min

    new_log_max = log_max + (log_span * 0.10)
    plt.xlim(10 ** log_min, 10 ** new_log_max)

    # MODIFICATION: Labels and ticks
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