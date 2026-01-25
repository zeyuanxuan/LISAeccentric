# coding:utf-8

import numpy as np
import scipy.constants as sciconsts
import scipy.special as scipy_special
import scipy.interpolate as sci_interpolate
import scipy.integrate as sci_integrate
import time
import sys
from multiprocessing import Process, Pool, cpu_count
import scipy.optimize as sciop
import random
import math
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os
from numba import njit, float64

# --- Helper Functions for Precise Evolution (Peters 1964) ---
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

# ==========================================
# 1. 物理常数与核心数学/物理函数 (保持原样)
# ==========================================

m_sun = 1.9891e30 * sciconsts.G / np.power(sciconsts.c, 3.0)
gama = 0.577215664901532860606512090082402431042159335
pi = sciconsts.pi
years = 365 * 24 * 3600.0
pc = 3.261 * sciconsts.light_year / sciconsts.c
AU = sciconsts.au / sciconsts.c


def J(n, x):  # 贝塞尔函数
    return scipy_special.jv(n, x)


def g(n, e):
    ne = n * e
    jn_2 = J(n - 2, ne)
    jn_1 = J(n - 1, ne)
    jn = J(n, ne)
    jn_p1 = J(n + 1, ne)
    jn_p2 = J(n + 2, ne)

    term1 = jn_2 - 2 * e * jn_1 + (2 / n) * jn + 2 * e * jn_p1 - jn_p2
    term2 = jn_2 - 2 * jn + jn_p2
    term3 = jn

    result = np.power(n, 4.0) / 32 * (
            np.power(term1, 2.0) +
            (1 - e * e) * np.power(term2, 2.0) +
            4 / (3 * n * n) * np.power(term3, 2.0)
    )
    return result


def h0(a, m1, m2, Dl):
    return np.sqrt(32 / 5) * m1 * m2 / Dl / a


def h(a, e, n, m1, m2, Dl):
    result = 2 / n * np.sqrt(g(n, e)) * h0(a, m1, m2, Dl)
    return result


# ==============================================================================
# Unified Noise Handling (Log-Log Interpolation & Injection Support)
# ==============================================================================

# --- Global Noise Data Storage ---
_LISA_NOISE_DATA = None


def _try_load_lisa_noise():
    """
    尝试加载 LISA_noise_ASD.csv。
    支持 Log-Log 预计算，并兼容 core.py 的注入机制。
    自动搜索当前目录和上一级目录。
    """
    global _LISA_NOISE_DATA
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 搜索路径策略：先找当前目录，再找上一级目录（兼容不同包结构）
        possible_paths = [
            os.path.join(current_dir, 'LISA_noise_ASD.csv'),
            os.path.join(os.path.dirname(current_dir), 'LISA_noise_ASD.csv')
        ]

        file_path = None
        for p in possible_paths:
            if os.path.exists(p):
                file_path = p
                break

        if file_path:
            # 尝试读取
            try:
                data = np.loadtxt(file_path, delimiter=',')
            except ValueError:
                data = np.loadtxt(file_path, delimiter=',', skiprows=1)

            # 1. 排序与清洗
            sort_idx = np.argsort(data[:, 0])
            sorted_data = data[sort_idx]

            f_data = sorted_data[:, 0]
            asd_data = sorted_data[:, 1]

            # 过滤非正值
            mask = (f_data > 0) & (asd_data > 0)
            f_data = f_data[mask]
            asd_data = asd_data[mask]

            # 2. 预计算 Log10 数据 (用于 Log-Log 插值)
            log_f = np.log10(f_data)
            log_asd = np.log10(asd_data)

            # 3. 计算低频延拓斜率 (Slope)
            # y = kx + b -> slope = (y1-y0)/(x1-x0)
            if len(log_f) >= 2:
                low_f_slope = (log_asd[1] - log_asd[0]) / (log_f[1] - log_f[0])
            else:
                low_f_slope = -2.5  # Fallback default

            _LISA_NOISE_DATA = {
                'f_min': f_data[0],
                'f_max': f_data[-1],
                'log_f': log_f,  # 存储 log(f)
                'log_asd': log_asd,  # 存储 log(ASD)
                'low_f_slope': low_f_slope,
                'log_f_0': log_f[0],
                'log_asd_0': log_asd[0],
                'use_file': True  # 标记位
            }
            # print(f"[Info] Loaded LISA noise from {os.path.basename(file_path)}")
        else:
            print(f"[Warning] LISA noise file not found. Using analytical fallback.")
            _LISA_NOISE_DATA = None

    except Exception as e:
        print(f"[Warning] Failed to load LISA noise file ({e}). Using analytical fallback.")
        _LISA_NOISE_DATA = None


# 初始化加载
_try_load_lisa_noise()


# --- SNR Functions ---

def _S_gal_N2A5_scalar(f):
    # Analytical Galactic Background (N2A5 Model)
    if f >= 1.0e-5 and f < 1.0e-3: return np.power(f, -2.3) * 10 ** -44.62 * 20.0 / 3.0
    if f >= 1.0e-3 and f < 10 ** -2.7: return np.power(f, -4.4) * 10 ** -50.92 * 20.0 / 3.0
    if f >= 10 ** -2.7 and f < 10 ** -2.4: return np.power(f, -8.8) * 10 ** -62.8 * 20.0 / 3.0
    if f >= 10 ** -2.4 and f <= 0.01: return np.power(f, -20.0) * 10 ** -89.68 * 20.0 / 3.0
    return 0.0


S_gal_N2A5 = np.vectorize(_S_gal_N2A5_scalar)


def _S_n_lisa_original(f):
    """Fallback Analytical Model (Robson+19 / N2A5)"""
    m1 = 5.0e9
    m2 = sciconsts.c * 0.41 / m1 / 2.0
    term_inst = 20.0 / 3.0 * (1 + (f / m2) ** 2) * (
                4.0 * (9.0e-30 / (2 * pi * f) ** 4 * (1 + 1.0e-4 / f)) + 2.96e-23 + 2.65e-23) / m1 ** 2
    return term_inst + S_gal_N2A5(f)


def S_n_lisa(f):
    """
    Unified Noise Calculator:
    1. Checks if file/injected data exists in _LISA_NOISE_DATA.
    2. If yes -> Log-Log Interpolation with Slope Extrapolation.
    3. If no  -> Fallback to analytical formula.
    """
    if _LISA_NOISE_DATA is not None and _LISA_NOISE_DATA.get('use_file', False):
        f_arr = np.atleast_1d(f)
        # Convert to Log space (protect against f<=0)
        log_f_in = np.log10(np.maximum(f_arr, 1e-30))

        # 1. Log-Log Interpolation
        # left=NaN (handle later), right=0.0 (ASD=1.0 for high freq)
        log_asd_out = np.interp(
            log_f_in,
            _LISA_NOISE_DATA['log_f'],
            _LISA_NOISE_DATA['log_asd'],
            left=np.nan,
            right=0.0
        )

        # 2. Handle Low Frequency Extrapolation (Slope)
        mask_low = np.isnan(log_asd_out)
        if np.any(mask_low):
            log_asd_out[mask_low] = _LISA_NOISE_DATA['log_asd_0'] + \
                                    _LISA_NOISE_DATA['low_f_slope'] * \
                                    (log_f_in[mask_low] - _LISA_NOISE_DATA['log_f_0'])

        # 3. Convert back to Linear ASD and square to get Sn(f)
        asd_out = np.power(10.0, log_asd_out)
        res = asd_out * asd_out

        if np.isscalar(f): return res[0]
        return res
    else:
        return _S_n_lisa_original(f)

def chirp_mass(m1, m2):
    return np.power(m1 * m2, 0.6) / (np.power(m1 + m2, 0.2))


def dforb_dt(m1, m2, a, e):
    ft = 1 / 2 / pi * np.sqrt(m1 + m2) * np.power(a, -3.0 / 2.0)
    et = e
    Mc = chirp_mass(m1, m2)
    Fe = (1 + 73 / 24 * et * et + 37 / 96 * np.power(et, 4.0)) / (np.power(1 - et * et, 7 / 2))
    fj2 = 96 * np.power(pi, 8 / 3) / 5 * np.power(Mc, 5 / 3) * np.power(ft * 2, 11 / 3) * Fe
    return fj2 / 2


# ==========================================
# 2. 核心计算逻辑
# ==========================================

def _core_calculator(args):
    """
    底层计算核心。
    Args: (m1_SI, m2_SI, a_SI, e, Dl_SI, tobs_SI, target_max_points, verbose)
    """
    # [修改] 增加 verbose 参数接收
    m1, m2, a, e, Dl, tobs, target_max_points, verbose = args

    # 1. 计算基频
    forb = 1 / 2 / pi * np.sqrt(m1 + m2) * np.power(a, -3.0 / 2.0)

    # 2. 确定 n 的范围
    e_calc = min(e, 1 - 1e-16)
    n_peak = np.sqrt(1 + e_calc) * np.power((1 - e_calc), -3.0 / 2.0)

    n_start = max(1, int(0.01 * n_peak))
    n_end = int(10 * n_peak)

    if n_end < n_start:
        return [], [], [], []

    # [修改] 智能稀疏采样逻辑 + verbose 控制
    step = 1
    total_harmonics = n_end - n_start

    # 如果总点数超过了设定的上限，则增加步长
    if total_harmonics > target_max_points:
        step = int(total_harmonics / target_max_points)
        # [核心] 只有当 verbose=True 时才打印
        if verbose:
            print(f"   [Info] Large harmonics ({total_harmonics}), downsampling step={step} (max={target_max_points})")

    n_arr = np.arange(n_start, n_end + 1, step, dtype=np.float64)

    df_dt_val = dforb_dt(m1, m2, a, e)
    h0_val = h0(a, m1, m2, Dl)
    g_vals = g(n_arr, e)

    hn_arr = 2 / n_arr * np.sqrt(g_vals) * h0_val
    fn_arr = n_arr * forb

    hnc2 = 2 * fn_arr ** 2 * hn_arr ** 2 / (n_arr * df_dt_val)
    hnc = np.sqrt(hnc2)
    hcn_bkg = 2 / n_arr * np.sqrt(g_vals) * h0_val
    Snfvec = hcn_bkg ** 2 / forb
    hc_avg2 = 2 * fn_arr ** 2 * hn_arr ** 2 / (forb) * tobs
    hc_avg = np.sqrt(hc_avg2)

    return fn_arr, hnc, hc_avg, Snfvec


# ==========================================
# 3. 封装接口
# ==========================================

def calculate_single_system(m1, m2, a, e, Dl, tobs=1.0*years, target_max_points=20000, verbose=True):
    """
    接口1: 单个系统计算
    默认: verbose=True (允许打印), target_max_points=20000 (高精度)
    """
    m1_si = m1
    m2_si = m2
    a_si = a
    Dl_si = Dl
    tobs_si = tobs

    # 传入 verbose=True
    args = (m1_si, m2_si, a_si, e, Dl_si, tobs_si, target_max_points, verbose)
    fn, hnc, hc_avg, snf = _core_calculator(args)

    return [fn, hc_avg, hnc, snf]


def process_population_batch(system_list_raw, tobs=1.0*years, n_cores=1, target_max_points=1000):
    """
    接口2: 批量处理系统
    强制: verbose=False (在 batch 内部屏蔽所有打印), target_max_points 默认为 1000 (低精度/高速度)
    """

    pool_args = []
    tobs_si = tobs

    # 强制静默
    batch_verbose = False

    for item in system_list_raw:
        # item: [id, Dl, a, e, m1, m2]
        Dl_si = item[1] * 1e3 * pc
        a_si = item[2] * AU
        e_val = item[3]
        m1_si = item[4] * m_sun
        m2_si = item[5] * m_sun

        # [核心] 将 target_max_points 和 verbose=False 传入元组
        pool_args.append((m1_si, m2_si, a_si, e_val, Dl_si, tobs_si, target_max_points, batch_verbose))

    logfrange = np.linspace(-6, 0, 1000)
    faxis = np.power(10., logfrange)
    Snf_tot = np.zeros_like(faxis)

    all_fn_lists = []
    all_hcavg_lists = []
    all_hnc_lists = []

    # [核心] 如果需要完全静默，这里也不打印；如果仅屏蔽 worker 打印，这里可以保留
    # 根据“屏蔽 print 输出”的要求，这里也加上 verbose 判断，默认为 False
    if batch_verbose:
        print(f"Start calculation for {len(system_list_raw)} systems (Sequential)...")

    t_start = time.time()

    results = []
    for i, args in enumerate(pool_args):
        results.append(_core_calculator(args))

    for res in results:
        fn_sys, hnc_sys, hc_avg_sys, Snf_sys = res
        if len(fn_sys) > 0:
            all_fn_lists.append(fn_sys)
            all_hcavg_lists.append(hc_avg_sys)
            all_hnc_lists.append(hnc_sys)
            if len(fn_sys) > 1:
                Snf_interp = np.interp(faxis, fn_sys, Snf_sys, left=0, right=0)
                Snf_tot += Snf_interp

    if batch_verbose:
        print(f"Calculation done in {time.time() - t_start:.2f} seconds.")

    return [faxis, Snf_tot, all_fn_lists, all_hcavg_lists, all_hnc_lists]

# [新增功能] 4. 单个系统绘图
# ==========================================
def plot_single_system_results(single_system_res, xlim=[1e-6, 1], ylim=[1e-23, 1e-14]):
    """
    接口4: 单个系统绘图
    输入: single_system_res (calculate_single_system 的返回值)
    """
    fn = single_system_res[0]
    hc_avg = single_system_res[1]
    hnc = single_system_res[2]  # This is scatter
    # Snf = single_system_res[3]

    # 生成 LISA 噪声曲线用于背景对比
    f_lisa = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 1000)
    hc_lisa = np.sqrt(f_lisa * S_n_lisa(f_lisa))

    plt.figure(figsize=(10, 7), dpi=100)

    # 1. Plot LISA Noise
    plt.plot(f_lisa, hc_lisa, color='black', linewidth=2, label='LISA Sensitivity ($\sqrt{f S_n(f)}$)')

    # 2. Plot hc_avg (Curve)
    if len(fn) > 0:
        plt.plot(fn, hc_avg, color='blue', linewidth=1.5, label=r'$h_{c, \mathrm{avg}} = \sqrt{2f^2h_n^2/f_{\rm orb} \times T_{\rm obs}}$ (Time-integrated spectrum - enclosed area reflects SNR)')

    # 3. Plot hc_scatter (Scatter points)
    if len(fn) > 0:
        plt.scatter(fn, hnc, color='red', s=4, alpha=0.7, zorder=5, label=r'$h_{c, n} = \sqrt{2f^2h_n^2/\dot{f}}$ (Instantaneous hc value for each harmonic)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [Hz]', fontsize=14)
    plt.ylabel('Characteristic Strain $h_c$', fontsize=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    plt.title('Single Eccentric Binary Spectrum', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_simulation_results(simulation_result_list,xlim=[1e-6,1],ylim=[1e-24, 1e-15]):
    """
    接口3: 绘图
    输入: simulation_result_list
    """
    # 解包，注意现在 list 长度为 5
    faxis = simulation_result_list[0]
    Snf_tot = simulation_result_list[1]
    all_fn_lists = simulation_result_list[2]
    all_hcavg_lists = simulation_result_list[3]
    all_hnc_lists = simulation_result_list[4]  # 这里接收了，虽然不画

    hc_background = np.sqrt(faxis * Snf_tot)
    sqrtfsnflist = np.sqrt(faxis * S_n_lisa(faxis))

    fig3 = plt.figure(figsize=(10, 8), dpi=100)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 画 LISA 灵敏度曲线
    plt.plot(faxis, sqrtfsnflist, label="LISA Noise", color='black', linewidth=2, zorder=10)

    # 画每个系统的折线图 (使用 hc_avg)
    label_added = False

    # zip遍历，忽略 hnc_lists
    for fn_sys, hc_avg_sys in zip(all_fn_lists, all_hcavg_lists):
        if len(fn_sys) > 0:
            lbl = "Individual Systems" if not label_added else None
            # 折线图
            plt.plot(fn_sys, hc_avg_sys, color='blue', alpha=0.3, linewidth=1, label=lbl)
            label_added = True

    # 画总背景噪声
    plt.plot(faxis, hc_background, label=r"Total Background ($\sqrt{f S_n(f)_{\mathrm{tot}}}$)", color='red', linestyle='--', linewidth=2,
             zorder=10)

    plt.xlabel("f [Hz]", fontsize=14)
    plt.ylabel("$h_c$", fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.2)
    plt.title("GW Source Population (Individual Spectra) and Background", fontsize=16)
    plt.show()


# ==========================================
# 5. Precise Orbit Evolution & Spectral History
# ==========================================


def peters_factor_func(e):
    """
    Peters (1964) function for semi-major axis evolution.
    a = c0 * peters_factor_func(e)
    """
    if e < 1e-10: return 0.0
    term1 = np.power(e, 12.0 / 19.0)
    term2 = 1.0 - e * e
    term3 = np.power(1.0 + (121.0 / 304.0) * e * e, 870.0 / 2299.0)
    return (term1 / term2) * term3


class MergerTimeAccelerator:
    """
    Pre-computes Peters (1964) merger time integral factors.
    Cached to 'merger_time_table.npz' for speed.
    """

    def __init__(self, cache_file='merger_time_table.npz'):
        # Determine path relative to this script
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()

        self.cache_file = os.path.join(base_dir, cache_file)

        # Try to load cache
        if self._load_cache():
            pass
        else:
            print("[System] Computing merger time table (first time run)...")
            self._compute_and_save_table()
            print(f"[System] Table computed and saved to {self.cache_file}")

        # Pchip Interpolator for monotonicity
        self.interpolator = PchipInterpolator(self.e_grid, self.f_vals)

    def _compute_and_save_table(self):
        # Logarithmic sampling for e -> 1
        x_vals = np.linspace(0, 5, 2000)
        self.e_grid = 1.0 - np.power(10, -x_vals)
        self.e_grid[0] = 0.0

        self.f_vals = np.array([self._compute_dimensionless_factor(e) for e in self.e_grid])
        np.savez(self.cache_file, e_grid=self.e_grid, f_vals=self.f_vals)

    def _load_cache(self):
        if not os.path.exists(self.cache_file):
            return False
        try:
            data = np.load(self.cache_file)
            self.e_grid = data['e_grid']
            self.f_vals = data['f_vals']
            return True
        except Exception as e:
            print(f"[Warning] Failed to load cache: {e}")
            return False

    def _compute_dimensionless_factor(self, e0):
        if e0 < 1e-6: return 1.0
        term_c0 = np.power(e0, 12.0 / 19.0) * np.power(1 + 121.0 / 304.0 * e0 ** 2, 870.0 / 2299.0)
        c0_norm = (1 - e0 ** 2) / term_c0

        def integrand(e):
            numer = np.power(e, 29.0 / 19.0) * np.power(1 + 121.0 / 304.0 * e ** 2, 1181.0 / 2299.0)
            denom = np.power(1 - e ** 2, 1.5)
            return numer / denom

        integral_val, _ = quad(integrand, 0, e0)
        return (48.0 / 19.0) * np.power(c0_norm, 4.0) * integral_val

    def get_factor(self, e):
        if e < 1e-4: return 1.0
        if e > 0.99999:
            return (768.0 / 425.0) * np.power(1 - e ** 2, 3.5)
        return self.interpolator(e)


# Initialize Global Accelerator
_ACCELERATOR = MergerTimeAccelerator()


def tmerger_integral(m1, m2, a0, e0):
    """
    Accurate merger time calculation using pre-computed Peters factors.
    Inputs: m1, m2 (seconds), a0 (seconds/light-seconds), e0.
    """
    beta = 64.0 / 5.0 * m1 * m2 * (m1 + m2)
    if beta == 0: return 1e99

    t_circ = np.power(a0, 4.0) / (4.0 * beta)

    if np.isscalar(e0):
        factor = _ACCELERATOR.get_factor(e0)
    else:
        factor = np.array([_ACCELERATOR.get_factor(e) for e in e0])

    return t_circ * factor


def GWtime(m1, m2, a1, e1):
    return tmerger_integral(m1, m2, a1, e1)


def solve_ae_after_time(m1, m2, a0, e0, dt):
    """
    Evolve system by dt seconds.
    Inputs: m1, m2 (seconds), a0 (seconds), e0 (dimensionless), dt (seconds).
    Returns: a_curr, e_curr
    """
    current_life = GWtime(m1, m2, a0, e0)
    if dt >= current_life:
        # System has merged
        return 0.0, 0.0

    t_rem_target = current_life - dt

    # Conservation of Peters constant: a = c0 * factor(e) -> c0 = a / factor(e)
    # But note: peters_factor_func here is defined such that a = c0 * peters_factor_func(e)
    # Check consistency: peters_factor_func(e) ~ e^(12/19) / (1-e^2) ...
    # This matches the relation used in tmerger logic.

    fact_e0 = peters_factor_func(e0)
    if fact_e0 == 0: return 0.0, 0.0  # Should not happen if e0 > 1e-10

    c0 = a0 / fact_e0

    # Solve for e_curr such that T_merger(e_curr) == t_rem_target
    try:
        # We need to find e such that GWtime(c0*func(e), e) - t_target = 0
        e_curr = brentq(lambda e: GWtime(m1, m2, c0 * peters_factor_func(e), e) - t_rem_target,
                        1e-6, e0, xtol=1e-12, maxiter=50)
    except Exception as e:
        print(f"[Warning] solve_ae failed: {e}. Returning initial values.")
        e_curr = e0

    a_curr = c0 * peters_factor_func(e_curr)
    return a_curr, e_curr


# ==========================================
# 5. Precise Evolving System Calculator (Modified)
# ==========================================

def get_orbit_at_f(target_f, c0, m_total_sec):
    """
    辅助函数：给定目标轨道频率 f，反解出此时的 e 和 a。
    利用守恒量 c0 = a / peters_factor_func(e)
    以及 f = 1/2pi * sqrt(M/a^3)
    => f = 1/2pi * sqrt(M / [c0 * P(e)]^3 )
    """
    # 目标半长轴 a_target
    # f = 1/(2pi) * sqrt(M/a^3)  => a^3 = M / (2pi f)^2 => a = (M / (2pi f)^2)^(1/3)
    omega = 2 * pi * target_f
    a_target = np.power(m_total_sec / (omega ** 2), 1.0 / 3.0)

    # 求解 e: a_target = c0 * peters_factor_func(e)
    # 即 peters_factor_func(e) - a_target / c0 = 0
    # P(e) 是单调递增的 (e=0->0, e=1->inf)
    target_val = a_target / c0

    # 边界检查
    if target_val <= 0: return 0.0, a_target  # Should not happen

    # 寻找根
    try:
        # e 的范围通常在 0 到 1 之间 (不含1)
        # 对于 P(e)，e接近1时值非常大，e接近0时值接近0
        # 我们使用 brentq
        e_sol = brentq(lambda e: peters_factor_func(e) - target_val, 1e-6, 0.99999, xtol=1e-12, maxiter=50)
    except ValueError:
        # 如果找不到根，说明可能已经圆化 (e very small) 或者数值误差
        # P(e) ~ e^(12/19). 如果 target_val 很小，e 也很小
        if target_val < 1e-4:
            e_sol = 1e-6
        else:
            e_sol = 0.99999  # Fallback

    return e_sol, a_target


def calculate_evolving_system(m1, m2, a, e, Dl, tobs_years=4.0, target_n_points=100,
                              all_harmonics=False, plot=True, verbose=True):
    """
    计算演化双星系统的频谱轨迹和积分频谱。
    [Update]:
    1. 在绘图信息框中增加距离参数 (Dl)。
    2. 修复 LaTeX 格式化问题。

    Returns:
        [unified_f_axis, total_hc_spectrum, snapshots_data, snr_val]
    """

    # 1. 单位转换
    m1_sec = m1 * m_sun
    m2_sec = m2 * m_sun
    m_total_sec = m1_sec + m2_sec
    a_sec = a * AU
    Dl_sec = Dl * 1e3 * pc
    tobs_sec = tobs_years * years

    # 2. 演化常数
    pf_val = peters_factor_func(e)
    c0 = a_sec / (pf_val if pf_val > 0 else 1e-10)

    r_isco = 6.0 * m_total_sec
    f_isco = 1.0 / (pi * np.power(6.0, 1.5) * m_total_sec)

    # 3. 确定演化范围
    f_start = 1.0 / (2.0 * pi) * np.sqrt(m_total_sec / np.power(a_sec, 3.0))
    t_merger_total = tmerger_integral(m1_sec, m2_sec, a_sec, e)
    t_end_limit = min(t_merger_total, tobs_sec)

    if t_end_limit >= t_merger_total * 0.999:
        f_end = min(f_isco, 1.0)
    else:
        a_final, e_final = solve_ae_after_time(m1_sec, m2_sec, a_sec, e, t_end_limit)
        f_end = 1.0 / (2.0 * pi) * np.sqrt(m_total_sec / np.power(a_final, 3.0))

    # 4. 时间/频率网格
    n_steps = 100
    if f_end <= f_start: f_end = f_start * 1.0001
    f_orb_grid = np.geomspace(f_start, f_end, n_steps)

    # 5. [Track] 锁定 hnc 用的谐波列表
    n_peak_0 = np.sqrt(1 + e) * np.power((1 - e), -1.5) if e < 0.999 else 1000
    n_max_0 = int(20 * n_peak_0)

    if all_harmonics:
        locked_n_indices = np.arange(1, n_max_0 + 1, dtype=int)
    else:
        if n_max_0 > target_n_points:
            locked_n_indices = np.unique(np.geomspace(1, n_max_0, target_n_points).astype(int))
        else:
            locked_n_indices = np.arange(1, n_max_0 + 1, dtype=int)

    if 2 not in locked_n_indices and n_max_0 >= 2:
        locked_n_indices = np.sort(np.append(locked_n_indices, 2))

    # 6. 准备总频谱容器
    unified_f_axis = np.geomspace(1e-6, 1.0, 2000)
    total_sq_strain = np.zeros_like(unified_f_axis)

    snapshots = []
    t_grid = []

    if verbose:
        mode_str = "ALL Tracks" if all_harmonics else f"Sparse Tracks (N~{len(locked_n_indices)})"
        print(f"   Calculating Evolving Strain... ({mode_str})")

    for i, f_curr in enumerate(f_orb_grid):
        e_curr, a_curr = get_orbit_at_f(f_curr, c0, m_total_sec)
        if a_curr <= r_isco or e_curr < 0: break

        t_rem_curr = tmerger_integral(m1_sec, m2_sec, a_curr, e_curr)
        t_elapsed = t_merger_total - t_rem_curr
        t_grid.append(t_elapsed)

        dt = 0 if i == 0 else t_grid[i] - t_grid[i - 1]

        # Part A: Track Calculation
        n_track = locked_n_indices
        fn_track = n_track * f_curr
        g_track = g(n_track, e_curr)
        h0_val = h0(a_curr, m1_sec, m2_sec, Dl_sec)
        hn_track = (2.0 / n_track) * np.sqrt(g_track) * h0_val

        df_dt_val = dforb_dt(m1_sec, m2_sec, a_curr, e_curr)
        dfn_dt_track = np.maximum(n_track * df_dt_val, 1e-50)

        hnc_sq_track = 2.0 * np.square(fn_track) * np.square(hn_track) / dfn_dt_track
        hnc_track = np.sqrt(hnc_sq_track)

        snapshots.append({
            't': t_elapsed, 'f_orb': f_curr, 'n': n_track,
            'freq': fn_track, 'hnc': hnc_track
        })

        # Part B: Spectrum Integration
        if dt > 0:
            n_lower_band = int(np.floor(1e-6 / f_curr))
            n_upper_band = int(np.ceil(1.0 / f_curr))
            n_peak = np.sqrt(1 + e_curr) * np.power((1 - e_curr), -1.5)
            n_phy_limit = int(100 * n_peak)

            n_start_calc = max(1, n_lower_band)
            n_end_calc = min(n_phy_limit, n_upper_band)

            if n_end_calc >= n_start_calc:
                if (n_end_calc - n_start_calc) > 3000:
                    n_calc = np.unique(np.geomspace(n_start_calc, n_end_calc, 3000).astype(int))
                else:
                    n_calc = np.arange(n_start_calc, n_end_calc + 1, dtype=int)

                fn_calc = n_calc * f_curr
                g_calc = g(n_calc, e_curr)
                hn_calc = (2.0 / n_calc) * np.sqrt(g_calc) * h0_val

                hc_avg_sq_contrib = (2.0 * np.square(fn_calc) * np.square(hn_calc) / f_curr) * dt

                interp_contrib = np.interp(unified_f_axis, fn_calc, hc_avg_sq_contrib, left=0, right=0)
                total_sq_strain += interp_contrib

    total_hc_avg = np.sqrt(total_sq_strain)

    # --- 7. SNR Calculation ---
    mask_snr = (unified_f_axis > 0) & (total_hc_avg > 0)
    snr_val = 0.0

    if np.any(mask_snr):
        f_integ = unified_f_axis[mask_snr]
        hc_integ = total_hc_avg[mask_snr]
        sn_vals = S_n_lisa(f_integ)  # S_n(f)

        integrand = (hc_integ ** 2) / (f_integ * sn_vals)
        snr_sq = np.trapz(integrand, np.log(f_integ))
        snr_val = np.sqrt(max(0, snr_sq))

    if verbose:
        print(f"   [Result] Integrated SNR: {snr_val:.4f}")

    sys_info = {
        'm1': m1, 'm2': m2, 'e0': e, 'tobs': tobs_years,
        'Dl': Dl,  # [New] Added Distance
        'f_start': f_start, 'f_end': f_end,
        'snr': snr_val
    }

    if plot:
        plot_evolving_spectrum(unified_f_axis, total_hc_avg, snapshots, sys_info)

    return unified_f_axis, total_hc_avg, snapshots, snr_val


def plot_evolving_spectrum(f_axis, hc_total, snapshots, sys_info, xlim=[1e-5, 1], ylim=[1e-23, 1e-15]):
    """
    绘图函数：显示谐波演化轨迹 (hnc) 和总积分频谱 (hc_avg)。
    [Update]: Added Distance (Dl) to info box.
    """
    import matplotlib.cm as cm

    plt.figure(figsize=(12, 9), dpi=100)

    # 1. Background Noise (zorder=0)
    f_bg = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 500)
    h_bg = np.sqrt(f_bg * S_n_lisa(f_bg))
    plt.loglog(f_bg, h_bg, 'k-', linewidth=2.0, label='LISA Sensitivity', alpha=0.25, zorder=0)

    # 2. Plot Total Integrated Spectrum (zorder=1)
    mask_pos = hc_total > 0
    plt.loglog(f_axis[mask_pos], hc_total[mask_pos], color='green', linewidth=3.5,
               label='Time-Integrated Spectrum', zorder=1, alpha=0.9)

    # 3. Plot Harmonic Tracks (zorder=2)
    harmonic_tracks = {}
    if len(snapshots) > 0:
        n_template = snapshots[0]['n']
        for n in n_template:
            harmonic_tracks[int(n)] = {'f': [], 'h': []}

        for snap in snapshots:
            for i, n in enumerate(snap['n']):
                harmonic_tracks[int(n)]['f'].append(snap['freq'][i])
                harmonic_tracks[int(n)]['h'].append(snap['hnc'][i])

    sorted_ns = sorted(harmonic_tracks.keys())
    colors = cm.jet(np.linspace(0, 1, len(sorted_ns)))

    num_labels = 8
    if len(sorted_ns) > num_labels:
        idx_to_label = np.linspace(0, len(sorted_ns) - 1, num_labels, dtype=int)
        ns_to_label = [sorted_ns[i] for i in idx_to_label]
    else:
        ns_to_label = sorted_ns

    if 2 in sorted_ns and 2 not in ns_to_label:
        ns_to_label.append(2)

    if len(sorted_ns) < 200:
        line_width = 1.5
        alpha_val = 0.8
    else:
        line_width = 1.0
        alpha_val = 0.5

    for idx, n in enumerate(sorted_ns):
        track = harmonic_tracks[n]
        f_arr = np.array(track['f'])
        h_arr = np.array(track['h'])

        if len(f_arr) > 1:
            plt.plot(f_arr, h_arr, color=colors[idx], linewidth=line_width, alpha=alpha_val, zorder=2)

            if n in ns_to_label:
                in_view_indices = np.where(
                    (f_arr >= xlim[0]) & (f_arr <= xlim[1]) &
                    (h_arr >= ylim[0]) & (h_arr <= ylim[1])
                )[0]

                if len(in_view_indices) > 0:
                    lbl_i = in_view_indices[0]
                    plt.text(f_arr[lbl_i], h_arr[lbl_i], f"n={n}",
                             fontsize=11, color=colors[idx], fontweight='bold',
                             verticalalignment='bottom', horizontalalignment='right', zorder=3)

    # Legends
    plt.plot([], [], color='blue', linewidth=line_width, alpha=alpha_val, label='Representative Harmonic Tracks')

    # 4. [System Info Box] with Dl and SNR
    snr_val = sys_info.get('snr', 0.0)

    line1 = r"$M_1=%.1f, M_2=%.1f\ M_\odot$" % (sys_info['m1'], sys_info['m2'])
    line2 = r"$e_0=%.3f, D_L=%.1f$ kpc" % (sys_info['e0'], sys_info['Dl'])  # Added Dl
    line3 = r"$T_{\rm obs}=%.1f$ yr" % (sys_info['tobs'])
    line4 = r"$f_{\rm orb}: %.2e \to %.2e$ Hz" % (sys_info['f_start'], sys_info['f_end'])
    line5 = r"${\bf SNR \approx %.1f}$" % (snr_val)

    info_str = "\n".join([line1, line2, line3, line4, line5])

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.text(0.97, 0.03, info_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=10)

    plt.xlabel('Frequency [Hz]', fontsize=14)
    plt.ylabel('Characteristic Strain', fontsize=14)
    plt.title('Evolving Binary Spectrum', fontsize=16)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, framealpha=0.95, fancybox=True)
    plt.tight_layout()
    plt.show()