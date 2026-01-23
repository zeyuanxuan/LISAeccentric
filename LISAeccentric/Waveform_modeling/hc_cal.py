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
# 2. 核心计算逻辑 (并行worker)
# ==========================================

def _core_calculator(args):
    """
    底层计算核心。
    Args: (m1_SI, m2_SI, a_SI, e, Dl_SI, tobs_SI)
    """
    m1, m2, a, e, Dl, tobs = args

    # 1. 计算基频
    forb = 1 / 2 / pi * np.sqrt(m1 + m2) * np.power(a, -3.0 / 2.0)

    # 2. 确定 n 的范围
    e_safe = min(e, 0.999)
    n_peak = np.sqrt(1 + e_safe) * np.power((1 - e_safe), -3.0 / 2.0)

    n_start = max(1, int(0.01 * n_peak))
    n_end = int(10 * n_peak)

    if n_end < n_start:
        return [], [], [], []

    n_arr = np.arange(n_start, n_end + 1, dtype=np.float64)

    df_dt_val = dforb_dt(m1, m2, a, e)
    h0_val = h0(a, m1, m2, Dl)

    g_vals = g(n_arr, e)
    hn_arr = 2 / n_arr * np.sqrt(g_vals) * h0_val
    fn_arr = n_arr * forb

    hnc2 = 2 * fn_arr ** 2 * hn_arr ** 2 / (n_arr * df_dt_val)
    hnc = np.sqrt(hnc2)  # 特征应变

    hcn_bkg = 2 / n_arr * np.sqrt(g_vals) * h0_val
    Snfvec = hcn_bkg ** 2 / forb

    hc_avg2 = 2 * fn_arr ** 2 * hn_arr ** 2 / (forb) * tobs
    hc_avg = np.sqrt(hc_avg2)

    return fn_arr, hnc, hc_avg, Snfvec


# ==========================================
# 3. 封装接口
# ==========================================

def calculate_single_system(m1, m2, a, e, Dl, tobs_years=1.0):
    """
    接口1: 单个系统计算
    输入: m1(Msun), m2(Msun), a(AU), e, Dl(Mpc)
    返回: List [fnlist, hclist, hcavglist, Snflist]
    """
    m1_si = m1 * m_sun
    m2_si = m2 * m_sun
    a_si = a * AU
    Dl_si = Dl * 1e3 * pc
    tobs_si = tobs_years * years

    args = (m1_si, m2_si, a_si, e, Dl_si, tobs_si)
    fn, hnc, hc_avg, snf = _core_calculator(args)

    return [fn, hc_avg, hnc, snf]


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
def process_population_batch(system_list_raw, tobs_years=1.0, n_cores=1):
    """
    接口2: 批量处理系统 (已修改为纯串行模式，移除了 Pool)
    输入:
        system_list_raw: 原始数据列表 list of lists
        n_cores: 保留参数以兼容接口，但不再使用。
    """

    pool_args = []
    tobs_si = tobs_years * years

    for item in system_list_raw:
        # item: [id, Dl, a, e, m1, m2]
        Dl_si = item[1] * 1e3 * pc
        a_si = item[2] * AU
        e_val = item[3]
        m1_si = item[4] * m_sun
        m2_si = item[5] * m_sun

        pool_args.append((m1_si, m2_si, a_si, e_val, Dl_si, tobs_si))

    # 定义公共频率轴
    logfrange = np.linspace(-6, 0, 1000)
    faxis = np.power(10., logfrange)
    Snf_tot = np.zeros_like(faxis)

    # 存储列表初始化
    all_fn_lists = []
    all_hcavg_lists = []
    all_hnc_lists = []

    # [修改] 不再使用多进程，使用单线程循环
    # 无论 n_cores 设置为多少，这里都强制串行
    print(f"Start calculation for {len(system_list_raw)} systems (Sequential mode)...")
    t_start = time.time()

    results = []
    # 直接使用简单的列表推导或循环
    for i, args in enumerate(pool_args):
        # 如果数据量非常大，可以取消下面这行的注释来查看进度
        # if i % 100 == 0: print(f"Processing {i}/{len(pool_args)}...")
        results.append(_core_calculator(args))

    # 结果聚合
    for res in results:
        fn_sys, hnc_sys, hc_avg_sys, Snf_sys = res

        if len(fn_sys) > 0:
            all_fn_lists.append(fn_sys)
            all_hcavg_lists.append(hc_avg_sys)
            all_hnc_lists.append(hnc_sys)

            # 背景噪声计算
            if len(fn_sys) > 1:
                Snf_interp = np.interp(faxis, fn_sys, Snf_sys, left=0, right=0)
                Snf_tot += Snf_interp

    print(f"Calculation done in {time.time() - t_start:.2f} seconds.")

    return [faxis, Snf_tot, all_fn_lists, all_hcavg_lists, all_hnc_lists]

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
# Main Execution
# ==========================================

if __name__ == '__main__':
    try:
        from GN_modeling import GN_BBH

        snapshot_data = GN_BBH.generate_snapshot_population(Gamma_rep=1.0, ync_age=6.0e6, ync_count=100, max_bh_mass=50)
    except ImportError:
        print("Creating dummy data...")
        snapshot_data = []
        for i in range(100):
            snapshot_data.append([
                i,
                8.0,  # Dl
                random.uniform(0.1, 2.0),  # a
                random.uniform(0.1, 0.9),  # e
                random.uniform(10, 50),  # m1
                random.uniform(10, 50)  # m2
            ])

    print(f"Input system list length: {len(snapshot_data)}")

    # 1. 单个测试
    test_res = calculate_single_system(m1=30, m2=30, a=0.5, e=0.6, Dl=8.0)
    print(f"Single test run produced {len(test_res[0])} harmonics.")

    # 2. 批量处理
    # 返回值包含：faxis, Snf, fn_lists, hcavg_lists, hnc_lists
    batch_results = process_population_batch(snapshot_data, tobs_years=1.0, n_cores=6)

    # 3. 绘图
    plot_simulation_results(batch_results)