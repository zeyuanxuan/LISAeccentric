# Import necessary packages
import numpy as np
import scipy.constants as sciconsts
import scipy.fftpack
import scipy.interpolate as sci_interpolate
import scipy.integrate as sci_integrate
from scipy.special import factorial
import scipy
import time
from scipy.optimize import root_scalar
from scipy.optimize import root
from math import sqrt
import matplotlib.pyplot as plt
import os
import time
import warnings  # Added to handle warning suppression
# ----------------- Numba Optimization Start -----------------
from numba import njit

import os
import shutil
import sys
import warnings


# -----------------------------------------------------------------------------
# [AUTO-FIX] Numba Cache Integrity Check
# -----------------------------------------------------------------------------
def _check_and_clear_numba_cache():
    """
    Automatically detects if Numba cache is corrupted (usually due to
    folder restructuring). If a path error occurs, it deletes the
    __pycache__ directory and prompts a restart.
    """
    try:
        from numba import njit

        # 1. Define a trivial test function to trigger compilation/cache loading
        @njit(cache=True)
        def _test_cache_integrity(x):
            return x + 1

        # 2. Attempt to run it. If cache is bad, this will crash.
        _test_cache_integrity(1)

    except Exception as e:
        # 3. Analyze the error. Look for typical path/import errors.
        error_str = str(e)
        is_path_error = "ModuleNotFoundError" in error_str or "No module named" in error_str

        if is_path_error:
            print("\n" + "!" * 60)
            print("[Auto-Fix] Numba cache path mismatch detected.")
            print("(This usually happens when files are moved or renamed)")
            print(f"Error Details: {e}")
            print("Attempting to clear old cache files...")

            # Locate the current file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(current_dir, "__pycache__")

            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    print(f"Successfully deleted cache directory: {cache_dir}")
                    print("!" * 60 + "\n")

                    # 4. CRITICAL: Exit and request restart.
                    # Python's internal import state is corrupted by the failed load.
                    # It is safer to restart the process than to continue.
                    print(">>> Cache cleared. Please RUN THE SCRIPT AGAIN to rebuild cache. <<<")
                    sys.exit(0)
                except OSError as err:
                    print(f"Failed to delete cache: {cache_dir}")
                    print(f"System Error: {err}")
            else:
                print("Could not find __pycache__ to delete. Please check environment.")
        else:
            # If it's a real syntax error, let it crash normally.
            raise e

# Run the check immediately
_check_and_clear_numba_cache()


m_sun = 1.98840987e30 * sciconsts.G / np.power(sciconsts.c, 3.0)
gama = 0.577215664901532860606512090082402431042159335
pi=sciconsts.pi
years = 365 * 24 * 3600.0
days=24*3600
pc = 3.261 * sciconsts.light_year/sciconsts.c
AU=sciconsts.au/sciconsts.c
C_val = sciconsts.c


def J(n, x):
    # scipy.special.jv 原生支持数组输入
    return scipy.special.jv(n, x)
def g(n, e):
    # n 可以是数组，e 是标量
    # 这里所有的计算都会自动广播 (Broadcasting)
    ne = n * e

    term1 = J(n - 2, ne) - 2 * e * J(n - 1, ne) + 2 / n * J(n, ne) + 2 * e * J(n + 1, ne) - J(n + 2, ne)
    term2 = J(n - 2, ne) - 2 * J(n, ne) + J(n + 2, ne)
    term3 = J(n, ne)

    # 注意：n**4 可能会很大，但在 float64 下通常没问题
    result = np.power(n, 4.0) / 32.0 * (
            np.power(term1, 2.0) +
            (1 - e * e) * np.power(term2, 2.0) +
            4.0 / (3.0 * n * n) * np.power(term3, 2.0)
    )
    return result
def h0(a, m1, m2, Dl):
    return np.sqrt(32 / 5) * m1 * m2 / Dl / a
def forb(M, a):
    return 1.0 / 2.0 / pi * np.sqrt(M) * np.power(a, -1.5)
def S_gal_N2A5(f):
    # 使用 np.where 或 boolean masking 来处理分段函数
    res = np.zeros_like(f)

    # 避免 log10(0) 或负数次幂的警告，可以先计算掩码
    # 这里的写法利用了 boolean indexing 赋值

    mask1 = (f >= 1.0e-5) & (f < 1.0e-3)
    res[mask1] = np.power(f[mask1], -2.3) * (10 ** -44.62) * 20.0 / 3.0

    mask2 = (f >= 1.0e-3) & (f < 10 ** -2.7)
    res[mask2] = np.power(f[mask2], -4.4) * (10 ** -50.92) * 20.0 / 3.0

    mask3 = (f >= 10 ** -2.7) & (f < 10 ** -2.4)
    res[mask3] = np.power(f[mask3], -8.8) * (10 ** -62.8) * 20.0 / 3.0

    mask4 = (f >= 10 ** -2.4) & (f <= 0.01)
    res[mask4] = np.power(f[mask4], -20.0) * (10 ** -89.68) * 20.0 / 3.0

    return res
def S_n_lisa(f):
    # 向量化版本的噪声计算
    m1_lisa = 5.0e9
    m2_lisa = C_val * 0.41 / m1_lisa / 2.0

    # 避免除以0，虽然 f 通常 > 0，但为了保险
    # 这里假设输入 f 都是正数

    noise_inst = 20.0 / 3.0 * (1 + np.power(f / m2_lisa, 2.0)) * (
            4.0 * (9.0e-30 / np.power(2 * pi * f, 4.0) * (1 + 1.0e-4 / f)) + 2.96e-23 + 2.65e-23
    ) / np.power(m1_lisa, 2.0)

    return noise_inst + S_gal_N2A5(f)
def tmerger_integral(m1, m2, a0, e0):
    m1=m1*m_sun
    m2=m2*m_sun
    a0=a0*AU
    beta = 64.0 / 5.0 * m1 * m2 * (m1 + m2)
    if beta == 0: return 1e99

    if e0 < 1e-4:
        return a0 ** 4 / (4.0 * beta)

    term_c0 = np.power(e0, 12.0 / 19.0) * np.power(1 + 121.0 / 304.0 * e0 ** 2, 870.0 / 2299.0)
    c0 = a0 * (1 - e0 ** 2) / term_c0

    # 使用 np.linspace 进行向量化积分
    steps = 1000
    e_vals = np.linspace(0, e0, steps, endpoint=False) + e0 / steps / 2.0  # 中点

    numer = np.power(e_vals, 29.0 / 19.0) * np.power(1 + 121.0 / 304.0 * e_vals ** 2, 1181.0 / 2299.0)
    denom = np.power(1 - e_vals ** 2, 1.5)
    integral_val = np.sum(numer / denom) * (e0 / steps)

    t_merger = (12.0 / 19.0) * (np.power(c0, 4.0) / beta) * integral_val
    return t_merger/years
def dSNR2dt_numpy(m1, m2, a, e, Dl):
    fmin = 1e-8
    fmax = 0.1
    fnumber = 16000

    f0 = 1. / 2. / pi * np.sqrt(m1 + m2) * np.power(a, -1.5)
    threshold = (fmax - fmin) / fnumber

    # --- High Eccentricity Mode ---
    if f0 * 10 < threshold:
        # print("Running Numpy High-E Mode")
        h = np.sqrt(32. / 5.) * m1 * m2 / Dl / a
        deltaf = (fmax - fmin) / fnumber

        # 1. 生成频率数组
        f_arr = np.linspace(fmin, fmax, fnumber)

        # 2. 计算对应的 n_avg
        rate = f_arr / f0
        n_avg = rate.astype(int)  # 向下取整

        # 3. 筛选有效点 (n_avg > 0)
        mask = n_avg > 0
        f_valid = f_arr[mask]
        n_valid = n_avg[mask]

        # 4. 向量化计算 g 和 Sn
        # 这一步会自动并行调用 Bessel 函数，非常快
        g_vals = g(n_valid, e)
        Sn_vals = S_n_lisa(f_valid)

        # 5. 积分累加
        # 避免除以0 (Sn_vals 可能有极小值，但 S_n_lisa_vec 只有 >1e-5 才有效)
        # 我们可以再加一层 mask 确保 Sn > 0
        valid_sn_mask = Sn_vals > 0

        term = g_vals[valid_sn_mask] / np.square(f_valid[valid_sn_mask]) / Sn_vals[valid_sn_mask]
        result = np.sum(term) * deltaf

        return result * 8 * h * h * f0

    # --- Low Eccentricity Mode ---
    else:
        # print("Running Numpy Low-E Mode")
        nmin = int(fmin / f0) + 1
        nmax = int(fmax / f0) + 2

        # 1. 生成 n 数组
        n_arr = np.arange(nmin, nmax)
        if len(n_arr) == 0:
            return 0.0

        # 2. 计算 fn, hn
        fn_arr = n_arr * forb(m1 + m2, a)

        val_h0 = h0(a, m1, m2, Dl)
        g_vals = g(n_arr, e)

        hn_arr = 2.0 / n_arr * np.sqrt(g_vals) * val_h0
        hnc2_arr = 4.0 * np.square(fn_arr) * np.square(hn_arr) / (2 * n_arr)

        # 3. 计算 Sn
        Sn_vals = S_n_lisa(fn_arr)

        # 4. 累加
        mask_sn = Sn_vals > 0
        # result = sum( hnc2 / fn^2 / Sn * n )
        terms = hnc2_arr[mask_sn] / np.square(fn_arr[mask_sn]) / Sn_vals[mask_sn] * n_arr[mask_sn]

        return np.sum(terms)
def SNR(m1, m2, a, e, Dl, tobs):
    m1=m1*m_sun
    m2=m2*m_sun
    a=a*AU
    Dl=Dl*1e3*pc
    tobs=tobs*years
    # 1. Merger Time Check
    tmerger = tmerger_lower(m1, m2, a, e)
    used_tobs = tobs

    if tmerger <= tobs:
        tmerger = tmerger_integral(m1/m_sun,m2/m_sun,a/AU,e)
        if tmerger <= tobs:
            print(f"[Warning] System evolves too fast! tmerger ({tmerger:.2e}) < tobs ({tobs:.2e}).")
            print(f"Approximation inaccurate. Adjusting tobs to : {tmerger}")
            used_tobs = tmerger

    # 2. Calculate Rate (Using Numpy Vectorization)
    rate_val = dSNR2dt_numpy(m1, m2, a, e, Dl)

    # 3. Final SNR^2
    final_snr2 = rate_val * used_tobs

    return np.sqrt(final_snr2)
def tmerger_lower(m1, m2, a, e):
    beta = 64 / 5 * m1 * m2 * (m1 + m2)
    tc = np.power(a, 4) / (4 * beta)
    t = tc * np.power(1 - e * e, 7 / 2)
    return t
def SNR_approx(m1, m2, a, e, Dl, tobs):
        h0max = np.sqrt(32 / 5) * m1 * m2 / (Dl * a * (1 - e))
        f0max = 2 * np.sqrt((m1 + m2) / (4 * pi * pi * np.power(a * (1 - e), 3.0)))
        sqrtsnf = np.sqrt(S_n_lisa(f0max))
        snrcur = h0max / sqrtsnf * np.sqrt(tobs * np.power(1 - e, 3 / 2))
        return snrcur
@njit(fastmath=True, cache=True)
def r2pn_numba(e, u, smr):
    cos_u = np.cos(u)
    e2 = e * e
    sqrt_1_e2 = np.sqrt(1 - e2)

    # 系数 1/(72*(1-e^2)^2)
    prefactor = 1.0 / (72.0 * (1 - e2) ** 2)

    term1 = (8 * smr ** 2 + 30 * smr + 72) * e ** 4
    term2 = (-16 * smr ** 2 - 876 * smr + 756) * e2
    term3 = (8 * smr ** 2 + 198 * smr + 360)

    term_cos = cos_u * (
            (-35 * smr ** 2 + 231 * smr - 72) * e ** 5 +
            (70 * smr ** 2 - 150 * smr - 468) * e ** 3 +
            (-35 * smr ** 2 + 567 * smr - 648) * e
    )

    term_sqrt = sqrt_1_e2 * (
            (360 - 144 * smr) * e2 + (144 * smr - 360) +
            cos_u * ((180 - 72 * smr) * e ** 3 + (72 * smr - 180) * e)
    )

    return prefactor * (term1 + term2 + term3 + term_cos + term_sqrt)
@njit(fastmath=True, cache=True)
def r3pn_numba(e, u, smr):
    pi_val = np.pi
    cos_u = np.cos(u)
    e2 = e * e

    prefactor = 1.0 / (181440.0 * (1 - e2) ** 3.5)

    term_const = (
            (-665280 * smr ** 2 + 1753920 * smr - 1814400) * e ** 6 +
            (725760 * smr ** 2 - 77490 * pi_val ** 2 * smr + 5523840 * smr - 3628800) * e ** 4 +
            (544320 * smr ** 2 + 154980 * pi_val ** 2 * smr - 14132160 * smr + 7257600) * e2 -
            604800 * smr ** 2 + 6854400 * smr
    )

    term_cos = cos_u * (
            (302400 * smr ** 2 - 1254960 * smr + 453600) * e ** 7 +
            (-1542240 * smr ** 2 - 38745 * pi_val ** 2 * smr + 6980400 * smr - 453600) * e ** 5 +
            (2177280 * smr ** 2 + 77490 * pi_val ** 2 * smr - 12373200 * smr + 4989600) * e ** 3 +
            (-937440 * smr ** 2 - 38745 * pi_val ** 2 * smr + 6647760 * smr - 4989600) * e
    )

    sqrt_term_content = (
            (-4480 * smr ** 3 - 25200 * smr ** 2 + 22680 * smr - 120960) * e ** 6 +
            (13440 * smr ** 3 + 4404960 * smr ** 2 + 116235 * pi_val ** 2 * smr - 12718296 * smr + 5261760) * e ** 4 +
            (-13440 * smr ** 3 + 2242800 * smr ** 2 + 348705 * pi_val ** 2 * smr - 19225080 * smr + 16148160) * e2 +
            4480 * smr ** 3 + 45360 * smr ** 2 - 8600904 * smr +
            cos_u * (
                    (-6860 * smr ** 3 + 550620 * smr ** 2 - 986580 * smr + 120960) * e ** 7 +
                    (20580 * smr ** 3 - 2458260 * smr ** 2 + 3458700 * smr - 2358720) * e ** 5 +
                    (
                                -20580 * smr ** 3 - 3539340 * smr ** 2 - 116235 * pi_val ** 2 * smr + 20173860 * smr - 16148160) * e ** 3 +
                    (6860 * smr ** 2 - 1220940 * smr ** 2 - 464940 * pi_val ** 2 * smr + 17875620 * smr - 4717440) * e
            ) +
            116235 * smr * pi_val ** 2 + 1814400
    )

    last_const = -77490 * smr * pi_val ** 2 - 1814400

    return prefactor * (term_const + term_cos + np.sqrt(1 - e2) * sqrt_term_content + last_const)
@njit(fastmath=True, cache=True)
def psi2pn_numba(e, u, smr):
    cos_u = np.cos(u)
    e2 = e * e
    sqrt_1_e2 = np.sqrt(1 - e2)
    denom = 12 * (1 - e2) ** 1.5 * (e * cos_u - 1) ** 5

    term1 = (-12 * smr ** 2 - 18 * smr) * e ** 6 + (20 * smr ** 2 - 26 * smr - 60) * e ** 4 + (
                -2 * smr ** 2 + 50 * smr + 75) * e2

    term_cos3 = cos_u ** 3 * ((-14 * smr ** 2 + 8 * smr - 147) * e ** 5 + (8 * smr ** 2 + 22 * smr + 42) * e ** 3)

    term_cos2 = cos_u ** 2 * ((17 * smr ** 2 - 17 * smr + 48) * e ** 6 + (-4 * smr ** 2 - 38 * smr + 153) * e ** 4 + (
                5 * smr ** 2 - 35 * smr + 114) * e2)

    term_cos1 = cos_u * ((-smr ** 2 + 97 * smr + 12) * e ** 5 + (-16 * smr ** 2 - 74 * smr - 81) * e ** 3 + (
                -smr ** 2 + 67 * smr - 246) * e)

    term_sqrt = sqrt_1_e2 * (
            e ** 3 * (36 * smr - 90) * cos_u ** 3 +
            ((180 - 72 * smr) * e ** 4 + (90 - 36 * smr) * e2) * cos_u ** 2 +
            ((144 * smr - 360) * e ** 3 + (90 - 36 * smr) * e) * cos_u +
            e2 * (180 - 72 * smr) + 36 * smr - 90
    )

    return (term1 + term_cos3 + term_cos2 - 36 * smr + term_cos1 + term_sqrt + 90) / denom
@njit(fastmath=True, cache=True)
def psi3pn_numba(e, u, eta):
    # eta is smr
    pi_val = np.pi
    cos_u = np.cos(u)
    cos_u2 = cos_u ** 2
    cos_u3 = cos_u ** 3
    cos_u4 = cos_u ** 4
    cos_u5 = cos_u ** 5
    e2 = e * e

    prefactor = 1.0 / (13440.0 * (1 - e2) ** 2.5 * (e * cos_u - 1) ** 7)

    # 为了避免代码过长导致的可读性问题，这里保持这种结构，但确保变量是本地的
    term_poly = (
            ((10080 * eta ** 3 + 40320 * eta ** 2 - 15120 * eta) * e ** 10 +
             (-52640 * eta ** 3 - 13440 * eta ** 2 + 483280 * eta) * e ** 8 +
             (84000 * eta ** 3 - 190400 * eta ** 2 - 17220 * pi_val ** 2 * eta - 50048 * eta - 241920) * e ** 6 +
             (-52640 * eta ** 3 + 516880 * eta ** 2 + 68880 * pi_val ** 2 * eta - 1916048 * eta + 262080) * e ** 4 +
             (4480 * eta ** 3 - 412160 * eta ** 2 - 30135 * pi_val ** 2 * eta + 553008 * eta + 342720) * e ** 2) +
            ((13440 * eta ** 3 + 94640 * eta ** 2 - 113680 * eta - 221760) * e ** 9 +
             (-11200 * eta ** 3 - 112000 * eta ** 2 + 12915 * pi_val ** 2 * eta + 692928 * eta - 194880) * e ** 7 +
             (
                         4480 * eta ** 3 + 8960 * eta ** 2 - 43050 * pi_val ** 2 * eta + 1127280 * eta - 147840) * e ** 5) * cos_u5 +
            ((-16240 * eta ** 3 + 12880 * eta ** 2 + 18480 * eta) * e ** 10 +
             (16240 * eta ** 3 - 91840 * eta ** 2 + 17220 * pi_val ** 2 * eta - 652192 * eta + 100800) * e ** 8 +
             (-55440 * eta ** 3 + 34160 * eta ** 2 - 30135 * pi_val ** 2 * eta - 2185040 * eta + 2493120) * e ** 6 +
             (
                         21840 * eta ** 3 + 86800 * eta ** 2 + 163590 * pi_val ** 2 * eta - 5713888 * eta + 228480) * e ** 4) * cos_u4 +
            ((560 * eta ** 3 - 137200 * eta ** 2 + 388640 * eta + 241920) * e ** 9 +
             (30800 * eta ** 3 - 264880 * eta ** 2 - 68880 * pi_val ** 2 * eta + 624128 * eta + 766080) * e ** 7 +
             (66640 * eta ** 3 + 612080 * eta ** 2 - 8610 * pi_val ** 2 * eta + 6666080 * eta - 6652800) * e ** 5 +
             (-30800 * eta ** 3 - 294000 * eta ** 2 - 223860 * pi_val ** 2 * eta + 9386432 * eta) * e ** 3) * cos_u3 +
            67200 * eta ** 2 +
            ((4480 * eta ** 3 - 20160 * eta ** 2 + 16800 * eta) * e ** 10 +
             (3920 * eta ** 3 + 475440 * eta ** 2 - 17220 * pi_val ** 2 * eta + 831952 * eta - 725760) * e ** 8 +
             (-75600 * eta ** 3 + 96880 * eta ** 2 + 154980 * pi_val ** 2 * eta - 3249488 * eta - 685440) * e ** 6 +
             (5040 * eta ** 3 - 659120 * eta ** 2 + 25830 * pi_val ** 2 * eta - 7356624 * eta + 6948480) * e ** 4 +
             (
                         -5040 * eta ** 3 + 190960 * eta ** 2 + 137760 * pi_val ** 2 * eta - 7307920 * eta + 107520) * e ** 2) * cos_u2 -
            761600 * eta +
            ((-2240 * eta ** 3 - 168000 * eta ** 2 - 424480 * eta) * e ** 9 +
             (28560 * eta ** 3 + 242480 * eta ** 2 + 34440 * pi_val ** 2 * eta - 1340224 * eta + 725760) * e ** 7 +
             (-33040 * eta ** 3 - 754880 * eta ** 2 - 172200 * pi_val ** 2 * eta + 5458480 * eta - 221760) * e ** 5 +
             (40880 * eta ** 3 + 738640 * eta ** 2 + 30135 * pi_val ** 2 * eta + 1554048 * eta - 2936640) * e ** 3 +
             (-560 * eta ** 3 - 100240 * eta ** 2 - 43050 * pi_val ** 2 * eta + 3284816 * eta - 389760) * e) * cos_u +
            np.sqrt(1 - e2) * (
                    ((-127680 * eta ** 2 + 544320 * eta - 739200) * e ** 7 +
                     (-53760 * eta ** 2 - 8610 * pi_val ** 2 * eta + 674240 * eta - 67200) * e ** 5) * cos_u5 +
                    ((161280 * eta ** 2 - 477120 * eta + 537600) * e ** 8 +
                     (477120 * eta ** 2 + 17220 * pi_val ** 2 * eta - 2894080 * eta + 2217600) * e ** 6 +
                     (268800 * eta ** 2 + 25830 * pi_val ** 2 * eta - 2721600 * eta + 1276800) * e ** 4) * cos_u4 +
                    ((-524160 * eta ** 2 + 1122240 * eta - 940800) * e ** 7 +
                     (-873600 * eta ** 2 - 68880 * pi_val ** 2 * eta + 7705600 * eta - 3897600) * e ** 5 +
                     (-416640 * eta ** 2 - 17220 * pi_val ** 2 * eta + 3357760 * eta - 3225600) * e ** 3) * cos_u3 +
                    ((604800 * eta ** 2 - 504000 * eta - 403200) * e ** 6 +
                     (1034880 * eta ** 2 + 103320 * pi_val ** 2 * eta - 11195520 * eta + 5779200) * e ** 4 +
                     (174720 * eta ** 2 - 17220 * pi_val ** 2 * eta - 486080 * eta + 2688000) * e ** 2) * cos_u2 +
                    ((-282240 * eta ** 2 - 450240 * eta + 1478400) * e ** 5 +
                     (-719040 * eta ** 2 - 68880 * pi_val ** 2 * eta + 8128960 * eta - 5040000) * e ** 3 +
                     (94080 * eta ** 2 + 25830 * pi_val ** 2 * eta - 1585920 * eta - 470400) * e) * cos_u -
                    67200 * eta ** 2 + 761600 * eta +
                    e ** 4 * (40320 * eta ** 2 + 309120 * eta - 672000) +
                    e ** 2 * (208320 * eta ** 2 + 17220 * pi_val ** 2 * eta - 2289280 * eta + 1680000) -
                    8610 * eta * pi_val ** 2 - 201600) + 8610 * eta * pi_val ** 2 + 201600
    )
    return prefactor * term_poly
@njit(fastmath=True, cache=True)
def get_residual_and_deriv(u, l, e, x, ephi, smr, PN_orbit):
    """ 计算开普勒方程残差及导数 (增加数值保护) """
    pi_val = np.pi
    sinu = np.sin(u)
    cosu = np.cos(u)

    # [保护1] 防止 e*cosu 极度接近 1 导致导数为 0
    one_minus_ecosu = 1.0 - e * cosu
    if one_minus_ecosu < 1e-12:
        one_minus_ecosu = 1e-12

    sqrt_1_e2 = np.sqrt(1 - e ** 2)

    val = -l + u - e * sinu
    deriv = one_minus_ecosu  # 这是 0PN 导数

    if PN_orbit >= 2:
        # [保护2] 防止 ephi 接近 0 导致除零 (ephi=0 时 betaphi -> 0)
        if np.abs(ephi) < 1e-10:
            betaphi = 0.5 * ephi  # 泰勒展开近似
        else:
            betaphi = (1 - np.sqrt(1 - ephi ** 2)) / ephi

        denom_atan = 1 - cosu * betaphi
        if np.abs(denom_atan) < 1e-15: denom_atan = 1e-15

        u_minus_v = -2 * np.arctan((sinu * betaphi) / denom_atan)

        denominator_2pn = 8 * sqrt_1_e2 * one_minus_ecosu
        term1 = -12 * (2 * smr - 5) * u_minus_v * (e * cosu - 1)
        term2 = -e * sqrt_1_e2 * (smr - 15) * smr * sinu
        l_2PN = (term1 + term2) / denominator_2pn
        val += l_2PN * x ** 2

    if PN_orbit >= 3:
        denominator_3pn = 6720 * (1 - e ** 2) ** 1.5 * one_minus_ecosu ** 3
        term1_3pn = (35 * (96 * (11 * smr ** 2 - 29 * smr + 30) * e ** 2 + 960 * smr ** 2 + smr * (
                -13184 + 123 * pi_val ** 2) + 8640) * u_minus_v * (e * cosu - 1) ** 3)
        term2_3pn = (3360 * (
                -12 * (2 * smr - 5) * u_minus_v + 12 * e * (2 * smr - 5) * cosu * u_minus_v + e * sqrt_1_e2 * (
                smr - 15) * smr * sinu) * (e * cosu - 1) ** 2)
        term3_3pn = (e * sqrt_1_e2 * (140 * (13 * e ** 4 - 11 * e ** 2 - 2) * smr ** 3 - 140 * (
                73 * e ** 4 - 325 * e ** 2 + 444) * smr ** 2 + (
                                              3220 * e ** 4 - 148960 * e ** 2 - 4305 * pi_val ** 2 + 143868) * smr + e ** 2 * (
                                              1820 * (e ** 2 - 1) * smr ** 3 - 140 * (
                                              83 * e ** 2 + 109) * smr ** 2 - (
                                                      1120 * e ** 2 + 4305 * pi_val ** 2 + 752) * smr + 67200) * cosu ** 2 - 2 * e * (
                                              1960 * (e ** 2 - 1) * smr ** 3 + 6720 * (e ** 2 - 5) * smr ** 2 + (
                                              -71820 * e ** 2 - 4305 * pi_val ** 2 + 69948) * smr + 67200) * cosu + 67200) * sinu)
        l_3PN = (term1_3pn + term2_3pn + term3_3pn) / denominator_3pn
        val += l_3PN * x ** 3

    return val, deriv
@njit(fastmath=True, cache=True)
def solve_u_series_robust(l_vec, e_vec, x_vec, ephi_vec, smr, PN_orbit):
    """ 加速求解 u(t)，包含步长限制和除零保护 """
    n = len(l_vec)
    u_vec = np.empty(n, dtype=np.float64)
    u_vec[0] = l_vec[0]  # Fallback initial

    for i in range(n):
        l = l_vec[i]
        e = e_vec[i]
        x = x_vec[i]
        ephi = ephi_vec[i]

        # 初始猜测 (Bracket Search)
        if i == 0:
            low = l - 1.5
            high = l + 1.5
        else:
            # 线性预测
            pred = u_vec[i - 1] + (l - l_vec[i - 1])
            low = pred - 1.0
            high = pred + 1.0

        # 快速定位区间
        for _ in range(20):
            fl, _ = get_residual_and_deriv(low, l, e, x, ephi, smr, PN_orbit)
            if fl < 0: break
            low -= 1.0
        for _ in range(20):
            fh, _ = get_residual_and_deriv(high, l, e, x, ephi, smr, PN_orbit)
            if fh > 0: break
            high += 1.0

        u_cur = 0.5 * (low + high)
        if i > 0 and (low < u_vec[i - 1] + (l - l_vec[i - 1]) < high):
            u_cur = u_vec[i - 1] + (l - l_vec[i - 1])

        # Newton-Raphson Loop
        for iter_j in range(50):
            f_val, df_du = get_residual_and_deriv(u_cur, l, e, x, ephi, smr, PN_orbit)

            if np.abs(f_val) < 1e-12: break

            # [关键修复] 严格的除零保护，不用 == 0
            if np.abs(df_du) < 1e-15:
                # 保持符号
                if df_du < 0:
                    df_du = -1e-15
                else:
                    df_du = 1e-15

            step = f_val / df_du

            # [关键修复] 步长限制 (Damped Newton)，防止飞出太远
            # 如果导数估计很差（因为没包含 PN 项），全步长会导致震荡或发散
            if step > 1.5: step = 1.5
            if step < -1.5: step = -1.5

            u_new = u_cur - step

            # Bisection Fallback (如果牛顿法跳出了区间)
            if not (low < u_new < high):
                u_new = 0.5 * (low + high)
                step = u_cur - u_new  # 更新 effective step 用于判断收敛

            u_cur = u_new

            # 更新区间
            f_new, _ = get_residual_and_deriv(u_cur, l, e, x, ephi, smr, PN_orbit)
            if f_new * fl > 0:  # 同号，移动下界
                low = u_cur
                fl = f_new
            else:
                high = u_cur

            if np.abs(high - low) < 1e-12: break

        u_vec[i] = u_cur
    return u_vec
@njit(fastmath=True, cache=True)
def compute_h_arrays_full(evec, uvec, xvec, phi, M, smr, R, theta, dt, PN_orbit):
    """
    全链路 Numba 加速：计算 r, dpsi, psi, h+, hx
    * 加入 Kahan Summation 算法以保护长时积分的相位精度
    """
    n = len(evec)
    rvec = np.empty(n, dtype=np.float64)
    dpsi_dt_vec = np.empty(n, dtype=np.float64)
    psiresult = np.empty(n, dtype=np.float64)
    hplusv = np.empty(n, dtype=np.float64)
    hcrossv = np.empty(n, dtype=np.float64)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    sin2_theta = sin_theta * sin_theta
    cos2_plus_1 = cos_theta * cos_theta + 1.0
    amp_fac = -M * smr / R

    # 1. 物理量计算 Loop (r, dpsi/dt)
    for i in range(n):
        e = evec[i]
        u = uvec[i]
        x = xvec[i]

        cos_u = np.cos(u)
        e2 = e * e
        sqrt_1_e2 = np.sqrt(1 - e2)
        ecos_m1 = e * cos_u - 1.0

        # 0PN / Newtonian
        r = (1.0 - e * cos_u) / x
        dp = (x ** 1.5) * sqrt_1_e2 / (ecos_m1 ** 2)

        # 1PN
        if PN_orbit >= 1:
            term_r_1pn = 2 * ecos_m1 / (e2 - 1) + (2 * (smr - 9) + e * (7 * smr - 6) * cos_u) / 6.0
            r += term_r_1pn
            term_dp_1pn = - (x ** 2.5) * e * (smr - 4) * (e - cos_u) / (sqrt_1_e2 * ecos_m1 ** 3)
            dp += term_dp_1pn

        # 2PN
        if PN_orbit >= 2:
            r += x * r2pn_numba(e, u, smr)
            dp += (x ** 3.5) * psi2pn_numba(e, u, smr)

        # 3PN
        if PN_orbit >= 3:
            r += x * x * r3pn_numba(e, u, smr)
            dp += (x ** 4.5) * psi3pn_numba(e, u, smr)

        rvec[i] = r * M
        dpsi_dt_vec[i] = dp / M

    # 2. 积分计算 Psi (使用 Kahan Summation 保护精度)
    # 普通累加: sum += val
    # Kahan累加: 引入补偿变量 c，记录低位丢失的信息
    current_psi = 0.0
    psi_compensation = 0.0  # 补偿变量
    psiresult[0] = 0.0

    for i in range(1, n):
        # 梯形法则增量
        delta_psi = 0.5 * (dpsi_dt_vec[i - 1] + dpsi_dt_vec[i]) * dt

        # Kahan Summation Core Logic
        y = delta_psi - psi_compensation
        t = current_psi + y
        psi_compensation = (t - current_psi) - y
        current_psi = t

        psiresult[i] = current_psi

    # 3. 波形 Strain 计算
    for i in range(n):
        if i == 0:
            rd = (rvec[1] - rvec[0]) / dt
        elif i == n - 1:
            rd = (rvec[n - 1] - rvec[n - 2]) / dt
        else:
            rd = (rvec[i + 1] - rvec[i - 1]) / (2 * dt)

        r = rvec[i]
        psid = dpsi_dt_vec[i]

        # 注意：这里我们计算 cos(2*psi)，如果 psi 极其巨大 (e.g. > 1e16)，
        # float64 依然会丢失 modulo 2pi 的精度。
        # 但对于 LISA/LIGO 应用，psi 通常在 1e6 ~ 1e9 量级，
        # Kahan Summation 已经足以保证 delta_psi 不被大数吞没。
        psit = psiresult[i] - phi

        cos2p = np.cos(2 * psit)
        sin2p = np.sin(2 * psit)

        common_term = -rd * rd + r * r * psid * psid + M / r

        # H_plus
        term1 = cos2p * common_term + 2 * r * rd * psid * sin2p
        term2 = -rd * rd - r * r * psid * psid + M / r
        hplusv[i] = amp_fac * (cos2_plus_1 * term1 + term2 * sin2_theta)

        # H_cross
        term_cross = common_term * sin2p - 2 * r * cos2p * rd * psid
        hcrossv[i] = 2 * amp_fac * cos_theta * term_cross

    return rvec, dpsi_dt_vec, psiresult, hplusv, hcrossv
def eccGW_waveform(f00, e0, timescale, m1, m2, theta, phi, R, l0=0, ts=None, PN_orbit=3, PN_reaction=2,
                    N=50, max_memory_GB=16.0, verbose=True):
    # 如果 verbose 为 True，vprint 就是 print；否则 vprint 是一个什么都不做的空函数
    vprint = print if verbose else lambda *args, **kwargs: None

    m1=m1*m_sun
    m2=m2*m_sun
    R=R*1e3*pc
    timescale=timescale*years
    # === [新增保护逻辑] 零偏心率自动修正 ===
    if e0 == 0:
        print("Warning: e0=0 is not supported by this eccentric template (singularities in PN terms).")
        print("         Automatically resetting e0 to 1e-5 to maintain stability.")
        e0 = 1e-5
    emin=0
    # ========================================
    vprint('================Ecc Inspiral GW waveform================')
    # =================================================================
    # 1. 内部辅助函数定义
    # =================================================================
    def J(n, x):
        return scipy.special.jv(n, x)
    def Jtilt(n, x):
        return 0.5 * (J(n - 1, x) - J(n + 1, x))
    #print(f"Loading 1.5PN Tail enhancement table (e0={e0:.6f})...")
    # --- 缓存与预计算逻辑 ---
    dist_to_1 = 1.0 - e0
    if dist_to_1 < 0: dist_to_1 = 0
    margin = min(0.0001, dist_to_1 * 0.05)
    required_safe_e0 = e0 + margin
    limit_e = 0.99999
    if required_safe_e0 > limit_e: required_safe_e0 = limit_e
    # [FIX] 使用 os.path.dirname(__file__) 锁定当前脚本目录
    # 这样无论你在哪里运行脚本，它都会在脚本所在的目录下寻找/创建 .npz 文件
    # 如果是交互式环境(无 __file__)，则回退到 getcwd
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_script_dir = os.getcwd()

    CACHE_FILE = os.path.join(current_script_dir, 'eccGW_1p5PN_table.npz')
    table_loaded = False
    kE_vals = None
    kJ_vals = None
    e_grid = None
    if os.path.exists(CACHE_FILE):
        try:
            data = np.load(CACHE_FILE)
            cached_e_grid = data['e_grid']
            cached_max_e = cached_e_grid[-1]
            if cached_max_e >= required_safe_e0 - 1e-8:
                #print(f"   Found valid table (max_e={cached_max_e:.6f}). Loading...")
                e_grid = cached_e_grid
                kE_vals = data['kE_vals']
                kJ_vals = data['kJ_vals']
                table_loaded = True
            else:
                pass
                #print(f"   [Cache Miss] Range insufficient. Recomputing...")
        except Exception as e:
            pass
            #print(f"   [Cache Warning] Failed to load cache: {e}. Recomputing...")
    if not table_loaded:
        def compute_kappa_single_point(e):
            e2 = e * e
            if e < 1e-6:
                return 1.0 + 1.625 * e2, 1.0 + 0.875 * e2
            local_N = int(30.0 / np.power(1.0 - e, 1.5)) + 20
            if local_N > 100000: local_N = 100000

            inv_e = 1.0 / e
            inv_e2 = inv_e * inv_e
            inv_e4 = inv_e2 * inv_e2

            cE1 = -e2 - 3 * inv_e2 + inv_e4 + 3
            cE2 = 1.0 / 3.0 - inv_e2 + inv_e4
            cE3 = -3 * e - 4 * inv_e2 * inv_e + 7 * inv_e
            cE4 = e2 + inv_e2 - 2
            cE5 = inv_e2 - 1

            sqrt_1_e2 = np.sqrt(1 - e2)
            inv_e3 = inv_e2 * inv_e
            cJ1 = -2 * inv_e4 - 1 + 3 * inv_e2
            cJ2 = 2 * (e + inv_e3 - 2 * inv_e)
            cJ3 = -inv_e + 2 * inv_e3
            cJ4 = 2 * (1 - inv_e2)

            sum_E = 0.0
            sum_J = 0.0

            for p in range(1, local_N + 1):
                pe = p * e
                jpe = scipy.special.jv(p, pe)
                jp_minus_1 = scipy.special.jv(p - 1, pe)
                jtpe = jp_minus_1 - inv_e * jpe
                p2 = p * p
                term_E = ((cE1 * p2 + cE2) * jpe ** 2 + cE3 * p * jtpe * jpe + (cE4 * p2 + cE5) * jtpe ** 2)
                current_add_E = 0.25 * (p * p2) * term_E
                sum_E += current_add_E
                term_J = (cJ1 * p * jpe ** 2 + (cJ2 * p2 + cJ3) * jtpe * jpe + cJ4 * p * jtpe ** 2)
                current_add_J = 0.5 * p2 * sqrt_1_e2 * term_J
                sum_J += current_add_J
                if p > 20:
                    abs_sum_E = abs(sum_E)
                    if abs_sum_E > 1e-100:
                        if abs(current_add_E) / abs_sum_E < 1e-13: break
                    elif abs(current_add_E) < 1e-13:
                        break
            return sum_E, sum_J

        target_max_e = max(0.95, required_safe_e0)
        if target_max_e > limit_e: target_max_e = limit_e
        print(f"   Fast-computation table not found. Building new table up to e={target_max_e:.6f}...")

        grid_size = 500
        split_point = max(0.9, target_max_e - 0.05)
        if split_point > target_max_e: split_point = target_max_e * 0.9
        e_grid_lin = np.linspace(0, split_point, int(grid_size * 0.3))
        log_start = np.log10(1.0 - split_point)
        log_end = np.log10(1.0 - target_max_e)
        e_grid_log = 1.0 - np.logspace(log_start, log_end, int(grid_size * 0.7))
        e_grid_log = np.sort(e_grid_log)
        e_grid = np.unique(np.concatenate((e_grid_lin, e_grid_log)))
        if e_grid[-1] < target_max_e: e_grid = np.append(e_grid, target_max_e)

        t_start_table = time.time()
        n_grid = len(e_grid)
        kE_vals = np.empty(n_grid)
        kJ_vals = np.empty(n_grid)
        for i in range(n_grid):
            val = e_grid[i]
            kE, kJ = compute_kappa_single_point(val)
            kE_vals[i] = kE
            kJ_vals[i] = kJ
            if i % 100 == 0:
                print(f"     Computing... {i}/{n_grid} (e={val:.5f})")
        vprint(f"   Table computed in {time.time() - t_start_table:.2f}s.")
        try:
            np.savez(CACHE_FILE, e_grid=e_grid, kE_vals=kE_vals, kJ_vals=kJ_vals)
            print(f"   Table saved to '{CACHE_FILE}'.")
        except Exception as e:
            print(f"   [Warning] Failed to save cache: {e}")

    kappaE_interp = sci_interpolate.interp1d(e_grid, kE_vals, kind='cubic', fill_value="extrapolate")
    kappaJ_interp = sci_interpolate.interp1d(e_grid, kJ_vals, kind='cubic', fill_value="extrapolate")

    def kappaE(e):
        if e > e_grid[-1]: return kE_vals[-1]
        if e < 1e-4: return 1.0
        return kappaE_interp(e)

    def kappaJ(e):
        if e > e_grid[-1]: return kJ_vals[-1]
        if e < 1e-4: return 1.0
        return kappaJ_interp(e)

    # --- 物理演化方程 ---
    def E2PN(e):
        return -e * smr / (30240 * np.power(1 - e * e, 9 / 2)) * (
                (2758560 * smr * smr - 4344852 * smr + 3786543) * np.power(e, 6.0) + (
                42810096 * smr * smr - 78112266 * smr + 46579718) * np.power(e, 4.0) + (
                        48711348 * smr * smr - 35583228 * smr - 36993396) * e * e + 4548096 * smr * smr + np.sqrt(
            1 - e * e) * ((2847600 - 1139040 * smr) * np.power(e, 4.0) + (
                35093520 - 14037408 * smr) * e * e - 5386752 * smr + 13466880) + 13509360 * smr - 15198032)

    def e_phi_func(et, x):
        e_phi_1PN = -et * (smr - 4)
        term1 = (41 * smr ** 2 - 659 * smr + 1152) * et ** 2
        term2 = 4 * smr ** 2 + 68 * smr
        term3 = np.sqrt(1 - et ** 2) * (288 * smr - 720)
        e_phi_2PN = (et / (96 * (et ** 2 - 1))) * (term1 + term2 + term3 - 1248)
        denominator = 26880 * (1 - et ** 2) ** (5 / 2)
        term1_3 = ((13440 * smr ** 2 + 483840 * smr - 940800) * et ** 4 + (
                255360 * smr ** 2 + 17220 * np.pi ** 2 * smr - 2880640 * smr + 2688000) * et ** 2 - 268800 * smr ** 2 + 2396800 * smr)
        term2_3 = np.sqrt(1 - et ** 2) * ((1050 * smr ** 3 - 134050 * smr ** 2 + 786310 * smr - 860160) * et ** 4 + (
                -18900 * smr ** 3 + 553980 * smr ** 2 + 4305 * np.pi ** 2 * smr - 1246368 * smr + 2042880) * et ** 2 + 276640 * smr ** 2 + 2674480 * smr - 17220 * smr * np.pi ** 2 - 1451520)
        e_phi_3PN = -et * (term1_3 + term2_3 - 17220 * smr * np.pi ** 2 - 1747200) / denominator
        return et + e_phi_1PN * x + e_phi_2PN * x ** 2 + e_phi_3PN * x ** 3

    def dx_de(x, e):
        if 1 - e * e <= 1e-8: return 0
        xdot = np.power(x, 5.0) * 2 * (37 * np.power(e, 4.0) + 292 * e * e + 96) * smr / (
                15 * np.power(1 - e * e, 7 / 2))
        if PN_reaction >= 1:
            xdot += np.power(x, 6.0) * smr / (420 * np.power(1 - e * e, 9 / 2)) * (
                    -(8288 * smr - 11717) * np.power(e, 6.0) - 14 * (10122 * smr - 12217) * np.power(e,
                                                                                                     4.0) - 120 * (
                            1330 * smr - 731) * e * e - 16 * (924 * smr + 743))
        if PN_reaction >= 1.5:
            xdot += np.power(x, 13 / 2) * 256 / 5 * smr * pi * kappaE(e)
        if PN_reaction >= 2:
            xdot += np.power(x, 7.0) * smr / (45360 * np.power(1 - e * e, 11 / 2)) * (
                    (1964256 * smr * smr - 3259980 * smr + 3523113) * np.power(e, 8.0) + (
                    64828848 * smr * smr - 123108426 * smr + 83424402) * np.power(e, 6.0) + (
                            16650606060 * smr * smr - 207204264 * smr + 783768) * np.power(e, 4.0) + (
                            61282032 * smr * smr + 15464736 * smr - 92846560) * e * e + 1903104 * smr * smr + np.sqrt(
                1 - e * e) * ((2646000 - 1058400 * smr) * np.power(e, 6.0) + (
                    64532160 - 25812864 * smr) * e * e - 580608 * smr + 1451520) + 4514976 * smr - 360224)
        xdot = xdot / M
        edot = np.power(x, 4.0) * (-e * (121 * e * e + 304) * smr / (15 * np.power(1 - e * e, 5 / 2)))
        if PN_reaction >= 1:
            edot += np.power(x, 5.0) * e * smr / (2520 * np.power(1 - e * e, 7 / 2)) * (
                    (93184 * smr - 125361) * np.power(e, 4.0) + 12 * (54271 * smr - 59834) * e * e + 8 * (
                    28588 * smr + 8451))
        if PN_reaction >= 1.5:
            edot += np.power(x, 11 / 2) * 128 * smr * pi / 5 / e * (
                    (e * e - 1) * kappaE(e) + np.sqrt(1 - e * e) * kappaJ(e))
        if PN_reaction >= 2:
            edot += np.power(x, 6.0) * E2PN(e)
        edot = edot / M
        if edot == 0: return 0
        return xdot / edot

    def x_e(e):
        try:
            val = xe1(e)
            if np.isnan(val): return 1.0 / 6.0
            if val >= 1.0 / 6.0: return False
            if val * 6.0 > (1.0 - e): return False
            return val
        except:
            return False

    def de_dt(e, t):
        x = x_e(e)
        if x is False: return 0
        edot = np.power(x, 4.0) * (-e * (121 * e * e + 304) * smr / (15 * np.power(1 - e * e, 5 / 2)))
        if PN_reaction >= 1: edot += np.power(x, 5.0) * e * smr / (2520 * np.power(1 - e * e, 7 / 2)) * (
                (93184 * smr - 125361) * np.power(e, 4.0) + 12 * (54271 * smr - 59834) * e * e + 8 * (
                28588 * smr + 8451))
        if PN_reaction >= 1.5: edot += np.power(x, 11 / 2) * 128 * smr * pi / 5 / e * (
                (e * e - 1) * kappaE(e) + np.sqrt(1 - e * e) * kappaJ(e))
        if PN_reaction >= 2: edot += np.power(x, 6.0) * E2PN(e)
        return edot / M

    def dx_dt(x, t):
        try:
            e = e_t(t)
        except NameError:
            return 0  # Should be fixed by logic order below

        if e <= emin or e >= e0:
            if x > 1.0 / 6.0: return 0
        xdot = np.power(x, 5.0) * 2 * (37 * np.power(e, 4.0) + 292 * e * e + 96) * smr / (
                15 * np.power(1 - e * e, 7 / 2))
        if PN_reaction >= 1: xdot += np.power(x, 6.0) * smr / (420 * np.power(1 - e * e, 9 / 2)) * (
                -(8288 * smr - 11717) * np.power(e, 6.0) - 14 * (10122 * smr - 12217) * np.power(e, 4.0) - 120 * (
                1330 * smr - 731) * e * e - 16 * (924 * smr + 743))
        if PN_reaction >= 1.5: xdot += np.power(x, 13 / 2) * 256 / 5 * smr * pi * kappaE(e)
        if PN_reaction >= 2:
            xdot += np.power(x, 7.0) * smr / (45360 * np.power(1 - e * e, 11 / 2)) * (
                    (1964256 * smr * smr - 3259980 * smr + 3523113) * np.power(e, 8.0) + (
                    64828848 * smr * smr - 123108426 * smr + 83424402) * np.power(e, 6.0) + (
                            16650606060 * smr * smr - 207204264 * smr + 783768) * np.power(e, 4.0) + (
                            61282032 * smr * smr + 15464736 * smr - 92846560) * e * e + 1903104 * smr * smr + np.sqrt(
                1 - e * e) * ((2646000 - 1058400 * smr) * np.power(e, 6.0) + (
                    64532160 - 25812864 * smr) * e * e - 580608 * smr + 1451520) + 4514976 * smr - 360224)
        return xdot / M

    def dl_dt(l, t):
        try:
            x = x_t(t);
            e = e_t(t)
        except:
            return 0
        result = (np.power(x, 3 / 2))
        if PN_orbit >= 1: result += np.power(x, 5 / 2) * 3 / (e * e - 1)
        if PN_orbit >= 2: result += np.power(x, 7 / 2) * ((26 * smr - 51) * e * e + 28 * smr - 18) / (
                4 * np.power(e * e - 1, 2.0))
        if PN_orbit >= 3: result += np.power(x, 9 / 2) * (-1) / (128 * np.power(1 - e * e, 7 / 2)) * (
                (1536 * smr - 3840) * np.power(e, 4.0) + (1920 - 768 * smr) * e * e - 768 * smr + np.sqrt(
            1 - e * e) * ((1040 * smr * smr - 1760 * smr + 2496) * np.power(e, 4.0) + (
                5120 * smr * smr + 123 * pi * pi * smr - 17856 * smr + 8544) * e * e + 896 * smr * smr - 14624 * smr + 492 * smr * pi * pi - 192) + 1920)
        return result / M

    def deltafvalue(a, e, M):
        n = np.power(a, -3 / 2) * np.sqrt(M)
        Porb = 1 / (n / (2 * pi))
        return 6 * np.power(2 * pi, 2 / 3) / (1 - e * e) * np.power(M, 2 / 3) * np.power(Porb, -5 / 3)

    # =================================================================
    # 2. 初始化与预检查
    # =================================================================
    M = m1 + m2
    smr = m1 * m2 / M / M
    a0 = np.power((m1 + m2) / (2 * pi * f00) ** 2, 1 / 3)
    deltaf = deltafvalue(a0, e0, M)
    f0 = f00 + deltaf / 2
    omega0 = f0 * 2 * pi
    x0 = np.power((m1 + m2) * omega0, 2 / 3)
    vprint(f'PN_EOM = {PN_orbit}; PN_Reaction = {PN_reaction}')
    vprint(f'm1, m2 = {m1/m_sun},{m2/m_sun} [m_sun] ; e0 = {e0}')
    vprint(f'f_orb = {f00} [Hz]; f_angular = {f0} [Hz]; f_GR = {deltaf} [Hz]')



    # [CHECK 1] 初始近星点保护
    rp_check = (1.0 - e0) / x0
    if rp_check < 6.0:
        print(f"ERROR: Initial Condition Unstable! rp = {rp_check:.2f} M (< 6M). Returning zeros.")
        return [np.array([0.0]), np.array([0.0]), np.array([0.0]), 0, [], np.array([0.0])]

    t_accuracy = int(timescale * f0 * max(np.power(1 - e0, -3 / 2) / 100, 1)) + 2000
    rtol0 = 2 / 3 * np.power(1 - e0, 3 / 2) / (timescale * f00) / 10
    if rtol0 < 1e-13: rtol0 = 1e-13

    # =================================================================
    # 3. 演化求解 (关键：必须注意依赖顺序)
    # =================================================================

    # (A) Map x(e)
    logonee = np.linspace(np.log10(1 - e0), np.log10(1 - emin), num=t_accuracy)
    e_temp = 1 - np.power(10., logonee)

    # ------------------ WARNING SUPPRESSION BLOCK START ------------------
    # 使用 warnings.catch_warnings 屏蔽 dx_de 积分时的 ODEintWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=scipy.integrate.ODEintWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # 积分得到 x(e)
        xe = sci_integrate.odeint(dx_de, x0, e_temp, rtol=rtol0, atol=0).T[0]

        # [Merger Check 1]: 在 x(e) 映射阶段检查 ISCO
        x_isco = 1.0 / 6.0
        # 找到所有符合物理条件的索引 (x < x_isco)
        valid_mask = xe < x_isco
        # 如果有任何点超出了 ISCO，我们需要截断
        if np.sum(~valid_mask) > 0:
            # 找到第一个非法的索引
            idx_merger_xe = np.argmax(~valid_mask)
            if idx_merger_xe > 1:
                #print(
                #    f"Warning: Reached ISCO (x >= 1/6) during x(e) mapping at e={e_temp[idx_merger_xe]}. Truncating map.")
                xe = xe[:idx_merger_xe]
                e_temp = e_temp[:idx_merger_xe]
            else:
                print("Error: Initial conditions are already past ISCO.")
                return [np.array([]), np.array([]), np.array([]), 0, [], np.array([])]

        # 建立插值函数
        xe1 = sci_interpolate.interp1d(e_temp, xe, fill_value="extrapolate")

        # 2. Slow Evolution: e(t)
        # 这里我们还是让它跑到 timescale，但会在后面做截断
        t_temp = np.linspace(0, timescale, num=t_accuracy)
        eresult = sci_integrate.odeint(de_dt, e0, t_temp, rtol=rtol0, atol=0).T[0]

        # 建立 e(t) 插值以便后续计算 x(t) 和 l(t)
        e_t = sci_interpolate.interp1d(t_temp, eresult, fill_value="extrapolate")

        # 3. Slow Evolution: x(t)
        xresult = sci_integrate.odeint(dx_dt, x0, t_temp, rtol=rtol0, atol=0).T[0]

        # [Merger Check 2]: 在时间演化结果中检查截断
        # 查找 x >= x_isco 或者 e 停止演化（导数为0）的地方
        # 由于 de_dt 在 merger 时返回 0，eresult 会在某处变成常数，但 xresult 可能会有些微漂移
        # 最稳健的方法是直接检查 xresult 是否越界
        merger_indices = np.where(xresult >= x_isco)[0]

        cutoff_index = len(t_temp)  # 默认不截断

        if len(merger_indices) > 0:
            cutoff_index = merger_indices[0]
            print(
                f"!!! MERGER REACHED !!! Truncating waveform at ISCO: t = {t_temp[cutoff_index]} s (x={xresult[cutoff_index]})")

        # 执行截断
        if cutoff_index < len(t_temp):
            t_temp = t_temp[:cutoff_index]
            eresult = eresult[:cutoff_index]
            xresult = xresult[:cutoff_index]
            # 更新 timescale，防止后面生成多余的 0 数据
            timescale = t_temp[-1]

        # 重新建立插值，因为 xresult 被截断了
        x_t = sci_interpolate.interp1d(t_temp, xresult, fill_value="extrapolate")
        e_t = sci_interpolate.interp1d(t_temp, eresult, fill_value="extrapolate")  # 更新 e_t 保证一致

        # 4. Slow Evolution: l(t)
        # 使用截断后的 t_temp 进行积分，节省时间
        lresult = sci_integrate.odeint(dl_dt, l0, t_temp, rtol=min(rtol0, 1e-8)).T[0]

    # ------------------ WARNING SUPPRESSION BLOCK END ------------------

    if len(t_temp) == 0:
        print("Waveform too short or invalid params.")
        return [np.array([]), np.array([]), np.array([]), 0, [], np.array([])]

    xlast = xresult[-1]
    flast = 1 / 2 / pi / M * np.power(xlast, 3 / 2)
    vprint(f'Evolution: {timescale / years} yr. f_initial: {f0} [Hz], f_final: {flast} [Hz]')
    # =================================================================
    # 4. 波形生成与采样
    # =================================================================
    if ts is None:
        controlnum2 = int(N * timescale * flast * np.power(1 - eresult[-1], -3 / 2)) + 1000
    else:
        controlnum2 = int(timescale / ts)
    if controlnum2 < 100: controlnum2 = 100

    # Memory Check
    est_memory_bytes = controlnum2 * 12 * 8
    max_bytes = max_memory_GB * (1024 ** 3)
    if est_memory_bytes > max_bytes:
        safe_controlnum2 = int(max_bytes / (12 * 8))
        print(f"[WARNING] Memory limit exceeded. Downgrading points: {controlnum2} -> {safe_controlnum2}")
        controlnum2 = safe_controlnum2

    t3_temp = np.linspace(0, timescale, num=controlnum2)
    dt = t3_temp[1] - t3_temp[0]

    # [CHECK 3] dt check
    if dt <= 0:
        print("Warning: dt=0, forced to 1e-16 to avoid ZeroDivisionError.")
        dt = 1e-16

    vprint(f'Waveform sampling: {1 / dt} Hz, Total points: {len(t3_temp)}')

    evec = np.interp(t3_temp, t_temp, eresult)
    xvec = np.interp(t3_temp, t_temp, xresult)
    lvec = np.interp(t3_temp, t_temp, lresult)
    ephivec = e_phi_func(evec, xvec)

    start_u = time.time()
    uvec = solve_u_series_robust(
        lvec.astype(np.float64), evec.astype(np.float64), xvec.astype(np.float64),
        ephivec.astype(np.float64), float(smr), int(PN_orbit)
    )

    start_h = time.time()
    rvec, dpsi_dt_vec, psiresult, hplusv, hcrossv = compute_h_arrays_full(
        evec.astype(np.float64), uvec.astype(np.float64), xvec.astype(np.float64),
        float(phi), float(M), float(smr), float(R), float(theta), float(dt), int(PN_orbit)
    )

    return [t3_temp, hplusv, hcrossv]
@njit(fastmath=True, cache=True)
def core_lisa_response_loop(t_arr, hplus, hcross, theta, phi, psi,
                                  fm, kappa, lamb, a, t0, ndot0):
    """
    修正后的核心循环，旨在严格匹配 'LISA_detectorresponse' (MED) 的逻辑。

    MED 逻辑:
      timelistnew = t + ndotx1(t)
      timelistnew -= t0
      timelistnew -= ndotx1(0)
      result = np.interp(timeline, timelistnew, signal)

    Numba 实现:
      time_new[i] = t + ndot - t0 - ndot0
    """
    n = len(t_arr)
    signal_static = np.empty(n, dtype=np.float64)
    time_new = np.empty(n, dtype=np.float64)

    # 预计算三角函数
    sin_theta = np.sin(theta);
    cos_theta = np.cos(theta)
    sin_2theta = np.sin(2 * theta);
    cos_2theta = np.cos(2 * theta)
    cos_2phi = np.cos(2 * phi);
    sin_2phi = np.sin(2 * phi)
    cos_2psi = np.cos(2 * psi);
    sin_2psi = np.sin(2 * psi)
    sqrt3 = np.sqrt(3.0);
    pi_val = np.pi

    for i in range(n):
        t = t_arr[i]
        alp = 2 * pi_val * fm * t + kappa

        # 1. 计算 ndotx1 (Roemer delay)
        # 对应 MED: ndotx1(t)
        ndot_t = a * sin_theta * np.cos(alp - phi)

        # 2. 构建插值用的 X 轴 (严格匹配 MED)
        # 对应 MED: t + ndotx1(t) - t0 - ndotx1(0)
        time_new[i] = t + ndot_t - t0 - ndot0

        # 3. 计算 Dplus (保持不变)
        arg_2lamb = 2 * lamb
        arg_2alp_2lamb = 2 * alp - 2 * lamb
        arg_4alp_2lamb = 4 * alp - 2 * lamb

        term1 = -36 * (sin_theta * sin_theta) * np.sin(arg_2alp_2lamb)
        sub_term2 = (cos_2phi * (9 * np.sin(arg_2lamb) - np.sin(arg_4alp_2lamb)) +
                     sin_2phi * (np.cos(arg_4alp_2lamb) - 9 * np.cos(arg_2lamb)))
        term2 = (3 + cos_2theta) * sub_term2

        arg_3a = 3 * alp - 2 * lamb - phi
        arg_1a = alp - 2 * lamb + phi
        sub_term3 = np.sin(arg_3a) - 3 * np.sin(arg_1a)
        term3 = -4 * sqrt3 * sin_2theta * sub_term3

        Dp = (sqrt3 / 64.0) * (term1 + term2 + term3)

        # 4. 计算 Dcross (保持不变)
        arg_2l_2p = 2 * lamb - 2 * phi
        arg_4a_2l_2p = 4 * alp - 2 * lamb - 2 * phi
        sub_term1_c = 9 * np.cos(arg_2l_2p) - np.cos(arg_4a_2l_2p)
        term1_c = sqrt3 * cos_theta * sub_term1_c

        sub_term2_c = np.cos(arg_3a) + 3 * np.cos(arg_1a)
        term2_c = -6 * sin_theta * sub_term2_c

        Dc = (1.0 / 16.0) * (term1_c + term2_c)

        # 5. 合成 Fplus, Fcross (保持不变)
        Fp = 0.5 * (cos_2psi * Dp - sin_2psi * Dc)
        Fc = 0.5 * (sin_2psi * Dp + cos_2psi * Dc)

        signal_static[i] = hplus[i] * Fp + hcross[i] * Fc

    return time_new, signal_static
def compute_LISA_response(t_src, hplus, hcross,
                          theta, phi, psi, t0=0,
                          kappa=0.0, lamb=0.0,
                          mode='interp'):
    """
    计算外部波形的 LISA 响应，支持两种输出模式。

    Args:
        t_src (array): 波源输入时间轴 (通常是均匀的)。
        hplus, hcross (array): 波源处的极化分量。
        theta, phi, psi (float): 几何角度 (rad)。
        t0 (float): 到达时间偏移 (s)。
        kappa, lamb (float): 初始轨道/相位参数。
        mode (str): 输出模式开关
            - 'interp' (默认): 输出时间轴与输入 t_src 完全一致。
                               会自动进行重采样，并在信号未覆盖区域补 0。
            - 'raw': 输出真实的探测器到达时间轴 (非均匀，受多普勒影响)。
                     不进行插值，保留最原始物理信息。

    Returns:
        t_out (array): 输出时间轴 (取决于 mode)。
        h_out (array): 输出 Strain。
    """
    # 1. 定义常量
    years = 365 * 24 * 3600.0
    ecc_lisa = 0.00965
    L = 5e9 / sciconsts.c
    a = L / ecc_lisa / (2 * np.sqrt(3))
    fm = 1 / years

    # 2. 计算初始对齐延迟 (t=0时刻的几何延迟)
    # 这一步确保了当 t_src=0 时，扣除 t0 后，物理上的相对延迟归零
    ndot0 = a * np.sin(theta) * np.cos(kappa - phi)

    # 3. 调用内核计算物理到达时间
    # t_physical 是这一串信号真实到达探测器的时间点 (非均匀)
    t_physical, h_modulated = core_lisa_response_loop(
        np.ascontiguousarray(t_src, dtype=np.float64),
        np.ascontiguousarray(hplus, dtype=np.float64),
        np.ascontiguousarray(hcross, dtype=np.float64),
        float(theta), float(phi), float(psi),
        float(fm), float(kappa), float(lamb), float(a), float(t0), float(ndot0)
    )

    # 4. 根据模式处理输出
    if mode == 'raw':
        # --- 模式二：输出非均匀物理时间 ---
        # 这里的 t_physical 包含了多普勒效应导致的网格变形
        return [t_physical, h_modulated]

    elif mode == 'interp':
        # --- 模式一：重采样回原始网格 ---
        # 我们希望知道：在 t_src 定义的那些时刻，探测器读数是多少？
        # 这是一个插值问题：已知 (t_physical, h_modulated)，求 f(t_src)

        # left=0.0, right=0.0 实现了“缺数据补0”的要求
        h_resampled = np.interp(t_src, t_physical, h_modulated, left=0.0, right=0.0)

        return [t_src, h_resampled]

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'interp' or 'raw'.")

# -----------------------------------------------------------------------------
# Numba 加速核心计算部分 (只负责算数，不改流程)
# -----------------------------------------------------------------------------

@njit(cache=True)
def get_lisa_noise_value(f):
    """
    计算单点 LISA 噪声值 (内部默认噪声)
    """
    if f <= 1.0e-9:
        return 1.0e100

    # --- S_gal_N2A50 ---
    s_gal = 0.0
    if f >= 1.0e-5 and f < 1.0e-3:
        s_gal = np.power(f, -2.3) * np.power(10, -44.62) * 20.0 / 3.0
    elif f >= 1.0e-3 and f < np.power(10, -2.7):
        s_gal = np.power(f, -4.4) * np.power(10, -50.92) * 20.0 / 3.0
    elif f >= np.power(10, -2.7) and f < np.power(10, -2.4):
        s_gal = np.power(f, -8.8) * np.power(10, -62.8) * 20.0 / 3.0
    elif f >= np.power(10, -2.4) and f <= 0.01:
        s_gal = np.power(f, -20.0) * np.power(10, -89.68) * 20.0 / 3.0

    # --- S_n_lisa ---
    m1 = 5.0e9
    c = 299792458.0
    m2 = c * 0.41 / m1 / 2.0
    pi = 3.141592653589793

    term1 = 20.0 / 3.0 * (1 + np.power(f / m2, 2.0))
    term2 = (4.0 * (9.0e-30 / np.power(2 * pi * f, 4.0) * (1 + 1.0e-4 / f)) + 2.96e-23 + 2.65e-23)
    s_instr = term1 * term2 / np.power(m1, 2.0)

    return s_instr + s_gal


@njit(cache=True)
def compute_sn_vector(xs):
    """内部生成噪声数组"""
    n = len(xs)
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        res[i] = get_lisa_noise_value(xs[i])
    return res


@njit(cache=True)
def compute_integral_sum(h1left, h1right, h2left, h2right,
                         h1_angle_left, h1_angle_right,
                         h2_angle_left, h2_angle_right,
                         snfleft, snfright,
                         xsl, xsr, phic, tobs):
    """
    计算积分求和 (加速原代码中的 ABval2 计算)
    """
    n = len(h1left)
    total_sum = 0.0

    for i in range(n):
        # 对应原代码公式
        term_left = 4.0 * h1left[i] * h2left[i] / snfleft[i] * np.cos(h1_angle_left[i] - (h2_angle_left[i] + phic))
        term_right = 4.0 * h1right[i] * h2right[i] / snfright[i] * np.cos(
            h1_angle_right[i] - (h2_angle_right[i] + phic))
        dx = xsr[i] - xsl[i]

        total_sum += (term_left + term_right) * dx / 2.0

    return total_sum * tobs * tobs


# -----------------------------------------------------------------------------
# 主接口
# -----------------------------------------------------------------------------

def inner_product(fs, waveform1, waveform2, phic, snf=None):
    num1 = len(waveform1)
    dt = 1.0 / fs
    tobs = num1 * dt

    # 频率轴
    xs = np.linspace(0, fs / 2.0, num=num1 // 2)

    # 1. FFT 处理
    hf_1 = scipy.fftpack.fft(waveform1)
    hf_1_abs = np.abs(hf_1)
    hf_1_angle = np.angle(hf_1)[0:num1 // 2]
    hf_1_norm = 2.0 / num1 * hf_1_abs[0:num1 // 2]

    hf_2 = scipy.fftpack.fft(waveform2)
    hf_2_abs = np.abs(hf_2)
    hf_2_angle = np.angle(hf_2)[0:num1 // 2]
    hf_2_norm = 2.0 / num1 * hf_2_abs[0:num1 // 2]

    # 2. 准备噪声 Snfvec (兼容性处理)
    if snf is None:
        Snfvec = compute_sn_vector(xs)
    else:
        try:
            # 尝试直接传数组 (假设用户函数支持向量化，这样快)
            Snfvec = snf(xs)
        except Exception:
            # 如果报错，说明不支持向量化，改为逐点生成
            Snfvec = np.array([snf(f) for f in xs])

    # 强制确保是 float64 数组，防止外部函数返回 list 或其它类型导致 numba 报错
    Snfvec = np.asarray(Snfvec, dtype=np.float64)

    # 3. 切片准备
    h1left = hf_1_norm[1:-1]
    h1right = hf_1_norm[2:]
    h1_angle_left = hf_1_angle[1:-1]
    h1_angle_right = hf_1_angle[2:]

    h2left = hf_2_norm[1:-1]
    h2right = hf_2_norm[2:]
    h2_angle_left = hf_2_angle[1:-1]
    h2_angle_right = hf_2_angle[2:]

    snfleft = Snfvec[1:-1]
    snfright = Snfvec[2:]

    xsl = xs[1:-1]
    xsr = xs[2:]

    # 4. 计算内积 (Numba 加速)
    ABval_raw = compute_integral_sum(
        h1left, h1right, h2left, h2right,
        h1_angle_left, h1_angle_right,
        h2_angle_left, h2_angle_right,
        snfleft, snfright,
        xsl, xsr, phic, tobs
    )

    ABval = abs(ABval_raw)
    return ABval

from scipy.optimize import brentq

def GWtime(m1, m2, a1, e1):
    # #print(m1/m_sun,m2/m_sun,a1/AU,e1)
    #if e1 >= 1.0 or a1 <= 0: return 0.0
    #factor = 1.6e13
    #return factor * (2 / m1 / m2 / (m1 + m2)) * np.power(a1 / 0.1, 4.0) * np.power(1 - e1 * e1, 7 / 2)
    return tmerger_integral(m1, m2, a1, e1)


def peters_factor_func(e):
    if e < 1e-10: return 0.0
    term1 = np.power(e, 12.0 / 19.0)
    term2 = 1.0 - e * e
    term3 = np.power(1.0 + (121.0 / 304.0) * e * e, 870.0 / 2299.0)
    return (term1 / term2) * term3


def solve_ae_after_time(m1, m2, a0, e0, dt):
    current_life = GWtime(m1, m2, a0, e0)
    #print(current_life,dt)
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



if __name__ == '__main__':
    Tobs = 1 * years
    fmax = 0.1
    fmin = 0  # 1e-5
    N = int(fmax * Tobs)
    f0 = fmax / N
    print('tobs [d]', Tobs / days, 'N', N, 'f0', f0)
    timelist = np.linspace(0, Tobs, 2 * N - 1)

    a0 = 0.01506869 * AU  # 0.1*AU#float(aeinfo[0])
    e0 = 0.9060675000000001  # 0.96679136  # 0.999#float(aeinfo[1])
    Dl = 8  # kpc
    theta = 2.6590048427983772
    phi = 5.4323352960796045
    psi = 1.8857738911294881
    theta2 = pi / 2  # 2.799259746842798#GWsource frame Line of sight direction
    phi2 = pi / 4  # 1.7382600210381554
    m1 = 8.255 * m_sun
    m2 = 21.29 * m_sun
    M = m1 + m2  # 29.532425000000003 * m_sun  #
    q = min(m1, m2) / max(m1, m2)  # 0.8463054136069103  #
    f00 = np.sqrt(M / (4 * pi * pi * np.power(a0, 3.0)))  #
    # print(f00, '!')
    t0 = 100  # 1 / f00 * np.power(1 - e0, 3 / 2)
    parameternames = ['theta', 'phi', 'psi', 'forb', '1-e', 'M', 'q', 'thetatwo', 'phitwo', 't0', 'Dl']  # fakeDl

    print('SNR', SNR(m1 / m_sun, m2 / m_sun, a0 / AU, e0, Dl, Tobs / years),
          SNR_approx(m1 / m_sun, m2 / m_sun, a0 / AU, e0, Dl * 1e3 * pc, Tobs))

    print('tmerger (yr)', tmerger_integral(m1 / m_sun, m2 / m_sun, a0 / AU, e0), tmerger_lower(m1, m2, a0, e0) / years)

    Dim = len(parameternames)

    tlist = timelist
    parametervalue = [theta, phi, psi, f00, 1 - e0, M, q, theta2, phi2, t0, Dl]  # last item fakeDl, in the unit of kpc

    a = time.time()
    # eccGW_waveform 内部逻辑正确，但注意它是否期待无量纲 mass？
    # 看代码开头: m1=m1*m_sun
    # 所以必须传入无量纲 Mass (m1/m_sun)
    hn1 = eccGW_waveform(f00, e0, Tobs / years, M / (1 + q) / m_sun, M * q / (1 + q) / m_sun, theta2, phi2, Dl)
    # hn1 = compute_LISA_response(hn1[0],hn1[1],hn1[2],pi/4,pi/4,pi/4)
    b = time.time()

    # hn3 = eccGW_waveform(f00,e0,Tobs,M / (1 + q), M * q / (1 + q),theta2,phi2,Dl*1e3*pc)
    c = time.time()

    # [Fix] 移除 .run()，直接调用
    if len(hn1[0]) > 1:
        SNR2 = np.sqrt(inner_product(1 / (hn1[0][1] - hn1[0][0]), hn1[1], hn1[1], 0))
    else:
        SNR2 = 0
        print("Waveform failed generation or too short.")

    # hn3 = detectorresponse(parametervalue,tlist)
    d = time.time()

    ee = time.time()
    print(b - a, c - b, d - c, 's')
    if len(hn1[0]) > 0:
        plt.plot(hn1[0], hn1[1], color='BLUE', label='NEW')
        # plt.plot(hn4[0], hn4[1], color='RED', linestyle='--', label='ORIGINAL')
        # plt.plot(hn3[0], hn3[1], color='ORANGE', linestyle=':', label='OLD')
        # plt.plot(hn4[0], hn4[1], color='BLACK', linestyle='-.', label='OLD')
        plt.legend()
        plt.xlabel("t [s]", fontsize=14)
        plt.ylabel("h", fontsize=14)
        plt.show()

    print(SNR2)
