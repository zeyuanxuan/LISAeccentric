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
from scipy.integrate import quad
import os
import shutil
import sys
import warnings
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator  # [新增] 用于快速插值
import warnings
from scipy.optimize import fsolve

# -----------------------------------------------------------------------------
# [AUTO-FIX] SciPy Version Compatibility Patch
# -----------------------------------------------------------------------------
if not hasattr(scipy.integrate, 'cumtrapz'):
    if hasattr(scipy.integrate, 'cumulative_trapezoid'):
        scipy.integrate.cumtrapz = scipy.integrate.cumulative_trapezoid
    else:
        # For very old scipy versions (unlikely but safe)
        from scipy.integrate import cumtrapz as _cumtrapz
        scipy.integrate.cumtrapz = _cumtrapz
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


# -----------------------------------------------------------------------------
# Constants (Extracted for Numba compatibility)
# -----------------------------------------------------------------------------
# Numba works best with pure floats, not module attributes inside the JIT function
CONST_G = sciconsts.G
CONST_c = sciconsts.c
CONST_pi = sciconsts.pi
CONST_au = sciconsts.au
m_sun = 1.98840987e30 * CONST_G / np.power(CONST_c, 3.0)
years = 365 * 24 * 3600.0
days = 24 * 3600
pc = 3.261 * sciconsts.light_year / CONST_c
AU = CONST_au / CONST_c
gama = 0.577215664901532860606512090082402431042159335
pi=sciconsts.pi
C_val = sciconsts.c


def J(n, x):
    return scipy.special.jv(n, x)
def g(n, e):
    ne = n * e

    term1 = J(n - 2, ne) - 2 * e * J(n - 1, ne) + 2 / n * J(n, ne) + 2 * e * J(n + 1, ne) - J(n + 2, ne)
    term2 = J(n - 2, ne) - 2 * J(n, ne) + J(n + 2, ne)
    term3 = J(n, ne)

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
# --- Global Noise Data Storage ---
_LISA_NOISE_DATA = None


def _try_load_lisa_noise():
    global _LISA_NOISE_DATA
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'LISA_noise_ASD.csv')

        if os.path.exists(file_path):
            try:
                data = np.loadtxt(file_path, delimiter=',')
            except ValueError:
                data = np.loadtxt(file_path, delimiter=',', skiprows=1)

            sort_idx = np.argsort(data[:, 0])
            sorted_data = data[sort_idx]

            f_data = sorted_data[:, 0]
            asd_data = sorted_data[:, 1]

            log_f = np.log10(f_data + 1e-30)
            log_asd = np.log10(asd_data + 1e-30)

            # Power-law exptrapolate
            # y = log(ASD), x = log(f)
            low_f_slope = (log_asd[1] - log_asd[0]) / (log_f[1] - log_f[0])

            _LISA_NOISE_DATA = {
                'f_min': f_data[0],
                'f_max': f_data[-1],
                'log_f': log_f,
                'log_asd': log_asd,
                'low_f_slope': low_f_slope,
                'log_f_0': log_f[0],
                'log_asd_0': log_asd[0]
            }
            #print(f"[Info] Successfully loaded LISA noise file: {file_path}")
        else:
            print(f"[Info] LISA noise file not found at {file_path}. Using default analytical model.")
    except Exception as e:
        print(f"[Warning] Failed to load LISA noise file ({e}). Using default analytical model.")
        _LISA_NOISE_DATA = None



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
    """original N2A5 Snf fallback"""
    m1 = 5.0e9
    m2 = sciconsts.c * 0.41 / m1 / 2.0
    return 20.0 / 3.0 * (1 + np.power(f / m2, 2.0)) * (4.0 * (
            9.0e-30 / np.power(2 * sciconsts.pi * f, 4.0) * (1 + 1.0e-4 / f)) + 2.96e-23 + 2.65e-23) / np.power(m1,
                                                                                                                2.0) + S_gal_N2A5(
        f)


def S_n_lisa(f):
    """
    modified Snf calculation
    """
    if _LISA_NOISE_DATA is not None:
        f_arr = np.atleast_1d(f)
        log_f_in = np.log10(np.maximum(f_arr, 1e-30))

        # left=NaN:
        # right=0.0: ASD=1.0 (log10(1)=0)
        log_asd_out = np.interp(
            log_f_in,
            _LISA_NOISE_DATA['log_f'],
            _LISA_NOISE_DATA['log_asd'],
            left=np.nan,
            right=0.0
        )

        mask_low = np.isnan(log_asd_out)

        if np.any(mask_low):
            # y = y0 + slope * (x - x0)
            log_asd_out[mask_low] = _LISA_NOISE_DATA['log_asd_0'] + \
                                    _LISA_NOISE_DATA['low_f_slope'] * \
                                    (log_f_in[mask_low] - _LISA_NOISE_DATA['log_f_0'])

        # ASD = 10^(log_asd)
        asd_out = np.power(10.0, log_asd_out)

        # Sn(f) = ASD^2
        res = asd_out * asd_out

        if np.isscalar(f):
            return res[0]
        return res
    else:
        # Fallback
        return _S_n_lisa_original(f)

def peters_factor_func(e):
    if e < 1e-10: return 0.0
    term1 = np.power(e, 12.0 / 19.0)
    term2 = 1.0 - e * e
    term3 = np.power(1.0 + (121.0 / 304.0) * e * e, 870.0 / 2299.0)
    return (term1 / term2) * term3
def tmerger_integral_old(m1, m2, a0, e0):
    m1 = m1
    m2 = m2
    a0 = a0
    beta = 64.0 / 5.0 * m1 * m2 * (m1 + m2)
    if beta == 0: return 1e99

    if e0 < 1e-4:
        return a0 ** 4 / (4.0 * beta)

    term_c0 = np.power(e0, 12.0 / 19.0) * np.power(1 + 121.0 / 304.0 * e0 ** 2, 870.0 / 2299.0)
    c0 = a0 * (1 - e0 ** 2) / term_c0

    def integrand(e):
        numer = np.power(e, 29.0 / 19.0) * np.power(1 + 121.0 / 304.0 * e ** 2, 1181.0 / 2299.0)
        denom = np.power(1 - e ** 2, 1.5)
        return numer / denom

    integral_val, abserr = quad(integrand, 0, e0)

    t_merger = (12.0 / 19.0) * (np.power(c0, 4.0) / beta) * integral_val

    # def tmerger1(m1,m2,a, e):
    #     beta = 64 / 5 * m1 * m2 * (m1 + m2)
    #     tc = np.power(a, 4) / (4 * beta)
    #     t = 768 / 425 * tc * np.power(1 - e * e, 7 / 2)
    #     return tc
    # print(t_merger/years,tmerger1(m1,m2,a0,e0)/years)

    return t_merger
class MergerTimeAccelerator:
    """
    Peters (1964) constant pre-computation table
    save as .npz
    """

    def __init__(self, cache_file='merger_time_table.npz'):
        self.cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_file)

        if self._load_cache():
            pass
            #print(f"[System] Loaded merger time table from {self.cache_file}")
        else:
            print("[System] Computing merger time table (first time run)...")
            self._compute_and_save_table()
            print(f"[System] Table computed and saved to {self.cache_file}")

        self.interpolator = PchipInterpolator(self.e_grid, self.f_vals)

    def _compute_and_save_table(self):
        x_vals = np.linspace(0, 5, 2000)
        self.e_grid = 1.0 - np.power(10, -x_vals)
        self.e_grid[0] = 0.0

        # F(e)
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

        # Factor = (48/19) * (c0/a)^4 * I
        return (48.0 / 19.0) * np.power(c0_norm, 4.0) * integral_val

    def get_factor(self, e):
        if e < 1e-4: return 1.0
        if e > 0.99999:
            return (768.0 / 425.0) * np.power(1 - e ** 2, 3.5)
        return self.interpolator(e)

_ACCELERATOR = MergerTimeAccelerator()

# use _ACCELERATOR.get_factor
def tmerger_integral(m1, m2, a0, e0):
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
def solve_ae_after_time0(m1, m2, a0, e0, dt):
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


def solve_ae_after_time01(m1, m2, a0, e0, dt):
    """
    Evolve system by dt seconds (Supports both forward and backward time evolution).
    Inputs: m1, m2 (seconds), a0 (seconds), e0 (dimensionless), dt (seconds).
    Returns: a_curr, e_curr
    """
    current_life = GWtime(m1, m2, a0, e0)

    if dt > 0 and dt >= current_life:
        return 0.0, 0.0

    t_rem_target = current_life - dt

    # c0
    fact_e0 = peters_factor_func(e0)
    if fact_e0 == 0: return 0.0, 0.0  # Should not happen if e0 > 1e-10

    c0 = a0 / fact_e0

    if dt > 0:
        e_lower, e_upper = 1e-6, e0
    elif dt < 0:
        e_lower, e_upper = e0, 1-1e-7
    else:
        # dt == 0
        return a0, e0

    try:
        e_curr = brentq(lambda e: GWtime(m1, m2, c0 * peters_factor_func(e), e) - t_rem_target,
                        e_lower, e_upper, xtol=1e-12, maxiter=50)
    except Exception as e:
        print(f"[Warning] solve_ae failed for dt={dt}: {e}. Returning initial values.")
        e_curr = e0

    a_curr = c0 * peters_factor_func(e_curr)
    return a_curr, e_curr

def solve_ae_after_time(m1, m2, a0, e0, dt):
    """
    Evolve system by dt seconds (supports forward and backward).
    Inputs: m1, m2 (seconds), a0 (seconds), e0 (dimensionless), dt (seconds).
    Returns: a_curr, e_curr
    """
    # ---- Special case: circular / near-circular orbit ----
    # Peters' eccentric inversion (c0 = a/peters_factor(e)) diverges as e->0,
    # because e is a fixed point of de/dt. Use the closed-form circular law
    # instead:  a(t)^4 = a0^4 - 4*beta*t,   e(t) = 0.
    CIRC_THRESHOLD = 1e-10
    if e0 < CIRC_THRESHOLD:
        beta = (64.0 / 5.0) * m1 * m2 * (m1 + m2)  # geometric units
        a4_new = a0**4 - 4.0 * beta * dt
        if a4_new <= 0.0:
            # Forward: merged. Backward: would imply a0 < 0 in the past.
            return 0.0, 0.0
        a_curr = a4_new ** 0.25
        return a_curr, 0.0

    # ---- Generic eccentric case (original code) ----
    current_life = GWtime(m1, m2, a0, e0)

    if dt > 0 and dt >= current_life:
        return 0.0, 0.0

    t_rem_target = current_life - dt

    fact_e0 = peters_factor_func(e0)
    if fact_e0 == 0:
        return 0.0, 0.0  # Should not happen for e0 above threshold

    c0 = a0 / fact_e0

    if dt > 0:
        e_lower, e_upper = 1e-6, e0
    elif dt < 0:
        e_lower, e_upper = e0, 1 - 1e-7
    else:
        return a0, e0

    try:
        e_curr = brentq(
            lambda e: GWtime(m1, m2, c0 * peters_factor_func(e), e) - t_rem_target,
            e_lower, e_upper, xtol=1e-12, maxiter=50
        )
    except Exception as exc:
        print(f"[Warning] solve_ae failed for dt={dt}: {exc}. Returning initial values.")
        e_curr = e0

    a_curr = c0 * peters_factor_func(e_curr)
    return a_curr, e_curr

def dSNR2dt_numpy_old(m1, m2, a, e, Dl):
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

        f_arr = np.linspace(fmin, fmax, fnumber)

        rate = f_arr / f0
        n_avg = rate.astype(int)

        mask = n_avg > 0
        f_valid = f_arr[mask]
        n_valid = n_avg[mask]

        g_vals = g(n_valid, e)
        Sn_vals = S_n_lisa(f_valid)

        valid_sn_mask = Sn_vals > 0

        term = g_vals[valid_sn_mask] / np.square(f_valid[valid_sn_mask]) / Sn_vals[valid_sn_mask]
        result = np.sum(term) * deltaf

        return result * 8 * h * h * f0

    # --- Low Eccentricity Mode ---
    else:
        # print("Running Numpy Low-E Mode")
        nmin = int(fmin / f0) + 1
        nmax = int(fmax / f0) + 2

        n_arr = np.arange(nmin, nmax)
        if len(n_arr) == 0:
            return 0.0

        fn_arr = n_arr * forb(m1 + m2, a)

        val_h0 = h0(a, m1, m2, Dl)
        g_vals = g(n_arr, e)

        hn_arr = 2.0 / n_arr * np.sqrt(g_vals) * val_h0
        hnc2_arr = 4.0 * np.square(fn_arr) * np.square(hn_arr) / (2 * n_arr)

        Sn_vals = S_n_lisa(fn_arr)

        mask_sn = Sn_vals > 0
        # result = sum( hnc2 / fn^2 / Sn * n )
        terms = hnc2_arr[mask_sn] / np.square(fn_arr[mask_sn]) / Sn_vals[mask_sn] * n_arr[mask_sn]

        return np.sum(terms)

@njit(fastmath=True, cache=True)
def _get_sn_val_jit(f, use_file, log_f_grid, log_asd_grid, low_f_slope, log_f_0, log_asd_0):
    """
    JIT-friendly noise calculator (Scalar Version)
    Matches logic of S_n_lisa exactly but runs inside Numba.
    """
    if not use_file:
        # Fallback Analytical
        m1_a = 5.0e9
        m2_a = CONST_c * 0.41 / m1_a / 2.0

        # S_gal part
        gal_res = 0.0
        if f >= 1.0e-5 and f < 1.0e-3:
            gal_res = np.power(f, -2.3) * 10 ** (-44.62) * 20.0 / 3.0
        elif f >= 1.0e-3 and f < 10 ** (-2.7):
            gal_res = np.power(f, -4.4) * 10 ** (-50.92) * 20.0 / 3.0
        elif f >= 10 ** (-2.7) and f < 10 ** (-2.4):
            gal_res = np.power(f, -8.8) * 10 ** (-62.8) * 20.0 / 3.0
        elif f >= 10 ** (-2.4) and f <= 0.01:
            gal_res = np.power(f, -20.0) * 10 ** (-89.68) * 20.0 / 3.0

        term1 = 20.0 / 3.0 * (1.0 + np.power(f / m2_a, 2.0))
        term2_inner = 9.0e-30 / np.power(2 * CONST_pi * f, 4.0) * (1.0 + 1.0e-4 / f)
        term2 = 4.0 * term2_inner + 2.96e-23 + 2.65e-23

        return term1 * term2 / np.power(m1_a, 2.0) + gal_res

    else:
        # Interpolation Logic
        f_safe = max(f, 1e-30)
        log_f_in = np.log10(f_safe)

        if log_f_in < log_f_grid[0]:
            log_asd = log_asd_0 + low_f_slope * (log_f_in - log_f_0)
        elif log_f_in > log_f_grid[-1]:
            log_asd = 0.0  # ASD = 1.0 -> log = 0
        else:
            log_asd = np.interp(log_f_in, log_f_grid, log_asd_grid)

        asd = 10.0 ** log_asd
        return asd * asd
@njit(fastmath=True, cache=True)
def _dSNR_high_E_kernel(f_arr, n_vals, g_vals, delta_log_f, use_file, log_f, log_asd, slope, lf0, lasd0):
    total = 0.0
    n_len = len(f_arr)

    for i in range(n_len):
        n = n_vals[i]
        if n > 0:
            f = f_arr[i]
            sn = _get_sn_val_jit(f, use_file, log_f, log_asd, slope, lf0, lasd0)

            if sn > 0:
                g = g_vals[i]
                term = g / (f * sn)
                total += term

    return total * delta_log_f
@njit(fastmath=True, cache=True)
def _dSNR_low_E_kernel(n_arr, g_vals, forb_val, h0_val, use_file, log_f, log_asd, slope, lf0, lasd0):
    total = 0.0
    n_len = len(n_arr)

    # Precompute constant: 8 * h0^2
    const_num = 8.0 * h0_val * h0_val

    for i in range(n_len):
        n = n_arr[i]
        fn = n * forb_val
        sn = _get_sn_val_jit(fn, use_file, log_f, log_asd, slope, lf0, lasd0)

        if sn > 0:
            g = g_vals[i]
            # Optimization: 8 * g * h0^2 / (n^2 * Sn)
            term = (const_num * g) / (n * n * sn)
            total += term

    return total
def dSNR2dt_numpy_fixedLISA(m1, m2, a, e, Dl):
    """
    Optimized dSNR2dt using Numba Kernels.
    Modifications: High E mode now uses log-uniform frequency grid.
    """
    fmin = 1e-8
    fmax = 0.1
    fnumber = 200

    f0 = 1. / 2. / pi * np.sqrt(m1 + m2) * np.power(a, -1.5)

    threshold = (fmax - fmin) / fnumber

    # Prepare Noise Data for JIT
    use_file = False
    log_f_g = np.array([0.0])
    log_asd_g = np.array([0.0])
    slope = 0.0
    lf0 = 0.0
    lasd0 = 0.0

    if _LISA_NOISE_DATA is not None and _LISA_NOISE_DATA.get('use_file', False):
        use_file = True
        log_f_g = _LISA_NOISE_DATA['log_f']
        log_asd_g = _LISA_NOISE_DATA['log_asd']
        slope = _LISA_NOISE_DATA['low_f_slope']
        lf0 = _LISA_NOISE_DATA['log_f_0']
        lasd0 = _LISA_NOISE_DATA['log_asd_0']

    # --- High Eccentricity Mode (修改部分) ---
    if f0 * 10 < threshold:
        h = np.sqrt(32. / 5.) * m1 * m2 / Dl / a

        f_arr = np.geomspace(fmin, fmax, fnumber)

        # d(ln f) = ln(fmax / fmin) / (N - 1)
        delta_log_f = np.log(fmax / fmin) / (fnumber - 1)

        # 2. Vectorized n calculation
        n_vals = (f_arr / f0).astype(np.int64)

        # 3. Calculate g using SciPy
        g_vals = g(n_vals, e)

        # 4. Kernel Summation (JIT)
        raw_sum = _dSNR_high_E_kernel(
            f_arr, n_vals, g_vals, delta_log_f,
            use_file, log_f_g, log_asd_g, slope, lf0, lasd0
        )

        return raw_sum * 8 * h * h * f0

    # --- Low Eccentricity Mode ---
    else:
        nmin = int(fmin / f0) + 1
        nmax = int(fmax / f0) + 2

        n_arr = np.arange(nmin, nmax)
        if len(n_arr) == 0:
            return 0.0

        # 1. Calculate g (SciPy)
        g_vals = g(n_arr, e)

        val_h0 = h0(a, m1, m2, Dl)
        val_forb = forb(m1 + m2, a)

        # 2. Kernel Summation (JIT)
        total_sum = _dSNR_low_E_kernel(
            n_arr, g_vals, val_forb, val_h0,
            use_file, log_f_g, log_asd_g, slope, lf0, lasd0
        )

        return total_sum

# =============================================================================
# PATCH for PN_waveform.py  --  SNR harmonic-integration band auto-tracking
# =============================================================================
#
# WHY:
#   dSNR2dt_numpy() hardcoded the integration band fmin=1e-8, fmax=0.1, fnumber=200.
#   That band is the LISA mHz band. For a LIGO noise curve the source harmonics
#   live at ~10-2000 Hz, entirely above fmax=0.1 Hz, so the harmonic sum picks up
#   essentially nothing and SNR collapses to ~0 (the thin top stripe you saw).
#
# WHAT THIS PATCH DOES:
#   The band is now derived from the *currently active* noise curve:
#     - use_file == False (default / no curve injected):
#         returns EXACTLY the legacy band (1e-8, 0.1, 200)  -> bit-identical results.
#     - use_file == True  (a curve was injected via update_noise_curve, e.g. LIGO):
#         follows that curve's [f_min, f_max] and rescales fnumber so the
#         log-frequency sampling density stays the same (~28.6 points / decade).
#
#   Nothing else changes. SNR(), the two JIT kernels, _get_sn_val_jit(), and
#   core.py's _inject_noise_data() are untouched.
#
# HOW TO APPLY (two edits in PN_waveform.py):
#   1. Paste the block "[1] INSERT" somewhere above dSNR2dt_numpy (e.g. right
#      before its 'def dSNR2dt_numpy(...)', around line 607).
#   2. Replace the whole body of the existing dSNR2dt_numpy (lines ~607-680)
#      with the block "[2] REPLACE" below. Only the first three lines of the
#      body differ from the original (the fmin/fmax/fnumber assignment).
#
#   dSNR2dt_numpy_old() (line ~469) is dead code (SNR uses dSNR2dt_numpy); leave
#   it as-is, or apply the same one-line change if you want it consistent.
# =============================================================================


# ----------------------------------------------------------------------------
# [1] INSERT  (module-level, above dSNR2dt_numpy)
# ----------------------------------------------------------------------------

# Legacy hardcoded integration band. Reproduces the original results exactly
# whenever no external noise curve is injected (use_file is False).
_DEFAULT_FMIN = 1e-8
_DEFAULT_FMAX = 0.1
_DEFAULT_FNUMBER = 200
_DEFAULT_DECADES = np.log10(_DEFAULT_FMAX / _DEFAULT_FMIN)  # = 7.0


def _get_integration_band():
    """
    Frequency band (fmin, fmax, fnumber) for the SNR harmonic integral.

    Default / analytical noise (use_file False)  -> (1e-8, 0.1, 200), i.e. the
    original hardcoded LISA band, so legacy results are reproduced bit-for-bit.

    Injected file noise with its own frequency axis (use_file True) -> that
    curve's [f_min, f_max], with fnumber rescaled to preserve the per-decade
    log-frequency sampling density of the default band (~28.6 points/decade).
    """
    nd = _LISA_NOISE_DATA
    if nd is not None and nd.get('use_file', False):
        try:
            fmin = float(nd['f_min'])
            fmax = float(nd['f_max'])
        except (KeyError, TypeError, ValueError):
            return _DEFAULT_FMIN, _DEFAULT_FMAX, _DEFAULT_FNUMBER

        if (not np.isfinite(fmin)) or (not np.isfinite(fmax)) \
                or (fmin <= 0.0) or (fmax <= fmin):
            return _DEFAULT_FMIN, _DEFAULT_FMAX, _DEFAULT_FNUMBER

        decades = np.log10(fmax / fmin)
        fnumber = int(round(_DEFAULT_FNUMBER * decades / _DEFAULT_DECADES))
        fnumber = max(fnumber, 16)  # floor so coarse curves still resolve the peak
        return fmin, fmax, fnumber

    return _DEFAULT_FMIN, _DEFAULT_FMAX, _DEFAULT_FNUMBER


# ----------------------------------------------------------------------------
# [2] REPLACE  (the existing dSNR2dt_numpy)
# ----------------------------------------------------------------------------
def dSNR2dt_numpy(m1, m2, a, e, Dl):
    """
    Optimized dSNR2dt using Numba Kernels.
    Modifications: High E mode uses a log-uniform frequency grid; the
    integration band now auto-tracks the active noise curve via
    _get_integration_band() (legacy band reproduced when use_file is False).
    """
    # --- only change vs. original: band is no longer hardcoded ---
    fmin, fmax, fnumber = _get_integration_band()

    f0 = 1. / 2. / pi * np.sqrt(m1 + m2) * np.power(a, -1.5)

    threshold = (fmax - fmin) / fnumber

    # Prepare Noise Data for JIT
    use_file = False
    log_f_g = np.array([0.0])
    log_asd_g = np.array([0.0])
    slope = 0.0
    lf0 = 0.0
    lasd0 = 0.0

    if _LISA_NOISE_DATA is not None and _LISA_NOISE_DATA.get('use_file', False):
        use_file = True
        log_f_g = _LISA_NOISE_DATA['log_f']
        log_asd_g = _LISA_NOISE_DATA['log_asd']
        slope = _LISA_NOISE_DATA['low_f_slope']
        lf0 = _LISA_NOISE_DATA['log_f_0']
        lasd0 = _LISA_NOISE_DATA['log_asd_0']

    # --- High Eccentricity Mode ---
    if f0 * 10 < threshold:
        h = np.sqrt(32. / 5.) * m1 * m2 / Dl / a

        f_arr = np.geomspace(fmin, fmax, fnumber)

        # d(ln f) = ln(fmax / fmin) / (N - 1)
        delta_log_f = np.log(fmax / fmin) / (fnumber - 1)

        # 2. Vectorized n calculation
        n_vals = (f_arr / f0).astype(np.int64)

        # 3. Calculate g using SciPy
        g_vals = g(n_vals, e)

        # 4. Kernel Summation (JIT)
        raw_sum = _dSNR_high_E_kernel(
            f_arr, n_vals, g_vals, delta_log_f,
            use_file, log_f_g, log_asd_g, slope, lf0, lasd0
        )

        return raw_sum * 8 * h * h * f0

    # --- Low Eccentricity Mode ---
    else:
        nmin = int(fmin / f0) + 1
        nmax = int(fmax / f0) + 2

        n_arr = np.arange(nmin, nmax)
        if len(n_arr) == 0:
            return 0.0

        # 1. Calculate g (SciPy)
        g_vals = g(n_arr, e)

        val_h0 = h0(a, m1, m2, Dl)
        val_forb = forb(m1 + m2, a)

        # 2. Kernel Summation (JIT)
        total_sum = _dSNR_low_E_kernel(
            n_arr, g_vals, val_forb, val_h0,
            use_file, log_f_g, log_asd_g, slope, lf0, lasd0
        )

        return total_sum

def SNR(m1, m2, a, e, Dl, tobs):
    m1=m1
    m2=m2
    a=a
    Dl=Dl
    tobs=tobs
    # 1. Merger Time Check
    tmerger = tmerger_lower(m1, m2, a, e)
    used_tobs = tobs

    if tmerger <= tobs:
        tmerger = tmerger_integral(m1,m2,a,e)
        if tmerger <= tobs:
            print(f"[Warning] System evolves too fast! analytical lower bound of tmerger ({tmerger:.2e} s) < tobs ({tobs:.2e} s).")
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
    """ solve Kepler eqs """
    pi_val = np.pi
    sinu = np.sin(u)
    cosu = np.cos(u)

    # in case e*cosu --> 1
    one_minus_ecosu = 1.0 - e * cosu
    if one_minus_ecosu < 1e-12:
        one_minus_ecosu = 1e-12

    sqrt_1_e2 = np.sqrt(1 - e ** 2)

    val = -l + u - e * sinu
    deriv = one_minus_ecosu  # 0PN

    if PN_orbit >= 2:
        # in case ephi --> 0
        if np.abs(ephi) < 1e-10:
            betaphi = 0.5 * ephi
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
    n = len(l_vec)
    u_vec = np.empty(n, dtype=np.float64)
    u_vec[0] = l_vec[0]  # Fallback initial

    for i in range(n):
        l = l_vec[i]
        e = e_vec[i]
        x = x_vec[i]
        ephi = ephi_vec[i]

        # (Bracket Search)
        if i == 0:
            low = l - 1.5
            high = l + 1.5
        else:
            pred = u_vec[i - 1] + (l - l_vec[i - 1])
            low = pred - 1.0
            high = pred + 1.0

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

            if np.abs(df_du) < 1e-15:
                if df_du < 0:
                    df_du = -1e-15
                else:
                    df_du = 1e-15

            step = f_val / df_du

            # (Damped Newton)
            if step > 1.5: step = 1.5
            if step < -1.5: step = -1.5

            u_new = u_cur - step

            # Bisection Fallback
            if not (low < u_new < high):
                u_new = 0.5 * (low + high)
                step = u_cur - u_new

            u_cur = u_new


            f_new, _ = get_residual_and_deriv(u_cur, l, e, x, ephi, smr, PN_orbit)
            if f_new * fl > 0:
                low = u_cur
                fl = f_new
            else:
                high = u_cur

            if np.abs(high - low) < 1e-12: break

        u_vec[i] = u_cur
    return u_vec
@njit(cache=True)#fastmath=True,
def compute_h_arrays_full(evec, uvec, xvec, phi, M, smr, R, theta, dt, PN_orbit):
    """
    compute r, dpsi, psi, h+, hx
    Kahan Summation
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

    # 1. (r, dpsi/dt)
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

    # 2. Psi (Kahan Summation)
    current_psi = 0.0
    psi_compensation = 0.0
    psiresult[0] = 0.0

    for i in range(1, n):
        delta_psi = 0.5 * (dpsi_dt_vec[i - 1] + dpsi_dt_vec[i]) * dt

        # Kahan Summation Core Logic
        y = delta_psi - psi_compensation
        t = current_psi + y
        psi_compensation = (t - current_psi) - y
        current_psi = t

        psiresult[i] = current_psi

    # 3. Strain
    for i in range(n):
        if i == 0:
            rd = (rvec[1] - rvec[0]) / dt
        elif i == n - 1:
            rd = (rvec[n - 1] - rvec[n - 2]) / dt
        else:
            rd = (rvec[i + 1] - rvec[i - 1]) / (2 * dt)

        r = rvec[i]
        psid = dpsi_dt_vec[i]

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
def eccGW_waveform0(f00, e0, timescale, m1, m2, theta, phi, R, l0=0, ts=None, PN_orbit=3, PN_reaction=2,
                    N=50, max_memory_GB=16.0, verbose=True):

    vprint = print if verbose else lambda *args, **kwargs: None

    m1=m1
    m2=m2
    R=R
    timescale=timescale
    # === protect e=0 (code cannot deal with e=0) ===
    if e0 == 0:
        print("Warning: e0=0 is not supported by this eccentric template (singularities in PN terms).")
        print("         Automatically resetting e0 to 1e-5 to maintain stability.")
        e0 = 1e-5
    emin=0
    # ========================================
    vprint('================Ecc Inspiral GW waveform================')
    # =================================================================
    # 1.
    # =================================================================
    def J(n, x):
        return scipy.special.jv(n, x)
    def Jtilt(n, x):
        return 0.5 * (J(n - 1, x) - J(n + 1, x))
    #print(f"Loading 1.5PN Tail enhancement table (e0={e0:.6f})...")

    dist_to_1 = 1.0 - e0
    if dist_to_1 < 0: dist_to_1 = 0
    margin = min(0.0001, dist_to_1 * 0.05)
    required_safe_e0 = e0 + margin
    limit_e = 0.99999
    if required_safe_e0 > limit_e: required_safe_e0 = limit_e

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
        print(f"   Fast-computation table not found. Building new table up to e={target_max_e}...")

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
    # 2.
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




    rp_check = (1.0 - e0) / x0
    if rp_check < 6.0:
        raise ValueError(f"ERROR: Initial Condition Unstable! rp = {rp_check:.2f} M (< 6M, inside ISCO).")
        #return [np.array([0.0]), np.array([0.0]), np.array([0.0]), 0, [], np.array([0.0])]

    t_accuracy = int(timescale * f0 * max(np.power(1 - e0, -3 / 2) / 100, 1)) + 2000
    rtol0 = 2 / 3 * np.power(1 - e0, 3 / 2) / (timescale * f00) / 10
    if rtol0 < 1e-13: rtol0 = 1e-13

    # =================================================================
    # 3.
    # =================================================================

    # (A) Map x(e)
    logonee = np.linspace(np.log10(1 - e0), np.log10(1 - emin), num=t_accuracy)
    e_temp = 1 - np.power(10., logonee)

    # ------------------ WARNING SUPPRESSION BLOCK START ------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=scipy.integrate.ODEintWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        xe = sci_integrate.odeint(dx_de, x0, e_temp, rtol=rtol0, atol=0).T[0]

        # [Merger Check 1]
        x_isco = 1.0 / 6.0

        valid_mask = xe < x_isco

        if np.sum(~valid_mask) > 0:

            idx_merger_xe = np.argmax(~valid_mask)
            if idx_merger_xe > 1:
                #print(
                #    f"Warning: Reached ISCO (x >= 1/6) during x(e) mapping at e={e_temp[idx_merger_xe]}. Truncating map.")
                xe = xe[:idx_merger_xe]
                e_temp = e_temp[:idx_merger_xe]
            else:
                print("Error: Initial conditions are already past ISCO.")
                return [np.array([]), np.array([]), np.array([]), 0, [], np.array([])]


        xe1 = sci_interpolate.interp1d(e_temp, xe, fill_value="extrapolate")

        # 2. Slow Evolution: e(t)
        t_temp = np.linspace(0, timescale, num=t_accuracy)
        eresult = sci_integrate.odeint(de_dt, e0, t_temp, rtol=rtol0, atol=0).T[0]

        e_t = sci_interpolate.interp1d(t_temp, eresult, fill_value="extrapolate")

        # 3. Slow Evolution: x(t)
        xresult = sci_integrate.odeint(dx_dt, x0, t_temp, rtol=rtol0, atol=0).T[0]

        # [Merger Check 2]
        merger_indices = np.where(xresult >= x_isco)[0]

        cutoff_index = len(t_temp)

        if len(merger_indices) > 0:
            cutoff_index = merger_indices[0]
            print(
                f"!!! MERGER REACHED !!! Truncating waveform at ISCO: t = {t_temp[cutoff_index]} s (x={xresult[cutoff_index]})")

        if cutoff_index < len(t_temp):
            t_temp = t_temp[:cutoff_index]
            eresult = eresult[:cutoff_index]
            xresult = xresult[:cutoff_index]
            timescale = t_temp[-1]

        x_t = sci_interpolate.interp1d(t_temp, xresult, fill_value="extrapolate")
        e_t = sci_interpolate.interp1d(t_temp, eresult, fill_value="extrapolate")  # 更新 e_t 保证一致

        # 4. Slow Evolution: l(t)
        lresult = sci_integrate.odeint(dl_dt, l0, t_temp, rtol=min(rtol0, 1e-8)).T[0]

    # ------------------ WARNING SUPPRESSION BLOCK END ------------------

    if len(t_temp) == 0:
        print("Waveform too short or invalid params.")
        return [np.array([]), np.array([]), np.array([]), 0, [], np.array([])]

    xlast = xresult[-1]
    flast = 1 / 2 / pi / M * np.power(xlast, 3 / 2)
    vprint(f'Evolution: {timescale / years} yr. f_initial: {f0} [Hz], f_final: {flast} [Hz]')
    # =================================================================
    # 4.
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

def eccGW_waveform(f00, e0, timescale, m1, m2, theta, phi, R, l0=0, ts=None, PN_orbit=3, PN_reaction=2,
                    N=50, max_memory_GB=16.0, verbose=True):
    vprint = print if verbose else lambda *args, **kwargs: None
    #print('!!!!')
    m1=m1
    m2=m2
    R=R
    timescale=timescale
    if e0 == 0:
        print("Warning: e0=0 is not supported by this eccentric template (singularities in PN terms).")
        print("         Automatically resetting e0 to 1e-5 to maintain stability.")
        e0 = 1e-5
    emin=0
    # ========================================
    vprint('================Ecc Inspiral GW waveform================')
    # =================================================================
    # 1.
    # =================================================================
    def J(n, x):
        return scipy.special.jv(n, x)
    def Jtilt(n, x):
        return 0.5 * (J(n - 1, x) - J(n + 1, x))
    #print(f"Loading 1.5PN Tail enhancement table (e0={e0:.6f})...")

    dist_to_1 = 1.0 - e0
    if dist_to_1 < 0: dist_to_1 = 0
    margin = min(0.0001, dist_to_1 * 0.05)
    required_safe_e0 = e0 + margin
    limit_e = 0.99999
    if required_safe_e0 > limit_e: required_safe_e0 = limit_e

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
        print(f"   Fast-computation table not found. Building new table up to e={target_max_e}...")

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


    def coupled_derivs(y, t):
            # y[0] = x, y[1] = e
            x = y[0]
            e = y[1]

            # 1. protect
            if x < 0: x = 1e-8

            # prtect e
            if e < 0: e = 0.0
            if e >= 1.0: e = 0.99999

            one_minus_e2 = 1.0 - e * e
            if one_minus_e2 < 1e-10: one_minus_e2 = 1e-10

            # --- dx/dt Calculation ---
            xdot = np.power(x, 5.0) * 2 * (37 * np.power(e, 4.0) + 292 * e * e + 96) * smr / (
                        15 * np.power(one_minus_e2, 3.5))

            if PN_reaction >= 1:
                term_1pn = -(8288 * smr - 11717) * np.power(e, 6.0) \
                           - 14 * (10122 * smr - 12217) * np.power(e, 4.0) \
                           - 120 * (1330 * smr - 731) * e * e \
                           - 16 * (924 * smr + 743)
                xdot += np.power(x, 6.0) * smr / (420 * np.power(one_minus_e2, 4.5)) * term_1pn

            if PN_reaction >= 1.5:
                xdot += np.power(x, 6.5) * 256 / 5 * smr * pi * kappaE(e)

            if PN_reaction >= 2:
                term_2pn = (1964256 * smr * smr - 3259980 * smr + 3523113) * np.power(e, 8.0) + \
                           (64828848 * smr * smr - 123108426 * smr + 83424402) * np.power(e, 6.0) + \
                           (16650606060 * smr * smr - 207204264 * smr + 783768) * np.power(e, 4.0) + \
                           (61282032 * smr * smr + 15464736 * smr - 92846560) * e * e + 1903104 * smr * smr + \
                           np.sqrt(one_minus_e2) * ((2646000 - 1058400 * smr) * np.power(e, 6.0) + (
                            64532160 - 25812864 * smr) * e * e - 580608 * smr + 1451520) + \
                           4514976 * smr - 360224
                xdot += np.power(x, 7.0) * smr / (45360 * np.power(one_minus_e2, 5.5)) * term_2pn

            dx_dt_val = xdot / M

            # --- de/dt Calculation ---
            if e <= 1e-9:
                edot_val = 0.0
            else:
                edot = np.power(x, 4.0) * (-e * (121 * e * e + 304) * smr / (15 * np.power(one_minus_e2, 2.5)))

                if PN_reaction >= 1:
                    term_e_1pn = (93184 * smr - 125361) * np.power(e, 4.0) + \
                                 12 * (54271 * smr - 59834) * e * e + \
                                 8 * (28588 * smr + 8451)
                    edot += np.power(x, 5.0) * e * smr / (2520 * np.power(one_minus_e2, 3.5)) * term_e_1pn

                if PN_reaction >= 1.5:
                    edot += np.power(x, 5.5) * 128 * smr * pi / 5 / e * (
                                (e * e - 1) * kappaE(e) + np.sqrt(one_minus_e2) * kappaJ(e))

                if PN_reaction >= 2:
                    edot += np.power(x, 6.0) * E2PN(e)

                edot_val = edot / M

            return [dx_dt_val, edot_val]

    def dl_dt(l, t, x, e):
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
    # 2.
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



    # [CHECK 1]
    rp_check = (1.0 - e0) / x0
    if rp_check < 6.0:
        raise ValueError(f"ERROR: Initial Condition Unstable! rp = {rp_check:.2f} M (< 6M, inside ISCO).")
        #return [np.array([0.0]), np.array([0.0]), np.array([0.0]), 0, [], np.array([0.0])]
    if True:
        t_accuracy = int(timescale * f0 * 2) + 5000
        t_temp = np.linspace(0, timescale, num=t_accuracy)

        y0 = [x0, e0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sol = sci_integrate.odeint(coupled_derivs, y0, t_temp, rtol=1e-12, atol=1e-14)

            xresult = sol[:, 0]
            eresult = sol[:, 1]

            # === (Merger Handling) ===
            x_isco = 1.0 / 6.0

            # find merger point: x >= ISCO or x --> NaN
            bad_mask = (xresult >= x_isco) | np.isnan(xresult) | np.isinf(xresult)
            bad_indices = np.where(bad_mask)[0]

            if len(bad_indices) > 0:
                cutoff_index = bad_indices[0]
                vprint(
                    f"   Merger detected at t={t_temp[cutoff_index]}s (x={xresult[cutoff_index]}). Truncating.")
            else:
                cutoff_index = len(t_temp)
                # check timescale
                if xresult[-1] < 0.1:
                    vprint(
                        f"   Simulation finished without merger (final x={xresult[-1]}).")

            if cutoff_index < 2:
                print("Error: Waveform truncated immediately.")
                return [np.array([]), np.array([]), np.array([]), 0, [], np.array([])]

            t_temp = t_temp[:cutoff_index]
            xresult = xresult[:cutoff_index]
            eresult = eresult[:cutoff_index]

            # fix e < 0
            eresult[eresult < 0] = 0.0

            # === l(t) ===
            dl_vals = np.zeros_like(xresult)
            for i in range(len(xresult)):
                dl_vals[i] = dl_dt(0, 0, xresult[i], eresult[i])

            lresult = sci_integrate.cumtrapz(dl_vals, t_temp, initial=0) + l0

        if len(t_temp) == 0:
            return [np.array([]), np.array([]), np.array([]), 0, [], np.array([])]

        timescale = t_temp[-1]

        x_t = sci_interpolate.interp1d(t_temp, xresult, fill_value="extrapolate")
        e_t = sci_interpolate.interp1d(t_temp, eresult, fill_value="extrapolate")
        l_t = sci_interpolate.interp1d(t_temp, lresult, fill_value="extrapolate")

        flast = 1 / 2 / pi / M * np.power(xresult[-1], 3 / 2)
        vprint(f'Evolution: {timescale / (365 * 24 * 3600)} yr. f_initial: {f0}, f_final: {flast}')
    # =================================================================
    # 4.
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
    LISA RESPONSE FUNCTION
      timelistnew = t + ndotx1(t)
      timelistnew -= t0
      timelistnew -= ndotx1(0)
      result = np.interp(timeline, timelistnew, signal)

      (time_new[i] = t + ndot - t0 - ndot0)
    """
    n = len(t_arr)
    signal_static = np.empty(n, dtype=np.float64)
    time_new = np.empty(n, dtype=np.float64)

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

        # 1. ndotx1 (Roemer delay)
        ndot_t = a * sin_theta * np.cos(alp - phi)

        # 2. t + ndotx1(t) - t0 - ndotx1(0)
        time_new[i] = t + ndot_t - t0 - ndot0

        # 3. Dplus
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

        # 4. Dcross
        arg_2l_2p = 2 * lamb - 2 * phi
        arg_4a_2l_2p = 4 * alp - 2 * lamb - 2 * phi
        sub_term1_c = 9 * np.cos(arg_2l_2p) - np.cos(arg_4a_2l_2p)
        term1_c = sqrt3 * cos_theta * sub_term1_c

        sub_term2_c = np.cos(arg_3a) + 3 * np.cos(arg_1a)
        term2_c = -6 * sin_theta * sub_term2_c

        Dc = (1.0 / 16.0) * (term1_c + term2_c)

        # 5. Fplus, Fcross
        Fp = 0.5 * (cos_2psi * Dp - sin_2psi * Dc)
        Fc = 0.5 * (sin_2psi * Dp + cos_2psi * Dc)

        signal_static[i] = hplus[i] * Fp + hcross[i] * Fc

    return time_new, signal_static
def compute_LISA_response(t_src, hplus, hcross,
                          theta, phi, psi, t0=0,
                          kappa=0.0, lamb=0.0,
                          mode='interp'):
    """
    Compute the LISA response of an external waveform, supporting two output modes.

    Args:
        t_src (array): Source-frame input time array (typically uniform).

        hplus, hcross (array): Polarization components at the source.

        theta, phi, psi (float): Geometric angles (rad).

        t0 (float): Arrival time offset (s).

        kappa, lamb (float): Initial orbital/phase parameters.

        mode (str): Output mode selector

            - 'interp' (default): The output time array exactly matches the input
                                  `t_src`. Resampling is performed automatically,
                                  and uncovered regions are zero-padded.

            - 'raw': Returns the true detector arrival time array
                     (non-uniform due to Doppler effects).
                     No interpolation is applied, preserving the
                     most original physical information.

    Returns:
        t_out (array): Output time array (depends on `mode`).

        h_out (array): Output strain.
    """

    years = 365 * 24 * 3600.0
    ecc_lisa = 0.00965
    L = 5e9 / sciconsts.c
    a = L / ecc_lisa / (2 * np.sqrt(3))
    fm = 1 / years

    # 2. initial delay time at t=0
    ndot0 = a * np.sin(theta) * np.cos(kappa - phi)

    # 3. physical arrival time
    t_physical, h_modulated = core_lisa_response_loop(
        np.ascontiguousarray(t_src, dtype=np.float64),
        np.ascontiguousarray(hplus, dtype=np.float64),
        np.ascontiguousarray(hcross, dtype=np.float64),
        float(theta), float(phi), float(psi),
        float(fm), float(kappa), float(lamb), float(a), float(t0), float(ndot0)
    )

    # 4. Process the output according to the selected mode
    if mode == 'raw':
        # --- Mode 2: Return the non-uniform physical time array ---
        # Here, `t_physical` includes the grid deformation caused by Doppler effects
        return [t_physical, h_modulated]

    elif mode == 'interp':
        # --- Mode 1: Resample back onto the original grid ---
        # We want to know: what is the detector response at the times
        # defined by `t_src`?
        # This is an interpolation problem:
        # given (t_physical, h_modulated), evaluate f(t_src)

        # left=0.0, right=0.0 implements zero-padding outside
        # the covered signal range
        h_resampled = np.interp(t_src, t_physical, h_modulated, left=0.0, right=0.0)

        return [t_src, h_resampled]

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'interp' or 'raw'.")


# -----------------------------------------------------------------------------
# Numba acceleration
# -----------------------------------------------------------------------------

@njit(cache=True)
def compute_integral_sum(h1left, h1right, h2left, h2right,
                         h1_angle_left, h1_angle_right,
                         h2_angle_left, h2_angle_right,
                         snfleft, snfright,
                         xsl, xsr, phic, tobs):
    """
    Compute the overlap integral:
    4 * Re[ h1 * conj(h2) / Sn ] / 4.

    Note that the Fourier-domain coefficient `hf` contains an additional
    factor of 1/2 relative to h0, which cancels the standard prefactor of 4.

    Numba acceleration is retained since this consists purely of array operations.
    """
    n = len(h1left)
    total_sum = 0.0

    for i in range(n):
        term_left = 1.0 * h1left[i] * h2left[i] / snfleft[i] * np.cos(h1_angle_left[i] - (h2_angle_left[i] + phic))
        term_right = 1.0 * h1right[i] * h2right[i] / snfright[i] * np.cos(
            h1_angle_right[i] - (h2_angle_right[i] + phic))

        dx = xsr[i] - xsl[i]

        # (f(x_i) + f(x_{i+1})) * dx / 2
        total_sum += (term_left + term_right) * dx / 2.0

    return total_sum * tobs * tobs



def inner_product0(fs, waveform1, waveform2, phic, snf=None):
    """
    compute (SNR^2)。
    """
    num1 = len(waveform1)
    dt = 1.0 / fs
    tobs = num1 * dt

    # 1. Construct the frequency array
    # Note: the FFT output is symmetric, so only the positive-frequency
    # half (0 to Nyquist) is needed
    xs = np.linspace(0, fs / 2.0, num=num1 // 2)

    # 2. FFT
    hf_1 = scipy.fftpack.fft(waveform1)
    hf_1_abs = np.abs(hf_1)
    hf_1_angle = np.angle(hf_1)[0:num1 // 2]
    # normalize
    hf_1_norm = 2.0 / num1 * hf_1_abs[0:num1 // 2]

    hf_2 = scipy.fftpack.fft(waveform2)
    hf_2_abs = np.abs(hf_2)
    hf_2_angle = np.angle(hf_2)[0:num1 // 2]
    hf_2_norm = 2.0 / num1 * hf_2_abs[0:num1 // 2]

    # 3. noise vector Snfvec
    if snf is None:
        Snfvec = S_n_lisa(xs)
    else:
        try:
            Snfvec = snf(xs)
        except Exception:
            Snfvec = np.array([snf(f) for f in xs])

    Snfvec = np.asarray(Snfvec, dtype=np.float64)

    # 4. Slice the data
    # Remove the DC component (index 0), since it is usually
    # unphysical or associated with effectively infinite noise.
    # Prepare arrays for the Numba-accelerated routine
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

    # Simple safeguard against division-by-zero errors if Snf becomes zero
    # (although the S_n_lisa logic should never return exactly zero)
    # If Snf is extremely small, replace it with a very large value
    mask_zero = (snfleft <= 0)
    if np.any(mask_zero):
        snfleft[mask_zero] = 1e100
    mask_zero_r = (snfright <= 0)
    if np.any(mask_zero_r):
        snfright[mask_zero_r] = 1e100

    # 5. Numba
    ABval_raw = compute_integral_sum(
        h1left, h1right, h2left, h2right,
        h1_angle_left, h1_angle_right,
        h2_angle_left, h2_angle_right,
        snfleft, snfright,
        xsl, xsr, phic, tobs
    )

    ABval = abs(ABval_raw)
    return ABval


def inner_product(fs, waveform1, waveform2, phic, snf=None):
    """
    Compute the inner product (SNR^2) between two waveforms.
    """
    num1 = len(waveform1)
    dt = 1.0 / fs
    tobs = num1 * dt

    # 1. Construct the frequency array
    # The true FFT frequency spacing is fs / num1;
    # keep only the first num1 // 2 frequency bins
    xs = np.arange(num1 // 2) * (fs / num1)

    # 2. FFT
    hf_1 = scipy.fftpack.fft(waveform1)
    hf_1_abs = np.abs(hf_1)
    hf_1_angle = np.angle(hf_1)[0:num1 // 2]
    # normalize
    hf_1_norm = 2.0 / num1 * hf_1_abs[0:num1 // 2]

    hf_2 = scipy.fftpack.fft(waveform2)
    hf_2_abs = np.abs(hf_2)
    hf_2_angle = np.angle(hf_2)[0:num1 // 2]
    hf_2_norm = 2.0 / num1 * hf_2_abs[0:num1 // 2]

    # 3. generate noise vector Snfvec
    if snf is None:
        Snfvec = S_n_lisa(xs)
    else:
        try:
            Snfvec = snf(xs)
        except Exception:
            Snfvec = np.array([snf(f) for f in xs])

    Snfvec = np.asarray(Snfvec, dtype=np.float64)

    # 4. Slice the data
    # Remove the DC component (index 0), since it is usually
    # unphysical or associated with effectively infinite noise.
    # Prepare arrays for the Numba-accelerated routine
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

    mask_zero = (snfleft <= 0)
    if np.any(mask_zero):
        snfleft[mask_zero] = 1e100
    mask_zero_r = (snfright <= 0)
    if np.any(mask_zero_r):
        snfright[mask_zero_r] = 1e100

    ABval_raw = compute_integral_sum(
        h1left, h1right, h2left, h2right,
        h1_angle_left, h1_angle_right,
        h2_angle_left, h2_angle_right,
        snfleft, snfright,
        xsl, xsr, phic, tobs
    )

    ABval = abs(ABval_raw)
    return ABval


def compute_characteristic_strain_numerical(h_t, ts, plot=False):
    """
    (Characteristic Strain Spectrum, hc,num)
    """
    import scipy.fftpack
    import matplotlib.pyplot as plt
    import numpy as np

    h_t = np.asarray(h_t)
    num1 = len(h_t)

    if num1 < 2:
        print("[Error] Waveform is too short to compute FFT.")
        return np.array([]), np.array([])

    dt = 1.0 / ts
    tobs_sec = num1 * dt

    xs = np.arange(num1 // 2) * (ts / num1)

    hf = scipy.fftpack.fft(h_t)
    hf_abs = np.abs(hf)

    hN = (2.0 / num1) * hf_abs[0:num1 // 2]
    hc_num = hN * np.sqrt(xs * tobs_sec)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        f_noise = np.logspace(-7, -1, 1000)

        try:
            Snf_vals = S_n_lisa(f_noise)
        except Exception:
            Snf_vals = np.array([S_n_lisa(f) for f in f_noise])

        hc_noise = np.sqrt(f_noise * Snf_vals)

        ax.loglog(f_noise, hc_noise, color='black', label='LISA Noise', zorder=1)

        ax.loglog(xs[1:], hc_num[1:], color='blue', label=r'Numerical $h_{c, \rm num}$', zorder=2)

        ax.set_xlim(1e-5, 0.1)
        ax.set_ylim(1e-24, 1e-17)

        ax.set_xlabel(r'$f$ [Hz]', fontsize=18)
        ax.set_ylabel(r'$h_c(f)$', fontsize=18)
        ax.legend(fontsize=12, loc='upper right', framealpha=1, edgecolor='k')
        ax.tick_params(axis='both', which='major', labelsize=14, direction='in')
        ax.tick_params(axis='both', which='minor', direction='in')
        plt.grid(True, which="both", ls="--", alpha=0.2)

        plt.tight_layout()
        plt.show()

    return xs, hc_num

"""
TEST VERSION!
"""

"""
Orbit-level evolution for the x-model (Hinder et al. 2010, PRD 82, 024033;
ADM-coordinate quasi-Keplerian parametrization from Memmesheimer,
Gopakumar & Schaefer 2004, PRD 70, 104011 / arXiv:gr-qc/0407049).

Stripped-down version of `eccGW_waveform`: evolves the orbital parameters
(x, e_t, l) but does NOT build the waveform.  Also returns the reduced
(specific) energy E and angular momentum Lz along the inspiral, computed
from (x, e_t) via the PN relations.

Returns:
    [t, x, e, l, E, Lz]
    t   : time grid [s]
    x   : PN frequency parameter x = (M omega)^(2/3)        [dimensionless]
    e   : time eccentricity e_t                             [dimensionless]
    l   : mean anomaly l                                    [rad]
    E   : reduced orbital energy xi = E/mu                  [geometrized, c^2 units]
    Lz  : reduced orbital angular momentum j = L/(M mu)     [geometrized, GM/c units]

PN COEFFICIENTS
---------------
Leading (Newtonian) terms for xi(x,e_t) and j(x,e_t) are exact and hard
coded.  The 1PN/2PN/3PN coefficient blocks are LEFT AS EXPLICIT, CLEARLY
MARKED placeholders: fill them from MGS04 Eq.(25) (ADM) matching your PN
order.  Cross-check the result against the E/Lz columns of the .traj file
before trusting higher orders.
"""
# scipy renamed cumtrapz -> cumulative_trapezoid in 1.14+; support both
if hasattr(sci_integrate, "cumulative_trapezoid"):
    _cumtrapz = sci_integrate.cumulative_trapezoid
else:
    _cumtrapz = sci_integrate.cumtrapz
# ============================================================================
#  MGS04 ADM forward relations  (E, h, eta) -> (n, e_t^2)
#  Transcribed VERBATIM from Memmesheimer-Gopakumar-Schaefer 2004,
#  arXiv:gr-qc/0407049, Eq.(20c) [n] and Eq.(20d) [e_t^2], ADM coordinates.
#
#  Conventions (geometrized, G = c = 1, total mass M = 1):
#    E  = reduced energy xi = E_phys/mu  (NEGATIVE for bound orbits)
#    h  = reduced angular momentum = J/(G M mu)  ->  here L/(M mu)
#  Recurring combos:  mE2 = (-2 E);  Eh2 = (-2 E h^2)  (== 1 - e_t^2 at Newt.)
#  Since c = 1, the 1/c^2,1/c^4,1/c^6 prefactors are all 1.
# ============================================================================
@njit(fastmath=True, cache=True)
def mgs_n_of_Eh(E, h, eta):
    """Mean motion n(E,h,eta), ADM, MGS04 Eq.(20c). Geometrized (M=1)."""
    mE2 = -2.0 * E
    Eh2 = -2.0 * E * h * h
    sqEh2 = np.sqrt(Eh2)
    n = mE2 ** 1.5 * (
        1.0
        + (mE2 / 8.0) * (-15.0 + eta)
        + (mE2 ** 2 / 128.0) * (
            555.0 + 30.0 * eta + 11.0 * eta ** 2
            + (192.0 / sqEh2) * (-5.0 + 2.0 * eta)
        )
        + (mE2 ** 3 / 3072.0) * (
            -29385.0 - 4995.0 * eta - 315.0 * eta ** 2 + 135.0 * eta ** 3
            - (16.0 / Eh2 ** 1.5) * (10080.0 + 123.0 * eta * np.pi ** 2
                                     - 13952.0 * eta + 1440.0 * eta ** 2)
            + (5760.0 / sqEh2) * (17.0 - 9.0 * eta + 2.0 * eta ** 2)
        )
    )
    return n
@njit(fastmath=True, cache=True)
def mgs_et2_of_Eh(E, h, eta):
    """Time eccentricity squared e_t^2(E,h,eta), ADM, MGS04 Eq.(20d)."""
    mE2 = -2.0 * E
    Eh2 = -2.0 * E * h * h
    sqEh2 = np.sqrt(Eh2)
    et2 = (
        1.0 + 2.0 * E * h * h
        + (mE2 / 4.0) * (-8.0 + 8.0 * eta - (-17.0 + 7.0 * eta) * Eh2)
        + (mE2 ** 2 / 8.0) * (
            8.0 + 4.0 * eta + 20.0 * eta ** 2
            - Eh2 * (112.0 - 47.0 * eta + 16.0 * eta ** 2)
            - 24.0 * sqEh2 * (-5.0 + 2.0 * eta)
            + (4.0 / Eh2) * (17.0 - 11.0 * eta)
            - (24.0 / sqEh2) * (5.0 - 2.0 * eta)
        )
        + (mE2 ** 3 / 192.0) * (
            24.0 * (-2.0 + 5.0 * eta) * (-23.0 + 10.0 * eta + 4.0 * eta ** 2)
            - 15.0 * (-528.0 + 200.0 * eta - 77.0 * eta ** 2 + 24.0 * eta ** 3) * Eh2
            - 72.0 * (265.0 - 193.0 * eta + 46.0 * eta ** 2) * sqEh2
            - (2.0 / Eh2) * (6732.0 + 117.0 * eta * np.pi ** 2
                             - 12508.0 * eta + 2004.0 * eta ** 2)
            + (2.0 / sqEh2) * (16380.0 - 19964.0 * eta
                               + 123.0 * eta * np.pi ** 2 + 3240.0 * eta ** 2)
            - (2.0 / Eh2 ** 1.5) * (10080.0 + 123.0 * eta * np.pi ** 2
                                    - 13952.0 * eta + 1440.0 * eta ** 2)
            + (96.0 / Eh2 ** 2) * (134.0 - 281.0 * eta
                                   + 5.0 * eta * np.pi ** 2 + 16.0 * eta ** 2)
        )
    )
    return et2
@njit(fastmath=True, cache=True)
def hinder_Mn_of_x_et(x, et, eta, PN_orbit):
    """Mean motion n from (x, e_t) via Hinder 2010 Eq.(A1)-(A4). Returns Mn.

    CRUCIAL conversion: x = (M omega)^{2/3}, omega = (2pi+Delta phi)/P, so
    x != (Mn)^{2/3} beyond Newtonian order (periastron advance differs them).
    Identical term-by-term to the dl_dt used in the x-model evolution.
    """
    e = et
    result = x ** 1.5
    if PN_orbit >= 1:
        result += x ** 2.5 * (3.0 / (e * e - 1.0))
    if PN_orbit >= 2:
        result += x ** 3.5 * (((26.0 * eta - 51.0) * e * e + 28.0 * eta - 18.0)
                              / (4.0 * (e * e - 1.0) ** 2))
    if PN_orbit >= 3:
        result += x ** 4.5 * (-1.0) / (128.0 * (1.0 - e * e) ** 3.5) * (
            (1536.0 * eta - 3840.0) * e ** 4
            + (1920.0 - 768.0 * eta) * e * e - 768.0 * eta
            + np.sqrt(1.0 - e * e) * (
                (1040.0 * eta ** 2 - 1760.0 * eta + 2496.0) * e ** 4
                + (5120.0 * eta ** 2 + 123.0 * np.pi ** 2 * eta
                   - 17856.0 * eta + 8544.0) * e * e
                + 896.0 * eta ** 2 - 14624.0 * eta
                + 492.0 * eta * np.pi ** 2 - 192.0) + 1920.0)
    return result
@njit(fastmath=True, cache=True)
def _invert_orbit_jit(xvec, evec, eta, PN_orbit):
    """JIT core: for each sample, get true n via Hinder A1-A4, then solve the
    2x2 system { mgs_n_of_Eh=n, mgs_et2_of_Eh=e_t^2 } for (E,h) by damped
    Newton with a numerical Jacobian and continuation seeding.

    Returns (Evec, hvec, max_residual, n_bad) where n_bad counts samples
    where the PN n(x,e_t) went unphysical (PN breakdown near merger).
    """
    nlen = xvec.shape[0]
    Evec = np.empty(nlen)
    hvec = np.empty(nlen)
    max_res = 0.0
    n_bad = 0

    E = -0.5 * xvec[0]
    h0sq = 1.0 - evec[0] * evec[0]
    if h0sq < 1e-12:
        h0sq = 1e-12
    h = np.sqrt(h0sq / xvec[0])

    for i in range(nlen):
        nt = hinder_Mn_of_x_et(xvec[i], evec[i], eta, PN_orbit)
        et2 = evec[i] * evec[i]

        if (not np.isfinite(nt)) or nt <= 0.0:
            # PN n(x,e_t) broke down; carry previous solution, flag.
            n_bad += 1
            Evec[i] = E
            hvec[i] = h
            continue

        # damped Newton with numerical (central-difference) Jacobian
        rn = 1.0e30
        for _ in range(100):
            f1 = mgs_n_of_Eh(E, h, eta) - nt
            f2 = mgs_et2_of_Eh(E, h, eta) - et2
            rn = np.sqrt(f1 * f1 + f2 * f2)
            if rn < 1.0e-13:
                break

            dE = 1.0e-8 * (abs(E) + 1.0e-12)
            dh = 1.0e-8 * (abs(h) + 1.0e-12)
            # d(n,et2)/dE
            n_Ep = mgs_n_of_Eh(E + dE, h, eta); n_Em = mgs_n_of_Eh(E - dE, h, eta)
            e_Ep = mgs_et2_of_Eh(E + dE, h, eta); e_Em = mgs_et2_of_Eh(E - dE, h, eta)
            J11 = (n_Ep - n_Em) / (2.0 * dE)
            J21 = (e_Ep - e_Em) / (2.0 * dE)
            # d(n,et2)/dh
            n_hp = mgs_n_of_Eh(E, h + dh, eta); n_hm = mgs_n_of_Eh(E, h - dh, eta)
            e_hp = mgs_et2_of_Eh(E, h + dh, eta); e_hm = mgs_et2_of_Eh(E, h - dh, eta)
            J12 = (n_hp - n_hm) / (2.0 * dh)
            J22 = (e_hp - e_hm) / (2.0 * dh)

            det = J11 * J22 - J12 * J21
            if abs(det) < 1.0e-30:
                det = 1.0e-30 if det >= 0 else -1.0e-30
            # Newton step: dp = J^{-1} f
            dE_step = (J22 * f1 - J12 * f2) / det
            dh_step = (-J21 * f1 + J11 * f2) / det

            # damping: cap relative step
            relcap = 0.5
            if abs(dE_step) > relcap * (abs(E) + 1e-12):
                dE_step = relcap * (abs(E) + 1e-12) * (1.0 if dE_step > 0 else -1.0)
            if abs(dh_step) > relcap * (abs(h) + 1e-12):
                dh_step = relcap * (abs(h) + 1e-12) * (1.0 if dh_step > 0 else -1.0)

            E_new = E - dE_step
            h_new = h - dh_step
            # keep in physical region
            if E_new >= 0.0:
                E_new = -1e-12
            if h_new <= 0.0:
                h_new = 0.5 * h
            E = E_new
            h = h_new

        Evec[i] = E
        hvec[i] = h
        if rn > max_res:
            max_res = rn

    return Evec, hvec, max_res, n_bad
def reduced_energy_and_h(xvec, evec, eta, PN_orbit=3, method='invert',
                         verbose=True):
    """Map (x, e_t) -> (E, h) along the inspiral.

    method='invert' (default): full PN. Two-step, gauge-consistent --
        (1) n(x,e_t) from Hinder Eq.(A1-A4); (2) solve MGS04 Eq.(20c,20d)
        for (E,h) by jitted damped Newton with continuation. E,h are ADM,
        gauge-dependent at 2PN+.
    method='newtonian': leading-order CLOSED FORM, fully vectorized --
        E = -x/2,  h = sqrt((1-e_t^2)/x).
        Instant; use as a baseline to compare against the full inversion.

    Returns (Evec, hvec, max_residual). For 'newtonian', residual is 0.0.
    """
    vprint = print if verbose else (lambda *a, **k: None)
    xvec = np.asarray(xvec, dtype=np.float64)
    evec = np.asarray(evec, dtype=np.float64)

    if method == 'newtonian':
        Evec = -0.5 * xvec
        one_m_e2 = np.clip(1.0 - evec ** 2, 1e-12, None)
        hvec = np.sqrt(one_m_e2 / xvec)
        vprint("  E,Lz via Newtonian closed form (E=-x/2, h=sqrt((1-e^2)/x)).")
        return Evec, hvec, 0.0

    if method != 'invert':
        raise ValueError("method must be 'invert' or 'newtonian'")

    Evec, hvec, max_res, n_bad = _invert_orbit_jit(xvec, evec,
                                                   float(eta), int(PN_orbit))
    vprint(f"  MGS04 inversion max residual: {max_res:.3e} (target <~1e-9)")
    if n_bad > 0:
        vprint(f"  [warn] {n_bad} sample(s) had unphysical n(x,e_t) "
               f"(PN breakdown near merger); previous solution carried.")
    return Evec, hvec, max_res
# ============================================================================
#  Orbit-only evolver  (mirrors eccGW_waveform up to the orbital level)
# ============================================================================
def ecc_orbit_evolution0(f00, e0, timescale, m1, m2,
                        l0=0.0, ts=None,
                        PN_orbit=3, PN_reaction=2,
                        N=50, max_memory_GB=16.0, verbose=True,
                        kappaE=None, kappaJ=None,
                        E_method='invert'):
    """Evolve (x, e_t, l) and return [t, x, e, l, E, Lz].

    Mirrors eccGW_waveform's argument list. kappaE/kappaJ are the 1.5PN
    tail-enhancement callables; if PN_reaction < 1.5 they are unused.
    E_method: 'invert' (full 3PN MGS04 inversion, jitted) or 'newtonian'
    (leading-order closed form E=-x/2, h=sqrt((1-e^2)/x); fast baseline).
    """
    vprint = print if verbose else (lambda *a, **k: None)

    if e0 == 0:
        vprint("Warning: e0=0 unsupported; resetting to 1e-5.")
        e0 = 1e-5

    M = m1 + m2
    smr = m1 * m2 / M / M     # symmetric mass ratio nu
    nu = smr

    if kappaE is None:
        kappaE = lambda e: 1.0
    if kappaJ is None:
        kappaJ = lambda e: 1.0

    # ---- 2PN edot helper (verbatim from parent) ---------------------------
    def E2PN(e):
        return -e * smr / (30240 * np.power(1 - e * e, 9 / 2)) * (
            (2758560 * smr * smr - 4344852 * smr + 3786543) * np.power(e, 6.0) + (
                42810096 * smr * smr - 78112266 * smr + 46579718) * np.power(e, 4.0) + (
                48711348 * smr * smr - 35583228 * smr - 36993396) * e * e + 4548096 * smr * smr + np.sqrt(
                1 - e * e) * ((2847600 - 1139040 * smr) * np.power(e, 4.0) + (
                    35093520 - 14037408 * smr) * e * e - 5386752 * smr + 13466880)
            + 13509360 * smr - 15198032)

    # ---- coupled dx/dt, de/dt (verbatim from parent) ----------------------
    def coupled_derivs(y, t):
        x = y[0]; e = y[1]
        if x < 0: x = 1e-8
        if e < 0: e = 0.0
        if e >= 1.0: e = 0.99999
        one_minus_e2 = 1.0 - e * e
        if one_minus_e2 < 1e-10: one_minus_e2 = 1e-10

        xdot = np.power(x, 5.0) * 2 * (37 * np.power(e, 4.0) + 292 * e * e + 96) * smr / (
            15 * np.power(one_minus_e2, 3.5))
        if PN_reaction >= 1:
            term_1pn = -(8288 * smr - 11717) * np.power(e, 6.0) \
                       - 14 * (10122 * smr - 12217) * np.power(e, 4.0) \
                       - 120 * (1330 * smr - 731) * e * e \
                       - 16 * (924 * smr + 743)
            xdot += np.power(x, 6.0) * smr / (420 * np.power(one_minus_e2, 4.5)) * term_1pn
        if PN_reaction >= 1.5:
            xdot += np.power(x, 6.5) * 256 / 5 * smr * pi * kappaE(e)
        if PN_reaction >= 2:
            term_2pn = (1964256 * smr * smr - 3259980 * smr + 3523113) * np.power(e, 8.0) + \
                       (64828848 * smr * smr - 123108426 * smr + 83424402) * np.power(e, 6.0) + \
                       (16650606060 * smr * smr - 207204264 * smr + 783768) * np.power(e, 4.0) + \
                       (61282032 * smr * smr + 15464736 * smr - 92846560) * e * e + 1903104 * smr * smr + \
                       np.sqrt(one_minus_e2) * ((2646000 - 1058400 * smr) * np.power(e, 6.0) + (
                           64532160 - 25812864 * smr) * e * e - 580608 * smr + 1451520) + \
                       4514976 * smr - 360224
            xdot += np.power(x, 7.0) * smr / (45360 * np.power(one_minus_e2, 5.5)) * term_2pn
        dx_dt_val = xdot / M

        if e <= 1e-9:
            edot_val = 0.0
        else:
            edot = np.power(x, 4.0) * (-e * (121 * e * e + 304) * smr / (15 * np.power(one_minus_e2, 2.5)))
            if PN_reaction >= 1:
                term_e_1pn = (93184 * smr - 125361) * np.power(e, 4.0) + \
                             12 * (54271 * smr - 59834) * e * e + \
                             8 * (28588 * smr + 8451)
                edot += np.power(x, 5.0) * e * smr / (2520 * np.power(one_minus_e2, 3.5)) * term_e_1pn
            if PN_reaction >= 1.5:
                edot += np.power(x, 5.5) * 128 * smr * pi / 5 / e * (
                    (e * e - 1) * kappaE(e) + np.sqrt(one_minus_e2) * kappaJ(e))
            if PN_reaction >= 2:
                edot += np.power(x, 6.0) * E2PN(e)
            edot_val = edot / M
        return [dx_dt_val, edot_val]

    # ---- dl/dt (verbatim from parent) -------------------------------------
    def dl_dt(x, e):
        result = np.power(x, 3 / 2)
        if PN_orbit >= 1:
            result += np.power(x, 5 / 2) * 3 / (e * e - 1)
        if PN_orbit >= 2:
            result += np.power(x, 7 / 2) * ((26 * smr - 51) * e * e + 28 * smr - 18) / (
                4 * np.power(e * e - 1, 2.0))
        if PN_orbit >= 3:
            result += np.power(x, 9 / 2) * (-1) / (128 * np.power(1 - e * e, 7 / 2)) * (
                (1536 * smr - 3840) * np.power(e, 4.0) + (1920 - 768 * smr) * e * e - 768 * smr + np.sqrt(
                    1 - e * e) * ((1040 * smr * smr - 1760 * smr + 2496) * np.power(e, 4.0) + (
                        5120 * smr * smr + 123 * pi * pi * smr - 17856 * smr + 8544) * e * e
                    + 896 * smr * smr - 14624 * smr + 492 * smr * pi * pi - 192) + 1920)
        return result / M

    def deltafvalue(a, e, Mtot):
        n = np.power(a, -3 / 2) * np.sqrt(Mtot)
        Porb = 1 / (n / (2 * pi))
        return 6 * np.power(2 * pi, 2 / 3) / (1 - e * e) * np.power(Mtot, 2 / 3) * np.power(Porb, -5 / 3)

    # ---- initial conditions (verbatim) ------------------------------------
    a0 = np.power((m1 + m2) / (2 * pi * f00) ** 2, 1 / 3)
    deltaf = deltafvalue(a0, e0, M)
    f0 = f00 + deltaf / 2
    omega0 = f0 * 2 * pi
    x0 = np.power((m1 + m2) * omega0, 2 / 3)

    vprint(f'PN_EOM={PN_orbit}; PN_Reaction={PN_reaction}')
    vprint(f'm1,m2={m1},{m2}; e0={e0}')
    vprint(f'f_orb={f00} Hz; f_angular={f0} Hz; deltaf={deltaf} Hz; x0={x0}')

    rp_check = (1.0 - e0) / x0
    if rp_check < 6.0:
        raise ValueError(f"Initial condition inside ISCO: rp={rp_check:.2f} M (<6M).")

    # ---- integrate x, e ---------------------------------------------------
    t_accuracy = int(timescale * f0 * 2) + 5000
    t_temp = np.linspace(0, timescale, num=t_accuracy)
    y0 = [x0, e0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = sci_integrate.odeint(coupled_derivs, y0, t_temp, rtol=1e-12, atol=1e-14)
        xresult = sol[:, 0]
        eresult = sol[:, 1]

        x_isco = 1.0 / 6.0
        bad_mask = (xresult >= x_isco) | np.isnan(xresult) | np.isinf(xresult)
        bad_indices = np.where(bad_mask)[0]
        if len(bad_indices) > 0:
            cutoff_index = bad_indices[0]
            vprint(f"   Merger/ISCO at t={t_temp[cutoff_index]} s (x={xresult[cutoff_index]}). Truncating.")
        else:
            cutoff_index = len(t_temp)
            if xresult[-1] < 0.1:
                vprint(f"   Finished without merger (final x={xresult[-1]}).")

        if cutoff_index < 2:
            vprint("Error: truncated immediately.")
            empty = np.array([])
            return [empty, empty, empty, empty, empty, empty]

        t_temp = t_temp[:cutoff_index]
        xresult = xresult[:cutoff_index]
        eresult = eresult[:cutoff_index]
        eresult[eresult < 0] = 0.0

        dl_vals = np.array([dl_dt(xresult[i], eresult[i]) for i in range(len(xresult))])
        lresult = _cumtrapz(dl_vals, t_temp, initial=0) + l0

    timescale = t_temp[-1]
    flast = 1 / 2 / pi / M * np.power(xresult[-1], 3 / 2)
    vprint(f'Evolution: {timescale/(365*24*3600):.4f} yr. f0={f0}, f_final={flast}')

    # ---- resample to output grid (same logic as parent) -------------------
    if ts is None:
        controlnum2 = int(N * timescale * flast * np.power(1 - eresult[-1], -3 / 2)) + 1000
    else:
        controlnum2 = int(timescale / ts)
    if controlnum2 < 100: controlnum2 = 100
    est_memory_bytes = controlnum2 * 12 * 8
    max_bytes = max_memory_GB * (1024 ** 3)
    if est_memory_bytes > max_bytes:
        controlnum2 = int(max_bytes / (12 * 8))
        vprint(f"[WARN] memory cap -> {controlnum2} points")

    t3 = np.linspace(0, timescale, num=controlnum2)

    evec = np.interp(t3, t_temp, eresult)
    xvec = np.interp(t3, t_temp, xresult)
    lvec = np.interp(t3, t_temp, lresult)

    # ---- E and Lz from (x, e_t) via MGS04 ADM inversion -------------------
    Evec, hvec, max_res = reduced_energy_and_h(xvec, evec, nu,
                                               PN_orbit=PN_orbit,
                                               method=E_method,
                                               verbose=verbose)
    Lzvec = hvec  # reduced angular momentum h = L/(M mu)

    vprint(f"Output points: {len(t3)}  dt={t3[1]-t3[0]:.4g} s")
    vprint(f"  x:  [{xvec.min():.4g}, {xvec.max():.4g}]")
    vprint(f"  e:  [{evec.min():.4g}, {evec.max():.4g}]")
    vprint(f"  E:  [{Evec.min():.4g}, {Evec.max():.4g}]  (reduced xi=E/mu, ADM)")
    vprint(f"  Lz: [{Lzvec.min():.4g}, {Lzvec.max():.4g}] (reduced h=L/(M mu), ADM)")

    return [t3, xvec, evec, lvec, Evec, Lzvec]

# ============================================================================
#  Orbit-only evolver  (Strictly preserves original logic, optimized for speed)
# ============================================================================
def ecc_orbit_evolution(f00, e0, timescale, m1, m2,
                        l0=0.0, ts=None,
                        PN_orbit=3, PN_reaction=2,
                        N=50, max_memory_GB=16.0, verbose=True,
                        kappaE=None, kappaJ=None,
                        E_method='invert'):
    vprint = print if verbose else (lambda *a, **k: None)

    if e0 == 0:
        vprint("Warning: e0=0 unsupported; resetting to 1e-5.")
        e0 = 1e-5

    M = m1 + m2
    smr = m1 * m2 / M / M  # symmetric mass ratio nu
    nu = smr

    if kappaE is None:
        kappaE = lambda e: 1.0
    if kappaJ is None:
        kappaJ = lambda e: 1.0

    # ---- 2PN edot helper (verbatim from parent) ---------------------------
    def E2PN(e):
        return -e * smr / (30240 * np.power(1 - e * e, 9 / 2)) * (
                (2758560 * smr * smr - 4344852 * smr + 3786543) * np.power(e, 6.0) + (
                42810096 * smr * smr - 78112266 * smr + 46579718) * np.power(e, 4.0) + (
                        48711348 * smr * smr - 35583228 * smr - 36993396) * e * e + 4548096 * smr * smr + np.sqrt(
            1 - e * e) * ((2847600 - 1139040 * smr) * np.power(e, 4.0) + (
                35093520 - 14037408 * smr) * e * e - 5386752 * smr + 13466880)
                + 13509360 * smr - 15198032)

    # ---- coupled dx/dt, de/dt (verbatim from parent) ----------------------
    def coupled_derivs(y, t):
        x = y[0];
        e = y[1]
        if x < 0: x = 1e-8
        if e < 0: e = 0.0
        if e >= 1.0: e = 0.99999
        one_minus_e2 = 1.0 - e * e
        if one_minus_e2 < 1e-10: one_minus_e2 = 1e-10

        xdot = np.power(x, 5.0) * 2 * (37 * np.power(e, 4.0) + 292 * e * e + 96) * smr / (
                15 * np.power(one_minus_e2, 3.5))
        if PN_reaction >= 1:
            term_1pn = -(8288 * smr - 11717) * np.power(e, 6.0) \
                       - 14 * (10122 * smr - 12217) * np.power(e, 4.0) \
                       - 120 * (1330 * smr - 731) * e * e \
                       - 16 * (924 * smr + 743)
            xdot += np.power(x, 6.0) * smr / (420 * np.power(one_minus_e2, 4.5)) * term_1pn
        if PN_reaction >= 1.5:
            xdot += np.power(x, 6.5) * 256 / 5 * smr * pi * kappaE(e)
        if PN_reaction >= 2:
            term_2pn = (1964256 * smr * smr - 3259980 * smr + 3523113) * np.power(e, 8.0) + \
                       (64828848 * smr * smr - 123108426 * smr + 83424402) * np.power(e, 6.0) + \
                       (16650606060 * smr * smr - 207204264 * smr + 783768) * np.power(e, 4.0) + \
                       (61282032 * smr * smr + 15464736 * smr - 92846560) * e * e + 1903104 * smr * smr + \
                       np.sqrt(one_minus_e2) * ((2646000 - 1058400 * smr) * np.power(e, 6.0) + (
                    64532160 - 25812864 * smr) * e * e - 580608 * smr + 1451520) + \
                       4514976 * smr - 360224
            xdot += np.power(x, 7.0) * smr / (45360 * np.power(one_minus_e2, 5.5)) * term_2pn
        dx_dt_val = xdot / M

        if e <= 1e-9:
            edot_val = 0.0
        else:
            edot = np.power(x, 4.0) * (-e * (121 * e * e + 304) * smr / (15 * np.power(one_minus_e2, 2.5)))
            if PN_reaction >= 1:
                term_e_1pn = (93184 * smr - 125361) * np.power(e, 4.0) + \
                             12 * (54271 * smr - 59834) * e * e + \
                             8 * (28588 * smr + 8451)
                edot += np.power(x, 5.0) * e * smr / (2520 * np.power(one_minus_e2, 3.5)) * term_e_1pn
            if PN_reaction >= 1.5:
                edot += np.power(x, 5.5) * 128 * smr * pi / 5 / e * (
                        (e * e - 1) * kappaE(e) + np.sqrt(one_minus_e2) * kappaJ(e))
            if PN_reaction >= 2:
                edot += np.power(x, 6.0) * E2PN(e)
            edot_val = edot / M
        return [dx_dt_val, edot_val]

    # ---- dl/dt (原样保持，但是可以支持向量化输入) -------------------------
    def dl_dt(x, e):
        result = np.power(x, 3 / 2)
        if PN_orbit >= 1:
            result += np.power(x, 5 / 2) * 3 / (e * e - 1)
        if PN_orbit >= 2:
            result += np.power(x, 7 / 2) * ((26 * smr - 51) * e * e + 28 * smr - 18) / (
                    4 * np.power(e * e - 1, 2.0))
        if PN_orbit >= 3:
            result += np.power(x, 9 / 2) * (-1) / (128 * np.power(1 - e * e, 7 / 2)) * (
                    (1536 * smr - 3840) * np.power(e, 4.0) + (1920 - 768 * smr) * e * e - 768 * smr + np.sqrt(
                1 - e * e) * ((1040 * smr * smr - 1760 * smr + 2496) * np.power(e, 4.0) + (
                    5120 * smr * smr + 123 * pi * pi * smr - 17856 * smr + 8544) * e * e
                              + 896 * smr * smr - 14624 * smr + 492 * smr * pi * pi - 192) + 1920)
        return result / M

    def deltafvalue(a, e, Mtot):
        n = np.power(a, -3 / 2) * np.sqrt(Mtot)
        Porb = 1 / (n / (2 * pi))
        return 6 * np.power(2 * pi, 2 / 3) / (1 - e * e) * np.power(Mtot, 2 / 3) * np.power(Porb, -5 / 3)

    # ---- initial conditions (严格保持原版) --------------------------------
    a0 = np.power((m1 + m2) / (2 * pi * f00) ** 2, 1 / 3)
    deltaf = deltafvalue(a0, e0, M)
    f0 = f00 + deltaf / 2
    omega0 = f0 * 2 * pi
    x0 = np.power((m1 + m2) * omega0, 2 / 3)

    vprint(f'PN_EOM={PN_orbit}; PN_Reaction={PN_reaction}')
    vprint(f'm1,m2={m1},{m2}; e0={e0}')
    vprint(f'f_orb={f00} Hz; f_angular={f0} Hz; deltaf={deltaf} Hz; x0={x0}')

    rp_check = (1.0 - e0) / x0
    if rp_check < 6.0:
        raise ValueError(f"Initial condition inside ISCO: rp={rp_check:.2f} M (<6M).")

    # ---- integrate x, e (严格保持原版的致密网格与精度设置) ----------------
    t_accuracy = int(timescale * f0 /10) + 5000
    t_temp = np.linspace(0, timescale, num=t_accuracy)
    y0 = [x0, e0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 严格保持原始的高容差设定 (rtol=1e-12, atol=1e-14)
        sol = sci_integrate.odeint(coupled_derivs, y0, t_temp, rtol=1e-12, atol=1e-14)
        xresult = sol[:, 0]
        eresult = sol[:, 1]

        x_isco = 1.0 / 6.0
        bad_mask = (xresult >= x_isco) | np.isnan(xresult) | np.isinf(xresult)
        bad_indices = np.where(bad_mask)[0]
        if len(bad_indices) > 0:
            cutoff_index = bad_indices[0]
            vprint(f"   Merger/ISCO at t={t_temp[cutoff_index]} s (x={xresult[cutoff_index]}). Truncating.")
        else:
            cutoff_index = len(t_temp)
            if xresult[-1] < 0.1:
                vprint(f"   Finished without merger (final x={xresult[-1]}).")

        if cutoff_index < 2:
            vprint("Error: truncated immediately.")
            empty = np.array([])
            return [empty, empty, empty, empty, empty, empty]

        t_temp = t_temp[:cutoff_index]
        xresult = xresult[:cutoff_index]
        eresult = eresult[:cutoff_index]
        eresult[eresult < 0] = 0.0

        # === 唯一优化的点：将原本的 for 循环替换为 Numpy 向量化计算 ===
        # 原代码：dl_vals = np.array([dl_dt(xresult[i], eresult[i]) for i in range(len(xresult))])
        dl_vals = dl_dt(xresult, eresult)

        lresult = _cumtrapz(dl_vals, t_temp, initial=0) + l0

    timescale = t_temp[-1]
    flast = 1 / 2 / pi / M * np.power(xresult[-1], 3 / 2)
    vprint(f'Evolution: {timescale / (365 * 24 * 3600):.4f} yr. f0={f0}, f_final={flast}')

    # ---- resample to output grid (严格保持原版的插值与内存限制) -----------
    if ts is None:
        controlnum2 = int(N * timescale * flast * np.power(1 - eresult[-1], -3 / 2)) + 1000
    else:
        controlnum2 = int(timescale / ts)

    if controlnum2 < 100: controlnum2 = 100
    est_memory_bytes = controlnum2 * 12 * 8
    max_bytes = max_memory_GB * (1024 ** 3)
    if est_memory_bytes > max_bytes:
        controlnum2 = int(max_bytes / (12 * 8))
        vprint(f"[WARN] memory cap -> {controlnum2} points")

    t3 = np.linspace(0, timescale, num=controlnum2)

    evec = np.interp(t3, t_temp, eresult)
    xvec = np.interp(t3, t_temp, xresult)
    lvec = np.interp(t3, t_temp, lresult)

    # ---- E and Lz from (x, e_t) via MGS04 ADM inversion -------------------
    Evec, hvec, max_res = reduced_energy_and_h(xvec, evec, nu,
                                               PN_orbit=PN_orbit,
                                               method=E_method,
                                               verbose=verbose)
    Lzvec = hvec  # reduced angular momentum h = L/(M mu)

    vprint(f"Output points: {len(t3)}  dt={t3[1] - t3[0]:.4g} s")
    vprint(f"  x:  [{xvec.min():.4g}, {xvec.max():.4g}]")
    vprint(f"  e:  [{evec.min():.4g}, {evec.max():.4g}]")
    vprint(f"  E:  [{Evec.min():.4g}, {Evec.max():.4g}]  (reduced xi=E/mu, ADM)")
    vprint(f"  Lz: [{Lzvec.min():.4g}, {Lzvec.max():.4g}] (reduced h=L/(M mu), ADM)")

    return [t3, xvec, evec, lvec, Evec, Lzvec]