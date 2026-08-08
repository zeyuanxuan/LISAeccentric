import numpy as np
import scipy.constants as sciconsts
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
import shutil  # Added for file operations
import importlib
import builtins     # For print overriding
import warnings     # For warning control
from scipy.optimize import newton
import contextlib
import io
import os
import sys
import threading
import time
from numba import njit

#transform to G=c=1 unit
m_sun = 1.98840987e30 * sciconsts.G / np.power(sciconsts.c, 3.0)
pi=sciconsts.pi
years = 365 * 24 * 3600.0
days=24*3600
pc = 3.261 * sciconsts.light_year/sciconsts.c
AU=sciconsts.au/sciconsts.c

try:
    from .GN_modeling import GN_BBH
    from .GC_modeling import GC_BBH
    from .Field_modeling import Field_BBH, Field_BBH_Elliptical
    from .Waveform_modeling import PN_waveform, hc_cal
except ImportError as e:
    import os
    print(f"CRITICAL ERROR: Package import failed in {__file__}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Error Details: {e}")
    raise e

# ==============================================================================
# Global Output Control (Dual Switches)
# ==============================================================================
_VERBOSE = True
_SHOW_WARNINGS = True


def set_output_control(verbose: bool = True, show_warnings: bool = True):
    global _VERBOSE, _SHOW_WARNINGS
    _VERBOSE = verbose
    _SHOW_WARNINGS = show_warnings
def set_verbose(verbose: bool):
    set_output_control(verbose=verbose, show_warnings=verbose)
def print(*args, **kwargs):
    if _VERBOSE:
        builtins.print(*args, **kwargs)

def mute_if_global_verbose_false(func):
    def wrapper(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            if not _VERBOSE:
                fnull = stack.enter_context(open(os.devnull, 'w'))
                stack.enter_context(contextlib.redirect_stdout(fnull))

            if not _SHOW_WARNINGS:
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("ignore")

            return func(*args, **kwargs)

    return wrapper






@njit(fastmath=True, cache=True)
def _geo_Esq_Lzsq(p, e):
    """Return (E_bar^2, Lz^2) for Schwarzschild equatorial geodesic, M=1."""
    denom = p * (p - 3.0 - e * e)
    Esq = (p - 2.0 - 2.0 * e) * (p - 2.0 + 2.0 * e) / denom
    Lzsq = p * p / (p - 3.0 - e * e)
    return Esq, Lzsq


@njit(fastmath=True, cache=True)
def _invert_geodesic_pe(Ebar_vec, Lz_vec, p_seed0, e_seed0):
    """For each (E_bar, Lz), solve the 2x2 geodesic system for (p, e) by
    damped Newton with numerical Jacobian + continuation. Returns
    (p_vec, e_vec, max_residual)."""
    n = Ebar_vec.shape[0]
    p_vec = np.empty(n)
    e_vec = np.empty(n)
    max_res = 0.0

    p = p_seed0
    e = e_seed0
    for i in range(n):
        Esq_t = Ebar_vec[i] * Ebar_vec[i]
        Lzsq_t = Lz_vec[i] * Lz_vec[i]
        rn = 1e30
        for _ in range(100):
            Esq, Lzsq = _geo_Esq_Lzsq(p, e)
            f1 = Esq - Esq_t
            f2 = Lzsq - Lzsq_t
            rn = np.sqrt(f1 * f1 + f2 * f2)
            if rn < 1e-13:
                break
            dp = 1e-8 * (abs(p) + 1e-12)
            de = 1e-8 * (abs(e) + 1e-12)
            Ep1, Lp1 = _geo_Esq_Lzsq(p + dp, e)
            Ep0, Lp0 = _geo_Esq_Lzsq(p - dp, e)
            Ee1, Le1 = _geo_Esq_Lzsq(p, e + de)
            Ee0, Le0 = _geo_Esq_Lzsq(p, e - de)
            J11 = (Ep1 - Ep0) / (2.0 * dp)
            J21 = (Lp1 - Lp0) / (2.0 * dp)
            J12 = (Ee1 - Ee0) / (2.0 * de)
            J22 = (Le1 - Le0) / (2.0 * de)
            det = J11 * J22 - J12 * J21
            if abs(det) < 1e-30:
                det = 1e-30 if det >= 0 else -1e-30
            dp_step = (J22 * f1 - J12 * f2) / det
            de_step = (-J21 * f1 + J11 * f2) / det
            cap = 0.5
            if abs(dp_step) > cap * (abs(p) + 1e-12):
                dp_step = cap * (abs(p) + 1e-12) * (1.0 if dp_step > 0 else -1.0)
            if abs(de_step) > cap * (abs(e) + 1e-12):
                de_step = cap * (abs(e) + 1e-12) * (1.0 if de_step > 0 else -1.0)
            p = p - dp_step
            e = e - de_step
            if e < 0.0:
                e = 0.0
            if e > 0.999999:
                e = 0.999999
            if p < 4.0:  # below this no bound geodesic; clamp
                p = 4.0
        p_vec[i] = p
        e_vec[i] = e
        if rn > max_res:
            max_res = rn
    return p_vec, e_vec, max_res

# ==============================================================================
# CompactBinary
# ==============================================================================
@dataclass
class CompactBinary:
    """
    Compact Binary System Data Object.

    Attributes:
        m1, m2 (float): Masses in Solar Mass [M_sun]. (Mandatory)
        a (float): Semi-major Axis in AU.
        e (float): Eccentricity (0 <= e < 1).
        Dl (float): Luminosity Distance in kpc.
        label (str): Environment/Origin tag (e.g., 'GN_Steadystate', 'Field').
        extra (dict): Storage for variable extra parameters (SNR, Lifetime, Rates, etc.).
    """

    m1: float
    m2: float
    a: float
    e: float
    Dl: float
    label: str

    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        """String representation (Updated to print all extra params)."""
        info = f"<CompactBinary [{self.label}]: M={self.m1:.1f}+{self.m2:.1f} m_sun, a={self.a:.3e}AU, e={self.e:.4f}, Dl={self.Dl:.1f}kpc"

        if self.extra:
            extra_str_list = []
            for k, v in self.extra.items():
                if isinstance(v, float):
                    val_str = f"{v:.2e}" if (abs(v) < 1e-2 or abs(v) > 1e4) and v != 0 else f"{v:.3f}"
                else:
                    val_str = str(v)
                extra_str_list.append(f"{k}={val_str}")

            info += " | " + ", ".join(extra_str_list)

        return info + ">"

    def info(self):
        """Print detailed information nicely."""
        print("=" * 50)
        print(f"Compact Binary System | Label: {self.label}")
        print("-" * 50)
        print(f"  > Primary Mass   (m1) : {self.m1} M_sun")
        print(f"  > Secondary Mass (m2) : {self.m2} M_sun")
        print(f"  > Semi-major Axis (a) : {self.a:.6e} AU")
        print(f"  > Eccentricity    (e) : {self.e:.6f}")
        print(f"  > Distance       (Dl) : {self.Dl:.2f} kpc")

        if self.extra:
            print("-" * 50)
            print("  > Extra / Derived Parameters:")
            for k, v in self.extra.items():
                val_str = f"{v:.4e}" if isinstance(v, float) else f"{v}"
                print(f"      - {k:<15}: {val_str}")
        print("=" * 50)

    @property
    def chirp_mass(self):
        """Calculate Chirp Mass [M_sun]."""
        mtot = self.m1 + self.m2
        return (self.m1 * self.m2) ** 0.6 / (mtot) ** 0.2

    # --------------------------------------------------------------------------
    # Added Analysis Methods
    # --------------------------------------------------------------------------
    @mute_if_global_verbose_false
    def compute_waveform(self, tobs_yr, initial_orbital_phase=0,
                         theta=np.pi / 4, phi=np.pi / 4,
                         PN_orbit=3, PN_reaction=2,
                         ts=None, points_per_peak=50,
                         max_memory_GB=16.0, verbose=True, plot=True):
        """
        Compute GW waveform for this system.

        Args:
            tobs_yr (float): Observation duration in years.
            initial_orbital_phase (float): Initial phase.
            theta, phi (float): Sky position angles.
            PN_orbit, PN_reaction (int): Post-Newtonian orders.
            ts (float, optional): Fixed time step in seconds.
            points_per_peak (int): Adaptive sampling density.
        """
        # Unit conversion
        m1_geo = self.m1 * m_sun
        m2_geo = self.m2 * m_sun
        Dl_geo = self.Dl * 1e3 * pc
        tobs_geo = tobs_yr * years

        # Calculate f_orb from a (assuming a is in AU)
        M_total = m1_geo + m2_geo
        a_geo = self.a * AU
        if a_geo > 0:
            f_orb = np.sqrt(M_total / (4 * pi ** 2 * np.power(a_geo, 3.0)))
        else:
            print("[CompactBinary] Error: Semi-major axis must be positive.")
            return None

        if verbose:
            print(f"\n[CompactBinary] Computing Waveform for {self.label}...")
            print(f"Params: m={self.m1}+{self.m2}, a={self.a} AU, e={self.e}, Dl={self.Dl} kpc")

        wf = PN_waveform.eccGW_waveform(
            f_orb, self.e, tobs_geo, m1_geo, m2_geo, theta, phi, Dl_geo,
            l0=initial_orbital_phase,
            ts=ts,
            N=points_per_peak,
            PN_orbit=PN_orbit, PN_reaction=PN_reaction,
            max_memory_GB=max_memory_GB, verbose=verbose
        )

        if plot and wf is not None:
            plt.figure(figsize=(8, 6), dpi=100)
            plt.plot(wf[0], wf[1], color='BLUE', label='h_plus')
            plt.xlabel("t [s]")
            plt.ylabel("Strain")
            plt.title(f"Waveform: {self.label}")
            plt.legend()
            plt.show()

        return wf

    @mute_if_global_verbose_false
    def get_spectrum(self, tobs_yr, ts=None, polarization='hplus',
                     theta=np.pi / 4, phi=np.pi / 4,
                     initial_orbital_phase=0,
                     PN_orbit=3, PN_reaction=2,
                     points_per_peak=50, max_memory_GB=16.0,
                     plot=True, verbose=True):
        """
        自动生成波形并计算数值特征应变谱 (hc, num)。

        Args:
            tobs_yr (float):  (years)
            ts (float, optional):  (seconds)。if None, use self-adaptive sampling
            polarization (str): 'hplus' or 'hcross'
            theta, phi (float): observational angles
            plot (bool):
            verbose (bool):

        Returns:
            (xs, hc_num): frequency axis and characteristic strain
        """
        if verbose:
            print(f"\n[CompactBinary] Computing Numerical Spectrum for {self.label}...")
            print(f"                Polarization: {polarization}, Tobs: {tobs_yr} yr")

        wf = self.compute_waveform(
            tobs_yr=tobs_yr,
            initial_orbital_phase=initial_orbital_phase,
            theta=theta,
            phi=phi,
            PN_orbit=PN_orbit,
            PN_reaction=PN_reaction,
            ts=ts,
            points_per_peak=points_per_peak,
            max_memory_GB=max_memory_GB,
            verbose=False,
            plot=False
        )

        if wf is None or len(wf[0]) < 2:
            print("[Error] Waveform generation failed or returned too few points.")
            return np.array([]), np.array([])

        t_arr, hplus, hcross = wf

        dt_actual = t_arr[1] - t_arr[0]
        if dt_actual <= 0:
            print("[Error] Invalid time step in generated waveform.")
            return np.array([]), np.array([])

        fs = 1.0 / dt_actual

        if polarization.lower() == 'hplus':
            h_target = hplus
        elif polarization.lower() == 'hcross':
            h_target = hcross
        else:
            print(f"[Warning] Unknown polarization '{polarization}', defaulting to 'hplus'.")
            h_target = hplus

        xs, hc_num = PN_waveform.compute_characteristic_strain_numerical(
            h_t=h_target,
            ts=fs,
            plot=plot
        )


        return xs, hc_num
    @mute_if_global_verbose_false
    def compute_snr_analytical0(self, tobs_yr, quick_analytical=False, verbose=True):
        """
        Compute Sky-Averaged SNR (Analytical).

        Args:
            tobs_yr (float): Observation duration in years.
            quick_analytical (bool): Use fast geometric approximation.
        """
        m1_s = self.m1 * m_sun
        m2_s = self.m2 * m_sun
        a_s = self.a * AU
        Dl_s = self.Dl * 1000.0 * pc
        tobs_s = tobs_yr * years

        snr = 0.0

        if quick_analytical:
            if self.a <= 0 or self.e >= 1.0:
                return 0.0

            used_tobs = tobs_s
            try:
                t_lower = PN_waveform.tmerger_lower(m1_s, m2_s, a_s, self.e)
                if t_lower <= tobs_s:
                    t_real = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, self.e)
                    if t_real <= tobs_s:
                        used_tobs = t_real
            except:
                pass  # Fallback to standard tobs

            rp_s = a_s * (1 - self.e)
            if rp_s > 0:
                term_f = (m1_s + m2_s) / (4 * pi * pi * np.power(rp_s, 3.0))
                f0max = 2 * np.sqrt(term_f)
                h0max = np.sqrt(32 / 5) * m1_s * m2_s / (Dl_s * a_s * (1 - self.e))
                Sn_val = PN_waveform.S_n_lisa(f0max)
                if Sn_val > 0:
                    snr = h0max / np.sqrt(Sn_val) * np.sqrt(used_tobs * np.power(1 - self.e, 1.5))
        else:
            snr = PN_waveform.SNR(m1_s, m2_s, a_s, self.e, Dl_s, tobs_s)

        if verbose:
            print(f"[CompactBinary] SNR ({'Quick' if quick_analytical else 'Full'}) = {snr:.4f}")

        # Optionally store in extra
        self.extra['snr_analytical'] = snr
        return snr

    @mute_if_global_verbose_false
    def compute_snr_analytical(self, tobs_yr, quick_analytical=False, verbose=True,
                               force_evolve=False, evolve_threshold=10.0,
                               n_segments=100, f_orb_max_Hz=0.1):
        """
        Compute Sky-Averaged SNR (Analytical).

        Args:
            tobs_yr (float):           Observation duration in years.
            quick_analytical (bool):   Use fast geometric approximation.
            force_evolve (bool):       Force time-evolving SNR integration
                                       (only used when quick_analytical=False).
            evolve_threshold (float):  Auto-trigger evolving integration when
                                       tmerger < evolve_threshold * tobs. Default 10.0.
            n_segments (int):          Number of log-spaced segments for the
                                       evolving-SNR integral. Default 200.
            f_orb_max_Hz (float):      Orbital-frequency cutoff [Hz]; integration
                                       stops once f_orb exceeds this. Default 0.1.
            verbose (bool):            Per-call status print. Default True.
        """
        m1_s = self.m1 * m_sun
        m2_s = self.m2 * m_sun
        a_s = self.a * AU
        Dl_s = self.Dl * 1000.0 * pc
        tobs_s = tobs_yr * years

        snr = 0.0

        if quick_analytical:
            if self.a <= 0 or self.e >= 1.0:
                return 0.0

            used_tobs = tobs_s
            try:
                t_lower = PN_waveform.tmerger_lower(m1_s, m2_s, a_s, self.e)
                if t_lower <= tobs_s:
                    t_real = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, self.e)
                    if t_real <= tobs_s:
                        used_tobs = t_real
            except:
                pass

            rp_s = a_s * (1 - self.e)
            if rp_s > 0:
                term_f = (m1_s + m2_s) / (4 * pi * pi * np.power(rp_s, 3.0))
                f0max = 2 * np.sqrt(term_f)
                h0max = np.sqrt(32 / 5) * m1_s * m2_s / (Dl_s * a_s * (1 - self.e))
                Sn_val = PN_waveform.S_n_lisa(f0max)
                if Sn_val > 0:
                    snr = h0max / np.sqrt(Sn_val) * np.sqrt(
                        used_tobs * np.power(1 - self.e, 1.5)
                    )

            if verbose:
                print(f"[CompactBinary] SNR (Quick) = {snr:.4f}")
            self.extra['snr_analytical'] = snr
            return snr

        # ===== Full Harmonic Integration: decide static vs evolving =====
        try:
            t_merger_s = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, self.e)
        except Exception:
            t_merger_s = np.inf

        use_evolving = force_evolve or (t_merger_s < evolve_threshold * tobs_s)

        if not use_evolving:
            snr = PN_waveform.SNR(m1_s, m2_s, a_s, self.e, Dl_s, tobs_s)
            if verbose:
                print(f"[CompactBinary] SNR (Full, non-evolving) = {snr:.4f}  "
                      f"[tmerger/tobs = {t_merger_s / tobs_s:.2e}]")
            self.extra['snr_analytical'] = snr
            return snr

        # ---- Time-evolving SNR integration ----
        t_end_s = min(tobs_s, t_merger_s)
        if t_end_s <= 0:
            self.extra['snr_analytical'] = 0.0
            return 0.0

        #log_min = np.log10(max(t_end_s * 1e-6, 1.0))
        log_min = np.log10(t_end_s * 1e-6)
        log_max = np.log10(t_end_s)
        tau = np.logspace(log_max, log_min, n_segments + 1)
        t_grid = t_end_s - tau
        t_grid[0] = 0.0
        t_grid[-1] = t_end_s

        M_tot = m1_s + m2_s
        snr2_total = 0.0
        n_used = 0
        truncated_by_freq = False

        a_cur, e_cur, t_cur = a_s, self.e, 0.0

        for i in range(n_segments):
            t_next = t_grid[i + 1]
            dt = t_next - t_cur
            if dt <= 0:
                continue

            if a_cur > 0:
                f_orb_cur = np.sqrt(M_tot / (4 * pi * pi * a_cur ** 3))
            else:
                f_orb_cur = np.inf
            if f_orb_cur > f_orb_max_Hz:
                truncated_by_freq = True
                break

            try:
                a_mid, e_mid = PN_waveform.solve_ae_after_time(
                    m1_s, m2_s, a_cur, e_cur, dt / 2.0
                )
            except Exception:
                a_mid, e_mid = a_cur, e_cur
            if (a_mid <= 0) or (e_mid >= 1.0) or (not np.isfinite(a_mid)):
                truncated_by_freq = True
                break
            f_orb_mid = np.sqrt(M_tot / (4 * pi * pi * a_mid ** 3))
            if f_orb_mid > f_orb_max_Hz:
                truncated_by_freq = True
                break

            try:
                dsnr = PN_waveform.SNR(m1_s, m2_s, a_mid, e_mid, Dl_s, dt)
            except Exception:
                dsnr = 0.0
            if np.isfinite(dsnr) and dsnr > 0:
                snr2_total += dsnr * dsnr
                n_used += 1

            try:
                a_new, e_new = PN_waveform.solve_ae_after_time(
                    m1_s, m2_s, a_cur, e_cur, dt
                )
            except Exception:
                truncated_by_freq = True
                break
            if (a_new <= 0) or (e_new >= 1.0) or (not np.isfinite(a_new)):
                truncated_by_freq = True
                break
            a_cur, e_cur, t_cur = a_new, e_new, t_next

        snr = np.sqrt(snr2_total)

        if verbose:
            reason = "forced" if force_evolve else \
                f"tmerger/tobs={t_merger_s / tobs_s:.2e} < {evolve_threshold}"
            trunc_msg = " [truncated at f_orb cutoff]" if truncated_by_freq else ""
            print(f"[CompactBinary] SNR (Full, evolving) = {snr:.4f}  "
                  f"[{reason}, n_seg_used={n_used}/{n_segments}, "
                  f"t_end={t_end_s / years:.3e} yr]{trunc_msg}")

        self.extra['snr_analytical'] = snr
        return snr

    @mute_if_global_verbose_false
    def compute_merger_time(self, verbose=True):
        """
        Compute time to merger in years.
        """
        m1_s = self.m1 * m_sun
        m2_s = self.m2 * m_sun
        a_s = self.a * AU

        t_sec = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, self.e)
        t_yr = t_sec / years

        if verbose:
            print(f"[CompactBinary] Merger Time = {t_yr:.4e} years")

        self.extra['merger_time_yr'] = t_yr
        return t_yr

    @mute_if_global_verbose_false
    def evolve_orbit(self, delta_t_yr: float, update_self=False, verbose=True):
        """
        Evolve orbit forward in time.

        Args:
            delta_t_yr (float): Time to evolve in years.
            update_self (bool): If True, updates the object's 'a' and 'e' attributes.
        Returns:
            (a_new_au, e_new): The evolved parameters.
        """
        m1_s = self.m1 * m_sun
        m2_s = self.m2 * m_sun
        a_s = self.a * AU
        delta_t_s = delta_t_yr * years

        a_new_geo, e_new = PN_waveform.solve_ae_after_time(m1_s, m2_s, a_s, self.e, delta_t_s)
        a_new_au = a_new_geo / AU

        if verbose:
            print(
                f"[CompactBinary] Evolved ({delta_t_yr} yr): a {self.a:.2e}->{a_new_au:.2e} AU, e {self.e:.4f}->{e_new:.4f}")

        if update_self:
            self.a = a_new_au
            self.e = e_new

        return a_new_au, e_new

    @mute_if_global_verbose_false
    def compute_characteristic_strain(self, tobs_yr, plot=True):
        """
        Compute characteristic strain (h_c).
        """
        m1_s = self.m1 * m_sun
        m2_s = self.m2 * m_sun
        a_s = self.a * AU
        Dl_s = self.Dl * 1e3 * pc
        tobs_s = tobs_yr * years

        print(f"[CompactBinary] Computing h_c for {self.label}...")
        res = hc_cal.calculate_single_system(
            m1=m1_s, m2=m2_s, a=a_s, e=self.e, Dl=Dl_s, tobs=tobs_s
        )
        if plot:
            hc_cal.plot_single_system_results(res)
        return res

    @mute_if_global_verbose_false
    def compute_characteristic_strain_evolve(self, tobs_yr=4.0, target_n_points=100,
                                             all_harmonics=False, plot=True, verbose=True):
        """
        [NEW] Compute the time-evolving harmonic spectrum and integrated characteristic strain.
        Uses precise orbital evolution (Peters 1964).

        Parameters:
            tobs_yr: Observation duration in years
            target_n_points: Max number of harmonics per snapshot (for sparse tracking)
            all_harmonics: If True, compute tracks for ALL harmonics in range (can be slow)
            plot: Whether to plot results
            verbose: Print status
        """
        if verbose:
            print(f"[CompactBinary] Computing evolving strain for {self.label}...")
            print(f"                m={self.m1}+{self.m2}, a={self.a:.4e} AU, e={self.e:.4f}, Tobs={tobs_yr} yr")

        results = hc_cal.calculate_evolving_system(
            m1=self.m1,
            m2=self.m2,
            a=self.a,
            e=self.e,
            Dl=self.Dl,
            tobs_years=tobs_yr,
            target_n_points=target_n_points,
            all_harmonics=all_harmonics,  # <--- Added
            plot=plot,
            verbose=verbose
        )
        return results
    @mute_if_global_verbose_false
    def compute_fpeak(self, verbose=True):
        """
        Calculate GW Peak Frequency [Hz] using Wen (2003) approximation.
        Formula: f_peak ~ f_orb * (1+e)^1.1954 / (1-e)^1.5
        """
        if self.a <= 0: return 0.0

        # Constants in SI
        G_si = sciconsts.G
        # Solar mass in kg
        M_sun_kg = 1.9884e30

        M_total_si = (self.m1 + self.m2) * M_sun_kg
        a_m = self.a * sciconsts.au

        # f_orb in Hz
        f_orb = (1.0 / (2 * np.pi)) * np.sqrt(G_si * M_total_si / (a_m ** 3))

        e = self.e
        if e < 0.0: e = 0.0
        if e >= 1.0: e = 0.999999

        factor = np.power(1 + e, 1.1954) / np.power(1 - e, 1.5)
        f_peak = f_orb * factor

        if verbose:
            print(f"[CompactBinary] f_peak = {f_peak:.4e} Hz (a={self.a:.2f} AU, e={e:.4f})")

        # Store in extra
        self.extra['f_peak_Hz'] = f_peak
        return f_peak

    @classmethod
    def from_list(cls, data_list: list, schema: str, aux_params: dict = None):
        """
        Factory method to create CompactBinary from list data.
        """
        if aux_params is None: aux_params = {}

        try:
            if schema == 'snapshot_std':
                # Format: [Label(0), Dist(1), SMA(2), Ecc(3), M1(4), M2(5), SNR(6)]
                return cls(
                    label=str(data_list[0]),
                    Dl=float(data_list[1]),
                    a=float(data_list[2]),
                    e=float(data_list[3]),
                    m1=float(data_list[4]),
                    m2=float(data_list[5]),
                    extra={'snr': float(data_list[6])}
                )

            elif schema == 'gn_prog':
                # Format: [m1, m2, a_init, e_init, i, a_outer, a_fin, e_fin, lifetime]
                # GN systems are at Galactic Center ~ 8.0 kpc
                return cls(
                    label="GN_Progenitor",
                    m1=float(data_list[0]),
                    m2=float(data_list[1]),
                    a=float(data_list[2]),  # Initial SMA
                    e=float(data_list[3]),  # Initial Ecc
                    Dl=8.0,
                    extra={
                        'inclination': data_list[4],
                        'a_outer': data_list[5],
                        'a_final': data_list[6],
                        'e_final': data_list[7],
                        'lifetime_yr': data_list[8]
                    }
                )

            elif schema == 'field_prog':
                # Format: [acur, e_init, e_final, Dl, rate, lifetime, tau]
                # Masses provided via aux_params (mandatory now)
                m1_val = aux_params.get('m1', 10.0)  # Default safeguard if not passed
                m2_val = aux_params.get('m2', 10.0)

                return cls(
                    label="Field_Progenitor",
                    m1=float(m1_val),
                    m2=float(m2_val),
                    a=float(data_list[0]),  # Initial SMA
                    e=float(data_list[1]),  # Initial Ecc
                    Dl=float(data_list[3]),
                    extra={
                        'e_final_LIGO': data_list[2],
                        'merger_rate': data_list[4],
                        'lifetime_yr': data_list[5],
                        'tau_relax': data_list[6]
                    }
                )
            else:
                raise ValueError(f"Unknown schema: {schema}")
        except Exception as err:
            raise ValueError(f"[CompactBinary] Parsing error for schema '{schema}': {err}\nData: {data_list}")

    def to_list(self, schema='snapshot_std'):
        """Convert object back to list format."""
        if schema == 'snapshot_std':
            snr = self.extra.get('snr', 0.0)
            return [self.label, self.Dl, self.a, self.e, self.m1, self.m2, snr]
        return []
# ==============================================================================
# 主接口类: leap
# ==============================================================================
class LISAeccentric:
    """
    leap Unified Interface.
    Modules: GN, GC, Field, Waveform.
    """

    def __init__(self):
        self.GN = self._GN_Handler()
        self.GC = self._GC_Handler()
        self.Field = self._Field_Handler()
        self.Waveform = self._Waveform_Handler()
        self.Noise = self._Noise_Handler()

    # ==========================================================================
    # MODULE 1: Galactic Nucleus (GN)
    # ==========================================================================
    class _GN_Handler:
            @mute_if_global_verbose_false
            def sample_eccentricities(self, n_samples=5000, max_bh_mass=100, plot=True):
                """Feature 1: Randomly sample N merger eccentricities (LIGO Band 10Hz)."""
                print(f"\n[GN] Sampling {n_samples} merger eccentricities (max_mass={max_bh_mass})...")
                e_samples = GN_BBH.generate_random_merger_eccentricities(n=n_samples, max_bh_mass=max_bh_mass)
                print(f'Sample Mean e (at 10Hz): {np.mean(e_samples):.4e}')
                if plot:
                    GN_BBH.plot_ecc_cdf_log(e_list=e_samples)
                return e_samples

            @mute_if_global_verbose_false
            def get_progenitor(self, n_inspect=3) -> List[CompactBinary]:
                """Feature 2: Get Progenitor Population (Initial States)."""
                print(f"\n[GN] Getting {n_inspect} Progenitor Systems...")

                # GN_BBH.get_random_merger_systems
                # [m1, m2, a1, e1, e2, i_init, a2, afin, efin, t_final]
                raw_data = GN_BBH.get_random_merger_systems(n=n_inspect)

                objs = []
                for row in raw_data:

                    obj = CompactBinary(
                        m1=float(row[0]),
                        m2=float(row[1]),
                        a=float(row[2]),
                        e=float(row[3]),
                        Dl=8.0,
                        label="GN_Progenitor"
                    )


                    obj.extra = {
                        'e2_init': float(row[4]),
                        'i_init_rad': float(row[5])/180*pi,
                        'a2_init': float(row[6]),
                        'a_final': float(row[7]),
                        'e_final': float(row[8]),
                        'lifetime_yr': float(row[9])
                    }

                    objs.append(obj)
                    print(obj)
                return objs

            @mute_if_global_verbose_false
            def get_snapshot(self, rate_gn=2.0, age_ync=6.0e6, n_ync_sys=100, max_bh_mass=100, plot=True) -> List[
                CompactBinary]:
                """Feature 3: Snapshot Generation (LISA Band / Current State)."""
                print(f"\n[GN] Generating Snapshot: Rate={rate_gn}/Myr, YNC Age={age_ync / 1e6} Myr")

                # GN_BBH.generate_snapshot_population
                # [label, dist, a, e, i, m1, m2, snr]
                raw_data = GN_BBH.generate_snapshot_population(
                    Gamma_rep=rate_gn, ync_age=age_ync, ync_count=n_ync_sys, max_bh_mass=max_bh_mass
                )


                raw_data.sort(key=lambda x: x[7], reverse=True)

                objs = []
                for row in raw_data:
                    obj = CompactBinary(
                        label=str(row[0]),
                        Dl=float(row[1]),
                        a=float(row[2]),
                        e=float(row[3]),
                        m1=float(row[5]),
                        m2=float(row[6])
                    )

                    obj.extra = {
                        'inclination_rad': float(row[4])/180*pi,
                        'snr': float(row[7])
                    }

                    objs.append(obj)

                print(f"Altogether {len(objs)} systems survived.")
                if plot:
                    GN_BBH.plot_snapshot_population(raw_data, title="Simulated MW Galactic Nucleus BBH Population")
                return objs

            @mute_if_global_verbose_false
            def calculate_fpeak_frequency(self, m1=None, m2=None, a_au=None, e=None, system=None):
                """
                Extra Feature: Calculate GW Peak Frequency.
                Supports parameter input OR CompactBinary object input.
                """
                if system is not None:
                    m1, m2, a_au, e = system.m1, system.m2, system.a, system.e
                elif isinstance(m1, CompactBinary):
                    system = m1
                    m1, m2, a_au, e = system.m1, system.m2, system.a, system.e

                if a_au <= 0: return 0.0

                G_si = 6.674e-11
                M_total_si = (m1 + m2) * 1.989e30
                a_m = a_au * 1.496e11
                f_orb = (1.0 / (2 * np.pi)) * np.sqrt(G_si * M_total_si / (a_m ** 3))

                if e < 0.0: e = 0.0
                if e >= 1.0: e = 0.999999
                factor = np.power(1 + e, 1.1954) / np.power(1 - e, 1.5)
                f_peak = f_orb * factor

                print(f"[GN Util] f_peak = {f_peak:.4e} Hz (a={a_au:.2f} AU, e={e:.4f})")
                return f_peak

            # ==========================================================================
            # MODULE 2: Globular Clusters (GC)
            # ==========================================================================

    class _GC_Handler:
        @mute_if_global_verbose_false
        def sample_eccentricities(self, n=5000, channel_name='Incluster', plot=True):
            """Feature 1: Sample eccentricities for GC BBH Mergers (LIGO band)."""
            print(f"\n[GC] Sampling {n} eccentricities (Channel: {channel_name})...")
            e_samples = GC_BBH.generate_ecc_samples_10Hz(channel_name=channel_name, size=n)
            if plot:
                GC_BBH.plot_ecc_cdf(e_samples, label=f"GC {channel_name}")
            return e_samples

        @mute_if_global_verbose_false
        def get_snapshot(self, mode='10_realizations', channel='all', n_random=500, plot=True) -> List[CompactBinary]:
            """Feature 2: Get BBH parameters from GC snapshots.
            channel options: 'all', 'incluster', 'ejected'."""
            print(f"\n[GC] Getting Snapshot (Mode: {mode}, Channel: {channel})...")
            if mode == '10_realizations':
                raw_data = GC_BBH.get_full_10_realizations(channel=channel)
                title = f"BBHs in MW GCs (10 Realizations, Channel: {channel})"
            elif mode == 'single':
                raw_data = GC_BBH.get_single_mw_realization(channel=channel)
                title = f"Single MW Realization (Channel: {channel})"
            elif mode == 'random':
                raw_data = GC_BBH.get_random_systems(n_random, channel=channel)
                title = f"Randomly Selected {n_random} Systems (Channel: {channel})"
            else:
                raise ValueError("Mode must be '10_realizations', 'single', or 'random'.")

            objs = [CompactBinary.from_list(row, schema='snapshot_std') for row in raw_data]
            print(f"Total systems retrieved: {len(objs)}")
            if plot:
                GC_BBH.plot_mw_gc_bbh_snapshot(raw_data, title=title)
            return objs

    # ==========================================================================
    # MODULE 3: Field (Fly-by)
    # ==========================================================================
    class _Field_Handler:
        @mute_if_global_verbose_false
        def run_simulation(self, galaxy_type='MW',
                           m1=10, m2=10, mp=0.6, fbh=7.5e-4, fgw=10,
                           formation_mod='starburst', arange_log=[2, 4.5],
                           n_sim_samples=200000, target_N=100000,
                           n0=0.1, rsun=8e3, Rl=2.6e3, h=1e3, sigmav=50e3,
                           age_mw=10e9, rrange_mw=[0.5, 15], blocknum_mw=29,
                           distance_Mpc=16.8, M_gal=1.0e12, Re=8.0e3,
                           age_ell=13e9, rrange_ell=[0.05, 100], blocknum_ell=60,
                           ell_n_sim=100000, ell_target_N=50000):
            """Section 0: Initialize/Re-run Simulation."""
            print(f"\n[Field] Running Simulation for {galaxy_type}...")
            if galaxy_type == 'MW':
                Field_BBH.simulate_and_save_default_population(
                    n_sim_samples=n_sim_samples, target_N=target_N,
                    m1=m1 * Field_BBH.m_sun, m2=m2 * Field_BBH.m_sun, mp=mp * Field_BBH.m_sun,
                    fgw=fgw, n0=n0 / (Field_BBH.pc ** 3), rsun=rsun * Field_BBH.pc,
                    Rl=Rl * Field_BBH.pc, h=h * Field_BBH.pc, sigmav=sigmav / sciconsts.c, fbh=fbh,
                    formation_mod=formation_mod, age=age_mw * Field_BBH.years,
                    rrange_kpc=rrange_mw, arange_log=arange_log, blocknum=blocknum_mw
                )
            elif galaxy_type == 'Elliptical':
                Field_BBH_Elliptical.simulate_and_save_default_population(
                    distance_Mpc=distance_Mpc, M_gal=M_gal * Field_BBH.m_sun, Re=Re * Field_BBH.pc,
                    m1=m1 * Field_BBH.m_sun, m2=m2 * Field_BBH.m_sun, mp=mp * Field_BBH.m_sun,
                    fbh=fbh, fgw=fgw, arange_log=arange_log,
                    formation_mod=formation_mod, age=age_ell * Field_BBH.years,
                    rrange_kpc=rrange_ell, blocknum=blocknum_ell,
                    n_sim_samples=ell_n_sim, target_N=ell_target_N
                )
            else:
                raise ValueError("galaxy_type must be 'MW' or 'Elliptical'")
            print("Simulation Data Saved.")

        @mute_if_global_verbose_false
        def sample_eccentricities(self, n=5000, galaxy_type='MW', plot=True):
            """Feature 1: Merger Eccentricity Sampling (LIGO Band)."""
            print(f"\n[Field] Sampling {n} eccentricities ({galaxy_type})...")
            module = Field_BBH_Elliptical if galaxy_type == 'Elliptical' else Field_BBH
            e_samples = module.generate_eccentricity_samples(size=n)
            print(f'Sample Mean e: {np.mean(e_samples):.4e}')
            if plot:
                module.plot_eccentricity_cdf(e_samples, label=f"Field {galaxy_type}")
            return e_samples

        @mute_if_global_verbose_false
        def get_progenitor(self, galaxy_type='MW', plot=True) -> List[CompactBinary]:
            """Feature 2: Get Progenitor Population (Initial States)."""
            print(f"\n[Field] Getting Progenitor Population ({galaxy_type})...")
            module = Field_BBH_Elliptical if galaxy_type == 'Elliptical' else Field_BBH

            # Retrieve masses from model metadata
            model = module._get_model()
            try:
                conversion_factor = Field_BBH.m_sun
                m1_solar = model.m1 / conversion_factor
                m2_solar = model.m2 / conversion_factor
            except AttributeError:
                m1_solar, m2_solar = 10.0, 10.0
                print("Warning: Could not read masses from simulation file. Using default 10.0 Msun.")

            raw_data = module.get_merger_progenitor_population()

            objs = [
                CompactBinary.from_list(
                    row,
                    schema='field_prog',
                    aux_params={'m1': m1_solar, 'm2': m2_solar}
                )
                for row in raw_data
            ]
            print(f"Total progenitors in library: {len(objs)} (Masses: {m1_solar:.1f}+{m2_solar:.1f})")

            if plot:
                module.plot_progenitor_sma_distribution(bins=50)
                module.plot_lifetime_cdf()
            return objs

        @mute_if_global_verbose_false
        def get_snapshot(self, mode='single', n_realizations=10, n_systems=500,
                         t_obs=10.0, t_window_Gyr=10.0, galaxy_type='MW', plot=True) -> List[CompactBinary]:
            """Feature 3: Snapshot Generation (LISA Band)."""
            print(f"\n[Field] Generating Snapshot ({galaxy_type}, Mode={mode})...")
            module = Field_BBH_Elliptical if galaxy_type == 'Elliptical' else Field_BBH

            if mode == 'single':
                if galaxy_type == 'Elliptical':
                    raw_data = module.get_single_realization(t_window_Gyr=t_window_Gyr, tobs_yr=t_obs)
                else:
                    raw_data = module.get_single_mw_realization(t_window_Gyr=t_window_Gyr, tobs_yr=t_obs)
                title = f"Snapshot ({galaxy_type} Single Realization)"
            elif mode == 'multi' and galaxy_type == 'MW':
                raw_data = module.get_multi_mw_realizations(n_realizations=n_realizations, t_window_Gyr=t_window_Gyr,
                                                            tobs_yr=t_obs)
                title = f"Snapshot ({galaxy_type} {n_realizations} Realizations)"
            elif mode == 'forced':
                raw_data = module.get_random_systems(n_systems=n_systems, t_window_Gyr=t_window_Gyr, tobs_yr=t_obs)
                title = f"Snapshot ({galaxy_type} Random {n_systems})"
            else:
                print("Invalid Mode.")
                return []

            objs = [CompactBinary.from_list(row, schema='snapshot_std') for row in raw_data]
            print(f"Systems found: {len(objs)}")
            if plot and len(raw_data) > 0:
                if galaxy_type == 'Elliptical':
                    module.plot_snapshot(raw_data, title=title, tobs_yr=t_obs)
                else:
                    module.plot_mw_field_bbh_snapshot(raw_data, title=title, tobs_yr=t_obs)
            return objs

    # ==========================================================================
    # MODULE 4: Waveform & Analysis
    # ==========================================================================
    class _Waveform_Handler:
        @mute_if_global_verbose_false
        def compute_waveform(self, m1_msun, m2_msun, a_au, e, Dl_kpc, tobs_yr,
                                    input_mode='a_au',  # <--- 默认为半长轴模式
                                    initial_orbital_phase=0,
                                    theta=np.pi / 4, phi=np.pi / 4,
                                    PN_orbit=3, PN_reaction=2,
                                    ts=None, points_per_peak=50,
                                    max_memory_GB=16.0, verbose=True, plot=True):
            """
            Generate GW waveform with flexible input modes.

            :param m1_msun:   Mass 1 [Solar Mass]
            :param m2_msun:   Mass 2 [Solar Mass]
            :param a_au:      Meaning fepends on input modes:
                              - input_mode='a_au':        Semi-major axis [AU] (default)
                              - input_mode='forb_Hz':     (radial) orbital frequency [Hz] (note here the name is a_au, but use freq)
                              - input_mode='fangular_Hz': angular orbital frequency [Hz]
            :param e:         Eccentricity
            :param Dl_kpc:    Luminosity Distance [kpc]
            :param tobs_yr:   Observation time [years]
            :param input_mode: Mode selector ('a_au', 'forb_Hz', 'fangular_Hz')

            Sampling parameters:
            :param ts:              Time step [seconds].
            :param points_per_peak: Sampling points per orbital peak.
            """
            # (G=c=1 units / seconds)
            m1 = m1_msun * m_sun
            m2 = m2_msun * m_sun
            M_total = m1 + m2
            Dl = Dl_kpc * 1e3 * pc
            tobs = tobs_yr * years

            label = f"Mode_{input_mode}"

            f_orb = 0.0
            a_au_display = 0.0

            if input_mode == 'a_au':
                real_a_au = a_au
                a = real_a_au * AU
                if a > 0:
                    f_orb = np.sqrt(M_total / (4 * pi ** 2 * np.power(a, 3.0)))
                a_au_display = real_a_au

            elif input_mode == 'forb_Hz':
                # a_au is actually frequency [Hz]
                f_orb = a_au

                if f_orb > 0:
                    a_s = np.power(M_total / ((2 * np.pi * f_orb) ** 2), 1.0 / 3.0)
                    a_au_display = a_s / AU

            elif input_mode == 'fangular_Hz':
                target_f0 = a_au

                def get_a0_from_f00(f_val, M):
                    return np.power(M / ((2 * np.pi * f_val) ** 2), 1.0 / 3.0)

                def deltafvalue(a, e_in, M):
                    n = np.power(a, -3 / 2) * np.sqrt(M)
                    Porb = 2 * pi / n
                    # Hinder+10
                    return 6 * np.power(2 * pi, 2 / 3) / (1 - e_in * e_in) * np.power(M, 2 / 3) * np.power(Porb, -5 / 3)

                def frequency_residual(f00_guess):
                    if f00_guess <= 0: return 1e6  # 保护
                    a_guess = get_a0_from_f00(f00_guess, M_total)
                    df = deltafvalue(a_guess, e, M_total)
                    f0_calc = f00_guess + df / 2.0
                    return f0_calc - target_f0

                try:
                    # Newton f_orb (initial guess = target_f0)
                    f_orb = newton(frequency_residual, x0=target_f0, tol=1e-7, maxiter=50)

                    a_s = get_a0_from_f00(f_orb, M_total)
                    a_au_display = a_s / AU
                except Exception as err:
                    print(f"[Error] Failed to solve f_orb from fangular_Hz: {err}")
                    return None
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}. Use 'a_au', 'forb_Hz', or 'fangular_Hz'.")

            if verbose:
                if ts is not None:
                    sampling_mode = f"Fixed TimeStep (ts={ts} s)"
                else:
                    sampling_mode = f"Adaptive Sampling (N={points_per_peak}/peak)"

                print(f"\n[Waveform] Generating Waveform...")
                print(f"           Mode:   {input_mode}")
                print(f"           Params: m={m1_msun}+{m2_msun} Msun, e={e:.3f}")
                print(f"           Input:  {a_au:.4e} ({input_mode})")
                print(f"           Derived: f_orb={f_orb:.4e} Hz, a~={a_au_display:.4f} AU")

            wf = PN_waveform.eccGW_waveform(
                f_orb, e, tobs, m1, m2, theta, phi, Dl,
                l0=initial_orbital_phase,
                ts=ts,
                N=points_per_peak,
                PN_orbit=PN_orbit, PN_reaction=PN_reaction,
                max_memory_GB=max_memory_GB, verbose=verbose
            )

            if plot:
                plt.figure(figsize=(8, 6), dpi=100)
                plt.plot(wf[0], wf[1], color='BLUE', label='h_plus')
                plt.xlabel("t [s]", fontsize=14)
                plt.ylabel("hplus", fontsize=14)
                plt.title(f"GW Waveform: {label}")
                plt.legend()
                plt.show()
            return wf


        def compute_orbit_evolution(self, m1_msun, m2_msun, a_au, e, tobs_yr,
                                    input_mode='a_au',
                                    initial_orbital_phase=0.0,
                                    PN_orbit=3, PN_reaction=2,
                                    ts=None, points_per_peak=50,
                                    max_memory_GB=16.0,
                                    E_method='invert',
                                    compute_a=False,
                                    compute_pe=False,
                                    e_zero_point='add_one',
                                    verbose=True, plot=True):
            """Evolve the orbit (no waveform) and return a dict of vectors.

            Mirrors compute_waveform's input handling (a_au / forb_Hz / fangular_Hz).

            Returns dict with keys:
                t, x, e_t, l, E, Lz                         (always)
                a_au                                        (if compute_a)
                p_pn, e_pn   (PN route: p=(1-e_t^2)/x, e=e_t)        (if compute_pe)
                p_geo, e_geo (geodesic route, E_bar=1+xi)            (if compute_pe)
            """
            m1 = m1_msun * m_sun
            m2 = m2_msun * m_sun
            M_total = m1 + m2
            tobs = tobs_yr * years

            # -- resolve input_mode -> f_orb (same logic as compute_waveform) --------
            f_orb = 0.0
            a_au_display = 0.0
            if input_mode == 'a_au':
                a = a_au * AU
                if a > 0:
                    f_orb = np.sqrt(M_total / (4 * pi ** 2 * a ** 3))
                a_au_display = a_au
            elif input_mode == 'forb_Hz':
                f_orb = a_au
                if f_orb > 0:
                    a_s = np.power(M_total / ((2 * pi * f_orb) ** 2), 1.0 / 3.0)
                    a_au_display = a_s / AU
            elif input_mode == 'fangular_Hz':
                from scipy.optimize import newton
                target_f0 = a_au

                def get_a0(f_val, M):
                    return np.power(M / ((2 * pi * f_val) ** 2), 1.0 / 3.0)

                def deltafvalue(a_, e_in, M):
                    n = np.power(a_, -3 / 2) * np.sqrt(M)
                    Porb = 2 * pi / n
                    return 6 * np.power(2 * pi, 2 / 3) / (1 - e_in * e_in) \
                        * np.power(M, 2 / 3) * np.power(Porb, -5 / 3)

                def resid(f00g):
                    if f00g <= 0:
                        return 1e6
                    ag = get_a0(f00g, M_total)
                    return f00g + deltafvalue(ag, e, M_total) / 2.0 - target_f0

                f_orb = newton(resid, x0=target_f0, tol=1e-7, maxiter=50)
                a_au_display = get_a0(f_orb, M_total) / AU
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")

            if verbose:
                smode = (f"Fixed ts={ts}s" if ts is not None
                         else f"Adaptive N={points_per_peak}/peak")
                print(f"\n[OrbitEvolution] {input_mode} | m={m1_msun}+{m2_msun} Msun, "
                      f"e={e:.3f}")
                print(f"           f_orb={f_orb:.4e} Hz, a~={a_au_display:.4f} AU, "
                      f"{smode}, E_method={E_method}")

            # -- evolve --------------------------------------------------------------
            t, xvec, evec, lvec, Evec, Lzvec = PN_waveform.ecc_orbit_evolution(
                f_orb, e, tobs, m1, m2,
                l0=initial_orbital_phase, ts=ts,
                PN_orbit=PN_orbit, PN_reaction=PN_reaction,
                N=points_per_peak, max_memory_GB=max_memory_GB,
                verbose=verbose, E_method=E_method,
            )

            out = dict(t=t, x=xvec, e_t=evec, l=lvec, E=Evec, Lz=Lzvec)

            if len(t) == 0:
                if verbose:
                    print("  [empty evolution]")
                return out

            # -- semi-major axis (approximate, leading PN: a = M/x) ------------------
            if compute_a:
                GM_over_c2_AU = (M_total) / AU  # M_total is in seconds == GM/c^3;
                # a_geom [units of M] = 1/x ; a_geom*M (seconds) = M/x; /AU -> AU.
                # In geometrized seconds, length = time, so a[s] = M_total / x.
                a_sec = M_total / xvec
                out['a_au'] = a_sec / AU
                if verbose:
                    print(f"  a_au (approx, a=M/x): "
                          f"[{out['a_au'].min():.4e}, {out['a_au'].max():.4e}] AU")

            # -- EMRI (p, e): output BOTH routes for comparison ----------------------
            if compute_pe:
                # ---- PN self-consistent route: p = a_r(1-e_t^2), e = e_t -----------
                # leading-order a_r = 1/x (geom, M=1)  ->  p_pn = (1-e_t^2)/x
                p_pn = (1.0 - evec ** 2) / xvec
                e_pn = evec.copy()
                out['p_pn'] = p_pn
                out['e_pn'] = e_pn

                # ---- geodesic route: invert Schwarzschild (E_bar, Lz) -> (p, e) ----
                # zero point: PN binding energy xi -> specific energy E_bar = 1 + xi
                if e_zero_point == 'add_one':
                    Ebar = 1.0 + Evec
                elif e_zero_point == 'as_is':
                    Ebar = Evec
                else:
                    raise ValueError("e_zero_point must be 'add_one' or 'as_is'")
                p_seed0 = (1.0 - evec[0] ** 2) / xvec[0]  # Newtonian seed
                e_seed0 = evec[0]
                p_geo, e_geo, res = _invert_geodesic_pe(
                    np.ascontiguousarray(Ebar),
                    np.ascontiguousarray(Lzvec),
                    float(p_seed0), float(e_seed0))
                out['p_geo'] = p_geo
                out['e_geo'] = e_geo

                if verbose:
                    print(f"  [PN route]       p=(1-e_t^2)/x, e=e_t : "
                          f"p:[{p_pn.min():.4g},{p_pn.max():.4g}]  "
                          f"e:[{e_pn.min():.4g},{e_pn.max():.4g}]")
                    print(f"  [geodesic route] (E_bar=1+xi) max res={res:.2e}: "
                          f"p:[{p_geo.min():.4g},{p_geo.max():.4g}]  "
                          f"e:[{e_geo.min():.4g},{e_geo.max():.4g}]")

            # -- optional plot -------------------------------------------------------
            if plot:
                import matplotlib.pyplot as plt
                npan = 2 + int(compute_a) + (1 if compute_pe else 0)
                fig, axes = plt.subplots(npan, 1, figsize=(9, 2.2 * npan),
                                         sharex=True)
                if npan == 1:
                    axes = [axes]
                k = 0
                axes[k].plot(t, xvec, color='#1f4e79');
                axes[k].set_ylabel('x');
                k += 1
                axes[k].plot(t, evec, color='#a6262e');
                axes[k].set_ylabel('e_t');
                k += 1
                if compute_a:
                    axes[k].plot(t, out['a_au'], color='#2a7f3f')
                    axes[k].set_ylabel('a [AU]');
                    k += 1
                if compute_pe:
                    axes[k].plot(t, out['p_geo'], color='#7a3aa1', label='p (geodesic)')
                    axes[k].plot(t, out['p_pn'], color='#d08a00', ls='--', label='p (PN)')
                    axes[k].set_ylabel('p');
                    axes[k].legend(loc='upper right');
                    k += 1
                axes[-1].set_xlabel('t [s]')
                for ax in axes:
                    ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()

            return out
        @mute_if_global_verbose_false
        def compute_LISA_response(self, dt_sample_sec, hplus, hcross,
                                  theta_sky=np.pi / 4, phi_sky=np.pi / 4, psi_sky=np.pi / 4,
                                  timeshift_sec=0.0,
                                  kappa=0.0, lamb=0.0, mode='interp', plot=True):
            """
            Compute LISA detector response.

            :param dt_sample_sec: Sampling interval [seconds] (Scalar).
                                  Must match the resolution of hplus/hcross.
            :param timeshift_sec: Time shift [seconds]. Shifts the signal in time domain.
                                  (e.g., propagation delay).
            """
            n_points = len(hplus)
            if len(hcross) != n_points:
                raise ValueError(f"[Error] Waveform length mismatch: hplus={n_points}, hcross={len(hcross)}")

            print(f"[Waveform] Computing LISA Response (dt={dt_sample_sec:.4e} s, shift={timeshift_sec} s)...")

            # (Standard Time Axis starting from 0)
            # timelist = 0, dt, 2dt, ...
            timelist = np.arange(n_points) * dt_sample_sec

            # pass timeshift_sec to basic functions
            response = PN_waveform.compute_LISA_response(
                timelist, hplus, hcross, theta_sky, phi_sky, psi_sky,
                t0=timeshift_sec,  # <--- Mapping here
                kappa=kappa, lamb=lamb, mode=mode
            )

            if plot:
                plt.figure(figsize=(8, 6), dpi=100)
                plt.plot(response[0], response[1], color='BLUE', label='Response')
                plt.xlabel("t [s]", fontsize=14)
                plt.ylabel("Detector Response")
                plt.title(f"LISA Response (Shift={timeshift_sec}s)")
                plt.legend()
                plt.show()
            return response

        @mute_if_global_verbose_false
        def compute_inner_product(self, dt_sample_sec, h1, h2, phase_difference=0):
            """
            Calculate inner product of two waveforms.

            :param dt_sample_sec: Sampling interval [seconds] (Scalar).
            """
            if len(h1) != len(h2):
                raise ValueError(f"[Error] Inner product length mismatch: {len(h1)} vs {len(h2)}")

            if len(h1) < 2: return 0.0

            # fs [Hz]
            fs = 1.0 / dt_sample_sec

            val = PN_waveform.inner_product(fs, h1, h2, phase_difference)
            print(f"[Analysis] Inner Product = {val:.4e}")
            return val

        @mute_if_global_verbose_false
        def compute_snr_numerical(self, dt_sample_sec, strainlist):
            """
            Estimate SNR using Numerical Inner Product.

            :param dt_sample_sec: Sampling interval [seconds] (Scalar).
            """
            val = self.compute_inner_product(dt_sample_sec, strainlist, strainlist)
            snr_num = np.sqrt(val)
            print(f"[Analysis] SNR_numerical = {snr_num:.4f}")
            return snr_num

        @mute_if_global_verbose_false
        def compute_snr_analytical0(self, m1_msun, m2_msun, a_au, e, Dl_kpc, tobs_yr, quick_analytical=False):
            """
            Estimate Sky-Averaged SNR (Analytical).
            m1_msun, m2_msun, a_au, e, Dl_kpc, tobs_yr
            quick_analytical (bool) - if True, use quick analytical method
            """
            # (G=c=1)
            m1_s = m1_msun * m_sun
            m2_s = m2_msun * m_sun
            a_s = a_au * AU
            Dl_s = Dl_kpc * 1000.0 * pc
            tobs_s = tobs_yr * years

            if quick_analytical:
                # === Quick Analytical (Geometric Approximation) ===
                if a_au <= 0 or e >= 1.0:
                    return 0.0

                used_tobs = tobs_s

                try:
                    # (Fast Check)
                    t_lower = PN_waveform.tmerger_lower(m1_s, m2_s, a_s, e)
                    #print('!',t_lower)

                    if t_lower <= tobs_s:
                        t_real = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, e)

                        if t_real <= tobs_s:
                            print(
                                f"[Warning] System evolves too fast! tmerger ({t_real:.2e} s) < tobs ({tobs_s:.2e} s).")
                            print(f"Approximation inaccurate. Adjusting tobs to : {t_real:.2e} s")
                            used_tobs = t_real

                except AttributeError:
                    try:
                        t_real = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, e)
                        if t_real <= tobs_s:
                            used_tobs = t_real
                    except Exception:
                        pass
                except Exception as e:
                    pass

                rp_s = a_s * (1 - e)
                if rp_s <= 0: return 0.0

                term_f = (m1_s + m2_s) / (4 * pi * pi * np.power(rp_s, 3.0))
                f0max = 2 * np.sqrt(term_f)

                h0max = np.sqrt(32 / 5) * m1_s * m2_s / (Dl_s * a_s * (1 - e))

                Sn_val = PN_waveform.S_n_lisa(f0max)

                if Sn_val <= 0:
                    snr = 0.0
                else:
                    sqrtsnf = np.sqrt(Sn_val)
                    snr = h0max / sqrtsnf * np.sqrt(used_tobs * np.power(1 - e, 1.5))

                print(f"[Analysis] SNR_analytical (Quick) = {snr:.4f}")
                return snr
            else:
                # === Full Numerical Integration ===
                snr = PN_waveform.SNR(m1_s, m2_s, a_s, e, Dl_s, tobs_s)
                print(f"[Analysis] SNR_analytical = {snr:.4f}")
                return snr

        @mute_if_global_verbose_false
        def compute_snr_analytical(self, m1_msun, m2_msun, a_au, e, Dl_kpc, tobs_yr,
                                   quick_analytical=False, verbose=True,
                                   force_evolve=False, evolve_threshold=10.0,
                                   n_segments=100, f_orb_max_Hz=0.1):
            """
            Estimate Sky-Averaged SNR (Analytical).

            Args:
                m1_msun, m2_msun (float): Component masses [Msun].
                a_au (float):              Semi-major axis [AU].
                e (float):                 Eccentricity.
                Dl_kpc (float):            Luminosity distance [kpc].
                tobs_yr (float):           Observation time [yr].
                quick_analytical (bool):   Fast geometric approximation.
                force_evolve (bool):       Force time-evolving SNR integration
                                           (only used when quick_analytical=False).
                evolve_threshold (float):  Auto-trigger evolving integration when
                                           tmerger < evolve_threshold * tobs.
                                           Default 10.0.
                n_segments (int):          Number of log-spaced time segments used
                                           for the evolving-SNR integral. Default 200.
                f_orb_max_Hz (float):      Orbital-frequency cutoff [Hz]; integration
                                           stops once f_orb exceeds this. Default 0.1.
                verbose (bool):            Per-call status print. Default True.
            """
            # ---- Geometric (G=c=1) unit conversion ----
            m1_s = m1_msun * m_sun
            m2_s = m2_msun * m_sun
            a_s = a_au * AU
            Dl_s = Dl_kpc * 1000.0 * pc
            tobs_s = tobs_yr * years

            if quick_analytical:
                # ===== Quick Analytical (Geometric Approximation) — unchanged =====
                if a_au <= 0 or e >= 1.0:
                    return 0.0

                used_tobs = tobs_s
                try:
                    t_lower = PN_waveform.tmerger_lower(m1_s, m2_s, a_s, e)
                    if t_lower <= tobs_s:
                        t_real = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, e)
                        if t_real <= tobs_s:
                            if verbose:
                                print(f"[Warning] System evolves too fast! "
                                      f"tmerger ({t_real:.2e} s) < tobs ({tobs_s:.2e} s).")
                                print(f"Approximation inaccurate. Adjusting tobs to : {t_real:.2e} s")
                            used_tobs = t_real
                except AttributeError:
                    try:
                        t_real = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, e)
                        if t_real <= tobs_s:
                            used_tobs = t_real
                    except Exception:
                        pass
                except Exception:
                    pass

                rp_s = a_s * (1 - e)
                if rp_s <= 0:
                    return 0.0

                term_f = (m1_s + m2_s) / (4 * pi * pi * np.power(rp_s, 3.0))
                f0max = 2 * np.sqrt(term_f)
                h0max = np.sqrt(32 / 5) * m1_s * m2_s / (Dl_s * a_s * (1 - e))
                Sn_val = PN_waveform.S_n_lisa(f0max)

                if Sn_val <= 0:
                    snr = 0.0
                else:
                    sqrtsnf = np.sqrt(Sn_val)
                    snr = h0max / sqrtsnf * np.sqrt(used_tobs * np.power(1 - e, 1.5))

                if verbose:
                    print(f"[Analysis] SNR_analytical (Quick) = {snr:.4f}")
                return snr

            # ===== Full Harmonic Integration: decide static vs evolving =====
            try:
                t_merger_s = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, e)
            except Exception:
                t_merger_s = np.inf

            use_evolving = force_evolve or (t_merger_s < evolve_threshold * tobs_s)

            if not use_evolving:
                # ---- Original (non-evolving) full-integration path ----
                snr = PN_waveform.SNR(m1_s, m2_s, a_s, e, Dl_s, tobs_s)
                if verbose:
                    print(f"[Analysis] SNR_analytical (Full, non-evolving) = {snr:.4f}  "
                          f"[tmerger/tobs = {t_merger_s / tobs_s:.2e}]")
                return snr

            # ---- Time-evolving SNR integration ----
            t_end_s = min(tobs_s, t_merger_s)
            if t_end_s <= 0:
                return 0.0

            #log_min = np.log10(max(t_end_s * 1e-6, 1.0))
            log_min = np.log10(t_end_s * 1e-6)
            log_max = np.log10(t_end_s)
            tau = np.logspace(log_max, log_min, n_segments + 1)
            t_grid = t_end_s - tau
            t_grid[0] = 0.0
            t_grid[-1] = t_end_s

            M_tot = m1_s + m2_s
            snr2_total = 0.0
            n_used = 0
            truncated_by_freq = False

            a_cur = a_s
            e_cur = e
            t_cur = 0.0

            for i in range(n_segments):
                t_next = t_grid[i + 1]
                dt = t_next - t_cur
                if dt <= 0:
                    continue

                if a_cur > 0:
                    f_orb_cur = np.sqrt(M_tot / (4 * pi * pi * a_cur ** 3))
                else:
                    f_orb_cur = np.inf
                if f_orb_cur > f_orb_max_Hz:
                    truncated_by_freq = True
                    break

                try:
                    a_mid, e_mid = PN_waveform.solve_ae_after_time(
                        m1_s, m2_s, a_cur, e_cur, dt / 2.0
                    )
                except Exception:
                    a_mid, e_mid = a_cur, e_cur

                if (a_mid <= 0) or (e_mid >= 1.0) or (not np.isfinite(a_mid)):
                    truncated_by_freq = True
                    break

                f_orb_mid = np.sqrt(M_tot / (4 * pi * pi * a_mid ** 3))
                if f_orb_mid > f_orb_max_Hz:
                    truncated_by_freq = True
                    break

                try:
                    dsnr = PN_waveform.SNR(m1_s, m2_s, a_mid, e_mid, Dl_s, dt)
                except Exception:
                    dsnr = 0.0
                if np.isfinite(dsnr) and dsnr > 0:
                    snr2_total += dsnr * dsnr
                    n_used += 1

                try:
                    a_new, e_new = PN_waveform.solve_ae_after_time(
                        m1_s, m2_s, a_cur, e_cur, dt
                    )
                except Exception:
                    truncated_by_freq = True
                    break
                if (a_new <= 0) or (e_new >= 1.0) or (not np.isfinite(a_new)):
                    truncated_by_freq = True
                    break

                a_cur, e_cur, t_cur = a_new, e_new, t_next

            snr = np.sqrt(snr2_total)

            if verbose:
                reason = "forced" if force_evolve else \
                    f"tmerger/tobs={t_merger_s / tobs_s:.2e} < {evolve_threshold}"
                trunc_msg = " [truncated at f_orb cutoff]" if truncated_by_freq else ""
                print(f"[Analysis] SNR_analytical (Full, evolving) = {snr:.4f}  "
                      f"[{reason}, n_seg_used={n_used}/{n_segments}, "
                      f"t_end={t_end_s / years:.3e} yr]{trunc_msg}")
            return snr

        @mute_if_global_verbose_false
        def compute_merger_time(self, m1_msun, m2_msun, a0_au, e0):
            """
            Estimate Merger Timescale.
            m1_msun, m2_msun, a_au, e
            """
            m1 = m1_msun * m_sun
            m2 = m2_msun * m_sun
            a0 = a0_au * AU

            t_merger = PN_waveform.tmerger_integral(m1, m2, a0, e0)
            print(f"[Analysis] T_merger = {t_merger/years:.4e} yrs")
            return t_merger/years

        @mute_if_global_verbose_false
        def evolve_orbit(self, m1_msun, m2_msun, a0_au, e0, delta_t_yr):
            """
            Estimate orbital parameters after given time.
            m1_msun, m2_msun, a0_au, e0, delta_t_yr
            """
            m1 = m1_msun * m_sun
            m2 = m2_msun * m_sun
            a0 = a0_au * AU
            delta_t = delta_t_yr * years

            a_new, e_new = PN_waveform.solve_ae_after_time(m1, m2, a0, e0, delta_t)
            a_new=a_new/AU

            print(f"[Analysis] After {delta_t_yr} yrs: a = {a_new:.4e} au, e = {e_new:.6f}")
            return a_new, e_new

        @mute_if_global_verbose_false
        def compute_characteristic_strain_single(self, m1_msun, m2_msun, a_au, e, Dl_kpc, tobs_yr, plot=True):
            """
            Compute h_c for single system.
            m1_msun, m2_msun, a_au, e, Dl_kpc, tobs_yr
            """
            m1 = m1_msun * m_sun
            m2 = m2_msun * m_sun
            a = a_au * AU
            Dl = Dl_kpc * 1e3 * pc
            tobs = tobs_yr * years
            label = "Manual_Input"
            print(f"[Analysis] Computing h_c for {label} (Tobs={tobs_yr} yr)...")
            res = hc_cal.calculate_single_system(
                m1=m1, m2=m2, a=a, e=e, Dl=Dl, tobs=tobs
            )
            if plot:
                hc_cal.plot_single_system_results(res)
            return res

        @mute_if_global_verbose_false
        def compute_characteristic_strain_evolve(self, m1_msun, m2_msun, a_au, e, Dl_kpc, tobs_yr=4.0,
                                                 target_n_points=100, all_harmonics=False, plot=True, verbose=True):
            """
            [NEW] Compute the time-evolving harmonic spectrum and integrated characteristic strain.
            Uses precise orbital evolution (Peters 1964).

            Parameters:
                m1_msun, m2_msun: Masses in Solar Masses
                a_au: Initial Semi-major axis in AU
                e: Initial Eccentricity
                Dl_kpc: Luminosity Distance in kpc
                tobs_yr: Observation duration in years
                target_n_points: Max number of harmonics per snapshot (downsampling target)
                all_harmonics: If True, calculate hnc for ALL harmonics in range (slow).
                plot: Whether to plot the results
                verbose: Whether to print status messages
            """
            if verbose:
                print(f"[Waveform] Computing evolving strain (Tobs={tobs_yr} yr)...")
                print(f"           m={m1_msun}+{m2_msun} Msun, a={a_au:.4e} AU, e={e:.4f}")

            results = hc_cal.calculate_evolving_system(
                m1=m1_msun,
                m2=m2_msun,
                a=a_au,
                e=e,
                Dl=Dl_kpc,
                tobs_years=tobs_yr,
                target_n_points=target_n_points,
                all_harmonics=all_harmonics,  # <--- Added
                plot=plot,
                verbose=verbose
            )
            return results

        @mute_if_global_verbose_false
        def compute_characteristic_strain_numerical(self, h_t, ts, plot=False):
            """
            Calculate the numerical characteristic strain spectrum (hc,num) from a waveform.

            :param h_t: Time-domain waveform strain sequence (array/list).
            :param ts: Sampling rate in Hz (1/dt).
            :param plot: Whether to plot the result against the LISA noise curve.
            :return: (xs, hc_num) - Frequency axis and characteristic strain spectrum.
            """
            print(f"[Analysis] Computing numerical characteristic strain (fs={ts} Hz)...")

            return PN_waveform.compute_characteristic_strain_numerical(h_t, ts, plot=plot)

        @mute_if_global_verbose_false
        def run_population_strain_analysis(self, binary_list: List[Any], tobs_yr, plot=True):
            """
            batch calculation of h_c。
            binary_list, tobs_yr
            """

            tobs = tobs_yr * years
            print(f"\n[Analysis] Computing h_c for Batch (N={len(binary_list)}, Tobs={tobs_yr} yr)...")
            snapshot_data_list = [b.to_list(schema='snapshot_std') for b in binary_list]
            batch_results = hc_cal.process_population_batch(snapshot_data_list, tobs=tobs)
            if plot:
                hc_cal.plot_simulation_results(batch_results)
            return batch_results

    # ==========================================================================
    # MODULE 5: Noise Handler (Updated)
    # ==========================================================================
    class _Noise_Handler:
        # Files shipped with the package. Everything else that looks like a
        # noise file in this folder is a leftover and may be cleaned away.
        CANONICAL_FILES = (
            'LISA_noise_ASD.csv',           # runtime curve read by PN_waveform
            'LISA_noise_ASD_official.csv',  # built-in model 'official'
            'LISA_noise_ASD_N2A5.csv',      # built-in model 'N2A5'
            'LIGO.txt',                     # built-in model 'LIGO'
        )
        BUILTIN_MODELS = ('official', 'N2A5', 'LIGO')
        _MISSING = object()

        def __init__(self):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.noise_file_path = os.path.join(current_dir, 'LISA_noise_ASD.csv')
            self.base_backup_name = 'LISA_noise_ASD_original'
            self.ligo_file_path = os.path.join(current_dir, 'LIGO.txt')
            # In-memory swap state. The default update/recover path never
            # touches any file inside the package, so an arbitrary number of
            # concurrent processes can swap curves independently.
            self._mem_backup = self._MISSING
            self._active_model = None
            self._persisted = False

        # ----------------------------------------------------------------------
        # Folder hygiene. Called only by update_noise_curve / recover_noise_curve
        # (i.e. when the curve is being changed), never on a plain read.
        # ----------------------------------------------------------------------
        @staticmethod
        def _file_is_usable(path):
            try:
                return os.path.isfile(path) and os.path.getsize(path) > 0
            except OSError:
                return False

        def repair_noise_dir(self, verbose=True):
            """
            Restore the package noise folder to its shipped state:
              * remove stray files left behind by file-swapping runs
                (LISA_noise_ASD_original_*.csv and similar), keeping only
                CANONICAL_FILES;
              * rebuild LISA_noise_ASD.csv from the official (or N2A5) model if
                it was moved away or truncated.
            The rebuild is atomic (write temp + os.replace) and every step
            swallows its own errors, so this is safe to call concurrently from
            many ranks. Set LEAP_NOISE_NO_REPAIR=1 to disable entirely.
            """
            if os.environ.get('LEAP_NOISE_NO_REPAIR', '') == '1':
                return
            target_dir = os.path.dirname(self.noise_file_path)
            removed = 0
            try:
                entries = os.listdir(target_dir)
            except OSError as e:
                if verbose:
                    print(f"[Noise] Repair: cannot list {target_dir} ({e})")
                return

            for fn in entries:
                if fn in self.CANONICAL_FILES:
                    continue
                if not (fn.startswith('LISA_noise_ASD') and fn.endswith('.csv')):
                    continue
                try:
                    os.remove(os.path.join(target_dir, fn))
                    removed += 1
                except OSError:
                    pass  # another rank removed it first

            if not self._file_is_usable(self.noise_file_path):
                source = os.path.join(target_dir, 'LISA_noise_ASD_official.csv')
                if not self._file_is_usable(source):
                    source = os.path.join(target_dir, 'LISA_noise_ASD_N2A5.csv')
                if self._file_is_usable(source):
                    tmp = f"{self.noise_file_path}.tmp.{os.getpid()}"
                    try:
                        shutil.copyfile(source, tmp)
                        os.replace(tmp, self.noise_file_path)
                        if verbose:
                            print(f"[Noise] Repair: rebuilt "
                                  f"{os.path.basename(self.noise_file_path)} from "
                                  f"{os.path.basename(source)}")
                    except OSError as e:
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
                        if verbose:
                            print(f"[Noise] Repair: rebuild failed ({e})")
                elif verbose:
                    print("[Noise] Repair: no canonical source found to rebuild "
                          "the runtime curve.")

            if removed and verbose:
                print(f"[Noise] Repair: removed {removed} stray file(s).")

        def generate_noise_data(self, model='official', f_min=1e-6, f_max=1.0, n_points=3000):
            """
            Generates LISA Noise ASD based on the selected model.
            integrated with Log-Log extrapolation for better physical accuracy.
            """

            flist = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

            if model == 'N2A5':
                # === N2A5 Analytical Model (unchanged) ===
                # Constants setup
                def S_gal_N2A5_vec(f):
                    conds = [
                        (f >= 1.0e-5) & (f < 1.0e-3),
                        (f >= 1.0e-3) & (f < 10 ** -2.7),
                        (f >= 10 ** -2.7) & (f < 10 ** -2.4),
                        (f >= 10 ** -2.4) & (f <= 0.01)
                    ]
                    funcs = [
                        lambda x: x ** -2.3 * 10 ** -44.62 * 20.0 / 3.0,
                        lambda x: x ** -4.4 * 10 ** -50.92 * 20.0 / 3.0,
                        lambda x: x ** -8.8 * 10 ** -62.8 * 20.0 / 3.0,
                        lambda x: x ** -20.0 * 10 ** -89.68 * 20.0 / 3.0
                    ]
                    return np.piecewise(f, conds, funcs + [0])

                def S_n_lisa_calc(f):
                    m1 = 5.0e9
                    m2 = sciconsts.c * 0.41 / m1 / 2.0
                    term1 = 20.0 / 3.0 * (1 + (f / m2) ** 2)
                    term2 = 4.0 * (9.0e-30 / (2 * np.pi * f) ** 4 * (1 + 1.0e-4 / f)) + 2.96e-23 + 2.65e-23
                    return term1 * term2 / m1 ** 2 + S_gal_N2A5_vec(f)

                Sn_list = S_n_lisa_calc(flist)
                return flist, np.sqrt(Sn_list)

            elif model == 'official':
                # === Official File Loading with Smart Extrapolation ===
                target_dir = os.path.dirname(self.noise_file_path)
                source_path = os.path.join(target_dir, "LISA_noise_ASD_official.csv")

                if not os.path.exists(source_path):
                    print(f"[Noise] Error: Official noise file not found at {source_path}")
                    return flist, np.zeros_like(flist)

                try:
                    data = np.loadtxt(source_path, delimiter=',')
                    # data = np.loadtxt(source_path, delimiter=',', skiprows=1)

                    data = data[data[:, 0].argsort()]
                    f_ref = data[:, 0]
                    asd_ref = data[:, 1]

                    mask = (f_ref > 0) & (asd_ref > 0)
                    f_ref = f_ref[mask]
                    asd_ref = asd_ref[mask]

                    log_f_ref = np.log10(f_ref)
                    log_asd_ref = np.log10(asd_ref)

                    # slope = (y2 - y1) / (x2 - x1)
                    if len(log_f_ref) >= 2:
                        slope_low = (log_asd_ref[1] - log_asd_ref[0]) / (log_f_ref[1] - log_f_ref[0])
                    else:
                        slope_low = 0  # Fallback


                    log_f_target = np.log10(flist)

                    # Interpolate
                    # Note: right=0.0 means log(ASD)=0 -> ASD=1.0
                    log_asd_target = np.interp(
                        log_f_target,
                        log_f_ref,
                        log_asd_ref,
                        left=np.nan,
                        right=0.0
                    )

                    # (Power-law Extrapolation)
                    # y = y0 + slope * (x - x0)
                    nan_mask = np.isnan(log_asd_target)
                    if np.any(nan_mask):
                        # x0 = log_f_ref[0], y0 = log_asd_ref[0]
                        log_asd_target[nan_mask] = log_asd_ref[0] + \
                                                   slope_low * (log_f_target[nan_mask] - log_f_ref[0])

                    asd_final = np.power(10.0, log_asd_target)

                    return flist, asd_final

                except Exception as e:
                    print(f"[Noise] Error processing official file: {e}")
                    return self.generate_noise_data(model='N2A5', f_min=f_min, f_max=f_max, n_points=n_points)

            elif model in ('LIGO', 'ligo', 'ground'):
                # === Ground-based ASD, read from LIGO.txt next to core.py ===
                # Returned on the file's own frequency grid: the ground-based
                # band (~1 Hz - 10 kHz) has nothing to do with the LISA
                # f_min/f_max defaults, so resampling onto flist is skipped.
                path = os.environ.get('LEAP_LIGO_NOISE', self.ligo_file_path)
                if not self._file_is_usable(path):
                    print(f"[Noise] Error: LIGO noise file not found at {path}")
                    return flist, np.zeros_like(flist)
                try:
                    try:
                        d = np.loadtxt(path, unpack=True)
                    except ValueError:
                        d = np.loadtxt(path, delimiter=',', unpack=True)
                    f_ref = np.asarray(d[0], dtype=float).ravel()
                    asd_ref = np.asarray(d[1], dtype=float).ravel()
                    order = np.argsort(f_ref)
                    f_ref, asd_ref = f_ref[order], asd_ref[order]
                    keep = (np.isfinite(f_ref) & np.isfinite(asd_ref)
                            & (f_ref > 0) & (asd_ref > 0))
                    f_ref, asd_ref = f_ref[keep], asd_ref[keep]
                    if len(f_ref) < 2:
                        print(f"[Noise] Error: too few valid points in {path}")
                        return flist, np.zeros_like(flist)
                    return f_ref, asd_ref
                except Exception as e:
                    print(f"[Noise] Error processing LIGO file: {e}")
                    return flist, np.zeros_like(flist)

            else:
                print(f"[Noise] Error: Unknown model '{model}'")
                return flist, np.zeros_like(flist)
        def _inject_noise_data(self):
            """
            Internal Method: Force-update the _LISA_NOISE_DATA global variable in PN_waveform.
            This bypasses file I/O issues and fixes the missing 'use_file' key bug.
            """
            if not os.path.exists(self.noise_file_path):
                print(f"[Noise] Warning: File not found for injection: {self.noise_file_path}")
                return

            try:
                try:
                    data = np.loadtxt(self.noise_file_path, delimiter=',')
                except ValueError:
                    data = np.loadtxt(self.noise_file_path, delimiter=',', skiprows=1)

                data = data[data[:, 0].argsort()]
                f_data = data[:, 0]
                asd_data = data[:, 1]

                mask = (f_data > 0) & (asd_data > 0)
                f_data = f_data[mask]
                asd_data = asd_data[mask]

                if len(f_data) < 2:
                    print("[Noise] Warning: Not enough valid points in noise file.")
                    return

                log_f = np.log10(f_data)
                log_asd = np.log10(asd_data)

                # Slope = dy / dx
                low_f_slope = (log_asd[1] - log_asd[0]) / (log_f[1] - log_f[0])

                noise_dict = {
                    'f_min': f_data[0],
                    'f_max': f_data[-1],
                    'log_f': log_f,
                    'log_asd': log_asd,
                    'low_f_slope': low_f_slope,
                    'log_f_0': log_f[0],
                    'log_asd_0': log_asd[0],
                    'use_file': True
                }


                PN_waveform._LISA_NOISE_DATA = noise_dict
                print(f"[Noise] Force-injected updated noise profile (Points: {len(f_data)})")
                print(f"        Slope: {low_f_slope:.4f}, f_range: [{f_data[0]:.1e}, {f_data[-1]:.1e}] Hz")

            except Exception as e:
                print(f"[Noise] Warning: Data injection failed: {e}")

        def _inject_noise_arrays(self, flist, asdlist, label='custom'):
            """
            Internal Method: build the _LISA_NOISE_DATA dict straight from
            arrays and push it into PN_waveform. Same payload as
            _inject_noise_data, but nothing is read from or written to disk,
            so concurrent processes cannot interfere with each other.
            The previous dict is kept in memory for recover_noise_curve().
            """
            f_data = np.asarray(flist, dtype=float).ravel()
            asd_data = np.asarray(asdlist, dtype=float).ravel()

            order = np.argsort(f_data)
            f_data, asd_data = f_data[order], asd_data[order]
            keep = (np.isfinite(f_data) & np.isfinite(asd_data)
                    & (f_data > 0) & (asd_data > 0))
            f_data, asd_data = f_data[keep], asd_data[keep]

            if len(f_data) < 2:
                print("[Noise] Warning: Not enough valid points to inject.")
                return False

            log_f = np.log10(f_data)
            log_asd = np.log10(asd_data)
            low_f_slope = (log_asd[1] - log_asd[0]) / (log_f[1] - log_f[0])

            if self._mem_backup is self._MISSING:
                prev = getattr(PN_waveform, '_LISA_NOISE_DATA', self._MISSING)
                # shallow copy: callers may flip flags such as 'use_file' on the
                # live dict, and that must not leak into the snapshot
                self._mem_backup = dict(prev) if isinstance(prev, dict) else prev

            PN_waveform._LISA_NOISE_DATA = {
                'f_min': f_data[0],
                'f_max': f_data[-1],
                'log_f': log_f,
                'log_asd': log_asd,
                'low_f_slope': low_f_slope,
                'log_f_0': log_f[0],
                'log_asd_0': log_asd[0],
                'use_file': True
            }
            self._active_model = label
            print(f"[Noise] In-memory injection '{label}' (Points: {len(f_data)})")
            print(f"        Slope: {low_f_slope:.4f}, "
                  f"f_range: [{f_data[0]:.1e}, {f_data[-1]:.1e}] Hz")
            return True

        def _restore_from_memory(self):
            """Put back the dict that was live before the first injection."""
            if self._mem_backup is self._MISSING:
                return False
            if self._mem_backup is None:
                if hasattr(PN_waveform, '_LISA_NOISE_DATA'):
                    del PN_waveform._LISA_NOISE_DATA
            else:
                PN_waveform._LISA_NOISE_DATA = self._mem_backup
            self._mem_backup = self._MISSING
            self._active_model = None
            return True

        @mute_if_global_verbose_false
        def _reload_dependencies(self):
            """
            Reload backend modules and inject data.
            """
            print("[Noise] Auto-reloading backend modules...")
            try:
                importlib.reload(GN_BBH)
                importlib.reload(Field_BBH)
                importlib.reload(Field_BBH_Elliptical)
                importlib.reload(hc_cal)
                importlib.reload(PN_waveform)

                self._inject_noise_data()

                print("[Noise] Modules reloaded & Data injected.")
            except Exception as e:
                print(f"[Noise] Warning: Module auto-reload failed. Error: {e}")

        @mute_if_global_verbose_false
        def update_noise_curve(self, data_list, persist=False, reload=False):
            """
            change noise curve, and backup the old one in memory

            input: data_list = [flist, ASDlist]   custom curve
                   data_list = 'LIGO'             built-in ground-based model
                   data_list = 'official'/'N2A5'  built-in LISA models

            By default nothing inside the package is written or moved: the
            curve is injected into PN_waveform and the previous one is kept in
            memory, which is what makes this safe under massive parallelism.
            Pass persist=True for the legacy behaviour (rewrite
            LISA_noise_ASD.csv and leave a backup file on disk) -- do not use
            that from concurrent jobs.
            """
            # a curve change is about to happen: tidy up whatever earlier runs
            # left in the package folder before doing anything else
            self.repair_noise_dir()

            if isinstance(data_list, str):
                model = data_list
                if model not in self.BUILTIN_MODELS:
                    print(f"[Noise] Error: Unknown model '{model}'. "
                          f"Available: {', '.join(self.BUILTIN_MODELS)}")
                    return
                data_list = list(self.generate_noise_data(model=model))
                label = model
            else:
                label = 'custom'

            if len(data_list) != 2:
                print("Error: Input must be [flist, ASDlist].")
                return

            if not persist:
                self._inject_noise_arrays(data_list[0], data_list[1], label=label)
                if reload:
                    # reload re-reads the on-disk curve, so re-inject afterwards
                    self._reload_dependencies()
                    self._inject_noise_arrays(data_list[0], data_list[1], label=label)
                return

            flist, asdlist = data_list[0], data_list[1]
            abs_path = os.path.abspath(self.noise_file_path)


            if os.path.exists(self.noise_file_path):
                i = 1
                while True:
                    backup_name = f"{self.base_backup_name}_{i}.csv"
                    backup_path = os.path.join(os.path.dirname(self.noise_file_path), backup_name)
                    if not os.path.exists(backup_path):
                        shutil.move(self.noise_file_path, backup_path)
                        print(f"[Noise] Backup created: {os.path.basename(backup_path)}")
                        break
                    i += 1

            try:
                sort_idx = np.argsort(flist)
                flist = flist[sort_idx]
                asdlist = asdlist[sort_idx]

                data_to_save = np.column_stack((flist, asdlist))
                np.savetxt(self.noise_file_path, data_to_save, delimiter=',', header='f,ASD', comments='')
                print(f"[Noise] Updated noise file at: {os.path.basename(abs_path)}")

                self._reload_dependencies()
                self._persisted = True

            except Exception as e:
                print(f"[Noise] Error writing file: {e}")

        @mute_if_global_verbose_false
        def recover_noise_curve(self, version=None):
            """
            Undo an update_noise_curve().

            If the curve was swapped in memory (the default), the previous dict
            is simply put back and `version` is ignored. Otherwise the legacy
            file-based restore is used. 'official' / 'N2A5' / 'LIGO' can also be
            passed to load a built-in model in memory.
            """
            # in-memory swap: restore and clean up, no file work needed
            if self._restore_from_memory():
                print("[Noise] Recovered noise from in-memory snapshot.")
                self.repair_noise_dir()
                return

            if version in self.BUILTIN_MODELS and not self._persisted:
                f, asd = self.generate_noise_data(model=version)
                self._inject_noise_arrays(f, asd, label=version)
                self.repair_noise_dir()
                return

            target_dir = os.path.dirname(self.noise_file_path)

            if version == 'official':
                source_name = "LISA_noise_ASD_official.csv"
            elif version == 'N2A5':
                source_name = "LISA_noise_ASD_N2A5.csv"
            elif version is None:
                source_name = f"{self.base_backup_name}_1.csv"
            else:
                source_name = f"{self.base_backup_name}_{version}.csv"

            source_path = os.path.join(target_dir, source_name)

            if not os.path.exists(source_path):
                print(f"[Noise] Error: Source file not found: {source_name}")
                return

            try:
                shutil.copyfile(source_path, self.noise_file_path)
                print(f"[Noise] Recovered noise from: {source_name}")
                self._reload_dependencies()
                self._persisted = False
                # restore is done, now drop the leftover backup files
                self.repair_noise_dir()
            except Exception as e:
                print(f"[Noise] Error recovering file: {e}")

        @mute_if_global_verbose_false
        def clean_backups(self):
            """Remove every non-canonical noise file and rebuild the runtime
            curve if it went missing."""
            print(f"[Noise] Cleaning backup files...")
            self.repair_noise_dir()

        @mute_if_global_verbose_false
        def get_noise_curve(self, plot=True):
            # An in-memory swap does not change the file, so report the dict
            # that PN_waveform is actually using whenever one is live.
            live = getattr(PN_waveform, '_LISA_NOISE_DATA', None)
            if self._active_model is not None and isinstance(live, dict) \
                    and 'log_f' in live and 'log_asd' in live:
                f = np.power(10.0, np.asarray(live['log_f'], dtype=float))
                asd = np.power(10.0, np.asarray(live['log_asd'], dtype=float))
                noise_char = np.sqrt(f) * asd
                print(f"[Noise] Active curve: in-memory '{self._active_model}'")
                if plot:
                    plt.figure(figsize=(8, 6))
                    plt.loglog(f, noise_char, color='black', linewidth=1.5,
                               label=f"Current Noise Curve ({self._active_model})")
                    plt.xlabel('Frequency [Hz]', fontsize=12)
                    plt.ylabel(r'$\sqrt{f S_n(f)}$ [unitless]', fontsize=12)
                    plt.title(f'Characteristic Noise Strain', fontsize=12)
                    plt.grid(True, which="both", ls="--", alpha=0.4)
                    plt.legend()
                    plt.show()
                return [f, noise_char]

            if not os.path.exists(self.noise_file_path):
                print(f"[Noise] Error: File not found.")
                return None
            try:
                try:
                    data = np.loadtxt(self.noise_file_path, delimiter=',')
                except ValueError:
                    data = np.loadtxt(self.noise_file_path, delimiter=',', skiprows=1)

                f = data[:, 0]
                asd = data[:, 1]
                noise_char = np.sqrt(f) * asd

                if plot:
                    plt.figure(figsize=(8, 6))
                    plt.loglog(f, noise_char, color='black', linewidth=1.5, label='Current Noise Curve')
                    plt.xlabel('Frequency [Hz]', fontsize=12)
                    plt.ylabel(r'$\sqrt{f S_n(f)}$ [unitless]', fontsize=12)
                    plt.title(f'Characteristic Noise Strain', fontsize=12)
                    plt.grid(True, which="both", ls="--", alpha=0.4)
                    plt.legend()
                    plt.show()
                return [f, noise_char]
            except Exception as e:
                print(f"[Noise] Error reading curve: {e}")
                return None


# ==============================================================================
# ADDED FEATURE: getMWcatalog (Milky Way Population Catalog Generator)
# ==============================================================================
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.lines import Line2D
import os
import random
import numpy as np
import copy

_CONST_AU = 1.495978707e11
_CONST_G = 6.67430e-11
_CONST_C = 299792458.0
_CONST_MSUN = 1.98847e30
_CONST_YEAR = 31557600.0


def _calculate_evaporation_tau(a_sec, n_sec3, sigmav_dimless, mp_sec):
    return 2.33e-3 * sigmav_dimless / (mp_sec * n_sec3 * a_sec) / 0.69315


def _calculate_ecrit_and_tmerger(surv_a_sec, surv_e, n_density_sec3, sigmav_dimless, age_sec, mp_sec, m1_sec, m2_sec):
    beta_ecrit = (85.0 / 3.0) * m1_sec * m2_sec * (m1_sec + m2_sec)
    forb_val = 1.0 / (2.0 * np.pi) * np.sqrt(m1_sec + m2_sec) * np.power(surv_a_sec, -1.5)
    term1 = 0.1 * sigmav_dimless / forb_val
    inner_mult = (27.0 / 4.0) * np.power(surv_a_sec, 29.0 / 7.0) * (mp_sec ** 2) / (m1_sec + m2_sec) * np.power(
        n_density_sec3 * np.pi / beta_ecrit, 2.0 / 7.0)
    term2 = np.sqrt(1.0 / sigmav_dimless * np.power(inner_mult, 7.0 / 12.0))
    b_val = np.minimum(term1, term2)
    T_val = np.minimum(1.0 / (n_density_sec3 * np.pi * b_val * b_val * sigmav_dimless), age_sec)
    ecrit_inner = 1.0 - np.power(beta_ecrit * T_val / np.power(surv_a_sec, 4.0), 2.0 / 7.0)
    ecrit = np.sqrt(np.maximum(0.0, ecrit_inner))
    beta_tmerger = (64.0 / 5.0) * m1_sec * m2_sec * (m1_sec + m2_sec)
    tc = np.power(surv_a_sec, 4.0) / (4.0 * beta_tmerger)
    tmerger_c = (768.0 / 425.0) * tc * np.power(1.0 - ecrit * ecrit, 3.5)
    tmerger_actual = (768.0 / 425.0) * tc * np.power(1.0 - surv_e * surv_e, 3.5)
    return tmerger_actual, tmerger_c


def _run_evaporation_simulation(pct=0.1):
    age_gyr = 10.0
    fbh = 7.5e-4
    n_blocks = 40
    r_min_kpc, r_max_kpc = (0.5, 15.0)
    log_a_min, log_a_max = (2.0, 4.5)

    pc_sec = 3.261 * sciconsts.light_year / _CONST_C
    m_sun_sec = _CONST_MSUN * _CONST_G / np.power(_CONST_C, 3.0)
    year_sec = _CONST_YEAR
    au_sec = _CONST_AU / _CONST_C

    age_sec = age_gyr * 1e9 * year_sec
    sigmav_dimless = 50e3 / _CONST_C
    mp_sec = 0.6 * m_sun_sec
    m1_sec = 10.0 * m_sun_sec
    m2_sec = 10.0 * m_sun_sec

    n0_sec3 = 0.1 / np.power(pc_sec, 3.0)
    R_sun_sec = 8.0 * 1e3 * pc_sec
    R_l_sec = 2.6 * 1e3 * pc_sec
    h_z_sec = 1.0 * 1e3 * pc_sec

    r_min_sec = r_min_kpc * 1e3 * pc_sec
    r_max_sec = r_max_kpc * 1e3 * pc_sec
    delta_r_sec = (r_max_sec - r_min_sec) / n_blocks

    global_res = []

    for i in range(n_blocks):
        r_mid_sec = r_min_sec + i * delta_r_sec + delta_r_sec / 2.0
        r_mid_kpc = r_mid_sec / (1e3 * pc_sec)
        vol_sec3 = 2 * np.pi * r_mid_sec * delta_r_sec * (2 * h_z_sec)
        n_density_sec3 = n0_sec3 * np.exp(-(r_mid_sec - R_sun_sec) / R_l_sec)

        N_real_block = vol_sec3 * n_density_sec3 * fbh
        N_gen = int(round(N_real_block * pct))

        if N_gen <= 0: continue

        log_a = np.random.uniform(log_a_min, log_a_max, N_gen)
        a_sec = np.power(10, log_a) * au_sec
        e_init = np.sqrt(np.random.random(N_gen))

        tau_sec = _calculate_evaporation_tau(a_sec, n_density_sec3, sigmav_dimless, mp_sec)
        prob1 = np.exp(-age_sec / tau_sec)
        survived_mask_1 = np.random.random(N_gen) < prob1
        n_survived_1 = np.sum(survived_mask_1)

        if n_survived_1 > 0:
            surv_a_sec = a_sec[survived_mask_1]
            surv_e = e_init[survived_mask_1]

            t_act, t_c = _calculate_ecrit_and_tmerger(
                surv_a_sec, surv_e, n_density_sec3, sigmav_dimless, age_sec, mp_sec, m1_sec, m2_sec
            )

            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                prob2 = np.exp(-t_c / t_act)
                prob2 = np.nan_to_num(prob2, nan=0.0, posinf=1.0)

            survived_mask_2 = np.random.random(n_survived_1) < prob2
            n_survived_final = np.sum(survived_mask_2)

            if n_survived_final > 0:
                final_a_sec = surv_a_sec[survived_mask_2]
                final_e = surv_e[survived_mask_2]
                phi = np.random.uniform(0, 2 * np.pi, n_survived_final)
                dl_kpc = np.sqrt((r_mid_kpc * np.cos(phi) - 8.0) ** 2 + (r_mid_kpc * np.sin(phi)) ** 2)

                for k in range(n_survived_final):
                    global_res.append(['Field', 10.0, 10.0, final_a_sec[k] / au_sec, final_e[k], dl_kpc[k], -1.0, {}])

    return global_res


def _plot_mw_catalog(catalog):
    if not catalog:
        print("[Catalog] No systems to plot.")
        return

    raw_labels = np.array([row[0] for row in catalog])
    a = np.array([row[3] for row in catalog], dtype=float)
    e = np.array([row[4] for row in catalog], dtype=float)
    snr = np.array([row[6] for row in catalog], dtype=float)

    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)

    x_min, x_max = (2e-3, 2e3)
    y_min, y_max = (1e-5, 1.2)
    a_grid = np.logspace(np.log10(x_min), np.log10(x_max), 1000) * _CONST_AU

    m1_m2_bg = 10.0 * _CONST_MSUN
    beta = 64 / 5 * _CONST_G ** 3 * m1_m2_bg * m1_m2_bg * (m1_m2_bg + m1_m2_bg) / _CONST_C ** 5

    merger_lines = [
        (1e10, r'$t_{\mathrm{merger}}=10^{10}\,\mathrm{yr}$', 'navy', '-.'),
        (1e7, r'$t_{\mathrm{merger}}=10^{7}\,\mathrm{yr}$', 'purple', '--'),
        (1e4, r'$t_{\mathrm{merger}}=10^{4}\,\mathrm{yr}$', 'maroon', ':')
    ]

    bg_handles, bg_labels = [], []
    for tyr, lbl, color, ls in merger_lines:
        term = (tyr * _CONST_YEAR * (425.0 / 768.0) * 4.0 * beta) / (a_grid ** 4)
        val_ome = np.power(term, 2.0 / 7.0)
        valid = (val_ome <= y_max) & (val_ome >= y_min)
        if np.any(valid):
            line, = ax.plot(a_grid[valid] / _CONST_AU, val_ome[valid], color=color, linestyle=ls, linewidth=2.0,
                            alpha=0.7, zorder=0)
            bg_handles.append(line)
            bg_labels.append(lbl)

    cmap, norm = copy.copy(mpl.colormaps['jet']), mcolors.LogNorm(vmin=0.1, vmax=100)
    scatter_handles, sc_final = [], None

    marker_settings = {
        'GN': {'marker': '*', 'scale': 1.4, 'legend_size': 14},
        'GC': {'marker': 'o', 'scale': 1.0, 'legend_size': 10},
        'Field': {'marker': '^', 'scale': 1.0, 'legend_size': 10}
    }

    for label_name, settings in marker_settings.items():
        mask = (raw_labels == label_name)
        if not np.any(mask): continue
        sub_a, sub_e, sub_snr = a[mask], e[mask], snr[mask]
        size_scale = settings['scale']

        # Evaporating
        m_evap = sub_snr < 0
        if np.any(m_evap):
            plot_a_evap = sub_a[m_evap]
            plot_e_evap = sub_e[m_evap]
            ax.scatter(plot_a_evap, 1.0 - plot_e_evap, s=20 * size_scale,
                       marker=settings['marker'], c=np.full_like(plot_a_evap, 0.1), cmap=cmap, norm=norm,
                       alpha=0.3, edgecolors='none', zorder=1)

        # Normal
        m_low, m_high = (sub_snr >= 0.0) & (sub_snr < 0.1), sub_snr >= 0.1
        if np.any(m_low):
            ax.scatter(sub_a[m_low], 1.0 - sub_e[m_low], s=20 * size_scale, c=sub_snr[m_low], marker=settings['marker'],
                       cmap=cmap, norm=norm, alpha=0.5, zorder=5)
        if np.any(m_high):
            idx = np.argsort(sub_snr[m_high])[::-1]
            s_plot = np.clip(np.sqrt(np.clip(sub_snr[m_high][idx], 1e-3, 200)) * 50, 20, 300) * size_scale
            sc_final = ax.scatter(sub_a[m_high][idx], 1.0 - sub_e[m_high][idx], s=s_plot, c=sub_snr[m_high][idx],
                                  marker=settings['marker'], cmap=cmap, norm=norm, edgecolors='k', linewidths=0.3,
                                  alpha=0.7, zorder=10)

        proxy = Line2D([], [], color='white', marker=settings['marker'], markeredgecolor='k', markerfacecolor='gray',
                       markersize=settings['legend_size'], label=label_name)
        scatter_handles.append(proxy)

    legend_bg = ax.legend(bg_handles, bg_labels, loc='lower left', bbox_to_anchor=(0.0, 0.0), fontsize=13, frameon=True,
                          framealpha=1.0, edgecolor='gray')
    legend_bg.set_zorder(100)
    ax.add_artist(legend_bg)

    legend_pop = ax.legend(handles=scatter_handles, loc='lower left', bbox_to_anchor=(0.0, 0.25), title='Population',
                           fontsize=13, frameon=True, framealpha=1.0, edgecolor='gray')
    legend_pop.set_zorder(101)

    stroke_effects = [path_effects.withStroke(linewidth=2, foreground='black')]
    common_txt_props = {
        'fontsize': 11, 'fontweight': 'bold', 'color': 'white',
        'verticalalignment': 'center', 'horizontalalignment': 'center',
        'multialignment': 'center', 'zorder': 105, 'path_effects': stroke_effects
    }

    ax.text(10 ** ((np.log10(100) + np.log10(x_max)) / 2),
            10 ** (np.log10(y_min) + (np.log10(y_max) - np.log10(y_min)) * (2.0 / 3.0)),
            "Wide\nField\nBBHs\n($a>100$au)", transform=ax.transData,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='gray', alpha=0.8, edgecolor='none'), **common_txt_props)
    ax.text(0.9, 0.4, "GN and \n GC BBHs", transform=ax.transData,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='gray', alpha=0.8, edgecolor='none'), **common_txt_props)

    ax.set_xscale('log');
    ax.set_yscale('log');
    ax.set_xlim(x_min, x_max);
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r'$a\ [\mathrm{au}]$', fontsize=26);
    ax.set_ylabel(r'$1 - e$', fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=22, length=8, width=1.5, direction='in', top=True, right=True)
    if sc_final:
        cbar = plt.colorbar(sc_final, pad=0.02)
        cbar.ax.set_title('SNR', fontsize=22, pad=12)
    plt.grid(True, which='both', ls='-', alpha=0.15, zorder=0)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# Spinner: a lightweight background-thread status indicator for long-running ops
# ==============================================================================
class _Spinner:
    """
    Minimal thread-backed spinner. Writes to real stdout (`sys.__stdout__`) so
    it still shows up even when the caller has redirected stdout elsewhere.

    Usage:
        with _Spinner("Doing thing") as sp:
            for i, item in enumerate(items):
                do_work(item)
                sp.update(f"({i+1}/{len(items)})")
    """
    _FRAMES = ['|', '/', '-', '\\']

    def __init__(self, message=""):
        self._base_msg = message
        self._suffix = ""
        self._stop_evt = threading.Event()
        self._thread = None
        # Only animate when we're actually writing to a TTY-like stream.
        # In non-TTY (piped, some CI logs, stale notebook kernels) we still
        # print a single start/end line so the user sees progress.
        self._stream = sys.__stdout__
        self._interactive = hasattr(self._stream, 'isatty') and self._stream.isatty()

    def __enter__(self):
        self._stop_evt.clear()
        if self._interactive:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            # Non-interactive: just announce the stage start once.
            self._stream.write(f"  {self._base_msg} ...\n")
            self._stream.flush()
        return self

    def update(self, suffix):
        """Update the trailing text shown next to the spinner frame."""
        self._suffix = suffix

    def _spin(self):
        i = 0
        while not self._stop_evt.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            line = f"\r  {frame} {self._base_msg} {self._suffix}"
            # Pad to clear any leftover characters from a previous, longer line.
            self._stream.write(line.ljust(80))
            self._stream.flush()
            i += 1
            time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._interactive:
            # Overwrite the spinner line with a final done/failed marker.
            mark = "✗" if exc_type is not None else "✓"
            final = f"\r  {mark} {self._base_msg} {self._suffix}"
            self._stream.write(final.ljust(80) + "\n")
            self._stream.flush()
        return False  # never swallow exceptions


@contextlib.contextmanager
def _silence_stdout():
    """
    Temporarily suppress noise from sub-function calls:
      - stdout prints (via redirect_stdout)
      - stderr writes (e.g. warnings module default stream)
      - RuntimeWarning / numpy FloatingPointError-style warnings
    We leave exceptions alone — errors still propagate normally.
    """
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), \
            contextlib.redirect_stderr(buf_err), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # numpy's internal divide-by-zero / invalid-value warnings go through
        # np.seterr, not the warnings module, so silence those too.
        old_np_err = np.seterr(divide='ignore', invalid='ignore', over='ignore')
        try:
            yield
        finally:
            np.seterr(**old_np_err)


def _print_catalog_config(tobs_yr, include_field_bkg, bkg_pct):
    """
    Print the list of populations that will be included in the catalog, along
    with the internal defaults used by getMWcatalog. Makes the 'what am I
    actually getting?' question answerable without reading the source.
    """
    print("=" * 64)
    print(f"[Catalog] Milky Way GW Source Catalog Generator")
    print("=" * 64)
    print(f"  Observation time (Tobs) : {tobs_yr} yr")
    print(f"  Included populations:")

    print(f"    • GN  (Galactic Nucleus)   "
          f"  loaded from gn_snapshots_aggregated.npy, sampled 1/20 (1 realization); rate_gn=3.0, age_ync=2~8 Myr")
    # ------------------------------------------------------

    print(f"    • GC  (Globular Clusters)  "
          f"  mode='single', channel='all'  (~1/10 of the CMC MW GC catalog)")
    print(f"    • Field (Fly-by induced)   "
          f"  loaded from field_snapshots_aggregated.npy, sampled 1:1000; LIGO mass distribution")

    if include_field_bkg:
        print(f"    • Wide-field background    "
              f"  evaporation model, sampled at {bkg_pct * 100:.1f}%")
    else:
        print(f"    • Wide-field background    DISABLED (set include_field_bkg=True to enable)")
    print("-" * 64)


# ==============================================================================
# getMWcatalog (refactored: Pre-loaded GN/Field, Smart Two-Step SNR)
# ==============================================================================
@mute_if_global_verbose_false
def getMWcatalog(self, plot=True, include_field_bkg=False, bkg_pct=0.001, tobs_yr=10.0):
    """
    Generate and optionally plot a full Milky Way GW Population Catalog.
    Combines populations from Galactic Nucleus (GN), Globular Clusters (GC), and Field.
    SNR is computed using a smart two-step method: quick geometric approximation first,
    followed by full harmonic integration ONLY for sources with SNR > 0.1.
    """
    import os
    import random
    import numpy as np

    include_evaporated = include_field_bkg
    evap_pct = bkg_pct

    # --- 0. Print the configuration upfront ---------------------------------
    _print_catalog_config(tobs_yr, include_field_bkg, bkg_pct)

    all_binaries = []
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 1. GN (Galactic Nucleus) -------------------------------------------
    gn_npy_name = 'gn_snapshots_aggregated.npy'
    gn_sample_divisor = 20
    target_file_gn = os.path.join(current_dir, gn_npy_name)

    if os.path.exists(target_file_gn):
        with _Spinner("Loading GN population") as sp:
            try:
                with _silence_stdout():
                    raw_gn = np.load(target_file_gn, allow_pickle=True)
                    if hasattr(raw_gn, 'item') and isinstance(raw_gn.item(), dict):
                        data_block_gn = raw_gn.item().get('data', [])
                    else:
                        data_block_gn = raw_gn

                    if isinstance(data_block_gn, np.ndarray):
                        all_gn_rows = data_block_gn.tolist()
                    else:
                        all_gn_rows = list(data_block_gn)

                    gn_added = 0
                    if len(all_gn_rows) > 0:
                        num_gn = max(1, len(all_gn_rows) // gn_sample_divisor)
                        num_gn = min(num_gn, len(all_gn_rows))
                        sampled_gn = random.sample(all_gn_rows, num_gn)

                        for row in sampled_gn:
                            cb = CompactBinary.from_list(
                                data_list=list(row), schema='snapshot_std'
                            )
                            cb.extra['source_label'] = cb.label
                            cb.label = "GN"
                            all_binaries.append(cb)
                        gn_added = num_gn

                sp.update(f"({gn_added} systems, 1/{gn_sample_divisor} of total)")
            except Exception as e:
                sp.update(f"(FAILED: {e})")
    else:
        print(f"[Catalog] ⚠️  Warning: GN file not found at {target_file_gn}. Skipping.")

    # --- 2. GC (Globular Clusters) ------------------------------------------
    if hasattr(self, 'GC'):
        with _Spinner("Sampling GC population") as sp:
            with _silence_stdout():
                gc_pops = self.GC.get_snapshot(
                    mode='single', channel='all', plot=False
                )
            for b in gc_pops:
                b.extra['source_label'] = b.label
                b.label = "GC"
            all_binaries.extend(gc_pops)
            sp.update(f"({len(gc_pops)} systems)")

    # --- 3. Field (fly-by / isolated) ---------------------------------------
    field_npy_name = 'field_snapshots_aggregated.npy'
    field_sample_divisor = 1000
    target_file_field = os.path.join(current_dir, field_npy_name)

    if os.path.exists(target_file_field):
        with _Spinner("Loading Field population") as sp:
            try:
                with _silence_stdout():
                    raw_field = np.load(target_file_field, allow_pickle=True)
                    if hasattr(raw_field, 'item') and isinstance(raw_field.item(), dict):
                        data_block_f = raw_field.item().get('data', [])
                    else:
                        data_block_f = raw_field

                    if isinstance(data_block_f, np.ndarray):
                        all_field_rows = data_block_f.tolist()
                    else:
                        all_field_rows = list(data_block_f)

                    field_added = 0
                    if len(all_field_rows) > 0:
                        num_f = max(1, len(all_field_rows) // field_sample_divisor)
                        num_f = min(num_f, len(all_field_rows))
                        sampled_field = random.sample(all_field_rows, num_f)

                        for row in sampled_field:
                            cb = CompactBinary.from_list(
                                data_list=list(row), schema='snapshot_std'
                            )
                            cb.extra['source_label'] = cb.label
                            cb.label = "Field"
                            all_binaries.append(cb)
                        field_added = num_f

                sp.update(f"({field_added} systems, 1/{field_sample_divisor} of total)")
            except Exception as e:
                sp.update(f"(FAILED: {e})")
    else:
        print(f"[Catalog] ⚠️  Warning: Field file not found at {target_file_field}. Skipping.")

    # --- 4. Smart Two-Step SNR Computation -----------------------------------
    formatted_catalog = []
    seen_gc_ae = set()
    total = len(all_binaries)

    full_recompute_count = 0

    with _Spinner(f"Computing SNR (Smart Two-Step) for {total} systems (tobs = {tobs_yr} yr)") as sp:
        with _silence_stdout():
            for idx, b in enumerate(all_binaries, start=1):

                # quick_analytical try
                snr_quick = b.compute_snr_analytical(tobs_yr=tobs_yr, quick_analytical=True, verbose=False)

                if snr_quick > 0.1:
                    snr_final = b.compute_snr_analytical(tobs_yr=tobs_yr, quick_analytical=False, verbose=False)
                    full_recompute_count += 1
                else:
                    snr_final = snr_quick

                b.extra['snr'] = snr_final

                if b.label == "GC" and snr_final < 0.1:
                    ae_tuple = (b.a, b.e)
                    if ae_tuple in seen_gc_ae:
                        continue
                    seen_gc_ae.add(ae_tuple)

                formatted_catalog.append(
                    [b.label, b.m1, b.m2, b.a, b.e, b.Dl, snr_final, b.extra]
                )

                if idx % 50 == 0 or idx == total:
                    sp.update(f"({idx}/{total})")

    # --- 4.5 Print SNR Performance Report ---
    print("\n" + "-" * 64)
    print(f"📊 SNR COMPUTATION REPORT (Smart Two-Step Method)")
    print("-" * 64)
    print(f"  • Total systems evaluated      : {total}")
    print(f"  • Full Harmonic Integrations   : {full_recompute_count}")
    print(f"  • Expensive calls saved        : {total - full_recompute_count}")
    print("-" * 64)

    # --- 5. Evaporated population (optional) --------------------------------
    if include_evaporated:
        with _Spinner(f"Simulating evaporated wide-field binaries ({evap_pct * 100:.1f}%)") as sp:
            with _silence_stdout():
                evap_systems = _run_evaporation_simulation(pct=evap_pct)
            for es in evap_systems:
                es[7]['source_label'] = 'Evaporated_Field'
            formatted_catalog.extend(evap_systems)
            sp.update(f"({len(evap_systems)} systems)")

    # --- 6. Summary & plot --------------------------------------------------
    print("-" * 64)
    print(f"[Catalog] ✅ Successfully generated {len(formatted_catalog)} systems.")
    print("=" * 64)

    if plot:
        _plot_mw_catalog(formatted_catalog)

    return formatted_catalog

# Re-attach after redefinition
LISAeccentric.getMWcatalog = getMWcatalog