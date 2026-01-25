import numpy as np
import scipy.constants as sciconsts
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
import shutil  # Added for file operations
import os
import importlib
import builtins     # For print overriding
import sys          # For stdout control
import contextlib   # For redirecting stdout
import warnings     # For warning control
from scipy.optimize import newton

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
    # 调试信息：如果相对导入失败，尝试打印详细路径信息
    import os

    print(f"CRITICAL ERROR: Package import failed in {__file__}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Error Details: {e}")
    # 再次抛出错误，强制终止程序，避免后面的 NameError
    raise e

# ==============================================================================
# Global Output Control (Dual Switches)
# ==============================================================================
_VERBOSE = True
_SHOW_WARNINGS = True


def set_output_control(verbose: bool = True, show_warnings: bool = True):
    """
    设置全局输出控制。

    Args:
        verbose (bool): 是否显示 print 输出 (stdout)。
        show_warnings (bool): 是否显示 RuntimeWarning 等警告信息 (stderr)。
    """
    global _VERBOSE, _SHOW_WARNINGS
    _VERBOSE = verbose
    _SHOW_WARNINGS = show_warnings
def set_verbose(verbose: bool):
    """
    [兼容旧接口] 简易设置函数。
    如果设置为 False，则同时关闭打印和警告。
    """
    set_output_control(verbose=verbose, show_warnings=verbose)
def print(*args, **kwargs):
    """
    覆盖本模块内的 print，根据 _VERBOSE 开关决定是否执行。
    """
    if _VERBOSE:
        builtins.print(*args, **kwargs)

def mute_if_global_verbose_false(func):
    """
    装饰器：根据全局开关 _VERBOSE 和 _SHOW_WARNINGS，
    决定是否在函数执行期间屏蔽 stdout 或过滤 warnings。
    (名称保持不变，但在内部集成了双开关逻辑)
    """

    def wrapper(*args, **kwargs):
        # 使用 ExitStack 灵活管理多个上下文管理器
        with contextlib.ExitStack() as stack:

            # 1. 如果关闭 Verbose，将标准输出重定向到空设备 (屏蔽外部库的 print)
            if not _VERBOSE:
                fnull = stack.enter_context(open(os.devnull, 'w'))
                stack.enter_context(contextlib.redirect_stdout(fnull))

            # 2. 如果关闭 Warnings，捕获并忽略所有警告
            if not _SHOW_WARNINGS:
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("ignore")

            # 执行原函数
            return func(*args, **kwargs)

    return wrapper

# ==============================================================================
# 核心数据类: CompactBinary
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
    # 核心参数 (必填，不允许为空)
    m1: float
    m2: float
    a: float
    e: float
    Dl: float
    label: str

    # 扩展参数
    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        """String representation (Updated to print all extra params)."""
        # 1. 基础物理量
        info = f"<CompactBinary [{self.label}]: M={self.m1:.1f}+{self.m2:.1f} m_sun, a={self.a:.3e}AU, e={self.e:.4f}, Dl={self.Dl:.1f}kpc"

        # 2. 动态遍历并添加 extra 中的所有信息
        if self.extra:
            extra_str_list = []
            for k, v in self.extra.items():
                # 针对浮点数做自适应格式化，避免输出过长
                if isinstance(v, float):
                    # 如果数值很大或很小，用科学计数法；否则保留3位小数
                    val_str = f"{v:.2e}" if (abs(v) < 1e-2 or abs(v) > 1e4) and v != 0 else f"{v:.3f}"
                else:
                    val_str = str(v)
                extra_str_list.append(f"{k}={val_str}")

            # 将 extra 信息拼接在后面
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
    def compute_snr_analytical(self, tobs_yr, quick_analytical=False, verbose=True):
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

        # 调用 hc_cal.calculate_evolving_system
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
# 主接口类: LISAeccentric
# ==============================================================================
class LISAeccentric:
    """
    LISAeccentric Unified Interface.
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
            def sample_eccentricities(self, n_samples=5000, max_bh_mass=50, plot=True):
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

                # 数据来源：GN_BBH.get_random_merger_systems
                # 格式: [m1, m2, a1, e1, e2, i_init, a2, afin, efin, t_final]
                raw_data = GN_BBH.get_random_merger_systems(n=n_inspect)

                objs = []
                for row in raw_data:
                    # 1. 实例化基础对象 (必填项)
                    # GN 核系统的距离默认设为 8.0 kpc
                    obj = CompactBinary(
                        m1=float(row[0]),
                        m2=float(row[1]),
                        a=float(row[2]),
                        e=float(row[3]),
                        Dl=8.0,
                        label="GN_Progenitor"
                    )

                    # 2. 将额外信息存入 extra 字典 (而不是 append)
                    obj.extra = {
                        'e2_init': float(row[4]),  # 新增: initial e2
                        'i_init_rad': float(row[5])/180*pi,  # 新增: initial inclination
                        'a2_init': float(row[6]),  # 新增: initial a2
                        'a_final': float(row[7]),  # 新增: final a1
                        'e_final': float(row[8]),  # 新增: final e1
                        'lifetime_yr': float(row[9])
                    }

                    objs.append(obj)
                    print(obj)  # 会打印出 extra 中的信息
                return objs

            @mute_if_global_verbose_false
            def get_snapshot(self, rate_gn=2.0, age_ync=6.0e6, n_ync_sys=100, max_bh_mass=50, plot=True) -> List[
                CompactBinary]:
                """Feature 3: Snapshot Generation (LISA Band / Current State)."""
                print(f"\n[GN] Generating Snapshot: Rate={rate_gn}/Myr, YNC Age={age_ync / 1e6} Myr")

                # 数据来源: GN_BBH.generate_snapshot_population
                # 格式: [label, dist, a, e, i, m1, m2, snr]
                raw_data = GN_BBH.generate_snapshot_population(
                    Gamma_rep=rate_gn, ync_age=age_ync, ync_count=n_ync_sys, max_bh_mass=max_bh_mass
                )

                # 按 SNR 排序 (索引 7)
                raw_data.sort(key=lambda x: x[7], reverse=True)

                objs = []
                for row in raw_data:
                    # 1. 实例化基础对象
                    obj = CompactBinary(
                        label=str(row[0]),
                        Dl=float(row[1]),
                        a=float(row[2]),
                        e=float(row[3]),
                        m1=float(row[5]),
                        m2=float(row[6])
                    )

                    # 2. 将额外信息存入 extra 字典
                    obj.extra = {
                        'inclination_rad': float(row[4])/180*pi,  # 新增: 当前时刻的倾角 i
                        'snr': float(row[7])
                    }

                    objs.append(obj)

                print(f"Altogether {len(objs)} systems survived.")
                if plot:
                    # 注意 plot 函数可能需要根据 raw_data 的新列结构进行微调，
                    # 但这里只负责传递数据
                    GN_BBH.plot_snapshot_population(raw_data, title="Simulated MW Galactic Nucleus BBH Population")
                return objs

            @mute_if_global_verbose_false
            def calculate_fpeak_frequency(self, m1=None, m2=None, a_au=None, e=None, system=None):
                """
                Extra Feature: Calculate GW Peak Frequency.
                Supports parameter input OR CompactBinary object input.
                """
                # 自动切换逻辑
                if system is not None:
                    m1, m2, a_au, e = system.m1, system.m2, system.a, system.e
                elif isinstance(m1, CompactBinary):
                    # 如果用户把对象传给了第一个参数
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
        def get_snapshot(self, mode='10_realizations', n_random=500, plot=True) -> List[CompactBinary]:
            """Feature 2: Get BBH parameters from GC snapshots."""
            print(f"\n[GC] Getting Snapshot (Mode: {mode})...")
            if mode == '10_realizations':
                raw_data = GC_BBH.get_full_10_realizations()
                title = "BBHs in MW GCs (10 Realizations)"
            elif mode == 'single':
                raw_data = GC_BBH.get_single_mw_realization()
                title = "Single MW Realization"
            elif mode == 'random':
                raw_data = GC_BBH.get_random_systems(n_random)
                title = f"Randomly Selected {n_random} Systems"
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

            【必填参数】:
            :param m1_msun:   Mass 1 [Solar Mass]
            :param m2_msun:   Mass 2 [Solar Mass]
            :param a_au:      根据 input_mode 不同，此参数的物理含义不同：
                              - input_mode='a_au':        输入为半长轴 [AU] (默认)
                              - input_mode='forb_Hz':     输入为轨道频率 f_orb [Hz] (注意：此时参数名虽叫a_au，但值应填频率)
                              - input_mode='fangular_Hz': 输入为角/峰值频率 [Hz]
            :param e:         Eccentricity
            :param Dl_kpc:    Luminosity Distance [kpc]
            :param tobs_yr:   Observation time [years]
            :param input_mode: Mode selector ('a_au', 'forb_Hz', 'fangular_Hz')

            【采样控制参数】:
            :param ts:              Time step [seconds].
            :param points_per_peak: Sampling points per orbital peak.
            """
            # 1. 基础物理量转换 (G=c=1 units / seconds)
            m1 = m1_msun * m_sun
            m2 = m2_msun * m_sun
            M_total = m1 + m2
            Dl = Dl_kpc * 1e3 * pc
            tobs = tobs_yr * years

            label = f"Mode_{input_mode}"

            f_orb = 0.0
            a_au_display = 0.0  # 仅用于打印显示，反推出来的 a

            # === 根据 input_mode 解释 a_au 参数 ===
            if input_mode == 'a_au':
                # --- 模式 A: 标准模式，a_au 即为 AU ---
                real_a_au = a_au
                a = real_a_au * AU
                if a > 0:
                    f_orb = np.sqrt(M_total / (4 * pi ** 2 * np.power(a, 3.0)))
                a_au_display = real_a_au

            elif input_mode == 'forb_Hz':
                # --- 模式 B: 输入为轨道频率 (Hz) ---
                # 此时 a_au 变量里存的是 Hz
                f_orb = a_au

                # 反推 a 用于显示
                if f_orb > 0:
                    a_s = np.power(M_total / ((2 * np.pi * f_orb) ** 2), 1.0 / 3.0)
                    a_au_display = a_s / AU

            elif input_mode == 'fangular_Hz':
                # --- 模式 C: 输入为角频率/峰值频率 (Hz), 求解 f_orb ---
                # 此时 a_au 变量里存的是 Hz
                target_f0 = a_au

                # 辅助函数: f -> a (G=c=1)
                def get_a0_from_f00(f_val, M):
                    return np.power(M / ((2 * np.pi * f_val) ** 2), 1.0 / 3.0)

                # 辅助函数: 计算频移 delta f
                def deltafvalue(a, e_in, M):
                    n = np.power(a, -3 / 2) * np.sqrt(M)
                    Porb = 2 * pi / n
                    # Hinder+10 近似公式
                    return 6 * np.power(2 * pi, 2 / 3) / (1 - e_in * e_in) * np.power(M, 2 / 3) * np.power(Porb, -5 / 3)

                # 求解残差函数
                def frequency_residual(f00_guess):
                    if f00_guess <= 0: return 1e6  # 保护
                    a_guess = get_a0_from_f00(f00_guess, M_total)
                    df = deltafvalue(a_guess, e, M_total)
                    f0_calc = f00_guess + df / 2.0
                    return f0_calc - target_f0

                try:
                    # 使用 Newton 法求解 f_orb (initial guess = target_f0)
                    f_orb = newton(frequency_residual, x0=target_f0, tol=1e-7, maxiter=50)

                    # 反推 a 用于显示
                    a_s = get_a0_from_f00(f_orb, M_total)
                    a_au_display = a_s / AU
                except Exception as err:
                    print(f"[Error] Failed to solve f_orb from fangular_Hz: {err}")
                    return None
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}. Use 'a_au', 'forb_Hz', or 'fangular_Hz'.")

            # --- 逻辑提示 ---
            if verbose:
                if ts is not None:
                    sampling_mode = f"Fixed TimeStep (ts={ts} s)"
                else:
                    sampling_mode = f"Adaptive Sampling (N={points_per_peak}/peak)"

                print(f"\n[Waveform] Generating Waveform...")
                print(f"           Mode:   {input_mode}")
                print(f"           Params: m={m1_msun}+{m2_msun} Msun, e={e:.3f}")
                # 显示原始输入值
                print(f"           Input:  {a_au:.4e} ({input_mode})")
                # 显示推导出的物理量
                print(f"           Derived: f_orb={f_orb:.4e} Hz, a~={a_au_display:.4f} AU")

            # 调用底层库
            wf = PN_waveform.eccGW_waveform(
                f_orb, e, tobs, m1, m2, theta, phi, Dl,
                l0=initial_orbital_phase,
                ts=ts,  # 优先参数
                N=points_per_peak,  # 次要参数 (若 ts 存在则被屏蔽)
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
        @mute_if_global_verbose_false
        def compute_LISA_response(self, dt_sample_sec, hplus, hcross,
                                  theta_sky=np.pi / 4, phi_sky=np.pi / 4, psi_sky=np.pi / 4,
                                  timeshift_sec=0.0,  # <--- 改名: 明确单位为秒
                                  kappa=0.0, lamb=0.0, mode='interp', plot=True):
            """
            Compute LISA detector response.

            :param dt_sample_sec: Sampling interval [seconds] (Scalar).
                                  Must match the resolution of hplus/hcross.
            :param timeshift_sec: Time shift [seconds]. Shifts the signal in time domain.
                                  (e.g., propagation delay).
            """
            # 1. 长度校验
            n_points = len(hplus)
            if len(hcross) != n_points:
                raise ValueError(f"[Error] Waveform length mismatch: hplus={n_points}, hcross={len(hcross)}")

            print(f"[Waveform] Computing LISA Response (dt={dt_sample_sec:.4e} s, shift={timeshift_sec} s)...")

            # 2. 内部重建标准相对时间轴 (Standard Time Axis starting from 0)
            # timelist = 0, dt, 2dt, ...
            timelist = np.arange(n_points) * dt_sample_sec

            # 3. 调用底层计算
            # 注意：将带单位的 timeshift_sec 传给底层 (假设底层参数仍叫 t0)
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
            # 1. 长度检查
            if len(h1) != len(h2):
                raise ValueError(f"[Error] Inner product length mismatch: {len(h1)} vs {len(h2)}")

            if len(h1) < 2: return 0.0

            # 2. 计算采样率 fs [Hz]
            fs = 1.0 / dt_sample_sec

            # 3. 调用底层
            val = PN_waveform.inner_product(fs, h1, h2, phase_difference)
            print(f"[Analysis] Inner Product = {val:.4e}")
            return val

        @mute_if_global_verbose_false
        def compute_snr_numerical(self, dt_sample_sec, strainlist):
            """
            Estimate SNR using Numerical Inner Product.

            :param dt_sample_sec: Sampling interval [seconds] (Scalar).
            """
            # 复用 inner_product
            val = self.compute_inner_product(dt_sample_sec, strainlist, strainlist)
            snr_num = np.sqrt(val)
            print(f"[Analysis] SNR_numerical = {snr_num:.4f}")
            return snr_num

        @mute_if_global_verbose_false
        def compute_snr_analytical(self, m1_msun, m2_msun, a_au, e, Dl_kpc, tobs_yr, quick_analytical=False):
            """
            Estimate Sky-Averaged SNR (Analytical).
            必填: m1_msun, m2_msun, a_au, e, Dl_kpc, tobs_yr
            可选: quick_analytical (bool) - 若为 True，使用快速几何近似计算。
            """
            # 1. 基础单位转换 (转为几何单位 G=c=1, 时间单位: 秒)
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

                # --- 优化逻辑: 先用快速下限判断，避免不必要的积分 ---
                try:
                    # 1. 先算合并时间下限 (Fast Check)
                    t_lower = PN_waveform.tmerger_lower(m1_s, m2_s, a_s, e)
                    #print('!',t_lower)
                    # 2. 只有当下限小于观测时间时，才需要算精确积分
                    if t_lower <= tobs_s:
                        t_real = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, e)

                        if t_real <= tobs_s:
                            print(
                                f"[Warning] System evolves too fast! tmerger ({t_real:.2e} s) < tobs ({tobs_s:.2e} s).")
                            print(f"Approximation inaccurate. Adjusting tobs to : {t_real:.2e} s")
                            used_tobs = t_real

                except AttributeError:
                    # 如果后端 PN_waveform 没有实现 tmerger_lower，则回退到直接算积分
                    try:
                        t_real = PN_waveform.tmerger_integral(m1_s, m2_s, a_s, e)
                        if t_real <= tobs_s:
                            used_tobs = t_real
                    except Exception:
                        pass
                except Exception as e:
                    # 其他计算错误(如 e=1)
                    pass

                # --- 3. 计算 SNR (几何近似) ---
                rp_s = a_s * (1 - e)
                if rp_s <= 0: return 0.0

                # 峰值频率
                term_f = (m1_s + m2_s) / (4 * pi * pi * np.power(rp_s, 3.0))
                f0max = 2 * np.sqrt(term_f)

                # 峰值幅度
                h0max = np.sqrt(32 / 5) * m1_s * m2_s / (Dl_s * a_s * (1 - e))

                # 噪声水平
                Sn_val = PN_waveform.S_n_lisa(f0max)

                if Sn_val <= 0:
                    snr = 0.0
                else:
                    sqrtsnf = np.sqrt(Sn_val)
                    # 使用修正后的 used_tobs
                    snr = h0max / sqrtsnf * np.sqrt(used_tobs * np.power(1 - e, 1.5))

                print(f"[Analysis] SNR_analytical (Quick) = {snr:.4f}")
                return snr
            else:
                # === Full Numerical Integration ===
                snr = PN_waveform.SNR(m1_s, m2_s, a_s, e, Dl_s, tobs_s)
                print(f"[Analysis] SNR_analytical = {snr:.4f}")
                return snr
        @mute_if_global_verbose_false
        def compute_merger_time(self, m1_msun, m2_msun, a0_au, e0):
            """
            Estimate Merger Timescale.
            必填: m1_msun, m2_msun, a_au, e
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
            必填: m1_msun, m2_msun, a0_au, e0, delta_t_yr
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
            必填: m1_msun, m2_msun, a_au, e, Dl_kpc, tobs_yr
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
        def run_population_strain_analysis(self, binary_list: List[Any], tobs_yr, plot=True):
            """
            批处理计算 h_c。
            必填: binary_list, tobs_yr
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
        def __init__(self):
            # 定位 CSV 文件路径
            # core.py 和 LISA_noise_ASD.csv 在同一级目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.noise_file_path = os.path.join(current_dir, 'LISA_noise_ASD.csv')
            self.base_backup_name = 'LISA_noise_ASD_original'

        def generate_noise_data(self, model='N2A5', f_min=1e-6, f_max=1.0, n_points=3000):
            """
            Generates LISA Noise ASD based on the selected model.
            integrated with Log-Log extrapolation for better physical accuracy.
            """
            # 1. 生成目标频率网格
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
                    # 1. 读取并清洗数据
                    data = np.loadtxt(source_path, delimiter=',')
                    # 若文件包含表头，解开下面注释:
                    # data = np.loadtxt(source_path, delimiter=',', skiprows=1)

                    data = data[data[:, 0].argsort()]
                    f_ref = data[:, 0]
                    asd_ref = data[:, 1]

                    mask = (f_ref > 0) & (asd_ref > 0)
                    f_ref = f_ref[mask]
                    asd_ref = asd_ref[mask]

                    # 2. 准备 Log-Log 空间数据
                    log_f_ref = np.log10(f_ref)
                    log_asd_ref = np.log10(asd_ref)

                    # 3. 计算低频端斜率 (Low-f Slope)
                    # slope = (y2 - y1) / (x2 - x1)
                    if len(log_f_ref) >= 2:
                        slope_low = (log_asd_ref[1] - log_asd_ref[0]) / (log_f_ref[1] - log_f_ref[0])
                    else:
                        slope_low = 0  # Fallback

                    # 4. 目标 Log 频率
                    log_f_target = np.log10(flist)

                    # 5. 执行插值 (左侧设为 NaN 以便后续处理，右侧设为 0.0 即 ASD=1.0)
                    # 注意：right=0.0 意味着 log(ASD)=0 -> ASD=1.0
                    log_asd_target = np.interp(
                        log_f_target,
                        log_f_ref,
                        log_asd_ref,
                        left=np.nan,
                        right=0.0
                    )

                    # 6. 处理低频 NaN (Power-law Extrapolation)
                    # y = y0 + slope * (x - x0)
                    nan_mask = np.isnan(log_asd_target)
                    if np.any(nan_mask):
                        # x0 = log_f_ref[0], y0 = log_asd_ref[0]
                        log_asd_target[nan_mask] = log_asd_ref[0] + \
                                                   slope_low * (log_f_target[nan_mask] - log_f_ref[0])

                    # 7. 还原回线性空间
                    asd_final = np.power(10.0, log_asd_target)

                    return flist, asd_final

                except Exception as e:
                    print(f"[Noise] Error processing official file: {e}")
                    # 出错时返回默认值
                    return self.generate_noise_data(model='N2A5', f_min=f_min, f_max=f_max, n_points=n_points)

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
                # 1. 直接读取 CSV
                # 尝试两种读取方式，防止 header 导致的错误
                try:
                    data = np.loadtxt(self.noise_file_path, delimiter=',')
                except ValueError:
                    data = np.loadtxt(self.noise_file_path, delimiter=',', skiprows=1)

                # 2. 预处理：按频率排序
                data = data[data[:, 0].argsort()]
                f_data = data[:, 0]
                asd_data = data[:, 1]

                # 过滤非正值（防止 log 报错）
                mask = (f_data > 0) & (asd_data > 0)
                f_data = f_data[mask]
                asd_data = asd_data[mask]

                if len(f_data) < 2:
                    print("[Noise] Warning: Not enough valid points in noise file.")
                    return

                # 3. 准备数据 (Log-Log 空间)
                log_f = np.log10(f_data)
                log_asd = np.log10(asd_data)

                # 计算低频斜率 (用于外推)
                # Slope = dy / dx
                low_f_slope = (log_asd[1] - log_asd[0]) / (log_f[1] - log_f[0])

                # 4. 构建字典
                # 注意：必须包含 'use_file': True，否则 PN_waveform 会忽略这些数据
                noise_dict = {
                    'f_min': f_data[0],
                    'f_max': f_data[-1],
                    'log_f': log_f,
                    'log_asd': log_asd,
                    'low_f_slope': low_f_slope,
                    'log_f_0': log_f[0],
                    'log_asd_0': log_asd[0],
                    'use_file': True  # <--- 关键修复：PN_waveform 原生加载器缺少此键
                }

                # 5. 强制注入
                # 直接修改 imported module 的全局变量
                PN_waveform._LISA_NOISE_DATA = noise_dict
                print(f"[Noise] Force-injected updated noise profile (Points: {len(f_data)})")
                print(f"        Slope: {low_f_slope:.4f}, f_range: [{f_data[0]:.1e}, {f_data[-1]:.1e}] Hz")

            except Exception as e:
                print(f"[Noise] Warning: Data injection failed: {e}")

        @mute_if_global_verbose_false
        def _reload_dependencies(self):
            """
            Reload backend modules and inject data.
            """
            print("[Noise] Auto-reloading backend modules...")
            try:
                # 1. 模块重载
                importlib.reload(GN_BBH)
                importlib.reload(Field_BBH)
                importlib.reload(Field_BBH_Elliptical)
                importlib.reload(hc_cal)
                importlib.reload(PN_waveform)

                # 2. 注入数据
                self._inject_noise_data()

                print("[Noise] Modules reloaded & Data injected.")
            except Exception as e:
                print(f"[Noise] Warning: Module auto-reload failed. Error: {e}")

        @mute_if_global_verbose_false
        def update_noise_curve(self, data_list):
            """
            更新噪声曲线文件，并自动备份旧文件。
            input: data_list = [flist, ASDlist]
            """
            if len(data_list) != 2:
                print("Error: Input must be [flist, ASDlist].")
                return

            flist, asdlist = data_list[0], data_list[1]
            abs_path = os.path.abspath(self.noise_file_path)

            # 1. 备份
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

            # 2. 写入
            try:
                # 确保 flist 递增排序，这对于后续插值很重要
                sort_idx = np.argsort(flist)
                flist = flist[sort_idx]
                asdlist = asdlist[sort_idx]

                data_to_save = np.column_stack((flist, asdlist))
                np.savetxt(self.noise_file_path, data_to_save, delimiter=',', header='f,ASD', comments='')
                print(f"[Noise] Updated noise file at: {os.path.basename(abs_path)}")

                # 3. 重新加载并注入
                self._reload_dependencies()

            except Exception as e:
                print(f"[Noise] Error writing file: {e}")

        @mute_if_global_verbose_false
        def recover_noise_curve(self, version=None):
            """恢复噪声曲线文件。"""
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
            except Exception as e:
                print(f"[Noise] Error recovering file: {e}")

        @mute_if_global_verbose_false
        def clean_backups(self):
            target_dir = os.path.dirname(self.noise_file_path)
            print(f"[Noise] Cleaning backup files...")
            count = 0
            try:
                for filename in os.listdir(target_dir):
                    if filename.startswith(self.base_backup_name) and filename.endswith(".csv"):
                        os.remove(os.path.join(target_dir, filename))
                        count += 1
            except Exception as e:
                print(f"[Noise] Error during cleaning: {e}")
            print(f"[Noise] Removed {count} backup file(s).")

        @mute_if_global_verbose_false
        def get_noise_curve(self, plot=True):
            if not os.path.exists(self.noise_file_path):
                print(f"[Noise] Error: File not found.")
                return None
            try:
                # 尝试读取，兼容有无 header
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