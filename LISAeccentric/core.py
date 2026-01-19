import numpy as np
import scipy.constants as sciconsts
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union

# ==============================================================================
# 修正：使用相对导入 (Relative Import)
# 只要 core.py 和 GN_modeling 文件夹在同一个包目录下，必须加 "."
# ==============================================================================
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
        """String representation."""
        info = f"<CompactBinary [{self.label}]: M={self.m1:.1f}+{self.m2:.1f}, a={self.a:.2f}AU, e={self.e:.4f}, Dl={self.Dl:.1f}kpc"
        if 'snr' in self.extra:
            info += f", SNR={self.extra['snr']:.2f}"
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
            print(f"[CompactBinary] Parsing error for schema '{schema}': {err}\nData: {data_list}")
            return None

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

    # ==========================================================================
    # MODULE 1: Galactic Nucleus (GN)
    # ==========================================================================
    class _GN_Handler:
        def sample_eccentricities(self, n_samples=5000, max_bh_mass=50, plot=True):
            """Feature 1: Randomly sample N merger eccentricities (LIGO Band 10Hz)."""
            print(f"\n[GN] Sampling {n_samples} merger eccentricities (max_mass={max_bh_mass})...")
            e_samples = GN_BBH.generate_random_merger_eccentricities(n=n_samples, max_bh_mass=max_bh_mass)
            print(f'Sample Mean e (at 10Hz): {np.mean(e_samples):.4e}')
            if plot:
                GN_BBH.plot_ecc_cdf_log(e_list=e_samples)
            return e_samples

        def get_progenitor(self, n_inspect=3) -> List[CompactBinary]:
            """Feature 2: Get Progenitor Population (Initial States)."""
            print(f"\n[GN] Getting {n_inspect} Progenitor Systems...")
            raw_data = GN_BBH.get_random_merger_systems(n=n_inspect)
            objs = []
            for row in raw_data:
                obj = CompactBinary.from_list(row, schema='gn_prog')
                objs.append(obj)
                print(obj)
            return objs

        def get_snapshot(self, rate_gn=2.0, age_ync=6.0e6, n_ync_sys=100, max_bh_mass=50, plot=True) -> List[
            CompactBinary]:
            """Feature 3: Snapshot Generation (LISA Band / Current State)."""
            print(f"\n[GN] Generating Snapshot: Rate={rate_gn}/Myr, YNC Age={age_ync / 1e6} Myr")
            raw_data = GN_BBH.generate_snapshot_population(
                Gamma_rep=rate_gn, ync_age=age_ync, ync_count=n_ync_sys, max_bh_mass=max_bh_mass
            )
            raw_data.sort(key=lambda x: x[6], reverse=True)
            objs = [CompactBinary.from_list(row, schema='snapshot_std') for row in raw_data]
            print(f"Altogether {len(objs)} systems survived.")
            if plot:
                GN_BBH.plot_snapshot_population(raw_data, title="Simulated MW Galactic Nucleus BBH Population")
            return objs

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
        def sample_eccentricities(self, n=5000, channel_name='Incluster', plot=True):
            """Feature 1: Sample eccentricities for GC BBH Mergers (LIGO band)."""
            print(f"\n[GC] Sampling {n} eccentricities (Channel: {channel_name})...")
            e_samples = GC_BBH.generate_ecc_samples_10Hz(channel_name=channel_name, size=n)
            if plot:
                GC_BBH.plot_ecc_cdf(e_samples, label=f"GC {channel_name}")
            return e_samples

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

        def sample_eccentricities(self, n=5000, galaxy_type='MW', plot=True):
            """Feature 1: Merger Eccentricity Sampling (LIGO Band)."""
            print(f"\n[Field] Sampling {n} eccentricities ({galaxy_type})...")
            module = Field_BBH_Elliptical if galaxy_type == 'Elliptical' else Field_BBH
            e_samples = module.generate_eccentricity_samples(size=n)
            print(f'Sample Mean e: {np.mean(e_samples):.4e}')
            if plot:
                module.plot_eccentricity_cdf(e_samples, label=f"Field {galaxy_type}")
            return e_samples

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
        def compute_waveform_system(self, m1=None, m2=None, a_au=None, e=None, Dl_kpc=None,
                                    tobs=0.5, theta=np.pi / 4, phi=np.pi / 4,
                                    l0=0, ts=None, PN_orbit=3, PN_reaction=2,
                                    N=50, max_memory_GB=16.0, verbose=True, plot=True,
                                    system=None):
            """
            Generate GW waveform.
            """
            # --- Input Switching Logic ---
            if system is not None:
                # User used 'system=obj' kwarg
                m1, m2, a_au, e, Dl_kpc = system.m1, system.m2, system.a, system.e, system.Dl
                label = system.label
            elif isinstance(m1, CompactBinary):
                # User passed object as first arg
                system = m1
                m1, m2, a_au, e, Dl_kpc = system.m1, system.m2, system.a, system.e, system.Dl
                label = system.label
            else:
                label = "Manual_Input"

            # --- Validation ---
            if m1 is None or m2 is None or a_au is None or e is None:
                print("Error: Missing waveform parameters.")
                return None

            print(f"\n[Waveform] Generating Waveform (m={m1}+{m2}, e={e:.3f})...")
            M_kg = (m1 + m2) * 1.98840987e30
            f_orb = np.sqrt(sciconsts.G * M_kg / (4 * sciconsts.pi ** 2 * np.power(a_au * sciconsts.au, 3.0)))

            wf = PN_waveform.eccGW_waveform(
                f_orb, e, tobs, m1, m2, theta, phi, Dl_kpc,
                l0=l0, ts=ts, PN_orbit=PN_orbit, PN_reaction=PN_reaction,
                N=N, max_memory_GB=max_memory_GB, verbose=verbose
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

        def compute_LISA_response(self, timelist, hplus, hcross,
                                  theta_sky=np.pi / 4, phi_sky=np.pi / 4, psi_sky=np.pi / 4,
                                  t0=0, kappa=0.0, lamb=0.0, mode='interp', plot=True):
            """
            Compute LISA detector response.
            """
            print(f"[Waveform] Computing LISA Response (Mode: {mode})...")
            response = PN_waveform.compute_LISA_response(
                timelist, hplus, hcross, theta_sky, phi_sky, psi_sky,
                t0=t0, kappa=kappa, lamb=lamb, mode=mode
            )
            if plot:
                plt.figure(figsize=(8, 6), dpi=100)
                plt.plot(response[0], response[1], color='BLUE', label='Response')
                plt.xlabel("t [s]", fontsize=14)
                plt.ylabel("Detector Response")
                plt.legend()
                plt.show()
            return response

        def compute_inner_product(self, timelist, h1, h2, phase_difference=0):
            """Calculate inner product of two waveforms."""
            if len(timelist) < 2: return 0.0
            fs = 1.0 / (timelist[1] - timelist[0])
            val = PN_waveform.inner_product(fs, h1, h2, phase_difference)
            print(f"[Analysis] Inner Product = {val:.4e}")
            return val

        def compute_snr_numerical(self, timelist, hplus, phase_difference=0):
            """Estimate SNR using Numerical Inner Product."""
            val = self.compute_inner_product(timelist, hplus, hplus, phase_difference)
            snr_num = np.sqrt(val)
            print(f"[Analysis] SNR_numerical = {snr_num:.4f}")
            return snr_num

        def compute_snr_analytical(self, m1=None, m2=None, a_au=None, e=None, Dl_kpc=None, tobs=0.5, system=None):
            """Estimate Sky-Averaged SNR (Analytical). Supports auto-switching input."""
            if system is not None:
                m1, m2, a_au, e, Dl_kpc = system.m1, system.m2, system.a, system.e, system.Dl
            elif isinstance(m1, CompactBinary):
                system = m1
                m1, m2, a_au, e, Dl_kpc = system.m1, system.m2, system.a, system.e, system.Dl

            if m1 is None: return 0.0
            snr = PN_waveform.SNR(m1, m2, a_au, e, Dl_kpc, tobs)
            print(f"[Analysis] SNR_analytical = {snr:.4f}")
            return snr

        def compute_merger_time(self, m1=None, m2=None, a_au=None, e=None, system=None):
            """Estimate Merger Timescale. Supports auto-switching input."""
            if system is not None:
                m1, m2, a_au, e = system.m1, system.m2, system.a, system.e
            elif isinstance(m1, CompactBinary):
                system = m1
                m1, m2, a_au, e = system.m1, system.m2, system.a, system.e

            if m1 is None: return np.inf
            t_merger = PN_waveform.tmerger_integral(m1, m2, a_au, e)
            print(f"[Analysis] T_merger = {t_merger:.4e} yrs")
            return t_merger

        def evolve_orbit(self, m1=None, m2=None, a0_au=None, e0=None, delta_t_years=None, system=None):
            """Estimate orbital parameters after given time. Supports auto-switching input."""
            # Handle arg shift if user passes system as first arg
            if isinstance(m1, CompactBinary):
                system = m1
                # If first arg is system, 2nd arg might be delta_t_years if positional
                if m2 is not None and delta_t_years is None:
                    delta_t_years = m2

            if system is not None:
                m1, m2, a0_au, e0 = system.m1, system.m2, system.a, system.e

            if m1 is None or delta_t_years is None: return None, None

            a_new, e_new = PN_waveform.solve_ae_after_time(m1, m2, a0_au, e0, delta_t_years)
            print(f"[Analysis] After {delta_t_years} yrs: a = {a_new:.4e} au, e = {e_new:.6f}")
            return a_new, e_new

        def compute_characteristic_strain_single(self, m1=None, m2=None, a_au=None, e=None, Dl_kpc=None,
                                                 tobs_years=0.5, plot=True, system=None):
            """Compute h_c for single system. Supports auto-switching input."""
            if system is not None:
                m1, m2, a_au, e, Dl_kpc = system.m1, system.m2, system.a, system.e, system.Dl
                label = system.label
            elif isinstance(m1, CompactBinary):
                system = m1
                m1, m2, a_au, e, Dl_kpc = system.m1, system.m2, system.a, system.e, system.Dl
                label = system.label
            else:
                label = "Manual_Input"

            if m1 is None: return None
            print(f"[Analysis] Computing h_c for {label} (Tobs={tobs_years} yr)...")
            res = hc_cal.calculate_single_system(
                m1=m1, m2=m2, a=a_au, e=e, Dl=Dl_kpc, tobs_years=tobs_years
            )
            if plot:
                hc_cal.plot_single_system_results(res)
            return res

        def run_population_strain_analysis(self, binary_list: List[CompactBinary], tobs_years=4.0, plot=True):
            """
            Compute characteristic strain for a list of CompactBinary objects.
            Takes a list of classes as input.
            """
            print(f"\n[Analysis] Computing h_c for Batch (N={len(binary_list)})...")
            snapshot_data_list = [b.to_list(schema='snapshot_std') for b in binary_list]
            batch_results = hc_cal.process_population_batch(snapshot_data_list, tobs_years=tobs_years)
            if plot:
                hc_cal.plot_simulation_results(batch_results)
            return batch_results