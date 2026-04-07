"""
fluid_calculator.py
-------------------
Module 3 of the Bio-Process Digital Twin.

Physics-based bioreactor engineering calculator covering:
  1. Reynolds Number (Re) — turbulence regime classification.
  2. Kolmogorov Eddy Length (λ) — shear stress / cell damage risk.
  3. Oxygen Mass Transfer (kLa) — estimated from empirical correlations.
  4. Power Input (P/V) — specific power per unit volume.
  5. Simple PID Feedback Controller — simulates DO setpoint control.

All numerical results are followed by an engineering interpretation
block that maps the value to practical process engineering decisions
(e.g., "reduce agitation" or "increase sparger flow").

Classes
-------
  BioreactorGeometry   – stores vessel + impeller geometry (data class)
  FluidCalculator      – performs all engineering calculations
  DOController         – discrete-time PID for dissolved oxygen control
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds from Nienow (2006) and ICH Q8 scale-up considerations
# ─────────────────────────────────────────────────────────────────────────────
RE_TURBULENT_THRESHOLD   = 10_000   # Re > 10 000 → fully turbulent
KOLMOGOROV_SAFE_RATIO    = 0.10     # λ/d_cell ≥ 0.10 → cells safe from shear
CHO_CELL_DIAMETER_UM     = 15.0     # typical CHO cell diameter (µm)
KLA_MIN_TARGET           = 5.0      # h⁻¹  — minimum for adequate O₂ supply
KLA_OPTIMAL_TARGET       = 15.0     # h⁻¹
PV_MAX_MAMMALIAN         = 100.0    # W/m³ — upper P/V limit for mammalian cells


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BioreactorGeometry:
    """
    Physical dimensions of a stirred-tank bioreactor (STR).

    All lengths in metres unless stated.
    Defaults approximate a 50-L pilot-scale single-use bioreactor.
    """
    vessel_diameter_m  : float = 0.38     # T
    liquid_height_m    : float = 0.76     # H  (H/T ≈ 2 for tall vessels)
    impeller_diameter_m: float = 0.15     # D  (D/T ≈ 0.4)
    n_impellers        : int   = 2
    impeller_type      : str   = "Rushton" # "Rushton" | "PBT" (pitched blade)

    @property
    def working_volume_m3(self) -> float:
        return math.pi / 4 * self.vessel_diameter_m**2 * self.liquid_height_m

    @property
    def working_volume_L(self) -> float:
        return self.working_volume_m3 * 1000

    # Impeller power number Np (from published correlations)
    @property
    def power_number(self) -> float:
        return 5.0 if self.impeller_type == "Rushton" else 1.3   # PBT


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FluidState:
    """
    Operating conditions at a given time point.
    """
    agitation_rpm : float   = 200.0   # N  (rev per second internally)
    temperature_C : float   = 37.0
    viscosity_Pa_s: float   = 1.2e-3  # µ  ≈ serum-free CHO medium at 37 °C
    density_kg_m3 : float   = 1010.0  # ρ
    gas_flow_vvm  : float   = 0.05    # volume gas / volume liquid / min


# ─────────────────────────────────────────────────────────────────────────────
class FluidCalculator:
    """
    Engineering calculations for stirred-tank bioreactor design and
    scale-up / scale-down studies.

    Example
    -------
    >>> geo   = BioreactorGeometry(vessel_diameter_m=0.38)
    >>> state = FluidState(agitation_rpm=200)
    >>> calc  = FluidCalculator(geo)
    >>> calc.full_report(state)
    """

    def __init__(
        self,
        geometry: BioreactorGeometry,
        output_dir: str = "outputs/fluid",
    ):
        self.geo        = geometry
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────
    # Core calculations
    # ──────────────────────────────────────────────

    def reynolds_number(self, state: FluidState) -> Tuple[float, str]:
        """
        Impeller Reynolds number:  Re = ρ N D² / µ

        Returns
        -------
        (Re, regime_label)
        """
        N  = state.agitation_rpm / 60.0        # rev/s
        Re = (state.density_kg_m3 * N
              * self.geo.impeller_diameter_m**2
              / state.viscosity_Pa_s)

        if Re < 10:
            regime = "Laminar"
        elif Re < 10_000:
            regime = "Transitional"
        else:
            regime = "Turbulent"
        return Re, regime

    def power_input(self, state: FluidState) -> Tuple[float, float]:
        """
        Total impeller power and specific power (P/V):
          P = Np × ρ × N³ × D⁵ × n_impellers

        Returns
        -------
        (P_watts, P_per_V_W_per_m3)
        """
        N   = state.agitation_rpm / 60.0
        Np  = self.geo.power_number
        P   = (Np * state.density_kg_m3
               * N**3
               * self.geo.impeller_diameter_m**5
               * self.geo.n_impellers)
        PV  = P / self.geo.working_volume_m3
        return P, PV

    def kolmogorov_length(self, state: FluidState) -> Tuple[float, float]:
        """
        Kolmogorov microscale of turbulence (Kolmogorov eddy length):
          λ = (ν³ / ε)^(1/4)

        where ν = kinematic viscosity, ε = specific energy dissipation rate.

        Returns
        -------
        (lambda_um, lambda_over_cell_diameter)  — λ in micrometres
        """
        _, PV    = self.power_input(state)
        nu       = state.viscosity_Pa_s / state.density_kg_m3   # m²/s
        epsilon  = PV / state.density_kg_m3                      # m²/s³
        # Avoid division by zero
        if epsilon <= 0:
            return float("inf"), float("inf")
        lambda_m  = (nu**3 / epsilon) ** 0.25
        lambda_um = lambda_m * 1e6
        ratio     = lambda_um / CHO_CELL_DIAMETER_UM
        return lambda_um, ratio

    def kla_estimate(self, state: FluidState) -> float:
        """
        Volumetric oxygen transfer coefficient (kLa) via the
        van't Riet correlation (1979):
          kLa = C × (P/V)^α × v_s^β

        Constants for a coalescing aqueous medium:
          C = 0.026,  α = 0.4,  β = 0.5

        v_s = superficial gas velocity (m/s) from vvm and vessel geometry.

        Returns
        -------
        kLa_per_h : float  (h⁻¹)
        """
        _, PV  = self.power_input(state)
        # Superficial gas velocity
        Q_m3s  = (state.gas_flow_vvm
                  * self.geo.working_volume_m3 / 60.0)          # m³/s
        A_cross = math.pi / 4 * self.geo.vessel_diameter_m**2   # m²
        vs      = Q_m3s / A_cross                                # m/s

        C, alpha, beta = 0.026, 0.4, 0.5
        kLa_per_s = C * (PV ** alpha) * (vs ** beta)
        return kLa_per_s * 3600   # convert to h⁻¹

    # ──────────────────────────────────────────────
    # Agitation sweep
    # ──────────────────────────────────────────────

    def agitation_sweep(
        self,
        rpm_range: Tuple[float, float] = (50, 400),
        n_points: int = 50,
        base_state: Optional[FluidState] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compute Re, kLa, Kolmogorov length and P/V over an agitation sweep
        and produce a 4-panel figure — essential for bioprocess scale-up work.
        """
        state    = base_state or FluidState()
        rpms     = np.linspace(rpm_range[0], rpm_range[1], n_points)
        re_vals, kla_vals, kol_vals, pv_vals = [], [], [], []

        for rpm in rpms:
            s         = FluidState(
                agitation_rpm  = rpm,
                temperature_C  = state.temperature_C,
                viscosity_Pa_s = state.viscosity_Pa_s,
                density_kg_m3  = state.density_kg_m3,
                gas_flow_vvm   = state.gas_flow_vvm,
            )
            re, _     = self.reynolds_number(s)
            kla       = self.kla_estimate(s)
            kol, _    = self.kolmogorov_length(s)
            _, pv     = self.power_input(s)
            re_vals.append(re); kla_vals.append(kla)
            kol_vals.append(kol); pv_vals.append(pv)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()

        datasets = [
            (re_vals,  "Reynolds Number (Re)",  "Re",       "#1A73E8"),
            (kla_vals, "kLa (h⁻¹)",             "kLa",      "#34A853"),
            (kol_vals, "Kolmogorov λ (µm)",      "λ (µm)",   "#EA4335"),
            (pv_vals,  "Specific Power P/V (W/m³)", "P/V",  "#FBBC05"),
        ]
        ref_lines = {
            1: [(KLA_MIN_TARGET,     "Min target", "grey", ":"),
                (KLA_OPTIMAL_TARGET, "Optimal",    "grey", "--")],
            2: [(CHO_CELL_DIAMETER_UM * KOLMOGOROV_SAFE_RATIO,
                 "Safe λ threshold", "grey", "--")],
            3: [(PV_MAX_MAMMALIAN, "Mammalian cell limit", "grey", "--")],
        }
        for i, (vals, title, ylabel, color) in enumerate(datasets):
            axes[i].plot(rpms, vals, color=color, linewidth=2)
            for ref_val, ref_label, rc, rs in ref_lines.get(i, []):
                axes[i].axhline(ref_val, color=rc, linestyle=rs,
                                linewidth=1.2, label=ref_label)
            axes[i].set_title(title); axes[i].set_xlabel("Agitation (rpm)")
            axes[i].set_ylabel(ylabel); axes[i].grid(True, alpha=0.35)
            if ref_lines.get(i):
                axes[i].legend(fontsize=8)

        plt.suptitle(
            f"Bioreactor Engineering Sweep — {self.geo.working_volume_L:.0f} L Vessel\n"
            f"(D={self.geo.impeller_diameter_m*100:.0f} cm, "
            f"{self.geo.impeller_type} impeller × {self.geo.n_impellers})",
            fontsize=12,
        )
        plt.tight_layout()
        path = save_path or str(self.output_dir / "agitation_sweep.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[PLOT] Agitation sweep saved → {path}")

    # ──────────────────────────────────────────────
    # Full console report
    # ──────────────────────────────────────────────

    def full_report(self, state: FluidState) -> dict:
        """
        Print a comprehensive engineering report for a given FluidState,
        including interpretation and recommended actions.
        """
        Re,  regime  = self.reynolds_number(state)
        P,   PV      = self.power_input(state)
        lam, ratio   = self.kolmogorov_length(state)
        kLa          = self.kla_estimate(state)

        header = (
            f"\n{'═'*60}\n"
            f"  BIOREACTOR FLUID DYNAMICS REPORT\n"
            f"  Vessel: {self.geo.working_volume_L:.0f} L  |  "
            f"Impeller: {self.geo.impeller_type} × {self.geo.n_impellers}  |  "
            f"Agitation: {state.agitation_rpm:.0f} rpm\n"
            f"{'═'*60}\n"
        )

        body = (
            f"  Reynolds Number  : {Re:>10,.0f}   [{regime}]\n"
            f"  Power Input P    : {P:>10.3f} W\n"
            f"  Spec. Power P/V  : {PV:>10.2f} W/m³\n"
            f"  Kolmogorov λ     : {lam:>10.2f} µm\n"
            f"  λ / cell diameter: {ratio:>10.3f}\n"
            f"  kLa              : {kLa:>10.2f} h⁻¹\n"
        )

        print(header + body)
        print("─" * 60)
        print("  INTERPRETATION")
        print("─" * 60)

        # ── Reynolds ─────────────────────────────────────────────────
        if Re < RE_TURBULENT_THRESHOLD:
            print(f"  ⚠  Re = {Re:,.0f} → flow is not fully turbulent.\n"
                  "     Poor bulk mixing expected. Increase agitation or\n"
                  "     switch to a lower-drag impeller (e.g., marine propeller).")
        else:
            print(f"  ✓  Re = {Re:,.0f} → fully turbulent. Adequate bulk mixing.")

        # ── Shear / Kolmogorov ────────────────────────────────────────
        if ratio < KOLMOGOROV_SAFE_RATIO:
            print(f"\n  ⚠  λ/d_cell = {ratio:.3f} < {KOLMOGOROV_SAFE_RATIO} SAFETY THRESHOLD.\n"
                  "     Eddies smaller than 10 % of cell diameter → CELL DAMAGE RISK.\n"
                  f"     MSAT Action: Reduce agitation RPM or increase D/T ratio.\n"
                  "     Consider switching from Rushton to PBT impeller (lower Np).")
        else:
            print(f"\n  ✓  λ/d_cell = {ratio:.3f} ≥ {KOLMOGOROV_SAFE_RATIO}. "
                  "Cells safe from turbulent shear.")

        # ── kLa ──────────────────────────────────────────────────────
        if kLa < KLA_MIN_TARGET:
            print(f"\n  ⚠  kLa = {kLa:.2f} h⁻¹ < {KLA_MIN_TARGET} h⁻¹ minimum.\n"
                  "     Oxygen transfer is insufficient for CHO cell culture.\n"
                  "     MSAT Action: Increase sparger flow (vvm) or agitation.\n"
                  "     Consider micro-sparger retrofit.")
        elif kLa < KLA_OPTIMAL_TARGET:
            print(f"\n  ~  kLa = {kLa:.2f} h⁻¹ — marginal; acceptable but not optimal.\n"
                  "     Monitor DO closely during peak growth (day 5–8).")
        else:
            print(f"\n  ✓  kLa = {kLa:.2f} h⁻¹ ≥ {KLA_OPTIMAL_TARGET} h⁻¹ optimal target.")

        # ── P/V ──────────────────────────────────────────────────────
        if PV > PV_MAX_MAMMALIAN:
            print(f"\n  ⚠  P/V = {PV:.1f} W/m³ exceeds {PV_MAX_MAMMALIAN} W/m³ limit for\n"
                  "     mammalian cells. High risk of hydrodynamic cell damage.\n"
                  "     MSAT Action: Reduce agitation immediately.")
        else:
            print(f"\n  ✓  P/V = {PV:.1f} W/m³ within mammalian cell tolerance "
                  f"(< {PV_MAX_MAMMALIAN} W/m³).")

        print("─" * 60 + "\n")

        return {"Re": Re, "regime": regime, "P_W": P, "PV_W_m3": PV,
                "kLa_h": kLa, "kolmogorov_um": lam, "lambda_ratio": ratio}


# ─────────────────────────────────────────────────────────────────────────────
# PID Dissolved Oxygen Controller
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PIDParameters:
    Kp: float = 2.0    # Proportional gain
    Ki: float = 0.05   # Integral gain
    Kd: float = 0.5    # Derivative gain
    setpoint: float = 40.0   # DO setpoint (%)
    output_min: float = 80.0  # min agitation rpm
    output_max: float = 350.0  # max agitation rpm


class DOController:
    """
    Discrete-time PID controller that manipulates agitation RPM to
    maintain a dissolved oxygen (DO) setpoint.

    Simulates the feedback control loop implemented in a bioreactor
    control system (e.g., Biostat B-DCU or Sartorius ambr° platform).

    Example
    -------
    >>> pid   = DOController(PIDParameters(setpoint=40.0))
    >>> rpms, dos = pid.simulate(initial_do=60.0, n_steps=300)
    >>> pid.plot_response(rpms, dos, save_path="outputs/fluid/pid_do.png")
    """

    def __init__(
        self,
        params: PIDParameters,
        output_dir: str = "outputs/fluid",
    ):
        self.params     = params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._integral  = 0.0
        self._prev_err  = 0.0

    def _step(self, do_meas: float, dt: float = 1.0) -> float:
        """Single PID update step. Returns new agitation RPM."""
        error          = self.params.setpoint - do_meas
        self._integral = np.clip(
            self._integral + error * dt,
            -500, 500,   # anti-windup clamp
        )
        derivative = (error - self._prev_err) / dt
        output     = (self.params.Kp * error
                      + self.params.Ki * self._integral
                      + self.params.Kd * derivative)
        self._prev_err = error
        return float(np.clip(output + 200,   # bias at 200 rpm
                             self.params.output_min,
                             self.params.output_max))

    def simulate(
        self,
        initial_do: float = 60.0,
        n_steps: int = 300,
        disturbance_at: Optional[int] = 150,
        disturbance_do_drop: float = 20.0,
    ) -> Tuple[List[float], List[float]]:
        """
        Simulate the DO control loop.

        Parameters
        ----------
        initial_do          : Starting DO (%).
        n_steps             : Number of time steps (each step = 1 min).
        disturbance_at      : Step at which a DO disturbance is injected.
        disturbance_do_drop : Magnitude of DO drop at disturbance (%).

        Returns
        -------
        (rpm_history, do_history)
        """
        # Simple first-order process model: DO responds to agitation with lag
        do        = initial_do
        tau       = 10.0          # process time constant (min)
        gain      = -0.08         # DO change per RPM unit (negative: more agitation → more O₂)
        rpm_history: List[float] = []
        do_history:  List[float] = []

        for step in range(n_steps):
            if disturbance_at and step == disturbance_at:
                do -= disturbance_do_drop   # inject disturbance

            rpm    = self._step(do)
            do_ss  = initial_do + gain * (rpm - 200)   # steady-state model
            do    += (do_ss - do) / tau                 # Euler integration

            rpm_history.append(rpm)
            do_history.append(float(np.clip(do, 0, 100)))

        return rpm_history, do_history

    def plot_response(
        self,
        rpm_history: List[float],
        do_history:  List[float],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot DO response and agitation RPM over time."""
        time = list(range(len(do_history)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

        ax1.plot(time, do_history, color="#1A73E8", linewidth=2, label="DO measured")
        ax1.axhline(self.params.setpoint, color="#EA4335", linestyle="--",
                    linewidth=1.5, label=f"Setpoint ({self.params.setpoint}%)")
        ax1.fill_between(time,
                         self.params.setpoint - 5, self.params.setpoint + 5,
                         alpha=0.10, color="green", label="±5% band")
        ax1.set_ylabel("Dissolved Oxygen (%)")
        ax1.set_title("PID Feedback Control — Dissolved Oxygen Regulation", fontsize=12)
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        ax2.plot(time, rpm_history, color="#FBBC05", linewidth=2, label="Agitation RPM")
        ax2.axhline(self.params.output_max, color="grey",
                    linestyle=":", label=f"Max RPM ({self.params.output_max})")
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Agitation (rpm)")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_path or str(self.output_dir / "pid_do_control.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[PLOT] PID response saved → {path}")

        # ── Interpretation ────────────────────────────────────────────
        do_arr   = np.array(do_history)
        settle   = next((i for i in range(len(do_arr))
                         if abs(do_arr[i] - self.params.setpoint) < 2.0), None)
        overshoot = max(do_arr) - self.params.setpoint
        sse      = np.mean((do_arr[-50:] - self.params.setpoint)**2)

        print("\n" + "─" * 60)
        print("  INTERPRETATION: PID Controller Performance")
        print("─" * 60)
        print(f"  Settling time : {'N/A' if settle is None else f'{settle} min'}"
              f"  (target: < 30 min)")
        print(f"  Overshoot     : {overshoot:.1f}%  (target: < 5%)")
        print(f"  Steady-state MSE: {sse:.4f}  (target: < 1.0)")

        if settle is not None and settle < 30 and overshoot < 5 and sse < 1.0:
            print("  ✓  Controller performs within bioprocess control criteria.")
        else:
            print("  ⚠  Controller tuning suboptimal. Recommended adjustments:")
            if overshoot >= 5:
                print(f"     • Reduce Kp ({self.params.Kp}) → decrease overshoot.")
            if settle is None or settle >= 30:
                print(f"     • Increase Ki ({self.params.Ki}) → speed up integral correction.")
        print("─" * 60 + "\n")
