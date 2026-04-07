"""
main.py — Bio-Process Digital Twin Entry Point
===============================================
Orchestrates all three analysis modules:
  1. MultivariateAnalyser  (PCA + Correlation + ANOVA)
  2. DeviationDetector     (Isolation Forest + LIR auto-draft)
  3. FluidCalculator       (Reynolds / kLa / Shear + PID control)

Run:
    python main.py

Outputs land in outputs/{multivariate,deviation,fluid}/
"""

import sys
from pathlib import Path

# ── make sure project root is on the path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.synthetic_batch_data import generate_batch_data
from bioprocess import (
    MultivariateAnalyser,
    DeviationDetector,
    BioreactorGeometry,
    FluidState,
    FluidCalculator,
    DOController,
    PIDParameters,
)


def run_module1(df):
    """
    ── MODULE 1: Multivariate Analysis ──────────────────────────────────────
    Correlation heatmap → PCA → ANOVA (pH band vs Titre)
    """
    print("\n" + "▓" * 60)
    print("  MODULE 1 — MULTIVARIATE STATISTICAL ANALYSIS")
    print("▓" * 60)

    analyser = MultivariateAnalyser(output_dir="outputs/multivariate")
    analyser.load_data(df)
    analyser.correlation_heatmap()
    analyser.run_pca(n_components=3)
    analyser.run_anova()


def run_module2(df):
    """
    ── MODULE 2: Deviation Detection ────────────────────────────────────────
    Train on normal batches → predict all → plot a deviated batch → LIR draft
    """
    print("\n" + "▓" * 60)
    print("  MODULE 2 — AI-BASED DEVIATION EARLY DETECTION")
    print("▓" * 60)

    df_normal = df[df["Is_Deviation"] == 0]
    df_all    = df.copy()

    detector = DeviationDetector(output_dir="outputs/deviation")
    detector.fit(df_normal)
    results = detector.predict(df_all)

    # Find the first batch that has predicted deviations
    alarm_batches = (results[results["Predicted_Dev"] == 1]["Batch_ID"].unique())
    target_batch  = alarm_batches[0] if len(alarm_batches) > 0 else df["Batch_ID"].iloc[0]

    detector.plot_anomaly_map(batch_id=target_batch)
    detector.generate_lir(batch_id=target_batch)
    detector.evaluate(df_labelled=df_all)


def run_module3():
    """
    ── MODULE 3: Fluid Dynamics Calculator ──────────────────────────────────
    Engineering report for a 50-L pilot vessel + PID DO controller simulation
    """
    print("\n" + "▓" * 60)
    print("  MODULE 3 — BIOREACTOR FLUID DYNAMICS CALCULATOR")
    print("▓" * 60)

    # --- 50-L pilot bioreactor with Rushton impeller ---
    geo50   = BioreactorGeometry(
        vessel_diameter_m   = 0.38,
        liquid_height_m     = 0.76,
        impeller_diameter_m = 0.15,
        n_impellers         = 2,
        impeller_type       = "Rushton",
    )
    state_nominal = FluidState(agitation_rpm=200, temperature_C=37.0)
    state_high    = FluidState(agitation_rpm=350, temperature_C=37.0)

    calc = FluidCalculator(geo50, output_dir="outputs/fluid")

    print("\n  ── Nominal operating point (200 rpm) ──")
    calc.full_report(state_nominal)

    print("  ── High-agitation scenario (350 rpm) ──")
    calc.full_report(state_high)

    calc.agitation_sweep(rpm_range=(50, 400), base_state=state_nominal)

    # --- PID DO controller simulation ---
    print("\n  ── PID Dissolved Oxygen Control Simulation ──")
    pid = DOController(
        PIDParameters(Kp=2.0, Ki=0.05, Kd=0.5, setpoint=40.0),
        output_dir="outputs/fluid",
    )
    rpms, dos = pid.simulate(
        initial_do=60.0, n_steps=300,
        disturbance_at=150, disturbance_do_drop=25.0,
    )
    pid.plot_response(rpms, dos)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  BIO-PROCESS DIGITAL TWIN & QUALITY CONTROL ENGINE")
    print("  Imperial College London | Computational Bioengineering")
    print("  MSAT Process Optimisation Prototype  v1.0")
    print("█" * 60)

    # ── Generate synthetic 30-batch fed-batch dataset ──────────────
    print("\n[DATA] Generating synthetic fed-batch CHO culture dataset...")
    df = generate_batch_data(n_batches=30, random_seed=42)
    print(f"[DATA] Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[DATA] Batches: {df['Batch_ID'].nunique()} | "
          f"Deviation batches: {df[df['Is_Deviation']==1]['Batch_ID'].nunique()}")

    run_module1(df)
    run_module2(df)
    run_module3()

    print("\n" + "█" * 60)
    print("  ALL MODULES COMPLETE.")
    print("  Output figures → outputs/{multivariate,deviation,fluid}/")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
