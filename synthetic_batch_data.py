"""
synthetic_batch_data.py
-----------------------
Generates realistic fed-batch bioreactor time-series data.
Variables are scaled to reflect real CHO cell culture conditions.

Variables
---------
  Inputs  : pH, Temperature (°C), DO (Dissolved Oxygen, %),
            Agitation (rpm), Feed_Rate (L/h)
  Outputs : Cell_Viability (%), Titre (g/L), Lactate (g/L),
            Glucose (g/L), VCD (Viable Cell Density, 1e6 cells/mL)

Deviation scenarios are injected at fixed time points to simulate
real process deviations (pH drift, DO crash) for anomaly detection.
"""

import numpy as np
import pandas as pd


def generate_batch_data(
    n_batches: int = 30,
    time_points: int = 144,       # 12 days × 12 samples/day (2-hr interval)
    random_seed: int = 42,
    deviation_fraction: float = 0.15,
) -> pd.DataFrame:
    """
    Simulate fed-batch CHO cell culture data.

    Parameters
    ----------
    n_batches          : Total number of batches to generate.
    time_points        : Number of time points per batch.
    random_seed        : NumPy random seed for reproducibility.
    deviation_fraction : Fraction of batches that contain deviations.

    Returns
    -------
    pd.DataFrame with columns:
        Batch_ID, Time_h, pH, Temperature, DO, Agitation,
        Feed_Rate, VCD, Cell_Viability, Titre, Lactate,
        Glucose, Is_Deviation
    """
    rng = np.random.default_rng(random_seed)
    records = []
    n_deviation_batches = int(n_batches * deviation_fraction)
    deviation_ids = rng.choice(n_batches, n_deviation_batches, replace=False)

    time_h = np.linspace(0, 240, time_points)   # 0 – 240 h

    for batch_id in range(n_batches):
        is_dev = batch_id in deviation_ids
        dev_start = rng.integers(60, 160) if is_dev else None

        # ── Gaussian noise scale ──────────────────────────────────────
        ph_noise      = rng.normal(0, 0.02, time_points)
        temp_noise    = rng.normal(0, 0.15, time_points)
        do_noise      = rng.normal(0, 1.5,  time_points)
        agit_noise    = rng.normal(0, 3,    time_points)

        # ── Nominal trajectories ──────────────────────────────────────
        pH          = 7.0  + ph_noise
        Temperature = 37.0 + temp_noise
        DO          = np.clip(35 - 0.05 * time_h + do_noise, 5, 70)
        Agitation   = np.clip(200 + 0.2 * time_h + agit_noise, 80, 350)
        Feed_Rate   = np.clip(0.05 + 0.003 * time_h + rng.normal(0, 0.005, time_points), 0, 0.5)

        # ── Cell growth (logistic model) ──────────────────────────────
        mu_max = 0.035 + rng.normal(0, 0.002)
        K      = 120
        VCD    = K / (1 + np.exp(-mu_max * (time_h - 80))) + rng.normal(0, 1, time_points)
        VCD    = np.clip(VCD, 0, 130)

        Cell_Viability = np.clip(95 - 0.1 * time_h + rng.normal(0, 1, time_points), 40, 99)
        Titre          = np.clip(0.005 * time_h**1.3 + rng.normal(0, 0.05, time_points), 0, 8)
        Glucose        = np.clip(20 - 0.08 * time_h + rng.normal(0, 0.3, time_points), 0.5, 25)
        Lactate        = np.clip(0.03 * time_h + rng.normal(0, 0.1, time_points), 0, 8)

        # ── Inject deviation ──────────────────────────────────────────
        is_dev_flags = np.zeros(time_points, dtype=int)
        if is_dev and dev_start is not None:
            dev_idx = time_h >= dev_start
            pH[dev_idx]           += rng.normal(0.4, 0.1, dev_idx.sum())   # pH drift
            DO[dev_idx]           -= rng.uniform(15, 25)                    # DO crash
            Cell_Viability[dev_idx] -= rng.uniform(10, 20)
            is_dev_flags[dev_idx]  = 1

        DO             = np.clip(DO, 0, 100)
        Cell_Viability = np.clip(Cell_Viability, 10, 99)

        for t_idx in range(time_points):
            records.append({
                "Batch_ID"       : f"Batch_{batch_id:03d}",
                "Time_h"         : round(float(time_h[t_idx]), 1),
                "pH"             : round(float(pH[t_idx]), 3),
                "Temperature"    : round(float(Temperature[t_idx]), 2),
                "DO"             : round(float(DO[t_idx]), 2),
                "Agitation"      : round(float(Agitation[t_idx]), 1),
                "Feed_Rate"      : round(float(Feed_Rate[t_idx]), 4),
                "VCD"            : round(float(VCD[t_idx]), 2),
                "Cell_Viability" : round(float(Cell_Viability[t_idx]), 2),
                "Titre"          : round(float(Titre[t_idx]), 4),
                "Glucose"        : round(float(Glucose[t_idx]), 3),
                "Lactate"        : round(float(Lactate[t_idx]), 3),
                "Is_Deviation"   : int(is_dev_flags[t_idx]),
            })

    df = pd.DataFrame(records)
    return df
