"""
deviation_detector.py
---------------------
Module 2 of the Bio-Process Digital Twin.

Implements an AI-based early deviation detection system using
Isolation Forest (unsupervised anomaly detection).

On detecting a deviation, the engine automatically drafts a
Lab Investigation Report (LIR) template — exactly what MSAT
engineers write after a manufacturing excursion.

Classes
-------
  DeviationDetector
      .fit(df_normal)         – train on normal batches only
      .predict(df)            – score all observations
      .plot_anomaly_map(...)  – time-series overlay with alarm flags
      .generate_lir(batch_id) – print auto-drafted LIR to stdout
      .evaluate(df_labelled)  – precision/recall against Is_Deviation label
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = ["pH", "Temperature", "DO", "Agitation",
                "Feed_Rate", "VCD", "Cell_Viability", "Glucose", "Lactate"]
CONTAMINATION = 0.08          # expected anomaly fraction ≈ deviation_fraction
N_ESTIMATORS  = 200
ALARM_LABEL   = "⚠  WARNING: Potential Deviation Detected"
CLEAR_LABEL   = "✓  Process Within Normal Range"


class DeviationDetector:
    """
    Isolation-Forest-based bioreactor deviation early-warning system.

    Mirrors the intent of a real-time PAT (Process Analytical Technology)
    monitoring system as described in FDA's PAT guidance (2004) and
    ICH Q10 pharmaceutical quality system.
    """

    def __init__(self, output_dir: str = "outputs/deviation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model   = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        self._scaler  = StandardScaler()
        self._fitted  = False
        self._results: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "DeviationDetector":
        """
        Train the Isolation Forest on normal process observations.

        Parameters
        ----------
        df : DataFrame — should contain only rows where Is_Deviation == 0.
        """
        X = df[FEATURE_COLS].values
        X_std = self._scaler.fit_transform(X)
        self._model.fit(X_std)
        self._fitted = True
        print(f"[MODEL] Isolation Forest trained on {len(df):,} normal observations "
              f"({df['Batch_ID'].nunique()} batches).")
        return self

    # ──────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score observations.  Adds columns:
          Anomaly_Score  : raw Isolation Forest anomaly score (lower = more anomalous)
          Predicted_Dev  : 1 if predicted deviation, 0 otherwise
          Status         : human-readable alarm label
        """
        self._check_fitted()
        X     = df[FEATURE_COLS].values
        X_std = self._scaler.transform(X)

        scores     = self._model.score_samples(X_std)   # lower → more anomalous
        pred_labels = self._model.predict(X_std)         # -1 = anomaly, +1 = normal
        pred_dev    = (pred_labels == -1).astype(int)

        result = df.copy()
        result["Anomaly_Score"] = scores
        result["Predicted_Dev"] = pred_dev
        result["Status"]        = result["Predicted_Dev"].map(
            {1: ALARM_LABEL, 0: CLEAR_LABEL}
        )
        self._results = result
        n_alarms = int(pred_dev.sum())
        print(f"[PREDICT] Alarms raised: {n_alarms:,} / {len(df):,} "
              f"observations ({100*n_alarms/len(df):.1f}%).")
        return result

    # ──────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────

    def plot_anomaly_map(
        self,
        batch_id: str,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot time-series of key process variables for a single batch,
        overlaying detected deviation windows in red.
        """
        self._check_results()
        df_b = self._results[self._results["Batch_ID"] == batch_id].sort_values("Time_h")
        if df_b.empty:
            print(f"[WARN] Batch '{batch_id}' not found in results.")
            return

        plot_vars = ["pH", "DO", "Cell_Viability", "Titre", "Anomaly_Score"]
        ylabels   = ["pH", "DO (%)", "Viability (%)", "Titre (g/L)", "Anomaly Score"]
        fig, axes = plt.subplots(len(plot_vars), 1, figsize=(12, 10), sharex=True)

        alarm_mask = df_b["Predicted_Dev"] == 1

        for ax, var, ylabel in zip(axes, plot_vars, ylabels):
            ax.plot(df_b["Time_h"], df_b[var], linewidth=1.5,
                    color="#1A73E8", label=var)
            # shade deviation windows
            if alarm_mask.any():
                ax.fill_between(df_b["Time_h"], ax.get_ylim()[0], ax.get_ylim()[1],
                                where=alarm_mask, alpha=0.20, color="red",
                                label="Predicted Deviation")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)

        axes[-1].set_xlabel("Process Time (h)")
        axes[0].set_title(f"Batch {batch_id} — Deviation Detection Overlay",
                          fontsize=12, pad=8)
        handles = [
            plt.Line2D([0], [0], color="#1A73E8", lw=2, label="Process Signal"),
            plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.25, label="Alarm Window"),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=9)
        plt.tight_layout()

        path = save_path or str(self.output_dir / f"anomaly_{batch_id}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[PLOT] Anomaly map saved → {path}")

    # ──────────────────────────────────────────────
    # LIR Auto-Draft
    # ──────────────────────────────────────────────

    def generate_lir(self, batch_id: str) -> str:
        """
        Auto-generate a Lab Investigation Report (LIR) draft for a
        batch where a deviation was predicted.

        The template follows the standard GMP LIR structure used in
        biopharmaceutical manufacturing.
        """
        self._check_results()
        df_b   = self._results[self._results["Batch_ID"] == batch_id]
        alarms = df_b[df_b["Predicted_Dev"] == 1]

        if alarms.empty:
            lir = (f"[LIR] No deviation predicted for {batch_id}. "
                   "LIR not required.")
            print(lir)
            return lir

        # Compute stats for the deviated window
        dev_start  = alarms["Time_h"].min()
        dev_end    = alarms["Time_h"].max()
        mean_ph    = alarms["pH"].mean()
        min_do     = alarms["DO"].min()
        mean_via   = alarms["Cell_Viability"].mean()
        final_tit  = alarms["Titre"].iloc[-1] if not alarms.empty else float("nan")
        n_alarms   = len(alarms)
        worst_var  = (alarms[["pH", "DO", "Cell_Viability"]]
                      .apply(lambda c: (c - df_b[c.name].mean()).abs().mean())
                      .idxmax())

        lir = textwrap.dedent(f"""
╔══════════════════════════════════════════════════════════════╗
║          LAB INVESTIGATION REPORT (LIR) — AUTO DRAFT        ║
║          [Generated by Bio-Process Digital Twin v1.0]        ║
╚══════════════════════════════════════════════════════════════╝

1. DEVIATION SUMMARY
   Batch ID        : {batch_id}
   Date Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}
   Detection Method: Isolation Forest (unsupervised anomaly detection)
   Alarm Count     : {n_alarms} time-points flagged
   Deviation Window: {dev_start:.0f} h – {dev_end:.0f} h (process time)

2. PROCESS PARAMETER EXCURSION DETAILS
   Primary variable : {worst_var}  (highest mean deviation from baseline)
   Mean pH (alarm)  : {mean_ph:.3f}   [Setpoint: 7.00 ± 0.05]
   Min DO  (alarm)  : {min_do:.1f}%  [NOR: 20–50%]
   Mean Viability   : {mean_via:.1f}% [Acceptance: ≥ 70%]
   Titre at alarm   : {final_tit:.3f} g/L

3. PROBABLE ROOT CAUSE (to be confirmed by MSAT review)
   The Isolation Forest model flagged the {dev_start:.0f}–{dev_end:.0f} h window
   as anomalous based on a multi-dimensional distance from the normal
   operating space learned from {len(df_b)} training batches.

   Hypothesis : {'DO control failure (agitation or sparger issue)'
                 if 'DO' in worst_var else
                 'pH control failure (CO2/base addition loop drift)'}

4. IMMEDIATE CONTAINMENT ACTIONS (Suggested)
   □ Increase DO sampling frequency to every 30 min.
   □ Check agitator RPM feedback loop and sparger integrity.
   □ Notify Process Engineering and QA within 2 hours.
   □ Flag batch for enhanced in-process testing (IPT).

5. CAPA RECOMMENDATION
   Corrective  : Recalibrate DO probe and agitator tachometer.
   Preventive  : Add real-time ML-based alarm to MES; set
                 Anomaly Score threshold at –0.10 for early warning.

6. IMPACT ASSESSMENT
   Estimated Titre impact : TBD (pending final harvest data)
   Batch disposition      : Under Review — Hold pending investigation

────────────────────────────────────────────────────────────────
  NOTE: This is an AI-generated draft. MSAT engineer review and
  approval is REQUIRED before submission to the QA system.
────────────────────────────────────────────────────────────────
        """).strip()

        print(lir)
        lir_path = self.output_dir / f"LIR_{batch_id}.txt"
        lir_path.write_text(lir)
        print(f"\n[LIR] Draft saved → {lir_path}")
        return lir

    # ──────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────

    def evaluate(self, df_labelled: pd.DataFrame) -> dict:
        """
        Compare Predicted_Dev against the ground-truth Is_Deviation column.

        Returns
        -------
        metrics : dict  {precision, recall, f1, confusion_matrix}
        """
        self._check_results()
        results_slim = self._results[["Batch_ID", "Time_h", "Predicted_Dev"]]
        truth_slim   = df_labelled[["Batch_ID", "Time_h", "Is_Deviation"]]
        merged = results_slim.merge(truth_slim, on=["Batch_ID", "Time_h"], how="inner")
        y_true = merged["Is_Deviation"].values
        y_pred = merged["Predicted_Dev"].values


        report = classification_report(y_true, y_pred,
                                       target_names=["Normal", "Deviation"],
                                       output_dict=True)
        cm     = confusion_matrix(y_true, y_pred)

        print("\n" + "─" * 60)
        print("  INTERPRETATION: Model Evaluation")
        print("─" * 60)
        print(classification_report(y_true, y_pred,
                                    target_names=["Normal", "Deviation"]))

        prec = report["Deviation"]["precision"]
        rec  = report["Deviation"]["recall"]

        if rec >= 0.80:
            print(f"  ✓ Recall = {rec:.2f} — model catches ≥ 80 % of real deviations.")
            print("    Suitable for a first-alert early-warning system.")
        else:
            print(f"  ⚠  Recall = {rec:.2f} — model misses > 20 % of deviations.")
            print("    Consider lowering the contamination parameter or adding "
                  "additional features (e.g., dCO2 trend).")
        if prec < 0.50:
            print(f"  ⚠  Precision = {prec:.2f} — high false-alarm rate.")
            print("    Recommend post-filtering with a 3-consecutive-point alarm rule.")
        print("─" * 60 + "\n")

        return {"precision": prec, "recall": rec,
                "f1": report["Deviation"]["f1-score"],
                "confusion_matrix": cm}

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model not trained. Call .fit(df_normal) first.")

    def _check_results(self) -> None:
        if self._results is None:
            raise RuntimeError("No predictions. Call .predict(df) first.")
