"""
multivariate_analysis.py
------------------------
Module 1 of the Bio-Process Digital Twin.

Performs multivariate statistical analysis on fed-batch bioreactor data:
  1. Pearson correlation heatmap  – identifies co-varying process parameters.
  2. PCA (Principal Component Analysis) – reduces dimensionality and ranks
     the process variables by their influence on product quality (Titre).
  3. One-way ANOVA                – tests whether mean Titre differs
     significantly across pH operating bands.

All results are accompanied by a plain-English interpretation block that
mirrors the kind of root-cause commentary an MSAT engineer writes in an LIR.

Classes
-------
  MultivariateAnalyser
      .load_data(df)
      .correlation_heatmap(save_path)
      .run_pca(n_components, save_path)
      .run_anova(save_path)
      .full_report()
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Constants / thresholds aligned with ICH Q8 / PAT guidance
# ─────────────────────────────────────────────────────────────────────────────
ALPHA              = 0.05          # significance level
STRONG_CORR_THRESH = 0.6           # |r| ≥ 0.6 → strong correlation
PC1_VARIANCE_WARN  = 0.50          # warn if PC1 explains < 50 % variance

PROCESS_VARS = ["pH", "Temperature", "DO", "Agitation", "Feed_Rate",
                "VCD", "Cell_Viability", "Glucose", "Lactate"]
TARGET_VAR   = "Titre"
PH_BANDS     = {"Low (6.8–6.95)"   : (6.80, 6.95),
                "Nominal (6.95–7.05)": (6.95, 7.05),
                "High (7.05–7.20)"  : (7.05, 7.20)}


class MultivariateAnalyser:
    """
    Multivariate statistical engine for bioreactor process data.

    Designed to mirror the exploratory data analysis phase that MSAT
    engineers perform after a manufacturing deviation campaign.
    """

    def __init__(self, output_dir: str = "outputs/multivariate"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._df: Optional[pd.DataFrame] = None
        self._scaler = StandardScaler()

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def load_data(self, df: pd.DataFrame) -> "MultivariateAnalyser":
        """
        Ingest a process DataFrame.

        Validates that required columns are present and drops rows with NaN
        in any numeric column.
        """
        required = set(PROCESS_VARS + [TARGET_VAR])
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self._df = df[list(required)].dropna().copy()
        print(f"[DATA] Loaded {len(self._df):,} observations "
              f"({df['Batch_ID'].nunique() if 'Batch_ID' in df.columns else '?'} batches).")
        return self

    def correlation_heatmap(
        self,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute Pearson correlation matrix and render a heatmap.

        Returns
        -------
        corr_matrix : pd.DataFrame
        """
        self._check_data()
        features = PROCESS_VARS + [TARGET_VAR]
        corr = self._df[features].corr(method="pearson")

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 9}, ax=ax,
        )
        ax.set_title("Process Variable Correlation Matrix\n"
                     "(Pearson r | Fed-Batch CHO Culture)",
                     fontsize=13, pad=12)
        plt.tight_layout()

        path = save_path or str(self.output_dir / "correlation_heatmap.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[PLOT] Correlation heatmap saved → {path}")

        # ── Interpretation ───────────────────────────────────────────
        titre_corr = corr[TARGET_VAR].drop(TARGET_VAR).sort_values(key=abs, ascending=False)
        strong     = titre_corr[titre_corr.abs() >= STRONG_CORR_THRESH]

        print("\n" + "─" * 60)
        print("  INTERPRETATION: Correlation Analysis")
        print("─" * 60)
        if strong.empty:
            print("  No single process variable shows a strong linear "
                  f"correlation with {TARGET_VAR} (|r| < {STRONG_CORR_THRESH}).\n"
                  "  Recommend multivariate (PCA) analysis.")
        else:
            for var, r in strong.items():
                direction = "positively" if r > 0 else "negatively"
                print(f"  • {var:20s} r = {r:+.3f}  → {direction} correlated with Titre.")
            print(f"\n  MSAT Action: Monitor {strong.index[0]} closely; "
                  "deviations in this variable are most likely to affect product yield.")
        print("─" * 60 + "\n")

        return corr

    def run_pca(
        self,
        n_components: int = 3,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Standardise process variables and run PCA.

        Produces:
          - Scree plot + explained variance bar chart.
          - PC loadings table (which process variables drive each PC).
          - Scores scatter (PC1 vs PC2, coloured by Titre quartile).

        Returns
        -------
        loadings_df : pd.DataFrame  (variables × PCs)
        """
        self._check_data()
        X      = self._df[PROCESS_VARS].values
        X_std  = self._scaler.fit_transform(X)
        n_comp = min(n_components, X_std.shape[1])

        pca          = PCA(n_components=n_comp, random_state=42)
        scores       = pca.fit_transform(X_std)
        explained    = pca.explained_variance_ratio_
        loadings_df  = pd.DataFrame(
            pca.components_.T,
            index=PROCESS_VARS,
            columns=[f"PC{i+1}" for i in range(n_comp)],
        )

        # ── Figure ────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # (a) Scree / variance explained
        axes[0].bar(range(1, n_comp + 1), explained * 100, color="#2176AE", alpha=0.85)
        axes[0].plot(range(1, n_comp + 1), np.cumsum(explained) * 100,
                     "o--", color="#E83151", label="Cumulative")
        axes[0].axhline(80, color="grey", linestyle=":", linewidth=1)
        axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Variance Explained (%)")
        axes[0].set_title("Scree Plot"); axes[0].legend()

        # (b) PC1 loadings bar chart
        pc1_load = loadings_df["PC1"].sort_values(key=abs, ascending=True)
        colors   = ["#E83151" if v > 0 else "#2176AE" for v in pc1_load]
        axes[1].barh(pc1_load.index, pc1_load.values, color=colors, alpha=0.85)
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Loading"); axes[1].set_title("PC1 Loadings")

        # (c) Score scatter coloured by Titre quartile
        titre_q   = pd.qcut(self._df[TARGET_VAR], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        palette   = {"Q1": "#4CAF50", "Q2": "#8BC34A", "Q3": "#FF9800", "Q4": "#F44336"}
        for q_lbl, color in palette.items():
            mask = titre_q == q_lbl
            axes[2].scatter(scores[mask, 0], scores[mask, 1],
                            c=color, label=f"Titre {q_lbl}", alpha=0.5, s=10)
        axes[2].set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
        axes[2].set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
        axes[2].set_title("PCA Score Plot (coloured by Titre quartile)")
        axes[2].legend(markerscale=2, fontsize=8)

        plt.suptitle("Principal Component Analysis – Bio-Process Variables",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        path = save_path or str(self.output_dir / "pca_analysis.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] PCA figure saved → {path}")

        # ── Interpretation ───────────────────────────────────────────
        top_pc1 = loadings_df["PC1"].abs().nlargest(3).index.tolist()
        print("\n" + "─" * 60)
        print("  INTERPRETATION: PCA")
        print("─" * 60)
        print(f"  PC1 explains {explained[0]*100:.1f}% of total process variance.")
        if explained[0] < PC1_VARIANCE_WARN:
            print("  ⚠  PC1 explains < 50 %. Process variability is distributed "
                  "across multiple factors — consider a stricter PAR review.")
        print(f"  Cumulative variance (PC1–PC{n_comp}): "
              f"{np.cumsum(explained)[-1]*100:.1f}%.")
        print(f"\n  Top drivers of PC1: {', '.join(top_pc1)}")
        print("  These variables are the primary candidates for process "
              "parameter tightening (Process Characterisation Study).")
        print("\n  MSAT Action: Assign Critical Process Parameters (CPPs) to the "
              f"top PC1 drivers ({top_pc1[0]}, {top_pc1[1]}) and refine their "
              "Normal Operating Ranges (NORs) in the next DoE campaign.")
        print("─" * 60 + "\n")

        return loadings_df

    def run_anova(self, save_path: Optional[str] = None) -> dict:
        """
        One-way ANOVA: tests whether mean Titre differs across pH bands.

        pH is discretised into three operating bands (Low / Nominal / High).
        Post-hoc interpretation indicates which band maximises Titre.

        Returns
        -------
        result : dict  {f_stat, p_value, groups, interpretation}
        """
        self._check_data()
        groups = {}
        for band_name, (lo, hi) in PH_BANDS.items():
            mask         = self._df["pH"].between(lo, hi)
            groups[band_name] = self._df.loc[mask, TARGET_VAR].values

        # Remove empty groups
        groups = {k: v for k, v in groups.items() if len(v) >= 3}
        if len(groups) < 2:
            print("[WARN] Insufficient data in pH bands for ANOVA.")
            return {}

        f_stat, p_value = stats.f_oneway(*groups.values())

        # ── Box-plot ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        data_list  = [v for v in groups.values()]
        labels_list = list(groups.keys())
        bp = ax.boxplot(data_list, labels=labels_list, patch_artist=True, notch=False)
        colors = ["#AED6F1", "#82E0AA", "#F9E79F"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(f"Titre by pH Band  |  One-way ANOVA: "
                     f"F = {f_stat:.2f}, p = {p_value:.4f}",
                     fontsize=11)
        ax.set_ylabel("Titre (g/L)"); ax.set_xlabel("pH Operating Band")
        sig_text = "*** p < 0.001" if p_value < 0.001 else (
                   "**  p < 0.01"  if p_value < 0.01  else (
                   "*   p < 0.05"  if p_value < ALPHA  else
                   "ns  (not significant)"))
        ax.text(0.98, 0.97, sig_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"))
        plt.tight_layout()
        path = save_path or str(self.output_dir / "anova_ph_titre.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[PLOT] ANOVA box-plot saved → {path}")

        # ── Interpretation ───────────────────────────────────────────
        best_band    = max(groups, key=lambda k: np.mean(groups[k]))
        group_means  = {k: f"{np.mean(v):.3f} g/L" for k, v in groups.items()}

        interp = self._anova_interpretation(p_value, f_stat, best_band, group_means)
        print("\n" + "─" * 60)
        print("  INTERPRETATION: One-Way ANOVA (pH band vs. Titre)")
        print("─" * 60)
        print(interp)
        print("─" * 60 + "\n")

        return {"f_stat": f_stat, "p_value": p_value,
                "groups": group_means, "interpretation": interp}

    def full_report(self) -> None:
        """Run all three analyses sequentially and print a summary banner."""
        print("\n" + "═" * 60)
        print("  BIO-PROCESS DIGITAL TWIN — MULTIVARIATE ANALYSIS REPORT")
        print("═" * 60 + "\n")
        self.correlation_heatmap()
        self.run_pca()
        self.run_anova()
        print("═" * 60)
        print("  Analysis complete. Check 'outputs/multivariate/' for plots.")
        print("═" * 60 + "\n")

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _check_data(self) -> None:
        if self._df is None:
            raise RuntimeError("No data loaded. Call .load_data(df) first.")

    @staticmethod
    def _anova_interpretation(
        p_value: float,
        f_stat:  float,
        best_band: str,
        group_means: dict,
    ) -> str:
        means_str = "\n    ".join([f"{k}: {v}" for k, v in group_means.items()])
        if p_value < ALPHA:
            verdict    = (f"STATISTICALLY SIGNIFICANT  (F = {f_stat:.2f}, "
                          f"p = {p_value:.4f} < α = {ALPHA})")
            action     = (f"pH operating band has a significant effect on Titre.\n"
                          f"  Optimal band: {best_band}  (highest mean Titre).\n"
                          f"  MSAT Action  : Tighten pH set-point to the '{best_band}' range\n"
                          f"                 in the next process characterisation study.\n"
                          f"                 Consider pH as a Critical Process Parameter (CPP).")
        else:
            verdict    = (f"NOT SIGNIFICANT  (F = {f_stat:.2f}, "
                          f"p = {p_value:.4f} ≥ α = {ALPHA})")
            action     = ("pH band does not significantly alter Titre within the\n"
                          "  tested range. Process is robust to pH variation.\n"
                          "  MSAT Action  : pH can be classified as a non-CPP for Titre;\n"
                          "                 retain current NOR without tightening.")
        return textwrap.dedent(f"""
  Result   : {verdict}
  Group means:
    {means_str}
  {action}""").strip()
