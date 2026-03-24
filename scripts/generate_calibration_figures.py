"""Generate publication-quality figures for calibration and discovery results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def setup_style():
    """Publication-quality matplotlib style."""
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def fig1_calibration_comparison():
    """Fig 1: Discovered vs literature thresholds for 3 domains."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Brusselator
    ax = axes[0]
    a_vals = [1.0, 1.5, 2.0]
    exact = [2.0, 3.25, 5.0]
    detected = [1.448, 2.782, 4.557]  # From calibration runs
    errors = [abs(d - e) / e * 100 for d, e in zip(detected, exact)]

    x = np.arange(len(a_vals))
    ax.bar(x - 0.15, exact, 0.3, label="Exact (analytical)", color="#2196F3", alpha=0.8)
    ax.bar(x + 0.15, detected, 0.3, label="Detected (our engine)", color="#FF5722", alpha=0.8)
    for j in range(len(a_vals)):
        ax.annotate(f"{errors[j]:.0f}%", (x[j], max(exact[j], detected[j]) + 0.1),
                    ha="center", fontsize=8, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels([f"a={a}" for a in a_vals])
    ax.set_ylabel("Critical b")
    ax.set_title("Brusselator Hopf (b_c = 1+a\u00b2)")
    ax.legend(loc="upper left")

    # NPB
    ax = axes[1]
    params = ["burst", "D_dilution"]
    lit_vals = [50.0, 0.35]
    det_vals = [53.6, 0.594]
    errs = [7.2, 69.7]

    x = np.arange(len(params))
    ax.barh(x - 0.15, lit_vals, 0.3, label="Literature", color="#2196F3", alpha=0.8)
    ax.barh(x + 0.15, det_vals, 0.3, label="Detected", color="#FF5722", alpha=0.8)
    for j in range(len(params)):
        ax.annotate(f"{errs[j]:.0f}%", (max(lit_vals[j], det_vals[j]) + 1, x[j]),
                    va="center", fontsize=8, color="#666")
    ax.set_yticks(x)
    ax.set_yticklabels(params)
    ax.set_xlabel("Critical value")
    ax.set_title("NPB Chemostat")
    ax.legend(loc="lower right")

    # Tumor-Immune
    ax = axes[2]
    params = ["a_t (default)", "a_t (calibrated)"]
    lit_vals = [0.20, 0.20]
    det_vals = [0.152, 0.295]
    errs = [24.0, 47.5]

    x = np.arange(len(params))
    ax.bar(x - 0.15, lit_vals, 0.3, label="Literature (Kuznetsov 1994)", color="#2196F3", alpha=0.8)
    ax.bar(x + 0.15, det_vals, 0.3, label="Detected", color="#FF5722", alpha=0.8)
    for j in range(len(params)):
        ax.annotate(f"{errs[j]:.0f}%", (x[j], max(lit_vals[j], det_vals[j]) + 0.01),
                    ha="center", fontsize=8, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels(params, fontsize=9)
    ax.set_ylabel("Critical a_t")
    ax.set_title("Tumor-Immune Escape")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig("output/figures/fig1_calibration.png")
    plt.close()
    print("Saved fig1_calibration.png")


def fig2_baseline_comparison():
    """Fig 2: Baseline method comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["Naive Sweep", "Gradient-Only", "Our Method"]
    tp = [0, 0, 2]
    fp = [3, 5, 6]
    fn = [3, 3, 1]
    errors = [100, 100, 4.7]

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax.bar(x - width, tp, width, label="True Positives", color="#4CAF50", alpha=0.8)
    bars2 = ax.bar(x, fp, width, label="False Positives", color="#F44336", alpha=0.8)
    bars3 = ax.bar(x + width, fn, width, label="False Negatives", color="#FF9800", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Count")
    ax.set_title("Discovery Method Comparison (3 calibrated domains)")
    ax.legend()

    # Add error annotation
    for j, (m, e) in enumerate(zip(methods, errors)):
        ax.annotate(f"Err: {e:.1f}%", (x[j], max(tp[j], fp[j], fn[j]) + 0.3),
                    ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("output/figures/fig2_baselines.png")
    plt.close()
    print("Saved fig2_baselines.png")


def fig3_dt_invariance():
    """Fig 3: dt-invariance summary."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Pie chart of pass/fail
    ax = axes[0]
    labels = ["Pass (<5%)", "Real Fail (>5%)", "Search Resolution"]
    sizes = [24, 3, 24]
    colors = ["#4CAF50", "#F44336", "#FFC107"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title("dt-Invariance Results (51 tested)")

    # Battery thermal invariant demonstration
    ax = axes[1]
    dts = [0.005, 0.01, 0.02]
    crits = [1.720, 1.720, 1.720]
    ax.plot(dts, crits, "o-", color="#2196F3", markersize=10, linewidth=2)
    ax.fill_between(dts, [c * 0.95 for c in crits], [c * 1.05 for c in crits],
                    alpha=0.2, color="#2196F3")
    ax.set_xlabel("Timestep (dt)")
    ax.set_ylabel("Critical I_load")
    ax.set_title("Battery Thermal: Perfect dt-Invariance")
    ax.set_ylim(1.5, 1.9)
    ax.annotate("I_load_c = 1.720\n(0.0% deviation)", (0.012, 1.74),
               fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("output/figures/fig3_dt_invariance.png")
    plt.close()
    print("Saved fig3_dt_invariance.png")


def fig4_discovery_summary():
    """Fig 4: Discovery counts by domain and type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # By domain
    ax = axes[0]
    domains = ["NPB", "Laser", "Gene-Met", "LV variants", "L-Stommel",
               "Earthquake", "Circadian", "Social-Epi", "Battery", "RCW"]
    counts = [17, 8, 6, 9, 3, 2, 1, 2, 1, 2]
    calibrated = [7, 3, 0, 0, 0, 0, 0, 0, 0, 0]

    y = np.arange(len(domains))
    ax.barh(y, counts, 0.6, label="Default params", color="#2196F3", alpha=0.7)
    ax.barh(y, calibrated, 0.6, label="Calibrated (literature)", color="#FF5722", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(domains)
    ax.set_xlabel("Validated Bifurcations")
    ax.set_title("Validated Discoveries by Domain")
    ax.legend()

    # By type
    ax = axes[1]
    types = ["Hopf\n(steady->osc)", "Inv. Hopf\n(osc->steady)",
             "Structural\nInvariant", "Policy\nEquation", "2D Phase\nDiagram"]
    type_counts = [28, 27, 4, 9, 31]
    colors = ["#4CAF50", "#2196F3", "#9C27B0", "#FF9800", "#607D8B"]
    ax.bar(range(len(types)), type_counts, color=colors, alpha=0.8)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types)
    ax.set_ylabel("Count")
    ax.set_title("Discovery Types")

    plt.tight_layout()
    plt.savefig("output/figures/fig4_summary.png")
    plt.close()
    print("Saved fig4_summary.png")


def fig5_predictions():
    """Fig 5: Testable predictions summary."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    predictions = [
        ("HIGH", "NPB: Burst >50 triggers oscillations",
         "Testable in T4-E.coli chemostat, 7.2% error vs literature"),
        ("HIGH", "NPB: Two dilution windows (D<0.07, 0.37<D<0.58)",
         "Run parallel chemostats at D=0.05, 0.2, 0.4, 0.7"),
        ("MEDIUM", "NPB: Paradox of enrichment with re-entry",
         "Novel prediction: N_in>5.4 re-destabilizes system"),
        ("MEDIUM", "Tumor: Escape at a_t>0.30/day (d_ti=0.05)",
         "Calibrated with Kuznetsov 1994 parameters"),
        ("HIGH", "Brusselator: b_c=1+a^2 within 9-28%",
         "Method validation on exact analytical benchmark"),
        ("LOW", "Tumor: Escape threshold invariant to cytokine coupling",
         "Structural invariant, testable with anti-TGF-beta"),
    ]

    colors = {"HIGH": "#4CAF50", "MEDIUM": "#FF9800", "LOW": "#F44336"}
    for i, (conf, pred, test) in enumerate(predictions):
        y = 0.9 - i * 0.15
        ax.add_patch(plt.Rectangle((0.02, y - 0.04), 0.08, 0.08,
                     color=colors[conf], alpha=0.8, transform=ax.transAxes))
        ax.text(0.06, y, conf, transform=ax.transAxes, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
        ax.text(0.12, y + 0.02, pred, transform=ax.transAxes, fontsize=10, va="center")
        ax.text(0.12, y - 0.02, test, transform=ax.transAxes, fontsize=8,
                va="center", color="#666", style="italic")

    ax.set_title("Testable Predictions from Autonomous Discovery Engine", fontsize=14, pad=20)
    plt.savefig("output/figures/fig5_predictions.png")
    plt.close()
    print("Saved fig5_predictions.png")


if __name__ == "__main__":
    setup_style()
    Path("output/figures").mkdir(parents=True, exist_ok=True)

    fig1_calibration_comparison()
    fig2_baseline_comparison()
    fig3_dt_invariance()
    fig4_discovery_summary()
    fig5_predictions()

    print("\nAll figures generated in output/figures/")
