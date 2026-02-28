"""Generate publication-quality ablation study figures.

Creates 4 figures:
1. Sampling strategy comparison (bar chart)
2. Analysis method comparison (bar chart with form correctness)
3. Data quantity convergence (line plot)
4. Combined ablation summary (2x2 panel)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/figures")
ABLATION_RESULTS = Path("output/ablation/ablation_results.json")

COLORS = {
    "primary": "#2196F3",
    "secondary": "#4CAF50",
    "warning": "#FF9800",
    "danger": "#F44336",
    "correct": "#4CAF50",
    "wrong": "#F44336",
}


def load_results() -> dict:
    """Load ablation results from JSON."""
    if not ABLATION_RESULTS.exists():
        logger.info("Running ablation study to generate results...")
        from simulating_anything.analysis.pipeline_ablation import run_pipeline_ablation
        return run_pipeline_ablation()

    with open(ABLATION_RESULTS) as f:
        return json.load(f)


def fig_sampling_strategy(results: dict) -> None:
    """Bar chart comparing sampling strategies for projectile."""
    data = results["sampling_strategy"]
    variants = [d["variant"] for d in data]
    r2_vals = [d["r_squared"] for d in data]
    correct = [d["correct_form"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS["correct"] if c else COLORS["warning"] for c in correct]
    bars = ax.bar(range(len(variants)), r2_vals, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.85)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("R$^2$", fontsize=12)
    ax.set_title("Sampling Strategy Ablation (Projectile)", fontsize=14, fontweight="bold")
    ax.set_ylim(0.999, 1.0001)

    # Add value labels
    for bar, r2, c in zip(bars, r2_vals, correct):
        label = f"{r2:.6f}"
        form_label = "correct" if c else ""
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00001,
                label, ha="center", va="bottom", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["correct"], label="Correct form (c = 1/g)"),
        Patch(facecolor=COLORS["warning"], label="Incorrect coefficient"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = OUTPUT_DIR / f"ablation_sampling.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {path}")
    plt.close(fig)


def fig_analysis_method(results: dict) -> None:
    """Bar chart comparing analysis methods for harmonic oscillator."""
    data = results["analysis_method"]
    variants = [d["variant"] for d in data]
    r2_vals = [d["r_squared"] for d in data]
    correct = [d["correct_form"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS["correct"] if c else COLORS["danger"] for c in correct]
    bars = ax.bar(range(len(variants)), r2_vals, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.85)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("R$^2$ (frequency accuracy)", fontsize=12)
    ax.set_title("Analysis Method Ablation (Harmonic Oscillator)", fontsize=14,
                 fontweight="bold")

    # Add value labels
    for bar, r2 in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{r2:.4f}", ha="center", va="bottom", fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["correct"], label="Correct functional form"),
        Patch(facecolor=COLORS["danger"], label="Wrong functional form"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = OUTPUT_DIR / f"ablation_analysis_method.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {path}")
    plt.close(fig)


def fig_data_quantity(results: dict) -> None:
    """Line plot of LV equilibrium convergence vs trajectory length."""
    data = results["data_quantity"]
    steps = [d["n_samples"] - 1 for d in data]  # n_samples = n_steps + 1
    r2_vals = [d["r_squared"] for d in data]
    correct = [d["correct_form"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(steps, r2_vals, "o-", color=COLORS["primary"], linewidth=2,
                markersize=8, markerfacecolor="white", markeredgewidth=2)

    # Mark correct vs incorrect
    for s, r2, c in zip(steps, r2_vals, correct):
        color = COLORS["correct"] if c else COLORS["warning"]
        ax.plot(s, r2, "o", color=color, markersize=10, zorder=5)

    # Threshold line
    ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, label="R$^2$ = 0.99")

    ax.set_xlabel("Trajectory Length (steps)", fontsize=12)
    ax.set_ylabel("R$^2$ (equilibrium accuracy)", fontsize=12)
    ax.set_title("Data Quantity Ablation (Lotka-Volterra)", fontsize=14, fontweight="bold")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["correct"],
               markersize=10, label="Equilibrium within 10%"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["warning"],
               markersize=10, label="Equilibrium error > 10%"),
        Line2D([0], [0], color="gray", linestyle="--", label="R$^2$ = 0.99"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = OUTPUT_DIR / f"ablation_data_quantity.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {path}")
    plt.close(fig)


def fig_combined(results: dict) -> None:
    """2x2 combined ablation figure for paper."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Sampling strategy
    ax = axes[0, 0]
    data = results["sampling_strategy"]
    variants = [d["variant"].replace(" (15x15)", "\n(15x15)").replace(" (narrow)", "\n(narrow)")
                for d in data]
    r2_vals = [d["r_squared"] for d in data]
    correct = [d["correct_form"] for d in data]
    colors = [COLORS["correct"] if c else COLORS["warning"] for c in correct]
    ax.bar(range(len(variants)), r2_vals, color=colors, alpha=0.85)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, fontsize=8)
    ax.set_ylabel("R$^2$", fontsize=10)
    ax.set_title("(a) Sampling Strategy", fontsize=11, fontweight="bold")
    ax.set_ylim(0.999, 1.0001)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Top-right: Analysis method
    ax = axes[0, 1]
    data = results["analysis_method"]
    variants = [d["variant"].replace(" (wrong form)", "\n(wrong form)") for d in data]
    r2_vals = [d["r_squared"] for d in data]
    correct = [d["correct_form"] for d in data]
    colors = [COLORS["correct"] if c else COLORS["danger"] for c in correct]
    ax.bar(range(len(variants)), r2_vals, color=colors, alpha=0.85)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, fontsize=8)
    ax.set_ylabel("R$^2$ (freq. accuracy)", fontsize=10)
    ax.set_title("(b) Analysis Method", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom-left: Data quantity
    ax = axes[1, 0]
    data = results["data_quantity"]
    steps = [d["n_samples"] - 1 for d in data]
    r2_vals = [d["r_squared"] for d in data]
    correct = [d["correct_form"] for d in data]
    ax.semilogx(steps, r2_vals, "o-", color=COLORS["primary"], linewidth=2,
                markersize=6, markerfacecolor="white", markeredgewidth=2)
    for s, r2, c in zip(steps, r2_vals, correct):
        color = COLORS["correct"] if c else COLORS["warning"]
        ax.plot(s, r2, "o", color=color, markersize=8, zorder=5)
    ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Trajectory Length (steps)", fontsize=10)
    ax.set_ylabel("R$^2$ (equilibrium)", fontsize=10)
    ax.set_title("(c) Data Quantity", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom-right: Summary table
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = (
        "ABLATION SUMMARY\n"
        "=" * 40 + "\n\n"
        "Sampling Strategy:\n"
        "  Grid > Random > Edge > Clustered\n"
        "  All strategies achieve R$^2$ > 0.999\n\n"
        "Analysis Method:\n"
        "  FFT $\\approx$ Zero-crossing > Autocorrelation\n"
        "  Polynomial fits data but wrong physics\n\n"
        "Data Quantity:\n"
        "  5000+ steps needed for convergence\n"
        "  Short trajectories miss equilibrium\n\n"
        "Key Insight:\n"
        "  Correct functional form matters more\n"
        "  than fitting accuracy (R$^2$)"
    )
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("(d) Summary", fontsize=11, fontweight="bold")

    plt.tight_layout(pad=2.0)

    for ext in ["png", "pdf"]:
        path = OUTPUT_DIR / f"ablation_combined.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {path}")
    plt.close(fig)


def main():
    """Generate all ablation figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ablation results...")
    results = load_results()

    logger.info("Generating figures...")
    fig_sampling_strategy(results)
    fig_analysis_method(results)
    fig_data_quantity(results)
    fig_combined(results)

    logger.info(f"\nAll ablation figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
