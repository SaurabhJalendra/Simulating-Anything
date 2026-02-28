"""Generate publication-quality sensitivity analysis figures.

Runs the sensitivity analysis from simulating_anything.analysis.sensitivity
and produces 3 figures (300dpi PNG + vector PDF) in output/sensitivity_figures/.

Figure 1: Noise sensitivity (projectile R^2 vs relative noise)
Figure 2: Data quantity sensitivity (projectile R^2 vs sample count)
Figure 3: Combined 2x2 sensitivity overview

Usage:
    python scripts/generate_sensitivity_figures.py
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulating_anything.analysis.sensitivity import (
    sensitivity_data_quantity,
    sensitivity_harmonic_oscillator,
    sensitivity_noise,
    sensitivity_param_range,
)

OUTPUT_DIR = Path("output/sensitivity_figures")

logger = logging.getLogger(__name__)


def setup_style() -> None:
    """Configure publication-quality matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    """Save figure as both 300dpi PNG and vector PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name} (.png + .pdf)")


def _find_r2_at_threshold(
    values: list[float],
    r_squared: list[float],
    r2_target: float = 0.999,
) -> float | None:
    """Interpolate to find the value where R^2 first drops below r2_target.

    Returns None if R^2 never drops below the target.
    """
    for i in range(len(r_squared) - 1):
        if r_squared[i] >= r2_target and r_squared[i + 1] < r2_target:
            # Linear interpolation on log scale for values
            frac = (r2_target - r_squared[i]) / (r_squared[i + 1] - r_squared[i])
            log_v0 = np.log10(max(values[i], 1e-12))
            log_v1 = np.log10(max(values[i + 1], 1e-12))
            return 10 ** (log_v0 + frac * (log_v1 - log_v0))
    return None


def _parse_omega_error(discovered_forms: list[str]) -> list[float]:
    """Extract relative omega error from HO discovered_form strings.

    Each string has the form: 'omega = X.XXXX (true: Y.YYYY)'.
    Returns list of |omega_measured - omega_true| / omega_true.
    """
    errors = []
    pattern = re.compile(r"omega\s*=\s*([\d.]+)\s*\(true:\s*([\d.]+)\)")
    for form in discovered_forms:
        m = pattern.search(form)
        if m:
            omega_measured = float(m.group(1))
            omega_true = float(m.group(2))
            rel_err = abs(omega_measured - omega_true) / max(omega_true, 1e-15)
            errors.append(rel_err)
        else:
            errors.append(np.nan)
    return errors


# ---------------------------------------------------------------------------
# Figure 1: Noise Sensitivity
# ---------------------------------------------------------------------------
def fig_noise_sensitivity() -> None:
    """Projectile R^2 vs observation noise level."""
    print("\n=== Figure 1: Noise Sensitivity ===")

    result = sensitivity_noise()
    values = np.array(result.values)
    r2 = np.array(result.r_squared)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot R^2 vs noise level (skip noise=0 on log scale)
    mask = values > 0
    ax.semilogx(values[mask], r2[mask], "o-", color="#1976D2",
                markersize=6, linewidth=1.5, label="Projectile")

    # If noise=0 exists, mark it at the left edge
    if not mask[0]:
        ax.axhline(r2[0], color="#1976D2", linewidth=0.8, linestyle=":",
                   alpha=0.5, label=f"No-noise baseline (R$^2$={r2[0]:.6f})")

    # Reference line at R^2 = 0.999
    ax.axhline(0.999, color="#D32F2F", linewidth=1, linestyle="--",
               alpha=0.7, label="R$^2$ = 0.999 threshold")

    # Find and annotate where R^2 crosses 0.999
    threshold_val = _find_r2_at_threshold(result.values, result.r_squared, 0.999)
    if threshold_val is not None:
        ax.annotate(
            f"R$^2$=0.999 at {threshold_val*100:.1f}% noise",
            xy=(threshold_val, 0.999),
            xytext=(threshold_val * 3, 0.996),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.2),
            color="#D32F2F",
            fontweight="bold",
        )
    else:
        # R^2 never drops below 0.999; annotate the lowest noise where R^2 is still high
        min_r2_idx = np.argmin(r2[mask])
        noise_at_min = values[mask][min_r2_idx]
        r2_at_min = r2[mask][min_r2_idx]
        ax.annotate(
            f"R$^2$={r2_at_min:.4f} at {noise_at_min*100:.0f}% noise",
            xy=(noise_at_min, r2_at_min),
            xytext=(noise_at_min / 5, r2_at_min - 0.003),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.2),
            color="#D32F2F",
            fontweight="bold",
        )

    ax.set_xlabel("Relative Noise Level")
    ax.set_ylabel("R$^2$ Score")
    ax.set_title("Discovery Robustness to Observation Noise")
    ax.legend(loc="lower left")
    ax.set_xlim(5e-4, 1.0)
    ax.set_ylim(min(r2) - 0.01, 1.002)

    fig.tight_layout()
    save(fig, "noise_sensitivity")


# ---------------------------------------------------------------------------
# Figure 2: Data Quantity Sensitivity
# ---------------------------------------------------------------------------
def fig_data_quantity() -> None:
    """Projectile R^2 vs number of samples."""
    print("\n=== Figure 2: Data Quantity Sensitivity ===")

    result = sensitivity_data_quantity()
    values = np.array(result.values)
    r2 = np.array(result.r_squared)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogx(values, r2, "s-", color="#388E3C",
                markersize=6, linewidth=1.5, label="Projectile")

    # Reference line at R^2 = 0.999
    ax.axhline(0.999, color="#D32F2F", linewidth=1, linestyle="--",
               alpha=0.7, label="R$^2$ = 0.999 threshold")

    # Find minimum samples for R^2 >= 0.999
    above_mask = r2 >= 0.999
    if np.any(above_mask):
        min_samples = values[above_mask][0]
        ax.axvline(min_samples, color="gray", linewidth=0.8, linestyle=":",
                   alpha=0.5)
        ax.annotate(
            f"R$^2$ $\\geq$ 0.999 from n={int(min_samples)}",
            xy=(min_samples, 0.999),
            xytext=(min_samples * 2.5, 0.997),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            color="#333333",
            fontweight="bold",
        )

    # Annotate each point with its R^2 value
    for v, r in zip(values, r2):
        ax.annotate(f"{r:.4f}", (v, r), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="#555555")

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("R$^2$ Score")
    ax.set_title("Discovery Quality vs Data Quantity")
    ax.legend(loc="lower right")
    ax.set_xlim(3, 2000)
    ax.set_ylim(min(r2) - 0.005, 1.002)

    fig.tight_layout()
    save(fig, "data_quantity")


# ---------------------------------------------------------------------------
# Figure 3: Combined Sensitivity (2x2)
# ---------------------------------------------------------------------------
def fig_combined_sensitivity() -> None:
    """2x2 subplot: projectile noise, data quantity, param range, HO noise."""
    print("\n=== Figure 3: Combined Sensitivity ===")

    # Run all four analyses
    noise_result = sensitivity_noise()
    data_result = sensitivity_data_quantity()
    range_result = sensitivity_param_range()
    ho_results = sensitivity_harmonic_oscillator()
    ho_noise = ho_results["noise"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- (1,1) Projectile noise sensitivity ---
    ax = axes[0, 0]
    values = np.array(noise_result.values)
    r2 = np.array(noise_result.r_squared)
    mask = values > 0
    ax.semilogx(values[mask], r2[mask], "o-", color="#1976D2",
                markersize=5, linewidth=1.5)
    if not mask[0]:
        ax.axhline(r2[0], color="#1976D2", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.axhline(0.999, color="#D32F2F", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Relative Noise Level")
    ax.set_ylabel("R$^2$ Score")
    ax.set_title("(a) Projectile: Noise Sensitivity")
    ax.set_xlim(5e-4, 1.0)
    ax.set_ylim(min(r2) - 0.01, 1.002)

    # --- (1,2) Projectile data quantity ---
    ax = axes[0, 1]
    values = np.array(data_result.values)
    r2 = np.array(data_result.r_squared)
    ax.semilogx(values, r2, "s-", color="#388E3C",
                markersize=5, linewidth=1.5)
    ax.axhline(0.999, color="#D32F2F", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("R$^2$ Score")
    ax.set_title("(b) Projectile: Data Quantity")
    ax.set_xlim(3, 2000)
    ax.set_ylim(min(r2) - 0.005, 1.002)

    # --- (2,1) Projectile parameter range ---
    ax = axes[1, 0]
    values = np.array(range_result.values)
    r2 = np.array(range_result.r_squared)
    ax.plot(values, r2, "D-", color="#F57C00",
            markersize=5, linewidth=1.5)
    ax.axhline(0.999, color="#D32F2F", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Parameter Range Fraction")
    ax.set_ylabel("R$^2$ Score")
    ax.set_title("(c) Projectile: Parameter Range")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(min(r2) - 0.005, 1.002)

    # --- (2,2) HO noise: omega relative error ---
    ax = axes[1, 1]
    ho_values = np.array(ho_noise.values)
    omega_errors = _parse_omega_error(ho_noise.discovered_form)
    omega_errors = np.array(omega_errors)

    # Use only nonzero noise levels for log scale
    ho_mask = ho_values > 0
    if np.any(ho_mask):
        ax.semilogx(ho_values[ho_mask], omega_errors[ho_mask] * 100, "^-",
                     color="#9C27B0", markersize=5, linewidth=1.5)
    # Mark zero-noise point at the left edge as a special marker
    if not ho_mask[0]:
        ax.axhline(omega_errors[0] * 100, color="#9C27B0", linewidth=0.6,
                   linestyle=":", alpha=0.4,
                   label=f"No-noise error: {omega_errors[0]*100:.3f}%")
        ax.legend(loc="upper left", fontsize=8)

    ax.set_xlabel("Relative Noise Level")
    ax.set_ylabel("Frequency Error (%)")
    ax.set_title(r"(d) Harmonic Oscillator: $\omega_0$ Error vs Noise")
    if np.any(ho_mask):
        ax.set_xlim(5e-4, 0.2)

    fig.suptitle("Sensitivity Analysis: Pipeline Robustness", fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "combined_sensitivity")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_style()
    print("Generating sensitivity analysis figures...")
    print(f"Output: {OUTPUT_DIR.resolve()}")

    fig_noise_sensitivity()
    fig_data_quantity()
    fig_combined_sensitivity()

    n_png = len(list(OUTPUT_DIR.glob("*.png")))
    n_pdf = len(list(OUTPUT_DIR.glob("*.pdf")))
    print(f"\nDone! Generated {n_png} PNG + {n_pdf} PDF in {OUTPUT_DIR}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
