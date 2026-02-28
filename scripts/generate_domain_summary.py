"""Generate a comprehensive domain summary table and taxonomy figure for all 14 domains."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# All 14 domains with metadata
DOMAINS = [
    {
        "name": "Projectile",
        "math_class": "Algebraic",
        "dim": 4,
        "method": "PySR",
        "target": r"$R = v_0^2 \sin(2\theta)/g$",
        "best_r2": 0.9999,
        "status": "Rediscovered",
    },
    {
        "name": "Lotka-Volterra",
        "math_class": "ODE (nonlinear)",
        "dim": 2,
        "method": "PySR + SINDy",
        "target": r"$\gamma/\delta$, $\alpha/\beta$, full ODE",
        "best_r2": 1.0,
        "status": "Rediscovered",
    },
    {
        "name": "Gray-Scott",
        "math_class": "PDE",
        "dim": "2 fields",
        "method": "PySR",
        "target": r"$\lambda \sim \sqrt{D_v}$",
        "best_r2": 0.985,
        "status": "Analyzed",
    },
    {
        "name": "SIR Epidemic",
        "math_class": "ODE (nonlinear)",
        "dim": 3,
        "method": "PySR + SINDy",
        "target": r"$R_0 = \beta/\gamma$",
        "best_r2": 1.0,
        "status": "Rediscovered",
    },
    {
        "name": "Double Pendulum",
        "math_class": "Chaotic ODE",
        "dim": 4,
        "method": "PySR",
        "target": r"$T = 2\pi\sqrt{L/g}$",
        "best_r2": 0.999993,
        "status": "Rediscovered",
    },
    {
        "name": "Harmonic Oscillator",
        "math_class": "ODE (linear)",
        "dim": 2,
        "method": "PySR + SINDy",
        "target": r"$\omega_0 = \sqrt{k/m}$",
        "best_r2": 1.0,
        "status": "Rediscovered",
    },
    {
        "name": "Lorenz Attractor",
        "math_class": "Chaotic ODE",
        "dim": 3,
        "method": "SINDy",
        "target": "Full 3-equation system",
        "best_r2": 0.99999,
        "status": "Rediscovered",
    },
    {
        "name": "Navier-Stokes 2D",
        "math_class": "PDE",
        "dim": "NxN field",
        "method": "PySR",
        "target": r"$\lambda = 2\nu|k|^2$",
        "best_r2": 1.0,
        "status": "Rediscovered",
    },
    {
        "name": "Van der Pol",
        "math_class": "ODE (nonlinear)",
        "dim": 2,
        "method": "PySR",
        "target": r"$T(\mu)$, $A \approx 2$",
        "best_r2": 0.99996,
        "status": "Rediscovered",
    },
    {
        "name": "Kuramoto",
        "math_class": "ODE (collective)",
        "dim": "N phases",
        "method": "PySR",
        "target": r"$r(K)$ sync transition",
        "best_r2": 0.9695,
        "status": "Rediscovered",
    },
    {
        "name": "Brusselator",
        "math_class": "ODE (nonlinear)",
        "dim": 2,
        "method": "PySR + SINDy",
        "target": r"$b_c = 1 + a^2$",
        "best_r2": 0.9964,
        "status": "Rediscovered",
    },
    {
        "name": "FitzHugh-Nagumo",
        "math_class": "ODE (nonlinear)",
        "dim": 2,
        "method": "SINDy",
        "target": "Full ODE system",
        "best_r2": 0.99999999,
        "status": "Rediscovered",
    },
    {
        "name": "Heat Equation 1D",
        "math_class": "PDE (linear)",
        "dim": "N points",
        "method": "PySR",
        "target": r"$\lambda_k = Dk^2$",
        "best_r2": 1.0,
        "status": "Rediscovered",
    },
    {
        "name": "Logistic Map",
        "math_class": "Discrete chaos",
        "dim": 1,
        "method": "PySR",
        "target": r"$\lambda(r)$ Lyapunov",
        "best_r2": 0.6287,
        "status": "Analyzed",
    },
]


def generate_summary_table():
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("SIMULATING ANYTHING -- 14-DOMAIN REDISCOVERY SUMMARY")
    print("=" * 100)
    print(f"{'#':>2}  {'Domain':<20} {'Math Class':<18} {'Method':<14} "
          f"{'Best R²':<12} {'Status':<14}")
    print("-" * 100)

    for i, d in enumerate(DOMAINS, 1):
        r2_str = f"{d['best_r2']:.6f}" if d["best_r2"] is not None else "Pending"
        print(f"{i:>2}  {d['name']:<20} {d['math_class']:<18} {d['method']:<14} "
              f"{r2_str:<12} {d['status']:<14}")

    print("-" * 100)
    n_complete = sum(1 for d in DOMAINS if d["status"] == "Rediscovered")
    n_analyzed = sum(1 for d in DOMAINS if d["status"] == "Analyzed")
    n_progress = sum(1 for d in DOMAINS if d["status"] == "In Progress")
    print(f"Total: {len(DOMAINS)} domains | "
          f"{n_complete} rediscovered | {n_analyzed} analyzed | {n_progress} in progress")
    print("=" * 100)


def generate_taxonomy_figure(output_dir: str = "output/figures"):
    """Generate a visual taxonomy of mathematical domain classes."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Organize by math class
    classes = {}
    for d in DOMAINS:
        mc = d["math_class"]
        if mc not in classes:
            classes[mc] = []
        classes[mc].append(d["name"])

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, len(classes) + 1)
    ax.axis("off")
    ax.set_title("Domain Taxonomy: 14 Domains Across 6 Mathematical Classes",
                 fontsize=16, fontweight="bold", pad=20)

    colors = {
        "Algebraic": "#e74c3c",
        "ODE (linear)": "#3498db",
        "ODE (nonlinear)": "#2ecc71",
        "ODE (collective)": "#1abc9c",
        "Chaotic ODE": "#e67e22",
        "PDE": "#9b59b6",
        "PDE (linear)": "#8e44ad",
        "Discrete chaos": "#f39c12",
    }

    y = len(classes) - 0.5
    for mc, domains in classes.items():
        color = colors.get(mc, "#95a5a6")
        # Class label
        ax.text(0, y, mc, fontsize=13, fontweight="bold", color=color,
                va="center")
        # Domain names
        domain_str = ", ".join(domains)
        ax.text(3.5, y, domain_str, fontsize=11, va="center", color="#2c3e50")

        # Draw connecting line
        ax.plot([2.8, 3.3], [y, y], color=color, linewidth=2)

        y -= 1

    plt.tight_layout()
    fig.savefig(output_path / "domain_taxonomy_14.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_path / "domain_taxonomy_14.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved taxonomy figure to {output_path / 'domain_taxonomy_14.png'}")


def generate_r2_barplot(output_dir: str = "output/figures"):
    """Generate R² bar chart for all completed domains."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    completed = [d for d in DOMAINS if d["best_r2"] is not None]
    names = [d["name"] for d in completed]
    r2s = [d["best_r2"] for d in completed]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e74c3c" if r > 0.999 else "#f39c12" if r > 0.99 else "#3498db"
              for r in r2s]
    bars = ax.bar(range(len(names)), r2s, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Best R²", fontsize=12)
    ax.set_title("Rediscovery Performance: Best R² Across 14 Domains",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0.97, 1.002)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(y=0.999, color="gray", linestyle=":", alpha=0.3)

    # Add R² labels on bars
    for bar, r2 in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{r2:.5f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path / "r2_summary_14domain.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_path / "r2_summary_14domain.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved R² bar plot to {output_path / 'r2_summary_14domain.png'}")


if __name__ == "__main__":
    generate_summary_table()
    generate_taxonomy_figure()
    generate_r2_barplot()
    print("\nAll figures generated.")
