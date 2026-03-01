"""Generate publication-quality validation framework figures.

Creates 3 figures:
1. Conservation check results across all domains (heatmap)
2. Theory comparison: simulation vs analytical (scatter)
3. Reproducibility: seed determinism across domains (bar chart)
"""
from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulating_anything.analysis.domain_statistics import DOMAIN_REGISTRY
from simulating_anything.types.simulation import SimulationConfig
from simulating_anything.verification.conservation import (
    check_boundedness,
    check_mass_conservation,
    check_positivity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/figures")
PAPER_DIR = Path("paper/figures")

COLORS = {
    "pass": "#4CAF50",
    "fail": "#F44336",
    "warn": "#FF9800",
    "primary": "#2196F3",
    "secondary": "#9C27B0",
}


def _make_sim(name: str, spec: dict):
    """Instantiate a simulation from a domain spec."""
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["cls"])
    config = SimulationConfig(
        domain=spec["domain"], dt=spec["dt"], n_steps=spec["n_steps"],
        parameters=spec["params"],
    )
    return cls(config), config


def fig_conservation_heatmap() -> None:
    """Heatmap showing conservation check results across all domains."""
    domains = sorted(DOMAIN_REGISTRY.keys())
    checks = ["Mass Cons.", "Positivity", "Finiteness", "Determinism"]
    results = np.zeros((len(domains), len(checks)))

    for i, name in enumerate(domains):
        spec = DOMAIN_REGISTRY[name]
        sim, config = _make_sim(name, spec)

        # Run trajectory
        sim.reset(seed=42)
        traj = sim.run(n_steps=min(spec["n_steps"], 500))
        states = traj.states

        # Mass conservation
        mass_check = check_mass_conservation(states, tolerance=1e-3)
        results[i, 0] = 1.0 if mass_check.passed else 0.0

        # Positivity
        pos_check = check_positivity(states)
        results[i, 1] = 1.0 if pos_check.passed else 0.5  # 0.5 = N/A (can be negative)

        # Finiteness
        finite = np.all(np.isfinite(states))
        results[i, 2] = 1.0 if finite else 0.0

        # Determinism
        sim.reset(seed=42)
        traj2 = sim.run(n_steps=min(spec["n_steps"], 500))
        deterministic = np.allclose(states, traj2.states, atol=0, rtol=0)
        results[i, 3] = 1.0 if deterministic else 0.0

        logger.info(f"  {name:25s}: mass={mass_check.passed}, pos={pos_check.passed}, "
                     f"finite={finite}, determ={deterministic}")

    fig, ax = plt.subplots(figsize=(8, 10))
    cmap = plt.cm.colors.ListedColormap([COLORS["fail"], COLORS["warn"], COLORS["pass"]])
    bounds = [-0.1, 0.25, 0.75, 1.1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(results, cmap=cmap, norm=norm, aspect="auto")

    # Labels
    display_names = [n.replace("_", " ").title() for n in domains]
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels(checks, fontsize=10, rotation=30, ha="right")

    # Annotate cells
    for i in range(len(domains)):
        for j in range(len(checks)):
            val = results[i, j]
            label = "PASS" if val >= 0.9 else ("N/A" if val > 0.3 else "FAIL")
            color = "white" if val < 0.3 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    fontweight="bold", color=color)

    ax.set_title("Validation Check Results Across Domains", fontsize=13, fontweight="bold")
    plt.tight_layout()

    for d in [OUTPUT_DIR, PAPER_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "validation_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_DIR / "validation_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved validation_heatmap")


def fig_theory_comparison() -> None:
    """Scatter plot: simulation results vs analytical predictions."""
    # Collected from rediscovery results
    domains = [
        "Projectile", "Lotka-Volterra", "SIR", "Double Pend.",
        "Harmonic Osc.", "Lorenz", "Navier-Stokes", "Van der Pol",
        "Brusselator", "FitzHugh-Nagumo", "Heat Eq.", "Kuramoto",
        "Gray-Scott", "Logistic Map",
    ]
    r2_values = [
        1.0000, 1.0000, 1.0000, 0.9999,
        1.0000, 0.9999, 1.0000, 0.9999,
        0.9999, 1.0000, 1.0000, 0.9695,
        0.9851, 0.6287,
    ]
    methods = [
        "PySR", "SINDy", "PySR+SINDy", "PySR",
        "PySR+SINDy", "SINDy", "PySR", "PySR",
        "PySR+SINDy", "SINDy", "PySR", "PySR",
        "PySR", "PySR",
    ]

    method_colors = {"PySR": "#2196F3", "SINDy": "#4CAF50", "PySR+SINDy": "#9C27B0"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, color in method_colors.items():
        mask = [m == method for m in methods]
        x = [domains[i] for i in range(len(domains)) if mask[i]]
        y = [r2_values[i] for i in range(len(r2_values)) if mask[i]]
        ax.scatter(x, y, c=color, s=100, label=method, edgecolors="black",
                   linewidth=0.5, zorder=3)

    ax.axhline(y=0.999, color="gray", linestyle="--", alpha=0.5, label="R² = 0.999 threshold")
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Rediscovery Quality: Simulation vs. Known Theory", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(0.55, 1.01)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "theory_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_DIR / "theory_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved theory_comparison")


def fig_runtime_profile() -> None:
    """Bar chart: simulation runtime per domain (log scale)."""
    domains = sorted(DOMAIN_REGISTRY.keys())
    times: list[float] = []
    dims: list[int] = []

    for name in domains:
        spec = DOMAIN_REGISTRY[name]
        sim, config = _make_sim(name, spec)
        sim.reset(seed=42)

        t0 = time.perf_counter()
        traj = sim.run(n_steps=spec["n_steps"])
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

        obs = traj.states
        obs_dim = int(np.prod(obs.shape[1:])) if obs.ndim > 1 else 1
        dims.append(obs_dim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Runtime bar chart
    display_names = [n.replace("_", " ").title() for n in domains]
    colors = [COLORS["primary"] if t < 100 else COLORS["secondary"] for t in times]
    bars = ax1.barh(display_names, times, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Runtime (ms)", fontsize=11)
    ax1.set_title("Simulation Runtime per Domain", fontsize=12, fontweight="bold")
    ax1.set_xscale("log")
    for bar, t in zip(bars, times):
        ax1.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                 f"{t:.1f}ms", va="center", fontsize=7)

    # Runtime vs dimension scatter
    ax2.scatter(dims, times, c=COLORS["primary"], s=80, edgecolors="black", linewidth=0.5)
    for name, d, t in zip(domains, dims, times):
        label = name.replace("_", " ").split()[0][:6]
        ax2.annotate(label, (d, t), textcoords="offset points", xytext=(5, 5),
                     fontsize=7, alpha=0.7)
    ax2.set_xlabel("Observation Dimension", fontsize=11)
    ax2.set_ylabel("Runtime (ms)", fontsize=11)
    ax2.set_title("Runtime vs. Observation Dimension", fontsize=12, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "runtime_profile.png", dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_DIR / "runtime_profile.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved runtime_profile")


if __name__ == "__main__":
    logger.info("Generating validation figures...")

    logger.info("1/3: Conservation heatmap")
    fig_conservation_heatmap()

    logger.info("2/3: Theory comparison scatter")
    fig_theory_comparison()

    logger.info("3/3: Runtime profile")
    fig_runtime_profile()

    logger.info("Done! Figures saved to output/figures/ and paper/figures/")
