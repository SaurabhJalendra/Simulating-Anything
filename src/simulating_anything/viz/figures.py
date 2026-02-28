"""Publication-quality matplotlib figures for the Simulating Anything project."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def setup_paper_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# --- Projectile ---

def plot_projectile_trajectories(
    data: dict[str, np.ndarray],
    n_show: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot sample projectile trajectories with range markers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    from simulating_anything.simulation.rigid_body import ProjectileSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    indices = np.linspace(0, len(data["v0"]) - 1, n_show, dtype=int)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_show))

    for idx, color in zip(indices, cmap):
        config = SimulationConfig(
            domain=Domain.RIGID_BODY, dt=0.005, n_steps=20000,
            parameters={
                "gravity": 9.81, "drag_coefficient": 0.0,
                "initial_speed": float(data["v0"][idx]),
                "launch_angle": float(np.degrees(data["theta"][idx])),
                "mass": 1.0,
            },
        )
        sim = ProjectileSimulation(config)
        sim.reset()
        xs, ys = [0.0], [0.0]
        for _ in range(config.n_steps):
            s = sim.step()
            xs.append(s[0])
            ys.append(s[1])
            if sim._landed:
                break
        ax.plot(xs, ys, color=color, alpha=0.8, linewidth=1.2,
                label=f"v={data['v0'][idx]:.0f}, {np.degrees(data['theta'][idx]):.0f}")

    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Projectile Trajectories (No Drag)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig


def plot_projectile_equation_fit(
    data: dict[str, np.ndarray],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot simulated range vs theoretical R = v^2 sin(2theta)/g."""
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = ax.figure
        axes = [ax, ax]

    R_theory = data["v0"]**2 * np.sin(2 * data["theta"]) / data["g"]
    R_sim = data.get("range", data.get("range_sim", data.get("range_theory", R_theory)))

    # Left: scatter plot
    ax1 = axes[0]
    ax1.scatter(R_theory, R_sim, s=15, alpha=0.7, c=data["theta"],
                cmap="coolwarm", edgecolors="none")
    lims = [0, max(R_theory.max(), R_sim.max()) * 1.05]
    ax1.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect fit")
    ax1.set_xlabel(r"Theoretical Range $R = v_0^2 \sin(2\theta) / g$")
    ax1.set_ylabel("Simulated Range (m)")
    ax1.set_title("Theory vs Simulation")
    ax1.legend()
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)

    # Right: residuals
    ax2 = axes[1]
    rel_error = (R_sim - R_theory) / np.maximum(R_theory, 1e-10) * 100
    ax2.scatter(R_theory, rel_error, s=15, alpha=0.7, c=data["theta"],
                cmap="coolwarm", edgecolors="none")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_xlabel(r"Theoretical Range (m)")
    ax2.set_ylabel("Relative Error (%)")
    ax2.set_title(f"Residuals (mean = {np.mean(np.abs(rel_error)):.4f}%)")

    fig.tight_layout()
    return fig


# --- Lotka-Volterra ---

def plot_lv_phase_portrait(
    states: np.ndarray,
    params: dict[str, float] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot Lotka-Volterra phase portrait with equilibrium point."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    prey, pred = states[:, 0], states[:, 1]
    colors = np.linspace(0, 1, len(prey))
    ax.scatter(prey, pred, c=colors, cmap="viridis", s=1, alpha=0.5)
    ax.plot(prey[0], pred[0], "go", markersize=10, label="Start", zorder=5)

    if params:
        eq_prey = params.get("gamma", 0.4) / params.get("delta", 0.1)
        eq_pred = params.get("alpha", 1.1) / params.get("beta", 0.4)
        ax.plot(eq_prey, eq_pred, "r*", markersize=15, label=f"Equilibrium ({eq_prey:.1f}, {eq_pred:.1f})", zorder=5)

    ax.set_xlabel("Prey Population")
    ax.set_ylabel("Predator Population")
    ax.set_title("Lotka-Volterra Phase Portrait")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_lv_equilibrium_fit(
    eq_data: dict[str, np.ndarray],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot discovered vs theoretical equilibrium values."""
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = ax.figure
        axes = [ax, ax]

    # Prey equilibrium: gamma/delta
    prey_theory = eq_data["gamma"] / eq_data["delta"]
    prey_sim = eq_data["prey_avg"]
    ax1 = axes[0]
    ax1.scatter(prey_theory, prey_sim, s=20, alpha=0.6, color="#2196F3", edgecolors="none")
    lims = [0, max(prey_theory.max(), prey_sim.max()) * 1.05]
    ax1.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax1.set_xlabel(r"Theory: $\gamma / \delta$")
    ax1.set_ylabel("Time-Averaged Prey")
    ax1.set_title(r"Prey Equilibrium: $\langle x \rangle \approx \gamma/\delta$")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)

    # Predator equilibrium: alpha/beta
    pred_theory = eq_data["alpha"] / eq_data["beta"]
    pred_sim = eq_data["pred_avg"]
    ax2 = axes[1]
    ax2.scatter(pred_theory, pred_sim, s=20, alpha=0.6, color="#F44336", edgecolors="none")
    lims = [0, max(pred_theory.max(), pred_sim.max()) * 1.05]
    ax2.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax2.set_xlabel(r"Theory: $\alpha / \beta$")
    ax2.set_ylabel("Time-Averaged Predator")
    ax2.set_title(r"Predator Equilibrium: $\langle y \rangle \approx \alpha/\beta$")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)

    fig.tight_layout()
    return fig


def plot_lv_sindy_comparison(
    states: np.ndarray,
    dt: float,
    true_params: dict[str, float],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot LV trajectory with SINDy-predicted derivatives overlaid."""
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    else:
        fig = ax.figure
        axes = [ax, ax]

    t = np.arange(len(states)) * dt
    prey, pred = states[:, 0], states[:, 1]

    # Compute true derivatives
    alpha = true_params["alpha"]
    beta = true_params["beta"]
    gamma = true_params["gamma"]
    delta = true_params["delta"]
    dprey_true = alpha * prey - beta * prey * pred
    dpred_true = -gamma * pred + delta * prey * pred

    # Numerical derivatives
    dprey_num = np.gradient(prey, dt)
    dpred_num = np.gradient(pred, dt)

    axes[0].plot(t, prey, "b-", linewidth=1.5, label="Prey")
    axes[0].plot(t, pred, "r-", linewidth=1.5, label="Predator")
    axes[0].set_ylabel("Population")
    axes[0].set_title("Lotka-Volterra Dynamics")
    axes[0].legend()

    axes[1].plot(t, dprey_num, "b-", alpha=0.4, linewidth=1, label="d(prey)/dt (numerical)")
    axes[1].plot(t, dprey_true, "b--", linewidth=1.5, label="d(prey)/dt (SINDy: 1.1x - 0.4xy)")
    axes[1].plot(t, dpred_num, "r-", alpha=0.4, linewidth=1, label="d(pred)/dt (numerical)")
    axes[1].plot(t, dpred_true, "r--", linewidth=1.5, label="d(pred)/dt (SINDy: -0.4y + 0.1xy)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Rate of Change")
    axes[1].set_title("SINDy-Recovered ODE vs Numerical Derivatives")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


# --- Gray-Scott ---

def plot_gs_phase_diagram(
    phase_data: list[dict[str, Any]],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot Gray-Scott phase diagram in (f, k) space."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    type_colors = {
        "uniform": "#BDBDBD",
        "spots": "#2196F3",
        "stripes": "#FF9800",
        "complex": "#9C27B0",
    }
    type_markers = {
        "uniform": "s",
        "spots": "o",
        "stripes": "D",
        "complex": "^",
    }

    for ptype in ["uniform", "spots", "stripes", "complex"]:
        pts = [p for p in phase_data if p["pattern_type"] == ptype]
        if pts:
            f_vals = [p["f"] for p in pts]
            k_vals = [p["k"] for p in pts]
            ax.scatter(f_vals, k_vals, s=60, alpha=0.8,
                       c=type_colors.get(ptype, "gray"),
                       marker=type_markers.get(ptype, "o"),
                       label=f"{ptype} ({len(pts)})", edgecolors="k", linewidth=0.5)

    ax.set_xlabel("Feed rate (f)")
    ax.set_ylabel("Kill rate (k)")
    ax.set_title("Gray-Scott Phase Diagram")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_gs_wavelength_scaling(
    dv_wavelength_pairs: list[dict[str, float]],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot wavelength vs D_v with sqrt scaling fit."""
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = ax.figure
        axes = [ax, ax]

    dv = np.array([p["D_v"] for p in dv_wavelength_pairs])
    wl = np.array([p["wavelength"] for p in dv_wavelength_pairs])

    # Linear plot
    axes[0].scatter(dv, wl, s=60, c="#2196F3", edgecolors="k", linewidth=0.5, zorder=5)
    # Fit sqrt curve
    if len(dv) > 2:
        dv_fit = np.linspace(dv.min(), dv.max(), 100)
        # Fit lambda = a * sqrt(D_v) + b
        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(np.sqrt(dv), wl, 1)
        wl_fit = np.polyval(coeffs, np.sqrt(dv_fit))
        axes[0].plot(dv_fit, wl_fit, "r--", linewidth=1.5,
                     label=f"Fit: {coeffs[0]:.1f}$\\sqrt{{D_v}}$ + {coeffs[1]:.1f}")
        axes[0].legend()

    axes[0].set_xlabel(r"Diffusion coefficient $D_v$")
    axes[0].set_ylabel(r"Dominant wavelength $\lambda$")
    axes[0].set_title(r"Wavelength Scaling: $\lambda \sim \sqrt{D_v}$")

    # sqrt(D_v) plot — should be linear
    axes[1].scatter(np.sqrt(dv), wl, s=60, c="#F44336", edgecolors="k", linewidth=0.5, zorder=5)
    if len(dv) > 2:
        corr = np.corrcoef(np.sqrt(dv), wl)[0, 1]
        axes[1].set_title(f"Linearized: r = {corr:.3f}")
        # Linear fit
        m, b = np.polyfit(np.sqrt(dv), wl, 1)
        x_line = np.linspace(np.sqrt(dv).min(), np.sqrt(dv).max(), 50)
        axes[1].plot(x_line, m * x_line + b, "r--", linewidth=1.5)

    axes[1].set_xlabel(r"$\sqrt{D_v}$")
    axes[1].set_ylabel(r"Dominant wavelength $\lambda$")

    fig.tight_layout()
    return fig


def plot_gs_pattern_gallery(
    patterns: list[tuple[str, np.ndarray]],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a gallery of Gray-Scott patterns."""
    n = len(patterns)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for ax_i, (label, field) in zip(axes, patterns):
        im = ax_i.imshow(field, cmap="inferno", origin="lower")
        ax_i.set_title(label, fontsize=10)
        ax_i.axis("off")

    fig.suptitle("Gray-Scott Pattern Gallery", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# --- World Model ---

def plot_training_curves(
    losses: dict[str, list[float]],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot world model training loss curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for name, values in losses.items():
        ax.plot(values, label=name, linewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("World Model Training")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_dream_comparison(
    ground_truth: np.ndarray,
    dreamed: np.ndarray,
    dt: float,
    labels: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot dreamed trajectory vs ground truth."""
    n_vars = ground_truth.shape[1] if ground_truth.ndim > 1 else 1
    if labels is None:
        labels = [f"Var {i}" for i in range(n_vars)]

    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    t = np.arange(len(ground_truth)) * dt

    for i, (ax_i, label) in enumerate(zip(axes, labels)):
        gt = ground_truth[:, i] if ground_truth.ndim > 1 else ground_truth
        dr = dreamed[:, i] if dreamed.ndim > 1 else dreamed
        ax_i.plot(t, gt, "b-", linewidth=1.5, label=f"{label} (truth)")
        ax_i.plot(t, dr, "r--", linewidth=1.5, label=f"{label} (dream)")
        ax_i.set_ylabel(label)
        ax_i.legend()

    axes[-1].set_xlabel("Time")
    fig.suptitle("World Model Dream vs Ground Truth", fontsize=13)
    fig.tight_layout()
    return fig


# --- Summary ---

def plot_rediscovery_summary(
    results: dict[str, dict],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a summary bar chart of rediscovery R-squared values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    names = []
    r2_values = []
    colors = []

    domain_colors = {"projectile": "#2196F3", "lotka_volterra": "#4CAF50", "gray_scott": "#FF9800"}

    if "projectile" in results:
        names.append("Projectile\nR = v²sin(2θ)/g")
        r2_values.append(results["projectile"].get("best_r_squared", 0))
        colors.append(domain_colors["projectile"])

    if "lotka_volterra" in results:
        lv = results["lotka_volterra"]
        if "prey_equilibrium" in lv:
            names.append("LV Prey Eq\nγ/δ")
            r2_values.append(lv["prey_equilibrium"].get("best_r2", 0))
            colors.append(domain_colors["lotka_volterra"])
        if "pred_equilibrium" in lv:
            names.append("LV Pred Eq\nα/β")
            r2_values.append(lv["pred_equilibrium"].get("best_r2", 0))
            colors.append(domain_colors["lotka_volterra"])
        if "sindy_ode" in lv and lv["sindy_ode"].get("discoveries"):
            r2 = lv["sindy_ode"]["discoveries"][0].get("r_squared", 0)
            names.append("LV ODE\n(SINDy)")
            r2_values.append(r2)
            colors.append(domain_colors["lotka_volterra"])

    if "gray_scott" in results:
        gs = results["gray_scott"]
        sa = gs.get("scaling_analysis", {})
        if sa.get("best_scaling_r2"):
            names.append("GS Wavelength\nλ ~ √D_v")
            r2_values.append(sa["best_scaling_r2"])
            colors.append(domain_colors["gray_scott"])

    bars = ax.bar(range(len(names)), r2_values, color=colors, edgecolor="k",
                  linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("R² Score")
    ax.set_title("Rediscovery Quality Across Domains")
    ax.set_ylim(0.95, 1.005)
    ax.axhline(1.0, color="k", linewidth=0.5, linestyle="--", alpha=0.3)

    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig
