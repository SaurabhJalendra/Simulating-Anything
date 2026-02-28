"""Generate all publication-quality figures from saved rediscovery data.

Run this after rediscovery results exist in output/rediscovery/.
Saves figures to output/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulating_anything.viz.figures import (
    setup_paper_style,
    plot_projectile_trajectories,
    plot_projectile_equation_fit,
    plot_lv_phase_portrait,
    plot_lv_equilibrium_fit,
    plot_lv_sindy_comparison,
    plot_gs_phase_diagram,
    plot_gs_wavelength_scaling,
    plot_gs_pattern_gallery,
    plot_rediscovery_summary,
)
from simulating_anything.rediscovery.gray_scott import (
    _run_gray_scott_jax,
    classify_pattern,
)

OUTPUT_DIR = Path("output/figures")
DATA_DIR = Path("output/rediscovery")


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as both PNG and PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


def generate_projectile_figures() -> dict:
    """Generate all projectile figures."""
    print("\n=== Projectile Figures ===")
    data = dict(np.load(DATA_DIR / "projectile" / "data.npz"))
    # Normalize key names (saved as range_sim to avoid Python builtin conflict)
    if "range_sim" in data and "range" not in data:
        data["range"] = data["range_sim"]
    with open(DATA_DIR / "projectile" / "results.json") as f:
        results = json.load(f)

    # Fig 1: Trajectories
    fig = plot_projectile_trajectories(data, n_show=12)
    save_fig(fig, "projectile_trajectories")

    # Fig 2: Equation fit
    fig = plot_projectile_equation_fit(data)
    save_fig(fig, "projectile_equation_fit")

    # Fig 3: Range vs angle for multiple speeds (custom)
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_v0 = np.unique(data["v0"])
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(unique_v0)))
    for v, color in zip(unique_v0, cmap):
        mask = data["v0"] == v
        thetas = data["theta"][mask]
        ranges = data["range"][mask]
        sort_idx = np.argsort(thetas)
        ax.plot(np.degrees(thetas[sort_idx]), ranges[sort_idx],
                color=color, linewidth=1.5, label=f"v={v:.0f} m/s")
    ax.set_xlabel("Launch Angle (degrees)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Projectile Range vs Launch Angle")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.axvline(45, color="k", linewidth=0.5, linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "projectile_range_vs_angle")

    return results


def generate_lv_figures() -> dict:
    """Generate all Lotka-Volterra figures."""
    print("\n=== Lotka-Volterra Figures ===")
    eq_data = dict(np.load(DATA_DIR / "lotka_volterra" / "equilibrium_data.npz"))
    ode_data = dict(np.load(DATA_DIR / "lotka_volterra" / "ode_data.npz"))
    with open(DATA_DIR / "lotka_volterra" / "results.json") as f:
        results = json.load(f)

    true_params = {
        "alpha": results["sindy_ode"]["true_alpha"],
        "beta": results["sindy_ode"]["true_beta"],
        "gamma": results["sindy_ode"]["true_gamma"],
        "delta": results["sindy_ode"]["true_delta"],
    }

    # Fig 1: Phase portrait
    fig = plot_lv_phase_portrait(ode_data["states"], true_params)
    save_fig(fig, "lv_phase_portrait")

    # Fig 2: Equilibrium fits
    fig = plot_lv_equilibrium_fit(eq_data)
    save_fig(fig, "lv_equilibrium_fit")

    # Fig 3: SINDy comparison
    fig = plot_lv_sindy_comparison(ode_data["states"], 0.01, true_params)
    save_fig(fig, "lv_sindy_comparison")

    # Fig 4: Time series (custom)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.arange(len(ode_data["states"])) * 0.01
    axes[0].plot(t, ode_data["states"][:, 0], "b-", linewidth=1.5, label="Prey")
    axes[0].plot(t, ode_data["states"][:, 1], "r-", linewidth=1.5, label="Predator")
    axes[0].axhline(true_params["gamma"] / true_params["delta"],
                     color="b", linestyle="--", alpha=0.5, label=r"$\gamma/\delta$ (prey eq)")
    axes[0].axhline(true_params["alpha"] / true_params["beta"],
                     color="r", linestyle="--", alpha=0.5, label=r"$\alpha/\beta$ (pred eq)")
    axes[0].set_ylabel("Population")
    axes[0].set_title("Lotka-Volterra Dynamics")
    axes[0].legend(fontsize=9)

    # Population ratio
    ratio = ode_data["states"][:, 0] / np.maximum(ode_data["states"][:, 1], 1e-10)
    axes[1].plot(t, ratio, "g-", linewidth=1)
    eq_ratio = (true_params["gamma"] / true_params["delta"]) / (
        true_params["alpha"] / true_params["beta"]
    )
    axes[1].axhline(eq_ratio, color="k", linestyle="--", alpha=0.5,
                     label=f"Eq ratio = {eq_ratio:.2f}")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Prey / Predator Ratio")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "lv_time_series")

    # Fig 5: Equilibrium error distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    prey_theory = eq_data["gamma"] / eq_data["delta"]
    pred_theory = eq_data["alpha"] / eq_data["beta"]
    prey_err = (eq_data["prey_avg"] - prey_theory) / prey_theory * 100
    pred_err = (eq_data["pred_avg"] - pred_theory) / pred_theory * 100
    axes[0].hist(prey_err, bins=30, color="#2196F3", alpha=0.7, edgecolor="k", linewidth=0.5)
    axes[0].axvline(0, color="k", linewidth=0.5)
    axes[0].set_xlabel("Relative Error (%)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Prey Equilibrium Error (mean={np.mean(np.abs(prey_err)):.2f}%)")
    axes[1].hist(pred_err, bins=30, color="#F44336", alpha=0.7, edgecolor="k", linewidth=0.5)
    axes[1].axvline(0, color="k", linewidth=0.5)
    axes[1].set_xlabel("Relative Error (%)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Predator Equilibrium Error (mean={np.mean(np.abs(pred_err)):.2f}%)")
    fig.tight_layout()
    save_fig(fig, "lv_equilibrium_errors")

    return results


def generate_gs_figures() -> dict:
    """Generate all Gray-Scott figures."""
    print("\n=== Gray-Scott Figures ===")
    with open(DATA_DIR / "gray_scott" / "results.json") as f:
        results = json.load(f)

    # Fig 1: Phase diagram
    fig = plot_gs_phase_diagram(results["phase_diagram"])
    save_fig(fig, "gs_phase_diagram")

    # Fig 2: Wavelength scaling
    sa = results.get("scaling_analysis", {})
    dv_pairs = sa.get("dv_wavelength_pairs", [])
    if dv_pairs:
        fig = plot_gs_wavelength_scaling(dv_pairs)
        save_fig(fig, "gs_wavelength_scaling")

    # Fig 3: Pattern gallery - generate representative patterns
    print("  Generating pattern gallery (4 representative patterns)...")
    gallery_params = [
        ("Uniform\nf=0.01, k=0.04", 0.01, 0.04),
        ("Spots\nf=0.035, k=0.065", 0.035, 0.065),
        ("Stripes\nf=0.04, k=0.063", 0.04, 0.063),
        ("Complex\nf=0.025, k=0.055", 0.025, 0.055),
    ]
    grid_size = 128
    rng = np.random.default_rng(42)
    u_init = np.ones((grid_size, grid_size), dtype=np.float64)
    v_init = np.zeros((grid_size, grid_size), dtype=np.float64)
    cx, cy = grid_size // 2, grid_size // 2
    r = max(grid_size // 10, 2)
    u_init[cx - r:cx + r, cy - r:cy + r] = 0.50
    v_init[cx - r:cx + r, cy - r:cy + r] = 0.25
    u_init += 0.05 * rng.standard_normal(u_init.shape)
    v_init += 0.05 * rng.standard_normal(v_init.shape)
    v_init = np.clip(v_init, 0, 1)

    patterns = []
    for label, f_val, k_val in gallery_params:
        try:
            _, v_final = _run_gray_scott_jax(
                u_init, v_init, 0.16, 0.08, f_val, k_val, 1.0, 1.0, 10000
            )
        except Exception:
            # Fallback to numpy if JAX not available
            from simulating_anything.rediscovery.gray_scott import _run_gray_scott_jax
            _, v_final = _run_gray_scott_jax(
                u_init, v_init, 0.16, 0.08, f_val, k_val, 1.0, 1.0, 10000
            )
        patterns.append((label, v_final))
        ptype = classify_pattern(v_final)
        print(f"    {label.split(chr(10))[0]}: classified as {ptype}")

    fig = plot_gs_pattern_gallery(patterns)
    save_fig(fig, "gs_pattern_gallery")

    # Fig 4: Pattern energy map (custom heatmap)
    f_values = sorted(set(p["f"] for p in results["phase_diagram"]))
    k_values = sorted(set(p["k"] for p in results["phase_diagram"]))
    energy_grid = np.zeros((len(k_values), len(f_values)))
    for p in results["phase_diagram"]:
        fi = f_values.index(p["f"])
        ki = k_values.index(p["k"])
        energy_grid[ki, fi] = np.log10(max(p["energy"], 1e-12))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(energy_grid, origin="lower", aspect="auto",
                   extent=[f_values[0], f_values[-1], k_values[0], k_values[-1]],
                   cmap="hot")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("log10(Pattern Energy)")
    ax.set_xlabel("Feed rate (f)")
    ax.set_ylabel("Kill rate (k)")
    ax.set_title("Gray-Scott Pattern Energy Landscape")
    fig.tight_layout()
    save_fig(fig, "gs_energy_landscape")

    return results


def generate_summary_figure(proj_results: dict, lv_results: dict, gs_results: dict) -> None:
    """Generate the cross-domain summary figure."""
    print("\n=== Summary Figure ===")
    all_results = {
        "projectile": proj_results,
        "lotka_volterra": lv_results,
        "gray_scott": gs_results,
    }
    fig = plot_rediscovery_summary(all_results)
    save_fig(fig, "rediscovery_summary")

    # Also generate a comprehensive table figure
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    table_data = [
        ["Projectile", r"$R = v_0^2 \sin(2\theta)/g$",
         proj_results.get("best_equation", "N/A")[:50],
         f"{proj_results.get('best_r_squared', 0):.6f}", "PySR"],
        ["LV (prey eq)", r"$x^* = \gamma/\delta$",
         lv_results.get("prey_equilibrium", {}).get("best", "N/A")[:50],
         f"{lv_results.get('prey_equilibrium', {}).get('best_r2', 0):.6f}", "PySR"],
        ["LV (pred eq)", r"$y^* = \alpha/\beta$",
         lv_results.get("pred_equilibrium", {}).get("best", "N/A")[:50],
         f"{lv_results.get('pred_equilibrium', {}).get('best_r2', 0):.6f}", "PySR"],
        ["LV (ODE prey)", r"$\dot{x} = \alpha x - \beta xy$",
         lv_results.get("sindy_ode", {}).get("discoveries", [{}])[0].get("expression", "N/A")[:50],
         f"{lv_results.get('sindy_ode', {}).get('discoveries', [{}])[0].get('r_squared', 0):.6f}",
         "SINDy"],
        ["GS (wavelength)", r"$\lambda \sim \sqrt{D_v}$",
         gs_results.get("scaling_analysis", {}).get("best_scaling_equation", "N/A")[:50],
         f"{gs_results.get('scaling_analysis', {}).get('best_scaling_r2', 0):.6f}", "PySR"],
    ]
    headers = ["Domain", "Target Law", "Discovered Expression", "R²", "Method"]
    table = ax.table(cellText=table_data, colLabels=headers, loc="center",
                     cellLoc="center", colColours=["#E3F2FD"] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Cross-Domain Rediscovery Results", fontsize=13, pad=20)
    fig.tight_layout()
    save_fig(fig, "results_table")


def main() -> None:
    setup_paper_style()
    print("Generating all publication-quality figures...")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    proj = generate_projectile_figures()
    lv = generate_lv_figures()
    gs = generate_gs_figures()
    generate_summary_figure(proj, lv, gs)

    # Count output files
    n_png = len(list(OUTPUT_DIR.glob("*.png")))
    n_pdf = len(list(OUTPUT_DIR.glob("*.pdf")))
    print(f"\nDone! Generated {n_png} PNG + {n_pdf} PDF figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
