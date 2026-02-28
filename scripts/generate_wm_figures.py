"""Generate world model training and dreaming figures.

Run after world model training to create visualization figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from simulating_anything.viz.figures import setup_paper_style

OUTPUT_DIR = Path("output/figures")
WM_DIR = Path("output/world_models")


def save_fig(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png")


def plot_all_loss_curves() -> None:
    """Plot training loss curves for all domains."""
    print("\n=== Loss Curves ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    domains = ["projectile", "lotka_volterra", "gray_scott"]
    titles = ["Projectile", "Lotka-Volterra", "Gray-Scott"]
    colors = {"total": "#2196F3", "recon": "#F44336", "kl": "#4CAF50"}

    for idx, (domain, title) in enumerate(zip(domains, titles)):
        path = WM_DIR / domain / "loss_curves.npz"
        if not path.exists():
            axes[idx].text(0.5, 0.5, "Not trained yet", ha="center", va="center",
                           transform=axes[idx].transAxes, fontsize=12)
            axes[idx].set_title(title)
            continue

        curves = dict(np.load(path))
        epochs = np.arange(1, len(curves["total"]) + 1)

        for name, color in colors.items():
            if name in curves:
                label_map = {"total": "Total", "recon": "Reconstruction", "kl": "KL Divergence"}
                axes[idx].plot(epochs, curves[name], color=color, linewidth=1.5,
                               label=label_map[name], alpha=0.8)

        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Loss")
        axes[idx].set_title(title)
        axes[idx].legend(fontsize=8)
        axes[idx].set_yscale("log")

    fig.suptitle("RSSM World Model Training Loss", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "wm_training_curves")


def plot_dream_comparison_vector(domain: str, title: str, var_names: list[str]) -> None:
    """Plot dream vs ground truth for vector domains."""
    path = WM_DIR / domain / "dream_comparison.npz"
    if not path.exists():
        return

    data = dict(np.load(path))
    gt = data["ground_truth"]
    dreamed = data["dreamed"]
    context_len = int(data["context_len"])

    n_vars = gt.shape[1]
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    t_full = np.arange(len(gt))
    t_dream = np.arange(context_len, context_len + len(dreamed))

    for i, (ax, name) in enumerate(zip(axes, var_names)):
        # Ground truth full
        ax.plot(t_full, gt[:, i], "b-", linewidth=1.5, label=f"{name} (ground truth)")
        # Context region shading
        ax.axvspan(0, context_len, alpha=0.1, color="green", label="Context" if i == 0 else None)
        # Dreamed continuation
        # Decode dreamed from symlog space for plotting
        dreamed_decoded = np.sign(dreamed[:, i]) * (np.exp(np.abs(dreamed[:, i])) - 1)
        ax.plot(t_dream, dreamed_decoded, "r--", linewidth=1.5, label=f"{name} (dreamed)")
        ax.axvline(context_len, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_ylabel(name)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"{title}: World Model Dream vs Ground Truth", fontsize=13)
    fig.tight_layout()
    save_fig(fig, f"wm_dream_{domain}")


def plot_dream_error_growth() -> None:
    """Plot dream error growth across domains."""
    print("\n=== Dream Error Growth ===")
    fig, ax = plt.subplots(figsize=(10, 5))

    domains = ["projectile", "lotka_volterra", "gray_scott"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    titles = ["Projectile", "Lotka-Volterra", "Gray-Scott"]

    for domain, color, title in zip(domains, colors, titles):
        path = WM_DIR / domain / "training_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            results = json.load(f)
        errors = results["dream_results"]["step_errors"]
        steps = np.arange(1, len(errors) + 1)
        ax.plot(steps, errors, color=color, linewidth=2, label=title, marker="o",
                markersize=3, alpha=0.8)

    ax.set_xlabel("Dream Step")
    ax.set_ylabel("MSE (symlog space)")
    ax.set_title("World Model Dream Error Growth")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "wm_dream_error_growth")


def plot_results_summary() -> None:
    """Plot summary bar chart of world model results."""
    print("\n=== World Model Summary ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    domains = []
    recon_losses = []
    dream_mses = []
    colors = []
    domain_colors = {"projectile": "#2196F3", "lotka_volterra": "#4CAF50", "gray_scott": "#FF9800"}
    domain_labels = {"projectile": "Projectile", "lotka_volterra": "Lotka-\nVolterra",
                     "gray_scott": "Gray-\nScott"}

    for domain in ["projectile", "lotka_volterra", "gray_scott"]:
        path = WM_DIR / domain / "training_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            r = json.load(f)
        domains.append(domain_labels[domain])
        recon_losses.append(r["final_recon"])
        dream_mses.append(r["dream_results"]["mse_symlog"])
        colors.append(domain_colors[domain])

    # Reconstruction loss
    bars = axes[0].bar(range(len(domains)), recon_losses, color=colors, edgecolor="k",
                       linewidth=0.5, alpha=0.85)
    axes[0].set_xticks(range(len(domains)))
    axes[0].set_xticklabels(domains)
    axes[0].set_ylabel("Reconstruction MSE (symlog)")
    axes[0].set_title("Final Reconstruction Loss")
    for bar, val in zip(bars, recon_losses):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Dream MSE
    bars = axes[1].bar(range(len(domains)), dream_mses, color=colors, edgecolor="k",
                       linewidth=0.5, alpha=0.85)
    axes[1].set_xticks(range(len(domains)))
    axes[1].set_xticklabels(domains)
    axes[1].set_ylabel("Dream MSE (symlog)")
    axes[1].set_title("Dream Quality (30 steps)")
    for bar, val in zip(bars, dream_mses):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("RSSM World Model Performance Summary", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "wm_summary")


def main():
    setup_paper_style()
    print("Generating world model figures...")

    plot_all_loss_curves()

    print("\n=== Projectile Dream ===")
    plot_dream_comparison_vector(
        "projectile", "Projectile",
        ["x (m)", "y (m)", "vx (m/s)", "vy (m/s)"]
    )

    print("\n=== Lotka-Volterra Dream ===")
    plot_dream_comparison_vector(
        "lotka_volterra", "Lotka-Volterra",
        ["Prey", "Predator"]
    )

    plot_dream_error_growth()
    plot_results_summary()

    n_figs = len(list(OUTPUT_DIR.glob("wm_*.png")))
    print(f"\nDone! Generated {n_figs} world model figures.")


if __name__ == "__main__":
    main()
