"""Generate pipeline architecture diagram for the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = Path("output/paper_figures")


def generate_pipeline_figure():
    """Create a clean pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-1.5, 2.5)

    # Stage definitions
    stages = [
        ("Problem\nArchitect", "#E3F2FD", "#1565C0"),
        ("Domain\nClassifier", "#E3F2FD", "#1565C0"),
        ("Simulation\nBuilder", "#E3F2FD", "#1565C0"),
        ("Ground-Truth\nSimulation", "#FFF3E0", "#E65100"),  # Domain-specific
        ("Exploration\n(RSSM)", "#E8F5E9", "#2E7D32"),
        ("Analysis\n(PySR/SINDy)", "#E8F5E9", "#2E7D32"),
        ("Communication\nAgent", "#E3F2FD", "#1565C0"),
    ]

    box_w = 1.6
    box_h = 1.0
    spacing = 0.35
    start_x = 0

    for i, (label, bg_color, border_color) in enumerate(stages):
        x = start_x + i * (box_w + spacing)
        y = 0.5

        rect = mpatches.FancyBboxPatch(
            (x, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=bg_color, edgecolor=border_color, linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y, label,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=border_color)

        # Stage number
        ax.text(x + box_w / 2, y + box_h / 2 + 0.15, f"Stage {i + 1}",
                ha="center", va="bottom", fontsize=7, color="gray")

        # Arrow between stages
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + box_w + spacing * 0.2, y),
                        xytext=(x + box_w + spacing * 0.05, y),
                        arrowprops=dict(arrowstyle="->", color="#666",
                                       lw=1.5))

    # Input/output labels
    ax.text(-0.3, 0.5, "Natural\nLanguage\nQuery", ha="right", va="center",
            fontsize=8, style="italic", color="#666")
    ax.annotate("", xy=(0, 0.5), xytext=(-0.15, 0.5),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    last_x = start_x + 6 * (box_w + spacing) + box_w
    ax.text(last_x + 0.3, 0.5, "Discovery\nReport\n(Equations)", ha="left",
            va="center", fontsize=8, style="italic", color="#666")
    ax.annotate("", xy=(last_x + 0.15, 0.5), xytext=(last_x + 0.05, 0.5),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    # Legend
    legend_y = -1.0
    patches = [
        mpatches.Patch(facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2,
                       label="LLM Agent (domain-agnostic)"),
        mpatches.Patch(facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2,
                       label="Domain-specific (only component)"),
        mpatches.Patch(facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2,
                       label="ML + Analysis (domain-agnostic)"),
    ]
    ax.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
              frameon=False, bbox_to_anchor=(0.5, -0.35))

    # Title
    ax.set_title("SimAnything: Seven-Stage Discovery Pipeline", fontsize=12,
                 fontweight="bold", pad=15)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "pipeline.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "pipeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pipeline figure to {OUTPUT_DIR / 'pipeline.pdf'}")


if __name__ == "__main__":
    generate_pipeline_figure()
