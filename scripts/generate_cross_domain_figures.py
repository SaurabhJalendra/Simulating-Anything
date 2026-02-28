"""Generate publication-quality cross-domain analogy visualizations.

Creates 5 figures from the cross-domain analogy analysis:
  1. Analogy network graph (nodes = domains, edges = analogies)
  2. Similarity heatmap (14x14 clustered matrix)
  3. R-squared by domain (horizontal bar chart)
  4. Analogy type distribution (donut + histogram)
  5. Domain connectivity (analogies per domain)

Runs on CPU (no WSL needed). Reads live analysis results from
simulating_anything.analysis.cross_domain.

Usage:
    python scripts/generate_cross_domain_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("WARNING: networkx not installed. Skipping network graph figure.")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulating_anything.analysis.cross_domain import (  # noqa: E402
    run_cross_domain_analysis,
)

OUTPUT_DIR = Path("output/cross_domain_figures")

# Domain R-squared values (best per domain)
R2_BY_DOMAIN: dict[str, float] = {
    "projectile": 1.0000,
    "lotka_volterra": 1.0000,
    "gray_scott": 0.9851,
    "sir_epidemic": 1.0000,
    "double_pendulum": 0.9999,
    "harmonic_oscillator": 1.0000,
    "lorenz": 0.9999,
    "navier_stokes": 1.0000,
    "van_der_pol": 0.9999,
    "kuramoto": 0.9695,
    "brusselator": 0.9964,
    "fitzhugh_nagumo": 1.0000,
    "heat_equation": 1.0000,
    "logistic_map": 0.6287,
}

# Color palette for math types
MATH_TYPE_COLORS: dict[str, str] = {
    "algebraic": "#4CAF50",
    "ode_linear": "#2196F3",
    "ode_nonlinear": "#FF9800",
    "pde": "#9C27B0",
    "chaotic": "#F44336",
}

# Color palette for analogy edge types
EDGE_TYPE_COLORS: dict[str, str] = {
    "structural": "#2196F3",
    "dimensional": "#4CAF50",
    "topological": "#F44336",
}


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
    """Save figure as both PNG (300dpi) and PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def _pretty_domain_name(name: str) -> str:
    """Convert snake_case domain name to a readable label."""
    replacements = {
        "projectile": "Projectile",
        "lotka_volterra": "Lotka-Volterra",
        "gray_scott": "Gray-Scott",
        "sir_epidemic": "SIR Epidemic",
        "double_pendulum": "Double Pendulum",
        "harmonic_oscillator": "Harmonic Oscillator",
        "lorenz": "Lorenz",
        "navier_stokes": "Navier-Stokes",
        "van_der_pol": "Van der Pol",
        "kuramoto": "Kuramoto",
        "brusselator": "Brusselator",
        "fitzhugh_nagumo": "FitzHugh-Nagumo",
        "heat_equation": "Heat Equation",
        "logistic_map": "Logistic Map",
    }
    return replacements.get(name, name.replace("_", " ").title())


def _pretty_math_type(math_type: str) -> str:
    """Convert math_type to a readable label."""
    replacements = {
        "algebraic": "Algebraic",
        "ode_linear": "ODE (Linear)",
        "ode_nonlinear": "ODE (Nonlinear)",
        "pde": "PDE",
        "chaotic": "Chaotic",
    }
    return replacements.get(math_type, math_type)


# ---------------------------------------------------------------------------
# Figure 1: Analogy Network Graph
# ---------------------------------------------------------------------------
def fig_analogy_network(results: dict) -> None:
    """Network graph with 14 domain nodes and analogy edges."""
    if not HAS_NETWORKX:
        print("  Skipping analogy_network (networkx not installed)")
        return

    print("\n=== Analogy Network Graph ===")

    G = nx.Graph()

    # Build domain -> math_type lookup
    domain_math_type: dict[str, str] = {}
    for dname, info in results["domain_signatures"].items():
        domain_math_type[dname] = info["math_type"]
        G.add_node(dname)

    # Add edges from analogies
    edge_data: list[dict] = []
    for analogy in results["analogies"]:
        a, b = analogy["domains"]
        strength = analogy["strength"]
        atype = analogy["type"]
        G.add_edge(a, b, weight=strength, analogy_type=atype)
        edge_data.append({"a": a, "b": b, "strength": strength, "type": atype})

    fig, ax = plt.subplots(figsize=(12, 10))

    # Layout: spring layout with seed for reproducibility
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Draw edges with color by type and width by strength
    for ed in edge_data:
        a, b = ed["a"], ed["b"]
        color = EDGE_TYPE_COLORS.get(ed["type"], "#888888")
        width = 1.0 + 3.5 * ed["strength"]
        nx.draw_networkx_edges(
            G, pos, edgelist=[(a, b)], width=width,
            edge_color=color, alpha=0.6, ax=ax,
        )

    # Draw nodes colored by math type
    node_colors = [MATH_TYPE_COLORS.get(domain_math_type[n], "#999999") for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, node_size=900, node_color=node_colors,
        edgecolors="white", linewidths=2.0, ax=ax,
    )

    # Labels
    labels = {n: _pretty_domain_name(n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels, font_size=8, font_weight="bold", ax=ax,
    )

    # Legend for math types
    math_type_handles = [
        mpatches.Patch(color=color, label=_pretty_math_type(mt))
        for mt, color in MATH_TYPE_COLORS.items()
    ]
    # Legend for edge types
    from matplotlib.lines import Line2D
    edge_type_handles = [
        Line2D([0], [0], color=color, linewidth=2.5, label=etype.capitalize())
        for etype, color in EDGE_TYPE_COLORS.items()
    ]

    legend1 = ax.legend(
        handles=math_type_handles, title="Math Type (Nodes)",
        loc="upper left", framealpha=0.9,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=edge_type_handles, title="Analogy Type (Edges)",
        loc="lower left", framealpha=0.9,
    )

    ax.set_title(
        "Cross-Domain Analogy Network: 17 Mathematical Isomorphisms Across 14 Domains",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.axis("off")

    save(fig, "analogy_network")


# ---------------------------------------------------------------------------
# Figure 2: Similarity Heatmap
# ---------------------------------------------------------------------------
def fig_similarity_heatmap(results: dict) -> None:
    """14x14 heatmap of domain similarity, clustered by math type."""
    print("\n=== Similarity Heatmap ===")

    domain_names = results["similarity_matrix"]["domain_names"]
    matrix = np.array(results["similarity_matrix"]["matrix"])

    # Reorder domains by math type so similar types are adjacent
    domain_math_type = {
        d: results["domain_signatures"][d]["math_type"] for d in domain_names
    }
    type_order = ["algebraic", "ode_linear", "ode_nonlinear", "pde", "chaotic"]
    ordered_indices = []
    for mt in type_order:
        for i, d in enumerate(domain_names):
            if domain_math_type[d] == mt:
                ordered_indices.append(i)
    # Include any domains not in type_order
    for i in range(len(domain_names)):
        if i not in ordered_indices:
            ordered_indices.append(i)

    ordered_names = [domain_names[i] for i in ordered_indices]
    ordered_matrix = matrix[np.ix_(ordered_indices, ordered_indices)]
    pretty_names = [_pretty_domain_name(n) for n in ordered_names]

    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(ordered_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

    # Annotate cells with strength values (skip diagonal and zero)
    for i in range(len(ordered_names)):
        for j in range(len(ordered_names)):
            val = ordered_matrix[i, j]
            if val > 0 and i != j:
                text_color = "white" if val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(pretty_names)))
    ax.set_xticklabels(pretty_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pretty_names)))
    ax.set_yticklabels(pretty_names, fontsize=9)

    # Draw math type group boundaries
    group_boundaries = []
    current_type = domain_math_type[ordered_names[0]]
    for i, d in enumerate(ordered_names):
        if domain_math_type[d] != current_type:
            group_boundaries.append(i - 0.5)
            current_type = domain_math_type[d]

    for boundary in group_boundaries:
        ax.axhline(y=boundary, color="black", linewidth=1.5, alpha=0.5)
        ax.axvline(x=boundary, color="black", linewidth=1.5, alpha=0.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Analogy Strength")
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "Domain Similarity Matrix (Clustered by Math Type)",
        fontsize=13, fontweight="bold", pad=15,
    )

    fig.tight_layout()
    save(fig, "similarity_heatmap")


# ---------------------------------------------------------------------------
# Figure 3: R-squared by Domain
# ---------------------------------------------------------------------------
def fig_r2_by_domain() -> None:
    """Horizontal bar chart of best R-squared per domain."""
    print("\n=== R-squared by Domain ===")

    # Sort domains by R-squared descending
    sorted_domains = sorted(R2_BY_DOMAIN.items(), key=lambda x: x[1], reverse=True)
    domain_names = [d[0] for d in sorted_domains]
    r2_values = [d[1] for d in sorted_domains]
    pretty_names = [_pretty_domain_name(d) for d in domain_names]

    # Lookup math type for coloring
    from simulating_anything.analysis.cross_domain import build_domain_signatures
    sigs = build_domain_signatures()
    sig_lookup = {s.name: s.math_type for s in sigs}
    bar_colors = [
        MATH_TYPE_COLORS.get(sig_lookup.get(d, ""), "#999999") for d in domain_names
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(domain_names))
    bars = ax.barh(y_pos, r2_values, color=bar_colors, edgecolor="white",
                   linewidth=0.5, height=0.7)

    # Reference line at R-squared = 0.999
    ax.axvline(x=0.999, color="#555555", linestyle="--", linewidth=1.2,
               alpha=0.7, label="R$^2$ = 0.999")

    # Show exact R-squared value at end of each bar
    for i, (bar, val) in enumerate(zip(bars, r2_values)):
        x_offset = 0.003 if val < 0.95 else -0.04
        text_color = "black" if val < 0.95 else "white"
        ha = "left" if val < 0.95 else "right"
        ax.text(
            val + x_offset, i, f"{val:.4f}",
            ha=ha, va="center", fontsize=9, fontweight="bold", color=text_color,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pretty_names, fontsize=10)
    ax.set_xlabel("Best R$^2$", fontsize=11)
    ax.set_xlim(0.55, 1.02)
    ax.invert_yaxis()

    # Legend for math types
    legend_handles = [
        mpatches.Patch(color=color, label=_pretty_math_type(mt))
        for mt, color in MATH_TYPE_COLORS.items()
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.2,
                   label="R$^2$ = 0.999")
    )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_title(
        "Symbolic Regression Accuracy Across 14 Domains",
        fontsize=13, fontweight="bold", pad=15,
    )

    fig.tight_layout()
    save(fig, "r2_by_domain")


# ---------------------------------------------------------------------------
# Figure 4: Analogy Type Distribution
# ---------------------------------------------------------------------------
def fig_analogy_distribution(results: dict) -> None:
    """Donut chart of analogy types + histogram of analogy strengths."""
    print("\n=== Analogy Type Distribution ===")

    analogy_counts = results["analogy_types"]
    all_analogies = results["analogies"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Donut chart ---
    labels = [atype.capitalize() for atype in analogy_counts.keys()]
    counts = list(analogy_counts.values())
    colors = [EDGE_TYPE_COLORS[atype] for atype in analogy_counts.keys()]
    total = sum(counts)

    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.78,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    # Center text with total count
    ax1.text(0, 0, f"{total}\nanalogies", ha="center", va="center",
             fontsize=14, fontweight="bold", color="#333333")
    ax1.set_title("Analogy Type Distribution", fontsize=12, fontweight="bold", pad=15)

    # --- Right: Histogram of analogy strengths ---
    strengths = [a["strength"] for a in all_analogies]
    type_labels = [a["type"] for a in all_analogies]

    # Separate strengths by type for stacked histogram
    strength_by_type: dict[str, list[float]] = {
        "structural": [], "dimensional": [], "topological": [],
    }
    for s, t in zip(strengths, type_labels):
        strength_by_type[t].append(s)

    bins = np.arange(0.55, 1.05, 0.05)
    bottom = np.zeros(len(bins) - 1)
    for atype in ["structural", "dimensional", "topological"]:
        vals = strength_by_type[atype]
        if vals:
            hist_vals, _ = np.histogram(vals, bins=bins)
            ax2.bar(
                bins[:-1] + 0.025, hist_vals, width=0.045, bottom=bottom,
                color=EDGE_TYPE_COLORS[atype], label=atype.capitalize(),
                edgecolor="white", linewidth=0.5,
            )
            bottom += hist_vals

    ax2.set_xlabel("Analogy Strength", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Analogy Strength Distribution", fontsize=12, fontweight="bold", pad=15)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.set_xlim(0.55, 1.05)

    fig.tight_layout()
    save(fig, "analogy_distribution")


# ---------------------------------------------------------------------------
# Figure 5: Domain Connectivity
# ---------------------------------------------------------------------------
def fig_domain_connectivity(results: dict) -> None:
    """Bar chart showing number of analogies per domain."""
    print("\n=== Domain Connectivity ===")

    # Count analogies per domain
    connectivity: dict[str, int] = {}
    for d in results["domain_signatures"]:
        connectivity[d] = 0
    for analogy in results["analogies"]:
        a, b = analogy["domains"]
        connectivity[a] = connectivity.get(a, 0) + 1
        connectivity[b] = connectivity.get(b, 0) + 1

    # Sort by connectivity descending
    sorted_domains = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)
    domain_names = [d[0] for d in sorted_domains]
    counts = [d[1] for d in sorted_domains]
    pretty_names = [_pretty_domain_name(d) for d in domain_names]

    # Math type lookup for coloring
    domain_math_type = {
        d: results["domain_signatures"][d]["math_type"] for d in domain_names
    }
    bar_colors = [
        MATH_TYPE_COLORS.get(domain_math_type.get(d, ""), "#999999")
        for d in domain_names
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(domain_names))
    bars = ax.barh(y_pos, counts, color=bar_colors, edgecolor="white",
                   linewidth=0.5, height=0.7)

    # Annotate bar ends with count
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + 0.15, i, str(count), ha="left", va="center",
                fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pretty_names, fontsize=10)
    ax.set_xlabel("Number of Analogies", fontsize=11)
    ax.set_xlim(0, max(counts) + 1.5)
    ax.invert_yaxis()

    # Legend for math types
    legend_handles = [
        mpatches.Patch(color=color, label=_pretty_math_type(mt))
        for mt, color in MATH_TYPE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_title(
        "Domain Connectivity: Analogies Per Domain",
        fontsize=13, fontweight="bold", pad=15,
    )

    fig.tight_layout()
    save(fig, "domain_connectivity")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate all cross-domain figures."""
    setup_style()

    print("Running cross-domain analogy analysis...")
    results = run_cross_domain_analysis(output_dir="output/cross_domain")
    print(f"  Found {results['n_analogies']} analogies across {results['n_domains']} domains")

    fig_analogy_network(results)
    fig_similarity_heatmap(results)
    fig_r2_by_domain()
    fig_analogy_distribution(results)
    fig_domain_connectivity(results)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("Files generated:")
    for name in [
        "analogy_network", "similarity_heatmap", "r2_by_domain",
        "analogy_distribution", "domain_connectivity",
    ]:
        print(f"  {name}.png (300dpi)")
        print(f"  {name}.pdf (vector)")


if __name__ == "__main__":
    main()
