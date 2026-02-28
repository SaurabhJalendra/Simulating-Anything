"""Aggregate all 14-domain rediscovery results into a unified report.

Reads output/rediscovery/*/results.json and produces:
1. A consolidated JSON with all results
2. A Markdown table suitable for the paper
3. Statistical summary (mean R², methods used, etc.)

Usage:
    python scripts/aggregate_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

RESULTS_DIR = Path("output/rediscovery")
OUTPUT_DIR = Path("output")


# Domain display metadata
DOMAIN_META = {
    "projectile": {"name": "Projectile", "class": "Algebraic", "target": "R=v^2sin(2t)/g"},
    "lotka_volterra": {"name": "Lotka-Volterra", "class": "Nonlinear ODE", "target": "Equilibrium + ODE"},
    "gray_scott": {"name": "Gray-Scott", "class": "PDE", "target": "Wavelength scaling"},
    "sir_epidemic": {"name": "SIR Epidemic", "class": "Nonlinear ODE", "target": "R0=b/g + ODE"},
    "double_pendulum": {"name": "Double Pendulum", "class": "Chaotic ODE", "target": "T=2pi*sqrt(L/g)"},
    "harmonic_oscillator": {"name": "Harmonic Oscillator", "class": "Linear ODE", "target": "w0=sqrt(k/m)"},
    "lorenz": {"name": "Lorenz Attractor", "class": "Chaotic ODE", "target": "Full 3-eq system"},
    "navier_stokes": {"name": "Navier-Stokes 2D", "class": "PDE", "target": "Decay rate=2v|k|^2"},
    "van_der_pol": {"name": "Van der Pol", "class": "Nonlinear ODE", "target": "T(mu), A~2"},
    "kuramoto": {"name": "Kuramoto", "class": "Collective ODE", "target": "r(K) transition"},
    "brusselator": {"name": "Brusselator", "class": "Nonlinear ODE", "target": "b_c=1+a^2 + ODE"},
    "fitzhugh_nagumo": {"name": "FitzHugh-Nagumo", "class": "Nonlinear ODE", "target": "Full ODE + f-I"},
    "heat_equation": {"name": "Heat Equation 1D", "class": "Linear PDE", "target": "lambda_k=D*k^2"},
    "logistic_map": {"name": "Logistic Map", "class": "Discrete Chaos", "target": "Feigenbaum + lambda"},
}


def extract_best_r2(result: dict) -> float | None:
    """Extract the best R² value from a results dict.

    Handles multiple result formats:
    - top-level: best_r_squared (projectile)
    - nested: sindy_ode.discoveries[0].r_squared (lorenz)
    - nested: scaling_analysis.best_scaling_r2 (gray_scott)
    - nested: sub_dict.best_r2 (most domains)
    """
    best = None

    # Top-level R² (projectile format)
    if "best_r_squared" in result:
        val = result["best_r_squared"]
        if val is not None:
            best = val

    # Nested dicts with best_r2
    for key in result:
        if isinstance(result[key], dict):
            # Standard best_r2
            if "best_r2" in result[key]:
                val = result[key]["best_r2"]
                if val is not None and (best is None or val > best):
                    best = val
            # Gray-Scott format
            if "best_scaling_r2" in result[key]:
                val = result[key]["best_scaling_r2"]
                if val is not None and (best is None or val > best):
                    best = val
            # SINDy with discoveries list (lorenz, etc.)
            if "discoveries" in result[key] and "r_squared" not in result[key]:
                for disc in result[key].get("discoveries", []):
                    if isinstance(disc, dict) and "r_squared" in disc:
                        val = disc["r_squared"]
                        if val is not None and (best is None or val > best):
                            best = val

    return best


def extract_methods(result: dict) -> list[str]:
    """Extract which methods were used."""
    methods = set()

    # Check for PySR/SINDy results in any nested dict
    for key in result:
        k_low = key.lower()
        if isinstance(result[key], dict):
            has_content = ("discoveries" in result[key] or "best_r2" in result[key]
                          or "best_scaling_r2" in result[key])
            if has_content:
                if "sindy" in k_low:
                    methods.add("SINDy")
                elif "pysr" in k_low or "scaling" in k_low:
                    methods.add("PySR")

    # Top-level discoveries (projectile format)
    if "discoveries" in result or "best_equation" in result:
        methods.add("PySR")

    # Check for SINDy in nested structures
    for key in result:
        if "sindy" in key.lower() and isinstance(result[key], dict):
            if "discoveries" in result[key] or "n_discoveries" in result[key]:
                methods.add("SINDy")

    return sorted(methods) if methods else ["Analysis"]


def extract_best_expression(result: dict) -> str:
    """Extract the best discovered expression."""
    best_r2 = -1.0
    best_expr = "N/A"

    # Top-level (projectile)
    if "best_equation" in result:
        val = result.get("best_r_squared", 0)
        if val is not None and val > best_r2:
            best_r2 = val
            best_expr = result["best_equation"]

    # Nested dicts
    for key in result:
        if isinstance(result[key], dict):
            # Standard best + best_r2
            if "best" in result[key] and "best_r2" in result[key]:
                r2 = result[key]["best_r2"]
                if r2 is not None and r2 > best_r2:
                    best_r2 = r2
                    best_expr = result[key]["best"]
            # Gray-Scott format
            if "best_scaling_equation" in result[key] and "best_scaling_r2" in result[key]:
                r2 = result[key]["best_scaling_r2"]
                if r2 is not None and r2 > best_r2:
                    best_r2 = r2
                    best_expr = result[key]["best_scaling_equation"]
            # SINDy discoveries list
            if "discoveries" in result[key] and "best" not in result[key]:
                for disc in result[key].get("discoveries", []):
                    if isinstance(disc, dict) and "r_squared" in disc:
                        r2 = disc["r_squared"]
                        if r2 is not None and r2 > best_r2:
                            best_r2 = r2
                            best_expr = disc.get("expression", "N/A")

    return best_expr


def aggregate():
    """Read all results and produce unified report."""
    all_results = {}
    summary_rows = []

    for domain_dir in sorted(RESULTS_DIR.iterdir()):
        results_file = domain_dir / "results.json"
        if not results_file.exists():
            continue

        domain = domain_dir.name
        with open(results_file) as f:
            result = json.load(f)

        best_r2 = extract_best_r2(result)
        methods = extract_methods(result)
        best_expr = extract_best_expression(result)
        meta = DOMAIN_META.get(domain, {"name": domain, "class": "?", "target": "?"})

        all_results[domain] = {
            "name": meta["name"],
            "math_class": meta["class"],
            "target": meta["target"],
            "best_r2": best_r2,
            "methods": methods,
            "best_expression": best_expr,
            "full_results": result,
        }

        summary_rows.append({
            "domain": meta["name"],
            "class": meta["class"],
            "method": " + ".join(methods),
            "best_r2": best_r2,
            "expression": best_expr[:50] + "..." if len(best_expr) > 50 else best_expr,
        })

    # Statistics
    r2_values = [r["best_r2"] for r in all_results.values() if r["best_r2"] is not None]
    n_perfect = sum(1 for r in r2_values if r >= 0.999)
    n_high = sum(1 for r in r2_values if r >= 0.99)

    stats = {
        "total_domains": len(all_results),
        "domains_with_r2": len(r2_values),
        "mean_r2": sum(r2_values) / len(r2_values) if r2_values else 0,
        "median_r2": sorted(r2_values)[len(r2_values) // 2] if r2_values else 0,
        "min_r2": min(r2_values) if r2_values else 0,
        "max_r2": max(r2_values) if r2_values else 0,
        "n_r2_above_999": n_perfect,
        "n_r2_above_99": n_high,
        "methods_used": list(set(m for r in all_results.values() for m in r["methods"])),
    }

    # Save consolidated JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    consolidated = {"statistics": stats, "domains": {}}
    for k, v in all_results.items():
        consolidated["domains"][k] = {
            "name": v["name"],
            "math_class": v["math_class"],
            "target": v["target"],
            "best_r2": v["best_r2"],
            "methods": v["methods"],
            "best_expression": v["best_expression"],
        }

    with open(OUTPUT_DIR / "rediscovery_summary.json", "w") as f:
        json.dump(consolidated, f, indent=2)

    # Generate Markdown table
    md_lines = [
        "# 14-Domain Rediscovery Results",
        "",
        "| # | Domain | Math Class | Method | Best R² | Key Discovery |",
        "|---|--------|------------|--------|---------|---------------|",
    ]

    for i, row in enumerate(summary_rows, 1):
        r2_str = f"{row['best_r2']:.6f}" if row["best_r2"] is not None else "N/A"
        md_lines.append(
            f"| {i} | {row['domain']} | {row['class']} | "
            f"{row['method']} | {r2_str} | `{row['expression']}` |"
        )

    md_lines.extend([
        "",
        "## Statistics",
        "",
        f"- **Total domains:** {stats['total_domains']}",
        f"- **Mean R²:** {stats['mean_r2']:.6f}",
        f"- **Median R²:** {stats['median_r2']:.6f}",
        f"- **R² >= 0.999:** {stats['n_r2_above_999']} domains",
        f"- **R² >= 0.99:** {stats['n_r2_above_99']} domains",
        f"- **Methods:** {', '.join(stats['methods_used'])}",
    ])

    with open(OUTPUT_DIR / "rediscovery_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    # Print summary
    print("\n" + "=" * 90)
    print("SIMULATING ANYTHING -- 14-DOMAIN REDISCOVERY RESULTS")
    print("=" * 90)
    print(f"{'#':>2}  {'Domain':<22} {'Class':<16} {'Method':<14} {'Best R²':<12}")
    print("-" * 90)

    for i, row in enumerate(summary_rows, 1):
        r2_str = f"{row['best_r2']:.6f}" if row["best_r2"] is not None else "N/A"
        print(f"{i:>2}  {row['domain']:<22} {row['class']:<16} {row['method']:<14} {r2_str:<12}")

    print("-" * 90)
    print(f"Mean R²: {stats['mean_r2']:.6f} | "
          f"R²>=0.999: {stats['n_r2_above_999']}/14 | "
          f"R²>=0.99: {stats['n_r2_above_99']}/14")
    print("=" * 90)

    print(f"\nSaved: {OUTPUT_DIR / 'rediscovery_summary.json'}")
    print(f"Saved: {OUTPUT_DIR / 'rediscovery_summary.md'}")


if __name__ == "__main__":
    aggregate()
