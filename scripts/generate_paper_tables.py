#!/usr/bin/env python3
"""Generate all paper tables and statistics from code.

Run: python scripts/generate_paper_tables.py

Produces:
  output/paper_tables/baseline_results.json
  output/paper_tables/baselines.tex
  output/paper_tables/significance.tex
  output/paper_tables/robustness.tex
  output/paper_tables/cost.tex
  output/paper_tables/summary.txt
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from simulating_anything.analysis.baselines import (
    generate_latex_table,
    run_all_baselines,
    save_results,
)
from simulating_anything.analysis.robustness import run_robustness_analysis
from simulating_anything.analysis.significance import (
    bootstrap_r_squared,
    permutation_test_r_squared,
    run_significance_analysis,
)
from simulating_anything.analysis.computational_cost import (
    measure_fitting_cost,
    measure_scaling,
    run_cost_analysis,
)


OUTPUT_DIR = Path("output/paper_tables")


def generate_baselines():
    """Generate baseline comparison table."""
    print("=" * 60)
    print("BASELINE COMPARISONS")
    print("=" * 60)

    comparisons = run_all_baselines()
    save_results(comparisons, OUTPUT_DIR)

    latex = generate_latex_table(comparisons)
    (OUTPUT_DIR / "baselines.tex").write_text(latex)
    print(f"  Generated baselines.tex ({len(comparisons)} domains)")

    for comp in comparisons:
        print(f"\n  {comp.domain}: {comp.target_equation}")
        for r in comp.results:
            check = "Y" if r.correct_form else "N"
            print(f"    {r.method:30s}  R²={r.r_squared:.6f}  form={check}  n={r.n_samples}")


def generate_significance():
    """Generate significance tests table."""
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 60)

    domains = {
        "projectile": {
            "gen": lambda n: _projectile_data(n),
            "fit": lambda v, t, r: _projectile_fit(v, t, r),
        },
        "harmonic_oscillator": {
            "gen": lambda n: _oscillator_data(n),
            "fit": lambda k, m, o: _oscillator_fit(k, m, o),
        },
    }

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Statistical significance of rediscovery results.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Domain & $R^2$ & $p$-value & 95\% CI & Significant \\",
        r"\midrule",
    ]

    for name, info in domains.items():
        data = info["gen"](225)
        y_true, y_pred = info["fit"](*data)
        report = run_significance_analysis(name, y_true, y_pred, n_permutations=5000)

        ci_low, ci_high = report.bootstrap_ci
        sig = r"\checkmark" if report.permutation_test.significant else r"$\times$"
        lines.append(
            f"  {name.replace('_', ' ').title()} & "
            f"{report.r_squared:.6f} & "
            f"{report.permutation_test.p_value:.4f} & "
            f"[{ci_low:.4f}, {ci_high:.4f}] & "
            f"{sig} \\\\"
        )
        print(f"  {name}: R²={report.r_squared:.6f}, p={report.permutation_test.p_value:.4f}, "
              f"CI=[{ci_low:.4f}, {ci_high:.4f}]")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT_DIR / "significance.tex").write_text("\n".join(lines))
    print(f"  Generated significance.tex")


def generate_robustness():
    """Generate robustness analysis table."""
    print("\n" + "=" * 60)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 60)

    report = run_robustness_analysis()

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Robustness analysis results.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Domain & Test & Metric & Threshold & Pass? \\",
        r"\midrule",
    ]

    for t in report.individual_tests:
        check = r"\checkmark" if t.passed else r"$\times$"
        lines.append(
            f"  {t.domain.replace('_', ' ').title()} & "
            f"{t.test_name.replace('_', ' ')} & "
            f"{t.metric_value:.4f} & {t.threshold:.2f} & {check} \\\\"
        )
        status = "PASS" if t.passed else "FAIL"
        print(f"  [{status}] {t.domain}.{t.test_name}: {t.details}")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT_DIR / "robustness.tex").write_text("\n".join(lines))
    print(f"\n  Pass rate: {report.pass_rate():.1%} ({report.n_passed()}/{len(report.individual_tests)})")
    print(f"  Generated robustness.tex")

    # Noise tolerance details
    for nr in report.noise_results:
        print(f"\n  Noise tolerance ({nr.domain}):")
        for nl, r2 in zip(nr.noise_levels, nr.r_squared_values):
            print(f"    {nl:6.1%} noise -> R²={r2:.6f}")


def generate_cost():
    """Generate computational cost table."""
    print("\n" + "=" * 60)
    print("COMPUTATIONAL COST")
    print("=" * 60)

    cost_report = run_cost_analysis(n_steps=1000, n_sweeps=10)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Computational cost per domain (1000 steps, 10 sweeps).}",
        r"\label{tab:cost}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Domain & Import (s) & Single Run (s) & 10-Sweep (s) \\",
        r"\midrule",
    ]

    for dc in cost_report.domains:
        import_t = next((s.wall_time_s for s in dc.stages if s.stage == "import"), 0)
        single_t = next((s.wall_time_s for s in dc.stages if s.stage == "single_run"), 0)
        sweep_t = next((s.wall_time_s for s in dc.stages if s.stage == "param_sweep"), 0)
        lines.append(
            f"  {dc.domain.replace('_', ' ').title()} & "
            f"{import_t:.4f} & {single_t:.4f} & {sweep_t:.4f} \\\\"
        )
        print(f"  {dc.domain:25s}  import={import_t:.4f}s  single={single_t:.4f}s  sweep={sweep_t:.4f}s")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT_DIR / "cost.tex").write_text("\n".join(lines))
    print(f"\n  Total time: {cost_report.total_time_s:.2f}s")
    print(f"  Generated cost.tex")

    # Scaling analysis
    print("\n  Fitting cost scaling:")
    scaling = measure_scaling()
    for s in scaling:
        print(f"    n={s.n_samples:6d}  time={s.wall_time_s:.6f}s  {s.notes}")


def generate_summary():
    """Generate summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = [
        "Paper Table Generation Summary",
        "=" * 40,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Output directory: {OUTPUT_DIR}",
        "",
        "Files generated:",
        "  - baseline_results.json",
        "  - baselines.tex",
        "  - significance.tex",
        "  - robustness.tex",
        "  - cost.tex",
        "  - summary.txt",
        "",
        "To include in paper:",
        r"  \input{output/paper_tables/baselines.tex}",
        r"  \input{output/paper_tables/significance.tex}",
        r"  \input{output/paper_tables/robustness.tex}",
        r"  \input{output/paper_tables/cost.tex}",
    ]

    text = "\n".join(summary)
    (OUTPUT_DIR / "summary.txt").write_text(text)
    print(text)


# Helper functions for significance tests
def _projectile_data(n):
    n_per = max(2, int(np.sqrt(n)))
    v0 = np.linspace(5, 50, n_per)
    theta = np.linspace(5, 85, n_per)
    V, T = np.meshgrid(v0, theta)
    return V.ravel(), T.ravel(), V.ravel()**2 * np.sin(2 * np.radians(T.ravel())) / 9.81


def _projectile_fit(v0, theta, R):
    feature = v0**2 * np.sin(2 * np.radians(theta))
    c = np.sum(feature * R) / np.sum(feature**2)
    return R, c * feature


def _oscillator_data(n):
    n_per = max(2, int(np.sqrt(n)))
    k = np.linspace(0.5, 20, n_per)
    m = np.linspace(0.5, 10, n_per)
    K, M = np.meshgrid(k, m)
    return K.ravel(), M.ravel(), np.sqrt(K.ravel() / M.ravel())


def _oscillator_fit(k, m, omega):
    feature = np.sqrt(k / m)
    c = np.sum(feature * omega) / np.sum(feature**2)
    return omega, c * feature


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    generate_baselines()
    generate_significance()
    generate_robustness()
    generate_cost()
    generate_summary()

    total = time.monotonic() - t0
    print(f"\nTotal generation time: {total:.1f}s")
