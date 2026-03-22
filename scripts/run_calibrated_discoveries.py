"""Run discovery campaigns with literature-calibrated parameters.

Compares discovered bifurcation thresholds against published values.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

from simulating_anything.analysis.literature_calibration import run_calibrated_campaign

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


CAMPAIGNS = [
    # ── Brusselator (exact analytical benchmark) ──
    {
        "domain": "brusselator",
        "sweep_param": "b",
        "sweep_range": (0.5, 4.0),
        "sim_module": "brusselator",
        "sim_class": "BrusselatorSimulation",
        "question": "Calibration: detect Hopf at b_c = 1 + a^2 = 2.0 (a=1.0)",
        "n_points": 300,
        "n_steps": 20000,
        "dt": 0.001,
    },
    # ── Brusselator at a=1.5 (b_c = 3.25) ──
    {
        "domain": "brusselator",
        "sweep_param": "b",
        "sweep_range": (1.0, 6.0),
        "sim_module": "brusselator",
        "sim_class": "BrusselatorSimulation",
        "question": "Calibration: detect Hopf at b_c = 1 + 1.5^2 = 3.25 (a=1.5)",
        "n_points": 300,
        "n_steps": 20000,
        "dt": 0.001,
        "extra_params": {"a": 1.5},
    },
    # ── Brusselator at a=2.0 (b_c = 5.0) ──
    {
        "domain": "brusselator",
        "sweep_param": "b",
        "sweep_range": (2.0, 8.0),
        "sim_module": "brusselator",
        "sim_class": "BrusselatorSimulation",
        "question": "Calibration: detect Hopf at b_c = 1 + 2^2 = 5.0 (a=2.0)",
        "n_points": 300,
        "n_steps": 20000,
        "dt": 0.001,
        "extra_params": {"a": 2.0},
    },
    # ── NPB: burst size sweep (literature: oscillations ~50-100) ──
    {
        "domain": "nutrient_phage_bacteria",
        "sweep_param": "burst",
        "sweep_range": (5.0, 200.0),
        "sim_module": "nutrient_phage_bacteria",
        "sim_class": "NutrientPhageBacteriaSimulation",
        "question": "Calibration: phage burst size for kill-the-winner oscillations",
        "n_points": 300,
        "n_steps": 10000,
        "dt": 0.01,
    },
    # ── NPB: dilution rate sweep (literature: oscillations ~0.35) ──
    {
        "domain": "nutrient_phage_bacteria",
        "sweep_param": "D_dilution",
        "sweep_range": (0.01, 0.8),
        "sim_module": "nutrient_phage_bacteria",
        "sim_class": "NutrientPhageBacteriaSimulation",
        "question": "Calibration: dilution rate for chemostat oscillation onset",
        "n_points": 300,
        "n_steps": 10000,
        "dt": 0.01,
    },
    # ── Tumor-Immune: growth rate sweep (literature: escape ~0.20) ──
    {
        "domain": "tumor_immune",
        "sweep_param": "a_t",
        "sweep_range": (0.05, 0.5),
        "sim_module": "tumor_immune",
        "sim_class": "TumorImmuneSimulation",
        "question": "Calibration: tumor growth rate for immune escape",
        "n_points": 300,
        "n_steps": 10000,
        "dt": 0.01,
    },
]


def main():
    print("=" * 60)
    print("LITERATURE CALIBRATION — Comparing Discoveries to Known Physics")
    print("=" * 60)

    all_results = []

    for i, camp in enumerate(CAMPAIGNS):
        domain = camp["domain"]
        param = camp["sweep_param"]
        print(f"\n--- Campaign {i+1}/{len(CAMPAIGNS)}: {domain}.{param} ---")
        print(f"    Question: {camp['question']}")

        # Handle extra params for Brusselator a variations
        extra = camp.get("extra_params", {})

        result = run_calibrated_campaign(
            domain=domain,
            sweep_param=param,
            sweep_range=camp["sweep_range"],
            sim_module=camp["sim_module"],
            sim_class=camp["sim_class"],
            question=camp["question"],
            n_points=camp.get("n_points", 300),
            n_steps=camp.get("n_steps", 10000),
            dt=camp.get("dt", 0.005),
        )

        all_results.append(result)

        # Print comparisons
        for c in result.comparisons:
            status = "MATCH" if c.match else "MISS"
            print(f"    [{status}] {c.parameter}: discovered={c.discovered_value:.4f}, "
                  f"literature={c.literature_value:.4f}, error={c.error_pct:.1f}%")

        if not result.comparisons:
            print(f"    No matching literature thresholds for {param}")
            print(f"    Discovered {len(result.discovered_thresholds)} bifurcations:")
            for d in result.discovered_thresholds:
                print(f"      {d.evidence.get('type', '?')} at {d.parameter}={d.critical_value:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    total_comparisons = sum(len(r.comparisons) for r in all_results)
    total_matches = sum(sum(1 for c in r.comparisons if c.match) for r in all_results)
    errors = [c.error_pct for r in all_results for c in r.comparisons if not c.discovered_value != c.discovered_value]

    print(f"  Total comparisons: {total_comparisons}")
    print(f"  Matches (within 20%): {total_matches}/{total_comparisons}")
    if errors:
        print(f"  Mean error: {sum(errors)/len(errors):.1f}%")
        print(f"  Min error: {min(errors):.1f}%")
        print(f"  Max error: {max(errors):.1f}%")


if __name__ == "__main__":
    main()
