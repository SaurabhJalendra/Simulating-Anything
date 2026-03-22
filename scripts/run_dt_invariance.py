"""Run dt-invariance validation on all validated bifurcations."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

from simulating_anything.analysis.dt_invariance import run_dt_invariance

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    print("=" * 60)
    print("dt-INVARIANCE VALIDATION")
    print("Testing all validated bifurcations at dt/2 and dt*2")
    print("=" * 60)

    results = run_dt_invariance()

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print(f"\nPassed ({len(passed)}/{len(results)}):")
    for r in passed:
        print(f"  {r.domain}: {r.parameter}={r.crit_original:.4f}, "
              f"dev={r.max_deviation_pct:.1f}%")

    if failed:
        print(f"\nFailed ({len(failed)}/{len(results)}):")
        for r in failed:
            print(f"  {r.domain}: {r.parameter}={r.crit_original:.4f}, "
                  f"dev={r.max_deviation_pct:.1f}%")

    print(f"\nPass rate: {len(passed)}/{len(results)} "
          f"({len(passed)/len(results)*100:.0f}%)")


if __name__ == "__main__":
    main()
