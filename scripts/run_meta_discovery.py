"""Run meta-analysis across all 192 domain rediscovery results.

Performs four cross-cutting analyses:
1. Universal Bifurcation Analysis (Lyapunov-based dynamics classification)
2. Hopf Scaling Law verification (sqrt(mu - mu_c) amplitude)
3. R-squared Distribution Analysis (by class, method, domain type)
4. Equation Complexity Analysis (PySR complexity vs R-squared)

Usage:
    python scripts/run_meta_discovery.py
    python scripts/run_meta_discovery.py --output output/meta_discovery
    python scripts/run_meta_discovery.py --rediscovery-dir output/rediscovery
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simulating_anything.analysis.meta_discovery import run_meta_discovery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta-analysis across all domain rediscovery results.",
    )
    parser.add_argument(
        "--output",
        default="output/meta_discovery",
        help="Output directory for analysis results (default: output/meta_discovery)",
    )
    parser.add_argument(
        "--rediscovery-dir",
        default=None,
        help="Path to rediscovery results (default: output/rediscovery)",
    )
    args = parser.parse_args()

    summary = run_meta_discovery(
        output_dir=args.output,
        rediscovery_dir=args.rediscovery_dir,
    )
    n = summary.get("n_domains_loaded", 0)
    print(f"\nDone. Analyzed {n} domains.")


if __name__ == "__main__":
    main()
