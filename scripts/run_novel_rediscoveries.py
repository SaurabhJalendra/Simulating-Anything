"""Run rediscovery analysis for novel V7 domains.

Runs PySR + SINDy equation recovery for the 2 new novel coupled systems:
- climate_epidemic: ENSO-driven disease dynamics (6D)
- neural_cardiac: brain-heart coupled oscillators (4D)

Must run in WSL2 for PySR (Julia) and PySINDy.

Usage:
    python scripts/run_novel_rediscoveries.py
    python scripts/run_novel_rediscoveries.py --domains climate_epidemic
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run novel domain rediscoveries"
    )
    parser.add_argument(
        "--domains",
        default="climate_epidemic,neural_cardiac",
        help="Comma-separated domain list",
    )
    parser.add_argument(
        "--pysr-iterations", type=int, default=40,
        help="PySR iterations (default: 40)",
    )
    args = parser.parse_args()

    domains = args.domains.split(",")

    runners = {
        "climate_epidemic": (
            "simulating_anything.rediscovery.climate_epidemic",
            "run_climate_epidemic_rediscovery",
        ),
        "neural_cardiac": (
            "simulating_anything.rediscovery.neural_cardiac",
            "run_neural_cardiac_rediscovery",
        ),
    }

    for domain in domains:
        domain = domain.strip()
        if domain not in runners:
            logger.warning(f"Unknown domain: {domain}")
            continue

        module_name, func_name = runners[domain]
        logger.info("=" * 60)
        logger.info(f"Running rediscovery: {domain}")
        logger.info("=" * 60)

        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_name)
            run_fn = getattr(mod, func_name)
            results = run_fn(n_iterations=args.pysr_iterations)

            out_dir = Path(f"output/rediscovery/{domain}")
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

            elapsed = time.time() - t0
            logger.info(f"  {domain} complete in {elapsed:.1f}s")

            if "sindy_ode" in results:
                sindy = results["sindy_ode"]
                logger.info(
                    f"  SINDy: {sindy['n_discoveries']} equations, "
                    f"mean R2={sindy['mean_r_squared']:.6f}"
                )
                for d in sindy["discoveries"]:
                    logger.info(f"    {d['expression']}  (R2={d['r_squared']:.6f})")

        except Exception as e:
            logger.error(f"  {domain} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
