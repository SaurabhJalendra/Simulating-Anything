"""Unified rediscovery runner -- runs all three domain rediscoveries."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_all_rediscoveries(
    output_dir: str | Path = "output/rediscovery",
    pysr_iterations: int = 40,
) -> dict:
    """Run all three domain rediscoveries and produce a unified report.

    Args:
        output_dir: Base output directory.
        pysr_iterations: Number of PySR iterations per run.

    Returns:
        Combined results dict with all discoveries.
    """
    from simulating_anything.rediscovery.gray_scott import run_gray_scott_analysis
    from simulating_anything.rediscovery.lotka_volterra import (
        run_lotka_volterra_rediscovery,
    )
    from simulating_anything.rediscovery.projectile import run_projectile_rediscovery

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    timings = {}

    # 1. Projectile
    logger.info("=" * 60)
    logger.info("REDISCOVERY 1/3: Projectile Range Equation")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        all_results["projectile"] = run_projectile_rediscovery(
            output_dir=output_path / "projectile",
            n_iterations=pysr_iterations,
        )
        timings["projectile"] = time.time() - t0
        logger.info(f"Projectile completed in {timings['projectile']:.1f}s")
    except Exception as e:
        logger.error(f"Projectile rediscovery failed: {e}")
        all_results["projectile"] = {"error": str(e)}
        timings["projectile"] = time.time() - t0

    # 2. Lotka-Volterra
    logger.info("=" * 60)
    logger.info("REDISCOVERY 2/3: Lotka-Volterra Equilibrium & ODE")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        all_results["lotka_volterra"] = run_lotka_volterra_rediscovery(
            output_dir=output_path / "lotka_volterra",
            n_iterations=pysr_iterations,
        )
        timings["lotka_volterra"] = time.time() - t0
        logger.info(f"Lotka-Volterra completed in {timings['lotka_volterra']:.1f}s")
    except Exception as e:
        logger.error(f"Lotka-Volterra rediscovery failed: {e}")
        all_results["lotka_volterra"] = {"error": str(e)}
        timings["lotka_volterra"] = time.time() - t0

    # 3. Gray-Scott
    logger.info("=" * 60)
    logger.info("REDISCOVERY 3/3: Gray-Scott Pattern Analysis")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        all_results["gray_scott"] = run_gray_scott_analysis(
            output_dir=output_path / "gray_scott",
        )
        timings["gray_scott"] = time.time() - t0
        logger.info(f"Gray-Scott completed in {timings['gray_scott']:.1f}s")
    except Exception as e:
        logger.error(f"Gray-Scott analysis failed: {e}")
        all_results["gray_scott"] = {"error": str(e)}
        timings["gray_scott"] = time.time() - t0

    # Build summary
    total_time = sum(timings.values())
    summary = {
        "total_time_seconds": total_time,
        "timings": timings,
        "results": all_results,
    }

    # Save combined results
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"All rediscoveries completed in {total_time:.1f}s")
    logger.info(f"Summary saved to {summary_file}")
    logger.info("=" * 60)

    return summary
