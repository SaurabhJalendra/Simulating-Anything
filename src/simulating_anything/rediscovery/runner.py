"""Unified rediscovery runner -- runs all five domain rediscoveries."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_domain(name: str, run_fn, output_path: Path, **kwargs) -> tuple[dict, float]:
    """Run a single domain rediscovery with timing and error handling."""
    t0 = time.time()
    try:
        result = run_fn(output_dir=output_path / name, **kwargs)
        elapsed = time.time() - t0
        logger.info(f"{name} completed in {elapsed:.1f}s")
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"{name} rediscovery failed: {e}")
        return {"error": str(e)}, elapsed


def run_all_rediscoveries(
    output_dir: str | Path = "output/rediscovery",
    pysr_iterations: int = 40,
    domains: list[str] | None = None,
) -> dict:
    """Run domain rediscoveries and produce a unified report.

    Args:
        output_dir: Base output directory.
        pysr_iterations: Number of PySR iterations per run.
        domains: Which domains to run (default: all six).

    Returns:
        Combined results dict with all discoveries.
    """
    from simulating_anything.rediscovery.double_pendulum import (
        run_double_pendulum_rediscovery,
    )
    from simulating_anything.rediscovery.gray_scott import run_gray_scott_analysis
    from simulating_anything.rediscovery.harmonic_oscillator import (
        run_harmonic_oscillator_rediscovery,
    )
    from simulating_anything.rediscovery.lotka_volterra import (
        run_lotka_volterra_rediscovery,
    )
    from simulating_anything.rediscovery.projectile import run_projectile_rediscovery
    from simulating_anything.rediscovery.sir_epidemic import run_sir_rediscovery

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Registry of all domains
    domain_registry = {
        "projectile": {
            "label": "Projectile Range Equation",
            "fn": run_projectile_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "lotka_volterra": {
            "label": "Lotka-Volterra Equilibrium & ODE",
            "fn": run_lotka_volterra_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "gray_scott": {
            "label": "Gray-Scott Pattern Analysis",
            "fn": run_gray_scott_analysis,
            "kwargs": {},
        },
        "sir_epidemic": {
            "label": "SIR Epidemic R0 & ODE",
            "fn": run_sir_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "double_pendulum": {
            "label": "Double Pendulum Period & Energy",
            "fn": run_double_pendulum_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
        "harmonic_oscillator": {
            "label": "Harmonic Oscillator Frequency & Damping",
            "fn": run_harmonic_oscillator_rediscovery,
            "kwargs": {"n_iterations": pysr_iterations},
        },
    }

    if domains is None:
        domains = list(domain_registry.keys())

    all_results = {}
    timings = {}

    for i, name in enumerate(domains, 1):
        if name not in domain_registry:
            logger.warning(f"Unknown domain '{name}' -- skipping")
            continue

        entry = domain_registry[name]
        logger.info("=" * 60)
        logger.info(f"REDISCOVERY {i}/{len(domains)}: {entry['label']}")
        logger.info("=" * 60)

        result, elapsed = _run_domain(
            name, entry["fn"], output_path, **entry["kwargs"]
        )
        all_results[name] = result
        timings[name] = elapsed

    # Build summary
    total_time = sum(timings.values())

    # Extract key metrics from each domain
    scorecard = {}
    for name, result in all_results.items():
        if "error" in result:
            scorecard[name] = {"status": "failed", "error": result["error"]}
            continue

        entry = {"status": "success"}

        # Try to extract best R² from various result keys
        for key in result:
            if isinstance(result[key], dict) and "best_r2" in result[key]:
                entry[f"{key}_r2"] = result[key]["best_r2"]
            if isinstance(result[key], dict) and "best" in result[key]:
                entry[f"{key}_expr"] = result[key]["best"]

        # Domain-specific metrics
        if "energy_conservation" in result:
            entry["energy_drift"] = result["energy_conservation"].get(
                "mean_final_drift", None
            )
        if "period_accuracy" in result:
            entry["period_error"] = result["period_accuracy"].get(
                "mean_relative_error", None
            )

        scorecard[name] = entry

    summary = {
        "n_domains": len(domains),
        "n_succeeded": sum(
            1 for s in scorecard.values() if s.get("status") == "success"
        ),
        "total_time_seconds": total_time,
        "timings": timings,
        "scorecard": scorecard,
        "results": all_results,
    }

    # Save combined results
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print scorecard
    logger.info("=" * 60)
    logger.info("REDISCOVERY SCORECARD")
    logger.info("=" * 60)
    for name, score in scorecard.items():
        status = score.get("status", "unknown")
        r2_keys = [k for k in score if k.endswith("_r2")]
        if r2_keys:
            best_r2 = max(score[k] for k in r2_keys)
            logger.info(f"  {name:20s}  {status:8s}  best R²={best_r2:.6f}")
        else:
            logger.info(f"  {name:20s}  {status:8s}")
    logger.info(f"\nTotal time: {total_time:.1f}s")
    logger.info(f"Summary saved to {summary_file}")
    logger.info("=" * 60)

    return summary
