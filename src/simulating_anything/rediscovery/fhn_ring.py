"""FHN Ring Network rediscovery.

Targets:
- Synchronization transition: order parameter r vs coupling D
- Critical coupling D_c for sync onset
- Traveling wave speed vs coupling
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.fhn_ring import FHNRingSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_sync_transition_data(
    n_D: int = 30,
    N: int = 20,
    n_transient: int = 5000,
    n_measure: int = 2000,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Sweep coupling D and measure steady-state synchronization.

    For each coupling strength, runs the ring to steady state and
    measures the mean-field order parameter.
    """
    D_values = np.linspace(0.0, 2.0, n_D)
    r_values = []

    for i, D in enumerate(D_values):
        config = SimulationConfig(
            domain=Domain.FHN_RING,
            dt=dt,
            n_steps=n_transient + n_measure,
            parameters={
                "a": 0.7, "b": 0.8, "eps": 0.08,
                "I": 0.5, "D": D, "N": float(N),
            },
        )
        sim = FHNRingSimulation(config)
        sim.reset(seed=42)

        # Skip transient
        for _ in range(n_transient):
            sim.step()

        # Measure order parameter over measurement window
        r_samples = []
        for _ in range(n_measure):
            sim.step()
            r_samples.append(sim.compute_synchronization())

        r_mean = np.mean(r_samples)
        r_values.append(r_mean)

        if (i + 1) % 10 == 0:
            logger.info(f"  D={D:.3f}: r={r_mean:.4f}")

    return {
        "D": D_values,
        "r": np.array(r_values),
        "N": N,
    }


def generate_wave_speed_data(
    n_D: int = 15,
    N: int = 20,
    n_transient: int = 3000,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Measure traveling wave speed as a function of coupling D.

    Only measures for nonzero coupling where waves can propagate.
    """
    D_values = np.linspace(0.1, 2.0, n_D)
    speeds = []

    for i, D in enumerate(D_values):
        config = SimulationConfig(
            domain=Domain.FHN_RING,
            dt=dt,
            n_steps=n_transient + 2000,
            parameters={
                "a": 0.7, "b": 0.8, "eps": 0.08,
                "I": 0.5, "D": D, "N": float(N),
            },
        )
        sim = FHNRingSimulation(config)
        sim.reset(seed=42)

        # Skip transient
        for _ in range(n_transient):
            sim.step()

        speed = sim.detect_traveling_wave()
        speeds.append(speed)

        if (i + 1) % 5 == 0:
            logger.info(f"  D={D:.3f}: wave speed={speed:.4f}")

    return {
        "D": D_values,
        "speed": np.array(speeds),
        "N": N,
    }


def run_fhn_ring_rediscovery(
    output_dir: str | Path = "output/rediscovery/fhn_ring",
    n_iterations: int = 40,
) -> dict:
    """Run FHN ring network rediscovery pipeline.

    1. Sweep coupling D, measure synchronization order parameter
    2. Estimate critical coupling D_c for synchronization onset
    3. Measure traveling wave speed vs coupling
    4. Run PySR to find r(D) relationship
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "fhn_ring",
        "targets": {
            "sync_transition": "order parameter r vs coupling D",
            "critical_coupling": "D_c for synchronization onset",
            "wave_speed": "traveling wave speed vs D",
        },
    }

    # --- Part 1: Synchronization transition ---
    logger.info("Part 1: Synchronization transition (r vs D)...")
    data = generate_sync_transition_data(n_D=30, N=20, dt=0.05)

    # Estimate D_c: first D where r > threshold
    threshold = 0.7
    above = data["r"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        D_c_est = data["D"][max(0, idx - 1)]
        results["D_c_estimate"] = float(D_c_est)
        logger.info(f"  D_c estimate: {D_c_est:.4f}")
    else:
        results["D_c_estimate"] = None
        logger.warning("  Could not detect synchronization transition")

    results["sync_transition"] = {
        "n_D": len(data["D"]),
        "D_range": [float(data["D"].min()), float(data["D"].max())],
        "max_r": float(np.max(data["r"])),
        "min_r": float(np.min(data["r"])),
    }

    # PySR: find r(D)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use data with nonzero coupling
        mask = data["D"] > 0.01
        if np.sum(mask) > 5:
            X = data["D"][mask].reshape(-1, 1)
            y = data["r"][mask]

            logger.info("  Running PySR: r = f(D)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["D_c"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square", "exp"],
                max_complexity=12,
                populations=15,
                population_size=30,
            )
            results["sync_pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {
                        "expression": d.expression,
                        "r_squared": d.evidence.fit_r_squared,
                    }
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["sync_pysr"]["best"] = best.expression
                results["sync_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["sync_pysr"] = {"error": str(e)}

    # --- Part 2: Wave speed ---
    logger.info("Part 2: Traveling wave speed vs D...")
    wave_data = generate_wave_speed_data(n_D=15, N=20, dt=0.05)

    results["wave_speed"] = {
        "n_D": len(wave_data["D"]),
        "D_range": [float(wave_data["D"].min()), float(wave_data["D"].max())],
        "max_speed": float(np.max(wave_data["speed"])),
        "mean_speed": float(np.mean(wave_data["speed"])),
    }

    # PySR on wave speed
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        nonzero = wave_data["speed"] > 0.01
        if np.sum(nonzero) > 3:
            X = wave_data["D"][nonzero].reshape(-1, 1)
            y = wave_data["speed"][nonzero]

            logger.info("  Running PySR: speed = f(D)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["D_c"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt"],
                max_complexity=8,
                populations=15,
                population_size=30,
            )
            results["wave_pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {
                        "expression": d.expression,
                        "r_squared": d.evidence.fit_r_squared,
                    }
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["wave_pysr"]["best"] = best.expression
                results["wave_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR wave speed failed: {e}")
        results["wave_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "sync_transition.npz",
        D=data["D"],
        r=data["r"],
    )
    np.savez(
        output_path / "wave_speed.npz",
        D=wave_data["D"],
        speed=wave_data["speed"],
    )

    return results
