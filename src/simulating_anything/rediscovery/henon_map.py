"""Henon map rediscovery.

Targets:
- Period-doubling bifurcation diagram
- Lyapunov exponent as function of a
- PySR fit of Lyapunov(a) in chaotic region
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.henon_map import HenonMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(a: float = 1.4, b: float = 0.3) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.HENON_MAP,
        dt=1.0,
        n_steps=1000,
        parameters={"a": a, "b": b, "x_0": 0.0, "y_0": 0.0},
    )


def generate_bifurcation_data(
    n_a: int = 500,
    a_min: float = 0.0,
    a_max: float = 1.4,
    b: float = 0.3,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram and period detection data."""
    a_values = np.linspace(a_min, a_max, n_a)

    # Bifurcation diagram points
    sim = HenonMapSimulation(_make_config(a=1.4, b=b))
    bif_data = sim.bifurcation_diagram(a_values, n_transient=500, n_plot=50)

    # Period detection
    periods = []
    for a in a_values:
        config_a = _make_config(a=a, b=b)
        sim_a = HenonMapSimulation(config_a)
        sim_a.reset()
        p = sim_a.detect_period(max_period=64)
        periods.append(p)

    return {
        "a_values": a_values,
        "periods": np.array(periods),
        "bif_a": bif_data["a"],
        "bif_x": bif_data["x"],
    }


def generate_lyapunov_data(
    n_a: int = 100,
    a_min: float = 0.0,
    a_max: float = 1.4,
    b: float = 0.3,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs a data."""
    a_values = np.linspace(a_min, a_max, n_a)
    lyapunovs = []

    for i, a in enumerate(a_values):
        config = _make_config(a=a, b=b)
        sim = HenonMapSimulation(config)
        sim.reset()
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        lyapunovs.append(lam)

        if (i + 1) % 25 == 0:
            logger.info(f"  a={a:.4f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov": np.array(lyapunovs),
    }


def run_henon_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/henon_map",
    n_iterations: int = 40,
) -> dict:
    """Run Henon map rediscovery."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "henon_map",
        "targets": {
            "lyapunov_classic": "~0.42 at a=1.4, b=0.3",
            "jacobian_det": "|b| = area contraction factor",
            "fixed_points": "x* = (-(1-b) +/- sqrt((1-b)^2 + 4a)) / (2a)",
        },
    }

    # Part 1: Bifurcation diagram
    logger.info("Part 1: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(n_a=200)
    results["bifurcation"] = {
        "n_a": 200,
        "a_range": [0.0, 1.4],
    }
    np.savez(
        output_path / "bifurcation_data.npz",
        a_values=bif_data["a_values"],
        periods=bif_data["periods"],
        bif_a=bif_data["bif_a"],
        bif_x=bif_data["bif_x"],
    )

    # Part 2: Lyapunov exponent
    logger.info("Part 2: Lyapunov exponent vs a...")
    lyap_data = generate_lyapunov_data(n_a=100)

    # Classic Lyapunov at a=1.4
    classic_config = _make_config(a=1.4, b=0.3)
    classic_sim = HenonMapSimulation(classic_config)
    classic_sim.reset()
    classic_lyap = classic_sim.compute_lyapunov(n_iterations=50000, n_transient=1000)
    results["classic_lyapunov"] = {
        "a": 1.4,
        "b": 0.3,
        "lyapunov": float(classic_lyap),
        "reference": 0.42,
    }
    logger.info(f"  Classic Lyapunov (a=1.4, b=0.3): {classic_lyap:.4f} (ref: ~0.42)")

    # Chaos onset
    chaotic = lyap_data["lyapunov"] > 0
    if np.any(chaotic):
        a_chaos = lyap_data["a"][np.argmax(chaotic)]
        results["chaos_onset"] = {
            "a_estimate": float(a_chaos),
        }
        logger.info(f"  Chaos onset: a ~ {a_chaos:.4f}")

    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "a_at_max": float(lyap_data["a"][np.argmax(lyap_data["lyapunov"])]),
    }

    # PySR: find Lyapunov(a) in chaotic region
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        chaotic_region = lyap_data["lyapunov"] > 0
        if np.sum(chaotic_region) > 5:
            X = lyap_data["a"][chaotic_region].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_region]

            logger.info("  Running PySR: lambda = f(a) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["a_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["log", "sqrt"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["lyapunov_pysr"] = {
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
                results["lyapunov_pysr"]["best"] = best.expression
                results["lyapunov_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["lyapunov_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "lyapunov_data.npz",
        a=lyap_data["a"],
        lyapunov=lyap_data["lyapunov"],
    )

    return results
