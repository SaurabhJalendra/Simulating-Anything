"""Logistic map rediscovery.

Targets:
- Period-doubling bifurcation points
- Feigenbaum constant delta ~ 4.669
- Lyapunov exponent as function of r
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.logistic_map import LogisticMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_bifurcation_data(
    n_r: int = 500,
    r_min: float = 2.5,
    r_max: float = 4.0,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram and period detection data."""
    r_values = np.linspace(r_min, r_max, n_r)

    # Bifurcation diagram points
    config = SimulationConfig(
        domain=Domain.LOGISTIC_MAP,
        dt=1.0,  # Discrete map
        n_steps=1000,
        parameters={"r": 3.5, "x_0": 0.5},
    )
    sim = LogisticMapSimulation(config)
    bif_data = sim.bifurcation_diagram(r_values, n_transient=500, n_plot=50)

    # Period detection
    periods = []
    for r in r_values:
        config_r = SimulationConfig(
            domain=Domain.LOGISTIC_MAP,
            dt=1.0,
            n_steps=100,
            parameters={"r": r, "x_0": 0.5},
        )
        sim_r = LogisticMapSimulation(config_r)
        sim_r.reset()
        p = sim_r.detect_period(max_period=64)
        periods.append(p)

    return {
        "r_values": r_values,
        "periods": np.array(periods),
        "bif_r": bif_data["r"],
        "bif_x": bif_data["x"],
    }


def generate_lyapunov_data(
    n_r: int = 100,
    r_min: float = 2.5,
    r_max: float = 4.0,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs r data."""
    r_values = np.linspace(r_min, r_max, n_r)
    lyapunovs = []

    for i, r in enumerate(r_values):
        config = SimulationConfig(
            domain=Domain.LOGISTIC_MAP,
            dt=1.0,
            n_steps=100,
            parameters={"r": r, "x_0": 0.5},
        )
        sim = LogisticMapSimulation(config)
        lam = sim.lyapunov_exponent(n_iterations=5000, n_transient=500)
        lyapunovs.append(lam)

        if (i + 1) % 25 == 0:
            logger.info(f"  r={r:.4f}: Lyapunov={lam:.4f}")

    return {
        "r": r_values,
        "lyapunov": np.array(lyapunovs),
    }


def estimate_feigenbaum(periods: np.ndarray, r_values: np.ndarray) -> dict:
    """Estimate Feigenbaum constant from period-doubling bifurcation points."""
    # Find bifurcation points where period doubles
    bif_points = []
    target_periods = [2, 4, 8, 16]

    for p in target_periods:
        # First r where period >= p
        idx = np.where(periods >= p)[0]
        if len(idx) > 0:
            bif_points.append(float(r_values[idx[0]]))

    result = {"bifurcation_points": bif_points}

    if len(bif_points) >= 3:
        deltas = []
        for i in range(len(bif_points) - 2):
            d1 = bif_points[i + 1] - bif_points[i]
            d2 = bif_points[i + 2] - bif_points[i + 1]
            if d2 > 1e-10:
                deltas.append(d1 / d2)

        if deltas:
            result["feigenbaum_estimates"] = deltas
            result["best_estimate"] = deltas[-1] if deltas else None
            # True Feigenbaum: 4.66920...
            logger.info(f"  Feigenbaum estimates: {deltas}")

    return result


def run_logistic_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/logistic_map",
    n_iterations: int = 40,
) -> dict:
    """Run logistic map rediscovery."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "logistic_map",
        "targets": {
            "feigenbaum": "delta ~ 4.669",
            "chaos_onset": "r_c ~ 3.5699",
            "lyapunov": "lambda(r) > 0 for chaos",
        },
    }

    # Part 1: Bifurcation diagram + period detection
    logger.info("Part 1: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(n_r=500)
    results["bifurcation"] = {
        "n_r": 500,
        "r_range": [2.5, 4.0],
    }

    # Feigenbaum constant
    feig = estimate_feigenbaum(bif_data["periods"], bif_data["r_values"])
    results["feigenbaum"] = feig

    # Part 2: Lyapunov exponent
    logger.info("Part 2: Lyapunov exponent vs r...")
    lyap_data = generate_lyapunov_data(n_r=100)

    # Find chaos onset (first r where lambda > 0)
    chaotic = lyap_data["lyapunov"] > 0
    if np.any(chaotic):
        r_chaos = lyap_data["r"][np.argmax(chaotic)]
        results["chaos_onset"] = {
            "r_estimate": float(r_chaos),
            "r_theory": 3.5699,
        }
        logger.info(f"  Chaos onset: r ~ {r_chaos:.4f} (theory: 3.5699)")

    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "r_at_max": float(lyap_data["r"][np.argmax(lyap_data["lyapunov"])]),
    }

    # PySR: find Lyapunov(r) in chaotic region
    try:
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        chaotic_region = lyap_data["lyapunov"] > 0
        if np.sum(chaotic_region) > 5:
            X = lyap_data["r"][chaotic_region].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_region]

            logger.info("  Running PySR: lambda = f(r) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["r"],
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
                    {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["lyapunov_pysr"]["best"] = best.expression
                results["lyapunov_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})"
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
        r=lyap_data["r"],
        lyapunov=lyap_data["lyapunov"],
    )

    return results
