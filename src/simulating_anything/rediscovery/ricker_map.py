"""Ricker map rediscovery.

Targets:
- Period-doubling bifurcation diagram
- Chaos onset at r ~ 2.0 (fixed point x*=K loses stability when |1-r| > 1)
- Lyapunov exponent as function of r
- Fixed point verification: x* = K
- Overcompensation dynamics
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.ricker_map import RickerMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    r: float = 2.0,
    K: float = 1.0,
    x_0: float = 0.5,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.RICKER_MAP,
        dt=1.0,
        n_steps=1000,
        parameters={"r": r, "K": K, "x_0": x_0},
    )


def generate_bifurcation_data(
    n_r: int = 500,
    r_min: float = 0.5,
    r_max: float = 3.5,
    K: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram and period detection data."""
    r_values = np.linspace(r_min, r_max, n_r)

    # Bifurcation diagram points
    sim = RickerMapSimulation(_make_config(r=2.0, K=K))
    bif_data = sim.bifurcation_diagram(r_values, n_transient=500, n_plot=50)

    # Period detection
    periods = []
    for r in r_values:
        config_r = _make_config(r=r, K=K)
        sim_r = RickerMapSimulation(config_r)
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
    r_min: float = 0.5,
    r_max: float = 3.5,
    K: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs r data."""
    r_values = np.linspace(r_min, r_max, n_r)
    lyapunovs = []

    for i, r in enumerate(r_values):
        config = _make_config(r=r, K=K)
        sim = RickerMapSimulation(config)
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        lyapunovs.append(lam)

        if (i + 1) % 25 == 0:
            logger.info(f"  r={r:.4f}: Lyapunov={lam:.4f}")

    return {
        "r": r_values,
        "lyapunov": np.array(lyapunovs),
    }


def estimate_feigenbaum(
    periods: np.ndarray,
    r_values: np.ndarray,
) -> dict:
    """Estimate Feigenbaum constant from period-doubling bifurcation points."""
    bif_points = []
    target_periods = [2, 4, 8, 16]

    for p in target_periods:
        idx = np.where(periods >= p)[0]
        if len(idx) > 0:
            bif_points.append(float(r_values[idx[0]]))

    result: dict = {"bifurcation_points": bif_points}

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
            logger.info(f"  Feigenbaum estimates: {deltas}")

    return result


def run_ricker_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/ricker_map",
    n_iterations: int = 40,
) -> dict:
    """Run Ricker map rediscovery.

    Discovers:
    - Bifurcation diagram and period-doubling cascade
    - Chaos onset at r ~ 2.0
    - Lyapunov exponent spectrum
    - Fixed point structure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "ricker_map",
        "targets": {
            "chaos_onset": "r_c ~ 2.0 (|1 - r| > 1)",
            "fixed_point": "x* = K",
            "lyapunov_at_fixed_point": "ln|1 - r|",
            "feigenbaum": "delta ~ 4.669",
        },
    }

    # Part 1: Fixed point analysis
    logger.info("Part 1: Fixed point analysis...")
    sim_fp = RickerMapSimulation(_make_config(r=1.5, K=1.0))
    fps = sim_fp.find_fixed_points()
    results["fixed_points"] = fps
    logger.info(f"  Fixed points: {fps}")

    # Verify convergence to K for r < 2
    sim_conv = RickerMapSimulation(_make_config(r=1.5, K=1.0, x_0=0.3))
    sim_conv.reset()
    for _ in range(1000):
        sim_conv.step()
    converged_x = float(sim_conv.observe()[0])
    results["convergence_test"] = {
        "r": 1.5,
        "K": 1.0,
        "final_x": converged_x,
        "error": abs(converged_x - 1.0),
    }
    logger.info(f"  r=1.5: converged to x={converged_x:.6f} (K=1.0)")

    # Part 2: Bifurcation diagram + period detection
    logger.info("Part 2: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(n_r=500)
    results["bifurcation"] = {
        "n_r": 500,
        "r_range": [0.5, 3.5],
    }
    np.savez(
        output_path / "bifurcation_data.npz",
        r_values=bif_data["r_values"],
        periods=bif_data["periods"],
        bif_r=bif_data["bif_r"],
        bif_x=bif_data["bif_x"],
    )

    # Feigenbaum constant from period-doubling
    feig = estimate_feigenbaum(bif_data["periods"], bif_data["r_values"])
    results["feigenbaum"] = feig

    # Part 3: Lyapunov exponent spectrum
    logger.info("Part 3: Lyapunov exponent vs r...")
    lyap_data = generate_lyapunov_data(n_r=100)

    # Chaos onset (first r where lambda > 0)
    chaotic = lyap_data["lyapunov"] > 0
    if np.any(chaotic):
        r_chaos = lyap_data["r"][np.argmax(chaotic)]
        results["chaos_onset"] = {
            "r_estimate": float(r_chaos),
            "r_theory": 2.0,
        }
        logger.info(f"  Chaos onset: r ~ {r_chaos:.4f} (theory: 2.0)")

    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "r_at_max": float(lyap_data["r"][np.argmax(lyap_data["lyapunov"])]),
    }

    # Check theoretical Lyapunov at fixed point x*=K: ln|1-r|
    for r_test in [1.5, 2.5, 3.0]:
        sim_test = RickerMapSimulation(_make_config(r=r_test, K=1.0))
        lam_num = sim_test.compute_lyapunov(n_iterations=10000, n_transient=500)
        lam_theory = np.log(abs(1.0 - r_test))
        logger.info(
            f"  r={r_test}: Lyapunov={lam_num:.4f} "
            f"(theory ln|1-r|={lam_theory:.4f})"
        )

    # PySR: fit Lyapunov(r) in chaotic region
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        chaotic_region = lyap_data["lyapunov"] > 0
        if np.sum(chaotic_region) > 5:
            X = lyap_data["r"][chaotic_region].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_region]

            logger.info("  Running PySR: lambda = f(r) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["r_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["log", "sqrt", "exp"],
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

    # Save results
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
