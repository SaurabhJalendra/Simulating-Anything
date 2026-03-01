"""Bouncing ball rediscovery.

Targets:
- Period-doubling cascade as amplitude increases
- Bifurcation diagram (amplitude vs steady-state velocity)
- Lyapunov exponent as function of amplitude
- Transition from periodic to chaotic bouncing
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.bouncing_ball import BouncingBallSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_bifurcation_data(
    n_A: int = 30,
    A_min: float = 0.01,
    A_max: float = 0.5,
    e: float = 0.5,
    omega: float = 2 * np.pi,
    g: float = 9.81,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram and period detection data.

    Args:
        n_A: number of amplitude values to sweep.
        A_min: minimum amplitude.
        A_max: maximum amplitude.
        e: coefficient of restitution.
        omega: table angular frequency.
        g: gravitational acceleration.

    Returns:
        Dict with A_values, periods, bifurcation A and velocity arrays.
    """
    A_values = np.linspace(A_min, A_max, n_A)

    # Bifurcation diagram
    config = SimulationConfig(
        domain=Domain.BOUNCING_BALL,
        dt=1.0,  # Discrete map
        n_steps=500,
        parameters={"e": e, "A": 0.1, "omega": omega, "g": g},
    )
    sim = BouncingBallSimulation(config)
    bif_data = sim.bifurcation_diagram(A_values, n_skip=200, n_record=100)

    # Period detection at each amplitude
    periods = []
    for A_val in A_values:
        config_A = SimulationConfig(
            domain=Domain.BOUNCING_BALL,
            dt=1.0,
            n_steps=500,
            parameters={"e": e, "A": A_val, "omega": omega, "g": g},
        )
        sim_A = BouncingBallSimulation(config_A)
        sim_A.reset()
        p = sim_A.period_detection(n_steps=500, max_period=64)
        periods.append(p)

    return {
        "A_values": A_values,
        "periods": np.array(periods),
        "bif_A": bif_data["A"],
        "bif_v": bif_data["velocity"],
    }


def generate_lyapunov_data(
    n_A: int = 30,
    A_min: float = 0.01,
    A_max: float = 0.5,
    e: float = 0.5,
    omega: float = 2 * np.pi,
    g: float = 9.81,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs amplitude data.

    Args:
        n_A: number of amplitude values.
        A_min: minimum amplitude.
        A_max: maximum amplitude.
        e: coefficient of restitution.
        omega: table angular frequency.
        g: gravitational acceleration.

    Returns:
        Dict with 'A' and 'lyapunov' arrays.
    """
    A_values = np.linspace(A_min, A_max, n_A)
    lyapunovs = []

    for i, A_val in enumerate(A_values):
        config = SimulationConfig(
            domain=Domain.BOUNCING_BALL,
            dt=1.0,
            n_steps=500,
            parameters={"e": e, "A": A_val, "omega": omega, "g": g},
        )
        sim = BouncingBallSimulation(config)
        sim.reset()
        lam = sim.compute_lyapunov(n_steps=5000, n_transient=500)
        lyapunovs.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  A={A_val:.4f}: Lyapunov={lam:.4f}")

    return {
        "A": A_values,
        "lyapunov": np.array(lyapunovs),
    }


def identify_period_doubling(
    periods: np.ndarray,
    A_values: np.ndarray,
) -> dict:
    """Identify period-doubling cascade from period data.

    Args:
        periods: detected periods at each amplitude.
        A_values: amplitude values.

    Returns:
        Dict with period-doubling bifurcation points.
    """
    bif_points = []
    target_periods = [2, 4, 8, 16]

    for p in target_periods:
        idx = np.where(periods >= p)[0]
        if len(idx) > 0:
            bif_points.append({
                "period": int(p),
                "A_threshold": float(A_values[idx[0]]),
            })

    result: dict = {"bifurcation_points": bif_points}

    # Estimate Feigenbaum-like ratio if enough points
    if len(bif_points) >= 3:
        A_vals = [bp["A_threshold"] for bp in bif_points]
        deltas = []
        for i in range(len(A_vals) - 2):
            d1 = A_vals[i + 1] - A_vals[i]
            d2 = A_vals[i + 2] - A_vals[i + 1]
            if d2 > 1e-10:
                deltas.append(d1 / d2)
        if deltas:
            result["feigenbaum_estimates"] = deltas
            logger.info(f"  Feigenbaum-like ratios: {deltas}")

    return result


def run_bouncing_ball_rediscovery(
    output_dir: str | Path = "output/rediscovery/bouncing_ball",
    n_iterations: int = 40,
) -> dict:
    """Run bouncing ball rediscovery.

    Args:
        output_dir: directory to save results.
        n_iterations: PySR iterations (for future symbolic regression).

    Returns:
        Results dict with bifurcation, period-doubling, and Lyapunov data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "bouncing_ball",
        "targets": {
            "period_doubling": "Period-doubling cascade as A increases",
            "chaos": "Chaotic bouncing at high amplitude",
            "lyapunov": "Positive Lyapunov for chaotic regime",
        },
    }

    # Part 1: Bifurcation diagram + period detection
    logger.info("Part 1: Bifurcation diagram + period detection...")
    bif_data = generate_bifurcation_data(n_A=30)
    results["bifurcation"] = {
        "n_A": 30,
        "A_range": [0.01, 0.5],
        "n_bif_points": len(bif_data["bif_A"]),
    }

    # Period-doubling analysis
    pd_result = identify_period_doubling(
        bif_data["periods"], bif_data["A_values"]
    )
    results["period_doubling"] = pd_result
    logger.info(
        f"  Found {len(pd_result['bifurcation_points'])} "
        f"period-doubling bifurcations"
    )

    # Part 2: Lyapunov exponent vs amplitude
    logger.info("Part 2: Lyapunov exponent vs amplitude...")
    lyap_data = generate_lyapunov_data(n_A=30)

    # Find chaos onset (first A where Lyapunov > 0)
    chaotic = lyap_data["lyapunov"] > 0
    if np.any(chaotic):
        A_chaos = lyap_data["A"][np.argmax(chaotic)]
        results["chaos_onset"] = {
            "A_estimate": float(A_chaos),
        }
        logger.info(f"  Chaos onset: A ~ {A_chaos:.4f}")

    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "A_at_max": float(lyap_data["A"][np.argmax(lyap_data["lyapunov"])]),
        "n_chaotic": int(np.sum(chaotic)),
        "n_total": len(lyap_data["lyapunov"]),
    }

    # Part 3: PySR on Lyapunov(A) in chaotic region (optional)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        chaotic_region = lyap_data["lyapunov"] > 0
        if np.sum(chaotic_region) > 5:
            X = lyap_data["A"][chaotic_region].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_region]

            logger.info("  Running PySR: lambda = f(A) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["A_"],
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
                results["lyapunov_pysr"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as exc:
        logger.warning(f"PySR failed: {exc}")
        results["lyapunov_pysr"] = {"error": str(exc)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "lyapunov_data.npz",
        A=lyap_data["A"],
        lyapunov=lyap_data["lyapunov"],
    )
    np.savez(
        output_path / "bifurcation_data.npz",
        A_values=bif_data["A_values"],
        periods=bif_data["periods"],
        bif_A=bif_data["bif_A"],
        bif_v=bif_data["bif_v"],
    )

    return results
