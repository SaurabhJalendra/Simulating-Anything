"""Restricted three-body problem rediscovery.

Targets:
- Jacobi constant conservation: C_J = -(vx^2+vy^2) + 2*Omega = const
- Lagrange point L4/L5 locations: (0.5-mu, +/-sqrt(3)/2)
- Lyapunov exponent dependence on mu (chaos characterization)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.three_body import ThreeBodySimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_jacobi_conservation_data(
    mu: float = 0.01,
    n_steps: int = 5000,
    dt: float = 0.005,
) -> dict[str, np.ndarray | float]:
    """Track Jacobi constant over a trajectory to verify conservation.

    Args:
        mu: Mass ratio m2/(m1+m2).
        n_steps: Number of integration steps.
        dt: Timestep.

    Returns:
        Dict with times, jacobi values, and drift statistics.
    """
    config = SimulationConfig(
        domain=Domain.THREE_BODY,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "mu": mu,
            "x0": 0.5,
            "y0": 0.5,
            "vx0": 0.1,
            "vy0": -0.1,
        },
    )
    sim = ThreeBodySimulation(config)
    sim.reset()

    CJ0 = sim.jacobi_constant()
    times = [0.0]
    jacobi_vals = [CJ0]

    for step_i in range(n_steps):
        sim.step()
        times.append((step_i + 1) * dt)
        jacobi_vals.append(sim.jacobi_constant())

    jacobi_arr = np.array(jacobi_vals)
    times_arr = np.array(times)
    drift = np.abs(jacobi_arr - CJ0) / (np.abs(CJ0) + 1e-15)

    return {
        "times": times_arr,
        "jacobi_values": jacobi_arr,
        "CJ0": float(CJ0),
        "max_drift": float(np.max(drift)),
        "mean_drift": float(np.mean(drift)),
        "final_drift": float(drift[-1]),
    }


def generate_lagrange_point_data(
    n_samples: int = 20,
) -> dict[str, np.ndarray]:
    """Sweep mass ratio mu and compute L4/L5 locations analytically.

    The L4/L5 equilateral points are at (0.5-mu, +/-sqrt(3)/2) for any mu.
    This generates data for PySR to rediscover the relationship.

    Args:
        n_samples: Number of mu values to sweep.

    Returns:
        Dict with mu values and L4/L5 coordinates.
    """
    mu_values = np.linspace(0.001, 0.1, n_samples)
    l4_x = []
    l4_y = []
    l5_x = []
    l5_y = []

    for mu in mu_values:
        config = SimulationConfig(
            domain=Domain.THREE_BODY,
            dt=0.01,
            n_steps=1,
            parameters={"mu": mu, "x0": 0.5, "y0": 0.5, "vx0": 0.0, "vy0": 0.0},
        )
        sim = ThreeBodySimulation(config)
        sim.reset()
        lps = sim.lagrange_points()

        l4_x.append(lps["L4"][0])
        l4_y.append(lps["L4"][1])
        l5_x.append(lps["L5"][0])
        l5_y.append(lps["L5"][1])

    return {
        "mu_values": mu_values,
        "l4_x": np.array(l4_x),
        "l4_y": np.array(l4_y),
        "l5_x": np.array(l5_x),
        "l5_y": np.array(l5_y),
        "l4_x_theory": 0.5 - mu_values,
        "l4_y_theory": np.full(n_samples, np.sqrt(3) / 2),
    }


def generate_lyapunov_vs_mu_data(
    n_samples: int = 15,
    n_lyap_steps: int = 3000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep mass ratio mu and compute Lyapunov exponent.

    Args:
        n_samples: Number of mu values.
        n_lyap_steps: Steps per Lyapunov computation.
        dt: Timestep.

    Returns:
        Dict with mu values and Lyapunov exponents.
    """
    mu_values = np.linspace(0.001, 0.1, n_samples)
    lyap_values = []

    for i, mu in enumerate(mu_values):
        config = SimulationConfig(
            domain=Domain.THREE_BODY,
            dt=dt,
            n_steps=1,
            parameters={
                "mu": mu,
                "x0": 0.5,
                "y0": 0.3,
                "vx0": 0.1,
                "vy0": -0.1,
            },
        )
        sim = ThreeBodySimulation(config)
        sim.reset()

        lyap = sim.compute_lyapunov(n_steps=n_lyap_steps, dt=dt)
        lyap_values.append(lyap)

        if (i + 1) % 5 == 0:
            logger.info(f"  Lyapunov measurement {i + 1}/{n_samples}")

    return {
        "mu_values": mu_values,
        "lyapunov": np.array(lyap_values),
    }


def run_three_body_rediscovery(
    output_dir: str | Path = "output/rediscovery/three_body",
    n_iterations: int = 40,
) -> dict:
    """Run the full three-body problem rediscovery.

    1. Verify Jacobi constant conservation
    2. Sweep mu, compute L4/L5, run PySR to find L4_x = 0.5 - mu
    3. Compute Lyapunov exponent vs mu

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "three_body",
        "targets": {
            "jacobi_constant": "C_J = -(vx^2+vy^2) + 2*Omega = const",
            "lagrange_l4_x": "L4_x = 0.5 - mu",
            "lagrange_l4_y": "L4_y = sqrt(3)/2",
            "lyapunov_vs_mu": "Positive Lyapunov => chaos",
        },
    }

    # --- Part 1: Jacobi constant conservation ---
    logger.info("Part 1: Verifying Jacobi constant conservation...")
    jacobi_data = generate_jacobi_conservation_data(
        mu=0.01, n_steps=5000, dt=0.005,
    )
    results["jacobi_conservation"] = {
        "CJ0": jacobi_data["CJ0"],
        "max_drift": jacobi_data["max_drift"],
        "mean_drift": jacobi_data["mean_drift"],
        "final_drift": jacobi_data["final_drift"],
    }
    logger.info(
        f"  Jacobi drift: max={jacobi_data['max_drift']:.2e}, "
        f"mean={jacobi_data['mean_drift']:.2e}"
    )

    # --- Part 2: Lagrange point L4 location via PySR ---
    logger.info("Part 2: Sweeping mu to find L4/L5 locations...")
    lp_data = generate_lagrange_point_data(n_samples=20)

    # Check analytical accuracy
    l4x_error = np.abs(lp_data["l4_x"] - lp_data["l4_x_theory"])
    results["lagrange_points"] = {
        "n_samples": len(lp_data["mu_values"]),
        "l4_x_max_error": float(np.max(l4x_error)),
        "l4_y_value": float(np.mean(lp_data["l4_y"])),
        "l4_y_theory": float(np.sqrt(3) / 2),
    }
    logger.info(
        f"  L4 x-coordinate max error from theory: {np.max(l4x_error):.2e}"
    )
    logger.info(
        f"  L4 y-coordinate: {np.mean(lp_data['l4_y']):.6f} "
        f"(theory: {np.sqrt(3)/2:.6f})"
    )

    # PySR: L4_x = f(mu) -- should find 0.5 - mu
    logger.info(
        f"  Running PySR for L4_x = f(mu) with {n_iterations} iterations..."
    )
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = lp_data["mu_values"].reshape(-1, 1)
        y = lp_data["l4_x"]

        discoveries = run_symbolic_regression(
            X,
            y,
            variable_names=["mu_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=10,
            populations=20,
            population_size=40,
        )

        results["l4x_pysr"] = {
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
            results["l4x_pysr"]["best"] = best.expression
            results["l4x_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["l4x_pysr"] = {"error": str(e)}

    # --- Part 3: Lyapunov exponent vs mu ---
    logger.info("Part 3: Computing Lyapunov exponent vs mass ratio...")
    lyap_data = generate_lyapunov_vs_mu_data(
        n_samples=15, n_lyap_steps=3000, dt=0.005,
    )

    positive_lyap = lyap_data["lyapunov"] > 0
    results["lyapunov_vs_mu"] = {
        "n_samples": len(lyap_data["mu_values"]),
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "min_lyapunov": float(np.min(lyap_data["lyapunov"])),
        "mean_lyapunov": float(np.mean(lyap_data["lyapunov"])),
        "n_chaotic": int(np.sum(positive_lyap)),
        "fraction_chaotic": float(np.mean(positive_lyap)),
    }
    logger.info(
        f"  Lyapunov range: [{np.min(lyap_data['lyapunov']):.3f}, "
        f"{np.max(lyap_data['lyapunov']):.3f}]"
    )
    logger.info(
        f"  Chaotic orbits: {np.sum(positive_lyap)}/{len(lyap_data['mu_values'])}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "jacobi_data.npz",
        times=jacobi_data["times"],
        jacobi_values=jacobi_data["jacobi_values"],
    )
    np.savez(
        output_path / "lagrange_data.npz",
        mu_values=lp_data["mu_values"],
        l4_x=lp_data["l4_x"],
        l4_y=lp_data["l4_y"],
    )
    np.savez(
        output_path / "lyapunov_data.npz",
        mu_values=lyap_data["mu_values"],
        lyapunov=lyap_data["lyapunov"],
    )

    return results
