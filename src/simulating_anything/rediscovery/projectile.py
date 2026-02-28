"""Projectile range equation rediscovery.

Target: R = v^2 * sin(2*theta) / g (no-drag case)
Method: Vary v0 and angle, compute range, run PySR to recover the equation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.rigid_body import ProjectileSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_projectile_data(
    n_speeds: int = 15,
    n_angles: int = 15,
    speed_range: tuple[float, float] = (10.0, 50.0),
    angle_range: tuple[float, float] = (10.0, 80.0),
    gravity: float = 9.81,
    drag_coefficient: float = 0.0,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate projectile trajectories and compute ranges.

    Returns dict with keys: v0, theta (radians), g, range, max_height.
    Each is a 1D array of shape (n_speeds * n_angles,).
    """
    speeds = np.linspace(speed_range[0], speed_range[1], n_speeds)
    angles_deg = np.linspace(angle_range[0], angle_range[1], n_angles)

    all_v0 = []
    all_theta = []
    all_g = []
    all_range = []
    all_max_height = []

    for v0 in speeds:
        for angle in angles_deg:
            config = SimulationConfig(
                domain=Domain.RIGID_BODY,
                dt=dt,
                n_steps=50000,
                parameters={
                    "gravity": gravity,
                    "drag_coefficient": drag_coefficient,
                    "initial_speed": float(v0),
                    "launch_angle": float(angle),
                    "mass": 1.0,
                },
            )
            sim = ProjectileSimulation(config)
            sim.reset()

            # Run until landing
            for _ in range(config.n_steps):
                state = sim.step()
                if sim._landed:
                    break

            x_final = state[0]
            # Collect max height from trajectory
            sim2 = ProjectileSimulation(config)
            sim2.reset()
            max_y = 0.0
            for _ in range(config.n_steps):
                s = sim2.step()
                max_y = max(max_y, s[1])
                if sim2._landed:
                    break

            all_v0.append(v0)
            all_theta.append(np.radians(angle))
            all_g.append(gravity)
            all_range.append(x_final)
            all_max_height.append(max_y)

    return {
        "v0": np.array(all_v0),
        "theta": np.array(all_theta),
        "g": np.array(all_g),
        "range": np.array(all_range),
        "max_height": np.array(all_max_height),
    }


def theoretical_range(v0: np.ndarray, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Compute theoretical range R = v^2 * sin(2*theta) / g."""
    return v0**2 * np.sin(2 * theta) / g


def run_projectile_rediscovery(
    output_dir: str | Path = "output/rediscovery/projectile",
    n_iterations: int = 40,
) -> dict:
    """Run the full projectile range equation rediscovery.

    1. Generate data (no drag) varying v0 and angle
    2. Run PySR to find R = f(v0, theta, g)
    3. Compare with known R = v^2*sin(2*theta)/g
    4. Save results

    Returns dict with discoveries, data stats, and comparison.
    """
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating projectile trajectory data (no drag)...")
    data = generate_projectile_data(
        n_speeds=15, n_angles=15, drag_coefficient=0.0, dt=0.001
    )

    # Feature matrix: [v0, theta, g]
    X = np.column_stack([data["v0"], data["theta"], data["g"]])
    y = data["range"]

    # Theoretical comparison
    R_theory = theoretical_range(data["v0"], data["theta"], data["g"])
    rel_error = np.abs(y - R_theory) / np.maximum(R_theory, 1e-10)
    logger.info(
        f"Simulation vs theory: mean relative error = {np.mean(rel_error):.4%}, "
        f"max = {np.max(rel_error):.4%}"
    )

    # Run PySR
    logger.info(f"Running PySR with {n_iterations} iterations...")
    discoveries = run_symbolic_regression(
        X,
        y,
        variable_names=["v0", "theta", "g"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "square"],
        max_complexity=20,
        populations=20,
        population_size=40,
    )

    # Build results
    results = {
        "domain": "projectile",
        "target_equation": "R = v0^2 * sin(2*theta) / g",
        "n_samples": len(y),
        "simulation_vs_theory_mean_error": float(np.mean(rel_error)),
        "simulation_vs_theory_max_error": float(np.max(rel_error)),
        "n_discoveries": len(discoveries),
        "discoveries": [],
    }

    for d in discoveries[:10]:
        results["discoveries"].append({
            "expression": d.expression,
            "confidence": d.confidence,
            "r_squared": d.evidence.fit_r_squared,
            "description": d.description,
        })

    if discoveries:
        best = discoveries[0]
        results["best_equation"] = best.expression
        results["best_r_squared"] = best.evidence.fit_r_squared
        logger.info(f"Best discovered equation: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")
    else:
        results["best_equation"] = "None found"
        results["best_r_squared"] = 0.0
        logger.warning("No equations discovered")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Save data for reproducibility
    np.savez(
        output_path / "data.npz",
        v0=data["v0"],
        theta=data["theta"],
        g=data["g"],
        range_sim=data["range"],
        range_theory=R_theory,
        max_height=data["max_height"],
    )

    return results
