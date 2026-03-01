"""Three-species food chain rediscovery.

Targets:
- ODE recovery via SINDy:
    dx/dt = a1*x - b1*x*y
    dy/dt = -a2*y + b1*x*y - b2*y*z
    dz/dt = -a3*z + b2*y*z
- Predator-free equilibrium: x* = a2/b1, y* = a1/b1
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.three_species import ThreeSpecies
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    a1: float = 1.0,
    b1: float = 0.5,
    a2: float = 0.5,
    b2: float = 0.2,
    a3: float = 0.3,
    x0: float = 1.0,
    y0: float = 0.5,
    z0: float = 0.5,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Generate a single long trajectory for SINDy ODE recovery.

    Returns dict with states array, time, and parameters.
    """
    config = SimulationConfig(
        domain=Domain.THREE_SPECIES,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a1": a1, "b1": b1, "a2": a2, "b2": b2, "a3": a3,
            "x0": x0, "y0": y0, "z0": z0,
        },
    )

    sim = ThreeSpecies(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "x": states_arr[:, 0],
        "y": states_arr[:, 1],
        "z": states_arr[:, 2],
        "dt": dt,
        "a1": a1, "b1": b1, "a2": a2, "b2": b2, "a3": a3,
    }


def generate_equilibrium_data(
    n_samples: int = 100,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep parameters and measure time-averaged populations.

    Varies a1, b1, a2, b2, a3 and computes time-averaged x, y, z.
    Returns dict with parameter arrays and measured averages.
    """
    rng = np.random.default_rng(42)

    all_a1, all_b1, all_a2, all_b2, all_a3 = [], [], [], [], []
    all_x_avg, all_y_avg, all_z_avg = [], [], []

    for i in range(n_samples):
        a1 = rng.uniform(0.5, 2.0)
        b1 = rng.uniform(0.2, 0.8)
        a2 = rng.uniform(0.2, 0.8)
        b2 = rng.uniform(0.1, 0.5)
        a3 = rng.uniform(0.1, 0.5)

        # Start with moderate populations
        x0 = rng.uniform(0.5, 2.0)
        y0 = rng.uniform(0.3, 1.5)
        z0 = rng.uniform(0.2, 1.0)

        config = SimulationConfig(
            domain=Domain.THREE_SPECIES,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a1": a1, "b1": b1, "a2": a2, "b2": b2, "a3": a3,
                "x0": x0, "y0": y0, "z0": z0,
            },
        )

        sim = ThreeSpecies(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            states.append(sim.step().copy())

        trajectory = np.array(states)
        # Skip initial transient (first 20%)
        skip = n_steps // 5
        x_avg = np.mean(trajectory[skip:, 0])
        y_avg = np.mean(trajectory[skip:, 1])
        z_avg = np.mean(trajectory[skip:, 2])

        all_a1.append(a1)
        all_b1.append(b1)
        all_a2.append(a2)
        all_b2.append(b2)
        all_a3.append(a3)
        all_x_avg.append(x_avg)
        all_y_avg.append(y_avg)
        all_z_avg.append(z_avg)

        if (i + 1) % 25 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} trajectories")

    return {
        "a1": np.array(all_a1),
        "b1": np.array(all_b1),
        "a2": np.array(all_a2),
        "b2": np.array(all_b2),
        "a3": np.array(all_a3),
        "x_avg": np.array(all_x_avg),
        "y_avg": np.array(all_y_avg),
        "z_avg": np.array(all_z_avg),
    }


def run_three_species_rediscovery(
    output_dir: str | Path = "output/rediscovery/three_species",
    n_iterations: int = 40,
    n_equilibrium_samples: int = 100,
) -> dict:
    """Run the full three-species food chain rediscovery.

    1. Generate a trajectory for SINDy ODE recovery
    2. Run SINDy to recover the three coupled ODEs
    3. Generate equilibrium data across parameter sweeps
    4. Compare with known results

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "three_species",
        "targets": {
            "ode_x": "dx/dt = a1*x - b1*x*y",
            "ode_y": "dy/dt = -a2*y + b1*x*y - b2*y*z",
            "ode_z": "dz/dt = -a3*z + b2*y*z",
            "equilibrium_x": "x* = a2/b1",
            "equilibrium_y": "y* = a1/b1",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for three-species food chain...")
    data = generate_trajectory_data(
        a1=1.0, b1=0.5, a2=0.5, b2=0.2, a3=0.3,
        n_steps=10000, dt=0.005,
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["x", "y", "z"],
            threshold=0.05,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries
            ],
        }
        if sindy_discoveries:
            best = sindy_discoveries[0]
            results["sindy_ode"]["best"] = best.expression
            results["sindy_ode"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  SINDy best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Equilibrium data ---
    logger.info("Part 2: Generating equilibrium sweep data...")
    eq_data = generate_equilibrium_data(
        n_samples=n_equilibrium_samples, n_steps=10000, dt=0.01,
    )

    results["equilibrium_data"] = {
        "n_samples": len(eq_data["a1"]),
        "x_avg_mean": float(np.mean(eq_data["x_avg"])),
        "y_avg_mean": float(np.mean(eq_data["y_avg"])),
        "z_avg_mean": float(np.mean(eq_data["z_avg"])),
    }

    logger.info(
        f"  Mean populations: x={np.mean(eq_data['x_avg']):.3f}, "
        f"y={np.mean(eq_data['y_avg']):.3f}, z={np.mean(eq_data['z_avg']):.3f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "trajectory_data.npz",
        states=data["states"],
    )
    np.savez(
        output_path / "equilibrium_data.npz",
        **{k: v for k, v in eq_data.items()},
    )

    return results
