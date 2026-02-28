"""Double pendulum rediscovery.

Targets:
- Energy conservation: E = const along trajectories
- Small-angle period: T = 2*pi*sqrt(L/g)
- SINDy recovery of equations of motion (simplified regime)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.chaotic_ode import DoublePendulumSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_energy_conservation_data(
    n_trajectories: int = 50,
    n_steps: int = 10000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate trajectories and compute energy drift.

    For each trajectory, compute total energy at every timestep.
    Energy conservation means E(t) = E(0) for all t.
    """
    rng = np.random.default_rng(42)

    all_params = []
    all_energy_drift = []
    all_mean_energy = []
    all_max_drift = []

    for i in range(n_trajectories):
        th1 = rng.uniform(-np.pi, np.pi)
        th2 = rng.uniform(-np.pi, np.pi)
        w1 = rng.uniform(-2, 2)
        w2 = rng.uniform(-2, 2)
        L1 = rng.uniform(0.5, 2.0)
        L2 = rng.uniform(0.5, 2.0)
        m1 = rng.uniform(0.5, 2.0)
        m2 = rng.uniform(0.5, 2.0)

        config = SimulationConfig(
            domain=Domain.CHAOTIC_ODE,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "m1": m1, "m2": m2, "L1": L1, "L2": L2, "g": 9.81,
                "theta1_0": th1, "theta2_0": th2,
                "omega1_0": w1, "omega2_0": w2,
            },
        )
        sim = DoublePendulumSimulation(config)
        sim.reset()

        energies = [sim.total_energy()]
        for _ in range(n_steps):
            sim.step()
            energies.append(sim.total_energy())

        energies = np.array(energies)
        E0 = energies[0]
        drift = np.abs(energies - E0) / max(abs(E0), 1e-10)

        all_params.append({"L1": L1, "L2": L2, "m1": m1, "m2": m2})
        all_energy_drift.append(drift[-1])
        all_mean_energy.append(np.mean(energies))
        all_max_drift.append(np.max(drift))

        if (i + 1) % 10 == 0:
            logger.info(f"  Energy check {i + 1}/{n_trajectories}: max drift = {np.max(drift):.2e}")

    return {
        "n_trajectories": n_trajectories,
        "final_drift": np.array(all_energy_drift),
        "max_drift": np.array(all_max_drift),
        "mean_energy": np.array(all_mean_energy),
    }


def generate_small_angle_period_data(
    n_samples: int = 100,
    n_steps: int = 50000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate small-angle pendulum 1 trajectories and measure period.

    For small angles and m2 << m1, pendulum 1 behaves like a simple pendulum
    with period T = 2*pi*sqrt(L1/g).
    """
    rng = np.random.default_rng(42)

    all_L1 = []
    all_g = []
    all_T_measured = []
    all_T_theory = []

    for i in range(n_samples):
        L1 = rng.uniform(0.3, 3.0)
        g_val = 9.81
        # Small angle: theta1 ~ 0.05 rad, m2 << m1 to decouple
        config = SimulationConfig(
            domain=Domain.CHAOTIC_ODE,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "m1": 10.0, "m2": 0.01,
                "L1": L1, "L2": 1.0, "g": g_val,
                "theta1_0": 0.05, "theta2_0": 0.0,
                "omega1_0": 0.0, "omega2_0": 0.0,
            },
        )
        sim = DoublePendulumSimulation(config)
        sim.reset()

        theta1_history = [sim.observe()[0]]
        for _ in range(n_steps):
            state = sim.step()
            theta1_history.append(state[0])

        theta1 = np.array(theta1_history)

        # Find period from zero crossings
        crossings = []
        for j in range(1, len(theta1)):
            if theta1[j - 1] < 0 and theta1[j] >= 0:
                # Linear interpolation for precise crossing
                frac = -theta1[j - 1] / (theta1[j] - theta1[j - 1])
                crossings.append((j - 1 + frac) * dt)

        if len(crossings) >= 3:
            periods = np.diff(crossings)
            T_measured = np.median(periods)
            T_theory = 2 * np.pi * np.sqrt(L1 / g_val)

            all_L1.append(L1)
            all_g.append(g_val)
            all_T_measured.append(T_measured)
            all_T_theory.append(T_theory)

        if (i + 1) % 25 == 0:
            logger.info(f"  Period measurement {i + 1}/{n_samples}")

    return {
        "L1": np.array(all_L1),
        "g": np.array(all_g),
        "T_measured": np.array(all_T_measured),
        "T_theory": np.array(all_T_theory),
    }


def run_double_pendulum_rediscovery(
    output_dir: str | Path = "output/rediscovery/double_pendulum",
    n_iterations: int = 40,
) -> dict:
    """Run the full double pendulum rediscovery.

    1. Energy conservation verification
    2. Small-angle period: T = 2*pi*sqrt(L/g) via PySR
    3. Report results
    """
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "double_pendulum",
        "targets": {
            "energy_conservation": "E(t) = E(0) for all t",
            "period": "T = 2*pi*sqrt(L/g)",
        },
    }

    # --- Part 1: Energy conservation ---
    logger.info("Part 1: Energy conservation verification...")
    energy_data = generate_energy_conservation_data(
        n_trajectories=50, n_steps=10000, dt=0.001
    )

    results["energy_conservation"] = {
        "n_trajectories": energy_data["n_trajectories"],
        "mean_final_drift": float(np.mean(energy_data["final_drift"])),
        "max_final_drift": float(np.max(energy_data["final_drift"])),
        "mean_max_drift": float(np.mean(energy_data["max_drift"])),
    }
    logger.info(
        f"  Energy drift: mean={np.mean(energy_data['final_drift']):.2e}, "
        f"max={np.max(energy_data['final_drift']):.2e}"
    )

    # --- Part 2: Small-angle period ---
    logger.info("Part 2: Small-angle period measurement...")
    period_data = generate_small_angle_period_data(n_samples=100, n_steps=50000, dt=0.001)

    rel_error = np.abs(period_data["T_measured"] - period_data["T_theory"]) / period_data["T_theory"]
    results["period_accuracy"] = {
        "n_samples": len(period_data["L1"]),
        "mean_relative_error": float(np.mean(rel_error)),
        "max_relative_error": float(np.max(rel_error)),
    }
    logger.info(f"  Period accuracy: mean error = {np.mean(rel_error):.4%}")

    # PySR: T = f(L, g)
    logger.info(f"  Running PySR for T = f(L, g) with {n_iterations} iterations...")
    X = np.column_stack([period_data["L1"], period_data["g"]])
    y = period_data["T_measured"]

    discoveries = run_symbolic_regression(
        X,
        y,
        variable_names=["L", "g"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "square"],
        max_complexity=15,
        populations=20,
        population_size=40,
    )

    results["period_pysr"] = {
        "n_discoveries": len(discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in discoveries[:5]
        ],
    }
    if discoveries:
        best = discoveries[0]
        results["period_pysr"]["best"] = best.expression
        results["period_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "period_data.npz",
        L1=period_data["L1"],
        g=period_data["g"],
        T_measured=period_data["T_measured"],
        T_theory=period_data["T_theory"],
    )

    return results
