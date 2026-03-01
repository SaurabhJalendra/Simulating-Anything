"""Kepler two-body orbit rediscovery.

Targets:
- Kepler's third law: T = 2*pi * a^(3/2) / sqrt(GM), i.e. T ~ a^(3/2)
- Energy conservation: E = 0.5*(v_r^2 + v_theta^2) - GM/r = const
- Angular momentum conservation: L = r * v_theta = const
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.kepler import KeplerOrbit
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_period_vs_sma_data(
    n_samples: int = 30,
    dt: float = 0.001,
    GM: float = 1.0,
    eccentricity: float = 0.3,
) -> dict[str, np.ndarray]:
    """Sweep semi-major axis and measure orbital period.

    For each value of a, run a full orbit (theta completes 2*pi),
    measure the elapsed time. PySR should find T = c * a^(3/2).

    Args:
        n_samples: Number of semi-major axis values to sweep.
        dt: Integration timestep.
        GM: Gravitational parameter.
        eccentricity: Orbital eccentricity (same for all orbits).

    Returns:
        Dict with arrays: a_values, T_measured, T_theory.
    """
    a_values = np.linspace(0.5, 5.0, n_samples)
    T_measured = []
    T_theory = []
    a_valid = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=Domain.KEPLER,
            dt=dt,
            n_steps=1,  # We manually step
            parameters={
                "GM": GM,
                "initial_r": a,
                "eccentricity": eccentricity,
            },
        )
        sim = KeplerOrbit(config)
        sim.reset()

        # Run until theta completes 2*pi
        theta_start = sim.observe()[1]
        total_theta = 0.0
        prev_theta = theta_start
        n_step = 0
        max_steps = int(200.0 / dt)  # Safety limit

        while total_theta < 2.0 * np.pi and n_step < max_steps:
            state = sim.step()
            n_step += 1
            curr_theta = state[1]
            d_theta = curr_theta - prev_theta
            # Handle angle wrapping (should not be needed for continuous theta)
            total_theta += d_theta
            prev_theta = curr_theta

            if not np.isfinite(state[0]) or state[0] <= 0:
                break

        if total_theta >= 2.0 * np.pi and np.isfinite(state[0]):
            T_meas = n_step * dt
            T_theo = 2.0 * np.pi * a**1.5 / np.sqrt(GM)
            T_measured.append(T_meas)
            T_theory.append(T_theo)
            a_valid.append(a)

        if (i + 1) % 10 == 0:
            logger.info(f"  Period measurement {i + 1}/{n_samples}")

    return {
        "a_values": np.array(a_valid),
        "T_measured": np.array(T_measured),
        "T_theory": np.array(T_theory),
        "GM": GM,
        "eccentricity": eccentricity,
    }


def generate_energy_conservation_data(
    n_orbits: int = 10,
    dt: float = 0.001,
    GM: float = 1.0,
    a: float = 1.0,
    eccentricity: float = 0.5,
) -> dict[str, np.ndarray]:
    """Track energy over multiple orbits to verify conservation.

    Args:
        n_orbits: Number of complete orbits to simulate.
        dt: Integration timestep.
        GM: Gravitational parameter.
        a: Semi-major axis.
        eccentricity: Orbital eccentricity.

    Returns:
        Dict with arrays: times, energies, E0, drift.
    """
    T_period = 2.0 * np.pi * a**1.5 / np.sqrt(GM)
    n_steps = int(n_orbits * T_period / dt)

    config = SimulationConfig(
        domain=Domain.KEPLER,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "GM": GM,
            "initial_r": a,
            "eccentricity": eccentricity,
        },
    )
    sim = KeplerOrbit(config)
    sim.reset()

    E0 = sim.energy
    times = [0.0]
    energies = [E0]

    for step_i in range(n_steps):
        sim.step()
        times.append((step_i + 1) * dt)
        energies.append(sim.energy)

    energies = np.array(energies)
    times = np.array(times)
    drift = np.abs(energies - E0) / np.abs(E0)

    return {
        "times": times,
        "energies": energies,
        "E0": E0,
        "max_drift": float(np.max(drift)),
        "mean_drift": float(np.mean(drift)),
        "final_drift": float(drift[-1]),
    }


def generate_angular_momentum_data(
    n_orbits: int = 10,
    dt: float = 0.001,
    GM: float = 1.0,
    a: float = 1.0,
    eccentricity: float = 0.5,
) -> dict[str, np.ndarray]:
    """Track angular momentum over multiple orbits to verify conservation.

    Args:
        n_orbits: Number of complete orbits to simulate.
        dt: Integration timestep.
        GM: Gravitational parameter.
        a: Semi-major axis.
        eccentricity: Orbital eccentricity.

    Returns:
        Dict with arrays: times, angular_momenta, L0, drift.
    """
    T_period = 2.0 * np.pi * a**1.5 / np.sqrt(GM)
    n_steps = int(n_orbits * T_period / dt)

    config = SimulationConfig(
        domain=Domain.KEPLER,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "GM": GM,
            "initial_r": a,
            "eccentricity": eccentricity,
        },
    )
    sim = KeplerOrbit(config)
    sim.reset()

    L0 = sim.angular_momentum
    times = [0.0]
    angular_momenta = [L0]

    for step_i in range(n_steps):
        sim.step()
        times.append((step_i + 1) * dt)
        angular_momenta.append(sim.angular_momentum)

    angular_momenta = np.array(angular_momenta)
    times = np.array(times)
    drift = np.abs(angular_momenta - L0) / np.abs(L0)

    return {
        "times": times,
        "angular_momenta": angular_momenta,
        "L0": L0,
        "max_drift": float(np.max(drift)),
        "mean_drift": float(np.mean(drift)),
        "final_drift": float(drift[-1]),
    }


def run_kepler_rediscovery(
    output_dir: str | Path = "output/rediscovery/kepler",
    n_iterations: int = 40,
) -> dict:
    """Run the full Kepler orbit rediscovery.

    1. Sweep semi-major axis a, measure period T, run PySR to find T ~ a^(3/2)
    2. Verify energy conservation over many orbits
    3. Verify angular momentum conservation

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "kepler",
        "targets": {
            "kepler_third_law": "T = 2*pi * a^(3/2) / sqrt(GM)",
            "energy_conservation": "E = 0.5*(v_r^2 + v_theta^2) - GM/r = const",
            "angular_momentum": "L = r * v_theta = const",
        },
    }

    # --- Part 1: Kepler's third law via period measurement ---
    logger.info("Part 1: Measuring orbital period vs semi-major axis...")
    period_data = generate_period_vs_sma_data(
        n_samples=30, dt=0.001, GM=1.0, eccentricity=0.3,
    )

    rel_error = np.abs(
        period_data["T_measured"] - period_data["T_theory"]
    ) / period_data["T_theory"]
    results["period_accuracy"] = {
        "n_samples": len(period_data["a_values"]),
        "mean_relative_error": float(np.mean(rel_error)),
        "max_relative_error": float(np.max(rel_error)),
    }
    logger.info(
        f"  Period accuracy: mean error = {np.mean(rel_error):.4%}"
    )

    # PySR: T = f(a) -- should find T ~ a^(3/2) since GM=1
    logger.info(
        f"  Running PySR for T = f(a) with {n_iterations} iterations..."
    )

    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = period_data["a_values"].reshape(-1, 1)
        y = period_data["T_measured"]

        discoveries = run_symbolic_regression(
            X,
            y,
            variable_names=["a_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square", "cube"],
            max_complexity=15,
            populations=20,
            population_size=40,
        )

        results["period_pysr"] = {
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
            results["period_pysr"]["best"] = best.expression
            results["period_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["period_pysr"] = {"error": str(e)}

    # --- Part 2: Energy conservation ---
    logger.info("Part 2: Verifying energy conservation...")
    energy_data = generate_energy_conservation_data(
        n_orbits=10, dt=0.001, GM=1.0, a=1.0, eccentricity=0.5,
    )
    results["energy_conservation"] = {
        "E0": energy_data["E0"],
        "max_drift": energy_data["max_drift"],
        "mean_drift": energy_data["mean_drift"],
        "final_drift": energy_data["final_drift"],
    }
    logger.info(
        f"  Energy drift: max={energy_data['max_drift']:.2e}, "
        f"mean={energy_data['mean_drift']:.2e}"
    )

    # --- Part 3: Angular momentum conservation ---
    logger.info("Part 3: Verifying angular momentum conservation...")
    L_data = generate_angular_momentum_data(
        n_orbits=10, dt=0.001, GM=1.0, a=1.0, eccentricity=0.5,
    )
    results["angular_momentum_conservation"] = {
        "L0": L_data["L0"],
        "max_drift": L_data["max_drift"],
        "mean_drift": L_data["mean_drift"],
        "final_drift": L_data["final_drift"],
    }
    logger.info(
        f"  Angular momentum drift: max={L_data['max_drift']:.2e}, "
        f"mean={L_data['mean_drift']:.2e}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "period_data.npz",
        a_values=period_data["a_values"],
        T_measured=period_data["T_measured"],
        T_theory=period_data["T_theory"],
    )
    np.savez(
        output_path / "energy_data.npz",
        times=energy_data["times"],
        energies=energy_data["energies"],
    )
    np.savez(
        output_path / "angular_momentum_data.npz",
        times=L_data["times"],
        angular_momenta=L_data["angular_momenta"],
    )

    return results
