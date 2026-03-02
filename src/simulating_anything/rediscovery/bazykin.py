"""Bazykin predator-prey rediscovery.

Targets:
- ODE recovery via SINDy:
    dx/dt = x*(1 - x) - x*y/(1 + alpha*x)
    dy/dt = -gamma*y + x*y/(1 + alpha*x) - delta*y^2
- Hopf bifurcation detection via gamma sweep
- Coexistence equilibrium verification
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.bazykin import BazykinSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.BAZYKIN


def generate_trajectory_data(
    alpha: float = 0.1,
    gamma: float = 0.1,
    delta: float = 0.01,
    x_0: float = 0.5,
    y_0: float = 0.5,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Generate a single long trajectory for SINDy ODE recovery.

    Returns dict with states array, time, and parameters.
    """
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha": alpha, "gamma": gamma, "delta": delta,
            "x_0": x_0, "y_0": y_0,
        },
    )

    sim = BazykinSimulation(config)
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
        "dt": dt,
        "alpha": alpha, "gamma": gamma, "delta": delta,
    }


def generate_bifurcation_data(
    n_gamma_values: int = 30,
    gamma_min: float = 0.01,
    gamma_max: float = 0.5,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep predator death rate gamma to detect Hopf bifurcation.

    For each gamma, runs a long trajectory and measures:
    - Time-averaged prey and predator populations
    - Oscillation amplitude (max - min in second half of trajectory)

    Returns dict with gamma values, averages, and amplitudes.
    """
    gamma_values = np.linspace(gamma_min, gamma_max, n_gamma_values)

    all_gamma = []
    all_x_avg = []
    all_y_avg = []
    all_x_amp = []
    all_y_amp = []

    for i, gamma_val in enumerate(gamma_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "alpha": 0.1, "gamma": float(gamma_val),
                "delta": 0.01, "x_0": 0.5, "y_0": 0.5,
            },
        )

        sim = BazykinSimulation(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)

        # Use second half to avoid transients
        half = n_steps // 2
        x_half = trajectory[half:, 0]
        y_half = trajectory[half:, 1]

        all_gamma.append(float(gamma_val))
        all_x_avg.append(float(np.mean(x_half)))
        all_y_avg.append(float(np.mean(y_half)))
        all_x_amp.append(float(np.max(x_half) - np.min(x_half)))
        all_y_amp.append(float(np.max(y_half) - np.min(y_half)))

        if (i + 1) % 10 == 0:
            logger.info(
                f"  Bifurcation sweep: {i + 1}/{n_gamma_values} gamma values"
            )

    return {
        "gamma": np.array(all_gamma),
        "x_avg": np.array(all_x_avg),
        "y_avg": np.array(all_y_avg),
        "x_amplitude": np.array(all_x_amp),
        "y_amplitude": np.array(all_y_amp),
    }


def run_bazykin_rediscovery(
    output_dir: str | Path = "output/rediscovery/bazykin",
    n_iterations: int = 40,
    n_gamma_values: int = 30,
) -> dict:
    """Run the full Bazykin predator-prey rediscovery.

    1. Generate a trajectory for SINDy ODE recovery
    2. Run SINDy to recover the coupled ODEs
    3. Sweep gamma to detect Hopf bifurcation
    4. Verify coexistence equilibrium

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "bazykin",
        "targets": {
            "ode_x": "dx/dt = x*(1-x) - x*y/(1+alpha*x)",
            "ode_y": "dy/dt = -gamma*y + x*y/(1+alpha*x) - delta*y^2",
            "equilibrium": "coexistence (x*, y*) from nullcline intersection",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for Bazykin model...")
    data = generate_trajectory_data(
        alpha=0.1, gamma=0.1, delta=0.01,
        n_steps=10000, dt=0.005,
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["x", "y"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
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
        for disc in sindy_discoveries:
            logger.info(f"  SINDy: {disc.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: Hopf bifurcation detection via gamma sweep ---
    logger.info("Part 2: Detecting Hopf bifurcation via gamma sweep...")
    bif_data = generate_bifurcation_data(
        n_gamma_values=n_gamma_values, gamma_min=0.01, gamma_max=0.5,
        n_steps=20000, dt=0.01,
    )

    # Detect gamma_c from data: find where oscillation amplitude drops
    # below threshold (stable equilibrium appears as gamma increases)
    amp_threshold = 0.01
    gamma_c_detected = None
    for j in range(len(bif_data["x_amplitude"])):
        if bif_data["x_amplitude"][j] < amp_threshold:
            gamma_c_detected = float(bif_data["gamma"][j])
            break

    results["bifurcation"] = {
        "n_gamma_values": n_gamma_values,
        "gamma_range": [
            float(bif_data["gamma"][0]),
            float(bif_data["gamma"][-1]),
        ],
        "gamma_c_detected": gamma_c_detected,
        "max_x_amplitude": float(np.max(bif_data["x_amplitude"])),
        "max_y_amplitude": float(np.max(bif_data["y_amplitude"])),
    }

    logger.info(f"  Detected gamma_c = {gamma_c_detected}")
    logger.info(
        f"  Max prey amplitude = {np.max(bif_data['x_amplitude']):.4f}"
    )

    # --- Part 3: Coexistence equilibrium verification ---
    logger.info("Part 3: Verifying coexistence equilibrium...")
    try:
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=0.01,
            n_steps=1000,
            parameters={
                "alpha": 0.1, "gamma": 0.1, "delta": 0.01,
                "x_0": 0.5, "y_0": 0.5,
            },
        )
        sim = BazykinSimulation(config)
        x_star, y_star = sim.coexistence_equilibrium()

        # Verify derivatives are zero at equilibrium
        eq_state = np.array([x_star, y_star])
        dy = sim._derivatives(eq_state)

        results["equilibrium"] = {
            "x_star": x_star,
            "y_star": y_star,
            "derivative_magnitude": float(np.linalg.norm(dy)),
            "is_stable": sim.is_stable(),
        }
        logger.info(f"  Equilibrium: x*={x_star:.6f}, y*={y_star:.6f}")
        logger.info(f"  |dy/dt| at eq = {np.linalg.norm(dy):.2e}")
        logger.info(f"  Stable: {sim.is_stable()}")
    except Exception as exc:
        logger.warning(f"Equilibrium computation failed: {exc}")
        results["equilibrium"] = {"error": str(exc)}

    # --- Part 4: PySR for bifurcation curve ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Fit x_amplitude as function of gamma
        X_bif = bif_data["gamma"].reshape(-1, 1)
        y_bif = bif_data["x_amplitude"]

        pysr_discoveries = run_symbolic_regression(
            X_bif,
            y_bif,
            variable_names=["g_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=15,
            populations=20,
            population_size=40,
        )

        results["pysr_amplitude"] = {
            "n_discoveries": len(pysr_discoveries),
            "discoveries": [
                {
                    "expression": disc.expression,
                    "r_squared": disc.evidence.fit_r_squared,
                }
                for disc in pysr_discoveries[:5]
            ],
        }
        if pysr_discoveries:
            best = pysr_discoveries[0]
            results["pysr_amplitude"]["best"] = best.expression
            results["pysr_amplitude"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  PySR amplitude: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as exc:
        logger.warning(f"PySR failed: {exc}")
        results["pysr_amplitude"] = {"error": str(exc)}

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
        output_path / "bifurcation_data.npz",
        **{k: v for k, v in bif_data.items()},
    )

    return results
