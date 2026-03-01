"""Rosenzweig-MacArthur predator-prey rediscovery.

Targets:
- ODE recovery via SINDy:
    dx/dt = r*x*(1 - x/K) - a*x*y/(1 + a*h*x)
    dy/dt = e*a*x*y/(1 + a*h*x) - d*y
- Hopf bifurcation: K_c = 2*d / (e*a - a*h*d)
- Paradox of enrichment: limit cycles for K > K_c
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.rosenzweig_macarthur import RosenzweigMacArthur
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use ROSENZWEIG_MACARTHUR domain enum value
_DOMAIN = Domain.ROSENZWEIG_MACARTHUR


def generate_trajectory_data(
    r: float = 1.0,
    K: float = 10.0,
    a: float = 0.5,
    h: float = 0.5,
    e: float = 0.5,
    d: float = 0.1,
    x_0: float = 1.0,
    y_0: float = 1.0,
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
            "r": r, "K": K, "a": a, "h": h, "e": e, "d": d,
            "x_0": x_0, "y_0": y_0,
        },
    )

    sim = RosenzweigMacArthur(config)
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
        "r": r, "K": K, "a": a, "h": h, "e": e, "d": d,
    }


def generate_bifurcation_data(
    n_K_values: int = 30,
    K_min: float = 2.0,
    K_max: float = 20.0,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep carrying capacity K to detect the Hopf bifurcation.

    For each K, runs a long trajectory and measures:
    - Time-averaged prey and predator populations
    - Oscillation amplitude (max - min in second half of trajectory)

    Returns dict with K values, averages, and amplitudes.
    """
    K_values = np.linspace(K_min, K_max, n_K_values)

    all_K = []
    all_x_avg = []
    all_y_avg = []
    all_x_amp = []
    all_y_amp = []

    for i, K_val in enumerate(K_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 1.0, "K": float(K_val), "a": 0.5, "h": 0.5,
                "e": 0.5, "d": 0.1, "x_0": 1.0, "y_0": 1.0,
            },
        )

        sim = RosenzweigMacArthur(config)
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

        all_K.append(float(K_val))
        all_x_avg.append(float(np.mean(x_half)))
        all_y_avg.append(float(np.mean(y_half)))
        all_x_amp.append(float(np.max(x_half) - np.min(x_half)))
        all_y_amp.append(float(np.max(y_half) - np.min(y_half)))

        if (i + 1) % 10 == 0:
            logger.info(f"  Bifurcation sweep: {i + 1}/{n_K_values} K values")

    return {
        "K": np.array(all_K),
        "x_avg": np.array(all_x_avg),
        "y_avg": np.array(all_y_avg),
        "x_amplitude": np.array(all_x_amp),
        "y_amplitude": np.array(all_y_amp),
    }


def run_rosenzweig_macarthur_rediscovery(
    output_dir: str | Path = "output/rediscovery/rosenzweig_macarthur",
    n_iterations: int = 40,
    n_K_values: int = 30,
) -> dict:
    """Run the full Rosenzweig-MacArthur rediscovery.

    1. Generate a trajectory for SINDy ODE recovery
    2. Run SINDy to recover the coupled ODEs
    3. Sweep K to detect Hopf bifurcation (paradox of enrichment)
    4. Compare critical K with theory

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "rosenzweig_macarthur",
        "targets": {
            "ode_x": "dx/dt = r*x*(1-x/K) - a*x*y/(1+a*h*x)",
            "ode_y": "dy/dt = e*a*x*y/(1+a*h*x) - d*y",
            "hopf_bifurcation": "K_c = 2*d/(e*a - a*h*d)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for Rosenzweig-MacArthur model...")
    data = generate_trajectory_data(
        r=1.0, K=10.0, a=0.5, h=0.5, e=0.5, d=0.1,
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
        for disc in sindy_discoveries:
            logger.info(f"  SINDy: {disc.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: Hopf bifurcation detection ---
    logger.info("Part 2: Detecting Hopf bifurcation via K sweep...")
    bif_data = generate_bifurcation_data(
        n_K_values=n_K_values, K_min=2.0, K_max=20.0,
        n_steps=20000, dt=0.01,
    )

    # Theoretical critical K
    # K_c = 2*d / (e*a - a*h*d) with defaults r=1, a=0.5, h=0.5, e=0.5, d=0.1
    ea = 0.5 * 0.5   # e*a = 0.25
    ahd = 0.5 * 0.5 * 0.1  # a*h*d = 0.025
    K_c_theory = 2.0 * 0.1 / (ea - ahd)  # = 0.2 / 0.225 = 0.889

    # Detect K_c from data: find where oscillation amplitude first exceeds threshold
    amp_threshold = 0.1
    K_c_detected = None
    for j, amp in enumerate(bif_data["x_amplitude"]):
        if amp > amp_threshold:
            K_c_detected = float(bif_data["K"][j])
            break

    results["bifurcation"] = {
        "n_K_values": n_K_values,
        "K_range": [float(bif_data["K"][0]), float(bif_data["K"][-1])],
        "K_c_theory": K_c_theory,
        "K_c_detected": K_c_detected,
        "max_x_amplitude": float(np.max(bif_data["x_amplitude"])),
        "max_y_amplitude": float(np.max(bif_data["y_amplitude"])),
    }

    logger.info(f"  Theoretical K_c = {K_c_theory:.4f}")
    logger.info(f"  Detected K_c = {K_c_detected}")
    logger.info(f"  Max prey amplitude = {np.max(bif_data['x_amplitude']):.4f}")

    # --- Part 3: PySR for bifurcation curve ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Fit x_amplitude as function of K
        X_bif = bif_data["K"].reshape(-1, 1)
        y_bif = bif_data["x_amplitude"]

        pysr_discoveries = run_symbolic_regression(
            X_bif,
            y_bif,
            variable_names=["K_"],
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
                {"expression": disc.expression, "r_squared": disc.evidence.fit_r_squared}
                for disc in pysr_discoveries[:5]
            ],
        }
        if pysr_discoveries:
            best = pysr_discoveries[0]
            results["pysr_amplitude"]["best"] = best.expression
            results["pysr_amplitude"]["best_r2"] = best.evidence.fit_r_squared
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
