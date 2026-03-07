"""Ivlev predator-prey model rediscovery.

Targets:
- SINDy ODE recovery of coupled equations:
    dN/dt = r*N*(1 - N/K) - a*(1 - exp(-b*N))*P
    dP/dt = e*a*(1 - exp(-b*N))*P - d*P
- PySR recovery of Ivlev functional response shape f(N) = a*(1-exp(-b*N))
- Coexistence equilibrium: N* = -ln(1 - d/(e*a)) / b
- Stability analysis via parameter sweep
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.ivlev_predator_prey import IvlevPredatorPreySimulation as IvlevPredatorPrey
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use DUFFING as placeholder domain enum value
_DOMAIN = Domain.IVLEV_PREDATOR_PREY


def generate_trajectory_data(
    r: float = 1.0,
    K: float = 10.0,
    a: float = 1.5,
    b: float = 0.3,
    e: float = 0.5,
    d: float = 0.3,
    N_0: float = 5.0,
    P_0: float = 2.0,
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
            "r": r, "K": K, "a": a, "b": b, "e": e, "d": d,
            "N_0": N_0, "P_0": P_0,
        },
    )

    sim = IvlevPredatorPrey(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "N": states_arr[:, 0],
        "P": states_arr[:, 1],
        "dt": dt,
        "r": r, "K": K, "a": a, "b": b, "e": e, "d": d,
    }


def generate_functional_response_data(
    n_points: int = 50,
    N_max: float = 20.0,
    a: float = 1.5,
    b: float = 0.3,
) -> dict[str, np.ndarray]:
    """Generate Ivlev functional response data for PySR fitting.

    Evaluates f(N) = a*(1 - exp(-b*N)) over a range of N values.

    Returns dict with N values and corresponding functional response values.
    """
    N_values = np.linspace(0.01, N_max, n_points)
    f_values = a * (1.0 - np.exp(-b * N_values))
    return {
        "N": N_values,
        "f_N": f_values,
        "a": a,
        "b": b,
    }


def generate_stability_sweep_data(
    n_K_values: int = 30,
    K_min: float = 2.0,
    K_max: float = 30.0,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep carrying capacity K to detect limit cycle onset.

    For each K, runs a long trajectory and measures:
    - Time-averaged prey and predator populations
    - Oscillation amplitude (max - min in second half)

    Returns dict with K values, averages, and amplitudes.
    """
    K_values = np.linspace(K_min, K_max, n_K_values)

    all_K = []
    all_N_avg = []
    all_P_avg = []
    all_N_amp = []
    all_P_amp = []

    for i, K_val in enumerate(K_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 1.0, "K": float(K_val), "a": 1.5, "b": 0.3,
                "e": 0.5, "d": 0.3, "N_0": 5.0, "P_0": 2.0,
            },
        )

        sim = IvlevPredatorPrey(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)

        # Use second half to avoid transients
        half = n_steps // 2
        N_half = trajectory[half:, 0]
        P_half = trajectory[half:, 1]

        all_K.append(float(K_val))
        all_N_avg.append(float(np.mean(N_half)))
        all_P_avg.append(float(np.mean(P_half)))
        all_N_amp.append(float(np.max(N_half) - np.min(N_half)))
        all_P_amp.append(float(np.max(P_half) - np.min(P_half)))

        if (i + 1) % 10 == 0:
            logger.info(f"  Stability sweep: {i + 1}/{n_K_values} K values")

    return {
        "K": np.array(all_K),
        "N_avg": np.array(all_N_avg),
        "P_avg": np.array(all_P_avg),
        "N_amplitude": np.array(all_N_amp),
        "P_amplitude": np.array(all_P_amp),
    }


def run_ivlev_predator_prey_rediscovery(
    output_dir: str | Path = "output/rediscovery/ivlev_predator_prey",
    n_iterations: int = 40,
    n_K_values: int = 30,
) -> dict:
    """Run the full Ivlev predator-prey rediscovery.

    1. Generate trajectory for SINDy ODE recovery
    2. Run SINDy to recover coupled ODEs
    3. Generate functional response data for PySR
    4. Run PySR to recover f(N) = a*(1-exp(-b*N))
    5. Sweep K to analyze stability transitions

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "ivlev_predator_prey",
        "targets": {
            "ode_N": "dN/dt = r*N*(1-N/K) - a*(1-exp(-b*N))*P",
            "ode_P": "dP/dt = e*a*(1-exp(-b*N))*P - d*P",
            "functional_response": "f(N) = a*(1-exp(-b*N))",
            "equilibrium_N": "N* = -ln(1 - d/(e*a)) / b",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for Ivlev predator-prey model...")
    data = generate_trajectory_data(
        r=1.0, K=10.0, a=1.5, b=0.3, e=0.5, d=0.3,
        n_steps=10000, dt=0.005,
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["N", "P"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": disc.expression, "r_squared": disc.evidence.fit_r_squared}
                for disc in sindy_discoveries
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

    # --- Part 2: PySR functional response recovery ---
    logger.info("Part 2: PySR recovery of Ivlev functional response...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        fr_data = generate_functional_response_data(n_points=100, N_max=20.0)
        X_fr = fr_data["N"].reshape(-1, 1)
        y_fr = fr_data["f_N"]

        pysr_discoveries = run_symbolic_regression(
            X_fr,
            y_fr,
            variable_names=["N_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "neg"],
            max_complexity=15,
            populations=20,
            population_size=40,
        )

        results["pysr_functional_response"] = {
            "n_discoveries": len(pysr_discoveries),
            "discoveries": [
                {"expression": disc.expression, "r_squared": disc.evidence.fit_r_squared}
                for disc in pysr_discoveries[:5]
            ],
        }
        if pysr_discoveries:
            best = pysr_discoveries[0]
            results["pysr_functional_response"]["best"] = best.expression
            results["pysr_functional_response"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  PySR functional response: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as exc:
        logger.warning(f"PySR failed: {exc}")
        results["pysr_functional_response"] = {"error": str(exc)}

    # --- Part 3: Stability sweep ---
    logger.info("Part 3: Stability analysis via K sweep...")
    sweep_data = generate_stability_sweep_data(
        n_K_values=n_K_values, K_min=2.0, K_max=30.0,
        n_steps=20000, dt=0.01,
    )

    # Detect onset of limit cycles from amplitude threshold
    amp_threshold = 0.1
    K_c_detected = None
    for j, amp in enumerate(sweep_data["N_amplitude"]):
        if amp > amp_threshold:
            K_c_detected = float(sweep_data["K"][j])
            break

    results["stability_sweep"] = {
        "n_K_values": n_K_values,
        "K_range": [float(sweep_data["K"][0]), float(sweep_data["K"][-1])],
        "K_c_detected": K_c_detected,
        "max_N_amplitude": float(np.max(sweep_data["N_amplitude"])),
        "max_P_amplitude": float(np.max(sweep_data["P_amplitude"])),
    }

    logger.info(f"  Detected K_c = {K_c_detected}")
    logger.info(f"  Max prey amplitude = {np.max(sweep_data['N_amplitude']):.4f}")

    # --- Part 4: Equilibrium verification ---
    logger.info("Part 4: Equilibrium verification...")
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=0.01,
        n_steps=500,
        parameters={
            "r": 1.0, "K": 10.0, "a": 1.5, "b": 0.3,
            "e": 0.5, "d": 0.3, "N_0": 5.0, "P_0": 2.0,
        },
    )
    sim = IvlevPredatorPrey(config)

    try:
        N_star, P_star = sim.coexistence_equilibrium()
        results["equilibrium"] = {
            "N_star": N_star,
            "P_star": P_star,
            "N_star_theory": -np.log(1.0 - 0.3 / (0.5 * 1.5)) / 0.3,
        }
        # Verify derivatives are near zero at equilibrium
        derivs = sim._derivatives(np.array([N_star, P_star]))
        results["equilibrium"]["deriv_magnitude"] = float(np.linalg.norm(derivs))
        logger.info(f"  N* = {N_star:.4f}, P* = {P_star:.4f}")
        logger.info(f"  |dN/dt, dP/dt| at eq = {np.linalg.norm(derivs):.2e}")
    except ValueError as exc:
        results["equilibrium"] = {"error": str(exc)}
        logger.warning(f"  Equilibrium not found: {exc}")

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
        output_path / "stability_sweep_data.npz",
        **{k: v for k, v in sweep_data.items()},
    )

    return results
