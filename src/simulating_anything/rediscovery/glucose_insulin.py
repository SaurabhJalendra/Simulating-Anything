"""Glucose-Insulin Bergman minimal model rediscovery.

Targets:
- ODE: dG/dt = -(p1+X)*G + p1*Gb, dX/dt = -p2*X + p3*(I-Ib),
        dI/dt = -n*(I-Ib) + gamma*max(G-h,0)  (via SINDy)
- Insulin sensitivity SI = p3/p2
- Glucose effectiveness SG = p1
- Glucose decay profile: time-to-half vs parameter variations
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.glucose_insulin import GlucoseInsulinSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    p1: float = 0.028,
    p2: float = 0.025,
    p3: float = 5e-6,
    n: float = 0.23,
    gamma: float = 0.01,
    n_steps: int = 10000,
    dt: float = 0.5,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.GLUCOSE_INSULIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "p1": p1, "p2": p2, "p3": p3,
            "n": n, "gamma": gamma,
            "G0": 300.0, "X0": 0.0, "I0": 11.0,
        },
    )
    sim = GlucoseInsulinSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "G": states_arr[:, 0],
        "X": states_arr[:, 1],
        "I": states_arr[:, 2],
        "p1": p1,
        "p2": p2,
        "p3": p3,
    }


def generate_sensitivity_data(
    n_p1: int = 20,
    n_p3: int = 20,
    dt: float = 0.5,
) -> dict[str, np.ndarray]:
    """Sweep p1 and p3, measure insulin sensitivity SI = p3/p2 and SG = p1."""
    p1_values = np.linspace(0.01, 0.06, n_p1)
    p3_values = np.linspace(1e-6, 1e-5, n_p3)
    p2 = 0.025

    si_measured = []
    sg_measured = []
    half_times = []

    for p1 in p1_values:
        for p3 in p3_values:
            config = SimulationConfig(
                domain=Domain.GLUCOSE_INSULIN,
                dt=dt,
                n_steps=1200,
                parameters={"p1": p1, "p3": p3, "G0": 300.0},
            )
            sim = GlucoseInsulinSimulation(config)
            si_measured.append(sim.insulin_sensitivity)
            sg_measured.append(sim.glucose_effectiveness)
            t_half = sim.time_to_half_peak(max_steps=1200)
            half_times.append(t_half)

    return {
        "p1": np.repeat(p1_values, n_p3),
        "p3": np.tile(p3_values, n_p1),
        "p2": p2,
        "SI": np.array(si_measured),
        "SG": np.array(sg_measured),
        "half_time": np.array(half_times),
        "SI_theory": np.tile(p3_values, n_p1) / p2,
        "SG_theory": np.repeat(p1_values, n_p3),
    }


def generate_decay_data(
    n_p1: int = 15,
    dt: float = 0.5,
) -> dict[str, np.ndarray]:
    """Sweep p1 and measure glucose decay half-time."""
    p1_values = np.linspace(0.01, 0.08, n_p1)
    half_times = []

    for p1 in p1_values:
        config = SimulationConfig(
            domain=Domain.GLUCOSE_INSULIN,
            dt=dt,
            n_steps=2000,
            parameters={"p1": p1, "G0": 300.0},
        )
        sim = GlucoseInsulinSimulation(config)
        t_half = sim.time_to_half_peak(max_steps=2000)
        half_times.append(t_half)

    return {
        "p1": p1_values,
        "half_time": np.array(half_times),
    }


def run_glucose_insulin_rediscovery(
    output_dir: str | Path = "output/rediscovery/glucose_insulin",
    n_iterations: int = 40,
) -> dict:
    """Run Bergman minimal model rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "glucose_insulin",
        "targets": {
            "ode": (
                "dG/dt = -(p1+X)*G + p1*Gb, "
                "dX/dt = -p2*X + p3*(I-Ib), "
                "dI/dt = -n*(I-Ib) + gamma*max(G-h,0)"
            ),
            "SI": "p3/p2 (insulin sensitivity)",
            "SG": "p1 (glucose effectiveness)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery...")
    data = generate_ode_data(n_steps=10000, dt=0.5)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.5,
            feature_names=["G", "X", "I"],
            threshold=0.001,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries[:5]
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
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Insulin sensitivity verification ---
    logger.info("Part 2: Insulin sensitivity SI = p3/p2...")
    sens_data = generate_sensitivity_data(n_p1=10, n_p3=10, dt=0.5)

    si_error = np.abs(sens_data["SI"] - sens_data["SI_theory"])
    sg_error = np.abs(sens_data["SG"] - sens_data["SG_theory"])

    results["insulin_sensitivity"] = {
        "SI_mean_error": float(np.mean(si_error)),
        "SI_max_error": float(np.max(si_error)),
        "SI_range": [float(np.min(sens_data["SI"])), float(np.max(sens_data["SI"]))],
        "SI_exact_match": bool(np.allclose(sens_data["SI"], sens_data["SI_theory"])),
    }
    results["glucose_effectiveness"] = {
        "SG_mean_error": float(np.mean(sg_error)),
        "SG_max_error": float(np.max(sg_error)),
        "SG_range": [float(np.min(sens_data["SG"])), float(np.max(sens_data["SG"]))],
        "SG_exact_match": bool(np.allclose(sens_data["SG"], sens_data["SG_theory"])),
    }
    logger.info(
        f"  SI exact: {results['insulin_sensitivity']['SI_exact_match']}, "
        f"SG exact: {results['glucose_effectiveness']['SG_exact_match']}"
    )

    # --- Part 3: PySR decay half-time ---
    logger.info("Part 3: PySR decay half-time vs p1...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        decay_data = generate_decay_data(n_p1=20, dt=0.5)
        valid = np.isfinite(decay_data["half_time"])
        if np.sum(valid) > 5:
            X = decay_data["p1"][valid].reshape(-1, 1)
            y = decay_data["half_time"][valid]

            logger.info("  Running PySR: t_half = f(p1)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["p1_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "log", "inv(x) = 1/x"],
                max_complexity=10,
                populations=20,
                population_size=40,
            )
            results["half_time_pysr"] = {
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
                results["half_time_pysr"]["best"] = best.expression
                results["half_time_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["half_time_pysr"] = {"error": str(e)}

    # --- Part 4: Basal state convergence ---
    logger.info("Part 4: Basal state convergence verification...")
    config = SimulationConfig(
        domain=Domain.GLUCOSE_INSULIN,
        dt=0.5,
        n_steps=2000,
        parameters={"G0": 300.0},
    )
    sim = GlucoseInsulinSimulation(config)
    error = sim.steady_state_error(n_steps=2000)
    results["basal_convergence"] = {
        "final_G_error": float(abs(error[0])),
        "final_X_error": float(abs(error[1])),
        "final_I_error": float(abs(error[2])),
        "converged": bool(np.linalg.norm(error) < 1.0),
    }
    logger.info(
        f"  Final error: |dG|={abs(error[0]):.4f}, "
        f"|dX|={abs(error[1]):.6f}, |dI|={abs(error[2]):.4f}"
    )

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
