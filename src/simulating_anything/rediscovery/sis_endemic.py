"""SIS endemic model rediscovery.

Targets:
- R0 = beta / gamma (basic reproduction number)
- Endemic equilibrium I* = N * (1 - 1/R0)
- SINDy recovery of SIS ODEs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sis_endemic import SISEndemicSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_sis_sweep_data(
    n_samples: int = 200,
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate SIS trajectories with varied beta/gamma to study R0 and endemic eq.

    For each parameter set, run to equilibrium and record:
    - Final infected fraction (should approach I* / N)
    - Whether endemic equilibrium was reached
    - Time to reach equilibrium
    """
    rng = np.random.default_rng(42)

    all_beta = []
    all_gamma = []
    all_R0 = []
    all_final_I_frac = []
    all_endemic_eq = []
    all_time_to_eq = []

    N0 = 1000.0

    for i in range(n_samples):
        beta = rng.uniform(0.05, 1.0)
        gamma = rng.uniform(0.05, 0.5)
        I_0 = N0 * rng.uniform(0.005, 0.05)

        config = SimulationConfig(
            domain=Domain.SIS_ENDEMIC,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta,
                "gamma": gamma,
                "N0": N0,
                "I_0": I_0,
            },
        )
        sim = SISEndemicSimulation(config)
        sim.reset()

        # Run to equilibrium
        prev_I = sim.observe()[1]
        eq_time = n_steps * dt
        for step in range(n_steps):
            state = sim.step()
            curr_I = state[1]
            # Check for equilibrium: relative change < threshold
            if step > 100 and abs(curr_I - prev_I) / max(curr_I, 1e-10) < 1e-8:
                eq_time = (step + 1) * dt
                break
            prev_I = curr_I

        final_state = sim.observe()
        final_I_frac = final_state[1] / N0

        r0 = beta / gamma
        # Theoretical endemic equilibrium: I*/N = 1 - 1/R0 when R0 > 1
        if r0 > 1.0:
            endemic_eq = 1.0 - 1.0 / r0
        else:
            endemic_eq = 0.0

        all_beta.append(beta)
        all_gamma.append(gamma)
        all_R0.append(r0)
        all_final_I_frac.append(final_I_frac)
        all_endemic_eq.append(endemic_eq)
        all_time_to_eq.append(eq_time)

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} SIS trajectories")

    return {
        "beta": np.array(all_beta),
        "gamma": np.array(all_gamma),
        "R0": np.array(all_R0),
        "final_I_frac": np.array(all_final_I_frac),
        "endemic_eq": np.array(all_endemic_eq),
        "time_to_eq": np.array(all_time_to_eq),
    }


def generate_sis_ode_data(
    n_steps: int = 3000,
    dt: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Generate a single SIS trajectory for SINDy ODE recovery."""
    N0 = 1000.0
    config = SimulationConfig(
        domain=Domain.SIS_ENDEMIC,  # placeholder
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": 0.5,
            "gamma": 0.2,
            "N0": N0,
            "I_0": 10.0,
        },
    )
    sim = SISEndemicSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "beta": 0.5,
        "gamma": 0.2,
        "N0": N0,
    }


def run_sis_endemic_rediscovery(
    output_dir: str | Path = "output/rediscovery/sis_endemic",
    n_iterations: int = 40,
    n_samples: int = 200,
) -> dict:
    """Run the full SIS endemic model rediscovery.

    1. Sweep beta/gamma parameter space
    2. Run PySR to find R0 = beta/gamma from endemic equilibrium data
    3. Run PySR to find I* = N*(1 - 1/R0) relationship
    4. Run SINDy to recover SIS ODEs
    5. Compare with known results
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sis_endemic",
        "targets": {
            "R0": "beta / gamma",
            "endemic_eq": "I* = N * (1 - 1/R0)",
            "ode_S": "dS/dt = -beta * S * I / N + gamma * I",
            "ode_I": "dI/dt = beta * S * I / N - gamma * I",
        },
    }

    # --- Part 1: R0 rediscovery via PySR ---
    logger.info("Part 1: Generating SIS parameter sweep data...")
    data = generate_sis_sweep_data(n_samples=n_samples, n_steps=5000, dt=0.1)

    # Filter to endemic cases (R0 > 1)
    mask = data["R0"] > 1.0
    X_filtered = np.column_stack([
        data["beta"][mask], data["gamma"][mask],
    ])

    logger.info(f"  {mask.sum()}/{len(mask)} cases with R0 > 1")

    # PySR: predict R0 directly from beta, gamma
    logger.info("  Running PySR for R0 = f(beta, gamma)...")
    r0_discoveries = run_symbolic_regression(
        X_filtered,
        data["R0"][mask],
        variable_names=["b_", "g_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        max_complexity=10,
        populations=20,
        population_size=40,
    )

    results["R0_pysr"] = {
        "n_endemic": int(mask.sum()),
        "n_discoveries": len(r0_discoveries),
        "discoveries": [
            {
                "expression": d.expression,
                "r_squared": d.evidence.fit_r_squared,
            }
            for d in r0_discoveries[:5]
        ],
    }
    if r0_discoveries:
        best = r0_discoveries[0]
        results["R0_pysr"]["best"] = best.expression
        results["R0_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best R0: {best.expression} "
            f"(R2={best.evidence.fit_r_squared:.6f})"
        )

    # PySR: predict endemic equilibrium I*/N from beta, gamma
    logger.info("  Running PySR for I*/N = f(beta, gamma)...")
    eq_discoveries = run_symbolic_regression(
        X_filtered,
        data["final_I_frac"][mask],
        variable_names=["b_", "g_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        max_complexity=15,
        populations=20,
        population_size=40,
    )

    results["endemic_eq_pysr"] = {
        "n_discoveries": len(eq_discoveries),
        "discoveries": [
            {
                "expression": d.expression,
                "r_squared": d.evidence.fit_r_squared,
            }
            for d in eq_discoveries[:5]
        ],
    }
    if eq_discoveries:
        best = eq_discoveries[0]
        results["endemic_eq_pysr"]["best"] = best.expression
        results["endemic_eq_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best I*/N: {best.expression} "
            f"(R2={best.evidence.fit_r_squared:.6f})"
        )

    # --- Part 2: SINDy ODE recovery ---
    logger.info("Part 2: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_sis_ode_data(n_steps=3000, dt=0.1)
        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["S", "I"],
            threshold=0.01,
            poly_degree=2,
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
            "true_beta": ode_data["beta"],
            "true_gamma": ode_data["gamma"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "sweep_data.npz",
        **{k: v for k, v in data.items()},
    )

    return results
