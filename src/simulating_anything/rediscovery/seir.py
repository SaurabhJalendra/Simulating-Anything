"""SEIR epidemic model rediscovery.

Targets:
- R0 = beta / gamma (basic reproduction number)
- Latent period effect on epidemic timing
- SINDy recovery of SEIR ODEs
- Population conservation S + E + I + R = N
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.seir import SEIRSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_seir_sweep_data(
    n_samples: int = 200,
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate SEIR trajectories with varied beta/gamma to study R0.

    For each parameter set, run to completion and record:
    - Peak infectious fraction
    - Final epidemic size (total fraction recovered)
    - Time to peak infectious
    - Peak exposed fraction
    """
    rng = np.random.default_rng(42)

    all_beta = []
    all_sigma = []
    all_gamma = []
    all_R0 = []
    all_peak_I = []
    all_peak_E = []
    all_final_size = []
    all_time_to_peak = []

    N = 1000.0

    for i in range(n_samples):
        beta = rng.uniform(0.1, 0.8)
        sigma = rng.uniform(0.05, 0.5)
        gamma = rng.uniform(0.02, 0.3)
        S_0 = N * rng.uniform(0.95, 1.0)
        E_0 = 0.0
        I_0 = N - S_0
        R_0_init = 0.0

        config = SimulationConfig(
            domain=Domain.SEIR,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta,
                "sigma": sigma,
                "gamma": gamma,
                "N": N,
                "S_0": S_0,
                "E_0": E_0,
                "I_0": I_0,
                "R_0_init": R_0_init,
            },
        )
        sim = SEIRSimulation(config)
        sim.reset()

        peak_I = 0.0
        peak_E = 0.0
        peak_time = 0
        for step in range(n_steps):
            state = sim.step()
            if state[2] > peak_I:
                peak_I = state[2]
                peak_time = step + 1
            if state[1] > peak_E:
                peak_E = state[1]
            # Early stop if epidemic is over
            if state[1] + state[2] < 1e-3 and step > 200:
                break

        final_R = sim.observe()[3]  # Final recovered

        all_beta.append(beta)
        all_sigma.append(sigma)
        all_gamma.append(gamma)
        all_R0.append(beta / gamma)
        all_peak_I.append(peak_I / N)
        all_peak_E.append(peak_E / N)
        all_final_size.append(final_R / N)
        all_time_to_peak.append(peak_time * dt)

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} SEIR trajectories")

    return {
        "beta": np.array(all_beta),
        "sigma": np.array(all_sigma),
        "gamma": np.array(all_gamma),
        "R0": np.array(all_R0),
        "peak_I": np.array(all_peak_I),
        "peak_E": np.array(all_peak_E),
        "final_size": np.array(all_final_size),
        "time_to_peak": np.array(all_time_to_peak),
    }


def generate_seir_ode_data(
    n_steps: int = 3000,
    dt: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Generate a single SEIR trajectory for SINDy ODE recovery."""
    N = 1000.0
    config = SimulationConfig(
        domain=Domain.SEIR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": 0.5,
            "sigma": 0.2,
            "gamma": 0.1,
            "N": N,
            "S_0": 990.0,
            "E_0": 5.0,
            "I_0": 5.0,
            "R_0_init": 0.0,
        },
    )
    sim = SEIRSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "beta": 0.5,
        "sigma": 0.2,
        "gamma": 0.1,
        "N": N,
    }


def run_seir_rediscovery(
    output_dir: str | Path = "output/rediscovery/seir",
    n_iterations: int = 40,
    n_samples: int = 200,
) -> dict:
    """Run the full SEIR epidemic rediscovery.

    1. Sweep beta/gamma/sigma parameter space
    2. Run PySR to find R0 = beta/gamma from peak_I and final_size data
    3. Run SINDy to recover SEIR ODEs
    4. Compare with known results
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "seir",
        "targets": {
            "R0": "beta / gamma",
            "ode_S": "dS/dt = -beta * S * I / N",
            "ode_E": "dE/dt = beta * S * I / N - sigma * E",
            "ode_I": "dI/dt = sigma * E - gamma * I",
            "ode_R": "dR/dt = gamma * I",
        },
    }

    # --- Part 1: R0 rediscovery via PySR ---
    logger.info("Part 1: Generating SEIR parameter sweep data...")
    data = generate_seir_sweep_data(n_samples=n_samples, n_steps=5000, dt=0.1)

    # PySR: predict R0 directly from beta, gamma
    # Filter to epidemics that actually occurred (R0 > 1)
    mask = data["R0"] > 1.0
    X_filtered = np.column_stack([data["beta"][mask], data["gamma"][mask]])

    logger.info(f"  {mask.sum()}/{len(mask)} epidemics with R0 > 1")
    logger.info("  Running PySR for R0 = f(beta, gamma)...")

    # Use b_ and g_ to avoid sympy conflicts
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
        "n_epidemics": int(mask.sum()),
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

    # PySR: predict final_size from beta, gamma
    logger.info("  Running PySR for final size = f(beta, gamma)...")
    X_fs = np.column_stack([data["beta"][mask], data["gamma"][mask]])
    y_fs = data["final_size"][mask]

    fs_discoveries = run_symbolic_regression(
        X_fs,
        y_fs,
        variable_names=["b_", "g_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log"],
        max_complexity=20,
        populations=20,
        population_size=40,
    )

    results["final_size_pysr"] = {
        "n_discoveries": len(fs_discoveries),
        "discoveries": [
            {
                "expression": d.expression,
                "r_squared": d.evidence.fit_r_squared,
            }
            for d in fs_discoveries[:5]
        ],
    }
    if fs_discoveries:
        best = fs_discoveries[0]
        results["final_size_pysr"]["best"] = best.expression
        results["final_size_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best final size: {best.expression} "
            f"(R2={best.evidence.fit_r_squared:.6f})"
        )

    # --- Part 2: SINDy ODE recovery ---
    logger.info("Part 2: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_seir_ode_data(n_steps=3000, dt=0.1)
        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["S", "E", "I", "R"],
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
            "true_sigma": ode_data["sigma"],
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
