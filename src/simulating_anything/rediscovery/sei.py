"""SEI epidemic model rediscovery.

Targets:
- R0 = beta / mu (basic reproduction number)
- Population decline rate with mu > 0
- Eventual total infection (S -> 0 when R0 > 1)
- SINDy recovery of SEI ODEs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sei import SEISimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_sei_sweep_data(
    n_samples: int = 200,
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate SEI trajectories with varied beta/sigma/mu to study R0.

    For each parameter set, run to completion and record:
    - Peak exposed fraction
    - Final infected fraction
    - Final susceptible fraction
    - Cumulative deaths (N0 - living)
    - Time to peak exposed
    """
    rng = np.random.default_rng(42)

    all_beta = []
    all_sigma = []
    all_mu = []
    all_R0 = []
    all_peak_E = []
    all_final_I = []
    all_final_S = []
    all_cumulative_deaths = []
    all_time_to_peak_E = []

    N0 = 1000.0

    for i in range(n_samples):
        beta = rng.uniform(0.1, 0.8)
        sigma = rng.uniform(0.05, 0.5)
        mu = rng.uniform(0.01, 0.2)
        S_0 = N0 * rng.uniform(0.95, 1.0)
        E_0 = (N0 - S_0) * 0.5
        I_0 = N0 - S_0 - E_0

        config = SimulationConfig(
            domain=Domain.SEI,  # Placeholder
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta,
                "sigma": sigma,
                "mu": mu,
                "N0": N0,
                "S_0": S_0,
                "E_0": E_0,
                "I_0": I_0,
            },
        )
        sim = SEISimulation(config)
        sim.reset()

        peak_E = 0.0
        peak_E_time = 0
        for step in range(n_steps):
            state = sim.step()
            if state[1] > peak_E:
                peak_E = state[1]
                peak_E_time = step + 1
            # Early stop if epidemic is over (E + I very small)
            if state[1] + state[2] < 1e-3 and step > 200:
                break

        final_state = sim.observe()
        final_S = final_state[0]
        final_I = final_state[2]
        living = final_state.sum()

        all_beta.append(beta)
        all_sigma.append(sigma)
        all_mu.append(mu)
        all_R0.append(beta / mu if mu > 0 else float("inf"))
        all_peak_E.append(peak_E / N0)
        all_final_I.append(final_I / N0)
        all_final_S.append(final_S / N0)
        all_cumulative_deaths.append((N0 - living) / N0)
        all_time_to_peak_E.append(peak_E_time * dt)

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} SEI trajectories")

    return {
        "beta": np.array(all_beta),
        "sigma": np.array(all_sigma),
        "mu": np.array(all_mu),
        "R0": np.array(all_R0),
        "peak_E": np.array(all_peak_E),
        "final_I": np.array(all_final_I),
        "final_S": np.array(all_final_S),
        "cumulative_deaths": np.array(all_cumulative_deaths),
        "time_to_peak_E": np.array(all_time_to_peak_E),
    }


def generate_sei_ode_data(
    n_steps: int = 3000,
    dt: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Generate a single SEI trajectory for SINDy ODE recovery."""
    N0 = 1000.0
    config = SimulationConfig(
        domain=Domain.SEI,  # Placeholder
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": 0.5,
            "sigma": 0.2,
            "mu": 0.05,
            "N0": N0,
            "S_0": 985.0,
            "E_0": 10.0,
            "I_0": 5.0,
        },
    )
    sim = SEISimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "beta": 0.5,
        "sigma": 0.2,
        "mu": 0.05,
        "N0": N0,
    }


def run_sei_rediscovery(
    output_dir: str | Path = "output/rediscovery/sei",
    n_iterations: int = 40,
    n_samples: int = 200,
) -> dict:
    """Run the full SEI epidemic rediscovery.

    1. Sweep beta/sigma/mu parameter space
    2. Run PySR to find R0 = beta/mu from sweep data
    3. Run SINDy to recover SEI ODEs
    4. Compare with known results
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sei",
        "targets": {
            "R0": "beta / mu",
            "ode_S": "dS/dt = -beta * S * I / N",
            "ode_E": "dE/dt = beta * S * I / N - sigma * E",
            "ode_I": "dI/dt = sigma * E - mu * I",
        },
    }

    # --- Part 1: R0 rediscovery via PySR ---
    logger.info("Part 1: Generating SEI parameter sweep data...")
    data = generate_sei_sweep_data(n_samples=n_samples, n_steps=5000, dt=0.1)

    # Filter to epidemics that actually occurred (R0 > 1)
    mask = data["R0"] > 1.0
    X_filtered = np.column_stack([
        data["beta"][mask], data["mu"][mask],
    ])

    logger.info(f"  {mask.sum()}/{len(mask)} epidemics with R0 > 1")

    # PySR: predict R0 directly from beta, mu
    logger.info("  Running PySR for R0 = f(beta, mu)...")
    r0_discoveries = run_symbolic_regression(
        X_filtered,
        data["R0"][mask],
        variable_names=["b_", "m_"],
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

    # PySR: predict cumulative deaths from beta, sigma, mu
    logger.info("  Running PySR for cumulative deaths = f(beta, sigma, mu)...")
    X_deaths = np.column_stack([
        data["beta"][mask], data["sigma"][mask], data["mu"][mask],
    ])
    death_discoveries = run_symbolic_regression(
        X_deaths,
        data["cumulative_deaths"][mask],
        variable_names=["b_", "s_", "m_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log"],
        max_complexity=20,
        populations=20,
        population_size=40,
    )

    results["deaths_pysr"] = {
        "n_discoveries": len(death_discoveries),
        "discoveries": [
            {
                "expression": d.expression,
                "r_squared": d.evidence.fit_r_squared,
            }
            for d in death_discoveries[:5]
        ],
    }
    if death_discoveries:
        best = death_discoveries[0]
        results["deaths_pysr"]["best"] = best.expression
        results["deaths_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best deaths: {best.expression} "
            f"(R2={best.evidence.fit_r_squared:.6f})"
        )

    # --- Part 2: SINDy ODE recovery ---
    logger.info("Part 2: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_sei_ode_data(n_steps=3000, dt=0.1)
        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["S", "E", "I"],
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
            "true_mu": ode_data["mu"],
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
