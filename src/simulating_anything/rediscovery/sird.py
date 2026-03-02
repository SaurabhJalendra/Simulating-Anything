"""SIRD epidemic model rediscovery.

Targets:
- R0 = beta / (gamma + mu) (basic reproduction number)
- CFR = mu / (gamma + mu) (case fatality rate)
- Final death toll proportional to CFR * total infected
- SINDy recovery of SIRD ODEs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sird import SIRDSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_sird_sweep_data(
    n_samples: int = 200,
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate SIRD trajectories with varied beta/gamma/mu to study R0 and CFR.

    For each parameter set, run to completion and record:
    - Peak infected fraction
    - Final epidemic size (total fraction that left S)
    - Final death toll (D at end)
    - Time to peak infected
    """
    rng = np.random.default_rng(42)

    all_beta = []
    all_gamma = []
    all_mu = []
    all_R0 = []
    all_CFR = []
    all_peak_I = []
    all_final_size = []
    all_final_D = []
    all_time_to_peak = []

    N0 = 1000.0

    for i in range(n_samples):
        beta = rng.uniform(0.1, 0.8)
        gamma = rng.uniform(0.02, 0.3)
        mu = rng.uniform(0.005, 0.1)
        S_0 = N0 * rng.uniform(0.95, 1.0)
        I_0 = N0 - S_0

        config = SimulationConfig(
            domain=Domain.SIRD,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta,
                "gamma": gamma,
                "mu": mu,
                "N0": N0,
                "S_0": S_0,
                "I_0": I_0,
                "R_0_init": 0.0,
                "D_0": 0.0,
            },
        )
        sim = SIRDSimulation(config)
        sim.reset()

        peak_I = 0.0
        peak_time = 0
        for step in range(n_steps):
            state = sim.step()
            if state[1] > peak_I:
                peak_I = state[1]
                peak_time = step + 1
            # Early stop if epidemic is over
            if state[1] < 1e-3 and step > 200:
                break

        final_state = sim.observe()
        final_S = final_state[0]
        final_D = final_state[3]

        all_beta.append(beta)
        all_gamma.append(gamma)
        all_mu.append(mu)
        all_R0.append(beta / (gamma + mu))
        all_CFR.append(mu / (gamma + mu))
        all_peak_I.append(peak_I / N0)
        all_final_size.append((N0 - final_S - final_D) / N0)  # fraction that recovered
        all_final_D.append(final_D / N0)
        all_time_to_peak.append(peak_time * dt)

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} SIRD trajectories")

    return {
        "beta": np.array(all_beta),
        "gamma": np.array(all_gamma),
        "mu": np.array(all_mu),
        "R0": np.array(all_R0),
        "CFR": np.array(all_CFR),
        "peak_I": np.array(all_peak_I),
        "final_size": np.array(all_final_size),
        "final_D": np.array(all_final_D),
        "time_to_peak": np.array(all_time_to_peak),
    }


def generate_sird_ode_data(
    n_steps: int = 3000,
    dt: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Generate a single SIRD trajectory for SINDy ODE recovery."""
    N0 = 1000.0
    config = SimulationConfig(
        domain=Domain.SIRD,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": 0.4,
            "gamma": 0.1,
            "mu": 0.02,
            "N0": N0,
            "S_0": 990.0,
            "I_0": 10.0,
            "R_0_init": 0.0,
            "D_0": 0.0,
        },
    )
    sim = SIRDSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "beta": 0.4,
        "gamma": 0.1,
        "mu": 0.02,
        "N0": N0,
    }


def run_sird_rediscovery(
    output_dir: str | Path = "output/rediscovery/sird",
    n_iterations: int = 40,
    n_samples: int = 200,
) -> dict:
    """Run the full SIRD epidemic rediscovery.

    1. Sweep beta/gamma/mu parameter space
    2. Run PySR to find R0 = beta/(gamma+mu) and CFR = mu/(gamma+mu)
    3. Run SINDy to recover SIRD ODEs
    4. Compare with known results
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sird",
        "targets": {
            "R0": "beta / (gamma + mu)",
            "CFR": "mu / (gamma + mu)",
            "ode_S": "dS/dt = -beta * S * I / N",
            "ode_I": "dI/dt = beta * S * I / N - gamma * I - mu * I",
            "ode_R": "dR/dt = gamma * I",
            "ode_D": "dD/dt = mu * I",
        },
    }

    # --- Part 1: R0 rediscovery via PySR ---
    logger.info("Part 1: Generating SIRD parameter sweep data...")
    data = generate_sird_sweep_data(n_samples=n_samples, n_steps=5000, dt=0.1)

    # Filter to epidemics that actually occurred (R0 > 1)
    mask = data["R0"] > 1.0
    X_filtered = np.column_stack([
        data["beta"][mask], data["gamma"][mask], data["mu"][mask],
    ])

    logger.info(f"  {mask.sum()}/{len(mask)} epidemics with R0 > 1")

    # PySR: predict R0 directly from beta, gamma, mu
    logger.info("  Running PySR for R0 = f(beta, gamma, mu)...")
    r0_discoveries = run_symbolic_regression(
        X_filtered,
        data["R0"][mask],
        variable_names=["b_", "g_", "m_"],
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

    # PySR: predict CFR from gamma, mu
    logger.info("  Running PySR for CFR = f(gamma, mu)...")
    X_cfr = np.column_stack([data["gamma"][mask], data["mu"][mask]])
    cfr_discoveries = run_symbolic_regression(
        X_cfr,
        data["CFR"][mask],
        variable_names=["g_", "m_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        max_complexity=10,
        populations=20,
        population_size=40,
    )

    results["CFR_pysr"] = {
        "n_discoveries": len(cfr_discoveries),
        "discoveries": [
            {
                "expression": d.expression,
                "r_squared": d.evidence.fit_r_squared,
            }
            for d in cfr_discoveries[:5]
        ],
    }
    if cfr_discoveries:
        best = cfr_discoveries[0]
        results["CFR_pysr"]["best"] = best.expression
        results["CFR_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best CFR: {best.expression} "
            f"(R2={best.evidence.fit_r_squared:.6f})"
        )

    # PySR: predict final death toll from beta, gamma, mu
    logger.info("  Running PySR for final death toll = f(beta, gamma, mu)...")
    death_discoveries = run_symbolic_regression(
        X_filtered,
        data["final_D"][mask],
        variable_names=["b_", "g_", "m_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log"],
        max_complexity=20,
        populations=20,
        population_size=40,
    )

    results["death_toll_pysr"] = {
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
        results["death_toll_pysr"]["best"] = best.expression
        results["death_toll_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best death toll: {best.expression} "
            f"(R2={best.evidence.fit_r_squared:.6f})"
        )

    # --- Part 2: SINDy ODE recovery ---
    logger.info("Part 2: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_sird_ode_data(n_steps=3000, dt=0.1)
        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["S", "I", "R", "D"],
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
