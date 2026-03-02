"""SIRS epidemic model rediscovery.

Targets:
- R0 = beta / gamma (basic reproduction number)
- Endemic equilibrium I* relationship with parameters
- Waning immunity rate sweep (oscillation analysis)
- SINDy recovery of SIRS ODEs (including xi*R waning term)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sirs import SIRSSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_sirs_sweep_data(
    n_samples: int = 200,
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate SIRS trajectories with varied beta/gamma/xi to study R0 and endemic state.

    For each parameter set, run the simulation and record:
    - Peak infected count
    - Final infected count (endemic level)
    - Final susceptible count
    - Whether oscillations occur
    - Time to peak
    """
    rng = np.random.default_rng(42)

    all_beta = []
    all_gamma = []
    all_xi = []
    all_R0 = []
    all_peak_I = []
    all_final_I = []
    all_final_S = []
    all_time_to_peak = []

    N0 = 1000.0

    for i in range(n_samples):
        beta = rng.uniform(0.1, 0.8)
        gamma = rng.uniform(0.02, 0.3)
        xi = rng.uniform(0.005, 0.2)
        I_0 = N0 * rng.uniform(0.005, 0.05)

        config = SimulationConfig(
            domain=Domain.SIRS,  # Placeholder domain
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta,
                "gamma": gamma,
                "xi": xi,
                "N0": N0,
                "I_0": I_0,
                "R_0_init": 0.0,
            },
        )
        sim = SIRSSimulation(config)
        sim.reset()

        peak_I = 0.0
        peak_time = 0
        for step in range(n_steps):
            state = sim.step()
            if state[1] > peak_I:
                peak_I = state[1]
                peak_time = step + 1

        final_state = sim.observe()
        all_beta.append(beta)
        all_gamma.append(gamma)
        all_xi.append(xi)
        all_R0.append(beta / gamma)
        all_peak_I.append(peak_I / N0)
        all_final_I.append(final_state[1] / N0)
        all_final_S.append(final_state[0] / N0)
        all_time_to_peak.append(peak_time * dt)

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} SIRS trajectories")

    return {
        "beta": np.array(all_beta),
        "gamma": np.array(all_gamma),
        "xi": np.array(all_xi),
        "R0": np.array(all_R0),
        "peak_I": np.array(all_peak_I),
        "final_I": np.array(all_final_I),
        "final_S": np.array(all_final_S),
        "time_to_peak": np.array(all_time_to_peak),
    }


def generate_sirs_ode_data(
    n_steps: int = 3000,
    dt: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Generate a single SIRS trajectory for SINDy ODE recovery."""
    N0 = 1000.0
    config = SimulationConfig(
        domain=Domain.SIRS,  # Placeholder domain
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": 0.5,
            "gamma": 0.15,
            "xi": 0.05,
            "N0": N0,
            "I_0": 10.0,
            "R_0_init": 0.0,
        },
    )
    sim = SIRSSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "beta": 0.5,
        "gamma": 0.15,
        "xi": 0.05,
        "N0": N0,
    }


def generate_sirs_waning_sweep(
    n_xi_values: int = 30,
    n_steps: int = 10000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep waning rate xi to study transition between SIR and SIS limits.

    At xi=0 (SIR): epidemic ends, no endemic state.
    At large xi (SIS-like): endemic equilibrium with higher I*.
    """
    xi_values = np.logspace(-3, 0, n_xi_values)
    N0 = 1000.0
    beta = 0.5
    gamma = 0.15

    all_xi = []
    all_final_I = []
    all_final_S = []
    all_max_I = []
    all_oscillation_count = []

    for xi in xi_values:
        config = SimulationConfig(
            domain=Domain.SIRS,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta,
                "gamma": gamma,
                "xi": xi,
                "N0": N0,
                "I_0": 10.0,
                "R_0_init": 0.0,
            },
        )
        sim = SIRSSimulation(config)
        sim.reset()

        I_trace = []
        for _ in range(n_steps):
            state = sim.step()
            I_trace.append(state[1])

        I_trace = np.array(I_trace)
        final_state = sim.observe()

        # Count oscillation peaks in the last half of the trajectory
        half = len(I_trace) // 2
        I_late = I_trace[half:]
        peaks = 0
        for j in range(1, len(I_late) - 1):
            if I_late[j] > I_late[j - 1] and I_late[j] > I_late[j + 1]:
                peaks += 1

        all_xi.append(xi)
        all_final_I.append(final_state[1] / N0)
        all_final_S.append(final_state[0] / N0)
        all_max_I.append(np.max(I_trace) / N0)
        all_oscillation_count.append(peaks)

    return {
        "xi": np.array(all_xi),
        "final_I": np.array(all_final_I),
        "final_S": np.array(all_final_S),
        "max_I": np.array(all_max_I),
        "oscillation_count": np.array(all_oscillation_count),
        "beta": beta,
        "gamma": gamma,
    }


def run_sirs_rediscovery(
    output_dir: str | Path = "output/rediscovery/sirs",
    n_iterations: int = 40,
    n_samples: int = 200,
) -> dict:
    """Run the full SIRS epidemic rediscovery.

    1. Sweep beta/gamma/xi parameter space
    2. Run PySR to find R0 = beta/gamma
    3. Run PySR to find endemic equilibrium relationship
    4. Run SINDy to recover SIRS ODEs
    5. Sweep waning rate xi
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sirs",
        "targets": {
            "R0": "beta / gamma",
            "endemic_I": "N * (1 - 1/R0) * xi / (xi + gamma)",
            "ode_S": "dS/dt = -beta * S * I / N + xi * R",
            "ode_I": "dI/dt = beta * S * I / N - gamma * I",
            "ode_R": "dR/dt = gamma * I - xi * R",
        },
    }

    # --- Part 1: R0 rediscovery via PySR ---
    logger.info("Part 1: Generating SIRS parameter sweep data...")
    data = generate_sirs_sweep_data(n_samples=n_samples, n_steps=5000, dt=0.1)

    # Filter to epidemics that actually occurred (R0 > 1)
    mask = data["R0"] > 1.0
    X_filtered = np.column_stack([
        data["beta"][mask], data["gamma"][mask],
    ])

    logger.info(f"  {mask.sum()}/{len(mask)} epidemics with R0 > 1")

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

    # PySR: predict endemic I* from beta, gamma, xi
    logger.info("  Running PySR for endemic I* = f(beta, gamma, xi)...")
    X_endemic = np.column_stack([
        data["beta"][mask], data["gamma"][mask], data["xi"][mask],
    ])
    endemic_discoveries = run_symbolic_regression(
        X_endemic,
        data["final_I"][mask],
        variable_names=["b_", "g_", "x_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        max_complexity=20,
        populations=20,
        population_size=40,
    )

    results["endemic_I_pysr"] = {
        "n_discoveries": len(endemic_discoveries),
        "discoveries": [
            {
                "expression": d.expression,
                "r_squared": d.evidence.fit_r_squared,
            }
            for d in endemic_discoveries[:5]
        ],
    }
    if endemic_discoveries:
        best = endemic_discoveries[0]
        results["endemic_I_pysr"]["best"] = best.expression
        results["endemic_I_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best endemic I*: {best.expression} "
            f"(R2={best.evidence.fit_r_squared:.6f})"
        )

    # --- Part 2: SINDy ODE recovery ---
    logger.info("Part 2: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_sirs_ode_data(n_steps=3000, dt=0.1)
        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["S", "I", "R"],
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
            "true_xi": ode_data["xi"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 3: Waning rate sweep ---
    logger.info("Part 3: Waning immunity rate sweep...")
    waning_data = generate_sirs_waning_sweep(n_xi_values=30, n_steps=10000, dt=0.1)
    results["waning_sweep"] = {
        "n_xi_values": len(waning_data["xi"]),
        "xi_range": [float(waning_data["xi"].min()), float(waning_data["xi"].max())],
        "final_I_range": [
            float(waning_data["final_I"].min()),
            float(waning_data["final_I"].max()),
        ],
        "beta": waning_data["beta"],
        "gamma": waning_data["gamma"],
    }

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
