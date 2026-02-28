"""Lotka-Volterra equilibrium and ODE rediscovery.

Targets:
- Equilibrium: prey_eq = gamma/delta, pred_eq = alpha/beta
- ODE: d(prey)/dt = alpha*prey - beta*prey*pred
        d(pred)/dt = delta*prey*pred - gamma*pred
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_equilibrium_data(
    n_samples: int = 200,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate LV trajectories with varied parameters, compute time-averaged populations.

    Varies alpha, beta, gamma, delta independently.
    Returns dict with parameter values and time-averaged prey/predator populations.
    """
    rng = np.random.default_rng(42)

    all_alpha = []
    all_beta = []
    all_gamma = []
    all_delta = []
    all_prey_avg = []
    all_pred_avg = []

    for i in range(n_samples):
        alpha = rng.uniform(0.5, 2.0)
        beta = rng.uniform(0.2, 0.8)
        gamma = rng.uniform(0.2, 0.8)
        delta = rng.uniform(0.05, 0.3)

        # Start near equilibrium to reduce transient effects
        prey_eq = gamma / delta
        pred_eq = alpha / beta
        prey_0 = prey_eq * rng.uniform(0.8, 1.2)
        pred_0 = pred_eq * rng.uniform(0.8, 1.2)

        config = SimulationConfig(
            domain=Domain.AGENT_BASED,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "prey_0": prey_0,
                "predator_0": pred_0,
            },
        )

        sim = LotkaVolterraSimulation(config)
        sim.reset()

        # Run simulation and collect states
        states = [sim.observe().copy()]
        for _ in range(n_steps):
            states.append(sim.step().copy())

        trajectory = np.array(states)
        # Skip initial transient (first 20%)
        skip = n_steps // 5
        prey_avg = np.mean(trajectory[skip:, 0])
        pred_avg = np.mean(trajectory[skip:, 1])

        all_alpha.append(alpha)
        all_beta.append(beta)
        all_gamma.append(gamma)
        all_delta.append(delta)
        all_prey_avg.append(prey_avg)
        all_pred_avg.append(pred_avg)

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} trajectories")

    return {
        "alpha": np.array(all_alpha),
        "beta": np.array(all_beta),
        "gamma": np.array(all_gamma),
        "delta": np.array(all_delta),
        "prey_avg": np.array(all_prey_avg),
        "pred_avg": np.array(all_pred_avg),
    }


def generate_ode_data(
    n_steps: int = 2000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate a single long LV trajectory for SINDy ODE recovery.

    Returns dict with states array and parameters.
    """
    config = SimulationConfig(
        domain=Domain.AGENT_BASED,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha": 1.1,
            "beta": 0.4,
            "gamma": 0.4,
            "delta": 0.1,
            "prey_0": 40.0,
            "predator_0": 9.0,
        },
    )

    sim = LotkaVolterraSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "alpha": 1.1,
        "beta": 0.4,
        "gamma": 0.4,
        "delta": 0.1,
    }


def run_lotka_volterra_rediscovery(
    output_dir: str | Path = "output/rediscovery/lotka_volterra",
    n_iterations: int = 40,
    n_equilibrium_samples: int = 200,
) -> dict:
    """Run the full Lotka-Volterra rediscovery.

    1. Generate trajectories with varied parameters
    2. Run PySR on time-averaged populations to find equilibrium expressions
    3. Run SINDy on a single trajectory to recover the ODEs
    4. Compare with known results

    Returns dict with all results.
    """
    from simulating_anything.analysis.equation_discovery import run_sindy
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "lotka_volterra",
        "targets": {
            "equilibrium_prey": "gamma / delta",
            "equilibrium_predator": "alpha / beta",
            "ode_prey": "d(prey)/dt = alpha*prey - beta*prey*pred",
            "ode_pred": "d(pred)/dt = delta*prey*pred - gamma*pred",
        },
    }

    # --- Part 1: Equilibrium rediscovery via PySR ---
    logger.info("Part 1: Generating equilibrium data...")
    eq_data = generate_equilibrium_data(
        n_samples=n_equilibrium_samples, n_steps=10000, dt=0.01
    )

    # Target: prey_avg ~ gamma/delta
    X_eq = np.column_stack([
        eq_data["alpha"], eq_data["beta"], eq_data["gamma"], eq_data["delta"]
    ])

    # Theoretical equilibria
    prey_theory = eq_data["gamma"] / eq_data["delta"]
    pred_theory = eq_data["alpha"] / eq_data["beta"]

    prey_err = np.abs(eq_data["prey_avg"] - prey_theory) / prey_theory
    pred_err = np.abs(eq_data["pred_avg"] - pred_theory) / pred_theory

    results["equilibrium_data"] = {
        "n_samples": len(eq_data["alpha"]),
        "prey_avg_vs_theory_mean_error": float(np.mean(prey_err)),
        "pred_avg_vs_theory_mean_error": float(np.mean(pred_err)),
    }

    logger.info(f"  Prey avg vs gamma/delta: mean error = {np.mean(prey_err):.2%}")
    logger.info(f"  Pred avg vs alpha/beta: mean error = {np.mean(pred_err):.2%}")

    # PySR for prey equilibrium
    # Note: PySR rejects 'alpha', 'beta', 'gamma', 'delta' as they conflict
    # with sympy function names. Use a_, b_, g_, d_ instead.
    pysr_var_names = ["a_", "b_", "g_", "d_"]
    logger.info("Running PySR for prey equilibrium (target: gamma/delta)...")
    prey_discoveries = run_symbolic_regression(
        X_eq,
        eq_data["prey_avg"],
        variable_names=pysr_var_names,
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        max_complexity=15,
        populations=20,
        population_size=40,
    )

    results["prey_equilibrium"] = {
        "n_discoveries": len(prey_discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in prey_discoveries[:5]
        ],
    }
    if prey_discoveries:
        best = prey_discoveries[0]
        results["prey_equilibrium"]["best"] = best.expression
        results["prey_equilibrium"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(f"  Best prey eq: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")

    # PySR for predator equilibrium
    logger.info("Running PySR for predator equilibrium (target: alpha/beta)...")
    pred_discoveries = run_symbolic_regression(
        X_eq,
        eq_data["pred_avg"],
        variable_names=pysr_var_names,
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        max_complexity=15,
        populations=20,
        population_size=40,
    )

    results["pred_equilibrium"] = {
        "n_discoveries": len(pred_discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in pred_discoveries[:5]
        ],
    }
    if pred_discoveries:
        best = pred_discoveries[0]
        results["pred_equilibrium"]["best"] = best.expression
        results["pred_equilibrium"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(f"  Best pred eq: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")

    # --- Part 2: ODE recovery via SINDy ---
    logger.info("Part 2: Generating ODE data for SINDy...")
    ode_data = generate_ode_data(n_steps=2000, dt=0.01)

    logger.info("Running SINDy to recover Lotka-Volterra ODEs...")
    sindy_discoveries = run_sindy(
        ode_data["states"],
        dt=ode_data["dt"],
        feature_names=["prey", "pred"],
        threshold=0.05,
        poly_degree=2,
    )

    results["sindy_ode"] = {
        "n_discoveries": len(sindy_discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in sindy_discoveries
        ],
        "true_alpha": ode_data["alpha"],
        "true_beta": ode_data["beta"],
        "true_gamma": ode_data["gamma"],
        "true_delta": ode_data["delta"],
    }
    for d in sindy_discoveries:
        logger.info(f"  SINDy: {d.expression}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "equilibrium_data.npz",
        **{k: v for k, v in eq_data.items()},
    )
    np.savez(
        output_path / "ode_data.npz",
        states=ode_data["states"],
    )

    return results
