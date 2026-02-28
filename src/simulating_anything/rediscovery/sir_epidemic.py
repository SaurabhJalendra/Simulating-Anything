"""SIR epidemic model rediscovery.

Targets:
- R0 = beta / gamma (basic reproduction number)
- Peak infected fraction relationship
- Final epidemic size
- SINDy recovery of SIR ODEs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.epidemiological import SIRSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_sir_sweep_data(
    n_samples: int = 200,
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate SIR trajectories with varied beta/gamma to study R0.

    For each parameter set, run to completion and record:
    - Peak infected fraction
    - Final epidemic size (total fraction infected)
    - Time to peak
    """
    rng = np.random.default_rng(42)

    all_beta = []
    all_gamma = []
    all_R0 = []
    all_peak_I = []
    all_final_size = []
    all_time_to_peak = []

    for i in range(n_samples):
        beta = rng.uniform(0.1, 0.8)
        gamma = rng.uniform(0.02, 0.3)
        S_0 = rng.uniform(0.95, 1.0)
        I_0 = 1.0 - S_0

        config = SimulationConfig(
            domain=Domain.EPIDEMIOLOGICAL,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta,
                "gamma": gamma,
                "S_0": S_0,
                "I_0": I_0,
                "R_0_init": 0.0,
            },
        )
        sim = SIRSimulation(config)
        sim.reset()

        peak_I = 0.0
        peak_time = 0
        states = [sim.observe().copy()]
        for step in range(n_steps):
            state = sim.step()
            states.append(state.copy())
            if state[1] > peak_I:
                peak_I = state[1]
                peak_time = step + 1
            # Early stop if epidemic is over
            if state[1] < 1e-6 and step > 100:
                break

        final_R = states[-1][2]  # Final recovered fraction = total infected

        all_beta.append(beta)
        all_gamma.append(gamma)
        all_R0.append(beta / gamma)
        all_peak_I.append(peak_I)
        all_final_size.append(final_R)
        all_time_to_peak.append(peak_time * dt)

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} SIR trajectories")

    return {
        "beta": np.array(all_beta),
        "gamma": np.array(all_gamma),
        "R0": np.array(all_R0),
        "peak_I": np.array(all_peak_I),
        "final_size": np.array(all_final_size),
        "time_to_peak": np.array(all_time_to_peak),
    }


def generate_sir_ode_data(
    n_steps: int = 3000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate a single SIR trajectory for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.EPIDEMIOLOGICAL,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": 0.3,
            "gamma": 0.1,
            "S_0": 0.99,
            "I_0": 0.01,
            "R_0_init": 0.0,
        },
    )
    sim = SIRSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "beta": 0.3,
        "gamma": 0.1,
    }


def run_sir_rediscovery(
    output_dir: str | Path = "output/rediscovery/sir_epidemic",
    n_iterations: int = 40,
    n_samples: int = 200,
) -> dict:
    """Run the full SIR epidemic rediscovery.

    1. Sweep beta/gamma parameter space
    2. Run PySR to find R0 = beta/gamma from peak_I and final_size data
    3. Run SINDy to recover SIR ODEs
    4. Compare with known results
    """
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "sir_epidemic",
        "targets": {
            "R0": "beta / gamma",
            "ode_S": "dS/dt = -beta * S * I",
            "ode_I": "dI/dt = beta * S * I - gamma * I",
            "ode_R": "dR/dt = gamma * I",
        },
    }

    # --- Part 1: R0 rediscovery via PySR ---
    logger.info("Part 1: Generating SIR parameter sweep data...")
    data = generate_sir_sweep_data(n_samples=n_samples, n_steps=5000, dt=0.1)

    # PySR: predict final_size from beta, gamma
    X = np.column_stack([data["beta"], data["gamma"]])
    y = data["final_size"]

    # Filter to epidemics that actually occurred (R0 > 1)
    mask = data["R0"] > 1.0
    X_filtered = X[mask]
    y_filtered = y[mask]

    logger.info(f"  {mask.sum()}/{len(mask)} epidemics with R0 > 1")
    logger.info(f"  Running PySR for final size = f(beta, gamma)...")

    # Use b_ and g_ to avoid sympy conflicts
    discoveries = run_symbolic_regression(
        X_filtered,
        y_filtered,
        variable_names=["b_", "g_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log"],
        max_complexity=20,
        populations=20,
        population_size=40,
    )

    results["final_size_pysr"] = {
        "n_epidemics": int(mask.sum()),
        "n_discoveries": len(discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in discoveries[:5]
        ],
    }
    if discoveries:
        best = discoveries[0]
        results["final_size_pysr"]["best"] = best.expression
        results["final_size_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")

    # PySR: predict R0 directly from peak characteristics
    # Use ratio beta/gamma as target
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
        "n_discoveries": len(r0_discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in r0_discoveries[:5]
        ],
    }
    if r0_discoveries:
        best = r0_discoveries[0]
        results["R0_pysr"]["best"] = best.expression
        results["R0_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(f"  Best R0: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")

    # --- Part 2: SINDy ODE recovery ---
    logger.info("Part 2: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_sir_ode_data(n_steps=3000, dt=0.1)
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
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
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
