"""Van der Pol oscillator rediscovery.

Targets:
- ODE: x'' - mu*(1-x^2)*x' + x = 0  (via SINDy)
- Limit cycle amplitude: A ~ 2 (constant for all mu > 0)
- Period scaling: T ~ 2*pi for small mu, T ~ (3-2*ln(2))*mu for large mu
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.van_der_pol import VanDerPolSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    mu: float = 1.0,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.VAN_DER_POL,
        dt=dt,
        n_steps=n_steps,
        parameters={"mu": mu, "x_0": 0.5, "v_0": 0.0},
    )
    sim = VanDerPolSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    times = np.arange(n_steps + 1) * dt

    return {
        "time": times,
        "states": states,
        "x": states[:, 0],
        "v": states[:, 1],
        "mu": mu,
        "dt": dt,
    }


def generate_period_data(
    n_mu: int = 30,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Generate period vs mu data across regimes."""
    mu_values = np.logspace(-1, 1.5, n_mu)  # 0.1 to ~31.6
    periods = []
    amplitudes = []

    for i, mu in enumerate(mu_values):
        config = SimulationConfig(
            domain=Domain.VAN_DER_POL,
            dt=dt,
            n_steps=1000,
            parameters={"mu": mu, "x_0": 0.1, "v_0": 0.0},
        )
        sim = VanDerPolSimulation(config)
        sim.reset()

        T = sim.measure_period(n_periods=5)
        periods.append(T)

        # Measure amplitude separately
        sim.reset()
        A = sim.measure_amplitude(n_periods=3)
        amplitudes.append(A)

        if (i + 1) % 10 == 0:
            logger.info(f"  mu={mu:.3f}: T={T:.4f}, A={A:.4f}")

    return {
        "mu": mu_values,
        "period": np.array(periods),
        "amplitude": np.array(amplitudes),
    }


def run_van_der_pol_rediscovery(
    output_dir: str | Path = "output/rediscovery/van_der_pol",
    n_iterations: int = 40,
) -> dict:
    """Run Van der Pol rediscovery pipeline.

    1. Generate trajectory for SINDy ODE recovery
    2. Generate period/amplitude vs mu data
    3. Run PySR to find T(mu) and A(mu)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "van_der_pol",
        "targets": {
            "ode": "x'' - mu*(1-x^2)*x' + x = 0",
            "amplitude": "A ~ 2 (limit cycle)",
            "period_small_mu": "T ~ 2*pi",
            "period_large_mu": "T ~ (3-2*ln(2))*mu",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at mu=1.0...")
    data = generate_ode_data(mu=1.0, n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy_discovery

        sindy_results = run_sindy_discovery(
            data["time"],
            data["states"],
            feature_names=["x", "v"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = sindy_results
        logger.info(f"  SINDy results: {sindy_results}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Period and amplitude vs mu ---
    logger.info("Part 2: Period and amplitude vs mu...")
    pa_data = generate_period_data(n_mu=30, dt=0.005)

    valid = np.isfinite(pa_data["period"]) & (pa_data["period"] > 0)
    results["period_data"] = {
        "n_samples": int(np.sum(valid)),
        "mu_range": [float(pa_data["mu"].min()), float(pa_data["mu"].max())],
        "period_range": [
            float(np.min(pa_data["period"][valid])),
            float(np.max(pa_data["period"][valid])),
        ],
        "mean_amplitude": float(np.mean(pa_data["amplitude"][valid])),
    }
    logger.info(f"  Mean amplitude: {np.mean(pa_data['amplitude'][valid]):.4f} (theory: 2.0)")

    # PySR: find T(mu)
    try:
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        X = pa_data["mu"][valid].reshape(-1, 1)
        y = pa_data["period"][valid]

        logger.info("  Running PySR: T = f(mu)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["mu"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["log", "sqrt"],
            max_complexity=12,
            populations=20,
            population_size=40,
        )
        results["period_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["period_pysr"]["best"] = best.expression
            results["period_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["period_pysr"] = {"error": str(e)}

    # PySR: find A(mu)
    try:
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        X = pa_data["mu"][valid].reshape(-1, 1)
        y = pa_data["amplitude"][valid]

        logger.info("  Running PySR: A = f(mu)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["mu"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt"],
            max_complexity=8,
            populations=15,
            population_size=30,
        )
        results["amplitude_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["amplitude_pysr"]["best"] = best.expression
            results["amplitude_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["amplitude_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "period_data.npz",
        mu=pa_data["mu"],
        period=pa_data["period"],
        amplitude=pa_data["amplitude"],
    )

    return results
