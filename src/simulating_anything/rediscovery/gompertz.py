"""Gompertz growth model rediscovery.

Targets:
- Analytical solution: N(t) = K * (N_0/K)^exp(-r*t) via trajectory match
- Inflection point: N_inflection = K / e via PySR
- Maximum growth rate: dN/dt_max = r * K / e via PySR
- ODE recovery via SINDy: dN/dt ~ r * N * ln(K/N)
- Growth rate scaling with r and K
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.gompertz import GompertzSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    r: float = 0.5,
    K: float = 100.0,
    N_0: float = 1.0,
    dt: float = 0.1,
    n_steps: int = 500,
) -> SimulationConfig:
    """Create a SimulationConfig for the Gompertz model."""
    return SimulationConfig(
        domain=Domain.GOMPERTZ,
        dt=dt,
        n_steps=n_steps,
        parameters={"r": r, "K": K, "N_0": N_0},
    )


def generate_growth_data(
    n_samples: int = 200,
    dt: float = 0.1,
    n_steps: int = 500,
) -> dict[str, np.ndarray]:
    """Generate parameter sweep data for PySR discovery of growth scaling.

    Sweeps r and K, records final population, inflection time, and max growth
    rate for each parameter combination.

    Args:
        n_samples: total number of (r, K) samples.
        dt: integration timestep.
        n_steps: number of integration steps per sample.

    Returns:
        Dict with arrays: 'r', 'K', 'N_0', 'inflection_N', 'max_dNdt',
        'inflection_time', 'final_N'.
    """
    n_r = int(np.sqrt(n_samples))
    n_K = n_samples // n_r

    r_values = np.linspace(0.1, 2.0, n_r)
    K_values = np.linspace(10.0, 200.0, n_K)

    r_arr = []
    K_arr = []
    N_0_arr = []
    inflection_N_arr = []
    max_dNdt_arr = []
    inflection_time_arr = []
    final_N_arr = []

    for r_val in r_values:
        for K_val in K_values:
            N_0 = 1.0
            config = _make_config(r=r_val, K=K_val, N_0=N_0, dt=dt, n_steps=n_steps)
            sim = GompertzSimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()

            r_arr.append(r_val)
            K_arr.append(K_val)
            N_0_arr.append(N_0)
            inflection_N_arr.append(sim.inflection_point)
            max_dNdt_arr.append(sim.max_growth_rate)
            inflection_time_arr.append(sim.inflection_time())
            final_N_arr.append(float(sim.observe()[0]))

    return {
        "r": np.array(r_arr),
        "K": np.array(K_arr),
        "N_0": np.array(N_0_arr),
        "inflection_N": np.array(inflection_N_arr),
        "max_dNdt": np.array(max_dNdt_arr),
        "inflection_time": np.array(inflection_time_arr),
        "final_N": np.array(final_N_arr),
    }


def generate_ode_data(
    r: float = 0.5,
    K: float = 100.0,
    N_0: float = 1.0,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses a fine timestep to produce a smooth trajectory suitable for numerical
    differentiation by SINDy.

    Args:
        r: growth rate.
        K: carrying capacity.
        N_0: initial population.
        n_steps: number of integration steps.
        dt: timestep for integration.

    Returns:
        Dict with 'time', 'states', 'N', and ground truth parameters.
    """
    config = _make_config(r=r, K=K, N_0=N_0, dt=dt, n_steps=n_steps)
    sim = GompertzSimulation(config)
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
        "r": r,
        "K": K,
        "N_0": N_0,
        "dt": dt,
    }


def generate_inflection_data(
    n_K: int = 30,
    r: float = 0.5,
    N_0: float = 1.0,
    dt: float = 0.1,
    n_steps: int = 1000,
) -> dict[str, np.ndarray]:
    """Sweep K and measure the inflection point for PySR.

    The inflection point of the Gompertz model is at N = K/e, so we sweep K
    and record both the theoretical and measured inflection points.

    Args:
        n_K: number of K values.
        r: growth rate (fixed).
        N_0: initial population.
        dt: timestep.
        n_steps: steps per run.

    Returns:
        Dict with 'K', 'inflection_N_theory', 'inflection_N_measured', etc.
    """
    K_values = np.linspace(10.0, 500.0, n_K)
    inflection_theory = []
    inflection_measured = []
    max_dNdt_theory = []
    max_dNdt_measured = []

    for K_val in K_values:
        config = _make_config(r=r, K=K_val, N_0=N_0, dt=dt, n_steps=n_steps)
        sim = GompertzSimulation(config)
        sim.reset()

        # Track maximum growth rate and corresponding N
        max_rate = 0.0
        N_at_max_rate = 0.0
        prev_N = N_0

        for _ in range(n_steps):
            sim.step()
            curr_N = float(sim.observe()[0])
            rate = (curr_N - prev_N) / dt
            if rate > max_rate:
                max_rate = rate
                N_at_max_rate = (curr_N + prev_N) / 2.0
            prev_N = curr_N

        inflection_theory.append(K_val / np.e)
        inflection_measured.append(N_at_max_rate)
        max_dNdt_theory.append(r * K_val / np.e)
        max_dNdt_measured.append(max_rate)

    return {
        "K": K_values,
        "inflection_N_theory": np.array(inflection_theory),
        "inflection_N_measured": np.array(inflection_measured),
        "max_dNdt_theory": np.array(max_dNdt_theory),
        "max_dNdt_measured": np.array(max_dNdt_measured),
    }


def run_gompertz_rediscovery(
    output_dir: str | Path = "output/rediscovery/gompertz",
    n_iterations: int = 40,
) -> dict:
    """Run Gompertz growth model rediscovery pipeline.

    Discovers:
    1. Analytical solution match: N(t) = K * (N_0/K)^exp(-r*t)
    2. Inflection point: N = K/e via PySR
    3. Maximum growth rate: dN/dt_max = r*K/e via PySR
    4. ODE recovery via SINDy
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "gompertz",
        "targets": {
            "analytical": "N(t) = K * (N_0/K)^exp(-r*t)",
            "inflection": "N_inflection = K / e",
            "max_growth_rate": "dN/dt_max = r * K / e",
            "ode": "dN/dt = r * N * ln(K/N)",
        },
    }

    # --- Part 1: Analytical solution verification ---
    logger.info("Part 1: Analytical solution verification...")
    config = _make_config(r=0.5, K=100.0, N_0=1.0, dt=0.01, n_steps=2000)
    sim = GompertzSimulation(config)
    sim.reset()

    times = np.arange(2001) * 0.01
    analytical = sim.analytical_solution(times)

    states = [sim.observe().copy()]
    for _ in range(2000):
        sim.step()
        states.append(sim.observe().copy())
    numerical = np.array(states)[:, 0]

    rel_error = np.abs(numerical - analytical) / np.maximum(analytical, 1e-10)
    mean_rel_error = float(np.mean(rel_error))
    max_rel_error = float(np.max(rel_error))

    results["analytical_match"] = {
        "mean_relative_error": mean_rel_error,
        "max_relative_error": max_rel_error,
        "n_points": len(times),
    }
    logger.info(
        f"  Analytical match: mean_rel_error={mean_rel_error:.2e}, "
        f"max_rel_error={max_rel_error:.2e}"
    )

    # --- Part 2: Inflection point via K sweep ---
    logger.info("Part 2: Inflection point via K sweep...")
    infl_data = generate_inflection_data(n_K=30, r=0.5, N_0=1.0, dt=0.05, n_steps=2000)

    correlation_infl = float(np.corrcoef(
        infl_data["inflection_N_theory"],
        infl_data["inflection_N_measured"],
    )[0, 1])
    mean_infl_error = float(np.mean(np.abs(
        infl_data["inflection_N_measured"] - infl_data["inflection_N_theory"]
    ) / infl_data["inflection_N_theory"]))

    results["inflection_point"] = {
        "n_points": len(infl_data["K"]),
        "correlation": correlation_infl,
        "mean_relative_error": mean_infl_error,
        "K_over_e_theory": True,
    }
    logger.info(
        f"  Inflection: correlation={correlation_infl:.4f}, "
        f"mean_rel_error={mean_infl_error:.4f}"
    )

    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = infl_data["K"].reshape(-1, 1)
        y = infl_data["inflection_N_measured"]

        logger.info("  Running PySR: N_inflection = f(K)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["K_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log"],
            max_complexity=8,
            populations=20,
            population_size=40,
        )
        results["inflection_pysr"] = {
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
            results["inflection_pysr"]["best"] = best.expression
            results["inflection_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR inflection failed: {e}")
        results["inflection_pysr"] = {"error": str(e)}

    # --- Part 3: Max growth rate via PySR ---
    logger.info("Part 3: Max growth rate = r*K/e...")
    growth_data = generate_growth_data(n_samples=200, dt=0.05, n_steps=2000)

    valid = np.isfinite(growth_data["max_dNdt"]) & (growth_data["max_dNdt"] > 0)
    results["growth_rate_data"] = {
        "n_samples": int(np.sum(valid)),
        "r_range": [float(growth_data["r"].min()), float(growth_data["r"].max())],
        "K_range": [float(growth_data["K"].min()), float(growth_data["K"].max())],
    }

    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = np.column_stack([growth_data["r"][valid], growth_data["K"][valid]])
        y = growth_data["max_dNdt"][valid]

        logger.info("  Running PySR: dN/dt_max = f(r, K)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["r_", "K_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log"],
            max_complexity=8,
            populations=20,
            population_size=40,
        )
        results["growth_rate_pysr"] = {
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
            results["growth_rate_pysr"]["best"] = best.expression
            results["growth_rate_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR growth rate failed: {e}")
        results["growth_rate_pysr"] = {"error": str(e)}

    # --- Part 4: SINDy ODE recovery ---
    logger.info("Part 4: SINDy ODE recovery...")
    ode_data = generate_ode_data(r=0.5, K=100.0, N_0=1.0, n_steps=5000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["N"],
            threshold=0.01,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
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

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
