"""Goodwin oscillator rediscovery.

Targets:
- ODE: dx/dt = a/(K^n+z^n) - b*x, dy/dt = alpha*x - beta*y, dz/dt = gamma*y - delta*z
- Oscillation onset at Hill coefficient n_c ~ 8 (Griffith's theorem)
- Period and amplitude vs Hill coefficient n
- Equilibrium verification
- SINDy ODE recovery for the linear terms
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.goodwin import GoodwinSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    a: float = 1.0,
    K: float = 1.0,
    n: float = 9.0,
    b: float = 0.1,
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.1,
    delta: float = 0.1,
    n_steps: int = 10000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.GOODWIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a, "K": K, "n": n, "b": b,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
        },
    )
    sim = GoodwinSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "x": states_arr[:, 0],
        "y": states_arr[:, 1],
        "z": states_arr[:, 2],
        "n": n,
    }


def generate_hill_sweep_data(
    n_range: tuple[float, float] = (2.0, 15.0),
    n_points: int = 30,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep Hill coefficient n and measure amplitude to find oscillation onset."""
    n_values = np.linspace(n_range[0], n_range[1], n_points)
    amplitudes = []

    for i, n_val in enumerate(n_values):
        config = SimulationConfig(
            domain=Domain.GOODWIN,
            dt=dt,
            n_steps=1000,
            parameters={"n": float(n_val)},
        )
        sim = GoodwinSimulation(config)
        sim.reset()

        amp = sim.measure_amplitude(transient_time=300.0)
        amplitudes.append(amp)

        if (i + 1) % 10 == 0:
            logger.info(f"  n={n_val:.2f}: amplitude={amp:.6f}")

    return {
        "n_values": n_values,
        "amplitudes": np.array(amplitudes),
    }


def generate_period_data(
    n_range: tuple[float, float] = (8.5, 15.0),
    n_points: int = 20,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep Hill coefficient n above threshold and measure oscillation periods."""
    n_values = np.linspace(n_range[0], n_range[1], n_points)
    periods = []

    for i, n_val in enumerate(n_values):
        config = SimulationConfig(
            domain=Domain.GOODWIN,
            dt=dt,
            n_steps=1000,
            parameters={"n": float(n_val)},
        )
        sim = GoodwinSimulation(config)
        sim.reset()
        period = sim.compute_period(
            n_transient=int(500.0 / dt),
            n_measure=int(800.0 / dt),
        )
        periods.append(period)

        if (i + 1) % 5 == 0:
            logger.info(f"  n={n_val:.2f}: period={period:.4f}")

    return {
        "n_values": n_values,
        "periods": np.array(periods),
    }


def run_goodwin_rediscovery(
    output_dir: str | Path = "output/rediscovery/goodwin",
    n_iterations: int = 40,
) -> dict:
    """Run Goodwin oscillator rediscovery pipeline.

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iteration count.

    Returns:
        Results dict with all discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "goodwin",
        "targets": {
            "ode": "dx/dt=a/(K^n+z^n)-b*x, dy/dt=alpha*x-beta*y, dz/dt=gamma*y-delta*z",
            "griffith_threshold": "n > 8 for 3-variable oscillation",
            "equilibrium": "(x*, y*, z*) from chain equations",
        },
    }

    # --- Part 1: Equilibrium verification ---
    logger.info("Part 1: Equilibrium verification...")
    config = SimulationConfig(
        domain=Domain.GOODWIN,
        dt=0.1,
        n_steps=1000,
        parameters={"n": 9.0},
    )
    sim = GoodwinSimulation(config)
    sim.reset()
    eq = sim.equilibrium()
    dy = sim._derivatives(np.array(eq))
    results["equilibrium"] = {
        "x_star": float(eq[0]),
        "y_star": float(eq[1]),
        "z_star": float(eq[2]),
        "derivative_at_eq": [float(d) for d in dy],
        "derivative_norm": float(np.linalg.norm(dy)),
    }
    logger.info(
        f"  Equilibrium: ({eq[0]:.6f}, {eq[1]:.6f}, {eq[2]:.6f}), "
        f"|f(x*)|={np.linalg.norm(dy):.2e}"
    )

    # --- Part 2: Hill coefficient sweep (oscillation onset) ---
    logger.info("Part 2: Hill coefficient sweep for oscillation onset...")
    sweep_data = generate_hill_sweep_data(n_range=(2.0, 15.0), n_points=30, dt=0.1)

    # Detect onset: first n where amplitude > threshold
    threshold = 0.01
    above = sweep_data["amplitudes"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        n_c_est = sweep_data["n_values"][max(0, idx - 1)]
        results["griffith_onset"] = {
            "n_c_estimate": float(n_c_est),
            "n_c_theory": 8.0,
            "relative_error": float(abs(n_c_est - 8.0) / 8.0),
        }
        logger.info(f"  n_c estimate: {n_c_est:.2f} (theory: ~8.0)")
    else:
        results["griffith_onset"] = {"error": "No oscillation detected in sweep"}

    # --- Part 3: SINDy ODE recovery ---
    logger.info("Part 3: SINDy ODE recovery at n=9...")
    data = generate_ode_data(n=9.0, n_steps=10000, dt=0.1)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.1,
            feature_names=["x", "y", "z"],
            threshold=0.005,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries[:6]
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

    # --- Part 4: Period vs Hill coefficient ---
    logger.info("Part 4: Period vs Hill coefficient...")
    try:
        period_data = generate_period_data(n_range=(8.5, 15.0), n_points=15, dt=0.1)
        finite = np.isfinite(period_data["periods"])
        if np.sum(finite) > 3:
            results["period"] = {
                "n_measured": int(np.sum(finite)),
                "n_range": [float(period_data["n_values"].min()),
                            float(period_data["n_values"].max())],
                "period_range": [
                    float(np.min(period_data["periods"][finite])),
                    float(np.max(period_data["periods"][finite])),
                ],
            }
            logger.info(
                f"  Measured {np.sum(finite)} periods in "
                f"n=[{period_data['n_values'].min():.2f}, "
                f"{period_data['n_values'].max():.2f}]"
            )
    except Exception as e:
        logger.warning(f"Period measurement failed: {e}")
        results["period"] = {"error": str(e)}

    # --- Part 5: PySR amplitude vs n ---
    logger.info("Part 5: PySR amplitude vs n...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use sweep data above threshold
        valid = sweep_data["amplitudes"] > threshold
        if np.sum(valid) > 5:
            X = sweep_data["n_values"][valid].reshape(-1, 1)
            y = sweep_data["amplitudes"][valid]

            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["n_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square", "sqrt"],
                max_complexity=10,
                populations=20,
                population_size=40,
            )
            results["amplitude_pysr"] = {
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
                results["amplitude_pysr"]["best"] = best.expression
                results["amplitude_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["amplitude_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
