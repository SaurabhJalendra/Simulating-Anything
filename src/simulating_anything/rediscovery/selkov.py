"""Selkov glycolysis model rediscovery.

Targets:
- ODE: dx/dt = -x + a*y + x^2*y, dy/dt = b - a*y - x^2*y  (via SINDy)
- Hopf bifurcation boundary b_c(a) in (a, b) parameter space
- Oscillation period and limit cycle amplitude
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.selkov import (
    SelkovSimulation,
    compute_hopf_boundary,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    a: float = 0.08,
    b: float = 0.6,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.SELKOV,
        dt=dt,
        n_steps=n_steps,
        parameters={"a": a, "b": b, "x_0": 0.5, "y_0": 0.5},
    )
    sim = SelkovSimulation(config)
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
        "a": a,
        "b": b,
    }


def generate_bifurcation_data(
    a: float = 0.08,
    n_b: int = 30,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep b and measure amplitude to find Hopf bifurcation."""
    b_values = np.linspace(0.1, 1.5, n_b)
    amplitudes = []

    for i, b in enumerate(b_values):
        config = SimulationConfig(
            domain=Domain.SELKOV,
            dt=dt,
            n_steps=1000,
            parameters={"a": a, "b": b, "x_0": b + 0.1, "y_0": 0.5},
        )
        sim = SelkovSimulation(config)
        sim.reset()

        # Transient
        for _ in range(int(500 / dt)):
            sim.step()

        # Measure amplitude
        x_vals = []
        for _ in range(int(200 / dt)):
            sim.step()
            x_vals.append(sim.observe()[0])

        amp = max(x_vals) - min(x_vals)
        amplitudes.append(amp)

        if (i + 1) % 10 == 0:
            logger.info(f"  b={b:.3f}: amplitude={amp:.4f}")

    return {
        "a": a,
        "b": b_values,
        "amplitude": np.array(amplitudes),
        "b_c_theory": float(compute_hopf_boundary(np.array([a]))[0]),
    }


def generate_period_data(
    a: float = 0.08,
    n_b: int = 20,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep b above Hopf threshold and measure oscillation periods."""
    b_c = float(compute_hopf_boundary(np.array([a]))[0])
    # Sweep b from just above b_c to well above
    b_values = np.linspace(b_c * 1.1, b_c * 3.0, n_b)
    periods = []

    for i, b in enumerate(b_values):
        config = SimulationConfig(
            domain=Domain.SELKOV,
            dt=dt,
            n_steps=1000,
            parameters={"a": a, "b": b, "x_0": 0.5, "y_0": 0.5},
        )
        sim = SelkovSimulation(config)
        sim.reset()
        period = sim.measure_period(n_periods=5)
        periods.append(period)

        if (i + 1) % 5 == 0:
            logger.info(f"  b={b:.3f}: period={period:.4f}")

    return {
        "a": a,
        "b": b_values,
        "period": np.array(periods),
        "b_c": b_c,
    }


def run_selkov_rediscovery(
    output_dir: str | Path = "output/rediscovery/selkov",
    n_iterations: int = 40,
) -> dict:
    """Run Selkov glycolysis model rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "selkov",
        "targets": {
            "ode": "dx/dt = -x + a*y + x^2*y, dy/dt = b - a*y - x^2*y",
            "hopf": "b_c(a) from trace(J) = 0",
            "fixed_point": "(x*, y*) = (b, b/(a+b^2))",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at a=0.08, b=0.6...")
    data = generate_ode_data(a=0.08, b=0.6, n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["x", "y"],
            threshold=0.01,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
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

    # --- Part 2: Hopf bifurcation ---
    logger.info("Part 2: Hopf bifurcation sweep...")
    bif_data = generate_bifurcation_data(a=0.08, n_b=30, dt=0.005)

    # Estimate b_c: first b where amplitude > threshold
    threshold = 0.05
    above = bif_data["amplitude"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        b_c_est = bif_data["b"][max(0, idx - 1)]
        b_c_theory = bif_data["b_c_theory"]
        results["hopf_bifurcation"] = {
            "b_c_estimate": float(b_c_est),
            "b_c_theory": float(b_c_theory),
            "relative_error": float(abs(b_c_est - b_c_theory) / b_c_theory),
        }
        logger.info(f"  b_c estimate: {b_c_est:.4f} (theory: {b_c_theory:.4f})")

    # PySR: find b_c as function of a
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Sweep multiple a values
        a_values = np.linspace(0.02, 0.2, 15)
        b_c_measured = []
        for a_val in a_values:
            bif = generate_bifurcation_data(a=a_val, n_b=20, dt=0.01)
            above_a = bif["amplitude"] > 0.05
            if np.any(above_a):
                idx_a = np.argmax(above_a)
                b_c_measured.append(bif["b"][max(0, idx_a - 1)])
            else:
                b_c_measured.append(np.nan)

        b_c_measured_arr = np.array(b_c_measured)
        valid = np.isfinite(b_c_measured_arr)
        if np.sum(valid) > 5:
            X = a_values[valid].reshape(-1, 1)
            y = b_c_measured_arr[valid]

            logger.info("  Running PySR: b_c = f(a)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["a_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square", "sqrt"],
                max_complexity=10,
                populations=20,
                population_size=40,
            )
            results["hopf_pysr"] = {
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
                results["hopf_pysr"]["best"] = best.expression
                results["hopf_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["hopf_pysr"] = {"error": str(e)}

    # --- Part 3: Period measurement ---
    logger.info("Part 3: Period vs b sweep...")
    try:
        period_data = generate_period_data(a=0.08, n_b=15, dt=0.005)
        finite = np.isfinite(period_data["period"])
        if np.sum(finite) > 3:
            results["period"] = {
                "n_measured": int(np.sum(finite)),
                "b_range": [float(period_data["b"].min()), float(period_data["b"].max())],
                "period_range": [
                    float(np.min(period_data["period"][finite])),
                    float(np.max(period_data["period"][finite])),
                ],
                "b_c": float(period_data["b_c"]),
            }
            logger.info(
                f"  Measured {np.sum(finite)} periods in "
                f"b=[{period_data['b'].min():.3f}, {period_data['b'].max():.3f}]"
            )
    except Exception as e:
        logger.warning(f"Period measurement failed: {e}")
        results["period"] = {"error": str(e)}

    # --- Part 4: Fixed point verification ---
    logger.info("Part 4: Fixed point verification...")
    a_test, b_test = 0.08, 0.6
    config = SimulationConfig(
        domain=Domain.SELKOV,
        dt=0.01,
        n_steps=1000,
        parameters={"a": a_test, "b": b_test, "x_0": b_test, "y_0": b_test / (a_test + b_test**2)},
    )
    sim = SelkovSimulation(config)
    sim.reset()
    fp = sim.fixed_point
    dy = sim._derivatives(np.array(fp))
    results["fixed_point"] = {
        "x_star": fp[0],
        "y_star": fp[1],
        "derivative_at_fp": [float(dy[0]), float(dy[1])],
        "derivative_norm": float(np.linalg.norm(dy)),
    }
    logger.info(
        f"  Fixed point: ({fp[0]:.6f}, {fp[1]:.6f}), "
        f"|f(x*)|={np.linalg.norm(dy):.2e}"
    )

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
