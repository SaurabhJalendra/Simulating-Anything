"""Bernoulli ODE rediscovery.

Targets:
- Analytical solution verification: y(t) vs exact formula
- Steady-state recovery: y_ss = (-a/b)^(1/(n-1)) via PySR
- Linearization verification: v = y^(1-n) follows a linear ODE
- SINDy ODE recovery: dy/dt = a*y + b*y^n

For a stable nontrivial steady state, we need a > 0, b < 0 (n=2).
This gives y_ss = -a/b, with stability f'(y_ss) = -a < 0.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.bernoulli_ode import BernoulliODESimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    a: float = 1.0,
    b: float = -1.0,
    n: float = 2.0,
    y_0: float = 0.1,
    dt: float = 0.01,
    n_steps: int = 500,
) -> SimulationConfig:
    """Create a SimulationConfig for the Bernoulli ODE."""
    return SimulationConfig(
        domain=Domain.BERNOULLI_ODE,
        dt=dt,
        n_steps=n_steps,
        parameters={"a": a, "b": b, "n": n, "y_0": y_0},
    )


def generate_solution_data(
    n_samples: int = 200,
    dt: float = 0.01,
    n_steps: int = 500,
) -> dict[str, np.ndarray]:
    """Generate parameter sweep data comparing numerical to analytical solutions.

    Sweeps a > 0 and b < 0 with fixed n=2 (stable regime) to produce
    (a, b, y_final_numerical, y_final_analytical) tuples.
    """
    n_side = int(np.sqrt(n_samples))
    a_values = np.linspace(0.1, 2.0, n_side)
    b_values = np.linspace(-2.0, -0.1, n_side)

    a_arr = []
    b_arr = []
    y_final_num = []
    y_final_ana = []
    y_ss_theory = []

    for a_val in a_values:
        for b_val in b_values:
            config = _make_config(a=a_val, b=b_val, n=2.0, y_0=0.1,
                                  dt=dt, n_steps=n_steps)
            sim = BernoulliODESimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()

            y_num = float(sim.observe()[0])
            t_final = n_steps * dt
            y_ana_arr = sim.analytical_solution(t_final)
            y_ana = float(y_ana_arr) if np.isfinite(y_ana_arr) else np.nan

            # Steady state for n=2: y_ss = -a/b
            ss = -a_val / b_val if -a_val / b_val > 0 else np.nan

            a_arr.append(a_val)
            b_arr.append(b_val)
            y_final_num.append(y_num)
            y_final_ana.append(y_ana)
            y_ss_theory.append(ss)

    return {
        "a": np.array(a_arr),
        "b": np.array(b_arr),
        "y_final_numerical": np.array(y_final_num),
        "y_final_analytical": np.array(y_final_ana),
        "y_ss_theory": np.array(y_ss_theory),
        "n_samples": len(a_arr),
        "n": 2.0,
        "dt": dt,
        "n_steps": n_steps,
    }


def generate_ode_data(
    a: float = 1.0,
    b: float = -1.0,
    n: float = 2.0,
    y_0: float = 0.1,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = _make_config(a=a, b=b, n=n, y_0=y_0, dt=dt, n_steps=n_steps)
    sim = BernoulliODESimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    times = np.arange(n_steps + 1) * dt

    return {
        "time": times,
        "states": states_arr,
        "y": states_arr[:, 0],
        "a": a,
        "b": b,
        "n": n,
        "dt": dt,
    }


def generate_linearization_data(
    a: float = 1.0,
    b: float = -1.0,
    n: float = 2.0,
    y_0: float = 0.1,
    n_steps: int = 1000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate data showing the linearization v = y^(1-n).

    Returns trajectory in both y and v coordinates, along with the
    analytical v(t) for comparison.
    """
    config = _make_config(a=a, b=b, n=n, y_0=y_0, dt=dt, n_steps=n_steps)
    sim = BernoulliODESimulation(config)
    result = sim.compute_linearized_trajectory(n_steps=n_steps, dt=dt)
    result["a"] = a
    result["b"] = b
    result["n"] = n
    return result


def generate_steady_state_data(
    n_a: int = 20,
    n_b: int = 20,
    n: float = 2.0,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep a > 0 and b < 0, measure steady-state y for PySR recovery.

    Uses the stable regime where y_ss = -a/b is an attractor.
    For n=2: y_ss = -a/b.
    For general n: y_ss = (-a/b)^(1/(n-1)).
    """
    a_values = np.linspace(0.1, 3.0, n_a)
    b_values = np.linspace(-3.0, -0.1, n_b)

    a_arr = []
    b_arr = []
    y_final_arr = []
    y_ss_theory_arr = []

    for i, a_val in enumerate(a_values):
        for b_val in b_values:
            config = _make_config(a=a_val, b=b_val, n=n, y_0=0.1,
                                  dt=dt, n_steps=3000)
            sim = BernoulliODESimulation(config)
            sim.reset()

            for _ in range(3000):
                sim.step()

            y_final = float(sim.observe()[0])

            ratio = -a_val / b_val
            if ratio > 0:
                y_ss = ratio ** (1.0 / (n - 1.0))
            else:
                y_ss = np.nan

            a_arr.append(a_val)
            b_arr.append(b_val)
            y_final_arr.append(y_final)
            y_ss_theory_arr.append(y_ss)

        if (i + 1) % 5 == 0:
            logger.info(f"  a sweep: {i + 1}/{n_a}")

    return {
        "a": np.array(a_arr),
        "b": np.array(b_arr),
        "y_final": np.array(y_final_arr),
        "y_ss_theory": np.array(y_ss_theory_arr),
        "n": n,
    }


def run_bernoulli_ode_rediscovery(
    output_dir: str | Path = "output/rediscovery/bernoulli_ode",
    n_iterations: int = 40,
) -> dict:
    """Run Bernoulli ODE rediscovery pipeline.

    Discovers:
    1. Analytical solution match (numerical vs exact)
    2. Steady-state formula: y_ss = (-a/b)^(1/(n-1)) via PySR
    3. Linearization verification: v = y^(1-n) is linear
    4. SINDy ODE recovery: dy/dt = a*y + b*y^n
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "bernoulli_ode",
        "targets": {
            "ode": "dy/dt = a*y + b*y^n",
            "steady_state": "y_ss = (-a/b)^(1/(n-1))",
            "linearization": "v = y^(1-n) -> dv/dt = (1-n)*(a*v + b)",
            "analytical": "v(t) = (v_0 + b/a)*exp((1-n)*a*t) - b/a",
        },
    }

    # --- Part 1: Analytical solution verification ---
    logger.info("Part 1: Analytical solution verification...")
    sol_data = generate_solution_data(n_samples=100, dt=0.01, n_steps=500)

    valid = np.isfinite(sol_data["y_final_analytical"]) & np.isfinite(
        sol_data["y_final_numerical"]
    )
    if np.sum(valid) > 0:
        errors = np.abs(
            sol_data["y_final_numerical"][valid]
            - sol_data["y_final_analytical"][valid]
        )
        rel_errors = errors / (
            np.abs(sol_data["y_final_analytical"][valid]) + 1e-15
        )
        results["analytical_verification"] = {
            "n_valid": int(np.sum(valid)),
            "mean_absolute_error": float(np.mean(errors)),
            "mean_relative_error": float(np.mean(rel_errors)),
            "max_relative_error": float(np.max(rel_errors)),
        }
        logger.info(
            f"  {int(np.sum(valid))} valid points, "
            f"mean rel error: {float(np.mean(rel_errors)):.2e}"
        )

    # --- Part 2: Linearization verification ---
    logger.info("Part 2: Linearization v = y^(1-n) verification...")
    lin_data = generate_linearization_data(
        a=1.0, b=-1.0, n=2.0, y_0=0.1, n_steps=1000, dt=0.01,
    )

    valid_lin = np.isfinite(lin_data["v"]) & np.isfinite(lin_data["v_analytical"])
    if np.sum(valid_lin) > 0:
        v_err = np.abs(
            lin_data["v"][valid_lin] - lin_data["v_analytical"][valid_lin]
        )
        results["linearization"] = {
            "n_valid": int(np.sum(valid_lin)),
            "mean_v_error": float(np.mean(v_err)),
            "max_v_error": float(np.max(v_err)),
        }
        logger.info(
            f"  v-space error: mean={float(np.mean(v_err)):.2e}, "
            f"max={float(np.max(v_err)):.2e}"
        )

    # --- Part 3: PySR steady-state formula ---
    logger.info("Part 3: Steady-state sweep for PySR y_ss = f(a, b)...")
    ss_data = generate_steady_state_data(n_a=15, n_b=15, n=2.0, dt=0.01)

    valid_ss = np.isfinite(ss_data["y_ss_theory"]) & (ss_data["y_final"] > 0.01)
    if np.sum(valid_ss) > 0:
        ss_err = np.abs(
            ss_data["y_final"][valid_ss] - ss_data["y_ss_theory"][valid_ss]
        )
        correlation = float(np.corrcoef(
            ss_data["y_final"][valid_ss], ss_data["y_ss_theory"][valid_ss],
        )[0, 1])
        results["steady_state_data"] = {
            "n_valid": int(np.sum(valid_ss)),
            "mean_error_vs_theory": float(np.mean(ss_err)),
            "correlation": correlation,
        }
        logger.info(
            f"  {int(np.sum(valid_ss))} valid points, "
            f"correlation={correlation:.4f}"
        )

    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = np.column_stack([
            ss_data["a"][valid_ss], ss_data["b"][valid_ss],
        ])
        y = ss_data["y_final"][valid_ss]

        logger.info("  Running PySR: y_ss = f(a, b)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["a_", "b_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt", "abs"],
            max_complexity=8,
            populations=20,
            population_size=40,
        )
        results["steady_state_pysr"] = {
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
            results["steady_state_pysr"]["best"] = best.expression
            results["steady_state_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["steady_state_pysr"] = {"error": str(e)}

    # --- Part 4: SINDy ODE recovery ---
    logger.info("Part 4: SINDy ODE recovery...")
    ode_data = generate_ode_data(
        a=1.0, b=-1.0, n=2.0, y_0=0.1, n_steps=5000, dt=0.01,
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["y"],
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

    np.savez(
        output_path / "steady_state_data.npz",
        a=ss_data["a"],
        b=ss_data["b"],
        y_final=ss_data["y_final"],
        y_ss_theory=ss_data["y_ss_theory"],
    )

    return results
