"""Brusselator rediscovery.

Targets:
- ODE: du/dt = a - (b+1)u + u^2*v, dv/dt = bu - u^2*v  (via SINDy)
- Hopf bifurcation: b_c = 1 + a^2
- Fixed point: (u*, v*) = (a, b/a)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.brusselator import BrusselatorSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    a: float = 1.0,
    b: float = 3.0,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.BRUSSELATOR,
        dt=dt,
        n_steps=n_steps,
        parameters={"a": a, "b": b, "u_0": 1.5, "v_0": 1.5},
    )
    sim = BrusselatorSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "u": states[:, 0],
        "v": states[:, 1],
        "a": a,
        "b": b,
    }


def generate_bifurcation_data(
    a: float = 1.0,
    n_b: int = 30,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep b and measure amplitude to find Hopf bifurcation."""
    b_values = np.linspace(0.5, 4.0, n_b)
    amplitudes = []

    for i, b in enumerate(b_values):
        config = SimulationConfig(
            domain=Domain.BRUSSELATOR,
            dt=dt,
            n_steps=1000,
            parameters={"a": a, "b": b, "u_0": a + 0.1, "v_0": b / a + 0.1},
        )
        sim = BrusselatorSimulation(config)
        sim.reset()

        # Transient
        for _ in range(int(200 / dt)):
            sim.step()

        # Measure amplitude
        u_vals = []
        for _ in range(int(100 / dt)):
            sim.step()
            u_vals.append(sim.observe()[0])

        amp = max(u_vals) - min(u_vals)
        amplitudes.append(amp)

        if (i + 1) % 10 == 0:
            logger.info(f"  b={b:.3f}: amplitude={amp:.4f}")

    return {
        "a": a,
        "b": b_values,
        "amplitude": np.array(amplitudes),
        "b_c_theory": 1 + a**2,
    }


def run_brusselator_rediscovery(
    output_dir: str | Path = "output/rediscovery/brusselator",
    n_iterations: int = 40,
) -> dict:
    """Run Brusselator rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "brusselator",
        "targets": {
            "ode": "du/dt = a-(b+1)u+u^2*v, dv/dt = bu-u^2*v",
            "hopf": "b_c = 1 + a^2",
            "fixed_point": "(u*, v*) = (a, b/a)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at a=1, b=3...")
    data = generate_ode_data(a=1.0, b=3.0, n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["u", "v"],
            threshold=0.05,
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
            logger.info(f"  SINDy best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Hopf bifurcation ---
    logger.info("Part 2: Hopf bifurcation sweep...")
    bif_data = generate_bifurcation_data(a=1.0, n_b=30, dt=0.005)

    # Estimate b_c: first b where amplitude > threshold
    threshold = 0.1
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
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        # Sweep multiple a values
        a_values = np.linspace(0.5, 2.0, 15)
        b_c_measured = []
        for a_val in a_values:
            bif = generate_bifurcation_data(a=a_val, n_b=20, dt=0.01)
            above_a = bif["amplitude"] > 0.1
            if np.any(above_a):
                idx_a = np.argmax(above_a)
                b_c_measured.append(bif["b"][max(0, idx_a - 1)])
            else:
                b_c_measured.append(np.nan)

        b_c_measured = np.array(b_c_measured)
        valid = np.isfinite(b_c_measured)
        if np.sum(valid) > 5:
            X = a_values[valid].reshape(-1, 1)
            y = b_c_measured[valid]

            logger.info("  Running PySR: b_c = f(a)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["a"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square"],
                max_complexity=8,
                populations=20,
                population_size=40,
            )
            results["hopf_pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["hopf_pysr"]["best"] = best.expression
                results["hopf_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["hopf_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
