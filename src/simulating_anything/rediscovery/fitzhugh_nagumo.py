"""FitzHugh-Nagumo rediscovery.

Targets:
- ODE: dv/dt = v - v^3/3 - w + I, dw/dt = eps*(v + a - b*w)  (via SINDy)
- f-I curve: firing frequency as a function of input current I
- Hopf bifurcation: critical current for oscillation onset
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.fitzhugh_nagumo import FitzHughNagumoSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    I: float = 0.5,
    n_steps: int = 10000,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.FITZHUGH_NAGUMO,
        dt=dt,
        n_steps=n_steps,
        parameters={"I": I, "a": 0.7, "b": 0.8, "eps": 0.08, "v_0": -1.0, "w_0": -0.5},
    )
    sim = FitzHughNagumoSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "v": states[:, 0],
        "w": states[:, 1],
        "I": I,
    }


def generate_fi_curve(
    n_I: int = 30,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Generate firing frequency vs input current (f-I curve)."""
    I_values = np.linspace(0.0, 1.5, n_I)
    frequencies = []

    for i, I in enumerate(I_values):
        config = SimulationConfig(
            domain=Domain.FITZHUGH_NAGUMO,
            dt=dt,
            n_steps=1000,
            parameters={"I": I, "a": 0.7, "b": 0.8, "eps": 0.08, "v_0": -1.0, "w_0": -0.5},
        )
        sim = FitzHughNagumoSimulation(config)
        sim.reset()
        freq = sim.measure_firing_frequency(n_spikes=5)
        frequencies.append(freq)

        if (i + 1) % 10 == 0:
            logger.info(f"  I={I:.3f}: freq={freq:.4f}")

    return {
        "I": I_values,
        "frequency": np.array(frequencies),
    }


def run_fitzhugh_nagumo_rediscovery(
    output_dir: str | Path = "output/rediscovery/fitzhugh_nagumo",
    n_iterations: int = 40,
) -> dict:
    """Run FitzHugh-Nagumo rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "fitzhugh_nagumo",
        "targets": {
            "ode": "dv/dt = v-v^3/3-w+I, dw/dt = eps*(v+a-b*w)",
            "fi_curve": "firing frequency vs input current",
            "hopf": "critical current for oscillation onset",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at I=0.5...")
    data = generate_ode_data(I=0.5, n_steps=20000, dt=0.02)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.02,
            feature_names=["v", "w"],
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
            logger.info(f"  SINDy best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: f-I curve ---
    logger.info("Part 2: f-I curve...")
    fi_data = generate_fi_curve(n_I=30, dt=0.02)

    # Find critical current (onset of oscillation)
    firing = fi_data["frequency"] > 0.001
    if np.any(firing):
        I_c_idx = np.argmax(firing)
        I_c = fi_data["I"][I_c_idx]
        results["fi_curve"] = {
            "I_critical": float(I_c),
            "max_frequency": float(np.max(fi_data["frequency"])),
            "n_oscillatory": int(np.sum(firing)),
        }
        logger.info(f"  Critical current: I_c ~ {I_c:.4f}")
        logger.info(f"  Max frequency: {np.max(fi_data['frequency']):.4f}")
    else:
        results["fi_curve"] = {"note": "No oscillations detected in range"}

    # PySR: find f(I) for I > I_c
    try:
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        if np.any(firing):
            X = fi_data["I"][firing].reshape(-1, 1)
            y = fi_data["frequency"][firing]

            if len(X) > 3:
                logger.info("  Running PySR: freq = f(I)...")
                discoveries = run_symbolic_regression(
                    X, y,
                    variable_names=["I"],
                    n_iterations=n_iterations,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sqrt", "square"],
                    max_complexity=10,
                    populations=15,
                    population_size=30,
                )
                results["fi_pysr"] = {
                    "n_discoveries": len(discoveries),
                    "discoveries": [
                        {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                        for d in discoveries[:5]
                    ],
                }
                if discoveries:
                    best = discoveries[0]
                    results["fi_pysr"]["best"] = best.expression
                    results["fi_pysr"]["best_r2"] = best.evidence.fit_r_squared
                    logger.info(
                        f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})"
                    )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["fi_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "fi_curve.npz",
        I=fi_data["I"],
        frequency=fi_data["frequency"],
    )

    return results
