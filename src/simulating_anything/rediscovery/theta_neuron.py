"""Theta neuron (Ermentrout-Kopell) rediscovery.

Targets:
- f-I curve: f = sqrt(I) / pi for I > 0 (via PySR)
- SNIC bifurcation at I_c = 0
- ODE recovery: dtheta/dt = 1 - cos(theta) + (1 + cos(theta)) * I (via SINDy)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.theta_neuron import ThetaNeuronSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    I: float = 0.5,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses a superthreshold current (I > 0) so theta evolves continuously.
    """
    config = SimulationConfig(
        domain=Domain.THETA_NEURON,  # placeholder
        dt=dt,
        n_steps=n_steps,
        parameters={"I": I, "theta_0": 0.0},
    )
    sim = ThetaNeuronSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "theta": states[:, 0],
        "I": I,
        "dt": dt,
    }


def generate_fi_curve(
    n_I: int = 30,
    dt: float = 0.01,
    t_max: float = 50.0,
    transient: float = 10.0,
) -> dict[str, np.ndarray]:
    """Generate firing frequency vs input current (f-I curve).

    Sweeps I from -0.5 to 3.0, measuring frequency for each value.
    """
    I_values = np.linspace(-0.5, 3.0, n_I)
    frequencies = []
    freq_theory = []

    for i, I in enumerate(I_values):
        config = SimulationConfig(
            domain=Domain.THETA_NEURON,  # placeholder
            dt=dt,
            n_steps=int(t_max / dt),
            parameters={"I": I, "theta_0": 0.0},
        )
        sim = ThetaNeuronSimulation(config)
        sim.reset()
        freq = sim.measure_frequency(t_max=t_max, transient=transient)
        frequencies.append(freq)
        freq_theory.append(ThetaNeuronSimulation.analytical_frequency(I))

        if (i + 1) % 10 == 0:
            logger.info(
                f"  I={I:.3f}: freq={freq:.4f} (theory={freq_theory[-1]:.4f})"
            )

    return {
        "I": I_values,
        "frequency": np.array(frequencies),
        "frequency_theory": np.array(freq_theory),
    }


def generate_bifurcation_data(
    n_I: int = 40,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep I across the SNIC bifurcation at I_c = 0.

    For I < 0, measures resting theta (fixed point).
    For I > 0, measures firing frequency.
    """
    I_values = np.linspace(-1.0, 1.0, n_I)
    resting_theta = []
    frequencies = []

    for i, I in enumerate(I_values):
        config = SimulationConfig(
            domain=Domain.THETA_NEURON,  # placeholder
            dt=dt,
            n_steps=5000,
            parameters={"I": I, "theta_0": 0.0},
        )
        sim = ThetaNeuronSimulation(config)
        sim.reset()

        if I < 0:
            # Run to steady state
            for _ in range(5000):
                sim.step()
            resting_theta.append(float(sim.observe()[0]))
            frequencies.append(0.0)
        else:
            freq = sim.measure_frequency(t_max=50.0, transient=10.0)
            frequencies.append(freq)
            resting_theta.append(float("nan"))

        if (i + 1) % 10 == 0:
            logger.info(f"  I={I:.3f}: freq={frequencies[-1]:.4f}")

    return {
        "I": I_values,
        "frequency": np.array(frequencies),
        "resting_theta": np.array(resting_theta),
    }


def run_theta_neuron_rediscovery(
    output_dir: str | Path = "output/rediscovery/theta_neuron",
    n_iterations: int = 40,
) -> dict:
    """Run theta neuron rediscovery pipeline.

    1. SINDy ODE recovery
    2. SNIC bifurcation detection (I_c ~ 0)
    3. PySR f-I curve: f = sqrt(I) / pi
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "theta_neuron",
        "targets": {
            "ode": "dtheta/dt = 1 - cos(theta) + (1 + cos(theta)) * I",
            "fi_curve": "f = sqrt(I) / pi for I > 0",
            "snic": "SNIC bifurcation at I_c = 0",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at I=0.5...")
    data = generate_ode_data(I=0.5, n_steps=20000, dt=0.01)

    # Build feature matrix with cos(theta) and sin(theta) for SINDy
    # The ODE is: dtheta/dt = 1 - cos(theta) + (1+cos(theta))*I
    #           = (1+I) - (1-I)*cos(theta)
    # SINDy needs custom features including cos(theta)
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        # For SINDy, use theta as state. The library will add polynomial terms;
        # the trig structure is hard for standard SINDy. We try anyway.
        sindy_discoveries = run_sindy(
            data["states"],
            dt=data["dt"],
            feature_names=["theta"],
            threshold=0.01,
            poly_degree=5,
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

    # --- Part 2: SNIC bifurcation detection ---
    logger.info("Part 2: SNIC bifurcation sweep...")
    bif_data = generate_bifurcation_data(n_I=40, dt=0.01)

    # Find I_c: first I where frequency > 0
    firing = bif_data["frequency"] > 0.001
    if np.any(firing):
        I_c_idx = np.argmax(firing)
        I_c = bif_data["I"][I_c_idx]
        results["snic_bifurcation"] = {
            "I_c_estimate": float(I_c),
            "I_c_theory": 0.0,
            "absolute_error": abs(float(I_c)),
        }
        logger.info(f"  I_c estimate: {I_c:.4f} (theory: 0.0)")
    else:
        results["snic_bifurcation"] = {"note": "No spiking detected in range"}

    # --- Part 3: PySR f-I curve ---
    logger.info("Part 3: f-I curve...")
    fi_data = generate_fi_curve(n_I=30, dt=0.01, t_max=50.0, transient=10.0)

    # Compute accuracy vs theory for I > 0
    superthreshold = fi_data["I"] > 0.01
    if np.any(superthreshold):
        meas = fi_data["frequency"][superthreshold]
        theo = fi_data["frequency_theory"][superthreshold]
        valid = (meas > 0) & np.isfinite(meas)
        if np.any(valid):
            rel_errors = np.abs(meas[valid] - theo[valid]) / theo[valid]
            results["fi_curve"] = {
                "n_superthreshold": int(np.sum(superthreshold)),
                "n_spiking": int(np.sum(valid)),
                "mean_relative_error": float(np.mean(rel_errors)),
                "max_frequency": float(np.max(meas)),
            }
            logger.info(
                f"  Mean relative error vs theory: {np.mean(rel_errors):.4f}"
            )

    # PySR: find f(I) for I > 0
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        supra = (fi_data["I"] > 0.05) & (fi_data["frequency"] > 0.001)
        if np.sum(supra) > 3:
            X = fi_data["I"][supra].reshape(-1, 1)
            y = fi_data["frequency"][supra]

            logger.info("  Running PySR: freq = f(I)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["I_ext"],
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
                    {
                        "expression": d.expression,
                        "r_squared": d.evidence.fit_r_squared,
                    }
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["fi_pysr"]["best"] = best.expression
                results["fi_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
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
        frequency_theory=fi_data["frequency_theory"],
    )

    return results
