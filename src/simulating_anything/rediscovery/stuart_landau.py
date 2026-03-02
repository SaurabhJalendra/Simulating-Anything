"""Stuart-Landau oscillator rediscovery.

Targets:
- ODE recovery via SINDy: dx/dt = mu*x - omega*y - (x^2+y^2)*(x - beta*y)
- Hopf bifurcation at mu = 0
- Limit cycle amplitude: r = sqrt(mu) via PySR
- Effective frequency: omega_eff = omega - beta*mu
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.stuart_landau import StuartLandauSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    mu: float = 1.0,
    omega: float = 1.0,
    beta: float = 0.0,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.STUART_LANDAU,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "mu": mu, "omega": omega, "beta": beta,
            "x_0": 0.5, "y_0": 0.0,
        },
    )
    sim = StuartLandauSimulation(config)
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
        "y": states[:, 1],
        "mu": mu,
        "omega": omega,
        "beta": beta,
        "dt": dt,
    }


def generate_amplitude_data(
    n_mu: int = 25,
    omega: float = 1.0,
    beta: float = 0.0,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep mu > 0 and measure amplitude to verify r = sqrt(mu)."""
    mu_values = np.linspace(0.1, 5.0, n_mu)
    amplitudes = []
    frequencies = []

    for i, mu in enumerate(mu_values):
        config = SimulationConfig(
            domain=Domain.STUART_LANDAU,
            dt=dt,
            n_steps=1000,
            parameters={
                "mu": mu, "omega": omega, "beta": beta,
                "x_0": 0.1, "y_0": 0.0,
            },
        )
        sim = StuartLandauSimulation(config)
        sim.reset()

        amp = sim.measure_amplitude(n_periods=3)
        amplitudes.append(amp)

        sim.reset()
        freq = sim.compute_frequency(n_periods=3)
        frequencies.append(freq)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  mu={mu:.3f}: amplitude={amp:.4f} "
                f"(theory={np.sqrt(mu):.4f}), freq={freq:.4f}"
            )

    return {
        "mu": mu_values,
        "amplitude": np.array(amplitudes),
        "frequency": np.array(frequencies),
        "amplitude_theory": np.sqrt(mu_values),
        "frequency_theory": omega - beta * mu_values,
    }


def generate_hopf_data(
    n_mu: int = 30,
    omega: float = 1.0,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep mu across 0 to detect Hopf bifurcation."""
    mu_values = np.linspace(-1.0, 2.0, n_mu)
    amplitudes = []

    for i, mu in enumerate(mu_values):
        config = SimulationConfig(
            domain=Domain.STUART_LANDAU,
            dt=dt,
            n_steps=1000,
            parameters={
                "mu": mu, "omega": omega, "beta": 0.0,
                "x_0": 0.1, "y_0": 0.0,
            },
        )
        sim = StuartLandauSimulation(config)
        sim.reset()

        if mu > 0:
            amp = sim.measure_amplitude(n_periods=3)
        else:
            # Run to steady state, measure final amplitude (should decay to 0)
            for _ in range(int(100 / dt)):
                sim.step()
            amp = sim.compute_amplitude()

        amplitudes.append(amp)

        if (i + 1) % 10 == 0:
            logger.info(f"  mu={mu:.3f}: amplitude={amp:.4f}")

    return {
        "mu": mu_values,
        "amplitude": np.array(amplitudes),
    }


def run_stuart_landau_rediscovery(
    output_dir: str | Path = "output/rediscovery/stuart_landau",
    n_iterations: int = 40,
) -> dict:
    """Run Stuart-Landau rediscovery pipeline.

    1. SINDy ODE recovery
    2. Hopf bifurcation detection (mu_c ~ 0)
    3. PySR amplitude law: r = sqrt(mu)
    4. PySR frequency law: omega_eff = omega + beta*mu
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "stuart_landau",
        "targets": {
            "ode": "dz/dt = (mu+i*omega)*z - (1+i*beta)*|z|^2*z",
            "amplitude": "r = sqrt(mu) for mu > 0",
            "hopf": "mu_c = 0",
            "frequency": "omega_eff = omega - beta*mu",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at mu=1.0, omega=1.0, beta=0...")
    data = generate_ode_data(mu=1.0, omega=1.0, beta=0.0, n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=data["dt"],
            feature_names=["x", "y"],
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
            logger.info(
                f"  SINDy best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Hopf bifurcation detection ---
    logger.info("Part 2: Hopf bifurcation sweep...")
    hopf_data = generate_hopf_data(n_mu=30, omega=1.0, dt=0.005)

    # Estimate mu_c: first mu where amplitude > threshold
    threshold = 0.05
    above = hopf_data["amplitude"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        mu_c_est = float(hopf_data["mu"][max(0, idx - 1)])
        results["hopf_bifurcation"] = {
            "mu_c_estimate": mu_c_est,
            "mu_c_theory": 0.0,
            "absolute_error": abs(mu_c_est),
        }
        logger.info(f"  mu_c estimate: {mu_c_est:.4f} (theory: 0.0)")

    # --- Part 3: PySR amplitude = f(mu) ---
    logger.info("Part 3: Amplitude vs mu...")
    amp_data = generate_amplitude_data(n_mu=25, omega=1.0, beta=0.0, dt=0.005)

    valid = (amp_data["amplitude"] > 0.01) & np.isfinite(amp_data["amplitude"])
    results["amplitude_data"] = {
        "n_samples": int(np.sum(valid)),
        "mu_range": [float(amp_data["mu"].min()), float(amp_data["mu"].max())],
        "mean_error_vs_theory": float(np.mean(np.abs(
            amp_data["amplitude"][valid] - amp_data["amplitude_theory"][valid]
        ))),
    }

    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = amp_data["mu"][valid].reshape(-1, 1)
        y = amp_data["amplitude"][valid]

        logger.info("  Running PySR: r = f(mu)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["mu"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt"],
            max_complexity=8,
            populations=20,
            population_size=40,
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
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["amplitude_pysr"] = {"error": str(e)}

    # --- Part 4: Frequency with nonzero beta ---
    logger.info("Part 4: Frequency vs mu with beta=0.5...")
    freq_data = generate_amplitude_data(
        n_mu=20, omega=1.0, beta=0.5, dt=0.005,
    )

    valid_f = (freq_data["frequency"] > 0) & np.isfinite(freq_data["frequency"])
    if np.sum(valid_f) > 3:
        results["frequency_data"] = {
            "n_samples": int(np.sum(valid_f)),
            "mean_error_vs_theory": float(np.mean(np.abs(
                freq_data["frequency"][valid_f]
                - freq_data["frequency_theory"][valid_f]
            ))),
        }

        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = freq_data["mu"][valid_f].reshape(-1, 1)
            y = freq_data["frequency"][valid_f]

            logger.info("  Running PySR: omega_eff = f(mu)...")
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
            results["frequency_pysr"] = {
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
                results["frequency_pysr"]["best"] = best.expression
                results["frequency_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        except Exception as e:
            logger.warning(f"PySR failed: {e}")
            results["frequency_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "amplitude_data.npz",
        mu=amp_data["mu"],
        amplitude=amp_data["amplitude"],
        frequency=amp_data["frequency"],
    )

    return results
