"""Autocatalator rediscovery.

Targets:
- ODE: da/dt = mu*(kappa+c) - a*b^2 - a,
       db/dt = (a*b^2 + a - b)/sigma,
       dc/dt = (b - c)/delta  (via SINDy)
- Hopf bifurcation as mu varies (onset of oscillations)
- Fixed point: b* = mu*kappa/(1-mu), c*=b*, a*=b*/(b*^2+1)
- Period and amplitude of limit cycle oscillations
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.autocatalator import AutocatalatorSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    mu: float = 0.002,
    kappa: float = 65.0,
    sigma: float = 0.005,
    delta: float = 0.2,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.AUTOCATALATOR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "mu": mu, "kappa": kappa, "sigma": sigma, "delta": delta,
            "a_0": 0.5, "b_0": 1.0, "c_0": 0.5,
        },
    )
    sim = AutocatalatorSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "a": states_arr[:, 0],
        "b": states_arr[:, 1],
        "c": states_arr[:, 2],
        "dt": dt,
        "mu": mu,
        "kappa": kappa,
        "sigma": sigma,
        "delta": delta,
    }


def generate_bifurcation_data(
    n_mu: int = 30,
    kappa: float = 65.0,
    sigma: float = 0.005,
    delta: float = 0.2,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Sweep mu and measure amplitude to find Hopf bifurcation."""
    mu_values = np.linspace(0.0005, 0.01, n_mu)
    amplitudes = []

    for i, mu in enumerate(mu_values):
        config = SimulationConfig(
            domain=Domain.AUTOCATALATOR,
            dt=dt,
            n_steps=1000,
            parameters={
                "mu": mu, "kappa": kappa, "sigma": sigma, "delta": delta,
                "a_0": 0.5, "b_0": 1.0, "c_0": 0.5,
            },
        )
        sim = AutocatalatorSimulation(config)
        sim.reset()

        # Transient
        for _ in range(int(500 / dt)):
            sim.step()

        # Measure amplitude of b
        b_vals = []
        for _ in range(int(500 / dt)):
            sim.step()
            b_vals.append(sim.observe()[1])

        amp = max(b_vals) - min(b_vals)
        amplitudes.append(amp)

        if (i + 1) % 10 == 0:
            logger.info(f"  mu={mu:.5f}: amplitude={amp:.4f}")

    return {
        "mu": mu_values,
        "amplitude": np.array(amplitudes),
        "kappa": kappa,
        "sigma": sigma,
        "delta": delta,
    }


def generate_period_data(
    n_mu: int = 20,
    kappa: float = 65.0,
    sigma: float = 0.005,
    delta: float = 0.2,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Sweep mu above Hopf threshold and measure oscillation periods."""
    # Scan a range that should include oscillatory regime
    mu_values = np.linspace(0.001, 0.008, n_mu)
    periods = []

    for i, mu in enumerate(mu_values):
        config = SimulationConfig(
            domain=Domain.AUTOCATALATOR,
            dt=dt,
            n_steps=1000,
            parameters={
                "mu": mu, "kappa": kappa, "sigma": sigma, "delta": delta,
                "a_0": 0.5, "b_0": 1.0, "c_0": 0.5,
            },
        )
        sim = AutocatalatorSimulation(config)
        sim.reset()
        period = sim.measure_period(n_periods=5)
        periods.append(period)

        if (i + 1) % 5 == 0:
            logger.info(f"  mu={mu:.5f}: period={period:.4f}")

    return {
        "mu": mu_values,
        "period": np.array(periods),
    }


def run_autocatalator_rediscovery(
    output_dir: str | Path = "output/rediscovery/autocatalator",
    n_iterations: int = 40,
) -> dict:
    """Run Autocatalator rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "autocatalator",
        "targets": {
            "ode": (
                "da/dt = mu*(kappa+c) - a*b^2 - a, "
                "db/dt = (a*b^2+a-b)/sigma, "
                "dc/dt = (b-c)/delta"
            ),
            "hopf": "Oscillation onset as mu varies",
            "fixed_point": "b* = mu*kappa/(1-mu), c*=b*, a*=b*/(b*^2+1)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery...")
    data = generate_ode_data(n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["a", "b", "c"],
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

    # --- Part 2: Hopf bifurcation ---
    logger.info("Part 2: Hopf bifurcation sweep over mu...")
    bif_data = generate_bifurcation_data(n_mu=30, dt=0.005)

    # Estimate mu_c: first mu where amplitude > threshold
    threshold = 0.05
    above = bif_data["amplitude"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        mu_c_est = bif_data["mu"][max(0, idx - 1)]
        results["hopf_bifurcation"] = {
            "mu_c_estimate": float(mu_c_est),
            "n_oscillatory": int(np.sum(above)),
            "mu_range": [float(bif_data["mu"][0]), float(bif_data["mu"][-1])],
        }
        logger.info(f"  mu_c estimate: {mu_c_est:.6f}")
    else:
        results["hopf_bifurcation"] = {"oscillatory_found": False}
        logger.info("  No oscillations detected in mu sweep")

    # PySR: find mu_c or amplitude as function of mu
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use amplitude vs mu data for symbolic fit
        valid = bif_data["amplitude"] > 0.01
        if np.sum(valid) > 5:
            X = bif_data["mu"][valid].reshape(-1, 1)
            y = bif_data["amplitude"][valid]

            logger.info("  Running PySR: amplitude = f(mu)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["mu_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
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
                results["amplitude_pysr"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["amplitude_pysr"] = {"error": str(e)}

    # --- Part 3: Period measurement ---
    logger.info("Part 3: Period vs mu sweep...")
    try:
        period_data = generate_period_data(n_mu=15, dt=0.005)
        finite = np.isfinite(period_data["period"])
        if np.sum(finite) > 3:
            results["period"] = {
                "n_measured": int(np.sum(finite)),
                "mu_range": [
                    float(period_data["mu"].min()),
                    float(period_data["mu"].max()),
                ],
                "period_range": [
                    float(np.min(period_data["period"][finite])),
                    float(np.max(period_data["period"][finite])),
                ],
            }
            logger.info(
                f"  Measured {np.sum(finite)} periods in "
                f"mu=[{period_data['mu'].min():.5f}, "
                f"{period_data['mu'].max():.5f}]"
            )
        else:
            results["period"] = {"n_measured": 0}
    except Exception as e:
        logger.warning(f"Period measurement failed: {e}")
        results["period"] = {"error": str(e)}

    # --- Part 4: Fixed point verification ---
    logger.info("Part 4: Fixed point verification...")
    config = SimulationConfig(
        domain=Domain.AUTOCATALATOR,
        dt=0.01,
        n_steps=1000,
        parameters={
            "mu": 0.002, "kappa": 65.0, "sigma": 0.005, "delta": 0.2,
        },
    )
    sim = AutocatalatorSimulation(config)
    sim.reset()
    fps = sim.fixed_points
    if fps:
        fp = fps[0]
        dy = sim._derivatives(fp)
        results["fixed_point"] = {
            "a_star": float(fp[0]),
            "b_star": float(fp[1]),
            "c_star": float(fp[2]),
            "derivative_at_fp": [float(d) for d in dy],
            "derivative_norm": float(np.linalg.norm(dy)),
        }
        logger.info(
            f"  Fixed point: ({fp[0]:.6f}, {fp[1]:.6f}, {fp[2]:.6f}), "
            f"|f(x*)|={np.linalg.norm(dy):.2e}"
        )

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
