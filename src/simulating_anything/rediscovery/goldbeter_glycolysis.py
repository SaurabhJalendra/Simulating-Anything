"""Goldbeter glycolysis model rediscovery.

Targets:
- ODE: dx/dt = v - sigma*phi(x,y), dy/dt = q*sigma*phi(x,y) - k_s*y (via SINDy)
- Hopf bifurcation boundary in v (oscillation onset/offset)
- Oscillation period and amplitude vs input flux v
- Allosteric rate function phi verification
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.goldbeter_glycolysis import (
    GoldbeterGlycolysisSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    v: float = 0.36,
    sigma: float = 0.1,
    q: float = 1.0,
    k_s: float = 0.02,
    L: float = 5e6,
    x_0: float = 1.0,
    y_0: float = 1.0,
    dt: float = 0.1,
    n_steps: int = 10000,
) -> SimulationConfig:
    """Create a SimulationConfig for the Goldbeter glycolysis model."""
    return SimulationConfig(
        domain=Domain.GOLDBETER_GLYCOLYSIS,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "v": v, "sigma": sigma, "q": q, "k_s": k_s, "L": L,
            "x_0": x_0, "y_0": y_0,
        },
    )


def generate_ode_data(
    v: float = 0.36,
    sigma: float = 0.1,
    q: float = 1.0,
    k_s: float = 0.02,
    L: float = 5e6,
    n_steps: int = 10000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = _make_config(v=v, sigma=sigma, q=q, k_s=k_s, L=L, dt=dt, n_steps=n_steps)
    sim = GoldbeterGlycolysisSimulation(config)
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
        "v": v,
        "sigma": sigma,
        "q": q,
        "k_s": k_s,
        "L": L,
    }


def generate_bifurcation_data(
    n_v: int = 30,
    v_range: tuple[float, float] = (0.1, 1.0),
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep v (input flux) and measure y-amplitude to find Hopf bifurcation."""
    v_values = np.linspace(v_range[0], v_range[1], n_v)
    amplitudes = []

    for i, v in enumerate(v_values):
        config = _make_config(v=v, dt=dt, n_steps=1000)
        sim = GoldbeterGlycolysisSimulation(config)
        sim.reset()

        # Transient
        for _ in range(int(500 / dt)):
            sim.step()

        # Measure amplitude
        y_vals = []
        for _ in range(int(300 / dt)):
            sim.step()
            y_vals.append(sim.observe()[1])

        amp = max(y_vals) - min(y_vals)
        amplitudes.append(amp)

        if (i + 1) % 10 == 0:
            logger.info(f"  v={v:.3f}: amplitude={amp:.4f}")

    return {
        "v_values": v_values,
        "amplitude": np.array(amplitudes),
    }


def generate_sigma_sweep_data(
    n_sigma: int = 20,
    sigma_range: tuple[float, float] = (0.01, 0.5),
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep sigma and measure oscillation onset."""
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], n_sigma)
    amplitudes = []

    for i, sigma in enumerate(sigma_values):
        config = _make_config(sigma=sigma, dt=dt, n_steps=1000)
        sim = GoldbeterGlycolysisSimulation(config)
        sim.reset()

        # Transient
        for _ in range(int(500 / dt)):
            sim.step()

        # Measure amplitude
        y_vals = []
        for _ in range(int(300 / dt)):
            sim.step()
            y_vals.append(sim.observe()[1])

        amp = max(y_vals) - min(y_vals)
        amplitudes.append(amp)

        if (i + 1) % 5 == 0:
            logger.info(f"  sigma={sigma:.4f}: amplitude={amp:.4f}")

    return {
        "sigma_values": sigma_values,
        "amplitude": np.array(amplitudes),
    }


def generate_period_data(
    n_v: int = 20,
    v_range: tuple[float, float] = (0.2, 0.6),
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep v and measure oscillation period where oscillatory."""
    v_values = np.linspace(v_range[0], v_range[1], n_v)
    periods = []

    for i, v in enumerate(v_values):
        config = _make_config(v=v, dt=dt, n_steps=1000)
        sim = GoldbeterGlycolysisSimulation(config)
        sim.reset()
        period = sim.compute_period(n_transient=3000, n_measure=5000)
        periods.append(period)

        if (i + 1) % 5 == 0:
            logger.info(f"  v={v:.3f}: period={period:.2f}")

    return {
        "v_values": v_values,
        "period": np.array(periods),
    }


def run_goldbeter_glycolysis_rediscovery(
    output_dir: str | Path = "output/rediscovery/goldbeter_glycolysis",
    n_iterations: int = 40,
) -> dict:
    """Run Goldbeter glycolysis model rediscovery pipeline.

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iteration count.

    Returns:
        Results dictionary with all rediscovery findings.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "goldbeter_glycolysis",
        "targets": {
            "ode": "dx/dt = v - sigma*phi(x,y), dy/dt = q*sigma*phi(x,y) - k_s*y",
            "phi": "phi(x,y) = x*(1+x)*(1+y)^2 / (L + (1+x)^2*(1+y)^2)",
            "hopf": "Oscillation onset as v varies",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at default parameters...")
    data = generate_ode_data(n_steps=10000, dt=0.1)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.1,
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

    # --- Part 2: Hopf bifurcation via v sweep ---
    logger.info("Part 2: Hopf bifurcation sweep (v)...")
    bif_data = generate_bifurcation_data(n_v=30, v_range=(0.1, 1.0), dt=0.1)

    # Estimate v_c: first v where amplitude > threshold
    threshold = 0.1
    above = bif_data["amplitude"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        v_c_est = float(bif_data["v_values"][max(0, idx - 1)])
        results["hopf_bifurcation_v"] = {
            "v_c_estimate": v_c_est,
            "n_oscillatory": int(np.sum(above)),
            "max_amplitude": float(np.max(bif_data["amplitude"])),
        }
        logger.info(f"  v_c estimate: {v_c_est:.4f}")

    # --- Part 3: sigma sweep ---
    logger.info("Part 3: Sigma sweep for oscillation onset...")
    sigma_data = generate_sigma_sweep_data(n_sigma=20, sigma_range=(0.01, 0.5), dt=0.1)

    above_sigma = sigma_data["amplitude"] > threshold
    if np.any(above_sigma):
        idx_s = np.argmax(above_sigma)
        sigma_c_est = float(sigma_data["sigma_values"][max(0, idx_s - 1)])
        results["sigma_sweep"] = {
            "sigma_c_estimate": sigma_c_est,
            "n_oscillatory": int(np.sum(above_sigma)),
            "max_amplitude": float(np.max(sigma_data["amplitude"])),
        }
        logger.info(f"  sigma_c estimate: {sigma_c_est:.4f}")
    else:
        results["sigma_sweep"] = {
            "n_oscillatory": 0,
            "note": "No oscillation detected in sigma range",
        }

    # --- Part 4: Allosteric rate function verification ---
    logger.info("Part 4: Allosteric rate function verification...")
    config = _make_config()
    sim = GoldbeterGlycolysisSimulation(config)

    # phi should be bounded in [0, 1)
    test_points = [(0.0, 0.0), (1.0, 1.0), (10.0, 10.0), (100.0, 100.0)]
    phi_values = []
    for x, y in test_points:
        phi_val = sim.phi(x, y)
        phi_values.append({"x": x, "y": y, "phi": float(phi_val)})

    results["phi_verification"] = {
        "test_points": phi_values,
        "phi_at_origin": float(sim.phi(0.0, 0.0)),
        "phi_bounded": all(0.0 <= p["phi"] < 1.0 for p in phi_values),
    }
    logger.info(f"  phi(0,0) = {sim.phi(0.0, 0.0):.6f}")
    logger.info(f"  phi(1,1) = {sim.phi(1.0, 1.0):.6f}")

    # --- Part 5: Period measurement ---
    logger.info("Part 5: Period vs v sweep...")
    try:
        period_data = generate_period_data(n_v=15, v_range=(0.2, 0.6), dt=0.1)
        finite = np.isfinite(period_data["period"])
        if np.sum(finite) > 3:
            results["period"] = {
                "n_measured": int(np.sum(finite)),
                "v_range": [
                    float(period_data["v_values"].min()),
                    float(period_data["v_values"].max()),
                ],
                "period_range": [
                    float(np.min(period_data["period"][finite])),
                    float(np.max(period_data["period"][finite])),
                ],
            }
            logger.info(f"  Measured {np.sum(finite)} periods")
        else:
            results["period"] = {
                "n_measured": int(np.sum(finite)),
                "note": "Few oscillatory points found",
            }
    except Exception as e:
        logger.warning(f"Period measurement failed: {e}")
        results["period"] = {"error": str(e)}

    # --- Part 6: PySR on bifurcation data (amplitude vs v) ---
    logger.info("Part 6: PySR amplitude vs v fit...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        osc_mask = bif_data["amplitude"] > threshold
        if np.sum(osc_mask) > 5:
            X = bif_data["v_values"][osc_mask].reshape(-1, 1)
            y = bif_data["amplitude"][osc_mask]

            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["v_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square", "sqrt"],
                max_complexity=10,
                populations=20,
                population_size=40,
            )
            results["pysr_amplitude"] = {
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
                results["pysr_amplitude"]["best"] = best.expression
                results["pysr_amplitude"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_amplitude"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
