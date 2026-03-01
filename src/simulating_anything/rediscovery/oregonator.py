"""Oregonator rediscovery.

Targets:
- ODE: du/dt = (u*(1-u) - f*v*(u-q)/(u+q))/eps, dv/dt = u - v,
  dw/dt = kw*(u - w)  (via SINDy)
- Hopf bifurcation as f varies
- Oscillation period and amplitude vs f
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.oregonator import Oregonator
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    eps: float = 0.04,
    f: float = 1.0,
    q: float = 0.002,
    kw: float = 0.5,
    n_steps: int = 50000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses small dt to resolve the fast u variable (timescale ~ eps).
    """
    config = SimulationConfig(
        domain=Domain.OREGONATOR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "eps": eps, "f": f, "q": q, "kw": kw,
            "u_0": 0.5, "v_0": 0.5, "w_0": 0.5,
        },
    )
    sim = Oregonator(config)
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
        "w": states[:, 2],
        "eps": eps,
        "f": f,
        "q": q,
        "kw": kw,
    }


def generate_oscillation_data(
    n_f: int = 20,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Sweep f and measure period and amplitude.

    The Oregonator oscillates for a range of f values. As f changes,
    the oscillation period and amplitude vary, and eventually the
    system may undergo a Hopf bifurcation.
    """
    f_values = np.linspace(0.3, 3.0, n_f)
    periods = []
    amplitudes_u = []

    for i, f_val in enumerate(f_values):
        config = SimulationConfig(
            domain=Domain.OREGONATOR,
            dt=dt,
            n_steps=1000,
            parameters={
                "eps": 0.04, "f": f_val, "q": 0.002, "kw": 0.5,
                "u_0": 0.5, "v_0": 0.5, "w_0": 0.5,
            },
        )
        sim = Oregonator(config)
        sim.reset()
        period = sim.measure_period(n_periods=3)
        amp = sim.measure_amplitude(n_periods=3)
        periods.append(period)
        amplitudes_u.append(amp[0])

        if (i + 1) % 5 == 0:
            logger.info(f"  f={f_val:.3f}: period={period:.4f}, u_amp={amp[0]:.4f}")

    return {
        "f": f_values,
        "period": np.array(periods),
        "amplitude_u": np.array(amplitudes_u),
    }


def generate_bifurcation_data(
    n_f: int = 30,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Sweep f and measure oscillation amplitude to detect Hopf bifurcation.

    At the Hopf bifurcation, the limit cycle disappears and the system
    converges to the fixed point.
    """
    f_values = np.linspace(0.3, 4.0, n_f)
    amplitudes = []

    for i, f_val in enumerate(f_values):
        config = SimulationConfig(
            domain=Domain.OREGONATOR,
            dt=dt,
            n_steps=1000,
            parameters={
                "eps": 0.04, "f": f_val, "q": 0.002, "kw": 0.5,
                "u_0": 0.5, "v_0": 0.5, "w_0": 0.5,
            },
        )
        sim = Oregonator(config)
        sim.reset()

        # Skip transient (200 time units)
        transient_steps = int(200.0 / dt)
        for _ in range(transient_steps):
            sim.step()

        # Measure amplitude of u
        u_vals = []
        measure_steps = int(100.0 / dt)
        for _ in range(measure_steps):
            sim.step()
            u_vals.append(sim.observe()[0])

        amp = max(u_vals) - min(u_vals)
        amplitudes.append(amp)

        if (i + 1) % 10 == 0:
            logger.info(f"  f={f_val:.3f}: amplitude={amp:.6f}")

    return {
        "f": f_values,
        "amplitude": np.array(amplitudes),
    }


def run_oregonator_rediscovery(
    output_dir: str | Path = "output/rediscovery/oregonator",
    n_iterations: int = 40,
) -> dict:
    """Run Oregonator rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "oregonator",
        "targets": {
            "ode": "du/dt=(u*(1-u)-f*v*(u-q)/(u+q))/eps, dv/dt=u-v, dw/dt=kw*(u-w)",
            "hopf": "Hopf bifurcation as f varies",
            "period": "Oscillation period vs f",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at default parameters...")
    data = generate_trajectory_data(
        eps=0.04, f=1.0, q=0.002, kw=0.5,
        n_steps=50000, dt=0.001,
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.001,
            feature_names=["u", "v", "w"],
            threshold=0.05,
            poly_degree=2,
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
    logger.info("Part 2: Hopf bifurcation sweep over f...")
    bif_data = generate_bifurcation_data(n_f=25, dt=0.001)

    # Estimate f_c: find transition from oscillating to non-oscillating
    threshold = 0.01
    oscillating = bif_data["amplitude"] > threshold
    if np.any(oscillating) and np.any(~oscillating):
        transitions = np.where(np.diff(oscillating.astype(int)) == -1)[0]
        if len(transitions) > 0:
            idx = transitions[0]
            f_c_est = float(0.5 * (bif_data["f"][idx] + bif_data["f"][idx + 1]))
        else:
            f_c_est = float(bif_data["f"][np.argmax(~oscillating)])
        results["hopf_bifurcation"] = {
            "f_c_estimate": f_c_est,
            "n_oscillatory": int(np.sum(oscillating)),
            "n_total": int(len(oscillating)),
        }
        logger.info(f"  f_c estimate: {f_c_est:.4f}")
    else:
        results["hopf_bifurcation"] = {
            "note": "All f values oscillating or all stable",
            "n_oscillatory": int(np.sum(oscillating)),
        }

    # --- Part 3: Period vs f (PySR) ---
    logger.info("Part 3: Oscillation data and PySR period fit...")
    osc_data = generate_oscillation_data(n_f=20, dt=0.001)

    # Filter finite periods for PySR
    valid = np.isfinite(osc_data["period"])
    if np.sum(valid) > 3:
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = osc_data["f"][valid].reshape(-1, 1)
            y = osc_data["period"][valid]

            logger.info("  Running PySR: period = g(f)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["f_stoich"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "log", "square"],
                max_complexity=10,
                populations=20,
                population_size=40,
            )
            results["period_pysr"] = {
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
                results["period_pysr"]["best"] = best.expression
                results["period_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        except Exception as e:
            logger.warning(f"PySR failed: {e}")
            results["period_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f_out:
        json.dump(results, f_out, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "bifurcation.npz",
        f=bif_data["f"],
        amplitude=bif_data["amplitude"],
    )

    return results
