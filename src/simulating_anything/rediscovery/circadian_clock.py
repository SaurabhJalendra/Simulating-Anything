"""Circadian clock (Gonze-Goodwin model) rediscovery.

Targets:
- Circadian period ~24h verification with default parameters
- Hill coefficient effect: sweep n to find oscillation onset
- Phase portrait: M-Fc-Fn limit cycle in 3D
- ODE recovery via SINDy for the linear transport terms
- Parameter sensitivity: v_s on amplitude, k1/k2 on period
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.circadian_clock import CircadianClockSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use GOODWIN as the closest domain enum for circadian models
_DOMAIN = Domain.CIRCADIAN_CLOCK


def _make_config(dt: float = 0.1, n_steps: int = 3000, **params) -> SimulationConfig:
    """Helper to create a SimulationConfig for the circadian clock."""
    defaults = {
        "v_s": 1.6, "K_I": 1.0, "n": 4.0,
        "v_m": 0.505, "K_m": 0.5, "k_s": 0.5,
        "v_d": 1.4, "K_d": 0.13, "k1": 0.33, "k2": 0.43,
        "M_0": 0.5, "Fc_0": 0.5, "Fn_0": 0.5,
    }
    defaults.update(params)
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters=defaults,
    )


def generate_ode_data(
    n_steps: int = 10000,
    dt: float = 0.1,
    **params,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Args:
        n_steps: number of simulation steps.
        dt: timestep size.
        **params: override default circadian parameters.

    Returns:
        Dict with time, states, and individual variable arrays.
    """
    config = _make_config(dt=dt, n_steps=n_steps, **params)
    sim = CircadianClockSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "M": states_arr[:, 0],
        "Fc": states_arr[:, 1],
        "Fn": states_arr[:, 2],
    }


def generate_hill_sweep_data(
    n_range: tuple[float, float] = (1.0, 8.0),
    n_points: int = 25,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep Hill coefficient n and measure oscillation amplitude.

    Args:
        n_range: (min_n, max_n) range for the sweep.
        n_points: number of n values to test.
        dt: timestep.

    Returns:
        Dict with 'n_values' and 'amplitudes' arrays.
    """
    n_values = np.linspace(n_range[0], n_range[1], n_points)
    amplitudes = []

    for i, n_val in enumerate(n_values):
        config = _make_config(dt=dt, n=float(n_val))
        sim = CircadianClockSimulation(config)
        sim.reset()

        amps = sim.measure_amplitude(transient_steps=3000, measure_steps=3000)
        amplitudes.append(amps[0])  # M amplitude

        if (i + 1) % 10 == 0:
            logger.info(f"  n={n_val:.2f}: M_amplitude={amps[0]:.6f}")

    return {
        "n_values": n_values,
        "amplitudes": np.array(amplitudes),
    }


def generate_period_data(
    n_steps_per_point: int = 15000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Measure period at different v_s and k1/k2 values.

    Returns:
        Dict with parameter arrays and measured periods.
    """
    # Sweep v_s (transcription rate affects amplitude and period)
    v_s_values = np.linspace(0.5, 1.2, 15)
    periods_vs = []

    for v_s in v_s_values:
        config = _make_config(dt=dt, n_steps=n_steps_per_point, v_s=float(v_s))
        sim = CircadianClockSimulation(config)
        sim.reset()
        period = sim.measure_period(transient_steps=5000, measure_steps=8000)
        periods_vs.append(period)

    # Sweep k1/k2 ratio (nuclear transport affects period)
    k1_values = np.linspace(1.0, 3.0, 15)
    periods_k1 = []

    for k1 in k1_values:
        config = _make_config(dt=dt, n_steps=n_steps_per_point, k1=float(k1))
        sim = CircadianClockSimulation(config)
        sim.reset()
        period = sim.measure_period(transient_steps=5000, measure_steps=8000)
        periods_k1.append(period)

    return {
        "v_s_values": v_s_values,
        "periods_vs": np.array(periods_vs),
        "k1_values": k1_values,
        "periods_k1": np.array(periods_k1),
    }


def generate_phase_portrait_data(
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate limit cycle data in (M, Fc, Fn) space after transient.

    Returns:
        Dict with M, Fc, Fn arrays for the limit cycle orbit.
    """
    config = _make_config(dt=dt, n_steps=n_steps)
    sim = CircadianClockSimulation(config)
    sim.reset()

    # Skip transient
    for _ in range(5000):
        sim.step()

    # Collect limit cycle
    m_vals, fc_vals, fn_vals = [], [], []
    for _ in range(n_steps):
        sim.step()
        state = sim.observe()
        m_vals.append(state[0])
        fc_vals.append(state[1])
        fn_vals.append(state[2])

    return {
        "M": np.array(m_vals),
        "Fc": np.array(fc_vals),
        "Fn": np.array(fn_vals),
    }


def run_circadian_clock_rediscovery(
    output_dir: str | Path = "output/rediscovery/circadian_clock",
    n_iterations: int = 40,
) -> dict:
    """Run circadian clock (Gonze-Goodwin model) rediscovery pipeline.

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iteration count.

    Returns:
        Results dict with all discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "circadian_clock",
        "model": "Gonze-Goodwin simplified circadian oscillator",
        "targets": {
            "ode": (
                "dM/dt = v_s*K_I^n/(K_I^n+Fn^n) - v_m*M/(K_m+M), "
                "dFc/dt = k_s*M - v_d*Fc/(K_d+Fc) - k1*Fc + k2*Fn, "
                "dFn/dt = k1*Fc - k2*Fn"
            ),
            "circadian_period": "~24h with default parameters",
            "hill_onset": "oscillation requires n >= 2",
        },
    }

    # --- Part 1: Period verification (~24h) ---
    logger.info("Part 1: Circadian period verification...")
    config = _make_config(dt=0.1, n_steps=3000)
    sim = CircadianClockSimulation(config)
    sim.reset()
    period = sim.measure_period(transient_steps=5000, measure_steps=8000)

    results["period_verification"] = {
        "measured_period": float(period),
        "target_period": 24.0,
        "period_error_pct": (
            float(abs(period - 24.0) / 24.0 * 100) if np.isfinite(period) else None
        ),
    }
    logger.info(f"  Circadian period: {period:.2f}h (target: ~24h)")

    # --- Part 2: Hill coefficient sweep ---
    logger.info("Part 2: Hill coefficient sweep for oscillation onset...")
    sweep_data = generate_hill_sweep_data(n_range=(1.0, 8.0), n_points=25, dt=0.1)

    threshold = 0.01
    above = sweep_data["amplitudes"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        n_c_est = sweep_data["n_values"][max(0, idx - 1)]
        results["hill_onset"] = {
            "n_c_estimate": float(n_c_est),
            "n_tested": len(sweep_data["n_values"]),
            "n_oscillating": int(np.sum(above)),
        }
        logger.info(f"  Oscillation onset at n ~ {n_c_est:.2f}")
    else:
        results["hill_onset"] = {"error": "No oscillation detected in sweep"}

    # --- Part 3: SINDy ODE recovery ---
    logger.info("Part 3: SINDy ODE recovery...")
    data = generate_ode_data(n_steps=10000, dt=0.1)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.1,
            feature_names=["M", "Fc", "Fn"],
            threshold=0.01,
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

    # --- Part 4: Parameter sensitivity ---
    logger.info("Part 4: Parameter sensitivity (v_s, k1)...")
    try:
        period_data = generate_period_data(dt=0.1)

        finite_vs = np.isfinite(period_data["periods_vs"])
        finite_k1 = np.isfinite(period_data["periods_k1"])

        results["parameter_sensitivity"] = {
            "v_s_sweep": {
                "n_finite": int(np.sum(finite_vs)),
                "v_s_range": [float(period_data["v_s_values"].min()),
                              float(period_data["v_s_values"].max())],
            },
            "k1_sweep": {
                "n_finite": int(np.sum(finite_k1)),
                "k1_range": [float(period_data["k1_values"].min()),
                             float(period_data["k1_values"].max())],
            },
        }

        if np.sum(finite_vs) > 2:
            results["parameter_sensitivity"]["v_s_sweep"]["period_range"] = [
                float(np.min(period_data["periods_vs"][finite_vs])),
                float(np.max(period_data["periods_vs"][finite_vs])),
            ]

        if np.sum(finite_k1) > 2:
            results["parameter_sensitivity"]["k1_sweep"]["period_range"] = [
                float(np.min(period_data["periods_k1"][finite_k1])),
                float(np.max(period_data["periods_k1"][finite_k1])),
            ]

        logger.info(
            f"  v_s: {np.sum(finite_vs)} finite periods, "
            f"k1: {np.sum(finite_k1)} finite periods"
        )
    except Exception as e:
        logger.warning(f"Parameter sensitivity failed: {e}")
        results["parameter_sensitivity"] = {"error": str(e)}

    # --- Part 5: Phase portrait ---
    logger.info("Part 5: Phase portrait (limit cycle)...")
    try:
        phase_data = generate_phase_portrait_data(n_steps=3000, dt=0.1)
        results["phase_portrait"] = {
            "M_range": [float(phase_data["M"].min()), float(phase_data["M"].max())],
            "Fc_range": [float(phase_data["Fc"].min()), float(phase_data["Fc"].max())],
            "Fn_range": [float(phase_data["Fn"].min()), float(phase_data["Fn"].max())],
            "n_points": len(phase_data["M"]),
        }
        logger.info(
            f"  Limit cycle M: [{phase_data['M'].min():.3f}, {phase_data['M'].max():.3f}]"
        )
    except Exception as e:
        logger.warning(f"Phase portrait failed: {e}")
        results["phase_portrait"] = {"error": str(e)}

    # --- Part 6: PySR period as function of k1 ---
    logger.info("Part 6: PySR period vs k1...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        period_data = generate_period_data(dt=0.1)
        finite_k1 = np.isfinite(period_data["periods_k1"])

        if np.sum(finite_k1) > 5:
            X = period_data["k1_values"][finite_k1].reshape(-1, 1)
            y = period_data["periods_k1"][finite_k1]

            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["k1_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square", "sqrt", "log"],
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
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
