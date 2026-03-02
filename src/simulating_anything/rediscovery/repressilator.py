"""Repressilator rediscovery.

Targets:
- ODE: dm_i/dt = -m_i + alpha/(1+p_j^n) + alpha0, dp_i/dt = beta*(m_i-p_i)
  (via SINDy)
- Oscillation onset: critical alpha for Hopf bifurcation
- Z3 cyclic symmetry: 120-degree phase shift between genes
- Period and amplitude dependence on alpha
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.repressilator import RepressilatorSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    alpha: float = 216.0,
    alpha0: float = 0.001,
    n: float = 2.0,
    beta: float = 5.0,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.REPRESSILATOR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha": alpha, "alpha0": alpha0, "n": n, "beta": beta,
            "m1_0": 1.0, "m2_0": 0.5, "m3_0": 0.2,
            "p1_0": 1.0, "p2_0": 0.5, "p3_0": 0.2,
        },
    )
    sim = RepressilatorSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "m1": states_arr[:, 0],
        "m2": states_arr[:, 1],
        "m3": states_arr[:, 2],
        "p1": states_arr[:, 3],
        "p2": states_arr[:, 4],
        "p3": states_arr[:, 5],
        "dt": dt,
        "alpha": alpha,
        "alpha0": alpha0,
        "n": n,
        "beta": beta,
    }


def generate_bifurcation_data(
    n_alpha: int = 30,
    alpha0: float = 0.001,
    n: float = 2.0,
    beta: float = 5.0,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Sweep alpha and measure oscillation amplitude to find onset."""
    alpha_values = np.linspace(1.0, 300.0, n_alpha)
    amplitudes = []

    for i, alpha in enumerate(alpha_values):
        config = SimulationConfig(
            domain=Domain.REPRESSILATOR,
            dt=dt,
            n_steps=1000,
            parameters={
                "alpha": alpha, "alpha0": alpha0, "n": n, "beta": beta,
                "m1_0": 1.0, "m2_0": 0.5, "m3_0": 0.2,
                "p1_0": 1.0, "p2_0": 0.5, "p3_0": 0.2,
            },
        )
        sim = RepressilatorSimulation(config)
        sim.reset()

        # Transient
        for _ in range(int(300 / dt)):
            sim.step()

        # Measure amplitude of p1
        p1_vals = []
        for _ in range(int(200 / dt)):
            sim.step()
            p1_vals.append(sim.observe()[3])

        amp = max(p1_vals) - min(p1_vals)
        amplitudes.append(amp)

        if (i + 1) % 10 == 0:
            logger.info(f"  alpha={alpha:.1f}: amplitude={amp:.4f}")

    return {
        "alpha": alpha_values,
        "amplitude": np.array(amplitudes),
        "alpha0": alpha0,
        "n": n,
        "beta": beta,
    }


def generate_period_data(
    n_alpha: int = 20,
    alpha0: float = 0.001,
    n: float = 2.0,
    beta: float = 5.0,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Sweep alpha above oscillation onset and measure periods."""
    alpha_values = np.linspace(50.0, 500.0, n_alpha)
    periods = []

    for i, alpha in enumerate(alpha_values):
        config = SimulationConfig(
            domain=Domain.REPRESSILATOR,
            dt=dt,
            n_steps=1000,
            parameters={
                "alpha": alpha, "alpha0": alpha0, "n": n, "beta": beta,
                "m1_0": 1.0, "m2_0": 0.5, "m3_0": 0.2,
                "p1_0": 1.0, "p2_0": 0.5, "p3_0": 0.2,
            },
        )
        sim = RepressilatorSimulation(config)
        sim.reset()
        period = sim.compute_period(n_transient=3000, n_measure=5000)
        periods.append(period)

        if (i + 1) % 5 == 0:
            logger.info(f"  alpha={alpha:.1f}: period={period:.4f}")

    return {
        "alpha": alpha_values,
        "period": np.array(periods),
    }


def run_repressilator_rediscovery(
    output_dir: str | Path = "output/rediscovery/repressilator",
    n_iterations: int = 40,
) -> dict:
    """Run Repressilator rediscovery pipeline.

    Steps:
    1. SINDy ODE recovery (6 coupled equations)
    2. Alpha sweep for oscillation onset (Hopf bifurcation)
    3. Period vs alpha measurement
    4. Z3 symmetry verification
    5. Fixed point verification
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "repressilator",
        "targets": {
            "ode": (
                "dm_i/dt = -m_i + alpha/(1+p_j^n) + alpha0, "
                "dp_i/dt = beta*(m_i - p_i)"
            ),
            "oscillation_onset": "Critical alpha for Hopf bifurcation",
            "z3_symmetry": "120-degree phase shift between genes",
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
            feature_names=["m1", "m2", "m3", "p1", "p2", "p3"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries[:10]
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

    # --- Part 2: Oscillation onset (Hopf bifurcation) ---
    logger.info("Part 2: Alpha sweep for oscillation onset...")
    bif_data = generate_bifurcation_data(n_alpha=30, dt=0.005)

    threshold = 0.5
    above = bif_data["amplitude"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        alpha_c_est = bif_data["alpha"][max(0, idx - 1)]
        results["oscillation_onset"] = {
            "alpha_c_estimate": float(alpha_c_est),
            "n_oscillatory": int(np.sum(above)),
            "alpha_range": [
                float(bif_data["alpha"][0]),
                float(bif_data["alpha"][-1]),
            ],
        }
        logger.info(f"  alpha_c estimate: {alpha_c_est:.2f}")
    else:
        results["oscillation_onset"] = {"oscillatory_found": False}
        logger.info("  No oscillations detected in alpha sweep")

    # PySR: amplitude as function of alpha
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        valid = bif_data["amplitude"] > 0.1
        if np.sum(valid) > 5:
            X = bif_data["alpha"][valid].reshape(-1, 1)
            y = bif_data["amplitude"][valid]

            logger.info("  Running PySR: amplitude = f(alpha)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["al_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "log"],
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
                results["amplitude_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["amplitude_pysr"] = {"error": str(e)}

    # --- Part 3: Period vs alpha ---
    logger.info("Part 3: Period vs alpha sweep...")
    try:
        period_data = generate_period_data(n_alpha=15, dt=0.005)
        finite = np.isfinite(period_data["period"])
        if np.sum(finite) > 3:
            results["period"] = {
                "n_measured": int(np.sum(finite)),
                "alpha_range": [
                    float(period_data["alpha"].min()),
                    float(period_data["alpha"].max()),
                ],
                "period_range": [
                    float(np.min(period_data["period"][finite])),
                    float(np.max(period_data["period"][finite])),
                ],
            }
            logger.info(
                f"  Measured {np.sum(finite)} periods in "
                f"alpha=[{period_data['alpha'].min():.1f}, "
                f"{period_data['alpha'].max():.1f}]"
            )
        else:
            results["period"] = {"n_measured": 0}
    except Exception as e:
        logger.warning(f"Period measurement failed: {e}")
        results["period"] = {"error": str(e)}

    # --- Part 4: Z3 symmetry verification ---
    logger.info("Part 4: Z3 symmetry verification...")
    config = SimulationConfig(
        domain=Domain.REPRESSILATOR,
        dt=0.005,
        n_steps=1000,
        parameters={
            "alpha": 216.0, "alpha0": 0.001, "n": 2.0, "beta": 5.0,
            "m1_0": 1.0, "m2_0": 0.5, "m3_0": 0.2,
            "p1_0": 1.0, "p2_0": 0.5, "p3_0": 0.2,
        },
    )
    sim = RepressilatorSimulation(config)
    sim.reset()
    z3_data = sim.verify_z3_symmetry(n_transient=5000, n_measure=5000)
    results["z3_symmetry"] = {
        "amplitudes": z3_data["amplitudes"],
        "amplitude_ratio_12": float(z3_data["amplitude_ratio_12"]),
        "amplitude_ratio_23": float(z3_data["amplitude_ratio_23"]),
        "amplitude_ratio_31": float(z3_data["amplitude_ratio_31"]),
        "phase_diffs": z3_data["phase_diffs"],
    }
    logger.info(
        f"  Amplitude ratios: "
        f"p1/p2={z3_data['amplitude_ratio_12']:.4f}, "
        f"p2/p3={z3_data['amplitude_ratio_23']:.4f}, "
        f"p3/p1={z3_data['amplitude_ratio_31']:.4f}"
    )

    # --- Part 5: Fixed point verification ---
    logger.info("Part 5: Symmetric fixed point verification...")
    fp = sim.symmetric_fixed_point
    dy = sim._derivatives(fp)
    results["fixed_point"] = {
        "m_star": float(fp[0]),
        "p_star": float(fp[3]),
        "derivative_at_fp": [float(d) for d in dy],
        "derivative_norm": float(np.linalg.norm(dy)),
    }
    logger.info(
        f"  Fixed point: m*=p*={fp[0]:.6f}, |f(x*)|={np.linalg.norm(dy):.2e}"
    )

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
