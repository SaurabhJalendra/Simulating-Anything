"""Delayed predator-prey rediscovery.

Targets:
- Critical delay tau_c where Hopf bifurcation destabilizes equilibrium
- Period scaling T(tau) for delay-induced oscillations
- Equilibrium (N*, P*) from the no-delay Rosenzweig-MacArthur model
- Optional PySR: T = f(tau) or amplitude = f(tau)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.delayed_predator_prey import (
    DelayedPredatorPreySimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    tau: float = 2.0,
    dt: float = 0.01,
    n_steps: int = 1000,
    **overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for the delayed predator-prey domain."""
    params: dict[str, float] = {
        "r": 1.0, "K": 3.0, "a": 0.5, "h": 0.1,
        "e": 0.6, "m": 0.4, "tau": tau,
        "N_0": 2.0, "P_0": 1.0,
    }
    params.update(overrides)
    return SimulationConfig(
        domain=Domain.DELAYED_PREDATOR_PREY,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


def generate_delay_sweep_data(
    n_tau: int = 30,
    tau_min: float = 0.0,
    tau_max: float = 5.0,
    dt: float = 0.01,
    n_steps: int = 50000,
    n_transient: int = 20000,
) -> dict[str, np.ndarray]:
    """Sweep tau and measure oscillation amplitude and period.

    Returns arrays of tau, amplitude, and period values.
    """
    tau_values = np.linspace(tau_min, tau_max, n_tau)
    amplitudes = []
    periods = []

    for i, tau_val in enumerate(tau_values):
        config = _make_config(tau=tau_val, dt=dt, n_steps=n_steps)
        sim = DelayedPredatorPreySimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(n_transient):
            sim.step()

        # Collect prey values
        N_vals = []
        for _ in range(n_steps):
            state = sim.step()
            N_vals.append(state[0])

        N_arr = np.array(N_vals)
        amp = float(np.max(N_arr) - np.min(N_arr))
        amplitudes.append(amp)

        # Period via FFT
        T = DelayedPredatorPreySimulation._fft_period(N_arr, dt)
        periods.append(T)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  tau={tau_val:.2f}: amplitude={amp:.4f}, period={T:.4f}"
            )

    return {
        "tau": tau_values,
        "amplitude": np.array(amplitudes),
        "period": np.array(periods),
    }


def generate_equilibrium_data(
    dt: float = 0.01,
    n_steps: int = 100000,
) -> dict[str, float]:
    """Run with tau=0 (Rosenzweig-MacArthur) and measure equilibrium.

    Compares final state to the analytical equilibrium.
    """
    config = _make_config(tau=0.0, dt=dt, n_steps=n_steps)
    sim = DelayedPredatorPreySimulation(config)
    sim.reset()

    # Run to steady state
    for _ in range(n_steps):
        sim.step()

    # Average over last section to get equilibrium
    N_vals, P_vals = [], []
    for _ in range(5000):
        state = sim.step()
        N_vals.append(state[0])
        P_vals.append(state[1])

    N_measured = float(np.mean(N_vals))
    P_measured = float(np.mean(P_vals))

    # Analytical equilibrium
    N_star, P_star = sim.find_equilibrium()

    return {
        "N_measured": N_measured,
        "P_measured": P_measured,
        "N_theory": N_star,
        "P_theory": P_star,
        "N_rel_error": abs(N_measured - N_star) / max(N_star, 1e-12),
        "P_rel_error": abs(P_measured - P_star) / max(P_star, 1e-12),
    }


def run_delayed_predator_prey_rediscovery(
    output_dir: str | Path = "output/rediscovery/delayed_predator_prey",
    n_iterations: int = 40,
) -> dict:
    """Run delayed predator-prey rediscovery pipeline.

    1. Verify equilibrium for tau=0 (Rosenzweig-MacArthur limit)
    2. Sweep tau to measure oscillation onset and period/amplitude
    3. Detect critical tau_c via Hopf bifurcation
    4. Optional PySR: fit T(tau) and amplitude(tau)

    Args:
        output_dir: Directory for results and data files.
        n_iterations: PySR iteration count.

    Returns:
        Dict with all rediscovery results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "delayed_predator_prey",
        "targets": {
            "equilibrium": "N* = m/(a*(e - m*h)), P* from prey eqn",
            "hopf_tau_c": "Critical delay for oscillation onset",
            "period_scaling": "T increases with tau",
            "no_delay_limit": "Rosenzweig-MacArthur model at tau=0",
        },
    }

    # --- Part 1: Equilibrium verification (tau=0) ---
    logger.info("Part 1: Equilibrium verification at tau=0...")
    eq_data = generate_equilibrium_data(dt=0.01, n_steps=100000)
    results["equilibrium"] = eq_data
    logger.info(
        f"  N*: measured={eq_data['N_measured']:.4f}, "
        f"theory={eq_data['N_theory']:.4f}, "
        f"error={eq_data['N_rel_error']:.6f}"
    )
    logger.info(
        f"  P*: measured={eq_data['P_measured']:.4f}, "
        f"theory={eq_data['P_theory']:.4f}, "
        f"error={eq_data['P_rel_error']:.6f}"
    )

    # --- Part 2: Delay sweep ---
    logger.info("Part 2: Delay sweep (tau = 0 to 5)...")
    sweep_data = generate_delay_sweep_data(
        n_tau=30, tau_min=0.0, tau_max=5.0,
        dt=0.01, n_steps=50000, n_transient=20000,
    )

    valid_period = (sweep_data["period"] > 0) & np.isfinite(sweep_data["period"])
    results["delay_sweep"] = {
        "n_tau_values": len(sweep_data["tau"]),
        "tau_range": [float(sweep_data["tau"][0]), float(sweep_data["tau"][-1])],
        "max_amplitude": float(np.max(sweep_data["amplitude"])),
        "n_oscillating": int(np.sum(sweep_data["amplitude"] > 0.05)),
        "period_range": [
            float(np.min(sweep_data["period"][valid_period]))
            if np.any(valid_period) else 0.0,
            float(np.max(sweep_data["period"][valid_period]))
            if np.any(valid_period) else 0.0,
        ],
    }

    # --- Part 3: Hopf bifurcation detection ---
    logger.info("Part 3: Hopf bifurcation detection...")
    config = _make_config(tau=0.0, dt=0.01, n_steps=1000)
    sim = DelayedPredatorPreySimulation(config)
    sim.reset()

    fine_tau = np.linspace(0.0, 5.0, 50)
    tau_c = sim.hopf_bifurcation_detect(
        fine_tau, n_steps=50000, n_transient=20000, amplitude_threshold=0.05,
    )
    results["hopf_bifurcation"] = {
        "tau_c_estimate": tau_c,
        "n_tau_tested": len(fine_tau),
        "amplitude_threshold": 0.05,
    }
    if tau_c is not None:
        logger.info(f"  Critical delay tau_c ~ {tau_c:.3f}")
    else:
        logger.info("  No Hopf bifurcation detected in range")

    # --- Part 4: PySR period fitting ---
    osc_mask = (sweep_data["amplitude"] > 0.05) & valid_period
    if np.sum(osc_mask) >= 5:
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = sweep_data["tau"][osc_mask].reshape(-1, 1)
            y = sweep_data["period"][osc_mask]

            logger.info("  Running PySR: T = f(tau)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["tau"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "log"],
                max_complexity=10,
                populations=15,
                population_size=30,
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
        except Exception as exc:
            logger.warning(f"PySR period fit failed: {exc}")
            results["period_pysr"] = {"error": str(exc)}

        # PySR amplitude fitting
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = sweep_data["tau"][osc_mask].reshape(-1, 1)
            y = sweep_data["amplitude"][osc_mask]

            logger.info("  Running PySR: amplitude = f(tau)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["tau"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt"],
                max_complexity=8,
                populations=15,
                population_size=30,
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
        except Exception as exc:
            logger.warning(f"PySR amplitude fit failed: {exc}")
            results["amplitude_pysr"] = {"error": str(exc)}
    else:
        logger.info("  Not enough oscillating points for PySR fitting")
        results["period_pysr"] = {"error": "Too few oscillating data points"}
        results["amplitude_pysr"] = {"error": "Too few oscillating data points"}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "delay_sweep.npz",
        tau=sweep_data["tau"],
        amplitude=sweep_data["amplitude"],
        period=sweep_data["period"],
    )

    return results
