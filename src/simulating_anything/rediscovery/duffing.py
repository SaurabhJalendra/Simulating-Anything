"""Duffing oscillator rediscovery.

Targets:
- ODE: x'' + delta*x' + alpha*x + beta*x^3 = gamma_f*cos(omega*t)  (via SINDy)
- Chaos onset as gamma_f increases
- Amplitude-frequency relationship
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.duffing import DuffingOscillator
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    alpha: float = 1.0,
    beta: float = 1.0,
    delta: float = 0.2,
    gamma_f: float = 0.3,
    omega: float = 1.0,
    n_steps: int = 10000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses moderate forcing so the trajectory is periodic (not chaotic),
    which gives SINDy clean data to recover the ODE coefficients.
    """
    config = SimulationConfig(
        domain=Domain.DUFFING,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "gamma_f": gamma_f,
            "omega": omega,
            "x_0": 0.5,
            "v_0": 0.0,
        },
    )
    sim = DuffingOscillator(config)
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
        "v": states[:, 1],
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "gamma_f": gamma_f,
        "omega": omega,
        "dt": dt,
    }


def generate_chaos_sweep(
    n_gamma: int = 30,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep gamma_f (forcing amplitude) to detect chaos vs periodic behavior.

    For each gamma_f value, compute a rough measure of trajectory complexity
    by checking the spread of Poincare section points (stroboscopic map at
    the forcing period T = 2*pi/omega).
    """
    gamma_values = np.linspace(0.0, 0.5, n_gamma)
    omega = 1.0
    T_force = 2 * np.pi / omega
    n_transient = int(100 * T_force / dt)
    n_poincare = 200
    poincare_spread = []

    for i, gf in enumerate(gamma_values):
        config = SimulationConfig(
            domain=Domain.DUFFING,
            dt=dt,
            n_steps=1000,
            parameters={
                "alpha": 1.0,
                "beta": 1.0,
                "delta": 0.2,
                "gamma_f": gf,
                "omega": omega,
                "x_0": 0.5,
                "v_0": 0.0,
            },
        )
        sim = DuffingOscillator(config)
        sim.reset()

        # Skip transient
        for _ in range(n_transient):
            sim.step()

        # Collect Poincare section points (stroboscopic at T_force)
        steps_per_period = int(round(T_force / dt))
        x_poincare = []
        for _ in range(n_poincare):
            for _ in range(steps_per_period):
                sim.step()
            x_poincare.append(sim.observe()[0])

        x_poincare = np.array(x_poincare)
        # Spread: standard deviation of Poincare points
        # Low spread = periodic, high spread = chaotic
        spread = float(np.std(x_poincare))
        poincare_spread.append(spread)

        if (i + 1) % 10 == 0:
            logger.info(f"  gamma_f={gf:.3f}: Poincare spread={spread:.6f}")

    return {
        "gamma_f": gamma_values,
        "poincare_spread": np.array(poincare_spread),
        "omega": omega,
    }


def generate_amplitude_frequency_data(
    n_omega: int = 25,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep forcing frequency omega and measure steady-state amplitude.

    This traces out the amplitude-frequency response curve, which for the
    Duffing oscillator bends to the right (hardening spring, beta > 0).
    """
    omega_values = np.linspace(0.3, 2.5, n_omega)
    amplitudes = []

    for i, omega in enumerate(omega_values):
        config = SimulationConfig(
            domain=Domain.DUFFING,
            dt=dt,
            n_steps=1000,
            parameters={
                "alpha": 1.0,
                "beta": 1.0,
                "delta": 0.2,
                "gamma_f": 0.3,
                "omega": omega,
                "x_0": 0.5,
                "v_0": 0.0,
            },
        )
        sim = DuffingOscillator(config)
        sim.reset()

        # Transient
        T_est = 2 * np.pi / omega
        transient_steps = max(int(100 / dt), int(30 * T_est / dt))
        for _ in range(transient_steps):
            sim.step()

        # Measure peak amplitude
        x_max = 0.0
        measure_steps = int(20 * T_est / dt)
        for _ in range(measure_steps):
            sim.step()
            x_max = max(x_max, abs(sim.observe()[0]))

        amplitudes.append(x_max)

        if (i + 1) % 10 == 0:
            logger.info(f"  omega={omega:.3f}: A={x_max:.4f}")

    return {
        "omega": omega_values,
        "amplitude": np.array(amplitudes),
    }


def run_duffing_rediscovery(
    output_dir: str | Path = "output/rediscovery/duffing",
    n_iterations: int = 40,
) -> dict:
    """Run Duffing oscillator rediscovery pipeline.

    1. Generate trajectory for SINDy ODE recovery (unforced case for clean
       coefficient identification)
    2. Sweep gamma_f to detect chaos vs periodic behavior
    3. Measure amplitude-frequency response curve
    4. Run PySR on amplitude-frequency data

    Args:
        output_dir: Directory to save results.
        n_iterations: Number of PySR iterations.

    Returns:
        Results dict with all discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "duffing",
        "targets": {
            "ode": "x'' + delta*x' + alpha*x + beta*x^3 = gamma_f*cos(omega*t)",
            "chaos": "Chaos onset with increasing gamma_f",
            "amplitude_freq": "Nonlinear amplitude-frequency response",
        },
    }

    # --- Part 1: SINDy ODE recovery (unforced) ---
    # Use gamma_f=0 so SINDy can cleanly identify alpha, beta, delta
    logger.info("Part 1: SINDy ODE recovery (unforced, alpha=1, beta=1, delta=0.2)...")
    data = generate_ode_data(
        alpha=1.0, beta=1.0, delta=0.2, gamma_f=0.0,
        n_steps=10000, dt=0.005,
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=data["dt"],
            feature_names=["x", "v"],
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

    # --- Part 2: Chaos sweep ---
    logger.info("Part 2: Chaos sweep (gamma_f from 0 to 0.5)...")
    chaos_data = generate_chaos_sweep(n_gamma=30, dt=0.005)

    # Detect chaos onset: first gamma_f where Poincare spread exceeds threshold
    chaos_threshold = 0.01
    above = chaos_data["poincare_spread"] > chaos_threshold
    if np.any(above):
        idx = np.argmax(above)
        gamma_c = float(chaos_data["gamma_f"][idx])
    else:
        gamma_c = float("inf")

    results["chaos_sweep"] = {
        "n_gamma": len(chaos_data["gamma_f"]),
        "gamma_range": [
            float(chaos_data["gamma_f"].min()),
            float(chaos_data["gamma_f"].max()),
        ],
        "max_spread": float(np.max(chaos_data["poincare_spread"])),
        "gamma_c_estimate": gamma_c,
    }
    logger.info(f"  Chaos onset estimate: gamma_f ~ {gamma_c:.3f}")

    # --- Part 3: Amplitude-frequency response ---
    logger.info("Part 3: Amplitude-frequency response...")
    af_data = generate_amplitude_frequency_data(n_omega=25, dt=0.005)

    valid = np.isfinite(af_data["amplitude"]) & (af_data["amplitude"] > 0)
    results["amplitude_frequency"] = {
        "n_samples": int(np.sum(valid)),
        "omega_range": [
            float(af_data["omega"].min()),
            float(af_data["omega"].max()),
        ],
        "max_amplitude": float(np.max(af_data["amplitude"][valid])),
        "resonant_omega": float(
            af_data["omega"][np.argmax(af_data["amplitude"])]
        ),
    }
    logger.info(
        f"  Resonant omega: {results['amplitude_frequency']['resonant_omega']:.3f}, "
        f"max A: {results['amplitude_frequency']['max_amplitude']:.4f}"
    )

    # PySR: find A(omega)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = af_data["omega"][valid].reshape(-1, 1)
        y = af_data["amplitude"][valid]

        logger.info("  Running PySR: A = f(omega)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["w"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=12,
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

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "chaos_data.npz",
        gamma_f=chaos_data["gamma_f"],
        poincare_spread=chaos_data["poincare_spread"],
    )
    np.savez(
        output_path / "amplitude_frequency.npz",
        omega=af_data["omega"],
        amplitude=af_data["amplitude"],
    )

    return results
