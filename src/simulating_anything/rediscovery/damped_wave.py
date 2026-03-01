"""Damped wave equation 1D rediscovery.

Targets:
- Dispersion: omega_k = sqrt(c^2 * k^2 - gamma^2/4)
- Decay rate: all modes decay at rate gamma/2
- Wave speed: pulse propagates at speed c
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.damped_wave import DampedWave1D
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_sim(
    c: float = 1.0,
    gamma: float = 0.1,
    N: int = 64,
    L: float = 2 * np.pi,
    dt: float = 0.001,
    n_steps: int = 5000,
) -> DampedWave1D:
    """Create a DampedWave1D simulation with the given parameters."""
    config = SimulationConfig(
        domain=Domain.DAMPED_WAVE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "c": c,
            "gamma": gamma,
            "N": float(N),
            "L": L,
        },
    )
    return DampedWave1D(config)


def generate_dispersion_data(
    n_c: int = 20,
    n_steps: int = 5000,
    dt: float = 0.001,
    N: int = 64,
    gamma: float = 0.0,
) -> dict[str, np.ndarray]:
    """Measure mode frequencies vs wavenumber for varying wave speed c.

    With gamma=0 (undamped), the frequency of mode m is exactly omega = c*k
    where k = 2*pi*m/L. We excite mode 1 on [0, 2*pi] so k=1, and sweep c
    to measure omega = c*1 = c.

    Returns dict with arrays: c_values, omega_measured, omega_theory.
    """
    L = 2 * np.pi
    mode = 1
    k_mode = 2 * np.pi * mode / L  # = 1.0

    c_values = np.linspace(0.5, 5.0, n_c)
    omega_measured = []
    omega_theory = []

    for i, c_val in enumerate(c_values):
        sim = _make_sim(c=c_val, gamma=gamma, N=N, L=L, dt=dt, n_steps=n_steps)
        sim.init_type = "sine"
        sim.reset()

        # Track displacement at a fixed grid point (quarter domain)
        track_idx = N // 4
        positions = [sim.observe()[track_idx]]
        for _ in range(n_steps):
            sim.step()
            positions.append(sim.observe()[track_idx])

        positions = np.array(positions)

        # Find positive-going zero crossings
        crossings = []
        for j in range(1, len(positions)):
            if positions[j - 1] < 0 and positions[j] >= 0:
                frac = -positions[j - 1] / (positions[j] - positions[j - 1])
                crossings.append((j - 1 + frac) * dt)

        if len(crossings) >= 3:
            periods = np.diff(crossings)
            T = float(np.median(periods))
            omega = 2 * np.pi / T
        else:
            omega = float("nan")

        omega_measured.append(omega)
        theory = np.sqrt(c_val ** 2 * k_mode ** 2 - gamma ** 2 / 4)
        omega_theory.append(theory)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  c={c_val:.2f}: omega_measured={omega:.4f}, "
                f"theory={theory:.4f}"
            )

    return {
        "c_values": c_values,
        "omega_measured": np.array(omega_measured),
        "omega_theory": np.array(omega_theory),
        "k_mode": k_mode,
    }


def generate_decay_rate_data(
    n_gamma: int = 25,
    n_steps: int = 5000,
    dt: float = 0.001,
    N: int = 64,
    c: float = 1.0,
) -> dict[str, np.ndarray]:
    """Measure mode amplitude decay rate vs damping coefficient gamma.

    Each mode amplitude decays as exp(-gamma/2 * t), so the decay rate = gamma/2.
    We excite mode 1, measure amplitude at start and end, and compute
    decay_rate = -ln(a_f / a_0) / t.

    Returns dict with arrays: gamma_values, decay_measured, decay_theory.
    """
    L = 2 * np.pi
    mode = 1

    gamma_values = np.linspace(0.05, 2.0, n_gamma)
    decay_measured = []
    decay_theory = []

    for i, gam in enumerate(gamma_values):
        sim = _make_sim(c=c, gamma=gam, N=N, L=L, dt=dt, n_steps=n_steps)
        sim.init_type = "sine"
        sim.reset()

        # Measure amplitude of mode 1 at t=0
        u_hat_0 = np.fft.fft(sim.observe()[:N])
        a0 = np.abs(u_hat_0[mode])

        # Run simulation
        for _ in range(n_steps):
            sim.step()

        u_hat_f = np.fft.fft(sim.observe()[:N])
        af = np.abs(u_hat_f[mode])

        t_total = n_steps * dt
        if af > 1e-15 and a0 > 1e-15:
            rate = -np.log(af / a0) / t_total
        else:
            rate = float("inf")

        decay_measured.append(rate)
        decay_theory.append(gam / 2.0)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  gamma={gam:.3f}: decay_measured={rate:.4f}, "
                f"theory={gam / 2:.4f}"
            )

    return {
        "gamma_values": gamma_values,
        "decay_measured": np.array(decay_measured),
        "decay_theory": np.array(decay_theory),
    }


def generate_wave_speed_data(
    n_c: int = 25,
    n_steps: int = 3000,
    dt: float = 0.001,
    N: int = 128,
) -> dict[str, np.ndarray]:
    """Sweep wave speed c and measure pulse propagation speed.

    A Gaussian pulse in a periodic domain propagates at speed c.
    We detect the pulse peak position over time and measure the speed.

    Returns dict with arrays: c_values, speed_measured, speed_theory.
    """
    L = 2 * np.pi

    c_values = np.linspace(0.5, 5.0, n_c)
    speed_measured = []
    speed_theory = []

    for i, c_val in enumerate(c_values):
        sim = _make_sim(
            c=c_val, gamma=0.0, N=N, L=L, dt=dt, n_steps=n_steps,
        )
        sim.init_type = "gaussian"
        sim.reset()

        # The Gaussian splits into left- and right-traveling pulses.
        # Track the right-traveling peak after they separate.
        separation_steps = min(n_steps // 3, 500)
        for _ in range(separation_steps):
            sim.step()

        u = sim.observe()[:N]
        peak1_idx = int(np.argmax(u))
        pos1 = sim.x[peak1_idx]
        t1 = separation_steps * dt

        # Run more steps
        more_steps = min(n_steps // 3, 500)
        for _ in range(more_steps):
            sim.step()

        u = sim.observe()[:N]
        peak2_idx = int(np.argmax(u))
        pos2 = sim.x[peak2_idx]
        t2 = (separation_steps + more_steps) * dt

        # Account for periodic boundaries
        delta_x = pos2 - pos1
        if delta_x < -L / 2:
            delta_x += L
        elif delta_x > L / 2:
            delta_x -= L

        delta_t = t2 - t1
        if delta_t > 0 and abs(delta_x) > 1e-10:
            speed = abs(delta_x) / delta_t
        else:
            speed = float("nan")

        speed_measured.append(speed)
        speed_theory.append(c_val)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  c={c_val:.2f}: speed_measured={speed:.4f}, "
                f"theory={c_val:.4f}"
            )

    return {
        "c_values": c_values,
        "speed_measured": np.array(speed_measured),
        "speed_theory": np.array(speed_theory),
    }


def run_damped_wave_rediscovery(
    output_dir: str | Path = "output/rediscovery/damped_wave",
    n_iterations: int = 40,
) -> dict:
    """Run the full damped wave equation rediscovery.

    1. Decay rate: gamma/2 for all modes (PySR: rate = f(gamma))
    2. Dispersion: omega = sqrt(c^2*k^2 - gamma^2/4) (PySR: omega = f(c))
    3. Wave speed verification
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "damped_wave",
        "targets": {
            "decay_rate": "rate = gamma/2",
            "dispersion": "omega = sqrt(c^2*k^2 - gamma^2/4)",
            "wave_speed": "pulse speed = c",
        },
    }

    # --- Part 1: Decay rate ---
    logger.info("Part 1: Measuring mode decay rate vs gamma...")
    decay_data = generate_decay_rate_data(n_gamma=25, n_steps=5000, dt=0.001)

    valid = np.isfinite(decay_data["decay_measured"])
    if np.sum(valid) > 5:
        rel_err = (
            np.abs(
                decay_data["decay_measured"][valid]
                - decay_data["decay_theory"][valid]
            )
            / decay_data["decay_theory"][valid]
        )
        results["decay_rate_data"] = {
            "n_samples": int(np.sum(valid)),
            "mean_relative_error": float(np.mean(rel_err)),
            "correlation": float(
                np.corrcoef(
                    decay_data["decay_measured"][valid],
                    decay_data["decay_theory"][valid],
                )[0, 1]
            ),
        }
        logger.info(f"  Mean relative error: {np.mean(rel_err):.4%}")

    # PySR: decay_rate = f(gamma)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = decay_data["gamma_values"][valid].reshape(-1, 1)
        y = decay_data["decay_measured"][valid]

        logger.info("Running PySR: decay_rate = f(gamma)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["g_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt"],
            max_complexity=8,
            populations=15,
            population_size=30,
        )
        results["decay_rate_pysr"] = {
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
            results["decay_rate_pysr"]["best"] = best.expression
            results["decay_rate_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["decay_rate_pysr"] = {"error": str(e)}

    # --- Part 2: Dispersion relation ---
    logger.info("Part 2: Measuring dispersion omega = f(c)...")
    disp_data = generate_dispersion_data(
        n_c=20, n_steps=5000, dt=0.001, gamma=0.0,
    )

    valid_disp = np.isfinite(disp_data["omega_measured"])
    if np.sum(valid_disp) > 5:
        disp_err = (
            np.abs(
                disp_data["omega_measured"][valid_disp]
                - disp_data["omega_theory"][valid_disp]
            )
            / disp_data["omega_theory"][valid_disp]
        )
        results["dispersion_data"] = {
            "n_samples": int(np.sum(valid_disp)),
            "mean_relative_error": float(np.mean(disp_err)),
            "correlation": float(
                np.corrcoef(
                    disp_data["omega_measured"][valid_disp],
                    disp_data["omega_theory"][valid_disp],
                )[0, 1]
            ),
        }
        logger.info(f"  Mean relative error: {np.mean(disp_err):.4%}")

    # PySR: omega = f(c)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = disp_data["c_values"][valid_disp].reshape(-1, 1)
        y = disp_data["omega_measured"][valid_disp]

        logger.info("Running PySR: omega = f(c)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["c_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=8,
            populations=15,
            population_size=30,
        )
        results["dispersion_pysr"] = {
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
            results["dispersion_pysr"]["best"] = best.expression
            results["dispersion_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["dispersion_pysr"] = {"error": str(e)}

    # --- Part 3: Wave speed ---
    logger.info("Part 3: Wave speed verification...")
    speed_data = generate_wave_speed_data(n_c=20, n_steps=3000, dt=0.001)

    valid_speed = np.isfinite(speed_data["speed_measured"])
    if np.sum(valid_speed) > 5:
        speed_err = (
            np.abs(
                speed_data["speed_measured"][valid_speed]
                - speed_data["speed_theory"][valid_speed]
            )
            / speed_data["speed_theory"][valid_speed]
        )
        results["wave_speed_data"] = {
            "n_samples": int(np.sum(valid_speed)),
            "mean_relative_error": float(np.mean(speed_err)),
        }
        logger.info(f"  Mean relative error: {np.mean(speed_err):.4%}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
