"""Harmonic oscillator rediscovery.

Targets:
- Natural frequency: omega_0 = sqrt(k/m)
- Period: T = 2*pi/omega_0 = 2*pi*sqrt(m/k)
- Damping ratio from amplitude decay
- SINDy recovery of x'' + 2*zeta*omega_0*x' + omega_0^2*x = 0
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.harmonic_oscillator import DampedHarmonicOscillator
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_frequency_data(
    n_samples: int = 200,
    n_steps: int = 10000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate undamped oscillator data to study omega = sqrt(k/m).

    Vary k and m, measure period from zero crossings, compute frequency.
    """
    rng = np.random.default_rng(42)

    all_k = []
    all_m = []
    all_T_measured = []
    all_T_theory = []
    all_omega_measured = []
    all_omega_theory = []

    for i in range(n_samples):
        k = rng.uniform(0.5, 20.0)
        m = rng.uniform(0.5, 5.0)

        config = SimulationConfig(
            domain=Domain.HARMONIC_OSCILLATOR,
            dt=dt,
            n_steps=n_steps,
            parameters={"k": k, "m": m, "c": 0.0, "x_0": 1.0, "v_0": 0.0},
        )
        sim = DampedHarmonicOscillator(config)
        sim.reset()

        # Collect x positions
        positions = [sim.observe()[0]]
        for _ in range(n_steps):
            state = sim.step()
            positions.append(state[0])

        positions = np.array(positions)

        # Find period from positive-going zero crossings
        crossings = []
        for j in range(1, len(positions)):
            if positions[j - 1] < 0 and positions[j] >= 0:
                frac = -positions[j - 1] / (positions[j] - positions[j - 1])
                crossings.append((j - 1 + frac) * dt)

        if len(crossings) >= 3:
            periods = np.diff(crossings)
            T_measured = float(np.median(periods))
            T_theory = 2 * np.pi / np.sqrt(k / m)

            all_k.append(k)
            all_m.append(m)
            all_T_measured.append(T_measured)
            all_T_theory.append(T_theory)
            all_omega_measured.append(2 * np.pi / T_measured)
            all_omega_theory.append(np.sqrt(k / m))

        if (i + 1) % 50 == 0:
            logger.info(f"  Frequency measurement {i + 1}/{n_samples}")

    return {
        "k": np.array(all_k),
        "m": np.array(all_m),
        "T_measured": np.array(all_T_measured),
        "T_theory": np.array(all_T_theory),
        "omega_measured": np.array(all_omega_measured),
        "omega_theory": np.array(all_omega_theory),
    }


def generate_damping_data(
    n_samples: int = 100,
    n_steps: int = 20000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate damped oscillator data to study amplitude decay.

    Vary damping coefficient c, measure decay rate from peak envelope.
    """
    rng = np.random.default_rng(42)

    all_k = []
    all_m = []
    all_c = []
    all_zeta = []
    all_decay_rate_measured = []
    all_decay_rate_theory = []

    for i in range(n_samples):
        k = rng.uniform(1.0, 10.0)
        m = rng.uniform(0.5, 3.0)
        # Underdamped: zeta < 1, so c < 2*sqrt(k*m)
        c_max = 2 * np.sqrt(k * m) * 0.9  # Stay well underdamped
        c = rng.uniform(0.01, c_max)
        zeta = c / (2 * np.sqrt(k * m))

        config = SimulationConfig(
            domain=Domain.HARMONIC_OSCILLATOR,
            dt=dt,
            n_steps=n_steps,
            parameters={"k": k, "m": m, "c": c, "x_0": 1.0, "v_0": 0.0},
        )
        sim = DampedHarmonicOscillator(config)
        sim.reset()

        positions = [sim.observe()[0]]
        for _ in range(n_steps):
            state = sim.step()
            positions.append(state[0])

        positions = np.array(positions)

        # Find peaks (local maxima)
        peaks = []
        peak_times = []
        for j in range(1, len(positions) - 1):
            if positions[j] > positions[j - 1] and positions[j] > positions[j + 1]:
                if positions[j] > 0.01:  # Only significant peaks
                    peaks.append(positions[j])
                    peak_times.append(j * dt)

        if len(peaks) >= 3:
            # Fit log(peak) = -decay_rate * t + const
            peaks = np.array(peaks)
            peak_times = np.array(peak_times)
            log_peaks = np.log(peaks)

            # Linear regression
            A = np.column_stack([peak_times, np.ones(len(peak_times))])
            coeffs = np.linalg.lstsq(A, log_peaks, rcond=None)[0]
            decay_rate_measured = -coeffs[0]  # Should be zeta * omega_0

            omega_0 = np.sqrt(k / m)
            decay_rate_theory = zeta * omega_0

            all_k.append(k)
            all_m.append(m)
            all_c.append(c)
            all_zeta.append(zeta)
            all_decay_rate_measured.append(decay_rate_measured)
            all_decay_rate_theory.append(decay_rate_theory)

        if (i + 1) % 25 == 0:
            logger.info(f"  Damping measurement {i + 1}/{n_samples}")

    return {
        "k": np.array(all_k),
        "m": np.array(all_m),
        "c": np.array(all_c),
        "zeta": np.array(all_zeta),
        "decay_rate_measured": np.array(all_decay_rate_measured),
        "decay_rate_theory": np.array(all_decay_rate_theory),
    }


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate a damped oscillator trajectory for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.HARMONIC_OSCILLATOR,
        dt=dt,
        n_steps=n_steps,
        parameters={"k": 4.0, "m": 1.0, "c": 0.4, "x_0": 1.0, "v_0": 0.0},
    )
    sim = DampedHarmonicOscillator(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "k": 4.0,
        "m": 1.0,
        "c": 0.4,
        "omega_0": 2.0,  # sqrt(4/1)
        "zeta": 0.1,  # 0.4 / (2*sqrt(4*1))
    }


def run_harmonic_oscillator_rediscovery(
    output_dir: str | Path = "output/rediscovery/harmonic_oscillator",
    n_iterations: int = 40,
) -> dict:
    """Run the full harmonic oscillator rediscovery.

    1. Frequency: omega_0 = sqrt(k/m) via PySR
    2. Damping: decay_rate = zeta * omega_0 via PySR
    3. SINDy ODE recovery
    """
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "harmonic_oscillator",
        "targets": {
            "frequency": "omega_0 = sqrt(k/m)",
            "period": "T = 2*pi*sqrt(m/k)",
            "damping": "decay_rate = c/(2m) = zeta*omega_0",
            "ode": "x'' + (c/m)*x' + (k/m)*x = 0",
        },
    }

    # --- Part 1: Frequency rediscovery ---
    logger.info("Part 1: Frequency measurement omega_0 = sqrt(k/m)...")
    freq_data = generate_frequency_data(n_samples=200, n_steps=10000, dt=0.001)

    rel_error = np.abs(freq_data["omega_measured"] - freq_data["omega_theory"]) / freq_data["omega_theory"]
    results["frequency_accuracy"] = {
        "n_samples": len(freq_data["k"]),
        "mean_relative_error": float(np.mean(rel_error)),
        "max_relative_error": float(np.max(rel_error)),
    }
    logger.info(f"  Frequency accuracy: mean error = {np.mean(rel_error):.4%}")

    # PySR: omega = f(k, m)
    logger.info(f"  Running PySR for omega = f(k, m) with {n_iterations} iterations...")
    X = np.column_stack([freq_data["k"], freq_data["m"]])
    y = freq_data["omega_measured"]

    discoveries = run_symbolic_regression(
        X,
        y,
        variable_names=["k_", "m_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "square"],
        max_complexity=15,
        populations=20,
        population_size=40,
    )

    results["frequency_pysr"] = {
        "n_discoveries": len(discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in discoveries[:5]
        ],
    }
    if discoveries:
        best = discoveries[0]
        results["frequency_pysr"]["best"] = best.expression
        results["frequency_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")

    # --- Part 2: Damping rediscovery ---
    logger.info("Part 2: Damping rate measurement...")
    damp_data = generate_damping_data(n_samples=100, n_steps=20000, dt=0.001)

    damp_error = np.abs(damp_data["decay_rate_measured"] - damp_data["decay_rate_theory"])
    damp_rel_error = damp_error / np.maximum(damp_data["decay_rate_theory"], 1e-10)
    results["damping_accuracy"] = {
        "n_samples": len(damp_data["k"]),
        "mean_absolute_error": float(np.mean(damp_error)),
        "mean_relative_error": float(np.mean(damp_rel_error)),
    }
    logger.info(f"  Damping accuracy: mean rel error = {np.mean(damp_rel_error):.4%}")

    # PySR: decay_rate = f(k, m, c)
    logger.info(f"  Running PySR for decay_rate = f(k, m, c)...")
    X_damp = np.column_stack([damp_data["k"], damp_data["m"], damp_data["c"]])
    y_damp = damp_data["decay_rate_measured"]

    damp_discoveries = run_symbolic_regression(
        X_damp,
        y_damp,
        variable_names=["k_", "m_", "c_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt"],
        max_complexity=15,
        populations=20,
        population_size=40,
    )

    results["damping_pysr"] = {
        "n_discoveries": len(damp_discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in damp_discoveries[:5]
        ],
    }
    if damp_discoveries:
        best = damp_discoveries[0]
        results["damping_pysr"]["best"] = best.expression
        results["damping_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")

    # --- Part 3: SINDy ODE recovery ---
    logger.info("Part 3: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_ode_data(n_steps=5000, dt=0.001)
        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x", "v"],
            threshold=0.01,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries
            ],
            "true_k": ode_data["k"],
            "true_m": ode_data["m"],
            "true_c": ode_data["c"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "frequency_data.npz",
        k=freq_data["k"],
        m=freq_data["m"],
        omega_measured=freq_data["omega_measured"],
        omega_theory=freq_data["omega_theory"],
        T_measured=freq_data["T_measured"],
        T_theory=freq_data["T_theory"],
    )

    np.savez(
        output_path / "damping_data.npz",
        **{k: v for k, v in damp_data.items()},
    )

    return results
