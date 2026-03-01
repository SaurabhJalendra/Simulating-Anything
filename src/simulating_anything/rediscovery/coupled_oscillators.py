"""Coupled oscillators rediscovery.

Targets:
- Beat frequency: omega_beat = sqrt((k + 2*kc)/m) - sqrt(k/m)
- Normal mode frequencies: omega_s = sqrt(k/m), omega_a = sqrt((k + 2*kc)/m)
- SINDy recovery of coupled ODEs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.coupled_oscillators import CoupledOscillators
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_beat_frequency_data(
    n_samples: int = 100,
    n_steps: int = 50000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate coupled oscillator data to study beat frequency.

    Sweep coupling constant kc, measure beat frequency from amplitude
    modulation envelope of oscillator 1.

    The beat frequency should satisfy:
        omega_beat = sqrt((k + 2*kc)/m) - sqrt(k/m)
    """
    rng = np.random.default_rng(42)

    all_k = []
    all_m = []
    all_kc = []
    all_beat_measured = []
    all_beat_theory = []

    for i in range(n_samples):
        k = rng.uniform(1.0, 10.0)
        m = rng.uniform(0.5, 3.0)
        kc = rng.uniform(0.05, 2.0)

        config = SimulationConfig(
            domain=Domain.COUPLED_OSCILLATORS,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "k": k, "m": m, "kc": kc,
                "x1_0": 1.0, "v1_0": 0.0,
                "x2_0": 0.0, "v2_0": 0.0,
            },
        )
        sim = CoupledOscillators(config)
        sim.reset()

        # Collect x1 positions
        positions = [sim.observe()[0]]
        for _ in range(n_steps):
            state = sim.step()
            positions.append(state[0])

        positions = np.array(positions)

        # Find amplitude envelope peaks (beat nodes)
        # The amplitude of x1 oscillates at the beat frequency
        # Find local maxima of |x1|
        abs_pos = np.abs(positions)
        envelope_peaks = []
        envelope_times = []
        # Use a window to find peaks of the envelope
        window = max(5, int(0.5 / (dt * np.sqrt(k / m))))  # ~half oscillation
        for j in range(window, len(abs_pos) - window):
            if abs_pos[j] == np.max(abs_pos[j - window:j + window + 1]):
                if abs_pos[j] > 0.3:  # Significant peak
                    envelope_peaks.append(abs_pos[j])
                    envelope_times.append(j * dt)

        # Find beat period from envelope peaks
        # Group nearby peaks and find the maximum within each beat cycle
        if len(envelope_peaks) < 4:
            continue

        envelope_peaks = np.array(envelope_peaks)
        envelope_times = np.array(envelope_times)

        # Find the peaks of the envelope (maxima of the maxima)
        beat_maxima_times = []
        for j in range(1, len(envelope_peaks) - 1):
            if (envelope_peaks[j] > envelope_peaks[j - 1]
                    and envelope_peaks[j] > envelope_peaks[j + 1]):
                beat_maxima_times.append(envelope_times[j])

        if len(beat_maxima_times) >= 2:
            beat_periods = np.diff(beat_maxima_times)
            T_beat = float(np.median(beat_periods))
            if T_beat > 0:
                beat_measured = 2 * np.pi / T_beat
                beat_theory = (
                    np.sqrt((k + 2 * kc) / m) - np.sqrt(k / m)
                )

                all_k.append(k)
                all_m.append(m)
                all_kc.append(kc)
                all_beat_measured.append(beat_measured)
                all_beat_theory.append(beat_theory)

        if (i + 1) % 25 == 0:
            logger.info(f"  Beat frequency measurement {i + 1}/{n_samples}")

    return {
        "k": np.array(all_k),
        "m": np.array(all_m),
        "kc": np.array(all_kc),
        "beat_measured": np.array(all_beat_measured),
        "beat_theory": np.array(all_beat_theory),
    }


def generate_normal_mode_data(
    n_samples: int = 50,
    n_steps: int = 30000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate data verifying normal mode frequencies.

    Excite symmetric and antisymmetric modes independently,
    measure oscillation frequency of each.
    """
    rng = np.random.default_rng(123)

    all_k = []
    all_m = []
    all_kc = []
    all_omega_s_measured = []
    all_omega_s_theory = []
    all_omega_a_measured = []
    all_omega_a_theory = []

    for i in range(n_samples):
        k = rng.uniform(1.0, 10.0)
        m = rng.uniform(0.5, 3.0)
        kc = rng.uniform(0.05, 2.0)

        omega_s_theory = np.sqrt(k / m)
        omega_a_theory = np.sqrt((k + 2.0 * kc) / m)

        # --- Symmetric mode: x1 = x2 ---
        config_s = SimulationConfig(
            domain=Domain.COUPLED_OSCILLATORS,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "k": k, "m": m, "kc": kc,
                "x1_0": 1.0, "v1_0": 0.0,
                "x2_0": 1.0, "v2_0": 0.0,
            },
        )
        sim_s = CoupledOscillators(config_s)
        sim_s.reset()

        positions_s = [sim_s.observe()[0]]
        for _ in range(n_steps):
            positions_s.append(sim_s.step()[0])
        positions_s = np.array(positions_s)

        omega_s_meas = _measure_frequency(positions_s, dt)

        # --- Antisymmetric mode: x1 = -x2 ---
        config_a = SimulationConfig(
            domain=Domain.COUPLED_OSCILLATORS,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "k": k, "m": m, "kc": kc,
                "x1_0": 1.0, "v1_0": 0.0,
                "x2_0": -1.0, "v2_0": 0.0,
            },
        )
        sim_a = CoupledOscillators(config_a)
        sim_a.reset()

        positions_a = [sim_a.observe()[0]]
        for _ in range(n_steps):
            positions_a.append(sim_a.step()[0])
        positions_a = np.array(positions_a)

        omega_a_meas = _measure_frequency(positions_a, dt)

        if omega_s_meas is not None and omega_a_meas is not None:
            all_k.append(k)
            all_m.append(m)
            all_kc.append(kc)
            all_omega_s_measured.append(omega_s_meas)
            all_omega_s_theory.append(omega_s_theory)
            all_omega_a_measured.append(omega_a_meas)
            all_omega_a_theory.append(omega_a_theory)

        if (i + 1) % 10 == 0:
            logger.info(f"  Normal mode measurement {i + 1}/{n_samples}")

    return {
        "k": np.array(all_k),
        "m": np.array(all_m),
        "kc": np.array(all_kc),
        "omega_s_measured": np.array(all_omega_s_measured),
        "omega_s_theory": np.array(all_omega_s_theory),
        "omega_a_measured": np.array(all_omega_a_measured),
        "omega_a_theory": np.array(all_omega_a_theory),
    }


def _measure_frequency(positions: np.ndarray, dt: float) -> float | None:
    """Measure oscillation frequency from positive-going zero crossings."""
    crossings = []
    for j in range(1, len(positions)):
        if positions[j - 1] < 0 and positions[j] >= 0:
            frac = -positions[j - 1] / (positions[j] - positions[j - 1])
            crossings.append((j - 1 + frac) * dt)

    if len(crossings) >= 3:
        periods = np.diff(crossings)
        T = float(np.median(periods))
        if T > 0:
            return 2.0 * np.pi / T
    return None


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate a coupled oscillator trajectory for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.COUPLED_OSCILLATORS,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "k": 4.0, "m": 1.0, "kc": 0.5,
            "x1_0": 1.0, "v1_0": 0.0,
            "x2_0": 0.0, "v2_0": 0.0,
        },
    )
    sim = CoupledOscillators(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "k": 4.0,
        "m": 1.0,
        "kc": 0.5,
    }


def run_coupled_oscillators_rediscovery(
    output_dir: str | Path = "output/rediscovery/coupled_oscillators",
    n_iterations: int = 40,
) -> dict:
    """Run the full coupled oscillators rediscovery.

    1. Beat frequency: omega_beat = sqrt((k+2*kc)/m) - sqrt(k/m) via PySR
    2. Normal mode frequencies via PySR
    3. SINDy ODE recovery
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "coupled_oscillators",
        "targets": {
            "beat_frequency": "omega_beat = sqrt((k+2*kc)/m) - sqrt(k/m)",
            "symmetric_mode": "omega_s = sqrt(k/m)",
            "antisymmetric_mode": "omega_a = sqrt((k+2*kc)/m)",
            "ode": "x1'' = -(k/m)*x1 - (kc/m)*(x1-x2)",
        },
    }

    # --- Part 1: Beat frequency rediscovery ---
    logger.info("Part 1: Beat frequency measurement...")
    beat_data = generate_beat_frequency_data(
        n_samples=100, n_steps=50000, dt=0.001,
    )

    if len(beat_data["k"]) > 0:
        rel_error = (
            np.abs(beat_data["beat_measured"] - beat_data["beat_theory"])
            / np.maximum(beat_data["beat_theory"], 1e-10)
        )
        results["beat_accuracy"] = {
            "n_samples": len(beat_data["k"]),
            "mean_relative_error": float(np.mean(rel_error)),
            "max_relative_error": float(np.max(rel_error)),
        }
        logger.info(
            f"  Beat accuracy: mean error = {np.mean(rel_error):.4%}"
        )

        # PySR: omega_beat = f(k, m, kc)
        logger.info(
            f"  Running PySR for omega_beat = f(k, m, kc) "
            f"with {n_iterations} iterations..."
        )
        X = np.column_stack([
            beat_data["k"], beat_data["m"], beat_data["kc"],
        ])
        y = beat_data["beat_measured"]

        discoveries = run_symbolic_regression(
            X,
            y,
            variable_names=["k_", "m_", "kc_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=20,
            populations=20,
            population_size=40,
        )

        results["beat_pysr"] = {
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
            results["beat_pysr"]["best"] = best.expression
            results["beat_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    else:
        results["beat_accuracy"] = {"n_samples": 0, "error": "no data"}

    # --- Part 2: Normal mode frequencies ---
    logger.info("Part 2: Normal mode frequency verification...")
    mode_data = generate_normal_mode_data(
        n_samples=50, n_steps=30000, dt=0.001,
    )

    if len(mode_data["k"]) > 0:
        s_error = (
            np.abs(
                mode_data["omega_s_measured"] - mode_data["omega_s_theory"]
            )
            / mode_data["omega_s_theory"]
        )
        a_error = (
            np.abs(
                mode_data["omega_a_measured"] - mode_data["omega_a_theory"]
            )
            / mode_data["omega_a_theory"]
        )
        results["normal_modes"] = {
            "n_samples": len(mode_data["k"]),
            "symmetric_mean_error": float(np.mean(s_error)),
            "antisymmetric_mean_error": float(np.mean(a_error)),
        }
        logger.info(
            f"  Symmetric mode error: {np.mean(s_error):.4%}"
        )
        logger.info(
            f"  Antisymmetric mode error: {np.mean(a_error):.4%}"
        )

        # PySR: omega_a = f(k, m, kc)
        logger.info("  Running PySR for omega_a = f(k, m, kc)...")
        X_a = np.column_stack([
            mode_data["k"], mode_data["m"], mode_data["kc"],
        ])
        y_a = mode_data["omega_a_measured"]

        a_discoveries = run_symbolic_regression(
            X_a,
            y_a,
            variable_names=["k_", "m_", "kc_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=20,
            populations=20,
            population_size=40,
        )

        results["antisymmetric_pysr"] = {
            "n_discoveries": len(a_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in a_discoveries[:5]
            ],
        }
        if a_discoveries:
            best = a_discoveries[0]
            results["antisymmetric_pysr"]["best"] = best.expression
            results["antisymmetric_pysr"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    else:
        results["normal_modes"] = {"n_samples": 0, "error": "no data"}

    # --- Part 3: SINDy ODE recovery ---
    logger.info("Part 3: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_ode_data(n_steps=5000, dt=0.001)
        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x1", "v1", "x2", "v2"],
            threshold=0.01,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries
            ],
            "true_k": ode_data["k"],
            "true_m": ode_data["m"],
            "true_kc": ode_data["kc"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    if len(beat_data["k"]) > 0:
        np.savez(
            output_path / "beat_data.npz",
            k=beat_data["k"],
            m=beat_data["m"],
            kc=beat_data["kc"],
            beat_measured=beat_data["beat_measured"],
            beat_theory=beat_data["beat_theory"],
        )

    if len(mode_data["k"]) > 0:
        np.savez(
            output_path / "mode_data.npz",
            k=mode_data["k"],
            m=mode_data["m"],
            kc=mode_data["kc"],
            omega_s_measured=mode_data["omega_s_measured"],
            omega_s_theory=mode_data["omega_s_theory"],
            omega_a_measured=mode_data["omega_a_measured"],
            omega_a_theory=mode_data["omega_a_theory"],
        )

    return results
