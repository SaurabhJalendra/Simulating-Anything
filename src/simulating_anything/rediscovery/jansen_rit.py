"""Jansen-Rit neural mass model rediscovery.

Targets:
- Alpha rhythm (~10 Hz) detection for default parameters (p=120 Hz)
- Bifurcation structure: oscillation onset/offset as p varies
- EEG output characterization (amplitude, frequency)
- ODE recovery via SINDy
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.jansen_rit import JansenRitSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    p: float = 120.0,
    n_steps: int = 10000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses default Jansen-Rit parameters with specified external input.
    """
    config = SimulationConfig(
        domain=Domain.JANSEN_RIT,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "A": 3.25, "B": 22.0,
            "a": 100.0, "b": 50.0,
            "C": 135.0,
            "vmax": 5.0, "v0": 6.0, "r": 0.56,
            "p": p,
        },
    )
    sim = JansenRitSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    eeg = sim.compute_eeg_output(states)

    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "eeg": eeg,
        "p": p,
    }


def generate_sweep_data(
    n_p: int = 30,
    p_range: tuple[float, float] = (50.0, 300.0),
    dt: float = 0.001,
    n_steps: int = 5000,
) -> dict[str, np.ndarray]:
    """Sweep external input p and measure EEG amplitude and frequency."""
    config = SimulationConfig(
        domain=Domain.JANSEN_RIT,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "A": 3.25, "B": 22.0,
            "a": 100.0, "b": 50.0,
            "C": 135.0,
            "vmax": 5.0, "v0": 6.0, "r": 0.56,
            "p": 120.0,
        },
    )
    sim = JansenRitSimulation(config)
    result = sim.input_sweep(p_range=p_range, n_p=n_p, n_steps=n_steps)
    return result


def generate_alpha_data(
    dt: float = 0.001,
    n_transient: int = 2000,
    n_measure: int = 5000,
) -> dict[str, float | np.ndarray]:
    """Measure alpha rhythm frequency at default parameters (p=120)."""
    config = SimulationConfig(
        domain=Domain.JANSEN_RIT,
        dt=dt,
        n_steps=10000,
        parameters={
            "A": 3.25, "B": 22.0,
            "a": 100.0, "b": 50.0,
            "C": 135.0,
            "vmax": 5.0, "v0": 6.0, "r": 0.56,
            "p": 120.0,
        },
    )
    sim = JansenRitSimulation(config)
    freq = sim.compute_frequency(n_transient=n_transient, n_measure=n_measure)

    # Also collect the EEG signal for amplitude analysis
    sim.reset()
    for _ in range(n_transient):
        sim.step()

    eeg = np.zeros(n_measure)
    for i in range(n_measure):
        sim.step()
        eeg[i] = sim._state[1] - sim._state[2]

    amplitude = float(np.max(eeg) - np.min(eeg))

    return {
        "frequency": freq,
        "amplitude": amplitude,
        "eeg_signal": eeg,
    }


def run_jansen_rit_rediscovery(
    output_dir: str | Path = "output/rediscovery/jansen_rit",
    n_iterations: int = 40,
) -> dict:
    """Run Jansen-Rit rediscovery pipeline.

    Targets:
    1. Alpha rhythm detection (~10 Hz) at default p=120
    2. Input sweep: bifurcation structure (oscillation onset/offset)
    3. EEG output characterization
    4. SINDy ODE recovery
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "jansen_rit",
        "targets": {
            "alpha_rhythm": "~10 Hz oscillation at p=120",
            "bifurcation": "oscillation onset/offset vs p",
            "eeg_output": "amplitude and frequency characterization",
            "ode": "6D ODE recovery via SINDy",
        },
    }

    # --- Part 1: Alpha rhythm detection ---
    logger.info("Part 1: Alpha rhythm detection at default parameters...")
    alpha_data = generate_alpha_data(dt=0.001, n_transient=2000, n_measure=5000)
    results["alpha_rhythm"] = {
        "frequency_hz": alpha_data["frequency"],
        "amplitude": alpha_data["amplitude"],
        "is_alpha": bool(5.0 <= alpha_data["frequency"] <= 15.0),
    }
    logger.info(
        f"  Dominant frequency: {alpha_data['frequency']:.2f} Hz, "
        f"amplitude: {alpha_data['amplitude']:.4f}"
    )

    # --- Part 2: Input sweep (bifurcation) ---
    logger.info("Part 2: Input sweep for bifurcation structure...")
    sweep_data = generate_sweep_data(n_p=30, p_range=(50.0, 300.0), n_steps=5000)

    # Detect oscillation onset and offset
    amp_threshold = 0.5
    oscillating = sweep_data["amplitude"] > amp_threshold

    p_onset = None
    p_offset = None
    if np.any(oscillating):
        idx_first = np.argmax(oscillating)
        p_onset = float(sweep_data["p_values"][idx_first])

        # Find offset: last oscillating index
        idx_last = len(oscillating) - 1 - np.argmax(oscillating[::-1])
        if idx_last < len(oscillating) - 1:
            p_offset = float(sweep_data["p_values"][idx_last + 1])

    results["bifurcation"] = {
        "p_onset": p_onset,
        "p_offset": p_offset,
        "max_amplitude": float(np.max(sweep_data["amplitude"])),
        "n_oscillating": int(np.sum(oscillating)),
    }
    logger.info(
        f"  Oscillation onset: p ~ {p_onset}, "
        f"max amplitude: {np.max(sweep_data['amplitude']):.4f}"
    )

    # --- Part 3: EEG output characterization ---
    logger.info("Part 3: EEG output characterization...")
    # Frequency at peak amplitude
    if np.any(oscillating):
        peak_amp_idx = np.argmax(sweep_data["amplitude"])
        peak_freq = sweep_data["frequency"][peak_amp_idx]
        peak_p = sweep_data["p_values"][peak_amp_idx]
        results["eeg_characterization"] = {
            "peak_amplitude_p": float(peak_p),
            "peak_frequency_hz": float(peak_freq),
            "frequency_range": [
                float(np.min(sweep_data["frequency"][oscillating])),
                float(np.max(sweep_data["frequency"][oscillating])),
            ],
        }
        logger.info(
            f"  Peak amplitude at p={peak_p:.1f}, "
            f"frequency={peak_freq:.2f} Hz"
        )
    else:
        results["eeg_characterization"] = {"note": "No oscillations detected"}

    # --- Part 4: SINDy ODE recovery ---
    logger.info("Part 4: SINDy ODE recovery...")
    ode_data = generate_ode_data(p=120.0, n_steps=20000, dt=0.001)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=0.001,
            feature_names=["y0", "y1", "y2", "y3", "y4", "y5"],
            threshold=0.01,
            poly_degree=2,
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

    # --- Part 5: PySR frequency vs p ---
    logger.info("Part 5: PySR frequency vs external input p...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use sweep data where oscillation is detected
        valid = sweep_data["frequency"] > 0.5
        if np.sum(valid) > 5:
            X = sweep_data["p_values"][valid].reshape(-1, 1)
            y = sweep_data["frequency"][valid]

            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["p_ext"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square", "log"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["pysr_frequency"] = {
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
                results["pysr_frequency"]["best"] = best.expression
                results["pysr_frequency"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        else:
            results["pysr_frequency"] = {
                "note": "Not enough oscillating points for PySR"
            }
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_frequency"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "sweep_data.npz",
        p_values=sweep_data["p_values"],
        amplitude=sweep_data["amplitude"],
        frequency=sweep_data["frequency"],
    )

    return results
