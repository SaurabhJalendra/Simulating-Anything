"""Hodgkin-Huxley rediscovery.

Targets:
- f-I curve: firing frequency as a function of injected current I_ext
- Rheobase current: minimum I for spiking
- Action potential shape: spike amplitude, duration, and refractory period
- Ionic current decomposition during action potential
- ODE recovery via SINDy (optional, gating nonlinearity is complex)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.hodgkin_huxley import HodgkinHuxleySimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_spike_data(
    I_ext: float = 10.0,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate a single action potential trajectory for analysis."""
    config = SimulationConfig(
        domain=Domain.HODGKIN_HUXLEY,
        dt=dt,
        n_steps=n_steps,
        parameters={"I_ext": I_ext},
    )
    sim = HodgkinHuxleySimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "V": states[:, 0],
        "n": states[:, 1],
        "m": states[:, 2],
        "h": states[:, 3],
        "I_ext": I_ext,
    }


def generate_fi_curve(
    n_I: int = 30,
    I_min: float = 0.0,
    I_max: float = 150.0,
    dt: float = 0.01,
    t_max: float = 500.0,
) -> dict[str, np.ndarray]:
    """Generate firing frequency vs injected current (f-I curve)."""
    I_values = np.linspace(I_min, I_max, n_I)

    config = SimulationConfig(
        domain=Domain.HODGKIN_HUXLEY,
        dt=dt,
        n_steps=1000,
        parameters={"I_ext": 0.0},
    )
    sim = HodgkinHuxleySimulation(config)
    fi_data = sim.compute_fi_curve(I_values, t_max=t_max)

    return fi_data


def measure_spike_properties(
    I_ext: float = 10.0,
    dt: float = 0.01,
) -> dict[str, float]:
    """Measure action potential amplitude, duration, and refractory period.

    Returns:
        Dict with spike_amplitude, spike_duration, refractory_period_ms.
    """
    data = generate_spike_data(I_ext=I_ext, n_steps=50000, dt=dt)
    V = data["V"]

    # Find spikes
    config = SimulationConfig(
        domain=Domain.HODGKIN_HUXLEY, dt=dt, n_steps=1000,
        parameters={"I_ext": I_ext},
    )
    sim = HodgkinHuxleySimulation(config)
    spike_indices = sim.detect_spikes(V, threshold=0.0)

    if len(spike_indices) < 2:
        return {
            "spike_amplitude": 0.0,
            "spike_duration_ms": 0.0,
            "refractory_period_ms": 0.0,
        }

    # Measure amplitude from first complete spike after transient
    first_spike = spike_indices[0]
    # Search for peak near first spike
    search_end = min(first_spike + int(2.0 / dt), len(V))
    peak_idx = first_spike + np.argmax(V[first_spike:search_end])
    amplitude = float(V[peak_idx])

    # Spike duration: time above threshold (0 mV)
    above = V[first_spike:search_end] > 0.0
    duration_ms = float(np.sum(above) * dt)

    # Refractory period: ISI between first two spikes
    if len(spike_indices) >= 2:
        isi = float((spike_indices[1] - spike_indices[0]) * dt)
    else:
        isi = 0.0

    return {
        "spike_amplitude": amplitude,
        "spike_duration_ms": duration_ms,
        "refractory_period_ms": isi,
    }


def measure_ionic_currents_during_spike(
    I_ext: float = 10.0,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Record ionic current decomposition during an action potential."""
    n_steps = 30000
    config = SimulationConfig(
        domain=Domain.HODGKIN_HUXLEY,
        dt=dt,
        n_steps=n_steps,
        parameters={"I_ext": I_ext},
    )
    sim = HodgkinHuxleySimulation(config)
    sim.reset()

    V_trace = []
    I_Na_trace = []
    I_K_trace = []
    I_L_trace = []

    for _ in range(n_steps):
        sim.step()
        state = sim.observe()
        V_trace.append(state[0])
        currents = sim.ionic_currents(state)
        I_Na_trace.append(currents["I_Na"])
        I_K_trace.append(currents["I_K"])
        I_L_trace.append(currents["I_L"])

    return {
        "V": np.array(V_trace),
        "I_Na": np.array(I_Na_trace),
        "I_K": np.array(I_K_trace),
        "I_L": np.array(I_L_trace),
        "time": np.arange(1, n_steps + 1) * dt,
    }


def run_hodgkin_huxley_rediscovery(
    output_dir: str | Path = "output/rediscovery/hodgkin_huxley",
    n_iterations: int = 40,
) -> dict:
    """Run Hodgkin-Huxley rediscovery pipeline.

    Steps:
    1. Generate f-I curve and find rheobase current
    2. Measure spike properties (amplitude, duration, refractory period)
    3. Verify ionic current decomposition
    4. Attempt PySR fit of f-I relationship
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "hodgkin_huxley",
        "targets": {
            "fi_curve": "firing frequency vs injected current",
            "rheobase": "minimum current for spiking",
            "spike_shape": "amplitude ~40-50mV, duration ~1ms",
            "refractory": "absolute and relative refractory periods",
            "ionic_decomposition": "I_Na + I_K + I_L balance",
        },
    }

    # --- Part 1: f-I curve ---
    logger.info("Part 1: Computing f-I curve (30 current values)...")
    fi_data = generate_fi_curve(n_I=30, I_min=0.0, I_max=150.0, dt=0.01)

    # Find rheobase (minimum current for spiking)
    firing = fi_data["frequency"] > 0.5  # At least 0.5 Hz
    if np.any(firing):
        rheobase_idx = np.argmax(firing)
        rheobase = float(fi_data["I"][rheobase_idx])
        max_freq = float(np.max(fi_data["frequency"]))
        n_firing = int(np.sum(firing))
        results["fi_curve"] = {
            "rheobase_current": rheobase,
            "max_frequency_Hz": max_freq,
            "n_firing_values": n_firing,
            "I_range": [float(fi_data["I"][0]), float(fi_data["I"][-1])],
        }
        logger.info(f"  Rheobase: I ~ {rheobase:.2f} uA/cm^2")
        logger.info(f"  Max frequency: {max_freq:.1f} Hz")
        logger.info(f"  Firing at {n_firing}/{len(fi_data['I'])} current values")

        # Check monotonicity of f-I curve (above rheobase)
        firing_freqs = fi_data["frequency"][firing]
        if len(firing_freqs) > 2:
            diffs = np.diff(firing_freqs)
            monotonic_frac = float(np.mean(diffs >= -0.5))
            results["fi_curve"]["monotonic_fraction"] = monotonic_frac
            logger.info(f"  Monotonicity: {monotonic_frac:.2%}")
    else:
        results["fi_curve"] = {"note": "No spiking detected in range"}
        logger.warning("  No spiking detected")

    # --- Part 2: Spike properties ---
    logger.info("Part 2: Measuring spike properties at I=10...")
    spike_props = measure_spike_properties(I_ext=10.0, dt=0.01)
    results["spike_properties"] = spike_props
    logger.info(f"  Amplitude: {spike_props['spike_amplitude']:.1f} mV")
    logger.info(f"  Duration: {spike_props['spike_duration_ms']:.2f} ms")
    logger.info(f"  Refractory period: {spike_props['refractory_period_ms']:.2f} ms")

    # --- Part 3: Ionic current decomposition ---
    logger.info("Part 3: Ionic current decomposition...")
    ionic_data = measure_ionic_currents_during_spike(I_ext=10.0, dt=0.01)

    # Verify that currents are large during spike
    peak_Na = float(np.min(ionic_data["I_Na"]))  # Na inward is negative
    peak_K = float(np.max(ionic_data["I_K"]))     # K outward is positive
    results["ionic_currents"] = {
        "peak_I_Na": peak_Na,
        "peak_I_K": peak_K,
        "mean_I_L": float(np.mean(np.abs(ionic_data["I_L"]))),
        "Na_dominates_upstroke": bool(abs(peak_Na) > abs(peak_K)),
    }
    logger.info(f"  Peak I_Na: {peak_Na:.1f} (inward)")
    logger.info(f"  Peak I_K: {peak_K:.1f} (outward)")

    # --- Part 4: PySR fit of f-I curve ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        if np.any(firing) and np.sum(firing) > 3:
            logger.info("Part 4: PySR fit of f-I curve...")
            X = fi_data["I"][firing].reshape(-1, 1)
            y = fi_data["frequency"][firing]

            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["I_ext"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["fi_pysr"] = {
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
                results["fi_pysr"]["best"] = best.expression
                results["fi_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  PySR best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["fi_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save f-I data
    np.savez(
        output_path / "fi_curve.npz",
        I=fi_data["I"],
        frequency=fi_data["frequency"],
    )

    return results
