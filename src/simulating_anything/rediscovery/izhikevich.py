"""Izhikevich neuron rediscovery.

Targets:
- f-I curve: firing frequency as a function of injected current I
- Pattern classification: RS, IB, CH, FS, TC, RZ, LTS across parameter space
- Spike timing analysis: ISI statistics for each pattern
- Reset mechanism: verify v->c, u->u+d at spike threshold v=30
- ODE recovery via SINDy (continuous dynamics between resets)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.izhikevich import (
    PATTERN_PRESETS,
    IzhikevichSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    pattern: str = "RS",
    I_ext: float = 10.0,
    n_steps: int = 10000,
    dt: float = 0.5,
) -> dict[str, np.ndarray | float | str]:
    """Generate trajectory data for a given pattern preset.

    Args:
        pattern: Preset name (RS, IB, CH, FS, TC, RZ, LTS).
        I_ext: External current.
        n_steps: Number of integration steps.
        dt: Timestep in ms.

    Returns:
        Dict with time, states, v, u arrays and metadata.
    """
    preset = IzhikevichSimulation.get_preset(pattern)
    params = {**preset, "I": I_ext}
    config = SimulationConfig(
        domain=Domain.IZHIKEVICH,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )
    sim = IzhikevichSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "v": states[:, 0],
        "u": states[:, 1],
        "I": I_ext,
        "pattern": pattern,
    }


def generate_fi_curve(
    pattern: str = "RS",
    n_I: int = 30,
    I_min: float = 0.0,
    I_max: float = 40.0,
    dt: float = 0.5,
    t_max: float = 1000.0,
) -> dict[str, np.ndarray]:
    """Generate firing frequency vs input current (f-I curve).

    Args:
        pattern: Preset name for (a, b, c, d) parameters.
        n_I: Number of current values to sweep.
        I_min: Minimum current.
        I_max: Maximum current.
        dt: Timestep in ms.
        t_max: Simulation time per current value (ms).

    Returns:
        Dict with 'I' and 'frequency' arrays.
    """
    preset = IzhikevichSimulation.get_preset(pattern)
    I_values = np.linspace(I_min, I_max, n_I)

    config = SimulationConfig(
        domain=Domain.IZHIKEVICH,
        dt=dt,
        n_steps=1000,
        parameters={**preset, "I": 0.0},
    )
    sim = IzhikevichSimulation(config)
    fi_data = sim.compute_fi_curve(I_values, t_max=t_max)

    return fi_data


def classify_all_presets(
    I_ext: float = 10.0,
    dt: float = 0.5,
    t_max: float = 1000.0,
) -> dict[str, dict[str, object]]:
    """Classify firing pattern for each named preset.

    Args:
        I_ext: External current for all presets.
        dt: Timestep in ms.
        t_max: Simulation time (ms).

    Returns:
        Dict mapping preset name to classification results.
    """
    results = {}
    for name, preset in PATTERN_PRESETS.items():
        params = {**preset, "I": I_ext}
        config = SimulationConfig(
            domain=Domain.IZHIKEVICH,
            dt=dt,
            n_steps=int(t_max / dt),
            parameters=params,
        )
        sim = IzhikevichSimulation(config)
        sim.reset()
        pattern = sim.classify_pattern(t_max=t_max)
        rate = sim.compute_firing_rate(t_max=t_max)
        results[name] = {
            "pattern": pattern,
            "firing_rate_per_ms": rate,
            "a": preset["a"],
            "b": preset["b"],
            "c": preset["c"],
            "d": preset["d"],
        }
    return results


def measure_spike_statistics(
    pattern: str = "RS",
    I_ext: float = 10.0,
    dt: float = 0.5,
    t_max: float = 1000.0,
    transient: float = 200.0,
) -> dict[str, float | int]:
    """Measure ISI statistics for a given pattern preset.

    Args:
        pattern: Preset name.
        I_ext: External current.
        dt: Timestep in ms.
        t_max: Total simulation time (ms).
        transient: Transient to discard (ms).

    Returns:
        Dict with mean_isi, std_isi, cv_isi, n_spikes.
    """
    preset = IzhikevichSimulation.get_preset(pattern)
    params = {**preset, "I": I_ext}
    n_steps = int(t_max / dt)
    transient_steps = int(transient / dt)

    config = SimulationConfig(
        domain=Domain.IZHIKEVICH,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )
    sim = IzhikevichSimulation(config)
    sim.reset()

    V_trace = np.zeros(n_steps + 1)
    V_trace[0] = sim.observe()[0]
    for i in range(1, n_steps + 1):
        sim.step()
        V_trace[i] = sim.observe()[0]

    V_steady = V_trace[transient_steps:]
    return sim.interspike_interval(V_steady, threshold=0.0)


def run_izhikevich_rediscovery(
    output_dir: str | Path = "output/rediscovery/izhikevich",
    n_iterations: int = 40,
) -> dict:
    """Run Izhikevich neuron rediscovery pipeline.

    Steps:
    1. f-I curve for Regular Spiking preset
    2. Pattern classification across all 7 presets
    3. Spike timing statistics comparison (RS vs FS vs IB)
    4. PySR fit of f-I relationship (optional)

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iteration count.

    Returns:
        Results dict with f-I curve, pattern classification, and spike stats.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "izhikevich",
        "targets": {
            "fi_curve": "firing frequency vs injected current",
            "patterns": "RS, IB, CH, FS, TC, RZ, LTS classification",
            "spike_timing": "ISI statistics across patterns",
            "reset": "v->c, u->u+d at v=30 mV",
        },
    }

    # --- Part 1: f-I curve for RS preset ---
    logger.info("Part 1: Computing f-I curve (RS preset, 30 current values)...")
    fi_data = generate_fi_curve(pattern="RS", n_I=30, I_min=0.0, I_max=40.0)

    firing = fi_data["frequency"] > 0.5
    if np.any(firing):
        rheobase_idx = int(np.argmax(firing))
        rheobase = float(fi_data["I"][rheobase_idx])
        max_freq = float(np.max(fi_data["frequency"]))
        n_firing = int(np.sum(firing))
        results["fi_curve"] = {
            "rheobase_current": rheobase,
            "max_frequency_Hz": max_freq,
            "n_firing_values": n_firing,
            "I_range": [float(fi_data["I"][0]), float(fi_data["I"][-1])],
        }
        logger.info(f"  Rheobase: I ~ {rheobase:.2f}")
        logger.info(f"  Max frequency: {max_freq:.1f} Hz")
        logger.info(f"  Firing at {n_firing}/{len(fi_data['I'])} current values")

        # Check monotonicity
        firing_freqs = fi_data["frequency"][firing]
        if len(firing_freqs) > 2:
            diffs = np.diff(firing_freqs)
            monotonic_frac = float(np.mean(diffs >= -0.5))
            results["fi_curve"]["monotonic_fraction"] = monotonic_frac
    else:
        results["fi_curve"] = {"note": "No spiking detected in range"}
        logger.warning("  No spiking detected")

    # --- Part 2: Pattern classification ---
    logger.info("Part 2: Classifying all 7 pattern presets at I=10...")
    preset_results = classify_all_presets(I_ext=10.0)
    results["pattern_classification"] = {}
    for name, info in preset_results.items():
        results["pattern_classification"][name] = {
            "pattern": info["pattern"],
            "firing_rate_per_ms": float(info["firing_rate_per_ms"]),
        }
        logger.info(f"  {name}: {info['pattern']} (rate={info['firing_rate_per_ms']:.4f}/ms)")

    # --- Part 3: Spike timing comparison ---
    logger.info("Part 3: Spike timing analysis (RS vs FS vs IB)...")
    for pattern_name in ["RS", "FS", "IB"]:
        stats = measure_spike_statistics(pattern=pattern_name, I_ext=10.0)
        results[f"spike_stats_{pattern_name}"] = {
            "mean_isi": stats["mean_isi"],
            "std_isi": stats["std_isi"],
            "cv_isi": stats["cv_isi"],
            "n_spikes": stats["n_spikes"],
        }
        logger.info(
            f"  {pattern_name}: mean_ISI={stats['mean_isi']:.2f}ms, "
            f"CV={stats['cv_isi']:.3f}, n_spikes={stats['n_spikes']}"
        )

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
