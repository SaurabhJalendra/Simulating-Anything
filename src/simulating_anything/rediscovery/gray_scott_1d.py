"""Gray-Scott 1D rediscovery.

Targets:
- Pulse regime detection: sweep (f, k) to find self-replicating pulse regions
- Pulse counting: detect splitting events as a function of parameters
- Pulse speed measurement
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.gray_scott_1d import GrayScott1DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    D_u: float = 0.16,
    D_v: float = 0.08,
    f: float = 0.04,
    k: float = 0.06,
    N: int = 256,
    L: float = 2.5,
    dt: float = 1.0,
    n_steps: int = 1000,
    seed: int = 42,
) -> SimulationConfig:
    """Create a SimulationConfig for 1D Gray-Scott."""
    return SimulationConfig(
        domain=Domain.GRAY_SCOTT_1D,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "D_u": D_u,
            "D_v": D_v,
            "f": f,
            "k": k,
            "N": float(N),
            "L": L,
        },
        seed=seed,
    )


def generate_pulse_regime_data(
    n_f: int = 8,
    n_k: int = 8,
    n_steps: int = 2000,
    N: int = 256,
) -> dict[str, np.ndarray]:
    """Sweep (f, k) parameter space to identify pulse regimes.

    Classifies outcomes as: no_pattern, single_pulse, multi_pulse, uniform_v.

    Args:
        n_f: Number of feed rate values to sweep.
        n_k: Number of kill rate values to sweep.
        n_steps: Simulation steps per parameter set.
        N: Grid resolution.

    Returns:
        Dict with f_values, k_values, pulse_counts, max_v arrays.
    """
    f_values = np.linspace(0.01, 0.07, n_f)
    k_values = np.linspace(0.04, 0.07, n_k)

    all_f = []
    all_k = []
    all_counts = []
    all_max_v = []

    # Compute CFL-safe dt
    D_max = 0.16
    dx = 2.5 / N
    dt = 0.5 * dx**2 / (4.0 * D_max)  # half the CFL limit for safety

    for i, f in enumerate(f_values):
        for j, k in enumerate(k_values):
            config = _make_config(f=f, k=k, N=N, dt=dt, n_steps=n_steps)
            sim = GrayScott1DSimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()

            n_pulses = sim.count_pulses()
            mv = sim.max_v

            all_f.append(f)
            all_k.append(k)
            all_counts.append(n_pulses)
            all_max_v.append(mv)

            logger.debug(
                f"  f={f:.3f}, k={k:.3f}: pulses={n_pulses}, max_v={mv:.4f}"
            )

        if (i + 1) % 4 == 0:
            logger.info(f"  Regime sweep: {i + 1}/{n_f} f values done")

    return {
        "f": np.array(all_f),
        "k": np.array(all_k),
        "pulse_count": np.array(all_counts),
        "max_v": np.array(all_max_v),
    }


def generate_splitting_data(
    n_k: int = 15,
    n_steps: int = 3000,
    N: int = 256,
) -> dict[str, np.ndarray]:
    """Detect pulse splitting by sweeping k at fixed f.

    At fixed f=0.04, varying k reveals transitions from single-pulse
    to multi-pulse (splitting) to extinction.

    Args:
        n_k: Number of kill rate values.
        n_steps: Steps per simulation.
        N: Grid resolution.

    Returns:
        Dict with k_values, pulse_counts, max_v arrays.
    """
    f_fixed = 0.04
    k_values = np.linspace(0.045, 0.065, n_k)
    dx = 2.5 / N
    dt = 0.5 * dx**2 / (4.0 * 0.16)

    all_k = []
    all_counts = []
    all_max_v = []

    for k in k_values:
        config = _make_config(f=f_fixed, k=k, N=N, dt=dt, n_steps=n_steps)
        sim = GrayScott1DSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        n_pulses = sim.count_pulses()
        mv = sim.max_v

        all_k.append(k)
        all_counts.append(n_pulses)
        all_max_v.append(mv)
        logger.debug(f"  k={k:.4f}: pulses={n_pulses}, max_v={mv:.4f}")

    return {
        "k": np.array(all_k),
        "pulse_count": np.array(all_counts),
        "max_v": np.array(all_max_v),
        "f": f_fixed,
    }


def generate_speed_data(
    n_samples: int = 5,
    track_steps: int = 200,
    warmup_steps: int = 1000,
    N: int = 256,
) -> dict[str, np.ndarray]:
    """Measure pulse speed for different parameter values.

    Args:
        n_samples: Number of parameter values to try.
        track_steps: Steps to track pulse movement.
        warmup_steps: Steps to let pattern stabilize before tracking.
        N: Grid resolution.

    Returns:
        Dict with f_values, speeds arrays.
    """
    f_values = np.linspace(0.03, 0.05, n_samples)
    k_fixed = 0.06
    dx = 2.5 / N
    dt = 0.5 * dx**2 / (4.0 * 0.16)

    all_f = []
    all_speeds = []

    for f in f_values:
        config = _make_config(f=f, k=k_fixed, N=N, dt=dt, n_steps=warmup_steps)
        sim = GrayScott1DSimulation(config)
        sim.reset()

        # Warmup
        for _ in range(warmup_steps):
            sim.step()

        # Measure speed only if a pulse exists
        if sim.count_pulses() >= 1:
            speed = sim.measure_pulse_speed(n_steps=track_steps)
            all_f.append(f)
            all_speeds.append(speed)
            logger.debug(f"  f={f:.3f}: speed={speed:.4f}")

    return {
        "f": np.array(all_f),
        "speed": np.array(all_speeds),
        "k": k_fixed,
    }


def run_gray_scott_1d_rediscovery(
    output_dir: str | Path = "output/rediscovery/gray_scott_1d",
    n_iterations: int = 40,
) -> dict:
    """Run Gray-Scott 1D rediscovery analysis.

    Sweeps parameter space to find pulse regimes, detect splitting
    bifurcations, and measure pulse speeds.

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iterations (for future symbolic regression).

    Returns:
        Results dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "gray_scott_1d",
        "targets": {
            "pulse_regime": "Self-replicating pulse detection in (f, k) space",
            "splitting": "Pulse splitting bifurcation vs k",
            "pulse_speed": "Pulse propagation speed",
        },
    }

    # 1. Pulse regime sweep
    logger.info("Sweeping (f, k) parameter space for pulse regimes...")
    regime_data = generate_pulse_regime_data(n_f=6, n_k=6, n_steps=1500, N=128)

    n_with_pulses = int(np.sum(regime_data["pulse_count"] > 0))
    results["pulse_regime"] = {
        "n_parameter_sets": len(regime_data["f"]),
        "n_with_pulses": n_with_pulses,
        "max_pulses_observed": int(np.max(regime_data["pulse_count"])),
        "f_range": [float(regime_data["f"].min()), float(regime_data["f"].max())],
        "k_range": [float(regime_data["k"].min()), float(regime_data["k"].max())],
    }
    logger.info(
        f"  {n_with_pulses}/{len(regime_data['f'])} parameter sets show pulses"
    )

    # 2. Splitting bifurcation
    logger.info("Detecting pulse splitting bifurcation...")
    split_data = generate_splitting_data(n_k=10, n_steps=2000, N=128)

    # Find where pulse count changes
    counts = split_data["pulse_count"]
    k_vals = split_data["k"]
    transitions = []
    for i in range(1, len(counts)):
        if counts[i] != counts[i - 1]:
            transitions.append({
                "k_before": float(k_vals[i - 1]),
                "k_after": float(k_vals[i]),
                "count_before": int(counts[i - 1]),
                "count_after": int(counts[i]),
            })

    results["splitting"] = {
        "n_k_values": len(k_vals),
        "pulse_counts": counts.tolist(),
        "k_values": k_vals.tolist(),
        "transitions": transitions,
        "n_transitions": len(transitions),
    }
    logger.info(f"  Found {len(transitions)} pulse count transitions")

    # 3. Pulse speed
    logger.info("Measuring pulse speeds...")
    speed_data = generate_speed_data(n_samples=4, track_steps=100, warmup_steps=500, N=128)

    if len(speed_data["speed"]) > 0:
        results["pulse_speed"] = {
            "n_measurements": len(speed_data["speed"]),
            "mean_speed": float(np.mean(speed_data["speed"])),
            "speeds": speed_data["speed"].tolist(),
            "f_values": speed_data["f"].tolist(),
        }
        logger.info(f"  Mean pulse speed: {np.mean(speed_data['speed']):.4f}")
    else:
        results["pulse_speed"] = {"n_measurements": 0, "note": "No stable pulses found"}
        logger.info("  No stable pulses found for speed measurement")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f_out:
        json.dump(results, f_out, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
