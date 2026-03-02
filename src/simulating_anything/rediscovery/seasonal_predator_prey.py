"""Seasonal predator-prey rediscovery.

Targets:
- Transition from fixed point -> limit cycle -> quasi-periodic -> chaos as epsilon increases
- Dominant frequency from FFT matches seasonal forcing frequency 1/T
- Equilibrium recovery at epsilon=0 (standard Rosenzweig-MacArthur dynamics)
- Period-doubling route to chaos with increasing seasonal amplitude
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.seasonal_predator_prey import (
    SeasonalPredatorPreySimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.SEASONAL_PREDATOR_PREY


def _make_config(
    r0: float = 1.0,
    K: float = 100.0,
    a: float = 0.5,
    h: float = 0.02,
    e: float = 0.4,
    d: float = 0.2,
    epsilon: float = 0.3,
    T: float = 12.0,
    N_0: float = 50.0,
    P_0: float = 10.0,
    dt: float = 0.01,
    n_steps: int = 10000,
) -> SimulationConfig:
    """Build a SimulationConfig for the seasonal predator-prey model."""
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r0": r0, "K": K, "a": a, "h": h,
            "e": e, "d": d, "epsilon": epsilon, "T": T,
            "N_0": N_0, "P_0": P_0,
        },
    )


def generate_trajectory_data(
    epsilon: float = 0.3,
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate a single trajectory for the seasonal predator-prey model.

    Args:
        epsilon: Seasonal forcing amplitude.
        n_steps: Number of simulation steps.
        dt: Timestep.

    Returns:
        Dict with states array, time, prey and predator arrays.
    """
    config = _make_config(epsilon=epsilon, dt=dt, n_steps=n_steps)
    sim = SeasonalPredatorPreySimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "states": states,
        "time": np.arange(n_steps + 1) * dt,
        "dt": dt,
        "N": states[:, 0],
        "P": states[:, 1],
        "epsilon": epsilon,
    }


def generate_epsilon_sweep_data(
    epsilon_values: np.ndarray | None = None,
    n_steps: int = 80000,
    dt: float = 0.01,
    transient_steps: int = 30000,
) -> dict[str, np.ndarray | list]:
    """Sweep epsilon and characterize dynamics at each value.

    For each epsilon, runs the simulation, discards transients, and computes:
    - Prey amplitude (max - min after transient)
    - Number of distinct peaks per seasonal period (period-doubling indicator)
    - Dominant FFT frequency

    Args:
        epsilon_values: Array of forcing amplitudes to sweep.
        n_steps: Total steps per simulation.
        dt: Timestep.
        transient_steps: Steps to discard as transient.

    Returns:
        Dict with epsilon_values, amplitudes, dominant frequencies,
        dominant periods, and regime classifications.
    """
    if epsilon_values is None:
        epsilon_values = np.linspace(0.0, 1.0, 30)

    amplitudes = np.zeros(len(epsilon_values))
    dominant_freqs = np.zeros(len(epsilon_values))
    dominant_periods = np.zeros(len(epsilon_values))
    mean_prey = np.zeros(len(epsilon_values))
    mean_pred = np.zeros(len(epsilon_values))
    regimes: list[str] = []

    for i, eps_val in enumerate(epsilon_values):
        config = _make_config(epsilon=eps_val, dt=dt, n_steps=n_steps)
        sim = SeasonalPredatorPreySimulation(config)
        sim.reset()

        # Run full simulation
        states_list = []
        for step_idx in range(n_steps):
            sim.step()
            if step_idx >= transient_steps:
                states_list.append(sim.observe().copy())

        if len(states_list) == 0:
            amplitudes[i] = 0.0
            dominant_freqs[i] = 0.0
            dominant_periods[i] = float("inf")
            mean_prey[i] = 0.0
            mean_pred[i] = 0.0
            regimes.append("trivial")
            continue

        states = np.array(states_list)
        prey = states[:, 0]
        pred = states[:, 1]

        # Amplitude of prey oscillation
        amplitudes[i] = float(np.max(prey) - np.min(prey))
        mean_prey[i] = float(np.mean(prey))
        mean_pred[i] = float(np.mean(pred))

        # FFT for dominant frequency
        freq_info = sim.detect_dominant_frequency(states, dt=dt)
        dominant_freqs[i] = freq_info["dominant_freq"]
        dominant_periods[i] = freq_info["dominant_period"]

        # Classify regime based on amplitude and frequency content
        if amplitudes[i] < 0.5:
            regimes.append("fixed_point")
        else:
            # Check if frequency matches seasonal forcing
            seasonal_freq = 1.0 / sim.T
            freq_ratio = dominant_freqs[i] / seasonal_freq if seasonal_freq > 0 else 0
            if abs(freq_ratio - 1.0) < 0.15:
                regimes.append("entrained")
            elif abs(freq_ratio - 0.5) < 0.15:
                regimes.append("period_doubled")
            else:
                regimes.append("complex")

        logger.info(
            f"  eps={eps_val:.3f}: amplitude={amplitudes[i]:.2f}, "
            f"freq={dominant_freqs[i]:.4f}, regime={regimes[-1]}"
        )

    return {
        "epsilon_values": epsilon_values,
        "amplitudes": amplitudes,
        "dominant_freqs": dominant_freqs,
        "dominant_periods": dominant_periods,
        "mean_prey": mean_prey,
        "mean_pred": mean_pred,
        "regimes": regimes,
    }


def generate_frequency_data(
    epsilon: float = 0.3,
    n_steps: int = 80000,
    dt: float = 0.01,
    transient_steps: int = 30000,
) -> dict[str, np.ndarray | float]:
    """Generate FFT frequency analysis data for a single epsilon value.

    Args:
        epsilon: Seasonal forcing amplitude.
        n_steps: Total simulation steps.
        dt: Timestep.
        transient_steps: Steps to discard.

    Returns:
        Dict with frequencies, power spectrum, dominant frequency/period,
        and the seasonal forcing frequency for comparison.
    """
    config = _make_config(epsilon=epsilon, dt=dt, n_steps=n_steps)
    sim = SeasonalPredatorPreySimulation(config)
    sim.reset()

    # Run and discard transient
    for _ in range(transient_steps):
        sim.step()

    analysis_steps = n_steps - transient_steps
    states_list = []
    for _ in range(analysis_steps):
        sim.step()
        states_list.append(sim.observe().copy())

    states = np.array(states_list)
    prey = states[:, 0]

    # FFT analysis
    prey_centered = prey - np.mean(prey)
    n = len(prey_centered)
    fft_vals = np.fft.rfft(prey_centered)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)

    # Dominant frequency (exclude DC)
    if len(power) > 1:
        peak_idx = np.argmax(power[1:]) + 1
        dominant_freq = float(freqs[peak_idx])
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float("inf")
    else:
        dominant_freq = 0.0
        dominant_period = float("inf")

    seasonal_freq = 1.0 / sim.T

    return {
        "freqs": freqs,
        "power": power,
        "dominant_freq": dominant_freq,
        "dominant_period": dominant_period,
        "seasonal_freq": seasonal_freq,
        "seasonal_period": sim.T,
        "epsilon": epsilon,
        "freq_ratio": dominant_freq / seasonal_freq if seasonal_freq > 0 else 0.0,
    }


def run_seasonal_predator_prey_rediscovery(
    output_dir: str | Path = "output/rediscovery/seasonal_predator_prey",
    n_iterations: int = 50,
) -> dict:
    """Run the full seasonal predator-prey rediscovery pipeline.

    1. Sweep epsilon to show dynamical transitions
    2. Measure dominant frequency from FFT and compare to seasonal forcing
    3. Verify equilibrium at epsilon=0
    4. Detect period-doubling cascade

    Args:
        output_dir: Directory for output files.
        n_iterations: Number of PySR iterations (for future symbolic regression).

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "seasonal_predator_prey",
        "targets": {
            "seasonal_frequency": "Dominant FFT frequency matches 1/T",
            "epsilon_transition": "Fixed point -> entrained -> complex as epsilon grows",
            "equilibrium_at_zero": "epsilon=0 recovers standard RM dynamics",
            "period_doubling": "Period-doubling route to chaos",
        },
    }

    # --- Part 1: Epsilon sweep ---
    logger.info("Part 1: Epsilon sweep for dynamical transitions...")
    sweep = generate_epsilon_sweep_data(
        epsilon_values=np.linspace(0.0, 1.0, 25),
        n_steps=60000,
        dt=0.01,
        transient_steps=20000,
    )
    regime_counts: dict[str, int] = {}
    for reg in sweep["regimes"]:
        regime_counts[reg] = regime_counts.get(reg, 0) + 1

    results["epsilon_sweep"] = {
        "n_epsilon_values": len(sweep["epsilon_values"]),
        "epsilon_range": [float(sweep["epsilon_values"][0]),
                          float(sweep["epsilon_values"][-1])],
        "max_amplitude": float(np.max(sweep["amplitudes"])),
        "min_amplitude": float(np.min(sweep["amplitudes"])),
        "regime_counts": regime_counts,
        "regimes": sweep["regimes"],
    }
    logger.info(f"  Regime counts: {regime_counts}")

    # --- Part 2: Frequency analysis at moderate forcing ---
    logger.info("Part 2: Frequency analysis at epsilon=0.3...")
    freq_data = generate_frequency_data(
        epsilon=0.3, n_steps=80000, dt=0.01, transient_steps=30000,
    )
    results["frequency_analysis"] = {
        "epsilon": freq_data["epsilon"],
        "dominant_freq": freq_data["dominant_freq"],
        "dominant_period": freq_data["dominant_period"],
        "seasonal_freq": freq_data["seasonal_freq"],
        "seasonal_period": freq_data["seasonal_period"],
        "freq_ratio": freq_data["freq_ratio"],
        "frequency_match": abs(freq_data["freq_ratio"] - 1.0) < 0.15,
    }
    logger.info(
        f"  Dominant freq={freq_data['dominant_freq']:.4f}, "
        f"seasonal freq={freq_data['seasonal_freq']:.4f}, "
        f"ratio={freq_data['freq_ratio']:.3f}"
    )

    # --- Part 3: Equilibrium at epsilon=0 ---
    logger.info("Part 3: Equilibrium verification at epsilon=0...")
    config_no_forcing = _make_config(epsilon=0.0, dt=0.01, n_steps=100000)
    sim_eq = SeasonalPredatorPreySimulation(config_no_forcing)
    sim_eq.reset()

    for _ in range(100000):
        sim_eq.step()

    final_state = sim_eq.observe()
    eq = sim_eq.coexistence_equilibrium()

    results["equilibrium_at_zero"] = {
        "final_N": float(final_state[0]),
        "final_P": float(final_state[1]),
        "theory_N_star": eq["N_star"],
        "theory_P_star": eq["P_star"],
        "equilibrium_valid": eq["valid"],
    }
    if eq["valid"] and np.isfinite(eq["N_star"]):
        n_err = abs(final_state[0] - eq["N_star"]) / max(eq["N_star"], 1e-10)
        results["equilibrium_at_zero"]["N_relative_error"] = float(n_err)
    logger.info(
        f"  Final state: N={final_state[0]:.2f}, P={final_state[1]:.2f}"
    )
    logger.info(
        f"  Theory: N*={eq['N_star']:.2f}, P*={eq['P_star']:.2f}"
    )

    # --- Part 4: Trajectory at default parameters ---
    logger.info("Part 4: Default trajectory generation...")
    traj_data = generate_trajectory_data(epsilon=0.3, n_steps=30000, dt=0.01)
    results["default_trajectory"] = {
        "n_steps": len(traj_data["time"]) - 1,
        "final_N": float(traj_data["N"][-1]),
        "final_P": float(traj_data["P"][-1]),
        "mean_N": float(np.mean(traj_data["N"])),
        "mean_P": float(np.mean(traj_data["P"])),
        "max_N": float(np.max(traj_data["N"])),
        "min_N": float(np.min(traj_data["N"])),
    }
    logger.info(
        f"  Mean prey={np.mean(traj_data['N']):.2f}, "
        f"mean pred={np.mean(traj_data['P']):.2f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
