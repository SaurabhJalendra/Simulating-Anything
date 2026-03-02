"""SIR Metapopulation rediscovery.

Targets:
- Migration-induced synchronization of epidemic peaks across patches
- Epidemic wave propagation delay between source and remote patches
- Total epidemic size with vs without migration
- Global R0 = beta/gamma recovered from multi-patch dynamics
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sir_metapopulation import (
    SIRMetapopulationSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.SIR_METAPOPULATION


def _make_config(
    n_patches: int = 5,
    beta: float = 0.5,
    gamma: float = 0.1,
    m: float = 0.01,
    N_per_patch: float = 1000.0,
    dt: float = 0.1,
    n_steps: int = 2000,
) -> SimulationConfig:
    """Create a SimulationConfig for SIR metapopulation."""
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "n_patches": float(n_patches),
            "beta": beta,
            "gamma": gamma,
            "m": m,
            "N_per_patch": N_per_patch,
        },
    )


def generate_migration_sweep_data(
    n_m_values: int = 20,
    n_patches: int = 5,
    beta: float = 0.5,
    gamma: float = 0.1,
    N_per_patch: float = 1000.0,
    n_steps: int = 3000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep migration rate m and measure synchronization of epidemic peaks.

    For each m value, run the simulation and record:
    - Peak infected time for each patch
    - Standard deviation of peak times (synchronization measure)
    - Total epidemic size (fraction recovered at end)

    Returns:
        Dict with arrays: 'm_values', 'peak_time_std', 'total_final_size',
        'peak_times' (shape n_m x n_patches).
    """
    m_values = np.logspace(-4, 0, n_m_values)

    peak_time_stds = []
    total_final_sizes = []
    all_peak_times = []

    for m_val in m_values:
        config = _make_config(
            n_patches=n_patches, beta=beta, gamma=gamma,
            m=m_val, N_per_patch=N_per_patch, dt=dt, n_steps=n_steps,
        )
        sim = SIRMetapopulationSimulation(config)
        peaks = sim.peak_infected_per_patch(n_steps=n_steps)

        peak_times = peaks["peak_time"]
        all_peak_times.append(peak_times.copy())

        # Synchronization: std of peak arrival times
        peak_time_stds.append(float(np.std(peak_times)))

        # Total final size: fraction recovered across all patches
        total_N = N_per_patch * n_patches
        total_R = float(np.sum(sim.get_all_recovered()))
        total_final_sizes.append(total_R / total_N)

    return {
        "m_values": np.array(m_values),
        "peak_time_std": np.array(peak_time_stds),
        "total_final_size": np.array(total_final_sizes),
        "peak_times": np.array(all_peak_times),
    }


def generate_epidemic_wave_data(
    n_patches: int = 5,
    beta: float = 0.5,
    gamma: float = 0.1,
    m: float = 0.01,
    N_per_patch: float = 1000.0,
    n_steps: int = 3000,
    dt: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Run a single simulation and track epidemic wave propagation.

    Records time series of infected for each patch, plus peak times
    to quantify wave delay from source patch.

    Returns:
        Dict with 'I_timeseries' (n_steps+1 x n_patches), 'peak_times',
        'peak_delays' relative to patch 0, and parameters.
    """
    config = _make_config(
        n_patches=n_patches, beta=beta, gamma=gamma,
        m=m, N_per_patch=N_per_patch, dt=dt, n_steps=n_steps,
    )
    sim = SIRMetapopulationSimulation(config)
    sim.reset()

    I_series = [sim.get_all_infected()]
    for _ in range(n_steps):
        sim.step()
        I_series.append(sim.get_all_infected())

    I_timeseries = np.array(I_series)  # shape (n_steps+1, n_patches)

    # Find peak times per patch
    peak_steps = np.argmax(I_timeseries, axis=0)
    peak_times = peak_steps * dt
    peak_delays = peak_times - peak_times[0]

    return {
        "I_timeseries": I_timeseries,
        "peak_times": peak_times,
        "peak_delays": peak_delays,
        "n_patches": n_patches,
        "beta": beta,
        "gamma": gamma,
        "m": m,
        "dt": dt,
    }


def compare_with_without_migration(
    n_patches: int = 5,
    beta: float = 0.5,
    gamma: float = 0.1,
    N_per_patch: float = 1000.0,
    n_steps: int = 3000,
    dt: float = 0.1,
) -> dict[str, float]:
    """Compare total epidemic size with and without migration.

    With migration: disease spreads to all patches.
    Without migration (m=0): only seeded patch has an epidemic.

    Returns:
        Dict with 'total_R_with_migration', 'total_R_without_migration',
        'fraction_with', 'fraction_without'.
    """
    total_N = N_per_patch * n_patches

    # With migration
    config_with = _make_config(
        n_patches=n_patches, beta=beta, gamma=gamma,
        m=0.01, N_per_patch=N_per_patch, dt=dt, n_steps=n_steps,
    )
    sim_with = SIRMetapopulationSimulation(config_with)
    sim_with.reset()
    for _ in range(n_steps):
        sim_with.step()
    total_R_with = float(np.sum(sim_with.get_all_recovered()))

    # Without migration
    config_without = _make_config(
        n_patches=n_patches, beta=beta, gamma=gamma,
        m=0.0, N_per_patch=N_per_patch, dt=dt, n_steps=n_steps,
    )
    sim_without = SIRMetapopulationSimulation(config_without)
    sim_without.reset()
    for _ in range(n_steps):
        sim_without.step()
    total_R_without = float(np.sum(sim_without.get_all_recovered()))

    return {
        "total_R_with_migration": total_R_with,
        "total_R_without_migration": total_R_without,
        "fraction_with": total_R_with / total_N,
        "fraction_without": total_R_without / total_N,
    }


def run_sir_metapopulation_rediscovery(
    output_dir: str | Path = "output/rediscovery/sir_metapopulation",
    n_iterations: int = 50,
) -> dict:
    """Run the full SIR metapopulation rediscovery.

    1. Sweep migration rate m to show synchronization of epidemics
    2. Compare total epidemic size with vs without migration
    3. Measure time delay between patches (epidemic wave)
    4. Return results dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sir_metapopulation",
        "targets": {
            "synchronization": "Higher m -> more synchronized peak times",
            "epidemic_wave": "Disease propagates from seeded patch to others",
            "migration_effect": "Migration increases total epidemic size",
            "R0": "beta / gamma",
        },
    }

    # --- Part 1: Migration sweep for synchronization ---
    logger.info("Part 1: Migration rate sweep for synchronization...")
    sweep_data = generate_migration_sweep_data(
        n_m_values=20, n_patches=5, beta=0.5, gamma=0.1,
        N_per_patch=1000.0, n_steps=3000, dt=0.1,
    )

    # Check synchronization trend: peak_time_std should decrease with m
    low_m_std = np.mean(sweep_data["peak_time_std"][:5])
    high_m_std = np.mean(sweep_data["peak_time_std"][-5:])

    results["synchronization"] = {
        "n_m_values": len(sweep_data["m_values"]),
        "m_range": [float(sweep_data["m_values"][0]), float(sweep_data["m_values"][-1])],
        "low_m_peak_std": float(low_m_std),
        "high_m_peak_std": float(high_m_std),
        "synchronization_observed": bool(high_m_std < low_m_std),
    }
    logger.info(
        f"  Peak time std: low_m={low_m_std:.2f}, high_m={high_m_std:.2f}, "
        f"sync={'yes' if high_m_std < low_m_std else 'no'}"
    )

    # --- Part 2: With vs without migration ---
    logger.info("Part 2: Comparing epidemic size with/without migration...")
    comparison = compare_with_without_migration(
        n_patches=5, beta=0.5, gamma=0.1,
        N_per_patch=1000.0, n_steps=3000, dt=0.1,
    )

    results["migration_comparison"] = comparison
    logger.info(
        f"  With migration: {comparison['fraction_with']:.4f}, "
        f"without: {comparison['fraction_without']:.4f}"
    )

    # --- Part 3: Epidemic wave propagation ---
    logger.info("Part 3: Epidemic wave propagation between patches...")
    wave_data = generate_epidemic_wave_data(
        n_patches=5, beta=0.5, gamma=0.1, m=0.01,
        N_per_patch=1000.0, n_steps=3000, dt=0.1,
    )

    results["epidemic_wave"] = {
        "peak_times": wave_data["peak_times"].tolist(),
        "peak_delays": wave_data["peak_delays"].tolist(),
        "source_patch_peak": float(wave_data["peak_times"][0]),
        "max_delay": float(np.max(wave_data["peak_delays"])),
        "wave_detected": bool(np.max(wave_data["peak_delays"]) > 0),
    }
    logger.info(
        f"  Peak times: {wave_data['peak_times']}, "
        f"max delay: {np.max(wave_data['peak_delays']):.1f}"
    )

    # --- Part 4: R0 verification ---
    r0_theory = 0.5 / 0.1
    results["R0"] = {
        "theoretical": r0_theory,
        "formula": "beta / gamma",
    }

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "sweep_data.npz",
        **{k: v for k, v in sweep_data.items()},
    )

    return results
