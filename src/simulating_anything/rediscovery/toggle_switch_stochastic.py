"""Stochastic toggle switch rediscovery.

Targets:
- Switching rate vs noise sigma: higher noise -> more frequent switching
- Ensemble comparison: stochastic trajectories vs deterministic bistable regions
- Basin of attraction mapping from deterministic nullclines
- Dwell time statistics in each stable state
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.toggle_switch_stochastic import (
    ToggleSwitchStochasticSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_switching_rate_data(
    n_sigma: int = 15,
    sigma_min: float = 0.05,
    sigma_max: float = 2.0,
    n_steps: int = 50000,
    dt: float = 0.01,
    n_trials: int = 5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Sweep noise intensity sigma and measure switching rate between states.

    For each sigma value, runs multiple trials and averages the switch count
    and mean dwell times.

    Args:
        n_sigma: number of sigma values in sweep.
        sigma_min: minimum noise intensity.
        sigma_max: maximum noise intensity.
        n_steps: simulation steps per trial.
        dt: timestep.
        n_trials: number of independent trials per sigma.
        seed: base random seed.

    Returns:
        Dictionary with sigma, switch_count, mean_dwell_high, mean_dwell_low arrays.
    """
    sigma_values = np.linspace(sigma_min, sigma_max, n_sigma)
    switch_counts = np.zeros(n_sigma)
    mean_dwell_high = np.zeros(n_sigma)
    mean_dwell_low = np.zeros(n_sigma)

    for i, sigma in enumerate(sigma_values):
        trial_switches = []
        trial_dwell_h = []
        trial_dwell_l = []

        for t in range(n_trials):
            config = SimulationConfig(
                domain=Domain.TOGGLE_SWITCH_STOCHASTIC,
                dt=dt,
                n_steps=n_steps,
                seed=seed + i * n_trials + t,
                parameters={
                    "alpha1": 5.0, "alpha2": 5.0,
                    "beta_hill": 2.0, "gamma_hill": 2.0,
                    "sigma": float(sigma),
                    "u_0": 0.5, "v_0": 0.5,
                },
            )
            sim = ToggleSwitchStochasticSimulation(config)
            traj = sim.run(n_steps=n_steps)
            u_trace = traj.states[:, 0]

            trial_switches.append(sim.count_switches(u_trace))

            dwell = sim.dwell_times(u_trace)
            if len(dwell["high_state_times"]) > 0:
                trial_dwell_h.append(float(np.mean(dwell["high_state_times"])))
            if len(dwell["low_state_times"]) > 0:
                trial_dwell_l.append(float(np.mean(dwell["low_state_times"])))

        switch_counts[i] = np.mean(trial_switches)
        mean_dwell_high[i] = np.mean(trial_dwell_h) if trial_dwell_h else 0.0
        mean_dwell_low[i] = np.mean(trial_dwell_l) if trial_dwell_l else 0.0

        if (i + 1) % 5 == 0:
            logger.info(
                f"  sigma={sigma:.3f}: switches={switch_counts[i]:.1f}, "
                f"dwell_high={mean_dwell_high[i]:.2f}, "
                f"dwell_low={mean_dwell_low[i]:.2f}"
            )

    return {
        "sigma": sigma_values,
        "switch_count": switch_counts,
        "mean_dwell_high": mean_dwell_high,
        "mean_dwell_low": mean_dwell_low,
        "n_steps": n_steps,
        "dt": dt,
        "n_trials": n_trials,
    }


def generate_ensemble_comparison(
    n_ensemble: int = 20,
    n_steps: int = 20000,
    dt: float = 0.01,
    sigma: float = 0.5,
    seed: int = 100,
) -> dict[str, object]:
    """Compare stochastic ensemble with deterministic bistable regions.

    Runs an ensemble of stochastic trajectories and checks what fraction
    end up in each basin, compared to deterministic predictions.

    Args:
        n_ensemble: number of stochastic trajectories.
        n_steps: simulation steps per trajectory.
        dt: timestep.
        sigma: noise intensity for stochastic runs.
        seed: base random seed.

    Returns:
        Dictionary with ensemble statistics and deterministic fixed points.
    """
    # Deterministic fixed points
    det_config = SimulationConfig(
        domain=Domain.TOGGLE_SWITCH_STOCHASTIC,
        dt=dt,
        n_steps=100,
        seed=seed,
        parameters={
            "alpha1": 5.0, "alpha2": 5.0,
            "beta_hill": 2.0, "gamma_hill": 2.0,
            "sigma": 0.0,
            "u_0": 0.5, "v_0": 0.5,
        },
    )
    det_sim = ToggleSwitchStochasticSimulation(det_config)
    fixed_points = det_sim.find_fixed_points()
    fp_classes = []
    for u_fp, v_fp in fixed_points:
        fp_classes.append({
            "u": u_fp, "v": v_fp,
            "type": det_sim.classify_fixed_point(u_fp, v_fp),
        })

    # Stochastic ensemble
    final_states = []
    switch_counts = []

    for i in range(n_ensemble):
        config = SimulationConfig(
            domain=Domain.TOGGLE_SWITCH_STOCHASTIC,
            dt=dt,
            n_steps=n_steps,
            seed=seed + i,
            parameters={
                "alpha1": 5.0, "alpha2": 5.0,
                "beta_hill": 2.0, "gamma_hill": 2.0,
                "sigma": sigma,
                "u_0": 0.5, "v_0": 0.5,
            },
        )
        sim = ToggleSwitchStochasticSimulation(config)
        traj = sim.run(n_steps=n_steps)
        final_states.append(traj.states[-1].copy())
        switch_counts.append(sim.count_switches(traj.states[:, 0]))

    final_states_arr = np.array(final_states)
    n_high_u = int(np.sum(final_states_arr[:, 0] > final_states_arr[:, 1]))
    n_high_v = int(n_ensemble - n_high_u)

    return {
        "fixed_points": fp_classes,
        "n_fixed_points": len(fixed_points),
        "n_ensemble": n_ensemble,
        "sigma": float(sigma),
        "n_high_u": n_high_u,
        "n_high_v": n_high_v,
        "frac_high_u": float(n_high_u / n_ensemble),
        "frac_high_v": float(n_high_v / n_ensemble),
        "mean_switches": float(np.mean(switch_counts)),
        "mean_final_u": float(np.mean(final_states_arr[:, 0])),
        "mean_final_v": float(np.mean(final_states_arr[:, 1])),
    }


def generate_basin_map(
    n_grid: int = 15,
    u_range: tuple[float, float] = (0.0, 5.5),
    v_range: tuple[float, float] = (0.0, 5.5),
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Map basins of attraction from deterministic nullclines.

    Runs deterministic trajectories from a grid of initial conditions and
    records which stable state each converges to.

    Args:
        n_grid: grid resolution in each dimension.
        u_range: range of initial u values.
        v_range: range of initial v values.
        n_steps: steps to run each trajectory.
        dt: timestep.

    Returns:
        Dictionary with u_grid, v_grid, basin arrays.
    """
    u_vals = np.linspace(u_range[0], u_range[1], n_grid)
    v_vals = np.linspace(v_range[0], v_range[1], n_grid)
    basin = np.zeros((n_grid, n_grid), dtype=np.int32)

    for i, u0 in enumerate(u_vals):
        for j, v0 in enumerate(v_vals):
            config = SimulationConfig(
                domain=Domain.TOGGLE_SWITCH_STOCHASTIC,
                dt=dt,
                n_steps=n_steps,
                seed=42,
                parameters={
                    "alpha1": 5.0, "alpha2": 5.0,
                    "beta_hill": 2.0, "gamma_hill": 2.0,
                    "sigma": 0.0,
                    "u_0": float(u0), "v_0": float(v0),
                },
            )
            sim = ToggleSwitchStochasticSimulation(config)
            traj = sim.run(n_steps=n_steps)
            final = traj.states[-1]
            # +1 = high u basin, -1 = high v basin, 0 = boundary
            if final[0] > final[1] + 0.1:
                basin[i, j] = 1
            elif final[1] > final[0] + 0.1:
                basin[i, j] = -1

    # Also get nullclines for reference
    ref_config = SimulationConfig(
        domain=Domain.TOGGLE_SWITCH_STOCHASTIC,
        dt=dt, n_steps=100, seed=42,
        parameters={
            "alpha1": 5.0, "alpha2": 5.0,
            "beta_hill": 2.0, "gamma_hill": 2.0,
            "sigma": 0.0,
        },
    )
    ref_sim = ToggleSwitchStochasticSimulation(ref_config)
    nullclines = ref_sim.compute_nullclines()

    return {
        "u_grid": u_vals,
        "v_grid": v_vals,
        "basin": basin,
        "nullclines": nullclines,
    }


def run_toggle_switch_stochastic_rediscovery(
    output_dir: str | Path = "output/rediscovery/toggle_switch_stochastic",
    n_sigma: int = 15,
    n_trials: int = 5,
) -> dict:
    """Run stochastic toggle switch rediscovery pipeline.

    Demonstrates:
    1. Switching rate increases with noise intensity sigma
    2. Ensemble stochastic trajectories explore both basins
    3. Deterministic basin of attraction structure from nullclines

    Args:
        output_dir: directory for results output.
        n_sigma: number of noise intensity values in sweep.
        n_trials: number of independent trials per sigma.

    Returns:
        Results dictionary with switching rate, ensemble, and basin data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "toggle_switch_stochastic",
        "targets": {
            "switching_rate_vs_sigma": "Higher noise -> more frequent switching",
            "ensemble_comparison": "Stochastic ensemble vs deterministic basins",
            "basin_mapping": "Deterministic basins from nullcline analysis",
        },
    }

    # --- Part 1: Switching rate vs noise ---
    logger.info("Part 1: Switching rate vs noise intensity...")
    sweep_data = generate_switching_rate_data(
        n_sigma=n_sigma,
        sigma_min=0.05,
        sigma_max=2.0,
        n_steps=50000,
        dt=0.01,
        n_trials=n_trials,
        seed=42,
    )

    # Check that switching increases with sigma (monotonic trend)
    switches = sweep_data["switch_count"]
    # Correlation between sigma and switching
    corr = float(np.corrcoef(sweep_data["sigma"], switches)[0, 1])
    results["switching_rate"] = {
        "sigma_switch_correlation": corr,
        "max_switches": float(np.max(switches)),
        "sigma_at_max_switches": float(
            sweep_data["sigma"][np.argmax(switches)]
        ),
        "min_switches": float(np.min(switches)),
    }
    logger.info(f"  Sigma-switch correlation: {corr:.4f}")

    # --- Part 2: Ensemble comparison ---
    logger.info("Part 2: Stochastic ensemble comparison...")
    ensemble = generate_ensemble_comparison(
        n_ensemble=20,
        n_steps=20000,
        dt=0.01,
        sigma=0.5,
        seed=100,
    )
    results["ensemble"] = ensemble

    # --- Part 3: Basin of attraction map ---
    logger.info("Part 3: Basin of attraction mapping...")
    basin_data = generate_basin_map(
        n_grid=10,
        n_steps=5000,
        dt=0.01,
    )
    n_high_u = int(np.sum(basin_data["basin"] == 1))
    n_high_v = int(np.sum(basin_data["basin"] == -1))
    n_boundary = int(np.sum(basin_data["basin"] == 0))
    results["basins"] = {
        "n_high_u_basin": n_high_u,
        "n_high_v_basin": n_high_v,
        "n_boundary": n_boundary,
        "grid_size": 10,
    }
    logger.info(
        f"  Basins: high_u={n_high_u}, high_v={n_high_v}, boundary={n_boundary}"
    )

    # --- Save results ---
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "switching_rate.npz",
        sigma=sweep_data["sigma"],
        switch_count=sweep_data["switch_count"],
        mean_dwell_high=sweep_data["mean_dwell_high"],
        mean_dwell_low=sweep_data["mean_dwell_low"],
    )

    return results
