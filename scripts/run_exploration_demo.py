"""Demonstrate uncertainty-driven exploration on real simulation domains.

Runs the UncertaintyDrivenExplorer on Lotka-Volterra (predator-prey oscillations)
and SIR (epidemic dynamics), showing how the explorer discovers interesting
parameter regions: high-amplitude oscillations for LV, and the R0=1 epidemic
threshold for SIR.

Runs on Windows with numpy only (no JAX needed).

Usage:
    python scripts/run_exploration_demo.py
    python scripts/run_exploration_demo.py --n_rounds 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulating_anything.exploration.uncertainty_driven import UncertaintyDrivenExplorer
from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
from simulating_anything.simulation.epidemiological import SIRSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig
from simulating_anything.types.trajectory import TrajectoryData, TrajectoryMetadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/exploration")

# Fixed parameters for Lotka-Volterra (only alpha and beta are swept)
LV_FIXED_PARAMS = {
    "gamma": 0.4,
    "delta": 0.1,
    "prey_0": 40.0,
    "predator_0": 9.0,
}

# Fixed parameters for SIR (only beta and gamma are swept)
SIR_FIXED_PARAMS = {
    "S_0": 0.99,
    "I_0": 0.01,
    "R_0_init": 0.0,
}


def compute_lv_oscillation_amplitude(states: np.ndarray) -> float:
    """Compute the peak-to-trough amplitude of prey oscillations.

    Uses the second half of the trajectory to avoid transient behavior.
    Returns amplitude normalized by mean population (dimensionless metric).
    """
    prey = states[:, 0]
    # Use second half to skip transients
    half = len(prey) // 2
    prey_stable = prey[half:]

    if len(prey_stable) < 10:
        return 0.0

    mean_pop = np.mean(prey_stable)
    if mean_pop < 1e-6:
        return 0.0

    amplitude = np.max(prey_stable) - np.min(prey_stable)
    return float(amplitude / mean_pop)


def compute_sir_peak_infected(states: np.ndarray) -> float:
    """Return the peak infected fraction from an SIR trajectory."""
    infected = states[:, 1]
    return float(np.max(infected))


def run_lv_exploration(n_rounds: int, n_points_per_dim: int = 10) -> dict:
    """Explore Lotka-Volterra parameter space for oscillation dynamics.

    Sweeps alpha (prey growth rate) and beta (predation rate) to find
    regions with large-amplitude oscillations versus damped/extinct dynamics.

    Args:
        n_rounds: Number of exploration rounds.
        n_points_per_dim: Grid resolution per parameter dimension.

    Returns:
        Results dict with exploration metrics, parameter map, and trajectories.
    """
    logger.info("=" * 60)
    logger.info("LOTKA-VOLTERRA EXPLORATION")
    logger.info("=" * 60)

    sweep_ranges = {
        "alpha": (0.5, 2.0),
        "beta": (0.1, 0.8),
    }

    explorer = UncertaintyDrivenExplorer(
        sweep_ranges=sweep_ranges,
        n_points_per_dim=n_points_per_dim,
        uncertainty_threshold=0.3,
        novelty_weight=0.5,
        seed=42,
    )

    explored_params = []
    amplitudes = []
    t0 = time.time()

    for round_idx in range(n_rounds):
        params = explorer.propose_parameters()

        # Merge swept params with fixed params
        full_params = {**LV_FIXED_PARAMS, **params}

        config = SimulationConfig(
            domain=Domain.AGENT_BASED,
            dt=0.01,
            n_steps=5000,
            parameters=full_params,
        )

        sim = LotkaVolterraSimulation(config)
        traj = sim.run(n_steps=5000)

        # Compute oscillation amplitude as the interestingness metric
        amplitude = compute_lv_oscillation_amplitude(traj.states)
        is_interesting = amplitude > 1.0  # Normalized amplitude > 1 is notable
        novelty_score = min(amplitude / 3.0, 1.0)  # Scale to [0, 1]

        # Build trajectory with metadata for the explorer
        explore_traj = TrajectoryData(
            parameters={**params},
            metadata=TrajectoryMetadata(
                novelty_score=novelty_score,
                interesting=is_interesting,
                explorer_strategy="uncertainty_driven",
            ),
        )
        explore_traj.states = traj.states

        explorer.update(explore_traj)

        # Boost uncertainty near interesting points to encourage local exploration
        if is_interesting:
            point = np.array([params[n] for n in explorer.param_names])
            dists = np.linalg.norm(explorer._grid_physical - point, axis=1)
            nearby = dists < 0.15 * np.max(dists)
            explorer._uncertainties[nearby] = np.minimum(
                explorer._uncertainties[nearby] * 1.5, 1.0
            )

        explored_params.append(params)
        amplitudes.append(amplitude)

        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            progress = explorer.get_exploration_progress()
            logger.info(
                f"  Round {round_idx + 1}/{n_rounds}: "
                f"alpha={params['alpha']:.3f}, beta={params['beta']:.3f}, "
                f"amplitude={amplitude:.2f}, "
                f"coverage={progress['coverage_fraction']:.1%}"
            )

    elapsed = time.time() - t0
    progress = explorer.get_exploration_progress()

    # Build discovery map: which regions have high oscillation amplitude
    alpha_vals = np.array([p["alpha"] for p in explored_params])
    beta_vals = np.array([p["beta"] for p in explored_params])
    amp_vals = np.array(amplitudes)

    # Identify interesting region
    interesting_mask = amp_vals > 1.0
    n_interesting = int(np.sum(interesting_mask))

    logger.info("-" * 40)
    logger.info(f"LV Exploration complete in {elapsed:.1f}s")
    logger.info(f"  Rounds: {n_rounds}, Coverage: {progress['coverage_fraction']:.1%}")
    logger.info(f"  Interesting points: {n_interesting}/{n_rounds}")
    if n_interesting > 0:
        logger.info(
            f"  Interesting alpha range: "
            f"[{np.min(alpha_vals[interesting_mask]):.3f}, "
            f"{np.max(alpha_vals[interesting_mask]):.3f}]"
        )
        logger.info(
            f"  Interesting beta range: "
            f"[{np.min(beta_vals[interesting_mask]):.3f}, "
            f"{np.max(beta_vals[interesting_mask]):.3f}]"
        )
    logger.info(f"  Max amplitude: {np.max(amp_vals):.3f}")
    logger.info(f"  Mean amplitude: {np.mean(amp_vals):.3f}")

    results = {
        "domain": "lotka_volterra",
        "sweep_ranges": {k: list(v) for k, v in sweep_ranges.items()},
        "fixed_params": LV_FIXED_PARAMS,
        "n_rounds": n_rounds,
        "elapsed_s": elapsed,
        "exploration_progress": progress,
        "n_interesting": n_interesting,
        "max_amplitude": float(np.max(amp_vals)),
        "mean_amplitude": float(np.mean(amp_vals)),
        "interesting_alpha_range": (
            [float(np.min(alpha_vals[interesting_mask])),
             float(np.max(alpha_vals[interesting_mask]))]
            if n_interesting > 0 else None
        ),
        "interesting_beta_range": (
            [float(np.min(beta_vals[interesting_mask])),
             float(np.max(beta_vals[interesting_mask]))]
            if n_interesting > 0 else None
        ),
    }

    # Save parameter map as npz
    lv_dir = OUTPUT_DIR / "lotka_volterra"
    lv_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        lv_dir / "discovery_map.npz",
        alpha=alpha_vals,
        beta=beta_vals,
        amplitude=amp_vals,
        interesting=interesting_mask,
        grid_physical=explorer._grid_physical,
        grid_uncertainties=explorer._uncertainties,
        grid_visited=explorer._visited,
    )
    logger.info(f"  Data saved to {lv_dir / 'discovery_map.npz'}")

    return results


def run_sir_exploration(n_rounds: int, n_points_per_dim: int = 10) -> dict:
    """Explore SIR parameter space to identify the R0=1 epidemic threshold.

    Sweeps beta (transmission rate) and gamma (recovery rate). The critical
    boundary is R0 = beta/gamma = 1: above this, epidemics occur (peak_I > 0.1);
    below, the disease dies out.

    Args:
        n_rounds: Number of exploration rounds.
        n_points_per_dim: Grid resolution per parameter dimension.

    Returns:
        Results dict with exploration metrics and R0 boundary analysis.
    """
    logger.info("=" * 60)
    logger.info("SIR EPIDEMIC EXPLORATION")
    logger.info("=" * 60)

    sweep_ranges = {
        "beta": (0.1, 0.8),
        "gamma": (0.02, 0.3),
    }

    explorer = UncertaintyDrivenExplorer(
        sweep_ranges=sweep_ranges,
        n_points_per_dim=n_points_per_dim,
        uncertainty_threshold=0.3,
        novelty_weight=0.5,
        seed=123,
    )

    explored_params = []
    peak_infecteds = []
    r0_values = []
    epidemic_flags = []
    t0 = time.time()

    for round_idx in range(n_rounds):
        params = explorer.propose_parameters()

        full_params = {**SIR_FIXED_PARAMS, **params}

        config = SimulationConfig(
            domain=Domain.EPIDEMIOLOGICAL,
            dt=0.01,
            n_steps=5000,
            parameters=full_params,
        )

        sim = SIRSimulation(config)
        traj = sim.run(n_steps=5000)

        peak_I = compute_sir_peak_infected(traj.states)
        r0 = params["beta"] / params["gamma"]
        is_epidemic = peak_I > 0.1

        # Points near the R0=1 boundary are most interesting for discovery
        distance_to_boundary = abs(r0 - 1.0)
        novelty_score = max(0.0, 1.0 - distance_to_boundary)

        explore_traj = TrajectoryData(
            parameters={**params},
            metadata=TrajectoryMetadata(
                novelty_score=novelty_score,
                interesting=distance_to_boundary < 0.3,
                explorer_strategy="uncertainty_driven",
            ),
        )
        explore_traj.states = traj.states

        explorer.update(explore_traj)

        # Boost uncertainty near the R0=1 boundary to focus exploration there
        if distance_to_boundary < 0.5:
            point = np.array([params[n] for n in explorer.param_names])
            dists = np.linalg.norm(explorer._grid_physical - point, axis=1)
            nearby = dists < 0.1 * np.max(dists)
            explorer._uncertainties[nearby] = np.minimum(
                explorer._uncertainties[nearby] * 1.3, 1.0
            )

        explored_params.append(params)
        peak_infecteds.append(peak_I)
        r0_values.append(r0)
        epidemic_flags.append(is_epidemic)

        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            progress = explorer.get_exploration_progress()
            logger.info(
                f"  Round {round_idx + 1}/{n_rounds}: "
                f"beta={params['beta']:.3f}, gamma={params['gamma']:.3f}, "
                f"R0={r0:.2f}, peak_I={peak_I:.4f}, "
                f"epidemic={'YES' if is_epidemic else 'no '}, "
                f"coverage={progress['coverage_fraction']:.1%}"
            )

    elapsed = time.time() - t0
    progress = explorer.get_exploration_progress()

    beta_vals = np.array([p["beta"] for p in explored_params])
    gamma_vals = np.array([p["gamma"] for p in explored_params])
    peak_vals = np.array(peak_infecteds)
    r0_vals = np.array(r0_values)
    epi_flags = np.array(epidemic_flags)

    n_epidemics = int(np.sum(epi_flags))
    n_no_epidemic = n_rounds - n_epidemics

    # Find boundary points: R0 close to 1
    boundary_mask = np.abs(r0_vals - 1.0) < 0.3
    n_boundary = int(np.sum(boundary_mask))

    # Verify that the explorer correctly separates epidemic vs non-epidemic
    # by checking whether R0 > 1 correlates with epidemic occurrence
    r0_above_1 = r0_vals > 1.0
    classification_accuracy = float(np.mean(r0_above_1 == epi_flags))

    logger.info("-" * 40)
    logger.info(f"SIR Exploration complete in {elapsed:.1f}s")
    logger.info(f"  Rounds: {n_rounds}, Coverage: {progress['coverage_fraction']:.1%}")
    logger.info(f"  Epidemics: {n_epidemics}/{n_rounds}, No epidemic: {n_no_epidemic}/{n_rounds}")
    logger.info(f"  Boundary points (|R0-1| < 0.3): {n_boundary}")
    logger.info(f"  R0>1 predicts epidemic: {classification_accuracy:.1%} accuracy")
    logger.info(f"  Max peak infected: {np.max(peak_vals):.4f}")

    results = {
        "domain": "sir",
        "sweep_ranges": {k: list(v) for k, v in sweep_ranges.items()},
        "fixed_params": SIR_FIXED_PARAMS,
        "n_rounds": n_rounds,
        "elapsed_s": elapsed,
        "exploration_progress": progress,
        "n_epidemics": n_epidemics,
        "n_no_epidemic": n_no_epidemic,
        "n_boundary_points": n_boundary,
        "r0_predicts_epidemic_accuracy": classification_accuracy,
        "max_peak_infected": float(np.max(peak_vals)),
    }

    # Save parameter map as npz
    sir_dir = OUTPUT_DIR / "sir"
    sir_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        sir_dir / "discovery_map.npz",
        beta=beta_vals,
        gamma=gamma_vals,
        peak_infected=peak_vals,
        r0=r0_vals,
        epidemic=epi_flags,
        boundary=boundary_mask,
        grid_physical=explorer._grid_physical,
        grid_uncertainties=explorer._uncertainties,
        grid_visited=explorer._visited,
    )
    logger.info(f"  Data saved to {sir_dir / 'discovery_map.npz'}")

    return results


def print_summary_table(lv_results: dict, sir_results: dict) -> None:
    """Print a formatted summary table of both explorations."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPLORATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<40} {'Lotka-Volterra':>14} {'SIR':>14}")
    logger.info("-" * 70)

    rows = [
        ("Rounds", str(lv_results["n_rounds"]), str(sir_results["n_rounds"])),
        ("Time (s)", f"{lv_results['elapsed_s']:.1f}", f"{sir_results['elapsed_s']:.1f}"),
        (
            "Coverage",
            f"{lv_results['exploration_progress']['coverage_fraction']:.1%}",
            f"{sir_results['exploration_progress']['coverage_fraction']:.1%}",
        ),
        (
            "Grid points",
            str(lv_results["exploration_progress"]["total_grid_points"]),
            str(sir_results["exploration_progress"]["total_grid_points"]),
        ),
        (
            "Mean uncertainty",
            f"{lv_results['exploration_progress']['mean_uncertainty']:.4f}",
            f"{sir_results['exploration_progress']['mean_uncertainty']:.4f}",
        ),
        (
            "Interesting points found",
            str(lv_results["n_interesting"]),
            str(sir_results["n_boundary_points"]),
        ),
        (
            "Key metric (max)",
            f"amp={lv_results['max_amplitude']:.2f}",
            f"peak_I={sir_results['max_peak_infected']:.4f}",
        ),
        (
            "Discovery",
            "Oscillation regions",
            f"R0 boundary ({sir_results['r0_predicts_epidemic_accuracy']:.0%} acc)",
        ),
    ]

    for label, lv_val, sir_val in rows:
        logger.info(f"  {label:<38} {lv_val:>14} {sir_val:>14}")

    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate uncertainty-driven exploration on simulation domains"
    )
    parser.add_argument(
        "--n_rounds", type=int, default=30,
        help="Number of exploration rounds per domain (default: 30)",
    )
    parser.add_argument(
        "--n_points_per_dim", type=int, default=10,
        help="Grid resolution per parameter dimension (default: 10)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Run explorations
    lv_results = run_lv_exploration(
        n_rounds=args.n_rounds,
        n_points_per_dim=args.n_points_per_dim,
    )
    sir_results = run_sir_exploration(
        n_rounds=args.n_rounds,
        n_points_per_dim=args.n_points_per_dim,
    )

    total_time = time.time() - t0

    # Print summary
    print_summary_table(lv_results, sir_results)
    logger.info(f"Total time: {total_time:.1f}s")

    # Save combined results JSON
    combined = {
        "total_time_s": total_time,
        "n_rounds_per_domain": args.n_rounds,
        "n_points_per_dim": args.n_points_per_dim,
        "lotka_volterra": lv_results,
        "sir": sir_results,
    }

    results_file = OUTPUT_DIR / "exploration_results.json"
    with open(results_file, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
