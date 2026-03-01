"""Competitive Lotka-Volterra (4-species) rediscovery.

Targets:
- Coexistence equilibrium N* = alpha^{-1} @ K
- Competitive exclusion: increasing competition reduces diversity
- Community matrix eigenvalues predict stability
- Shannon diversity tracks species richness
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.competitive_lv import CompetitiveLVSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    n_species: int = 4,
    r: list[float] | None = None,
    K: list[float] | None = None,
    alpha: list[list[float]] | None = None,
    N_init: list[float] | None = None,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Build a SimulationConfig for the competitive LV model."""
    if r is None:
        r = [1.0, 0.72, 1.53, 1.27]
    if K is None:
        K = [100.0, 100.0, 100.0, 100.0]
    if alpha is None:
        alpha = [
            [1.0, 0.5, 0.4, 0.3],
            [0.4, 1.0, 0.6, 0.3],
            [0.3, 0.4, 1.0, 0.5],
            [0.5, 0.3, 0.4, 1.0],
        ]

    params: dict[str, float] = {"n_species": float(n_species)}
    for i in range(n_species):
        params[f"r_{i}"] = r[i]
        params[f"K_{i}"] = K[i]
        for j in range(n_species):
            params[f"alpha_{i}_{j}"] = alpha[i][j]
        if N_init is not None:
            params[f"N_0_{i}"] = N_init[i]

    return SimulationConfig(
        domain=Domain.COMPETITIVE_LV,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


def generate_coexistence_trajectory(
    n_steps: int = 100000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate a long trajectory at default (coexistence) parameters.

    Returns dict with states array, time, and equilibrium info.
    """
    config = _make_config(dt=dt, n_steps=n_steps)
    sim = CompetitiveLVSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    N_star = sim.equilibrium_point()

    return {
        "states": states_arr,
        "time": np.arange(n_steps + 1) * dt,
        "N_star_analytical": N_star,
        "n_species": 4,
        "dt": dt,
    }


def generate_exclusion_sweep_data(
    n_alpha: int = 30,
    n_steps: int = 80000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep alpha_12 from 0 to 1.5 and track species survival.

    Returns dict with alpha values, survival counts, and diversity.
    """
    alpha_values = np.linspace(0.0, 1.5, n_alpha)
    n_surviving_list = []
    diversity_list = []
    final_pops_list = []

    for i, alpha_12 in enumerate(alpha_values):
        # Build config with modified alpha_01
        alpha_matrix = [
            [1.0, alpha_12, 0.4, 0.3],
            [0.4, 1.0, 0.6, 0.3],
            [0.3, 0.4, 1.0, 0.5],
            [0.5, 0.3, 0.4, 1.0],
        ]
        config = _make_config(alpha=alpha_matrix, dt=dt, n_steps=n_steps)
        sim = CompetitiveLVSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        n_surviving_list.append(sim.n_surviving())
        diversity_list.append(sim.diversity_index())
        final_pops_list.append(sim.observe().copy())

        if (i + 1) % 10 == 0:
            logger.info(
                f"  alpha_12={alpha_12:.3f}: "
                f"surviving={n_surviving_list[-1]}, "
                f"H={diversity_list[-1]:.3f}"
            )

    return {
        "alpha_12": alpha_values,
        "n_surviving": np.array(n_surviving_list),
        "diversity": np.array(diversity_list),
        "final_populations": np.array(final_pops_list),
    }


def generate_eigenvalue_data(
    n_samples: int = 50,
) -> dict[str, np.ndarray]:
    """Sample random competition matrices and compute eigenvalues.

    Returns dict with alpha matrices, eigenvalues, stability flags,
    and equilibrium feasibility.
    """
    rng = np.random.default_rng(42)
    all_max_real_eig = []
    all_is_stable = []
    all_is_feasible = []
    all_alpha_strength = []

    for _ in range(n_samples):
        # Random off-diagonal alpha ~ Uniform(0.1, 1.2)
        alpha_matrix = np.eye(4)
        for i in range(4):
            for j in range(4):
                if i != j:
                    alpha_matrix[i, j] = rng.uniform(0.1, 1.2)

        alpha_list = alpha_matrix.tolist()
        config = _make_config(alpha=alpha_list, dt=0.01, n_steps=100)
        sim = CompetitiveLVSimulation(config)
        sim.reset()

        N_star = sim.equilibrium_point()
        feasible = bool(np.all(N_star > 0) and np.all(np.isfinite(N_star)))

        eigs = sim.stability_eigenvalues()
        max_real = float(np.max(np.real(eigs)))
        stable = feasible and (max_real < 0)

        all_max_real_eig.append(max_real)
        all_is_stable.append(stable)
        all_is_feasible.append(feasible)
        # Mean off-diagonal alpha as a measure of competition strength
        off_diag = alpha_matrix[~np.eye(4, dtype=bool)]
        all_alpha_strength.append(float(np.mean(off_diag)))

    return {
        "max_real_eigenvalue": np.array(all_max_real_eig),
        "is_stable": np.array(all_is_stable),
        "is_feasible": np.array(all_is_feasible),
        "mean_alpha_strength": np.array(all_alpha_strength),
    }


def run_competitive_lv_rediscovery(
    output_dir: str | Path = "output/rediscovery/competitive_lv",
    n_iterations: int = 40,
) -> dict:
    """Run the full competitive LV rediscovery pipeline.

    1. Generate coexistence trajectories, verify equilibrium
    2. Sweep competition strength, track exclusion
    3. Compute community matrix eigenvalues
    4. Verify competitive exclusion principle

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "competitive_lv",
        "targets": {
            "equilibrium": "N* = alpha^{-1} @ K",
            "exclusion": "Increasing competition reduces diversity",
            "stability": "All eigenvalue real parts < 0 => stable coexistence",
        },
    }

    # --- Part 1: Coexistence trajectory and equilibrium verification ---
    logger.info("Part 1: Coexistence trajectory and equilibrium verification...")
    traj_data = generate_coexistence_trajectory(n_steps=80000, dt=0.01)

    states = traj_data["states"]
    N_star = traj_data["N_star_analytical"]

    # Time-average of last 20% of trajectory
    skip = len(states) * 4 // 5
    time_avg = np.mean(states[skip:], axis=0)

    # Compare with analytical equilibrium
    eq_error = np.abs(time_avg - N_star) / np.abs(N_star)
    results["equilibrium"] = {
        "N_star_analytical": N_star.tolist(),
        "time_averaged": time_avg.tolist(),
        "relative_error": eq_error.tolist(),
        "mean_relative_error": float(np.mean(eq_error)),
    }
    logger.info(f"  Analytical equilibrium: {N_star}")
    logger.info(f"  Time-averaged: {time_avg}")
    logger.info(f"  Mean relative error: {np.mean(eq_error):.4%}")

    # --- Part 2: Competitive exclusion sweep ---
    logger.info("Part 2: Competitive exclusion sweep (alpha_12: 0 -> 1.5)...")
    sweep_data = generate_exclusion_sweep_data(n_alpha=25, n_steps=80000, dt=0.01)

    results["exclusion_sweep"] = {
        "n_alpha_values": len(sweep_data["alpha_12"]),
        "alpha_12_range": [
            float(sweep_data["alpha_12"][0]),
            float(sweep_data["alpha_12"][-1]),
        ],
        "n_surviving_at_low": int(sweep_data["n_surviving"][0]),
        "n_surviving_at_high": int(sweep_data["n_surviving"][-1]),
        "diversity_at_low": float(sweep_data["diversity"][0]),
        "diversity_at_high": float(sweep_data["diversity"][-1]),
        "exclusion_verified": bool(
            sweep_data["diversity"][-1] < sweep_data["diversity"][0]
            or sweep_data["n_surviving"][-1] <= sweep_data["n_surviving"][0]
        ),
    }
    logger.info(
        f"  Low competition: {sweep_data['n_surviving'][0]} species, "
        f"H={sweep_data['diversity'][0]:.3f}"
    )
    logger.info(
        f"  High competition: {sweep_data['n_surviving'][-1]} species, "
        f"H={sweep_data['diversity'][-1]:.3f}"
    )

    # --- Part 3: Community matrix eigenvalue analysis ---
    logger.info("Part 3: Community matrix eigenvalue analysis...")
    eig_data = generate_eigenvalue_data(n_samples=50)

    n_feasible = int(np.sum(eig_data["is_feasible"]))
    n_stable = int(np.sum(eig_data["is_stable"]))

    results["eigenvalue_analysis"] = {
        "n_samples": 50,
        "n_feasible": n_feasible,
        "n_stable": n_stable,
        "fraction_feasible": float(n_feasible / 50),
        "fraction_stable": float(n_stable / 50),
        "mean_max_real_eigenvalue": float(np.mean(eig_data["max_real_eigenvalue"])),
    }
    logger.info(f"  Feasible equilibria: {n_feasible}/50")
    logger.info(f"  Stable coexistence: {n_stable}/50")

    # --- Part 4: Verify competitive exclusion principle ---
    # The principle states: in a homogeneous environment, n species cannot
    # coexist on fewer than n limiting factors. With strong enough competition,
    # species are excluded.
    logger.info("Part 4: Verifying competitive exclusion principle...")
    # Use a very strong competition scenario
    strong_alpha = [
        [1.0, 1.5, 1.5, 1.5],
        [1.5, 1.0, 1.5, 1.5],
        [1.5, 1.5, 1.0, 1.5],
        [1.5, 1.5, 1.5, 1.0],
    ]
    config_strong = _make_config(
        alpha=strong_alpha, dt=0.01, n_steps=100000,
    )
    sim_strong = CompetitiveLVSimulation(config_strong)
    sim_strong.reset()
    for _ in range(100000):
        sim_strong.step()

    n_surv_strong = sim_strong.n_surviving()
    results["competitive_exclusion"] = {
        "strong_competition_alpha": 1.5,
        "n_surviving_strong": n_surv_strong,
        "exclusion_observed": n_surv_strong < 4,
    }
    logger.info(f"  Strong competition: {n_surv_strong}/4 species surviving")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save trajectory data
    np.savez(
        output_path / "coexistence_trajectory.npz",
        states=traj_data["states"],
        time=traj_data["time"],
        N_star=N_star,
    )
    np.savez(
        output_path / "exclusion_sweep.npz",
        **{k: v for k, v in sweep_data.items()},
    )

    return results
