"""May-Leonard cyclic competition rediscovery.

Targets:
- Interior fixed point x* = K / (1 + a + b) for symmetric 4-species case
- Heteroclinic cycle period as function of competition strength a
- Cyclic dominance sequence detection (1 -> 2 -> 3 -> 4 -> 1)
- Biodiversity oscillations and total population dynamics
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.may_leonard import MayLeonardSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    n_species: int = 4,
    a: float = 1.5,
    b: float = 0.5,
    r: float = 1.0,
    K: float = 1.0,
    x_init: list[float] | None = None,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Build a SimulationConfig for the May-Leonard model."""
    params: dict[str, float] = {
        "n_species": float(n_species),
        "a": a,
        "b": b,
        "r": r,
        "K": K,
    }
    if x_init is not None:
        for i, val in enumerate(x_init):
            params[f"x_0_{i}"] = val

    return SimulationConfig(
        domain=Domain.MAY_LEONARD,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


def generate_heteroclinic_trajectory(
    n_steps: int = 50000,
    dt: float = 0.01,
    a: float = 1.5,
    b: float = 0.5,
) -> dict[str, np.ndarray | float | int]:
    """Generate a long trajectory showing heteroclinic cycling.

    Returns dict with states, time, dominance sequence, and biodiversity.
    """
    # Use slightly asymmetric initial conditions to break symmetry
    x_init = [0.26, 0.25, 0.24, 0.25]
    config = _make_config(a=a, b=b, x_init=x_init, dt=dt, n_steps=n_steps)
    sim = MayLeonardSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    x_star = sim.compute_interior_fixed_point()
    dominance = sim.compute_dominance_index(states_arr)
    total_pop = sim.compute_total_population(states_arr)

    # Compute biodiversity over time
    biodiv = np.array([
        sim.biodiversity_index(states_arr[i]) for i in range(len(states_arr))
    ])

    return {
        "states": states_arr,
        "time": np.arange(n_steps + 1) * dt,
        "x_star": x_star,
        "dominance": dominance,
        "total_population": total_pop,
        "biodiversity": biodiv,
        "n_species": 4,
        "a": a,
        "b": b,
        "dt": dt,
    }


def generate_competition_sweep_data(
    n_a: int = 25,
    n_steps: int = 40000,
    dt: float = 0.01,
    b: float = 0.5,
) -> dict[str, np.ndarray]:
    """Sweep competition strength a and measure cycle period and biodiversity.

    Returns dict with a_values, periods, biodiversity, total_pop.
    """
    a_values = np.linspace(1.05, 3.0, n_a)
    periods = []
    biodiversity = []
    total_pop = []
    is_cyclic = []

    for i, a_val in enumerate(a_values):
        x_init = [0.26, 0.25, 0.24, 0.25]
        config = _make_config(a=a_val, b=b, x_init=x_init, dt=dt, n_steps=n_steps)
        sim = MayLeonardSimulation(config)
        sim.reset()

        states_list = [sim.observe().copy()]
        for _ in range(n_steps):
            sim.step()
            states_list.append(sim.observe().copy())
        traj = np.array(states_list)

        # Measure period from total population peaks
        total = sim.compute_total_population(traj)
        skip = len(total) // 3
        total_tail = total[skip:]
        peaks = []
        for j in range(1, len(total_tail) - 1):
            if total_tail[j] > total_tail[j - 1] and total_tail[j] > total_tail[j + 1]:
                peaks.append(j)
        if len(peaks) >= 2:
            peak_diffs = np.diff(peaks) * dt
            periods.append(float(np.mean(peak_diffs)))
        else:
            periods.append(np.inf)

        # Check for cyclic dominance
        dominance = sim.compute_dominance_index(traj)
        changes = np.diff(dominance)
        n_transitions = int(np.sum(changes != 0))
        is_cyclic.append(n_transitions >= 4)

        biodiversity.append(sim.biodiversity_index())
        total_pop.append(float(np.sum(sim.observe())))

        if (i + 1) % 5 == 0:
            logger.info(
                f"  a={a_val:.2f}: period={periods[-1]:.2f}, "
                f"H={biodiversity[-1]:.3f}, cyclic={is_cyclic[-1]}"
            )

    return {
        "a_values": a_values,
        "periods": np.array(periods),
        "biodiversity": np.array(biodiversity),
        "total_population": np.array(total_pop),
        "is_cyclic": np.array(is_cyclic),
    }


def generate_fixed_point_data(
    n_samples: int = 30,
) -> dict[str, np.ndarray]:
    """Compute interior fixed point as function of a and b.

    For the symmetric case with all r=1, K=1, the interior fixed point
    should be x_i* = 1 / (1 + a + b) for the 4-species system where
    the row sum of alpha is (1 + a + b).

    Returns dict with parameter values and fixed points.
    """
    a_values = np.linspace(1.0, 3.0, n_samples)
    b_values = np.full(n_samples, 0.5)
    fixed_points = []
    row_sums = []

    for a_val, b_val in zip(a_values, b_values):
        config = _make_config(a=a_val, b=b_val)
        sim = MayLeonardSimulation(config)
        sim.reset()

        x_star = sim.compute_interior_fixed_point()
        fixed_points.append(x_star.copy())

        # Row sum of alpha matrix: 1 + a + b (for n=4 with 2 non-zero off-diag)
        row_sum = float(np.sum(sim.alpha[0]))
        row_sums.append(row_sum)

    return {
        "a_values": a_values,
        "b_values": b_values,
        "fixed_points": np.array(fixed_points),
        "row_sums": np.array(row_sums),
        "theoretical_x_star": 1.0 / np.array(row_sums),
    }


def run_may_leonard_rediscovery(
    output_dir: str | Path = "output/rediscovery/may_leonard",
    n_iterations: int = 40,
) -> dict:
    """Run the full May-Leonard rediscovery pipeline.

    1. Generate heteroclinic trajectory, verify cycling
    2. Sweep competition strength, track period and biodiversity
    3. Verify interior fixed point formula
    4. Check eigenvalue structure (unstable interior)

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "may_leonard",
        "targets": {
            "fixed_point": "x* = K / row_sum(alpha) for symmetric case",
            "heteroclinic": "Cyclic dominance with period T(a, b)",
            "biodiversity": "Shannon entropy oscillates during cycling",
        },
    }

    # --- Part 1: Heteroclinic trajectory ---
    logger.info("Part 1: Heteroclinic trajectory and cyclic dominance...")
    traj_data = generate_heteroclinic_trajectory(n_steps=50000, dt=0.01)

    dominance = traj_data["dominance"]
    total_pop = traj_data["total_population"]
    biodiv = traj_data["biodiversity"]
    x_star = traj_data["x_star"]

    # Detect cyclic dominance
    changes = np.diff(dominance)
    transition_indices = np.where(changes != 0)[0]
    n_transitions = len(transition_indices)

    # Check if transitions follow cyclic pattern
    if n_transitions >= 4:
        transition_species = dominance[transition_indices + 1]
        # Check for cyclic pattern: differences should be consistent
        diffs = np.diff(transition_species.astype(int)) % 4
        is_cyclic = bool(np.all(diffs == diffs[0]))
    else:
        is_cyclic = False

    results["heteroclinic"] = {
        "n_transitions": n_transitions,
        "is_cyclic": is_cyclic,
        "x_star": x_star.tolist(),
        "total_pop_mean": float(np.mean(total_pop)),
        "total_pop_std": float(np.std(total_pop)),
        "biodiversity_mean": float(np.mean(biodiv)),
        "biodiversity_std": float(np.std(biodiv)),
    }
    logger.info(f"  Transitions detected: {n_transitions}")
    logger.info(f"  Cyclic dominance: {is_cyclic}")
    logger.info(f"  Interior fixed point: {x_star}")

    # --- Part 2: Competition strength sweep ---
    logger.info("Part 2: Competition strength sweep (a: 1.05 -> 3.0)...")
    sweep_data = generate_competition_sweep_data(n_a=25, n_steps=40000, dt=0.01)

    finite_periods = sweep_data["periods"][np.isfinite(sweep_data["periods"])]
    results["competition_sweep"] = {
        "n_a_values": len(sweep_data["a_values"]),
        "a_range": [
            float(sweep_data["a_values"][0]),
            float(sweep_data["a_values"][-1]),
        ],
        "n_cyclic": int(np.sum(sweep_data["is_cyclic"])),
        "mean_period": float(np.mean(finite_periods)) if len(finite_periods) > 0 else None,
        "period_range": (
            [float(np.min(finite_periods)), float(np.max(finite_periods))]
            if len(finite_periods) > 0
            else None
        ),
    }
    logger.info(
        f"  Cyclic dynamics in {np.sum(sweep_data['is_cyclic'])}/{len(sweep_data['a_values'])} "
        f"parameter values"
    )
    if len(finite_periods) > 0:
        logger.info(
            f"  Period range: [{np.min(finite_periods):.2f}, {np.max(finite_periods):.2f}]"
        )

    # --- Part 3: Fixed point verification ---
    logger.info("Part 3: Interior fixed point verification...")
    fp_data = generate_fixed_point_data(n_samples=30)

    # For symmetric case, x_i* should equal K / row_sum = 1 / (1 + a + b)
    measured_x = fp_data["fixed_points"][:, 0]  # All species same by symmetry
    theoretical_x = fp_data["theoretical_x_star"]
    fp_error = np.abs(measured_x - theoretical_x) / np.abs(theoretical_x)

    results["fixed_point"] = {
        "mean_relative_error": float(np.mean(fp_error)),
        "max_relative_error": float(np.max(fp_error)),
        "formula_verified": bool(np.all(fp_error < 0.01)),
        "n_samples": len(fp_data["a_values"]),
    }
    logger.info(f"  Fixed point formula error: {np.mean(fp_error):.6e}")
    logger.info(f"  Formula x*=K/(1+a+b) verified: {np.all(fp_error < 0.01)}")

    # --- Part 4: Eigenvalue analysis ---
    logger.info("Part 4: Eigenvalue analysis at interior fixed point...")
    config = _make_config(a=1.5, b=0.5)
    sim = MayLeonardSimulation(config)
    sim.reset()
    eigs = sim.stability_eigenvalues()
    max_real = float(np.max(np.real(eigs)))

    results["eigenvalues"] = {
        "eigenvalues_real": [float(np.real(e)) for e in eigs],
        "eigenvalues_imag": [float(np.imag(e)) for e in eigs],
        "max_real_part": max_real,
        "interior_unstable": max_real > 0,
    }
    logger.info(f"  Eigenvalues: {eigs}")
    logger.info(f"  Max real part: {max_real:.4f}")
    logger.info(f"  Interior fixed point unstable: {max_real > 0}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save trajectory data
    np.savez(
        output_path / "heteroclinic_trajectory.npz",
        states=traj_data["states"],
        time=traj_data["time"],
        dominance=dominance,
        total_population=total_pop,
        biodiversity=biodiv,
    )
    np.savez(
        output_path / "competition_sweep.npz",
        **{k: v for k, v in sweep_data.items()},
    )

    return results
