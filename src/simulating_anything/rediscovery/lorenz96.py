"""Lorenz-96 model rediscovery.

Targets:
- Chaos transition: Lyapunov exponent as a function of F
- Energy statistics: mean energy vs F
- Attractor dimension estimate vs N and F
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.lorenz96 import Lorenz96
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_chaos_transition_data(
    n_F: int = 30,
    N: int = 36,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep F to map the transition to chaos.

    For each F value, compute the Lyapunov exponent and classify
    the dynamics (decaying, periodic, chaotic).
    """
    F_values = np.linspace(1.0, 20.0, n_F)
    lyapunov_exps = []
    attractor_types = []

    for i, F in enumerate(F_values):
        config = SimulationConfig(
            domain=Domain.LORENZ96,
            dt=dt,
            n_steps=n_steps,
            parameters={"N": float(N), "F": F},
        )
        sim = Lorenz96(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        # Classify dynamics
        if lam < -0.1:
            atype = "decaying"
        elif lam < 0.1:
            atype = "periodic_or_marginal"
        else:
            atype = "chaotic"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(f"  F={F:.1f}: Lyapunov={lam:.4f}, type={atype}")

    return {
        "F": F_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "attractor_type": np.array(attractor_types),
    }


def generate_energy_data(
    n_F: int = 25,
    N: int = 36,
    n_transient: int = 5000,
    n_measure: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep F to measure energy statistics.

    For each F, compute mean, std, min, and max energy after discarding transient.
    """
    F_values = np.linspace(2.0, 20.0, n_F)
    energy_means = []
    energy_stds = []

    for i, F in enumerate(F_values):
        config = SimulationConfig(
            domain=Domain.LORENZ96,
            dt=dt,
            n_steps=n_transient + n_measure,
            parameters={"N": float(N), "F": F},
        )
        sim = Lorenz96(config)
        sim.reset()

        # Skip transient
        for _ in range(n_transient):
            sim.step()

        # Measure energy
        energies = []
        for _ in range(n_measure):
            sim.step()
            energies.append(sim.energy)

        energy_means.append(np.mean(energies))
        energy_stds.append(np.std(energies))

        if (i + 1) % 10 == 0:
            logger.info(
                f"  F={F:.1f}: energy={energy_means[-1]:.2f} +/- {energy_stds[-1]:.2f}"
            )

    return {
        "F": F_values,
        "energy_mean": np.array(energy_means),
        "energy_std": np.array(energy_stds),
    }


def generate_dimension_data(
    N_values: list[int] | None = None,
    F: float = 8.0,
    n_transient: int = 5000,
    n_measure: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Measure attractor dimension proxy vs system size N.

    Uses the correlation dimension estimate via the fraction of active modes
    (principal components above a threshold). This is a rough proxy for the
    true attractor dimension.
    """
    if N_values is None:
        N_values = [8, 12, 20, 36, 48]

    dimension_estimates = []
    active_modes_list = []

    for i, N in enumerate(N_values):
        config = SimulationConfig(
            domain=Domain.LORENZ96,
            dt=dt,
            n_steps=n_transient + n_measure,
            parameters={"N": float(N), "F": F},
        )
        sim = Lorenz96(config)
        sim.reset()

        # Skip transient
        for _ in range(n_transient):
            sim.step()

        # Collect states for PCA
        states = []
        for _ in range(n_measure):
            sim.step()
            states.append(sim.observe())

        states_arr = np.array(states)
        # Center data
        states_centered = states_arr - np.mean(states_arr, axis=0)

        # Compute SVD for dimension estimate
        _, s, _ = np.linalg.svd(states_centered, full_matrices=False)
        # Fraction of variance explained
        var_explained = s**2 / np.sum(s**2)
        cumvar = np.cumsum(var_explained)
        # Number of modes for 95% variance
        n_active = int(np.searchsorted(cumvar, 0.95) + 1)

        dimension_estimates.append(n_active)
        active_modes_list.append(n_active)

        logger.info(
            f"  N={N}: active modes (95% variance) = {n_active}/{N}"
        )

    return {
        "N": np.array(N_values),
        "dimension_estimate": np.array(dimension_estimates),
        "F": F,
    }


def run_lorenz96_rediscovery(
    output_dir: str | Path = "output/rediscovery/lorenz96",
    n_iterations: int = 40,
) -> dict:
    """Run the full Lorenz-96 rediscovery.

    1. Sweep F to map chaos transition (Lyapunov exponent)
    2. Measure energy statistics vs F
    3. Estimate attractor dimension vs N
    4. Verify fixed point and decay for F=0

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "lorenz96",
        "targets": {
            "equations": "dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F",
            "fixed_point": "x_i = F for all i",
            "chaos_onset": "F >= ~8 for N=36",
            "energy_scaling": "mean energy grows with F",
        },
    }

    # --- Part 1: Chaos transition sweep ---
    logger.info("Part 1: Mapping chaos transition (F sweep)...")
    chaos_data = generate_chaos_transition_data(n_F=30, N=36, n_steps=20000, dt=0.01)

    n_chaotic = int(np.sum(chaos_data["attractor_type"] == "chaotic"))
    n_decaying = int(np.sum(chaos_data["attractor_type"] == "decaying"))
    logger.info(f"  Found {n_chaotic} chaotic, {n_decaying} decaying regimes")

    results["chaos_transition"] = {
        "n_F_values": len(chaos_data["F"]),
        "n_chaotic": n_chaotic,
        "n_decaying": n_decaying,
        "F_range": [float(chaos_data["F"][0]), float(chaos_data["F"][-1])],
    }

    # Find approximate critical F (first positive Lyapunov)
    mask_positive = chaos_data["lyapunov_exponent"] > 0.1
    if np.any(mask_positive):
        F_c_approx = float(chaos_data["F"][np.argmax(mask_positive)])
        results["chaos_transition"]["F_c_approx"] = F_c_approx
        logger.info(f"  Approximate critical F: {F_c_approx:.1f} (expected: ~8)")

    # Lyapunov at F=8
    mask_F8 = np.argmin(np.abs(chaos_data["F"] - 8.0))
    lam_F8 = float(chaos_data["lyapunov_exponent"][mask_F8])
    results["chaos_transition"]["lyapunov_at_F8"] = lam_F8
    logger.info(f"  Lyapunov at F~8: {lam_F8:.4f}")

    # --- Part 2: Energy statistics ---
    logger.info("Part 2: Energy statistics vs F...")
    energy_data = generate_energy_data(n_F=25, N=36, dt=0.01)

    results["energy_statistics"] = {
        "n_F_values": len(energy_data["F"]),
        "F_range": [float(energy_data["F"][0]), float(energy_data["F"][-1])],
        "energy_mean_range": [
            float(np.min(energy_data["energy_mean"])),
            float(np.max(energy_data["energy_mean"])),
        ],
    }

    # Check that energy grows with F (monotonic in the mean)
    energy_correlation = float(np.corrcoef(energy_data["F"], energy_data["energy_mean"])[0, 1])
    results["energy_statistics"]["energy_F_correlation"] = energy_correlation
    logger.info(f"  Energy-F correlation: {energy_correlation:.4f}")

    # --- Part 3: Attractor dimension vs N ---
    logger.info("Part 3: Attractor dimension estimate vs N...")
    dim_data = generate_dimension_data(N_values=[8, 12, 20, 36, 48], F=8.0, dt=0.01)

    results["dimension_scaling"] = {
        "N_values": dim_data["N"].tolist(),
        "dimension_estimates": dim_data["dimension_estimate"].tolist(),
        "F": dim_data["F"],
    }

    # Check that dimension grows with N
    dim_correlation = float(
        np.corrcoef(dim_data["N"].astype(float), dim_data["dimension_estimate"].astype(float))[
            0, 1
        ]
    )
    results["dimension_scaling"]["dimension_N_correlation"] = dim_correlation
    logger.info(f"  Dimension-N correlation: {dim_correlation:.4f}")

    # --- Part 4: Fixed point verification ---
    logger.info("Part 4: Verifying fixed point x_i = F...")
    config_fp = SimulationConfig(
        domain=Domain.LORENZ96,
        dt=0.01,
        n_steps=1000,
        parameters={"N": 36.0, "F": 8.0},
    )
    sim_fp = Lorenz96(config_fp)
    sim_fp.reset()

    fp = sim_fp.fixed_point
    derivs = sim_fp._derivatives(fp)
    fp_residual = float(np.max(np.abs(derivs)))
    results["fixed_point"] = {
        "value": float(fp[0]),
        "max_derivative_residual": fp_residual,
        "is_exact": fp_residual < 1e-12,
    }
    logger.info(f"  Fixed point x_i = {fp[0]:.1f}, max |dx/dt| = {fp_residual:.2e}")

    # --- Part 5: Decay for F=0 ---
    logger.info("Part 5: Verifying decay for F=0...")
    config_decay = SimulationConfig(
        domain=Domain.LORENZ96,
        dt=0.01,
        n_steps=2000,
        parameters={"N": 36.0, "F": 0.0},
        seed=42,
    )
    sim_decay = Lorenz96(config_decay)
    # Initialize with random state
    sim_decay.reset(seed=42)
    # Override with non-trivial initial condition for decay test
    sim_decay._state = np.random.default_rng(42).standard_normal(36) * 0.5

    energies_decay = []
    for _ in range(2000):
        sim_decay.step()
        energies_decay.append(sim_decay.energy)

    final_energy = energies_decay[-1]
    results["decay_F0"] = {
        "initial_energy": float(energies_decay[0]),
        "final_energy": float(final_energy),
        "decayed": bool(final_energy < 0.01),
    }
    logger.info(
        f"  F=0 decay: energy {energies_decay[0]:.4f} -> {final_energy:.6f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "chaos_transition.npz",
        F=chaos_data["F"],
        lyapunov_exponent=chaos_data["lyapunov_exponent"],
    )
    np.savez(
        output_path / "energy_data.npz",
        F=energy_data["F"],
        energy_mean=energy_data["energy_mean"],
        energy_std=energy_data["energy_std"],
    )
    np.savez(
        output_path / "dimension_data.npz",
        N=dim_data["N"],
        dimension_estimate=dim_data["dimension_estimate"],
    )

    return results
