"""Kuramoto-Sivashinsky equation rediscovery.

Targets:
- Energy evolution: L2 norm saturates to finite value (spatiotemporal chaos)
- Lyapunov exponent: positive, indicating chaos (estimated from trajectory divergence)
- Correlation length: scales with domain size L
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.kuramoto_sivashinsky import KuramotoSivashinsky
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_energy_evolution_data(
    L: float = 32.0 * np.pi,
    N: int = 128,
    n_steps: int = 5000,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Track energy (L2 norm) vs time for the KS equation.

    The energy should grow from the small initial perturbation, then
    saturate to a statistically steady chaotic state.
    """
    config = SimulationConfig(
        domain=Domain.KURAMOTO_SIVASHINSKY,
        dt=dt,
        n_steps=n_steps,
        parameters={"L": L, "N": float(N), "viscosity": 1.0},
    )
    sim = KuramotoSivashinsky(config)
    sim.reset()

    times = [0.0]
    energies = [sim.energy]
    means = [sim.spatial_mean]

    for i in range(n_steps):
        sim.step()
        times.append((i + 1) * dt)
        energies.append(sim.energy)
        means.append(sim.spatial_mean)

    return {
        "time": np.array(times),
        "energy": np.array(energies),
        "spatial_mean": np.array(means),
        "L": L,
        "N": N,
    }


def generate_lyapunov_data(
    L: float = 32.0 * np.pi,
    N: int = 128,
    n_steps: int = 2000,
    dt: float = 0.05,
    n_trials: int = 5,
    perturbation: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Estimate the largest Lyapunov exponent from trajectory divergence.

    Runs a reference trajectory and perturbed trajectories, measuring the
    exponential growth rate of their separation.
    """
    lyapunov_estimates = []

    for trial in range(n_trials):
        seed = 42 + trial
        config = SimulationConfig(
            domain=Domain.KURAMOTO_SIVASHINSKY,
            dt=dt,
            n_steps=n_steps,
            parameters={"L": L, "N": float(N), "viscosity": 1.0},
            seed=seed,
        )

        # Reference trajectory
        sim_ref = KuramotoSivashinsky(config)
        sim_ref.reset(seed=seed)

        # Let transient decay
        n_transient = 500
        for _ in range(n_transient):
            sim_ref.step()

        # Perturbed trajectory
        sim_pert = KuramotoSivashinsky(config)
        sim_pert.reset(seed=seed)
        for _ in range(n_transient):
            sim_pert.step()

        # Add small perturbation and sync spectral state
        rng = np.random.default_rng(seed + 1000)
        delta = rng.normal(0, perturbation, size=N)
        sim_pert._state = sim_ref._state.copy() + delta
        sim_pert._u_hat = np.fft.fft(sim_pert._state)

        # Measure divergence
        log_stretching = 0.0
        n_renorm = 0
        n_measure = n_steps - n_transient

        for _ in range(n_measure):
            sim_ref.step()
            sim_pert.step()

            diff = sim_pert._state - sim_ref._state
            dist = np.sqrt(np.mean(diff ** 2))

            if dist > 0 and np.isfinite(dist):
                log_stretching += np.log(dist / perturbation)
                n_renorm += 1

                # Renormalize perturbation and sync spectral state
                sim_pert._state = sim_ref._state + diff * (perturbation / dist)
                sim_pert._u_hat = np.fft.fft(sim_pert._state)

        if n_renorm > 0:
            lam = log_stretching / (n_renorm * dt)
            lyapunov_estimates.append(lam)
            logger.info(f"  Trial {trial}: Lyapunov = {lam:.4f}")

    return {
        "lyapunov_estimates": np.array(lyapunov_estimates),
        "lyapunov_mean": float(np.mean(lyapunov_estimates)) if lyapunov_estimates else 0.0,
        "lyapunov_std": float(np.std(lyapunov_estimates)) if lyapunov_estimates else 0.0,
        "L": L,
        "N": N,
    }


def generate_spatial_correlation_data(
    L_values: np.ndarray | None = None,
    N: int = 256,
    n_steps: int = 3000,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Measure correlation length vs domain size L.

    For the KS equation, the correlation length is expected to be
    approximately independent of L for large L (extensive chaos).
    """
    if L_values is None:
        L_values = np.array([
            10 * np.pi, 20 * np.pi, 32 * np.pi, 50 * np.pi, 64 * np.pi,
        ])

    all_L = []
    all_corr_len = []

    for L in L_values:
        config = SimulationConfig(
            domain=Domain.KURAMOTO_SIVASHINSKY,
            dt=dt,
            n_steps=n_steps,
            parameters={"L": L, "N": float(N), "viscosity": 1.0},
        )
        sim = KuramotoSivashinsky(config)
        sim.reset()

        # Run to steady state
        for _ in range(2000):
            sim.step()

        # Average correlation length over several snapshots
        corr_lengths = []
        for _ in range(n_steps - 2000):
            sim.step()
            if sim._step_count % 50 == 0:
                cl = sim.correlation_length()
                if np.isfinite(cl) and cl > 0:
                    corr_lengths.append(cl)

        if corr_lengths:
            mean_cl = np.mean(corr_lengths)
            all_L.append(L)
            all_corr_len.append(mean_cl)
            logger.info(f"  L={L:.1f}: correlation_length={mean_cl:.2f}")

    return {
        "L": np.array(all_L),
        "correlation_length": np.array(all_corr_len),
    }


def run_kuramoto_sivashinsky_rediscovery(
    output_dir: str | Path = "output/rediscovery/kuramoto_sivashinsky",
    n_iterations: int = 40,
) -> dict:
    """Run the full KS equation rediscovery.

    1. Track energy evolution (saturation to chaos)
    2. Estimate Lyapunov exponent (positive for chaos)
    3. Measure correlation length vs domain size

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "kuramoto_sivashinsky",
        "targets": {
            "equation": "u_t + u*u_x + u_xx + u_xxxx = 0",
            "energy_saturation": "L2 norm saturates (no blowup)",
            "lyapunov": "Positive Lyapunov exponent (spatiotemporal chaos)",
            "correlation_length": "Finite correlation length (extensive chaos)",
        },
    }

    # --- Part 1: Energy evolution ---
    logger.info("Part 1: Tracking energy evolution...")
    energy_data = generate_energy_evolution_data(n_steps=3000, dt=0.05)

    energy = energy_data["energy"]
    means = energy_data["spatial_mean"]

    # Check energy saturation (compare first and second half)
    n = len(energy)
    first_half_energy = np.mean(energy[n // 4: n // 2])
    second_half_energy = np.mean(energy[n // 2:])

    # Check spatial mean conservation
    mean_drift = np.abs(means[-1] - means[0])

    results["energy_evolution"] = {
        "n_steps": len(energy) - 1,
        "initial_energy": float(energy[0]),
        "final_energy": float(energy[-1]),
        "max_energy": float(np.max(energy)),
        "first_half_mean": float(first_half_energy),
        "second_half_mean": float(second_half_energy),
        "energy_bounded": bool(np.all(np.isfinite(energy))),
        "spatial_mean_drift": float(mean_drift),
        "spatial_mean_conserved": bool(mean_drift < 1e-8),
    }
    logger.info(f"  Energy: initial={energy[0]:.4e}, final={energy[-1]:.4e}")
    logger.info(f"  Spatial mean drift: {mean_drift:.4e}")

    # --- Part 2: Lyapunov exponent ---
    logger.info("Part 2: Estimating Lyapunov exponent...")
    lyap_data = generate_lyapunov_data(n_steps=2000, dt=0.05, n_trials=3)

    results["lyapunov"] = {
        "mean": lyap_data["lyapunov_mean"],
        "std": lyap_data["lyapunov_std"],
        "n_trials": len(lyap_data["lyapunov_estimates"]),
        "is_chaotic": bool(lyap_data["lyapunov_mean"] > 0),
        "estimates": lyap_data["lyapunov_estimates"].tolist(),
    }
    logger.info(
        f"  Lyapunov: {lyap_data['lyapunov_mean']:.4f} "
        f"+/- {lyap_data['lyapunov_std']:.4f}"
    )

    # --- Part 3: Correlation length ---
    logger.info("Part 3: Measuring correlation length vs L...")
    corr_data = generate_spatial_correlation_data(
        L_values=np.array([10 * np.pi, 20 * np.pi, 32 * np.pi]),
        N=128,
        n_steps=2500,
        dt=0.05,
    )

    if len(corr_data["L"]) > 1:
        results["correlation_length"] = {
            "L_values": corr_data["L"].tolist(),
            "correlation_lengths": corr_data["correlation_length"].tolist(),
            "n_points": len(corr_data["L"]),
        }
        logger.info(f"  Correlation lengths: {corr_data['correlation_length']}")

    # --- Part 4: Stability threshold check ---
    logger.info("Part 4: Checking stability threshold (small L)...")
    L_small = 2 * np.pi  # Below critical L = 2*pi*sqrt(2) ~ 8.89
    config_small = SimulationConfig(
        domain=Domain.KURAMOTO_SIVASHINSKY,
        dt=0.01,
        n_steps=2000,
        parameters={"L": L_small, "N": 64.0, "viscosity": 1.0},
    )
    sim_small = KuramotoSivashinsky(config_small)
    sim_small.init_type = "sine"
    sim_small.reset()
    for _ in range(2000):
        sim_small.step()
    final_energy_small = sim_small.energy

    results["stability_threshold"] = {
        "L_small": L_small,
        "L_critical": 2 * np.pi * np.sqrt(2),
        "final_energy": float(final_energy_small),
        "decayed_to_zero": bool(final_energy_small < 1e-10),
    }
    logger.info(
        f"  L={L_small:.2f} (< L_c={2*np.pi*np.sqrt(2):.2f}): "
        f"energy={final_energy_small:.4e}, decayed={final_energy_small < 1e-10}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save energy data
    np.savez(
        output_path / "energy_evolution.npz",
        time=energy_data["time"],
        energy=energy_data["energy"],
        spatial_mean=energy_data["spatial_mean"],
    )

    return results
