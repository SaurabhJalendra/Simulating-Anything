"""Swinging Atwood machine rediscovery.

Targets:
- Energy conservation: E = const (within numerical tolerance)
- Mass ratio sweep: Lyapunov exponent as function of mu = M/m
- Identify regular vs chaotic regimes
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.swinging_atwood import SwingingAtwoodSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Domain enum value used for config construction
_DOMAIN = Domain.SWINGING_ATWOOD


def generate_energy_conservation_data(
    n_trajectories: int = 50,
    n_steps: int = 10000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Track total energy over time to verify conservation.

    Energy should be conserved to within numerical tolerance for the
    Hamiltonian system integrated with RK4.
    """
    rng = np.random.default_rng(42)

    all_final_drift = []
    all_max_drift = []
    all_mean_energy = []

    for i in range(n_trajectories):
        M_val = rng.uniform(1.0, 5.0)
        m_val = rng.uniform(0.5, 3.0)
        r_0 = rng.uniform(0.8, 2.0)
        theta_0 = rng.uniform(-0.5, 0.5)
        r_dot_0 = rng.uniform(-0.2, 0.2)
        theta_dot_0 = rng.uniform(-0.3, 0.3)

        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "M": M_val, "m": m_val, "g": 9.81,
                "r_min": 0.1, "r_0": r_0, "theta_0": theta_0,
                "r_dot_0": r_dot_0, "theta_dot_0": theta_dot_0,
            },
        )
        sim = SwingingAtwoodSimulation(config)
        sim.reset()

        E0 = sim.total_energy()
        energies = [E0]
        for _ in range(n_steps):
            sim.step()
            energies.append(sim.total_energy())

        energies = np.array(energies)
        drift = np.abs(energies - E0) / max(abs(E0), 1e-10)

        all_final_drift.append(drift[-1])
        all_max_drift.append(np.max(drift))
        all_mean_energy.append(np.mean(energies))

        if (i + 1) % 10 == 0:
            logger.info(
                f"  Energy check {i + 1}/{n_trajectories}: "
                f"max drift = {np.max(drift):.2e}"
            )

    return {
        "n_trajectories": n_trajectories,
        "final_drift": np.array(all_final_drift),
        "max_drift": np.array(all_max_drift),
        "mean_energy": np.array(all_mean_energy),
    }


def generate_mass_ratio_sweep_data(
    n_mu: int = 20,
    n_steps: int = 50000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Sweep mass ratio mu = M/m and measure Lyapunov exponent.

    For mu = 1, the system is integrable.
    For mu != 1, the system is generally chaotic.
    """
    mu_values = np.linspace(0.5, 5.0, n_mu)
    lyapunov_exps = []

    for i, mu in enumerate(mu_values):
        M_val = mu * 1.0  # m = 1.0
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "M": M_val, "m": 1.0, "g": 9.81,
                "r_min": 0.1, "r_0": 1.0, "theta_0": 0.5,
                "r_dot_0": 0.0, "theta_dot_0": 0.0,
            },
        )
        sim = SwingingAtwoodSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(2000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  mu={mu:.2f}: Lyapunov={lam:.4f}"
            )

    return {
        "mu": mu_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_swinging_atwood_rediscovery(
    output_dir: str | Path = "output/rediscovery/swinging_atwood",
    n_iterations: int = 40,
) -> dict:
    """Run the full swinging Atwood machine rediscovery.

    1. Energy conservation verification
    2. Mass ratio sweep for Lyapunov exponent
    3. Report results

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "swinging_atwood",
        "targets": {
            "energy_conservation": "E(t) = E(0) for all t",
            "chaos_vs_mu": "Lyapunov exponent as function of mu = M/m",
            "integrable_case": "mu = 1 is integrable (near-zero Lyapunov)",
        },
    }

    # --- Part 1: Energy conservation ---
    logger.info("Part 1: Energy conservation verification...")
    energy_data = generate_energy_conservation_data(
        n_trajectories=50, n_steps=10000, dt=0.001,
    )

    results["energy_conservation"] = {
        "n_trajectories": energy_data["n_trajectories"],
        "mean_final_drift": float(np.mean(energy_data["final_drift"])),
        "max_final_drift": float(np.max(energy_data["final_drift"])),
        "mean_max_drift": float(np.mean(energy_data["max_drift"])),
    }
    logger.info(
        f"  Energy drift: mean={np.mean(energy_data['final_drift']):.2e}, "
        f"max={np.max(energy_data['final_drift']):.2e}"
    )

    # --- Part 2: Mass ratio sweep ---
    logger.info("Part 2: Mass ratio sweep for Lyapunov exponent...")
    mu_data = generate_mass_ratio_sweep_data(
        n_mu=20, n_steps=50000, dt=0.001,
    )

    # Classify regimes
    lam = mu_data["lyapunov_exponent"]
    mu_vals = mu_data["mu"]
    n_chaotic = int(np.sum(lam > 0.1))
    n_regular = int(np.sum(lam < 0.01))

    results["mass_ratio_sweep"] = {
        "n_mu_values": len(mu_vals),
        "mu_range": [float(mu_vals[0]), float(mu_vals[-1])],
        "n_chaotic": n_chaotic,
        "n_regular": n_regular,
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
        "lyapunov_at_mu1": None,
    }

    # Find closest mu to 1.0 and record its Lyapunov
    idx_mu1 = int(np.argmin(np.abs(mu_vals - 1.0)))
    lam_at_mu1 = float(lam[idx_mu1])
    results["mass_ratio_sweep"]["lyapunov_at_mu1"] = lam_at_mu1
    logger.info(
        f"  Lyapunov at mu~1: {lam_at_mu1:.4f} "
        f"(should be near zero for integrable case)"
    )
    logger.info(
        f"  Chaotic regimes: {n_chaotic}/{len(mu_vals)}, "
        f"max Lyapunov = {np.max(lam):.4f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "energy_data.npz",
        final_drift=energy_data["final_drift"],
        max_drift=energy_data["max_drift"],
    )
    np.savez(
        output_path / "mass_ratio_sweep.npz",
        mu=mu_data["mu"],
        lyapunov_exponent=mu_data["lyapunov_exponent"],
    )

    return results
