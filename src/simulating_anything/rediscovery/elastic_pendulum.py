"""Elastic pendulum rediscovery.

Targets:
- Radial frequency: omega_r = sqrt(k/m)
- Angular frequency (small angle): omega_theta = sqrt(g/L0)
- Energy conservation: E = const (no dissipation)
- SINDy ODE recovery of the elastic pendulum equations
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.elastic_pendulum import ElasticPendulum
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Domain enum value used for config construction
_DOMAIN = Domain.ELASTIC_PENDULUM


def generate_frequency_data(
    n_samples: int = 200,
    n_steps: int = 20000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate data to study radial frequency omega_r = sqrt(k/m).

    Sweep k and m, start with a small radial perturbation from equilibrium
    at theta=0 (pure radial mode), and measure the oscillation frequency
    from zero crossings of (r - r_eq).
    """
    rng = np.random.default_rng(42)

    all_k = []
    all_m = []
    all_L0 = []
    all_omega_measured = []
    all_omega_theory = []

    for i in range(n_samples):
        k_val = rng.uniform(1.0, 20.0)
        m_val = rng.uniform(0.5, 5.0)
        L0_val = rng.uniform(0.5, 3.0)

        # Equilibrium length
        r_eq = L0_val + m_val * 9.81 / k_val
        # Small radial perturbation, no angular motion
        r_0 = r_eq + 0.01

        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "k": k_val, "m": m_val, "L0": L0_val, "g": 9.81,
                "r_0": r_0, "r_dot_0": 0.0,
                "theta_0": 0.0, "theta_dot_0": 0.0,
            },
        )
        sim = ElasticPendulum(config)
        sim.reset()

        # Collect radial displacement from equilibrium
        r_displacements = [sim.observe()[0] - r_eq]
        for _ in range(n_steps):
            state = sim.step()
            r_displacements.append(state[0] - r_eq)

        r_displacements = np.array(r_displacements)

        # Find period from positive-going zero crossings
        crossings = []
        for j in range(1, len(r_displacements)):
            if r_displacements[j - 1] < 0 and r_displacements[j] >= 0:
                frac = -r_displacements[j - 1] / (
                    r_displacements[j] - r_displacements[j - 1]
                )
                crossings.append((j - 1 + frac) * dt)

        if len(crossings) >= 3:
            periods = np.diff(crossings)
            T_measured = float(np.median(periods))
            omega_measured = 2 * np.pi / T_measured
            omega_theory = np.sqrt(k_val / m_val)

            all_k.append(k_val)
            all_m.append(m_val)
            all_L0.append(L0_val)
            all_omega_measured.append(omega_measured)
            all_omega_theory.append(omega_theory)

        if (i + 1) % 50 == 0:
            logger.info(f"  Radial frequency measurement {i + 1}/{n_samples}")

    return {
        "k": np.array(all_k),
        "m": np.array(all_m),
        "L0": np.array(all_L0),
        "omega_measured": np.array(all_omega_measured),
        "omega_theory": np.array(all_omega_theory),
    }


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
        k_val = rng.uniform(1.0, 20.0)
        m_val = rng.uniform(0.5, 5.0)
        L0_val = rng.uniform(0.5, 3.0)
        r_eq = L0_val + m_val * 9.81 / k_val
        r_0 = r_eq + rng.uniform(-0.1, 0.1)
        r_dot_0 = rng.uniform(-0.5, 0.5)
        theta_0 = rng.uniform(-0.3, 0.3)
        theta_dot_0 = rng.uniform(-0.5, 0.5)

        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "k": k_val, "m": m_val, "L0": L0_val, "g": 9.81,
                "r_0": r_0, "r_dot_0": r_dot_0,
                "theta_0": theta_0, "theta_dot_0": theta_dot_0,
            },
        )
        sim = ElasticPendulum(config)
        sim.reset()

        E0 = sim.total_energy
        energies = [E0]
        for _ in range(n_steps):
            sim.step()
            energies.append(sim.total_energy)

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


def run_elastic_pendulum_rediscovery(
    output_dir: str | Path = "output/rediscovery/elastic_pendulum",
    n_iterations: int = 40,
) -> dict:
    """Run the full elastic pendulum rediscovery.

    1. Energy conservation verification
    2. Radial frequency: omega_r = sqrt(k/m) via PySR
    3. Report results
    """
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "elastic_pendulum",
        "targets": {
            "energy_conservation": "E(t) = E(0) for all t",
            "radial_frequency": "omega_r = sqrt(k/m)",
            "angular_frequency": "omega_theta = sqrt(g/L0) (small angle)",
        },
    }

    # --- Part 1: Energy conservation ---
    logger.info("Part 1: Energy conservation verification...")
    energy_data = generate_energy_conservation_data(
        n_trajectories=50, n_steps=10000, dt=0.001
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

    # --- Part 2: Radial frequency rediscovery ---
    logger.info("Part 2: Radial frequency measurement omega_r = sqrt(k/m)...")
    freq_data = generate_frequency_data(n_samples=200, n_steps=20000, dt=0.001)

    rel_error = (
        np.abs(freq_data["omega_measured"] - freq_data["omega_theory"])
        / freq_data["omega_theory"]
    )
    results["frequency_accuracy"] = {
        "n_samples": len(freq_data["k"]),
        "mean_relative_error": float(np.mean(rel_error)),
        "max_relative_error": float(np.max(rel_error)),
    }
    logger.info(f"  Frequency accuracy: mean error = {np.mean(rel_error):.4%}")

    # PySR: omega_r = f(k, m)
    logger.info(
        f"  Running PySR for omega_r = f(k, m) with {n_iterations} iterations..."
    )
    X = np.column_stack([freq_data["k"], freq_data["m"]])
    y = freq_data["omega_measured"]

    discoveries = run_symbolic_regression(
        X,
        y,
        variable_names=["k_", "m_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "square"],
        max_complexity=15,
        populations=20,
        population_size=40,
    )

    results["frequency_pysr"] = {
        "n_discoveries": len(discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in discoveries[:5]
        ],
    }
    if discoveries:
        best = discoveries[0]
        results["frequency_pysr"]["best"] = best.expression
        results["frequency_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})"
        )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "frequency_data.npz",
        k=freq_data["k"],
        m=freq_data["m"],
        L0=freq_data["L0"],
        omega_measured=freq_data["omega_measured"],
        omega_theory=freq_data["omega_theory"],
    )

    return results
