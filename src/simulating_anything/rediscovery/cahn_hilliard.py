"""Cahn-Hilliard rediscovery: coarsening law and energy decay.

Targets:
- Mass conservation: integral(u) dx = const
- Free energy monotonically decreases
- Coarsening law: L(t) ~ t^(1/3) (Lifshitz-Slyozov)
- PySR: L(t) = f(t) should recover t^(1/3) scaling
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.cahn_hilliard import CahnHilliardSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_coarsening_data(
    n_snapshots: int = 30,
    n_steps: int = 50000,
    dt: float = 1e-4,
    N: int = 64,
    epsilon: float = 0.02,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate coarsening data: characteristic length vs time.

    Runs a single spinodal decomposition simulation and measures the
    characteristic domain size at multiple time points.

    Args:
        n_snapshots: Number of measurement points.
        n_steps: Total simulation steps.
        dt: Time step.
        N: Grid resolution.
        epsilon: Interface width parameter.
        seed: Random seed.

    Returns:
        Dict with 'times' and 'length_scales' arrays.
    """
    config = SimulationConfig(
        domain=Domain.CAHN_HILLIARD,
        dt=dt,
        n_steps=n_steps,
        parameters={"M": 1.0, "epsilon": epsilon, "N": float(N), "L": 1.0},
        seed=seed,
    )
    sim = CahnHilliardSimulation(config)
    sim.reset(seed=seed)

    # Let the system evolve past the initial transient before measuring
    transient_steps = n_steps // 10
    for _ in range(transient_steps):
        sim.step()

    remaining = n_steps - transient_steps
    snapshot_interval = max(1, remaining // n_snapshots)

    times = []
    length_scales = []

    for step_idx in range(remaining):
        sim.step()
        if (step_idx + 1) % snapshot_interval == 0:
            t = (transient_steps + step_idx + 1) * dt
            L_char = sim._characteristic_length()
            times.append(t)
            length_scales.append(L_char)
            if len(times) % 10 == 0:
                logger.info(f"  t={t:.4f}: L={L_char:.4f}")

    return {
        "times": np.array(times),
        "length_scales": np.array(length_scales),
    }


def generate_energy_data(
    n_steps: int = 10000,
    dt: float = 1e-4,
    N: int = 64,
    epsilon: float = 0.05,
    seed: int = 42,
    sample_interval: int = 100,
) -> dict[str, np.ndarray]:
    """Generate energy vs time data for a spinodal decomposition run.

    Args:
        n_steps: Total simulation steps.
        dt: Time step.
        N: Grid resolution.
        epsilon: Interface width parameter.
        seed: Random seed.
        sample_interval: Steps between energy measurements.

    Returns:
        Dict with 'times', 'energies', and 'masses' arrays.
    """
    config = SimulationConfig(
        domain=Domain.CAHN_HILLIARD,
        dt=dt,
        n_steps=n_steps,
        parameters={"M": 1.0, "epsilon": epsilon, "N": float(N), "L": 1.0},
        seed=seed,
    )
    sim = CahnHilliardSimulation(config)
    sim.reset(seed=seed)

    times = [0.0]
    energies = [sim.compute_free_energy()]
    masses = [sim.compute_total_mass()]

    for step_idx in range(n_steps):
        sim.step()
        if (step_idx + 1) % sample_interval == 0:
            t = (step_idx + 1) * dt
            times.append(t)
            energies.append(sim.compute_free_energy())
            masses.append(sim.compute_total_mass())

    return {
        "times": np.array(times),
        "energies": np.array(energies),
        "masses": np.array(masses),
    }


def run_cahn_hilliard_rediscovery(
    output_dir: str | Path = "output/rediscovery/cahn_hilliard",
    n_iterations: int = 40,
) -> dict:
    """Run the full Cahn-Hilliard rediscovery.

    1. Verify mass conservation
    2. Verify energy monotonic decrease
    3. Measure coarsening law L(t)
    4. PySR: find L(t) = f(t), target t^(1/3)

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "cahn_hilliard",
        "targets": {
            "mass_conservation": "integral(u) = const",
            "energy_decrease": "dE/dt <= 0",
            "coarsening_law": "L(t) ~ t^(1/3)",
        },
    }

    # --- Part 1: Energy and mass tracking ---
    logger.info("Part 1: Energy decay and mass conservation...")
    edata = generate_energy_data(
        n_steps=10000, dt=1e-4, N=64, epsilon=0.05, seed=42, sample_interval=100
    )

    # Mass conservation check
    mass_init = edata["masses"][0]
    mass_final = edata["masses"][-1]
    mass_drift = abs(mass_final - mass_init)
    mass_rel_drift = mass_drift / max(abs(mass_init), 1e-15) if mass_init != 0 else mass_drift

    results["mass_conservation"] = {
        "initial_mass": float(mass_init),
        "final_mass": float(mass_final),
        "absolute_drift": float(mass_drift),
        "relative_drift": float(mass_rel_drift),
    }
    logger.info(f"  Mass drift: {mass_drift:.2e} (relative: {mass_rel_drift:.2e})")

    # Energy decrease check
    energies = edata["energies"]
    energy_diffs = np.diff(energies)
    n_increasing = int(np.sum(energy_diffs > 1e-12))
    energy_monotonic = n_increasing == 0

    results["energy_decrease"] = {
        "initial_energy": float(energies[0]),
        "final_energy": float(energies[-1]),
        "energy_ratio": float(energies[-1] / energies[0]) if energies[0] > 0 else 0.0,
        "n_increasing_steps": n_increasing,
        "is_monotonic": energy_monotonic,
    }
    logger.info(
        f"  Energy: {energies[0]:.6f} -> {energies[-1]:.6f} "
        f"(monotonic: {energy_monotonic})"
    )

    # --- Part 2: Coarsening law ---
    logger.info("Part 2: Coarsening analysis...")
    cdata = generate_coarsening_data(
        n_snapshots=30, n_steps=50000, dt=1e-4, N=64, epsilon=0.02, seed=42
    )

    times = cdata["times"]
    lengths = cdata["length_scales"]

    # Filter out early transient (first few points) and any degenerate values
    valid = (times > 0) & (lengths > 0) & (lengths < 1.0) & np.isfinite(lengths)

    if np.sum(valid) >= 5:
        log_t = np.log(times[valid])
        log_L = np.log(lengths[valid])

        # Linear fit in log-log space: log(L) = a * log(t) + b
        coeffs = np.polyfit(log_t, log_L, 1)
        exponent = coeffs[0]
        prefactor = np.exp(coeffs[1])

        # Correlation
        L_pred = prefactor * times[valid] ** exponent
        ss_res = np.sum((lengths[valid] - L_pred) ** 2)
        ss_tot = np.sum((lengths[valid] - np.mean(lengths[valid])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        results["coarsening"] = {
            "n_valid_points": int(np.sum(valid)),
            "fitted_exponent": float(exponent),
            "theoretical_exponent": 1.0 / 3.0,
            "exponent_error": float(abs(exponent - 1.0 / 3.0)),
            "prefactor": float(prefactor),
            "r_squared": float(r2),
        }
        logger.info(
            f"  Coarsening exponent: {exponent:.4f} "
            f"(theory: 0.333, error: {abs(exponent - 1./3.):.4f})"
        )
        logger.info(f"  Power-law R2: {r2:.6f}")
    else:
        results["coarsening"] = {"error": "Too few valid data points"}
        logger.warning("  Too few valid coarsening data points")

    # --- Part 3: PySR symbolic regression ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        if np.sum(valid) >= 5:
            X = times[valid].reshape(-1, 1)
            y = lengths[valid]

            logger.info("Running PySR: L(t) = f(t)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["t"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "cube"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["coarsening_pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {
                        "expression": d.expression,
                        "r_squared": d.evidence.fit_r_squared,
                    }
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["coarsening_pysr"]["best"] = best.expression
                results["coarsening_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["coarsening_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
