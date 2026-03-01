"""Vicsek model rediscovery -- flocking / active matter.

Targets:
- Order-disorder phase transition: phi(eta) from 1 to 0
- Critical noise eta_c where phi crosses 0.5
- Density dependence: higher density -> easier to flock (lower eta_c)
- Connection to Kuramoto synchronization
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.vicsek import VicsekSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_noise_sweep_data(
    n_eta: int = 20,
    N: int = 100,
    L: float = 10.0,
    v0: float = 0.5,
    R: float = 1.0,
    n_steps: int = 500,
    n_avg: int = 200,
    n_trials: int = 3,
    seed_base: int = 42,
) -> dict[str, np.ndarray]:
    """Generate order parameter phi vs noise eta data.

    Sweeps eta from 0 to 1 and measures steady-state order parameter.
    """
    eta_values = np.linspace(0.0, 1.0, n_eta)
    phi_mean_all = np.zeros(n_eta)

    for trial in range(n_trials):
        config = SimulationConfig(
            domain=Domain.VICSEK,
            dt=1.0,
            n_steps=n_steps,
            parameters={
                "N": float(N),
                "L": L,
                "v0": v0,
                "R": R,
                "eta": 0.0,
            },
        )
        sim = VicsekSimulation(config)
        sweep = sim.order_parameter_sweep(
            eta_values=eta_values,
            n_steps=n_steps,
            n_avg=n_avg,
            seed=seed_base + trial,
        )
        phi_mean_all += sweep["phi_mean"]

    phi_mean_all /= n_trials

    return {
        "eta": eta_values,
        "phi_mean": phi_mean_all,
        "N": N,
        "L": L,
        "density": N / L ** 2,
    }


def generate_density_sweep_data(
    N_values: list[int] | None = None,
    L: float = 10.0,
    eta: float = 0.3,
    v0: float = 0.5,
    R: float = 1.0,
    n_steps: int = 500,
    n_avg: int = 200,
    n_trials: int = 3,
    seed_base: int = 100,
) -> dict[str, np.ndarray]:
    """Generate order parameter vs density by varying N at fixed L.

    Higher density should lead to stronger alignment (higher phi).
    """
    if N_values is None:
        N_values = [20, 50, 100, 200, 400]

    densities = []
    phi_values = []

    for N in N_values:
        phi_trials = []
        for trial in range(n_trials):
            config = SimulationConfig(
                domain=Domain.VICSEK,
                dt=1.0,
                n_steps=n_steps,
                parameters={
                    "N": float(N),
                    "L": L,
                    "v0": v0,
                    "R": R,
                    "eta": eta,
                },
            )
            sim = VicsekSimulation(config)
            sim.reset(seed=seed_base + trial)

            # Run transient
            for _ in range(n_steps - n_avg):
                sim.step()

            # Measure steady-state phi
            vals = []
            for _ in range(n_avg):
                sim.step()
                vals.append(sim.order_parameter())
            phi_trials.append(np.mean(vals))

        densities.append(N / L ** 2)
        phi_values.append(np.mean(phi_trials))
        logger.info(f"  N={N}, rho={N / L ** 2:.2f}: phi={np.mean(phi_trials):.4f}")

    return {
        "N": np.array(N_values),
        "density": np.array(densities),
        "phi": np.array(phi_values),
        "eta": eta,
        "L": L,
    }


def run_vicsek_rediscovery(
    output_dir: str | Path = "output/rediscovery/vicsek",
    n_iterations: int = 40,
) -> dict:
    """Run Vicsek model rediscovery pipeline.

    1. Sweep noise eta, measure order parameter phi
    2. Identify critical noise eta_c (phi crossing 0.5)
    3. Sweep density, measure phi dependence
    4. Optional PySR: fit phi(eta) curve

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iteration count.

    Returns:
        Results dict with discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "vicsek",
        "targets": {
            "order_parameter": "phi = |1/N sum exp(i*theta_i)|",
            "critical_noise": "eta_c where phi crosses 0.5",
            "density_dependence": "higher rho -> lower eta_c (easier flocking)",
        },
    }

    # --- Part 1: Noise sweep (order-disorder transition) ---
    logger.info("Part 1: Noise sweep (phi vs eta)...")
    data = generate_noise_sweep_data(
        n_eta=20, N=100, L=10.0, n_steps=500, n_avg=200, n_trials=3,
    )

    results["noise_sweep"] = {
        "n_eta": len(data["eta"]),
        "eta_range": [float(data["eta"].min()), float(data["eta"].max())],
        "phi_range": [float(data["phi_mean"].min()), float(data["phi_mean"].max())],
        "density": float(data["density"]),
    }

    # Estimate eta_c: interpolate where phi crosses 0.5
    phi = data["phi_mean"]
    eta = data["eta"]
    eta_c = None
    for j in range(len(phi) - 1):
        if phi[j] >= 0.5 and phi[j + 1] < 0.5:
            # Linear interpolation
            frac = (0.5 - phi[j]) / (phi[j + 1] - phi[j])
            eta_c = eta[j] + frac * (eta[j + 1] - eta[j])
            break

    if eta_c is not None:
        results["eta_c_estimate"] = float(eta_c)
        logger.info(f"  eta_c estimate: {eta_c:.4f}")
    else:
        logger.warning("  Could not detect eta_c (phi may not cross 0.5)")
        # Store approximate eta_c from midpoint of phi range
        mid_idx = np.argmin(np.abs(phi - 0.5))
        results["eta_c_estimate"] = float(eta[mid_idx])
        logger.info(f"  Approximate eta_c (closest to 0.5): {eta[mid_idx]:.4f}")

    # --- Part 2: PySR on phi(eta) ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use data where phi varies meaningfully
        mask = (phi > 0.05) & (phi < 0.95)
        if np.sum(mask) >= 5:
            X = eta[mask].reshape(-1, 1)
            y = phi[mask]

            logger.info("  Running PySR: phi = f(eta)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["eta"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square", "exp"],
                max_complexity=12,
                populations=20,
                population_size=40,
            )
            results["phi_pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["phi_pysr"]["best"] = best.expression
                results["phi_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["phi_pysr"] = {"error": str(e)}

    # --- Part 3: Density sweep ---
    logger.info("Part 2: Density sweep (phi vs rho)...")
    density_data = generate_density_sweep_data(
        N_values=[20, 50, 100, 200, 400],
        eta=0.3,
        n_steps=500,
        n_avg=200,
        n_trials=3,
    )
    results["density_sweep"] = {
        "N_values": density_data["N"].tolist(),
        "densities": density_data["density"].tolist(),
        "phi_values": density_data["phi"].tolist(),
        "eta": float(density_data["eta"]),
    }

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "noise_sweep.npz",
        eta=data["eta"],
        phi_mean=data["phi_mean"],
    )
    np.savez(
        output_path / "density_sweep.npz",
        N=density_data["N"],
        density=density_data["density"],
        phi=density_data["phi"],
    )

    return results
