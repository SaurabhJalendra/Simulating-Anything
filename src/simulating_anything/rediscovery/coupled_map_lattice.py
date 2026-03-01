"""Coupled Map Lattice rediscovery.

Targets:
- Synchronization transition: spatial variance drops with increasing eps
- Order parameter vs coupling strength curve
- Largest Lyapunov exponent as function of eps
- Critical coupling for synchronization onset
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.coupled_map_lattice import (
    CoupledMapLatticeSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_coupling_sweep_data(
    n_eps: int = 30,
    eps_min: float = 0.0,
    eps_max: float = 0.7,
    N: int = 100,
    r: float = 3.9,
    n_transient: int = 2000,
    n_measure: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Sweep coupling strength and measure order parameter.

    Returns dict with keys: eps, variance_mean, variance_std,
    mean_field_mean, sync_order.
    """
    eps_values = np.linspace(eps_min, eps_max, n_eps)
    config = SimulationConfig(
        domain=Domain.COUPLED_MAP_LATTICE,
        dt=1.0,
        n_steps=1000,
        parameters={"N": float(N), "r": r, "eps": 0.0},
    )
    sim = CoupledMapLatticeSimulation(config)
    data = sim.coupling_sweep(
        eps_values,
        n_transient=n_transient,
        n_measure=n_measure,
        seed=seed,
    )
    return data


def generate_lyapunov_sweep_data(
    n_eps: int = 20,
    eps_min: float = 0.0,
    eps_max: float = 0.7,
    N: int = 50,
    r: float = 3.9,
    n_steps: int = 3000,
    n_transient: int = 1000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Compute largest Lyapunov exponent vs coupling strength."""
    eps_values = np.linspace(eps_min, eps_max, n_eps)
    lyapunovs = []

    for i, eps_val in enumerate(eps_values):
        config = SimulationConfig(
            domain=Domain.COUPLED_MAP_LATTICE,
            dt=1.0,
            n_steps=1000,
            parameters={"N": float(N), "r": r, "eps": eps_val},
        )
        sim = CoupledMapLatticeSimulation(config)
        lam = sim.compute_lyapunov(
            n_steps=n_steps, n_transient=n_transient, seed=seed,
        )
        lyapunovs.append(lam)

        if (i + 1) % 5 == 0:
            logger.info(f"  eps={eps_val:.3f}: Lyapunov={lam:.4f}")

    return {
        "eps": eps_values,
        "lyapunov": np.array(lyapunovs),
    }


def generate_space_time_data(
    eps_values: list[float] | None = None,
    N: int = 100,
    r: float = 3.9,
    n_steps: int = 200,
    n_transient: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate space-time diagrams for representative coupling strengths."""
    if eps_values is None:
        eps_values = [0.0, 0.1, 0.3, 0.5]

    diagrams = {}
    for eps_val in eps_values:
        config = SimulationConfig(
            domain=Domain.COUPLED_MAP_LATTICE,
            dt=1.0,
            n_steps=1000,
            parameters={"N": float(N), "r": r, "eps": eps_val},
        )
        sim = CoupledMapLatticeSimulation(config)
        diagram = sim.space_time_diagram(
            n_steps=n_steps, n_transient=n_transient, seed=seed,
        )
        diagrams[f"eps_{eps_val:.2f}"] = diagram

    return diagrams


def run_coupled_map_lattice_rediscovery(
    output_dir: str | Path = "output/rediscovery/coupled_map_lattice",
    n_iterations: int = 40,
) -> dict:
    """Run CML rediscovery analysis.

    Targets:
    1. Coupling sweep: spatial variance vs eps
    2. Synchronization transition detection
    3. Lyapunov exponent vs eps
    4. PySR: fit order parameter curve
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "coupled_map_lattice",
        "targets": {
            "synchronization": "spatial variance -> 0 as eps -> 1",
            "spatiotemporal_chaos": "positive Lyapunov for intermediate eps",
            "order_parameter": "sync_order = 1 - var/var_uncoupled",
        },
    }

    # Part 1: Coupling sweep
    logger.info("Part 1: Coupling sweep (spatial variance vs eps)...")
    sweep_data = generate_coupling_sweep_data(n_eps=30, eps_max=0.7)
    results["coupling_sweep"] = {
        "n_eps": 30,
        "eps_range": [0.0, 0.7],
        "min_variance": float(np.min(sweep_data["variance_mean"])),
        "max_variance": float(np.max(sweep_data["variance_mean"])),
        "max_sync_order": float(np.max(sweep_data["sync_order"])),
    }

    # Detect synchronization threshold: where sync_order > 0.5
    sync_threshold_idx = np.where(sweep_data["sync_order"] > 0.5)[0]
    if len(sync_threshold_idx) > 0:
        eps_sync = float(sweep_data["eps"][sync_threshold_idx[0]])
        results["synchronization_threshold"] = {
            "eps_critical": eps_sync,
            "method": "sync_order > 0.5",
        }
        logger.info(f"  Synchronization threshold: eps ~ {eps_sync:.3f}")
    else:
        results["synchronization_threshold"] = {"eps_critical": None}

    # Part 2: Lyapunov exponent sweep
    logger.info("Part 2: Lyapunov exponent vs coupling...")
    lyap_data = generate_lyapunov_sweep_data(n_eps=20, eps_max=0.7, N=50)
    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "eps_at_max": float(lyap_data["eps"][np.argmax(lyap_data["lyapunov"])]),
    }

    # Find eps where Lyapunov crosses zero (chaos suppression)
    positive = lyap_data["lyapunov"] > 0
    if np.any(positive) and np.any(~positive):
        # Look for last positive -> first non-positive transition
        transitions = np.where(np.diff(positive.astype(int)) == -1)[0]
        if len(transitions) > 0:
            eps_suppress = float(lyap_data["eps"][transitions[-1] + 1])
            results["chaos_suppression"] = {
                "eps_estimate": eps_suppress,
                "method": "Lyapunov zero crossing",
            }
            logger.info(f"  Chaos suppression: eps ~ {eps_suppress:.3f}")

    # Part 3: PySR symbolic regression on order parameter curve
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = sweep_data["eps"].reshape(-1, 1)
        y = sweep_data["sync_order"]

        logger.info("  Running PySR: sync_order = f(eps)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["eps"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "exp"],
            max_complexity=12,
            populations=15,
            population_size=30,
        )
        results["sync_order_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["sync_order_pysr"]["best"] = best.expression
            results["sync_order_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["sync_order_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save numerical data
    np.savez(
        output_path / "coupling_sweep.npz",
        eps=sweep_data["eps"],
        variance_mean=sweep_data["variance_mean"],
        sync_order=sweep_data["sync_order"],
    )
    np.savez(
        output_path / "lyapunov_sweep.npz",
        eps=lyap_data["eps"],
        lyapunov=lyap_data["lyapunov"],
    )

    return results
