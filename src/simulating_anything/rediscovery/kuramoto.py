"""Kuramoto coupled oscillators rediscovery.

Targets:
- Critical coupling K_c for synchronization transition
  - Uniform on [-1,1]: K_c = 4/pi ~ 1.273
- Order parameter: r = sqrt(1 - K_c/K) for K > K_c
- Mean-field self-consistency equation
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.kuramoto import KuramotoSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_sync_transition_data(
    n_K: int = 40,
    N: int = 100,
    n_trials: int = 5,
    n_transient: int = 5000,
    n_measure: int = 2000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate order parameter r vs coupling K data.

    Sweeps K from 0 to 4 and measures steady-state order parameter.
    """
    K_values = np.linspace(0.1, 4.0, n_K)
    r_mean = []
    r_std = []

    for i, K in enumerate(K_values):
        r_trials = []
        for trial in range(n_trials):
            config = SimulationConfig(
                domain=Domain.KURAMOTO,
                dt=dt,
                n_steps=n_transient + n_measure,
                parameters={"N": float(N), "K": K, "omega_std": 1.0},
            )
            sim = KuramotoSimulation(config)
            r = sim.measure_steady_state_r(
                n_transient_steps=n_transient,
                n_measure_steps=n_measure,
                seed=42 + trial,
            )
            r_trials.append(r)

        r_mean.append(np.mean(r_trials))
        r_std.append(np.std(r_trials))

        if (i + 1) % 10 == 0:
            logger.info(f"  K={K:.3f}: r={np.mean(r_trials):.4f} +/- {np.std(r_trials):.4f}")

    return {
        "K": K_values,
        "r_mean": np.array(r_mean),
        "r_std": np.array(r_std),
        "N": N,
        "n_trials": n_trials,
    }


def generate_finite_size_data(
    N_values: list[int] | None = None,
    K: float = 2.0,
    n_trials: int = 10,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate order parameter vs system size N at fixed K > K_c."""
    if N_values is None:
        N_values = [10, 20, 50, 100, 200, 500]

    all_N = []
    all_r = []

    for N in N_values:
        r_trials = []
        for trial in range(n_trials):
            config = SimulationConfig(
                domain=Domain.KURAMOTO,
                dt=dt,
                n_steps=8000,
                parameters={"N": float(N), "K": K, "omega_std": 1.0},
            )
            sim = KuramotoSimulation(config)
            r = sim.measure_steady_state_r(
                n_transient_steps=5000,
                n_measure_steps=3000,
                seed=100 + trial,
            )
            r_trials.append(r)

        all_N.append(N)
        all_r.append(np.mean(r_trials))
        logger.info(f"  N={N}: r={np.mean(r_trials):.4f}")

    return {
        "N": np.array(all_N),
        "r": np.array(all_r),
        "K": K,
    }


def run_kuramoto_rediscovery(
    output_dir: str | Path = "output/rediscovery/kuramoto",
    n_iterations: int = 40,
) -> dict:
    """Run Kuramoto model rediscovery pipeline.

    1. Sweep coupling K, measure order parameter r
    2. Run PySR to find r(K) relationship
    3. Estimate critical coupling K_c
    4. Finite-size scaling analysis
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "kuramoto",
        "targets": {
            "critical_coupling": "K_c = 4/pi ~ 1.273 for uniform[-1,1]",
            "order_parameter": "r = sqrt(1 - K_c/K) for K > K_c",
        },
    }

    # --- Part 1: Sync transition ---
    logger.info("Part 1: Synchronization transition (r vs K)...")
    data = generate_sync_transition_data(n_K=40, N=100, n_trials=5, dt=0.01)

    # Estimate K_c: first K where r > threshold
    threshold = 0.3
    above = data["r_mean"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        K_c_est = data["K"][max(0, idx - 1)]
        results["K_c_estimate"] = float(K_c_est)
        K_c_theory = 4 / np.pi  # ~1.273 for uniform[-1,1]
        results["K_c_theory"] = float(K_c_theory)
        results["K_c_relative_error"] = float(abs(K_c_est - K_c_theory) / K_c_theory)
        logger.info(f"  K_c estimate: {K_c_est:.4f} (theory: {K_c_theory:.4f})")
    else:
        logger.warning("  Could not detect synchronization transition")

    results["sync_transition"] = {
        "n_K": len(data["K"]),
        "K_range": [float(data["K"].min()), float(data["K"].max())],
        "max_r": float(np.max(data["r_mean"])),
    }

    # PySR: find r(K) for K > K_c
    try:
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        # Use only supercritical data
        supercritical = data["r_mean"] > 0.1
        if np.sum(supercritical) > 5:
            X = data["K"][supercritical].reshape(-1, 1)
            y = data["r_mean"][supercritical]

            logger.info("  Running PySR: r = f(K) for K > K_c...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["K"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=12,
                populations=20,
                population_size=40,
            )
            results["order_param_pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["order_param_pysr"]["best"] = best.expression
                results["order_param_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["order_param_pysr"] = {"error": str(e)}

    # --- Part 2: Finite-size scaling ---
    logger.info("Part 2: Finite-size scaling...")
    fs_data = generate_finite_size_data(K=2.0, n_trials=5)
    results["finite_size"] = {
        "N_values": fs_data["N"].tolist(),
        "r_values": fs_data["r"].tolist(),
        "K": fs_data["K"],
    }

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "sync_transition.npz",
        K=data["K"],
        r_mean=data["r_mean"],
        r_std=data["r_std"],
    )
    np.savez(
        output_path / "finite_size.npz",
        N=fs_data["N"],
        r=fs_data["r"],
    )

    return results
