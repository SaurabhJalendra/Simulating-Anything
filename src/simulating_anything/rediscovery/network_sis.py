"""Network SIS epidemic model rediscovery.

Targets:
- Epidemic threshold: tau_c = beta_c/gamma = 1/lambda_max(A)
- Prevalence vs beta/gamma relationship
- Network topology comparison: ER, regular, complete
- Optional PySR: prevalence = f(tau, lambda_max)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.network_sis import NetworkSISSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_threshold_sweep_data(
    N: int = 50,
    mean_degree: float = 6.0,
    n_beta: int = 30,
    n_transient: int = 8000,
    n_measure: int = 2000,
    dt: float = 0.01,
    seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Sweep beta/gamma ratio across the epidemic threshold for an ER network.

    Returns dict with tau (beta/gamma), prevalence, and theoretical threshold.
    """
    config = SimulationConfig(
        domain=Domain.NETWORK_SIS,
        dt=dt,
        n_steps=1000,
        parameters={
            "N": float(N),
            "beta": 0.1,
            "gamma": 0.1,
            "mean_degree": mean_degree,
            "initial_fraction": 0.2,
        },
    )
    sim = NetworkSISSimulation(config)
    sim.reset(seed=seed)

    # Compute theoretical threshold
    tau_c = sim.compute_epidemic_threshold()
    lam_max = sim.spectral_radius()

    # Sweep tau = beta/gamma from below to above threshold
    gamma = 0.1
    tau_min = max(0.01, tau_c * 0.2)
    tau_max = tau_c * 5.0
    tau_values = np.linspace(tau_min, tau_max, n_beta)
    beta_values = tau_values * gamma

    all_tau = []
    all_prevalence = []

    for i, (tau, beta) in enumerate(zip(tau_values, beta_values)):
        sim.beta = beta
        sim.gamma = gamma
        sim.reset(seed=seed)
        for _ in range(n_transient):
            sim.step()
        prev_vals = []
        for _ in range(n_measure):
            sim.step()
            prev_vals.append(sim.compute_prevalence())
        prevalence = float(np.mean(prev_vals))

        all_tau.append(tau)
        all_prevalence.append(prevalence)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  Sweep {i + 1}/{n_beta}: tau={tau:.3f}, prevalence={prevalence:.4f}"
            )

    return {
        "tau": np.array(all_tau),
        "prevalence": np.array(all_prevalence),
        "tau_c_theory": float(tau_c),
        "lambda_max": float(lam_max),
        "N": N,
        "mean_degree": mean_degree,
        "gamma": gamma,
    }


def generate_topology_comparison_data(
    N: int = 50,
    mean_degree: float = 6.0,
    n_beta: int = 20,
    n_transient: int = 8000,
    n_measure: int = 2000,
    dt: float = 0.01,
    seed: int = 42,
) -> dict[str, dict]:
    """Compare epidemic thresholds across network topologies.

    Returns dict keyed by network type with sweep data and thresholds.
    """
    results = {}
    gamma = 0.1

    for net_type in ["erdos_renyi", "regular", "complete"]:
        config = SimulationConfig(
            domain=Domain.NETWORK_SIS,
            dt=dt,
            n_steps=1000,
            parameters={
                "N": float(N),
                "beta": 0.1,
                "gamma": gamma,
                "mean_degree": mean_degree,
                "initial_fraction": 0.2,
            },
        )
        sim = NetworkSISSimulation(config)
        sim.network_type = net_type
        sim.reset(seed=seed)

        tau_c = sim.compute_epidemic_threshold()
        lam_max = sim.spectral_radius()
        degrees = sim.degree_distribution()

        # Sweep tau around the threshold
        tau_min = max(0.01, tau_c * 0.3)
        tau_max = max(tau_c * 4.0, 0.5)
        tau_values = np.linspace(tau_min, tau_max, n_beta)

        prevalences = []
        for tau in tau_values:
            sim.beta = tau * gamma
            sim.gamma = gamma
            sim.reset(seed=seed)
            for _ in range(n_transient):
                sim.step()
            prev_vals = []
            for _ in range(n_measure):
                sim.step()
                prev_vals.append(sim.compute_prevalence())
            prevalences.append(float(np.mean(prev_vals)))

        results[net_type] = {
            "tau": tau_values.tolist(),
            "prevalence": prevalences,
            "tau_c": float(tau_c),
            "lambda_max": float(lam_max),
            "mean_degree": float(np.mean(degrees)),
            "degree_std": float(np.std(degrees)),
        }
        logger.info(
            f"  {net_type}: tau_c={tau_c:.4f}, lambda_max={lam_max:.2f}, "
            f"<k>={np.mean(degrees):.1f}"
        )

    return results


def run_network_sis_rediscovery(
    output_dir: str | Path = "output/rediscovery/network_sis",
    n_iterations: int = 40,
    N: int = 50,
    mean_degree: float = 6.0,
) -> dict:
    """Run the full Network SIS epidemic rediscovery.

    1. Sweep beta/gamma across the epidemic threshold
    2. Compare network topologies
    3. Optional PySR: fit prevalence = f(tau, lambda_max)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "network_sis",
        "targets": {
            "epidemic_threshold": "tau_c = 1/lambda_max(A)",
            "prevalence": "endemic prevalence above threshold",
        },
    }

    # --- Part 1: Threshold sweep ---
    logger.info("Part 1: Generating epidemic threshold sweep data...")
    sweep_data = generate_threshold_sweep_data(
        N=N, mean_degree=mean_degree, n_beta=30,
    )

    # Detect threshold from data: where prevalence first exceeds a small value
    tau_arr = sweep_data["tau"]
    prev_arr = sweep_data["prevalence"]
    threshold_mask = prev_arr > 0.01
    if np.any(threshold_mask):
        tau_c_empirical = float(tau_arr[np.argmax(threshold_mask)])
    else:
        tau_c_empirical = float(tau_arr[-1])

    results["threshold_sweep"] = {
        "n_points": len(tau_arr),
        "tau_c_theory": sweep_data["tau_c_theory"],
        "tau_c_empirical": tau_c_empirical,
        "lambda_max": sweep_data["lambda_max"],
        "max_prevalence": float(np.max(prev_arr)),
        "threshold_error": abs(tau_c_empirical - sweep_data["tau_c_theory"]),
    }
    logger.info(
        f"  Threshold: theory={sweep_data['tau_c_theory']:.4f}, "
        f"empirical={tau_c_empirical:.4f}"
    )

    # --- Part 2: Topology comparison ---
    logger.info("Part 2: Comparing network topologies...")
    topo_data = generate_topology_comparison_data(
        N=N, mean_degree=mean_degree, n_beta=15,
    )
    results["topology_comparison"] = {}
    for net_type, data in topo_data.items():
        results["topology_comparison"][net_type] = {
            "tau_c": data["tau_c"],
            "lambda_max": data["lambda_max"],
            "mean_degree": data["mean_degree"],
        }
        logger.info(
            f"  {net_type}: tau_c={data['tau_c']:.4f}, "
            f"lambda_max={data['lambda_max']:.2f}"
        )

    # --- Part 3: PySR symbolic regression (optional) ---
    logger.info("Part 3: PySR symbolic regression...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use threshold sweep data: predict prevalence from tau
        mask = prev_arr > 0.001  # Only above-threshold data
        if np.sum(mask) > 5:
            X = tau_arr[mask].reshape(-1, 1)
            y = prev_arr[mask]
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["tau"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "exp"],
                max_complexity=15,
                populations=20,
                population_size=40,
            )
            results["prevalence_pysr"] = {
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
                results["prevalence_pysr"]["best"] = best.expression
                results["prevalence_pysr"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        else:
            results["prevalence_pysr"] = {"error": "Not enough above-threshold data"}

    except ImportError:
        logger.warning("PySR not available -- skipping symbolic regression")
        results["prevalence_pysr"] = {"error": "PySR not installed"}
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["prevalence_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "sweep_data.npz",
        tau=tau_arr,
        prevalence=prev_arr,
    )

    return results
