"""Bak-Sneppen model rediscovery.

Targets:
- SOC threshold f_c ~ 2/3 = 0.667 for 1D ring
- Avalanche size distribution: power law P(s) ~ s^{-tau}
- Gap evolution: minimum fitness converges to f_c from below
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.bak_sneppen import BakSneppen
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_threshold_data(
    N_values: list[int] | None = None,
    n_transient: int = 10000,
    n_measure: int = 10000,
    n_trials: int = 5,
) -> dict[str, np.ndarray]:
    """Measure SOC threshold vs system size N.

    For each N, run multiple trials and average the measured threshold.
    The threshold should converge to ~2/3 as N increases.
    """
    if N_values is None:
        N_values = [20, 50, 100, 200, 500]

    thresholds_mean = []
    thresholds_std = []

    for N in N_values:
        trial_thresholds = []
        for trial in range(n_trials):
            config = SimulationConfig(
                domain=Domain.BAK_SNEPPEN,
                dt=1.0,
                n_steps=n_transient + n_measure,
                parameters={"N": float(N)},
                seed=42 + trial,
            )
            sim = BakSneppen(config)
            threshold = sim.measure_soc_threshold(
                n_transient=n_transient,
                n_measure=n_measure,
                seed=42 + trial,
            )
            trial_thresholds.append(threshold)

        thresholds_mean.append(np.mean(trial_thresholds))
        thresholds_std.append(np.std(trial_thresholds))
        logger.info(
            f"  N={N}: threshold={np.mean(trial_thresholds):.4f}"
            f" +/- {np.std(trial_thresholds):.4f}"
        )

    return {
        "N": np.array(N_values),
        "threshold_mean": np.array(thresholds_mean),
        "threshold_std": np.array(thresholds_std),
    }


def generate_avalanche_data(
    N: int = 100,
    n_transient: int = 10000,
    n_avalanches: int = 2000,
    n_trials: int = 3,
) -> dict[str, np.ndarray]:
    """Measure avalanche size distribution.

    Collect avalanche sizes and fit a power law exponent tau.
    For the 1D Bak-Sneppen model, tau ~ 1.07 (mean-field) or ~1.4 (1D).
    """
    all_sizes = []

    for trial in range(n_trials):
        config = SimulationConfig(
            domain=Domain.BAK_SNEPPEN,
            dt=1.0,
            n_steps=n_transient + n_avalanches * 100,
            parameters={"N": float(N)},
            seed=100 + trial,
        )
        sim = BakSneppen(config)
        sizes = sim.measure_avalanche_distribution(
            n_transient=n_transient,
            n_avalanches=n_avalanches,
            seed=100 + trial,
        )
        all_sizes.extend(sizes)

    all_sizes = np.array(all_sizes, dtype=np.float64)

    # Fit power law exponent using log-log linear regression on the CCDF
    # P(S >= s) ~ s^{-(tau-1)}, so slope of log-CCDF gives -(tau-1)
    tau_estimate = None
    if len(all_sizes) > 10:
        # Use unique values for binning
        unique_sizes, counts = np.unique(all_sizes[all_sizes > 0], return_counts=True)
        if len(unique_sizes) > 3:
            # Cumulative distribution (survival function)
            ccdf = np.cumsum(counts[::-1])[::-1] / len(all_sizes)
            mask = unique_sizes > 0
            if np.sum(mask) > 3:
                log_s = np.log(unique_sizes[mask])
                log_ccdf = np.log(ccdf[mask])
                # Linear fit in log-log space
                valid = np.isfinite(log_s) & np.isfinite(log_ccdf)
                if np.sum(valid) > 3:
                    coeffs = np.polyfit(log_s[valid], log_ccdf[valid], 1)
                    tau_estimate = 1.0 - coeffs[0]  # tau = 1 - slope

    result = {
        "sizes": all_sizes,
        "n_avalanches": len(all_sizes),
        "mean_size": float(np.mean(all_sizes)) if len(all_sizes) > 0 else 0.0,
        "max_size": float(np.max(all_sizes)) if len(all_sizes) > 0 else 0.0,
    }
    if tau_estimate is not None:
        result["tau_estimate"] = float(tau_estimate)
        logger.info(f"  Power law exponent tau ~ {tau_estimate:.3f}")

    return result


def generate_gap_evolution_data(
    N: int = 100,
    n_steps: int = 20000,
) -> dict[str, np.ndarray]:
    """Track how the fitness gap (minimum fitness) evolves over time.

    Starting from random initial conditions, the minimum fitness should
    gradually increase and fluctuate around f_c ~ 2/3.
    """
    config = SimulationConfig(
        domain=Domain.BAK_SNEPPEN,
        dt=1.0,
        n_steps=n_steps,
        parameters={"N": float(N)},
        seed=42,
    )
    sim = BakSneppen(config)
    data = sim.measure_gap_evolution(n_steps=n_steps, seed=42)

    # Compute running average of minimum fitness in windows
    window = min(500, n_steps // 10)
    if len(data["min_fitness"]) >= window:
        running_avg = np.convolve(
            data["min_fitness"],
            np.ones(window) / window,
            mode="valid",
        )
    else:
        running_avg = data["min_fitness"]

    return {
        "steps": data["steps"],
        "min_fitness": data["min_fitness"],
        "mean_fitness": data["mean_fitness"],
        "running_avg_min": running_avg,
    }


def run_bak_sneppen_rediscovery(
    output_dir: str | Path = "output/rediscovery/bak_sneppen",
    n_iterations: int = 40,
) -> dict:
    """Run Bak-Sneppen model rediscovery pipeline.

    1. Measure SOC threshold vs N
    2. Measure avalanche size distribution (power law)
    3. Track gap evolution
    4. PySR: fit threshold(N) relationship

    Args:
        output_dir: Directory for output files.
        n_iterations: Number of PySR iterations.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "bak_sneppen",
        "targets": {
            "soc_threshold": "f_c ~ 2/3 = 0.667 for 1D ring",
            "avalanche_exponent": "P(s) ~ s^{-tau}, tau ~ 1.07-1.4",
            "gap_convergence": "min fitness -> f_c over time",
        },
    }

    # --- Part 1: SOC threshold vs N ---
    logger.info("Part 1: SOC threshold vs system size N...")
    threshold_data = generate_threshold_data(
        N_values=[20, 50, 100, 200, 500],
        n_transient=5000,
        n_measure=5000,
        n_trials=3,
    )

    results["threshold_vs_N"] = {
        "N_values": threshold_data["N"].tolist(),
        "threshold_mean": threshold_data["threshold_mean"].tolist(),
        "threshold_std": threshold_data["threshold_std"].tolist(),
    }

    # Best threshold estimate: use largest N
    best_threshold = threshold_data["threshold_mean"][-1]
    results["soc_threshold_estimate"] = float(best_threshold)
    results["soc_threshold_theory"] = 2.0 / 3.0
    results["soc_threshold_error"] = float(
        abs(best_threshold - 2.0 / 3.0) / (2.0 / 3.0)
    )
    logger.info(
        f"  Best threshold estimate: {best_threshold:.4f}"
        f" (theory: 0.6667, error: {results['soc_threshold_error']:.2%})"
    )

    # --- Part 2: Avalanche distribution ---
    logger.info("Part 2: Avalanche size distribution...")
    aval_data = generate_avalanche_data(
        N=100, n_transient=5000, n_avalanches=1000, n_trials=3,
    )
    results["avalanche"] = {
        "n_avalanches": aval_data["n_avalanches"],
        "mean_size": aval_data["mean_size"],
        "max_size": aval_data["max_size"],
    }
    if "tau_estimate" in aval_data:
        results["avalanche"]["tau_estimate"] = aval_data["tau_estimate"]

    # --- Part 3: Gap evolution ---
    logger.info("Part 3: Gap evolution...")
    gap_data = generate_gap_evolution_data(N=100, n_steps=10000)

    # Measure the final gap value (average of last 10% of steps)
    n_final = max(1, len(gap_data["min_fitness"]) // 10)
    final_avg_min = float(np.mean(gap_data["min_fitness"][-n_final:]))
    final_avg_mean = float(np.mean(gap_data["mean_fitness"][-n_final:]))

    results["gap_evolution"] = {
        "n_steps": len(gap_data["steps"]),
        "final_mean_min_fitness": final_avg_min,
        "final_mean_fitness": final_avg_mean,
    }
    logger.info(
        f"  Final avg min fitness: {final_avg_min:.4f},"
        f" final avg mean fitness: {final_avg_mean:.4f}"
    )

    # --- Part 4: PySR on threshold vs N ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        if len(threshold_data["N"]) >= 4:
            X = threshold_data["N"].reshape(-1, 1).astype(np.float64)
            y = threshold_data["threshold_mean"]

            logger.info("  Running PySR: threshold = f(N)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["N_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["log", "sqrt"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["threshold_pysr"] = {
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
                results["threshold_pysr"]["best"] = best.expression
                results["threshold_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression}"
                    f" (R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["threshold_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data arrays
    np.savez(
        output_path / "threshold_data.npz",
        N=threshold_data["N"],
        threshold_mean=threshold_data["threshold_mean"],
        threshold_std=threshold_data["threshold_std"],
    )
    np.savez(
        output_path / "gap_evolution.npz",
        steps=gap_data["steps"],
        min_fitness=gap_data["min_fitness"],
        mean_fitness=gap_data["mean_fitness"],
    )

    return results
