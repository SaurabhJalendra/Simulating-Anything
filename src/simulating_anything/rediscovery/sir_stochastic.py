"""Stochastic SIR epidemic model rediscovery.

Targets:
- Ensemble mean trajectory converges to deterministic SIR
- Variance scaling: Var ~ sigma^2 (noise intensity)
- Variance scaling: Var ~ 1/N (population size)
- Extinction probability vs R0 near critical threshold
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sir_stochastic import SIRStochasticSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ensemble_comparison_data(
    n_runs: int = 50,
    n_steps: int = 2000,
    dt: float = 0.1,
    beta: float = 0.5,
    gamma: float = 0.1,
    N0: float = 1000.0,
    sigma: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Compare ensemble mean with deterministic SIR trajectory.

    Runs multiple stochastic simulations and compares their mean
    against the noise-free deterministic solution.

    Returns:
        Dictionary with deterministic trajectory, ensemble mean/std,
        and error metrics.
    """
    config = SimulationConfig(
        domain=Domain.SIR_STOCHASTIC,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        parameters={
            "beta": beta,
            "gamma": gamma,
            "N0": N0,
            "sigma": sigma,
            "I_0": 10.0,
        },
    )
    sim = SIRStochasticSimulation(config)

    # Deterministic trajectory (RK4, no noise)
    det_traj = sim.run_deterministic(n_steps=n_steps)

    # Stochastic ensemble
    ensemble = sim.run_ensemble(n_runs=n_runs, n_steps=n_steps, base_seed=seed)

    ensemble_mean = np.mean(ensemble, axis=0)
    ensemble_std = np.std(ensemble, axis=0)

    # Mean absolute error between ensemble mean and deterministic
    mae = np.mean(np.abs(ensemble_mean - det_traj))

    # Relative error at peak infected (where deviation is largest)
    det_peak_idx = np.argmax(det_traj[:, 1])
    det_peak_I = det_traj[det_peak_idx, 1]
    ens_peak_I = ensemble_mean[det_peak_idx, 1]
    if det_peak_I > 0:
        peak_rel_error = abs(ens_peak_I - det_peak_I) / det_peak_I
    else:
        peak_rel_error = 0.0

    return {
        "deterministic": det_traj,
        "ensemble_mean": ensemble_mean,
        "ensemble_std": ensemble_std,
        "mae": mae,
        "peak_rel_error": peak_rel_error,
        "n_runs": n_runs,
        "beta": beta,
        "gamma": gamma,
        "N0": N0,
        "sigma": sigma,
        "dt": dt,
    }


def generate_variance_scaling_data(
    sigma_values: np.ndarray | None = None,
    N_values: np.ndarray | None = None,
    n_runs: int = 30,
    n_steps: int = 500,
    dt: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Measure variance scaling with noise intensity and population size.

    Part 1: Fix N, sweep sigma -- expect Var ~ sigma^2
    Part 2: Fix sigma, sweep N -- expect Var ~ 1/N

    Returns:
        Dictionary with sigma_values, N_values, and measured variances.
    """
    if sigma_values is None:
        sigma_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    if N_values is None:
        N_values = np.array([100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0])

    # Part 1: Sweep sigma (fix N=1000)
    sigma_variances = np.zeros(len(sigma_values))
    N_fixed = 1000.0

    for i, sigma in enumerate(sigma_values):
        config = SimulationConfig(
            domain=Domain.SIR_STOCHASTIC,
            dt=dt,
            n_steps=n_steps,
            seed=seed,
            parameters={
                "beta": 0.5,
                "gamma": 0.1,
                "N0": N_fixed,
                "sigma": sigma,
                "I_0": 10.0,
            },
        )
        sim = SIRStochasticSimulation(config)
        ensemble = sim.run_ensemble(n_runs=n_runs, n_steps=n_steps, base_seed=seed + i * 100)

        # Measure variance of I at a fixed time point (step n_steps//2)
        mid_I = ensemble[:, n_steps // 2, 1]
        sigma_variances[i] = np.var(mid_I)

        logger.info(f"  sigma={sigma:.3f}: Var(I)={sigma_variances[i]:.4f}")

    # Part 2: Sweep N (fix sigma=0.1)
    n_variances = np.zeros(len(N_values))
    sigma_fixed = 0.1

    for i, N0 in enumerate(N_values):
        I_0 = max(1.0, N0 * 0.01)  # 1% initially infected
        config = SimulationConfig(
            domain=Domain.SIR_STOCHASTIC,
            dt=dt,
            n_steps=n_steps,
            seed=seed,
            parameters={
                "beta": 0.5,
                "gamma": 0.1,
                "N0": N0,
                "sigma": sigma_fixed,
                "I_0": I_0,
            },
        )
        sim = SIRStochasticSimulation(config)
        ensemble = sim.run_ensemble(n_runs=n_runs, n_steps=n_steps, base_seed=seed + i * 100)

        # Measure variance of I/N (fraction) at midpoint
        mid_I_frac = ensemble[:, n_steps // 2, 1] / N0
        n_variances[i] = np.var(mid_I_frac)

        logger.info(f"  N={N0:.0f}: Var(I/N)={n_variances[i]:.6f}")

    return {
        "sigma_values": sigma_values,
        "sigma_variances": sigma_variances,
        "N_values": N_values,
        "n_variances": n_variances,
        "n_runs": n_runs,
    }


def generate_extinction_data(
    R0_values: np.ndarray | None = None,
    n_runs: int = 50,
    n_steps: int = 2000,
    dt: float = 0.1,
    N0: float = 1000.0,
    sigma: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Measure extinction probability as a function of R0.

    Extinction is defined as I < 0.5 before the epidemic peaks.
    Stochastic effects can cause extinction even when R0 > 1.

    Returns:
        Dictionary with R0_values, extinction probabilities, and mean peak I.
    """
    if R0_values is None:
        R0_values = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0])

    gamma = 0.1
    extinction_probs = np.zeros(len(R0_values))
    mean_peak_I = np.zeros(len(R0_values))

    for i, r0 in enumerate(R0_values):
        beta = r0 * gamma
        extinctions = 0
        peak_Is = []

        for run in range(n_runs):
            config = SimulationConfig(
                domain=Domain.SIR_STOCHASTIC,
                dt=dt,
                n_steps=n_steps,
                seed=seed + i * n_runs + run,
                parameters={
                    "beta": beta,
                    "gamma": gamma,
                    "N0": N0,
                    "sigma": sigma,
                    "I_0": 10.0,
                },
            )
            sim = SIRStochasticSimulation(config)
            sim.reset(seed=seed + i * n_runs + run)

            max_I = 0.0
            extinct = False
            for _ in range(n_steps):
                state = sim.step()
                if state[1] > max_I:
                    max_I = state[1]
                # Extinction: I drops below 0.5 after initial period
                if state[1] < 0.5 and sim._step_count > 50:
                    extinct = True
                    break

            if extinct and max_I < N0 * 0.05:
                extinctions += 1
            peak_Is.append(max_I)

        extinction_probs[i] = extinctions / n_runs
        mean_peak_I[i] = np.mean(peak_Is)

        logger.info(
            f"  R0={r0:.2f}: extinction={extinction_probs[i]:.2f}, "
            f"mean_peak_I={mean_peak_I[i]:.1f}"
        )

    return {
        "R0_values": R0_values,
        "extinction_probs": extinction_probs,
        "mean_peak_I": mean_peak_I,
        "n_runs": n_runs,
        "N0": N0,
        "sigma": sigma,
    }


def run_sir_stochastic_rediscovery(
    output_dir: str | Path = "output/rediscovery/sir_stochastic",
    n_iterations: int = 50,
) -> dict:
    """Run the full stochastic SIR rediscovery pipeline.

    1. Compare ensemble mean with deterministic SIR
    2. Measure variance scaling with sigma and N
    3. Measure extinction probability vs R0

    Args:
        output_dir: directory for results output.
        n_iterations: not used for PySR here, kept for API consistency.

    Returns:
        Results dictionary with all analysis data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sir_stochastic",
        "targets": {
            "ensemble_mean": "Ensemble mean converges to deterministic SIR",
            "variance_sigma": "Var(I) ~ sigma^2",
            "variance_N": "Var(I/N) ~ 1/N",
            "extinction": "P(extinction) increases for R0 near 1",
        },
    }

    # --- Part 1: Ensemble mean vs deterministic ---
    logger.info("Part 1: Ensemble mean vs deterministic SIR...")
    comparison = generate_ensemble_comparison_data(
        n_runs=50, n_steps=2000, dt=0.1,
    )
    results["ensemble_comparison"] = {
        "mae": float(comparison["mae"]),
        "peak_rel_error": float(comparison["peak_rel_error"]),
        "n_runs": int(comparison["n_runs"]),
    }
    logger.info(
        f"  MAE={comparison['mae']:.4f}, "
        f"peak_rel_error={comparison['peak_rel_error']:.4f}"
    )

    # --- Part 2: Variance scaling ---
    logger.info("Part 2: Variance scaling with sigma and N...")
    variance_data = generate_variance_scaling_data(
        n_runs=30, n_steps=500, dt=0.1,
    )

    # Check sigma^2 scaling
    sigma_vals = variance_data["sigma_values"]
    sigma_vars = variance_data["sigma_variances"]
    # Log-log correlation between sigma^2 and variance
    valid_sigma = sigma_vars > 0
    if np.sum(valid_sigma) > 2:
        log_sigma2 = np.log(sigma_vals[valid_sigma] ** 2)
        log_var = np.log(sigma_vars[valid_sigma])
        corr_sigma = float(np.corrcoef(log_sigma2, log_var)[0, 1])
    else:
        corr_sigma = 0.0

    results["variance_sigma_scaling"] = {
        "sigma_values": [float(x) for x in sigma_vals],
        "variances": [float(x) for x in sigma_vars],
        "log_log_correlation": corr_sigma,
    }
    logger.info(f"  Var ~ sigma^2 correlation: {corr_sigma:.4f}")

    # Check 1/N scaling
    n_vals = variance_data["N_values"]
    n_vars = variance_data["n_variances"]
    valid_n = n_vars > 0
    if np.sum(valid_n) > 2:
        log_inv_n = np.log(1.0 / n_vals[valid_n])
        log_var_n = np.log(n_vars[valid_n])
        corr_n = float(np.corrcoef(log_inv_n, log_var_n)[0, 1])
    else:
        corr_n = 0.0

    results["variance_n_scaling"] = {
        "N_values": [float(x) for x in n_vals],
        "variances": [float(x) for x in n_vars],
        "log_log_correlation": corr_n,
    }
    logger.info(f"  Var ~ 1/N correlation: {corr_n:.4f}")

    # --- Part 3: Extinction probability ---
    logger.info("Part 3: Extinction probability vs R0...")
    extinction_data = generate_extinction_data(
        n_runs=50, n_steps=2000, dt=0.1,
    )
    results["extinction"] = {
        "R0_values": [float(x) for x in extinction_data["R0_values"]],
        "extinction_probs": [float(x) for x in extinction_data["extinction_probs"]],
        "mean_peak_I": [float(x) for x in extinction_data["mean_peak_I"]],
    }

    # --- Save results ---
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
