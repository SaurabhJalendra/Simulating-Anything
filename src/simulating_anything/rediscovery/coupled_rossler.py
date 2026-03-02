"""Coupled Rossler systems rediscovery.

Targets:
- Synchronization transition as epsilon increases (phase -> complete sync)
- Critical coupling epsilon_c for synchronization
- Conditional Lyapunov exponent negative for epsilon > epsilon_c
- Independent chaos at epsilon=0, synchronized state at large epsilon
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.coupled_rossler import CoupledRosslerSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_sync_sweep_data(
    n_eps: int = 30,
    eps_min: float = 0.0,
    eps_max: float = 0.3,
    n_steps: int = 5000,
    n_transient: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep coupling strength and measure steady-state sync error.

    At each epsilon value, runs the coupled system, discards transient,
    and computes time-averaged synchronization error.

    Args:
        n_eps: Number of epsilon values to sample.
        eps_min: Minimum coupling strength.
        eps_max: Maximum coupling strength.
        n_steps: Measurement steps after transient.
        n_transient: Transient steps to discard.
        dt: Timestep.

    Returns:
        Dict with 'epsilon' and 'mean_error' arrays.
    """
    eps_values = np.linspace(eps_min, eps_max, n_eps)
    mean_errors = np.empty(n_eps)

    for i, eps_val in enumerate(eps_values):
        config = SimulationConfig(
            domain=Domain.COUPLED_ROSSLER,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": 0.2, "b": 0.2, "c": 5.7,
                "epsilon": float(eps_val),
                "x1_0": 1.0, "y1_0": 0.0, "z1_0": 0.0,
                "x2_0": 0.5, "y2_0": 0.5, "z2_0": 0.5,
            },
        )
        sim = CoupledRosslerSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(n_transient):
            sim.step()

        # Measure average error
        error_sum = 0.0
        for _ in range(n_steps):
            sim.step()
            error_sum += sim.sync_error()

        mean_errors[i] = error_sum / n_steps

        if (i + 1) % 5 == 0:
            logger.info(
                f"  epsilon={eps_val:.3f}: mean_error={mean_errors[i]:.4f}"
            )

    return {
        "epsilon": eps_values,
        "mean_error": mean_errors,
    }


def estimate_critical_coupling(
    sweep_data: dict[str, np.ndarray],
    error_threshold: float = 1.0,
) -> float:
    """Estimate critical coupling from a sync sweep.

    epsilon_c is the smallest epsilon where the mean sync error drops below
    the given threshold. Uses linear interpolation between sweep points.

    Args:
        sweep_data: Output of generate_sync_sweep_data().
        error_threshold: Threshold for "synchronized" state.

    Returns:
        Estimated epsilon_c.
    """
    eps = sweep_data["epsilon"]
    errors = sweep_data["mean_error"]

    # Find first crossing below threshold
    for j in range(len(errors) - 1):
        if errors[j] >= error_threshold and errors[j + 1] < error_threshold:
            frac = (error_threshold - errors[j]) / (errors[j + 1] - errors[j])
            return float(eps[j] + frac * (eps[j + 1] - eps[j]))

    # If all above threshold, return max eps
    if np.all(errors >= error_threshold):
        return float(eps[-1])
    # If all below threshold, return min eps
    return float(eps[0])


def generate_lyapunov_data(
    eps_values: np.ndarray | None = None,
    n_compute: int = 20000,
    n_transient: int = 3000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Compute max Lyapunov exponent of the full 6D system vs epsilon.

    Args:
        eps_values: Coupling strengths to test.
        n_compute: Steps for Lyapunov computation.
        n_transient: Transient steps.
        dt: Timestep.

    Returns:
        Dict with 'epsilon' and 'lyapunov' arrays.
    """
    if eps_values is None:
        eps_values = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])

    lyap_values = np.empty(len(eps_values))

    for i, eps_val in enumerate(eps_values):
        config = SimulationConfig(
            domain=Domain.COUPLED_ROSSLER,
            dt=dt,
            n_steps=n_compute,
            parameters={
                "a": 0.2, "b": 0.2, "c": 5.7,
                "epsilon": float(eps_val),
                "x1_0": 1.0, "y1_0": 0.0, "z1_0": 0.0,
                "x2_0": 0.5, "y2_0": 0.5, "z2_0": 0.5,
            },
        )
        sim = CoupledRosslerSimulation(config)
        sim.reset()
        lam = sim.compute_lyapunov(
            n_transient=n_transient, n_compute=n_compute,
        )
        lyap_values[i] = lam
        logger.info(f"  epsilon={eps_val:.3f}: Lyapunov={lam:.4f}")

    return {"epsilon": eps_values, "lyapunov": lyap_values}


def run_coupled_rossler_rediscovery(
    output_dir: str | Path = "output/rediscovery/coupled_rossler",
    n_iterations: int = 40,
) -> dict:
    """Run the full coupled Rossler synchronization rediscovery.

    1. Sweep epsilon from 0 to 0.3 and measure steady-state sync error
    2. Identify critical coupling epsilon_c
    3. Compute Lyapunov exponents vs epsilon
    4. Verify independent chaos at epsilon=0
    5. Verify synchronization at large epsilon

    Args:
        output_dir: Directory for output files.
        n_iterations: Not used directly; kept for API compatibility.

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "coupled_rossler",
        "targets": {
            "phenomenon": "Coupled Rossler synchronization via diffusive x-coupling",
            "sync_transition": "Phase -> complete sync as epsilon increases",
            "critical_coupling": "epsilon_c where sync error drops significantly",
            "lyapunov": "Positive at eps=0 (chaos), decreases with coupling",
        },
    }

    # --- Part 1: Synchronization sweep ---
    logger.info("Part 1: Sweeping coupling strength epsilon...")
    sweep_data = generate_sync_sweep_data(
        n_eps=30, eps_min=0.0, eps_max=0.3,
        n_steps=5000, n_transient=5000, dt=0.01,
    )

    eps_c = estimate_critical_coupling(sweep_data, error_threshold=1.0)
    logger.info(f"  Estimated epsilon_c = {eps_c:.4f}")

    results["sync_sweep"] = {
        "n_eps_values": len(sweep_data["epsilon"]),
        "eps_range": [
            float(sweep_data["epsilon"][0]),
            float(sweep_data["epsilon"][-1]),
        ],
        "min_error": float(np.min(sweep_data["mean_error"])),
        "max_error": float(np.max(sweep_data["mean_error"])),
        "eps_c_estimate": eps_c,
    }

    # --- Part 2: Lyapunov exponents vs epsilon ---
    logger.info("Part 2: Computing Lyapunov exponents vs epsilon...")
    lyap_data = generate_lyapunov_data(
        eps_values=np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]),
        n_compute=20000, n_transient=3000, dt=0.01,
    )

    results["lyapunov"] = {
        "epsilon": lyap_data["epsilon"].tolist(),
        "values": lyap_data["lyapunov"].tolist(),
        "n_positive": int(np.sum(lyap_data["lyapunov"] > 0)),
        "lyap_at_zero": float(lyap_data["lyapunov"][0]),
    }

    # --- Part 3: Verify independent chaos at epsilon=0 ---
    logger.info("Part 3: Verifying independent chaos at epsilon=0...")
    config_uncoupled = SimulationConfig(
        domain=Domain.COUPLED_ROSSLER,
        dt=0.01,
        n_steps=10000,
        parameters={
            "a": 0.2, "b": 0.2, "c": 5.7,
            "epsilon": 0.0,
            "x1_0": 1.0, "y1_0": 0.0, "z1_0": 0.0,
            "x2_0": 0.5, "y2_0": 0.5, "z2_0": 0.5,
        },
    )
    sim_uncoupled = CoupledRosslerSimulation(config_uncoupled)
    sim_uncoupled.reset()

    # Skip transient
    for _ in range(5000):
        sim_uncoupled.step()

    # Measure sync error -- should be large (independent chaos)
    errors_uncoupled = []
    for _ in range(5000):
        sim_uncoupled.step()
        errors_uncoupled.append(sim_uncoupled.sync_error())
    mean_err_uncoupled = float(np.mean(errors_uncoupled))
    logger.info(f"  Mean sync error at eps=0: {mean_err_uncoupled:.4f}")

    results["uncoupled_verification"] = {
        "epsilon": 0.0,
        "mean_sync_error": mean_err_uncoupled,
        "independent_chaos": bool(mean_err_uncoupled > 1.0),
    }

    # --- Part 4: Verify synchronization at large epsilon ---
    logger.info("Part 4: Verifying sync at epsilon=0.3...")
    config_sync = SimulationConfig(
        domain=Domain.COUPLED_ROSSLER,
        dt=0.01,
        n_steps=20000,
        parameters={
            "a": 0.2, "b": 0.2, "c": 5.7,
            "epsilon": 0.3,
            "x1_0": 1.0, "y1_0": 0.0, "z1_0": 0.0,
            "x2_0": 0.5, "y2_0": 0.5, "z2_0": 0.5,
        },
    )
    sim_sync = CoupledRosslerSimulation(config_sync)
    sim_sync.reset()

    for _ in range(20000):
        sim_sync.step()
    final_error = sim_sync.sync_error()
    logger.info(f"  Final sync error at eps=0.3: {final_error:.6e}")

    results["sync_verification"] = {
        "epsilon": 0.3,
        "final_sync_error": float(final_error),
        "synchronized": bool(final_error < 1e-2),
    }

    # --- Part 5: Conditional Lyapunov exponent ---
    logger.info("Part 5: Computing conditional Lyapunov exponents...")
    cond_eps_values = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    cond_lyap = np.empty(len(cond_eps_values))

    config_cond = SimulationConfig(
        domain=Domain.COUPLED_ROSSLER,
        dt=0.01,
        n_steps=30000,
        parameters={
            "a": 0.2, "b": 0.2, "c": 5.7,
            "epsilon": 0.05,
        },
    )
    sim_cond = CoupledRosslerSimulation(config_cond)
    sim_cond.reset()

    for i, eps_val in enumerate(cond_eps_values):
        lam = sim_cond.conditional_lyapunov(
            epsilon=float(eps_val),
            n_steps=20000,
            n_transient=3000,
        )
        cond_lyap[i] = lam
        logger.info(f"  epsilon={eps_val:.3f}: cond_Lyapunov={lam:.4f}")

    results["conditional_lyapunov"] = {
        "epsilon": cond_eps_values.tolist(),
        "values": cond_lyap.tolist(),
        "n_negative": int(np.sum(cond_lyap < 0)),
        "n_positive": int(np.sum(cond_lyap > 0)),
    }

    # Find zero crossing for conditional Lyapunov
    for j in range(len(cond_lyap) - 1):
        if cond_lyap[j] > 0 and cond_lyap[j + 1] <= 0:
            frac = -cond_lyap[j] / (cond_lyap[j + 1] - cond_lyap[j])
            eps_c_lyap = cond_eps_values[j] + frac * (
                cond_eps_values[j + 1] - cond_eps_values[j]
            )
            results["conditional_lyapunov"]["eps_c_from_lyapunov"] = float(
                eps_c_lyap
            )
            logger.info(
                f"  eps_c from conditional Lyapunov: {eps_c_lyap:.4f}"
            )
            break

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save sweep data
    np.savez(
        output_path / "sync_sweep.npz",
        epsilon=sweep_data["epsilon"],
        mean_error=sweep_data["mean_error"],
    )
    np.savez(
        output_path / "lyapunov_data.npz",
        epsilon=lyap_data["epsilon"],
        lyapunov=lyap_data["lyapunov"],
    )

    return results
