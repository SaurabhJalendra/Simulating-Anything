"""Coupled Lorenz systems rediscovery.

Targets:
- Critical coupling eps_c for chaos synchronization transition
- Synchronization error decay rate vs eps
- Conditional Lyapunov exponent negative for eps > eps_c
- Transient synchronization time vs eps
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.coupled_lorenz import CoupledLorenzSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_sync_sweep_data(
    n_eps: int = 25,
    eps_min: float = 0.0,
    eps_max: float = 10.0,
    n_steps: int = 10000,
    n_transient: int = 5000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep coupling strength and measure steady-state sync error.

    At each eps value, runs the coupled system, discards transient,
    and computes time-averaged synchronization error.
    """
    eps_values = np.linspace(eps_min, eps_max, n_eps)
    mean_errors = np.empty(n_eps)

    for i, eps_val in enumerate(eps_values):
        config = SimulationConfig(
            domain=Domain.COUPLED_LORENZ,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0,
                "eps": eps_val,
                "x1_0": 1.0, "y1_0": 1.0, "z1_0": 1.0,
                "x2_0": -5.0, "y2_0": 5.0, "z2_0": 25.0,
            },
        )
        sim = CoupledLorenzSimulation(config)
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
                f"  eps={eps_val:.2f}: mean_error={mean_errors[i]:.4f}"
            )

    return {
        "eps": eps_values,
        "mean_error": mean_errors,
    }


def generate_transient_data(
    eps_values: np.ndarray | None = None,
    n_steps: int = 20000,
    dt: float = 0.005,
    threshold: float = 0.1,
) -> dict[str, np.ndarray]:
    """Measure synchronization transient time for different coupling strengths.

    The transient time is defined as the first time the sync error drops
    below `threshold` and stays below for at least 100 consecutive steps.

    Args:
        eps_values: Coupling strengths to test (default: 6 values above eps_c).
        n_steps: Maximum integration steps.
        dt: Timestep.
        threshold: Error threshold for synchronization.

    Returns:
        Dict with 'eps' and 'transient_time' arrays.
    """
    if eps_values is None:
        eps_values = np.array([2.0, 3.0, 4.0, 5.0, 7.0, 10.0])

    transient_times = np.empty(len(eps_values))
    confirm_steps = 100

    for i, eps_val in enumerate(eps_values):
        config = SimulationConfig(
            domain=Domain.COUPLED_LORENZ,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0,
                "eps": eps_val,
                "x1_0": 1.0, "y1_0": 1.0, "z1_0": 1.0,
                "x2_0": -5.0, "y2_0": 5.0, "z2_0": 25.0,
            },
        )
        sim = CoupledLorenzSimulation(config)
        sim.reset()

        t_sync = np.inf
        consecutive_below = 0

        for step in range(n_steps):
            sim.step()
            err = sim.sync_error()
            if err < threshold:
                consecutive_below += 1
                if consecutive_below >= confirm_steps:
                    # Synchronization achieved
                    t_sync = (step - confirm_steps + 1) * dt
                    break
            else:
                consecutive_below = 0

        transient_times[i] = t_sync
        logger.info(
            f"  eps={eps_val:.2f}: transient_time={t_sync:.2f}"
        )

    return {
        "eps": eps_values,
        "transient_time": transient_times,
    }


def estimate_critical_coupling(
    sweep_data: dict[str, np.ndarray],
    error_threshold: float = 1.0,
) -> float:
    """Estimate critical coupling eps_c from a sync sweep.

    eps_c is the smallest eps where the mean sync error drops below
    the given threshold. Uses linear interpolation between sweep points.

    Args:
        sweep_data: Output of generate_sync_sweep_data().
        error_threshold: Threshold for "synchronized" state.

    Returns:
        Estimated eps_c.
    """
    eps = sweep_data["eps"]
    errors = sweep_data["mean_error"]

    # Find first crossing below threshold
    for j in range(len(errors) - 1):
        if errors[j] >= error_threshold and errors[j + 1] < error_threshold:
            # Linear interpolation
            frac = (error_threshold - errors[j]) / (errors[j + 1] - errors[j])
            return float(eps[j] + frac * (eps[j + 1] - eps[j]))

    # If all above threshold, return max eps
    if np.all(errors >= error_threshold):
        return float(eps[-1])
    # If all below threshold, return min eps
    return float(eps[0])


def run_coupled_lorenz_rediscovery(
    output_dir: str | Path = "output/rediscovery/coupled_lorenz",
    n_iterations: int = 40,
) -> dict:
    """Run the full coupled Lorenz synchronization rediscovery.

    1. Sweep eps from 0 to 10 and measure steady-state sync error
    2. Identify critical coupling eps_c
    3. Measure synchronization transient time vs eps
    4. Compute conditional Lyapunov exponents

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "coupled_lorenz",
        "targets": {
            "phenomenon": "Chaos synchronization via diffusive coupling",
            "critical_coupling": "eps_c ~ 1-2 for sigma=10, rho=28, beta=8/3",
            "sync_error_decay": "||e(t)|| -> 0 exponentially for eps > eps_c",
            "conditional_lyapunov": "lambda_cond < 0 for eps > eps_c",
        },
    }

    # --- Part 1: Synchronization sweep ---
    logger.info("Part 1: Sweeping coupling strength eps...")
    sweep_data = generate_sync_sweep_data(
        n_eps=25, eps_min=0.0, eps_max=10.0,
        n_steps=10000, n_transient=5000, dt=0.005,
    )

    eps_c = estimate_critical_coupling(sweep_data, error_threshold=1.0)
    logger.info(f"  Estimated eps_c = {eps_c:.3f}")

    results["sync_sweep"] = {
        "n_eps_values": len(sweep_data["eps"]),
        "eps_range": [float(sweep_data["eps"][0]), float(sweep_data["eps"][-1])],
        "min_error": float(np.min(sweep_data["mean_error"])),
        "max_error": float(np.max(sweep_data["mean_error"])),
        "eps_c_estimate": eps_c,
    }

    # Check monotonicity in the high-eps regime
    high_mask = sweep_data["eps"] > eps_c
    if np.sum(high_mask) > 1:
        high_errors = sweep_data["mean_error"][high_mask]
        results["sync_sweep"]["high_eps_max_error"] = float(np.max(high_errors))
        results["sync_sweep"]["high_eps_mean_error"] = float(np.mean(high_errors))

    # --- Part 2: Transient time measurement ---
    logger.info("Part 2: Measuring synchronization transient times...")
    transient_data = generate_transient_data(
        eps_values=np.array([2.0, 3.0, 4.0, 5.0, 7.0, 10.0]),
        n_steps=20000, dt=0.005, threshold=0.1,
    )

    results["transient_times"] = {
        "eps": transient_data["eps"].tolist(),
        "times": transient_data["transient_time"].tolist(),
        "n_synchronized": int(
            np.sum(np.isfinite(transient_data["transient_time"]))
        ),
    }

    # --- Part 3: Conditional Lyapunov exponents ---
    logger.info("Part 3: Computing conditional Lyapunov exponents...")
    lyap_eps_values = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    lyap_values = np.empty(len(lyap_eps_values))

    config_base = SimulationConfig(
        domain=Domain.COUPLED_LORENZ,
        dt=0.005,
        n_steps=30000,
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0, "eps": 5.0},
    )
    sim_lyap = CoupledLorenzSimulation(config_base)
    sim_lyap.reset()

    for i, eps_val in enumerate(lyap_eps_values):
        lam = sim_lyap.conditional_lyapunov(
            eps=eps_val, n_steps=30000, n_transient=5000,
        )
        lyap_values[i] = lam
        logger.info(f"  eps={eps_val:.1f}: cond_Lyapunov={lam:.4f}")

    results["conditional_lyapunov"] = {
        "eps": lyap_eps_values.tolist(),
        "lyapunov": lyap_values.tolist(),
        "n_negative": int(np.sum(lyap_values < 0)),
        "n_positive": int(np.sum(lyap_values > 0)),
    }

    # Find zero crossing of conditional Lyapunov (another eps_c estimate)
    for j in range(len(lyap_values) - 1):
        if lyap_values[j] > 0 and lyap_values[j + 1] <= 0:
            frac = -lyap_values[j] / (lyap_values[j + 1] - lyap_values[j])
            eps_c_lyap = lyap_eps_values[j] + frac * (
                lyap_eps_values[j + 1] - lyap_eps_values[j]
            )
            results["conditional_lyapunov"]["eps_c_from_lyapunov"] = float(
                eps_c_lyap
            )
            logger.info(f"  eps_c from conditional Lyapunov: {eps_c_lyap:.3f}")
            break

    # --- Part 4: Verify synchronization at high coupling ---
    logger.info("Part 4: Verifying synchronization at eps=5.0...")
    config_sync = SimulationConfig(
        domain=Domain.COUPLED_LORENZ,
        dt=0.005,
        n_steps=15000,
        parameters={
            "sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0,
            "eps": 5.0,
            "x1_0": 1.0, "y1_0": 1.0, "z1_0": 1.0,
            "x2_0": -5.0, "y2_0": 5.0, "z2_0": 25.0,
        },
    )
    sim_verify = CoupledLorenzSimulation(config_sync)
    sim_verify.reset()

    # Run and measure final error
    for _ in range(15000):
        sim_verify.step()
    final_error = sim_verify.sync_error()
    logger.info(f"  Final sync error at eps=5.0: {final_error:.6e}")

    results["verification"] = {
        "eps": 5.0,
        "final_sync_error": float(final_error),
        "synchronized": bool(final_error < 1e-4),
    }

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save sweep data
    np.savez(
        output_path / "sync_sweep.npz",
        eps=sweep_data["eps"],
        mean_error=sweep_data["mean_error"],
    )
    np.savez(
        output_path / "transient_data.npz",
        eps=transient_data["eps"],
        transient_time=transient_data["transient_time"],
    )
    np.savez(
        output_path / "lyapunov_data.npz",
        eps=lyap_eps_values,
        lyapunov=lyap_values,
    )

    return results
