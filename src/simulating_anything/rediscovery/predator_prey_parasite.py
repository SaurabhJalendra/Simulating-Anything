"""Predator-prey-parasite model rediscovery.

Targets:
- ODE recovery via SINDy: 3-variable predator-prey-parasite system
- Equilibrium classification: prey-only, prey-predator, prey-parasite, coexistence
- Lambda sweep: effect of infection rate on prey regulation
- Biological control analysis: parasite reduces prey peaks
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.predator_prey_parasite import (
    PredatorPreyParasiteSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.PREDATOR_PREY_PARASITE


def generate_ode_data(
    n_steps: int = 20000,
    dt: float = 0.01,
    **param_overrides: float,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Args:
        n_steps: Number of integration steps.
        dt: Timestep.
        **param_overrides: Override default parameters.

    Returns:
        Dict with time, states, N, P, Z arrays.
    """
    params: dict[str, float] = {
        "r": 1.0, "K": 10.0, "a": 0.1, "e": 0.5, "d": 0.2,
        "lambda_": 0.15, "mu": 0.3, "alpha": 0.05,
        "N_0": 5.0, "P_0": 2.0, "Z_0": 1.0,
    }
    params.update(param_overrides)

    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )
    sim = PredatorPreyParasiteSimulation(config)
    sim.reset(seed=42)

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "N": states_arr[:, 0],
        "P": states_arr[:, 1],
        "Z": states_arr[:, 2],
    }


def generate_lambda_sweep(
    n_lambda: int = 25,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep infection rate lambda to measure parasite effect on prey.

    Args:
        n_lambda: Number of lambda values.
        n_steps: Steps per simulation.
        dt: Timestep.

    Returns:
        Dict with lambda_values, N_avg, P_avg, Z_avg arrays.
    """
    lambda_values = np.linspace(0.0, 0.5, n_lambda)
    N_arr = []
    P_arr = []
    Z_arr = []

    for i, lam_val in enumerate(lambda_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 1.0, "K": 10.0, "a": 0.1, "e": 0.5, "d": 0.2,
                "lambda_": lam_val, "mu": 0.3, "alpha": 0.05,
                "N_0": 5.0, "P_0": 2.0, "Z_0": 1.0,
            },
        )
        sim = PredatorPreyParasiteSimulation(config)
        sim.reset(seed=42)

        # Run to steady state
        for _ in range(n_steps):
            sim.step()

        # Average over last 10%
        tail_steps = max(1, int(n_steps * 0.1))
        tail_states = []
        for _ in range(tail_steps):
            sim.step()
            tail_states.append(sim.observe().copy())

        tail_arr = np.array(tail_states)
        N_arr.append(np.mean(tail_arr[:, 0]))
        P_arr.append(np.mean(tail_arr[:, 1]))
        Z_arr.append(np.mean(tail_arr[:, 2]))

        if (i + 1) % 5 == 0:
            logger.info(
                f"  lambda={lam_val:.3f}: N={N_arr[-1]:.3f}, "
                f"P={P_arr[-1]:.3f}, Z={Z_arr[-1]:.3f}"
            )

    return {
        "lambda_values": lambda_values,
        "N_avg": np.array(N_arr),
        "P_avg": np.array(P_arr),
        "Z_avg": np.array(Z_arr),
    }


def classify_long_time_behavior(
    states: np.ndarray,
    threshold_std: float = 0.01,
) -> str:
    """Classify long-time behavior as fixed point, limit cycle, or irregular.

    Args:
        states: Array of shape (n_steps, 3) from post-transient regime.
        threshold_std: Below this normalized std, consider fixed point.

    Returns:
        One of 'fixed_point', 'limit_cycle', 'irregular'.
    """
    means = np.mean(states, axis=0)
    stds = np.std(states, axis=0)

    # Normalized variation
    norm_std = stds / (np.abs(means) + 1e-10)

    if np.all(norm_std < threshold_std):
        return "fixed_point"

    # Check for periodicity using autocorrelation on first variable
    x = states[:, 0] - means[0]
    if len(x) > 100:
        # Compute autocorrelation
        n = len(x)
        acf = np.correlate(x, x, mode="full")[n - 1:]
        acf = acf / (acf[0] + 1e-10)

        # Look for secondary peak (period detection)
        # Skip first 10 points to avoid main peak
        skip = min(10, n // 10)
        if len(acf) > skip + 10:
            secondary = acf[skip:]
            peaks = np.where(
                (secondary[1:-1] > secondary[:-2])
                & (secondary[1:-1] > secondary[2:])
            )[0] + 1
            if len(peaks) > 0 and secondary[peaks[0]] > 0.3:
                return "limit_cycle"

    return "irregular"


def run_predator_prey_parasite_rediscovery(
    output_dir: str | Path = "output/rediscovery/predator_prey_parasite",
    n_iterations: int = 40,
) -> dict:
    """Run predator-prey-parasite rediscovery pipeline.

    1. Generate trajectory for SINDy ODE recovery
    2. Classify long-time behavior
    3. Sweep lambda to study parasite effect
    4. Verify equilibrium computation
    5. Biological control analysis
    6. SINDy ODE recovery

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iterations.

    Returns:
        Results dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "predator_prey_parasite",
        "targets": {
            "ode_N": "dN/dt = r*N*(1-N/K) - a*N*P - lambda*N*Z",
            "ode_P": "dP/dt = e*a*N*P - d*P",
            "ode_Z": "dZ/dt = lambda*N*Z - mu*Z - alpha*Z*P",
            "equilibria": "Multiple equilibria classification",
            "biological_control": "Parasite reduces prey peak",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery...")
    data = generate_ode_data(n_steps=20000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["N", "P", "Z"],
            threshold=0.05,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries[:6]
            ],
        }
        if sindy_discoveries:
            best = sindy_discoveries[0]
            results["sindy_ode"]["best"] = best.expression
            results["sindy_ode"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  SINDy best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: Classify long-time behavior ---
    logger.info("Part 2: Classify long-time behavior...")
    # Use post-transient data (last 50%)
    n_total = data["states"].shape[0]
    post_transient = data["states"][n_total // 2:]
    behavior = classify_long_time_behavior(post_transient)
    results["behavior"] = {
        "classification": behavior,
        "n_points_analyzed": len(post_transient),
    }
    logger.info(f"  Long-time behavior: {behavior}")

    # --- Part 3: Lambda sweep ---
    logger.info("Part 3: Lambda sweep (infection rate effect)...")
    lambda_data = generate_lambda_sweep(n_lambda=20, n_steps=20000, dt=0.01)

    # Check if parasite emerges at high lambda
    Z_above_threshold = lambda_data["Z_avg"] > 0.01
    n_parasite_present = int(np.sum(Z_above_threshold))
    results["lambda_sweep"] = {
        "n_lambda_values": len(lambda_data["lambda_values"]),
        "n_parasite_present": n_parasite_present,
        "max_Z": float(np.max(lambda_data["Z_avg"])),
        "N_range": [
            float(np.min(lambda_data["N_avg"])),
            float(np.max(lambda_data["N_avg"])),
        ],
    }
    logger.info(
        f"  Parasite present in {n_parasite_present}/{len(lambda_data['lambda_values'])} "
        f"parameter values"
    )

    # --- Part 4: Equilibrium verification ---
    logger.info("Part 4: Equilibrium verification...")
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=0.01,
        n_steps=1000,
        parameters={
            "r": 1.0, "K": 10.0, "a": 0.1, "e": 0.5, "d": 0.2,
            "lambda_": 0.15, "mu": 0.3, "alpha": 0.05,
            "N_0": 5.0, "P_0": 2.0, "Z_0": 1.0,
        },
    )
    sim = PredatorPreyParasiteSimulation(config)
    equilibria = sim.compute_equilibria()
    eq_results = {}
    for name, eq in equilibria.items():
        if eq is not None:
            # Verify: derivatives should be near zero at equilibrium
            deriv = sim._derivatives(eq)
            max_deriv = float(np.max(np.abs(deriv)))
            eq_results[name] = {
                "state": eq.tolist(),
                "max_derivative": max_deriv,
                "is_equilibrium": max_deriv < 1e-8,
            }
        else:
            eq_results[name] = None
    results["equilibria"] = eq_results
    n_eq = sum(1 for v in eq_results.values() if v is not None)
    logger.info(f"  Found {n_eq} equilibria")

    # --- Part 5: Biological control analysis ---
    logger.info("Part 5: Biological control analysis...")
    # Compare prey dynamics with and without parasite
    data_no_parasite = generate_ode_data(
        n_steps=20000, dt=0.01, lambda_=0.0,
    )
    data_with_parasite = generate_ode_data(
        n_steps=20000, dt=0.01, lambda_=0.15,
    )

    # Compare prey peak and mean
    N_no_para = data_no_parasite["N"]
    N_with_para = data_with_parasite["N"]

    # Skip transient (first 20%)
    skip = len(N_no_para) // 5
    N_peak_no = float(np.max(N_no_para[skip:]))
    N_peak_with = float(np.max(N_with_para[skip:]))
    N_mean_no = float(np.mean(N_no_para[skip:]))
    N_mean_with = float(np.mean(N_with_para[skip:]))

    results["biological_control"] = {
        "prey_peak_without_parasite": N_peak_no,
        "prey_peak_with_parasite": N_peak_with,
        "prey_mean_without_parasite": N_mean_no,
        "prey_mean_with_parasite": N_mean_with,
        "peak_reduction_pct": float(
            (N_peak_no - N_peak_with) / (N_peak_no + 1e-10) * 100
        ),
        "mean_reduction_pct": float(
            (N_mean_no - N_mean_with) / (N_mean_no + 1e-10) * 100
        ),
    }
    logger.info(
        f"  Prey peak reduction: {results['biological_control']['peak_reduction_pct']:.1f}%"
    )

    # --- Part 6: PySR for prey dynamics ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use lambda sweep data: N_avg = f(lambda)
        valid = lambda_data["lambda_values"] > 0.01
        if np.sum(valid) > 5:
            X = lambda_data["lambda_values"][valid].reshape(-1, 1)
            y = lambda_data["N_avg"][valid]

            logger.info("  Running PySR: N_avg = f(lambda)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["lam_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["prey_pysr"] = {
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
                results["prey_pysr"]["best"] = best.expression
                results["prey_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as exc:
        logger.warning(f"PySR failed: {exc}")
        results["prey_pysr"] = {"error": str(exc)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
