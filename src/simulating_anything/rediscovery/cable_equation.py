"""Cable equation (passive neurite) rediscovery.

Targets:
- Space constant: V(x) ~ exp(-|x|/lambda)
- Time constant: tau_m = R_m * C_m
- Space constant tracks lambda_e parameter (PySR: lambda_meas = f(lambda_e))
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.cable_equation import CableEquationSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_space_constant_data(
    n_lambda: int = 20,
    n_steps: int = 10000,
    dt: float = 0.01,
    N: int = 100,
) -> dict[str, np.ndarray]:
    """Sweep lambda_e and measure space constant from steady-state profile.

    Args:
        n_lambda: number of lambda_e values to test.
        n_steps: steps to reach steady state for each value.
        dt: timestep in ms.
        N: number of grid points.

    Returns:
        Dict with lambda_set, lambda_measured, V_peak arrays.
    """
    lambda_values = np.linspace(0.2, 2.0, n_lambda)

    all_lambda_set = []
    all_lambda_meas = []
    all_V_peak = []

    for i, lam in enumerate(lambda_values):
        config = SimulationConfig(
            domain=Domain.CABLE_EQUATION,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "tau_m": 10.0, "lambda_e": lam, "L": 10.0,
                "N": float(N), "R_m": 1.0, "I0": 1.0,
                "inject_x": 0.5,
            },
        )
        sim = CableEquationSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        lam_meas = sim.measure_space_constant()
        V_peak = float(sim.observe()[sim._inject_idx])

        all_lambda_set.append(lam)
        all_lambda_meas.append(lam_meas)
        all_V_peak.append(V_peak)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  lambda_e={lam:.3f}: measured={lam_meas:.4f}, "
                f"V_peak={V_peak:.4f}"
            )

    return {
        "lambda_set": np.array(all_lambda_set),
        "lambda_measured": np.array(all_lambda_meas),
        "V_peak": np.array(all_V_peak),
    }


def run_cable_equation_rediscovery(
    output_dir: str | Path = "output/rediscovery/cable_equation",
    n_iterations: int = 40,
) -> dict:
    """Run cable equation rediscovery.

    Demonstrates:
    1. Steady-state exponential decay V ~ exp(-|x|/lambda)
    2. Measured space constant matches lambda_e parameter
    3. Time constant measurement
    4. PySR symbolic regression on lambda_measured vs lambda_set

    Args:
        output_dir: directory for output files.
        n_iterations: PySR iteration count.

    Returns:
        Results dict with all measurements and discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "cable_equation",
        "targets": {
            "space_constant": "V(x) ~ exp(-|x-x0| / lambda)",
            "time_constant": "tau_m = R_m * C_m",
            "space_constant_identity": "lambda_measured = lambda_e",
        },
    }

    # 1. Generate space constant sweep data
    logger.info("Generating space constant sweep data...")
    data = generate_space_constant_data(
        n_lambda=20, n_steps=10000, dt=0.01, N=100
    )

    valid = (data["lambda_measured"] > 0.01) & np.isfinite(data["lambda_measured"])

    if np.sum(valid) > 3:
        rel_err = np.abs(
            data["lambda_measured"][valid] - data["lambda_set"][valid]
        ) / data["lambda_set"][valid]
        corr = float(np.corrcoef(
            data["lambda_set"][valid],
            data["lambda_measured"][valid],
        )[0, 1])
        results["space_constant_data"] = {
            "n_samples": int(np.sum(valid)),
            "mean_relative_error": float(np.mean(rel_err)),
            "correlation": corr,
        }
        logger.info(
            f"  Space constant: {np.sum(valid)} valid, "
            f"mean rel error: {np.mean(rel_err):.4%}, "
            f"correlation: {corr:.6f}"
        )

    # 2. Measure time constant
    logger.info("Measuring time constant...")
    config_tau = SimulationConfig(
        domain=Domain.CABLE_EQUATION,
        dt=0.01,
        n_steps=10000,
        parameters={
            "tau_m": 10.0, "lambda_e": 0.5, "L": 5.0,
            "N": 100.0, "R_m": 1.0, "I0": 1.0,
        },
    )
    sim_tau = CableEquationSimulation(config_tau)
    tau_meas = sim_tau.measure_time_constant(n_steps=10000)
    results["time_constant"] = {
        "tau_m_set": 10.0,
        "tau_m_measured": tau_meas,
        "relative_error": abs(tau_meas - 10.0) / 10.0 if tau_meas > 0 else None,
    }
    logger.info(
        f"  tau_m set=10.0, measured={tau_meas:.4f}, "
        f"error={abs(tau_meas - 10.0) / 10.0:.4%}" if tau_meas > 0
        else "  tau_m measurement failed"
    )

    # 3. Verify exponential spatial profile
    logger.info("Verifying exponential spatial profile...")
    config_profile = SimulationConfig(
        domain=Domain.CABLE_EQUATION,
        dt=0.01,
        n_steps=20000,
        parameters={
            "tau_m": 10.0, "lambda_e": 0.5, "L": 10.0,
            "N": 200.0, "R_m": 1.0, "I0": 1.0,
            "inject_x": 0.5,
        },
    )
    sim_profile = CableEquationSimulation(config_profile)
    sim_profile.reset()
    for _ in range(20000):
        sim_profile.step()

    V_ss = sim_profile.observe()
    V_analytical = sim_profile.steady_state_analytical(sim_profile.x, 1.0)
    inject_idx = sim_profile._inject_idx

    # Compare shape (not amplitude, due to finite cable effects)
    # Normalize both to peak
    V_norm = V_ss / V_ss[inject_idx] if V_ss[inject_idx] > 0 else V_ss
    V_ana_norm = (V_analytical / V_analytical[inject_idx]
                  if V_analytical[inject_idx] > 0 else V_analytical)

    # Use interior region away from boundaries
    margin = sim_profile.N // 10
    interior = slice(margin, sim_profile.N - margin)
    if V_ss[inject_idx] > 0 and V_analytical[inject_idx] > 0:
        profile_corr = float(np.corrcoef(
            V_norm[interior], V_ana_norm[interior]
        )[0, 1])
    else:
        profile_corr = 0.0

    results["spatial_profile"] = {
        "peak_voltage": float(V_ss[inject_idx]),
        "analytical_peak": float(V_analytical[inject_idx]),
        "profile_correlation": profile_corr,
    }
    logger.info(f"  Profile correlation with analytical: {profile_corr:.6f}")

    # 4. PySR: lambda_measured = f(lambda_set)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = data["lambda_set"][valid].reshape(-1, 1)
        y = data["lambda_measured"][valid]

        logger.info("Running PySR: lambda_measured = f(lambda_set)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["lam"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt"],
            max_complexity=6,
            populations=15,
            population_size=30,
        )
        results["space_constant_pysr"] = {
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
            results["space_constant_pysr"]["best"] = best.expression
            results["space_constant_pysr"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["space_constant_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
