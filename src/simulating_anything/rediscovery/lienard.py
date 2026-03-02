"""Lienard oscillator rediscovery.

Targets:
- ODE: x'' + mu*(x^2-1)*x' + x + alpha*x^3 = 0  (via SINDy)
- Limit cycle amplitude dependence on mu and alpha
- Period dependence on mu and alpha
- Energy: E = v^2/2 + x^2/2 + alpha*x^4/4
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.lienard import LienardSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    mu: float = 1.0,
    alpha: float = 0.1,
    n_steps: int = 10000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Args:
        mu: Nonlinear damping parameter.
        alpha: Cubic stiffness coefficient.
        n_steps: Number of integration steps.
        dt: Timestep.

    Returns:
        Dict with time, states, x, v, and parameter values.
    """
    config = SimulationConfig(
        domain=Domain.LIENARD,
        dt=dt,
        n_steps=n_steps,
        parameters={"mu": mu, "alpha": alpha, "x_0": 2.0, "v_0": 0.0},
    )
    sim = LienardSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    times = np.arange(n_steps + 1) * dt

    return {
        "time": times,
        "states": states,
        "x": states[:, 0],
        "v": states[:, 1],
        "mu": mu,
        "alpha": alpha,
        "dt": dt,
    }


def generate_mu_sweep(
    n_mu: int = 20,
    alpha: float = 0.1,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep mu and measure limit cycle amplitude and period.

    Args:
        n_mu: Number of mu values to sweep.
        alpha: Fixed cubic stiffness coefficient.
        dt: Timestep for integration.

    Returns:
        Dict with mu values, measured periods, and amplitudes.
    """
    mu_values = np.linspace(0.5, 5.0, n_mu)
    periods = []
    amplitudes = []

    for i, mu in enumerate(mu_values):
        config = SimulationConfig(
            domain=Domain.LIENARD,
            dt=dt,
            n_steps=1000,
            parameters={"mu": mu, "alpha": alpha, "x_0": 2.0, "v_0": 0.0},
        )
        sim = LienardSimulation(config)
        sim.reset()

        T = sim.measure_period(n_periods=5)
        periods.append(T)

        # Measure amplitude separately (fresh reset)
        sim.reset()
        A = sim.measure_amplitude(n_periods=3)
        amplitudes.append(A)

        if (i + 1) % 5 == 0:
            logger.info(f"  mu={mu:.3f}: T={T:.4f}, A={A:.4f}")

    return {
        "mu": mu_values,
        "period": np.array(periods),
        "amplitude": np.array(amplitudes),
        "alpha": alpha,
    }


def generate_alpha_sweep(
    n_alpha: int = 15,
    mu: float = 1.0,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep alpha (cubic stiffness) and measure period and amplitude.

    Args:
        n_alpha: Number of alpha values to sweep.
        mu: Fixed damping parameter.
        dt: Timestep for integration.

    Returns:
        Dict with alpha values, measured periods, and amplitudes.
    """
    alpha_values = np.linspace(0.0, 1.0, n_alpha)
    periods = []
    amplitudes = []

    for i, alpha in enumerate(alpha_values):
        config = SimulationConfig(
            domain=Domain.LIENARD,
            dt=dt,
            n_steps=1000,
            parameters={"mu": mu, "alpha": alpha, "x_0": 2.0, "v_0": 0.0},
        )
        sim = LienardSimulation(config)
        sim.reset()

        T = sim.measure_period(n_periods=5)
        periods.append(T)

        sim.reset()
        A = sim.measure_amplitude(n_periods=3)
        amplitudes.append(A)

        if (i + 1) % 5 == 0:
            logger.info(f"  alpha={alpha:.3f}: T={T:.4f}, A={A:.4f}")

    return {
        "alpha": alpha_values,
        "period": np.array(periods),
        "amplitude": np.array(amplitudes),
        "mu": mu,
    }


def generate_energy_check(
    mu: float = 0.0,
    alpha: float = 0.1,
    n_steps: int = 10000,
    dt: float = 0.001,
) -> dict[str, np.ndarray | float]:
    """Check energy conservation for mu=0 (conservative limit).

    When mu=0, the damping vanishes and E = v^2/2 + x^2/2 + alpha*x^4/4
    should be exactly conserved (up to RK4 numerical drift).

    Args:
        mu: Damping parameter (0 for conservation check).
        alpha: Cubic stiffness coefficient.
        n_steps: Number of integration steps.
        dt: Timestep.

    Returns:
        Dict with energy trajectory and drift statistics.
    """
    config = SimulationConfig(
        domain=Domain.LIENARD,
        dt=dt,
        n_steps=n_steps,
        parameters={"mu": mu, "alpha": alpha, "x_0": 2.0, "v_0": 0.0},
    )
    sim = LienardSimulation(config)
    sim.reset()

    energies = [sim.compute_energy()]
    for _ in range(n_steps):
        sim.step()
        energies.append(sim.compute_energy())

    energies = np.array(energies)
    E0 = energies[0]
    drift = np.abs(energies - E0) / max(abs(E0), 1e-15)

    return {
        "energies": energies,
        "E0": float(E0),
        "E_final": float(energies[-1]),
        "max_relative_drift": float(np.max(drift)),
        "mean_relative_drift": float(np.mean(drift)),
        "mu": mu,
        "alpha": alpha,
    }


def run_lienard_rediscovery(
    output_dir: str | Path = "output/rediscovery/lienard",
    n_iterations: int = 40,
) -> dict:
    """Run Lienard oscillator rediscovery pipeline.

    1. Generate trajectory for SINDy ODE recovery
    2. Sweep mu to measure period and amplitude dependence
    3. Sweep alpha to measure restoring force effects on period
    4. Verify energy conservation in conservative limit (mu=0)
    5. Run PySR to find T(mu) and A(alpha)

    Args:
        output_dir: Directory to save results.
        n_iterations: Number of PySR iterations.

    Returns:
        Results dict with all discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "lienard",
        "targets": {
            "ode": "x'' + mu*(x^2-1)*x' + x + alpha*x^3 = 0",
            "limit_cycle": "Stable limit cycle for mu > 0 (Lienard theorem)",
            "energy": "E = v^2/2 + x^2/2 + alpha*x^4/4 (conserved when mu=0)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery (mu=1.0, alpha=0.1)...")
    data = generate_ode_data(mu=1.0, alpha=0.1, n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=data["dt"],
            feature_names=["x", "v"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries[:5]
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
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: mu sweep ---
    logger.info("Part 2: Period and amplitude vs mu (alpha=0.1)...")
    mu_data = generate_mu_sweep(n_mu=20, alpha=0.1, dt=0.005)

    valid_mu = np.isfinite(mu_data["period"]) & (mu_data["period"] > 0)
    results["mu_sweep"] = {
        "n_samples": int(np.sum(valid_mu)),
        "mu_range": [float(mu_data["mu"].min()), float(mu_data["mu"].max())],
        "period_range": [
            float(np.min(mu_data["period"][valid_mu])),
            float(np.max(mu_data["period"][valid_mu])),
        ],
        "mean_amplitude": float(np.mean(mu_data["amplitude"][valid_mu])),
    }
    logger.info(f"  Mean amplitude: {results['mu_sweep']['mean_amplitude']:.4f}")

    # PySR: find T(mu)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = mu_data["mu"][valid_mu].reshape(-1, 1)
        y = mu_data["period"][valid_mu]

        logger.info("  Running PySR: T = f(mu_)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["mu_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["log", "sqrt"],
            max_complexity=12,
            populations=20,
            population_size=40,
        )
        results["period_mu_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["period_mu_pysr"]["best"] = best.expression
            results["period_mu_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR period(mu) failed: {e}")
        results["period_mu_pysr"] = {"error": str(e)}

    # --- Part 3: alpha sweep ---
    logger.info("Part 3: Period and amplitude vs alpha (mu=1.0)...")
    alpha_data = generate_alpha_sweep(n_alpha=15, mu=1.0, dt=0.005)

    valid_alpha = np.isfinite(alpha_data["period"]) & (alpha_data["period"] > 0)
    results["alpha_sweep"] = {
        "n_samples": int(np.sum(valid_alpha)),
        "alpha_range": [
            float(alpha_data["alpha"].min()),
            float(alpha_data["alpha"].max()),
        ],
        "period_range": [
            float(np.min(alpha_data["period"][valid_alpha])),
            float(np.max(alpha_data["period"][valid_alpha])),
        ],
        "amplitude_range": [
            float(np.min(alpha_data["amplitude"][valid_alpha])),
            float(np.max(alpha_data["amplitude"][valid_alpha])),
        ],
    }

    # PySR: find T(alpha)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = alpha_data["alpha"][valid_alpha].reshape(-1, 1)
        y = alpha_data["period"][valid_alpha]

        logger.info("  Running PySR: T = f(alpha_)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["alpha_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=10,
            populations=15,
            population_size=30,
        )
        results["period_alpha_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["period_alpha_pysr"]["best"] = best.expression
            results["period_alpha_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR period(alpha) failed: {e}")
        results["period_alpha_pysr"] = {"error": str(e)}

    # --- Part 4: Energy conservation check ---
    logger.info("Part 4: Energy conservation check (mu=0)...")
    energy_data = generate_energy_check(mu=0.0, alpha=0.1, n_steps=10000, dt=0.001)

    results["energy_conservation"] = {
        "E0": energy_data["E0"],
        "E_final": energy_data["E_final"],
        "max_relative_drift": energy_data["max_relative_drift"],
        "mean_relative_drift": energy_data["mean_relative_drift"],
        "conserved": energy_data["max_relative_drift"] < 1e-6,
    }
    logger.info(
        f"  E0={energy_data['E0']:.6f}, E_final={energy_data['E_final']:.6f}, "
        f"max drift={energy_data['max_relative_drift']:.2e}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "mu_sweep.npz",
        mu=mu_data["mu"],
        period=mu_data["period"],
        amplitude=mu_data["amplitude"],
    )
    np.savez(
        output_path / "alpha_sweep.npz",
        alpha=alpha_data["alpha"],
        period=alpha_data["period"],
        amplitude=alpha_data["amplitude"],
    )

    return results
