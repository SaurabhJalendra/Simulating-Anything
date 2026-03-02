"""Gauss map rediscovery.

Targets:
- Period-doubling bifurcation diagram across beta
- Lyapunov exponent as function of beta
- Fixed point structure and stability
- Chaos onset detection
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.gauss_map import GaussMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    alpha: float = 6.2,
    beta: float = -0.5,
    x_0: float = 0.5,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.GAUSS_MAP,  # Placeholder until GAUSS_MAP enum is added
        dt=1.0,
        n_steps=1000,
        parameters={"alpha": alpha, "beta": beta, "x_0": x_0},
    )


def generate_bifurcation_data(
    n_beta: int = 500,
    beta_min: float = -1.0,
    beta_max: float = 1.0,
    alpha: float = 6.2,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram and period detection data across beta."""
    beta_values = np.linspace(beta_min, beta_max, n_beta)

    # Bifurcation diagram points
    sim = GaussMapSimulation(_make_config(alpha=alpha, beta=-0.5))
    bif_data = sim.bifurcation_data(
        param_name="beta",
        param_range=(beta_min, beta_max),
        n_params=n_beta,
        n_transient=500,
        n_plot=50,
    )

    # Period detection
    periods = []
    for beta in beta_values:
        config = _make_config(alpha=alpha, beta=beta)
        sim_b = GaussMapSimulation(config)
        sim_b.reset()
        p = sim_b.detect_period(max_period=64)
        periods.append(p)

    return {
        "beta_values": beta_values,
        "periods": np.array(periods),
        "bif_param": bif_data["param"],
        "bif_x": bif_data["x"],
    }


def generate_lyapunov_data(
    n_beta: int = 100,
    beta_min: float = -1.0,
    beta_max: float = 1.0,
    alpha: float = 6.2,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs beta data."""
    beta_values = np.linspace(beta_min, beta_max, n_beta)
    lyapunovs = []

    for i, beta in enumerate(beta_values):
        config = _make_config(alpha=alpha, beta=beta)
        sim = GaussMapSimulation(config)
        lam = sim.compute_lyapunov(n_transient=1000, n_compute=5000)
        lyapunovs.append(lam)

        if (i + 1) % 25 == 0:
            logger.info(f"  beta={beta:.4f}: Lyapunov={lam:.4f}")

    return {
        "beta": beta_values,
        "lyapunov": np.array(lyapunovs),
    }


def estimate_feigenbaum(
    periods: np.ndarray,
    beta_values: np.ndarray,
) -> dict:
    """Estimate Feigenbaum constant from period-doubling bifurcation points."""
    bif_points = []
    target_periods = [2, 4, 8, 16]

    for p in target_periods:
        idx = np.where(periods >= p)[0]
        if len(idx) > 0:
            bif_points.append(float(beta_values[idx[0]]))

    result: dict = {"bifurcation_points": bif_points}

    if len(bif_points) >= 3:
        deltas = []
        for i in range(len(bif_points) - 2):
            d1 = bif_points[i + 1] - bif_points[i]
            d2 = bif_points[i + 2] - bif_points[i + 1]
            if abs(d2) > 1e-10:
                deltas.append(abs(d1 / d2))

        if deltas:
            result["feigenbaum_estimates"] = deltas
            result["best_estimate"] = deltas[-1] if deltas else None
            logger.info(f"  Feigenbaum estimates: {deltas}")

    return result


def run_gauss_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/gauss_map",
    n_iterations: int = 40,
) -> dict:
    """Run Gauss map rediscovery.

    Discovers:
    - Bifurcation diagram and period-doubling cascade
    - Chaos onset in the beta parameter space
    - Lyapunov exponent spectrum
    - Fixed point structure and stability
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "gauss_map",
        "targets": {
            "map": "x_{n+1} = exp(-alpha * x_n^2) + beta",
            "derivative": "f'(x) = -2*alpha*x*exp(-alpha*x^2)",
            "chaos": "period-doubling route to chaos as beta varies",
            "feigenbaum": "delta ~ 4.669",
        },
    }

    # Part 1: Fixed point analysis
    logger.info("Part 1: Fixed point analysis...")
    sim_fp = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5))
    fps = sim_fp.find_fixed_points()
    results["fixed_points"] = fps
    for fp in fps:
        logger.info(
            f"  x*={fp['x']:.6f}, eigenvalue={fp['eigenvalue']:.6f}, "
            f"stability={fp['stability']}"
        )

    # Verify convergence to stable fixed point
    if any(fp["stability"] == "stable" for fp in fps):
        stable_fp = next(fp for fp in fps if fp["stability"] == "stable")
        sim_conv = GaussMapSimulation(_make_config(alpha=6.2, beta=-0.5, x_0=0.3))
        sim_conv.reset()
        for _ in range(1000):
            sim_conv.step()
        converged_x = float(sim_conv.observe()[0])
        results["convergence_test"] = {
            "alpha": 6.2,
            "beta": -0.5,
            "final_x": converged_x,
            "expected_x": stable_fp["x"],
            "error": abs(converged_x - stable_fp["x"]),
        }
        logger.info(
            f"  Converged to x={converged_x:.6f} "
            f"(expected {stable_fp['x']:.6f})"
        )

    # Part 2: Bifurcation diagram + period detection
    logger.info("Part 2: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(n_beta=500)
    results["bifurcation"] = {
        "n_beta": 500,
        "beta_range": [-1.0, 1.0],
    }
    np.savez(
        output_path / "bifurcation_data.npz",
        beta_values=bif_data["beta_values"],
        periods=bif_data["periods"],
        bif_param=bif_data["bif_param"],
        bif_x=bif_data["bif_x"],
    )

    # Feigenbaum constant from period-doubling
    feig = estimate_feigenbaum(bif_data["periods"], bif_data["beta_values"])
    results["feigenbaum"] = feig

    # Part 3: Lyapunov exponent spectrum
    logger.info("Part 3: Lyapunov exponent vs beta...")
    lyap_data = generate_lyapunov_data(n_beta=100)

    # Chaos onset (first beta where lambda > 0)
    chaotic = lyap_data["lyapunov"] > 0
    if np.any(chaotic):
        beta_chaos = lyap_data["beta"][np.argmax(chaotic)]
        results["chaos_onset"] = {
            "beta_estimate": float(beta_chaos),
        }
        logger.info(f"  Chaos onset: beta ~ {beta_chaos:.4f}")

    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "beta_at_max": float(
            lyap_data["beta"][np.argmax(lyap_data["lyapunov"])]
        ),
    }

    # Part 4: Verify Lyapunov at specific parameter values
    logger.info("Part 4: Specific Lyapunov checks...")
    for beta_test in [-0.8, -0.5, 0.0, 0.5]:
        sim_test = GaussMapSimulation(
            _make_config(alpha=6.2, beta=beta_test)
        )
        lam = sim_test.compute_lyapunov(n_transient=1000, n_compute=10000)
        logger.info(f"  beta={beta_test}: Lyapunov={lam:.4f}")

    # PySR: fit Lyapunov(beta) in chaotic region
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        chaotic_region = lyap_data["lyapunov"] > 0
        if np.sum(chaotic_region) > 5:
            X = lyap_data["beta"][chaotic_region].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_region]

            logger.info(
                "  Running PySR: lambda = f(beta) for chaotic region..."
            )
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["b_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["log", "sqrt", "exp"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["lyapunov_pysr"] = {
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
                results["lyapunov_pysr"]["best"] = best.expression
                results["lyapunov_pysr"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["lyapunov_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "lyapunov_data.npz",
        beta=lyap_data["beta"],
        lyapunov=lyap_data["lyapunov"],
    )

    return results
