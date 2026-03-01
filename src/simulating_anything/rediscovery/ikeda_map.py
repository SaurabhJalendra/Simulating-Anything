"""Ikeda map rediscovery.

Targets:
- Bifurcation diagram: period-doubling route to chaos as u increases
- Lyapunov exponent as function of u (positive for u > ~0.8)
- Fixed point tracking as function of u
- Dissipative property: |det(J)| = u^2
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.ikeda_map import IkedaMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(u: float = 0.9) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.IKEDA_MAP,
        dt=1.0,
        n_steps=1000,
        parameters={"u": u, "x_0": 0.0, "y_0": 0.0},
    )


def generate_bifurcation_data(
    n_u: int = 500,
    u_min: float = 0.1,
    u_max: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram and period detection data."""
    u_values = np.linspace(u_min, u_max, n_u)

    # Bifurcation diagram points
    sim = IkedaMapSimulation(_make_config(u=0.9))
    bif_data = sim.bifurcation_sweep(u_values, n_transient=500, n_record=50)

    # Period detection
    periods = []
    for u_val in u_values:
        config_u = _make_config(u=u_val)
        sim_u = IkedaMapSimulation(config_u)
        sim_u.reset()
        p = sim_u.detect_period(max_period=64)
        periods.append(p)

    return {
        "u_values": u_values,
        "periods": np.array(periods),
        "bif_u": bif_data["u"],
        "bif_x": bif_data["x"],
        "bif_y": bif_data["y"],
    }


def generate_lyapunov_data(
    n_u: int = 100,
    u_min: float = 0.1,
    u_max: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs u data."""
    u_values = np.linspace(u_min, u_max, n_u)
    lyapunovs = []

    for i, u_val in enumerate(u_values):
        config = _make_config(u=u_val)
        sim = IkedaMapSimulation(config)
        sim.reset()
        lam = sim.compute_lyapunov(n_steps=5000, n_transient=500)
        lyapunovs.append(lam)

        if (i + 1) % 25 == 0:
            logger.info(f"  u={u_val:.4f}: Lyapunov={lam:.4f}")

    return {
        "u": u_values,
        "lyapunov": np.array(lyapunovs),
    }


def generate_fixed_point_data(
    n_u: int = 50,
    u_min: float = 0.1,
    u_max: float = 1.0,
) -> dict[str, list]:
    """Track fixed points as a function of u."""
    u_values = np.linspace(u_min, u_max, n_u)
    all_fps = []

    for u_val in u_values:
        config = _make_config(u=u_val)
        sim = IkedaMapSimulation(config)
        fps = sim.find_fixed_points()
        all_fps.append({
            "u": float(u_val),
            "n_fixed_points": len(fps),
            "fixed_points": [fp.tolist() for fp in fps],
        })

    return {"u_values": u_values.tolist(), "fixed_points": all_fps}


def run_ikeda_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/ikeda_map",
    n_iterations: int = 40,
) -> dict:
    """Run Ikeda map rediscovery.

    Args:
        output_dir: Directory for output files.
        n_iterations: Number of PySR iterations.

    Returns:
        Results dict with bifurcation, Lyapunov, and fixed point data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "ikeda_map",
        "targets": {
            "dissipative": "|det(J)| = u^2 (area contraction)",
            "lyapunov": "lambda > 0 for chaotic regime (u > ~0.8)",
            "fixed_points": "numerical root finding for equilibria",
        },
    }

    # Part 1: Bifurcation diagram
    logger.info("Part 1: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(n_u=50)
    results["bifurcation"] = {
        "n_u": 50,
        "u_range": [0.1, 1.0],
    }
    np.savez(
        output_path / "bifurcation_data.npz",
        u_values=bif_data["u_values"],
        periods=bif_data["periods"],
        bif_u=bif_data["bif_u"],
        bif_x=bif_data["bif_x"],
        bif_y=bif_data["bif_y"],
    )

    # Part 2: Lyapunov exponent
    logger.info("Part 2: Lyapunov exponent vs u...")
    lyap_data = generate_lyapunov_data(n_u=50)

    # Chaos onset (first u where lambda > 0)
    chaotic = lyap_data["lyapunov"] > 0
    if np.any(chaotic):
        u_chaos = lyap_data["u"][np.argmax(chaotic)]
        results["chaos_onset"] = {
            "u_estimate": float(u_chaos),
        }
        logger.info(f"  Chaos onset: u ~ {u_chaos:.4f}")

    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "u_at_max": float(lyap_data["u"][np.argmax(lyap_data["lyapunov"])]),
    }

    # Verify dissipative property at classic params
    classic_config = _make_config(u=0.9)
    classic_sim = IkedaMapSimulation(classic_config)
    classic_sim.reset()
    # At any point, |det(J)| should equal u^2
    classic_sim.step()
    x, y = classic_sim.observe()
    jac = classic_sim.compute_jacobian(x, y)
    det_j = abs(np.linalg.det(jac))
    results["dissipative"] = {
        "u": 0.9,
        "u_squared": 0.81,
        "det_jacobian": float(det_j),
        "matches": abs(det_j - 0.81) < 0.01,
    }
    logger.info(f"  |det(J)| = {det_j:.6f}, u^2 = 0.81")

    # Part 3: Fixed point tracking
    logger.info("Part 3: Fixed point tracking...")
    fp_data = generate_fixed_point_data(n_u=20)
    results["fixed_points"] = {
        "n_u": 20,
        "data": fp_data["fixed_points"],
    }

    # PySR: find Lyapunov(u) in chaotic region
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        chaotic_region = lyap_data["lyapunov"] > 0
        if np.sum(chaotic_region) > 5:
            X = lyap_data["u"][chaotic_region].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_region]

            logger.info("  Running PySR: lambda = f(u) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["u_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["log", "sqrt"],
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
                results["lyapunov_pysr"]["best_r2"] = best.evidence.fit_r_squared
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
        u=lyap_data["u"],
        lyapunov=lyap_data["lyapunov"],
    )

    return results
