"""Lozi map rediscovery.

Targets:
- Period-doubling bifurcation diagram as a varies
- Lyapunov exponent as function of a (positive in chaotic regime)
- Fixed point verification: x* = 1/(1+a-b), y* = b*x*
- Jacobian determinant |det(J)| = |b| (constant area contraction)
- Attractor correlation dimension estimate
- PySR fit of Lyapunov(a) in chaotic region
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.lozi_map import LoziMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(a: float = 1.7, b: float = 0.5) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.LOZI_MAP,
        dt=1.0,
        n_steps=1000,
        parameters={"a": a, "b": b, "x_0": 0.0, "y_0": 0.0},
    )


def generate_bifurcation_data(
    n_a: int = 500,
    a_min: float = 0.0,
    a_max: float = 1.8,
    b: float = 0.5,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram and period detection data."""
    a_values = np.linspace(a_min, a_max, n_a)

    # Bifurcation diagram points
    sim = LoziMapSimulation(_make_config(a=1.7, b=b))
    bif_data = sim.bifurcation_diagram(a_values, n_transient=500, n_plot=50)

    # Period detection
    periods = []
    for a in a_values:
        config_a = _make_config(a=a, b=b)
        sim_a = LoziMapSimulation(config_a)
        sim_a.reset()
        p = sim_a.detect_period(max_period=64)
        periods.append(p)

    return {
        "a_values": a_values,
        "periods": np.array(periods),
        "bif_a": bif_data["a"],
        "bif_x": bif_data["x"],
    }


def generate_lyapunov_data(
    n_a: int = 100,
    a_min: float = 0.0,
    a_max: float = 1.8,
    b: float = 0.5,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs a data."""
    a_values = np.linspace(a_min, a_max, n_a)
    lyapunovs = []

    for i, a in enumerate(a_values):
        config = _make_config(a=a, b=b)
        sim = LoziMapSimulation(config)
        sim.reset()
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        lyapunovs.append(lam)

        if (i + 1) % 25 == 0:
            logger.info(f"  a={a:.4f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov": np.array(lyapunovs),
    }


def generate_fixed_point_data(
    n_a: int = 50,
    a_min: float = 0.1,
    a_max: float = 1.8,
    b: float = 0.5,
) -> dict:
    """Generate fixed point data across a range of a values."""
    a_values = np.linspace(a_min, a_max, n_a)
    fixed_points = []

    for a in a_values:
        config = _make_config(a=a, b=b)
        sim = LoziMapSimulation(config)
        fps = sim.fixed_points
        entry = {
            "a": float(a),
            "b": float(b),
            "n_fixed_points": len(fps),
            "fixed_points": [fp.tolist() for fp in fps],
        }
        fixed_points.append(entry)

    return {
        "a_values": a_values,
        "fixed_points": fixed_points,
    }


def run_lozi_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/lozi_map",
    n_iterations: int = 40,
) -> dict:
    """Run Lozi map rediscovery."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "lozi_map",
        "targets": {
            "fixed_point": "x* = 1/(1+a-b), y* = b*x*",
            "jacobian_det": "|det(J)| = |b| (constant area contraction)",
            "lyapunov_sum": "lambda_1 + lambda_2 = ln|b|",
            "attractor": "strange attractor at a=1.7, b=0.5",
        },
    }

    # Part 1: Fixed point verification
    logger.info("Part 1: Fixed point verification...")
    b = 0.5
    a_test = 1.7
    config = _make_config(a=a_test, b=b)
    sim = LoziMapSimulation(config)
    fps = sim.fixed_points

    fp_results = []
    for fp in fps:
        x, y = fp
        # Verify f(x*) = x*
        x_new = 1.0 - a_test * abs(x) + y
        y_new = b * x
        error = np.sqrt((x_new - x) ** 2 + (y_new - y) ** 2)
        fp_results.append({
            "x": float(x),
            "y": float(y),
            "error": float(error),
        })
    results["fixed_points"] = {
        "a": a_test,
        "b": b,
        "n_fixed_points": len(fps),
        "points": fp_results,
    }
    logger.info(f"  Found {len(fps)} fixed points")

    # Part 2: Jacobian determinant verification
    logger.info("Part 2: Jacobian determinant = -b...")
    det_j = sim.jacobian_determinant
    results["jacobian_determinant"] = {
        "det_J": float(det_j),
        "negative_b": float(-b),
        "abs_det_J": float(abs(det_j)),
        "abs_b": float(abs(b)),
        "match": bool(abs(det_j - (-b)) < 1e-14),
    }
    logger.info(f"  det(J) = {det_j}, -b = {-b}")

    # Part 3: Bifurcation diagram
    logger.info("Part 3: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(n_a=200)
    results["bifurcation"] = {
        "n_a": 200,
        "a_range": [0.0, 1.8],
    }
    np.savez(
        output_path / "bifurcation_data.npz",
        a_values=bif_data["a_values"],
        periods=bif_data["periods"],
        bif_a=bif_data["bif_a"],
        bif_x=bif_data["bif_x"],
    )

    # Part 4: Lyapunov exponent sweep
    logger.info("Part 4: Lyapunov exponent vs a...")
    lyap_data = generate_lyapunov_data(n_a=100)

    # Classic Lyapunov at a=1.7, b=0.5
    classic_config = _make_config(a=1.7, b=0.5)
    classic_sim = LoziMapSimulation(classic_config)
    classic_sim.reset()
    classic_lyap = classic_sim.compute_lyapunov(
        n_iterations=50000, n_transient=1000,
    )
    results["classic_lyapunov"] = {
        "a": 1.7,
        "b": 0.5,
        "lyapunov": float(classic_lyap),
    }
    logger.info(f"  Classic Lyapunov (a=1.7, b=0.5): {classic_lyap:.4f}")

    # Lyapunov spectrum sum verification: lambda_1 + lambda_2 = ln|b|
    spectrum = classic_sim.compute_lyapunov_spectrum(
        n_iterations=50000, n_transient=1000,
    )
    lyap_sum = float(spectrum[0] + spectrum[1])
    ln_b = float(np.log(abs(b)))
    results["lyapunov_spectrum"] = {
        "lambda_1": float(spectrum[0]),
        "lambda_2": float(spectrum[1]),
        "sum": lyap_sum,
        "ln_abs_b": ln_b,
        "sum_error": float(abs(lyap_sum - ln_b)),
    }
    logger.info(
        f"  Lyapunov spectrum: [{spectrum[0]:.4f}, {spectrum[1]:.4f}], "
        f"sum={lyap_sum:.4f}, ln|b|={ln_b:.4f}"
    )

    # Chaos onset
    chaotic = lyap_data["lyapunov"] > 0
    if np.any(chaotic):
        a_chaos = lyap_data["a"][np.argmax(chaotic)]
        results["chaos_onset"] = {
            "a_estimate": float(a_chaos),
        }
        logger.info(f"  Chaos onset: a ~ {a_chaos:.4f}")

    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "a_at_max": float(lyap_data["a"][np.argmax(lyap_data["lyapunov"])]),
    }

    # Part 5: Attractor dimension
    logger.info("Part 5: Attractor correlation dimension...")
    dim_sim = LoziMapSimulation(_make_config(a=1.7, b=0.5))
    dim_sim.reset()
    dim = dim_sim.compute_attractor_dimension(n_steps=10000)
    results["attractor_dimension"] = {
        "correlation_dimension": float(dim),
        "note": "Lozi attractor dimension ~1.2 expected",
    }
    logger.info(f"  Correlation dimension: {dim:.3f}")

    # Part 6: PySR fit of Lyapunov(a) in chaotic region
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        chaotic_region = lyap_data["lyapunov"] > 0
        if np.sum(chaotic_region) > 5:
            X = lyap_data["a"][chaotic_region].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_region]

            logger.info("  Running PySR: lambda = f(a) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["a_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["log", "sqrt", "abs"],
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
        a=lyap_data["a"],
        lyapunov=lyap_data["lyapunov"],
    )

    return results
