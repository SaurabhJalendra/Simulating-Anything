"""Tinkerbell map rediscovery.

Targets:
- Strange attractor at classic parameters (a=0.9, b=-0.6013, c=2.0, d=0.5)
- Lyapunov exponent sweep over parameter a
- Bifurcation diagram with period-doubling route to chaos
- Fixed point analysis
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.tinkerbell_map import TinkerbellMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    a: float = 0.9,
    b: float = -0.6013,
    c: float = 2.0,
    d: float = 0.5,
    x_0: float = -0.72,
    y_0: float = -0.64,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.TINKERBELL_MAP,
        dt=1.0,
        n_steps=1000,
        parameters={
            "a": a, "b": b, "c": c, "d": d,
            "x_0": x_0, "y_0": y_0,
        },
    )


def generate_lyapunov_data(
    n_a: int = 100,
    a_min: float = -0.5,
    a_max: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs a data."""
    a_values = np.linspace(a_min, a_max, n_a)
    lyapunovs = []

    for i, a_val in enumerate(a_values):
        config = _make_config(a=a_val)
        sim = TinkerbellMapSimulation(config)
        sim.reset()
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        lyapunovs.append(lam)

        if (i + 1) % 25 == 0:
            logger.info(f"  a={a_val:.4f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov": np.array(lyapunovs),
    }


def generate_bifurcation_data(
    n_a: int = 500,
    a_min: float = -0.5,
    a_max: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram and period detection data."""
    a_values = np.linspace(a_min, a_max, n_a)

    sim = TinkerbellMapSimulation(_make_config())
    bif_data = sim.bifurcation_diagram(a_values, n_transient=500, n_plot=50)

    # Period detection
    periods = []
    for a_val in a_values:
        config_a = _make_config(a=a_val)
        sim_a = TinkerbellMapSimulation(config_a)
        sim_a.reset()
        p = sim_a.detect_period(max_period=64, n_transient=1000)
        periods.append(p)

    return {
        "a_values": a_values,
        "periods": np.array(periods),
        "bif_a": bif_data["a"],
        "bif_x": bif_data["x"],
    }


def run_tinkerbell_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/tinkerbell_map",
    n_iterations: int = 40,
) -> dict:
    """Run Tinkerbell map rediscovery analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "tinkerbell_map",
        "targets": {
            "attractor": "Strange attractor at a=0.9, b=-0.6013, c=2.0, d=0.5",
            "lyapunov_classic": "Positive Lyapunov exponent at classic params",
            "fixed_points": "Solve x^2+(a-1)*x-y^2+b*y=0, 2*x*y+c*x+(d-1)*y=0",
        },
    }

    # Part 1: Fixed points at classic parameters
    logger.info("Part 1: Fixed point analysis...")
    classic_config = _make_config()
    sim = TinkerbellMapSimulation(classic_config)
    sim.reset()
    fps = sim.find_fixed_points()
    results["fixed_points"] = {
        "count": len(fps),
        "points": [fp.tolist() for fp in fps],
    }
    for i, fp in enumerate(fps):
        det_j = sim.jacobian_determinant_at(fp[0], fp[1])
        jac = sim.compute_jacobian(fp[0], fp[1])
        eigenvalues = np.linalg.eigvals(jac)
        results["fixed_points"][f"fp_{i}"] = {
            "x": float(fp[0]),
            "y": float(fp[1]),
            "det_J": float(det_j),
            "eigenvalues": [complex(e).real for e in eigenvalues],
            "stable": bool(np.all(np.abs(eigenvalues) < 1.0)),
        }
        logger.info(
            f"  FP{i}: ({fp[0]:.6f}, {fp[1]:.6f}), "
            f"det(J)={det_j:.4f}, |eig|={np.abs(eigenvalues)}"
        )

    # Part 2: Lyapunov exponent at classic parameters
    logger.info("Part 2: Lyapunov exponent at classic parameters...")
    classic_lyap = sim.compute_lyapunov(n_iterations=50000, n_transient=1000)
    spectrum = sim.compute_lyapunov_spectrum(
        n_iterations=50000, n_transient=1000,
    )
    results["classic_lyapunov"] = {
        "a": 0.9,
        "b": -0.6013,
        "c": 2.0,
        "d": 0.5,
        "lyapunov_max": float(classic_lyap),
        "spectrum": spectrum.tolist(),
        "is_chaotic": bool(classic_lyap > 0),
    }
    logger.info(
        f"  Classic Lyapunov: {classic_lyap:.4f}, "
        f"spectrum: {spectrum}"
    )

    # Part 3: Bifurcation diagram
    logger.info("Part 3: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(n_a=200)
    results["bifurcation"] = {
        "n_a": 200,
        "a_range": [-0.5, 1.0],
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

    chaotic = lyap_data["lyapunov"] > 0
    valid = ~np.isnan(lyap_data["lyapunov"])
    chaotic_valid = chaotic & valid
    if np.any(chaotic_valid):
        a_chaos = lyap_data["a"][np.argmax(chaotic_valid)]
        results["chaos_onset"] = {
            "a_estimate": float(a_chaos),
        }
        logger.info(f"  Chaos onset: a ~ {a_chaos:.4f}")

    valid_lyap = lyap_data["lyapunov"][valid]
    valid_a = lyap_data["a"][valid]
    if len(valid_lyap) > 0:
        results["lyapunov"] = {
            "max_lyapunov": float(np.max(valid_lyap)),
            "a_at_max": float(valid_a[np.argmax(valid_lyap)]),
        }

    # Part 5: PySR on Lyapunov(a) in chaotic region
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        chaotic_region = (lyap_data["lyapunov"] > 0) & valid
        if np.sum(chaotic_region) > 5:
            X = lyap_data["a"][chaotic_region].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_region]

            logger.info("  Running PySR: lambda = f(a) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["a_"],
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
        a=lyap_data["a"],
        lyapunov=lyap_data["lyapunov"],
    )

    return results
