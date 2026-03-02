"""Cubic map rediscovery.

Targets:
- Period-doubling bifurcation diagram
- Fixed points: x*=0 always, x*=+/-sqrt(r-1) for r > 1
- Stability transitions: origin at |r|=1, nontrivial at r=2
- Odd symmetry: f(-x) = -f(x)
- Lyapunov exponent as function of r
- Feigenbaum constant from period-doubling cascade
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.cubic_map import CubicMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    r: float = 2.5,
    x_0: float = 0.5,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CUBIC_MAP,  
        dt=1.0,
        n_steps=1000,
        parameters={"r": r, "x_0": x_0},
    )


def generate_bifurcation_data(
    n_r: int = 500,
    r_min: float = 0.5,
    r_max: float = 3.0,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram and period detection data."""
    r_values = np.linspace(r_min, r_max, n_r)

    # Bifurcation diagram points
    sim = CubicMapSimulation(_make_config(r=2.5))
    bif_data = sim.bifurcation_diagram(r_values, n_transient=500, n_plot=50)

    # Period detection
    periods = []
    for r in r_values:
        config_r = _make_config(r=r)
        sim_r = CubicMapSimulation(config_r)
        sim_r.reset()
        p = sim_r.detect_period(max_period=64)
        periods.append(p)

    return {
        "r_values": r_values,
        "periods": np.array(periods),
        "bif_r": bif_data["r"],
        "bif_x": bif_data["x"],
    }


def generate_lyapunov_data(
    n_r: int = 100,
    r_min: float = 0.5,
    r_max: float = 3.0,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs r data."""
    r_values = np.linspace(r_min, r_max, n_r)
    lyapunovs = []

    for i, r in enumerate(r_values):
        config = _make_config(r=r)
        sim = CubicMapSimulation(config)
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        lyapunovs.append(lam)

        if (i + 1) % 25 == 0:
            logger.info(f"  r={r:.4f}: Lyapunov={lam:.4f}")

    return {
        "r": r_values,
        "lyapunov": np.array(lyapunovs),
    }


def analyze_fixed_points(
    r_values: np.ndarray | None = None,
) -> dict:
    """Analyze fixed point structure across r values.

    Verifies:
    - x*=0 exists for all r
    - x*=+/-sqrt(r-1) exists for r > 1
    - Stability transitions at r=1 (origin) and r=2 (nontrivial)
    """
    if r_values is None:
        r_values = np.array([0.5, 0.9, 1.0, 1.1, 1.5, 1.9, 2.0, 2.1, 2.5])

    results = []
    for r in r_values:
        sim = CubicMapSimulation(_make_config(r=r))
        fps = sim.find_fixed_points()
        results.append({
            "r": float(r),
            "n_fixed_points": len(fps),
            "fixed_points": fps,
        })

    return {"fixed_point_sweep": results}


def verify_symmetry(
    r: float = 2.5,
    n_steps: int = 100,
) -> dict:
    """Verify odd symmetry of the cubic map: f(-x) = -f(x).

    Checks that if x_0 -> x_1 -> x_2 -> ..., then
    -x_0 -> -x_1 -> -x_2 -> ...
    """
    sim_pos = CubicMapSimulation(_make_config(r=r, x_0=0.3))
    sim_neg = CubicMapSimulation(_make_config(r=r, x_0=-0.3))
    sim_pos.reset()
    sim_neg.reset()

    max_error = 0.0
    for _ in range(n_steps):
        sim_pos.step()
        sim_neg.step()
        x_pos = sim_pos.observe()[0]
        x_neg = sim_neg.observe()[0]
        error = abs(x_pos + x_neg)
        max_error = max(max_error, error)

    return {
        "r": r,
        "n_steps": n_steps,
        "max_symmetry_error": max_error,
        "symmetric": max_error < 1e-10,
    }


def estimate_feigenbaum(
    periods: np.ndarray,
    r_values: np.ndarray,
) -> dict:
    """Estimate Feigenbaum constant from period-doubling bifurcation points."""
    bif_points = []
    target_periods = [2, 4, 8, 16]

    for p in target_periods:
        idx = np.where(periods >= p)[0]
        if len(idx) > 0:
            bif_points.append(float(r_values[idx[0]]))

    result: dict = {"bifurcation_points": bif_points}

    if len(bif_points) >= 3:
        deltas = []
        for i in range(len(bif_points) - 2):
            d1 = bif_points[i + 1] - bif_points[i]
            d2 = bif_points[i + 2] - bif_points[i + 1]
            if d2 > 1e-10:
                deltas.append(d1 / d2)

        if deltas:
            result["feigenbaum_estimates"] = deltas
            result["best_estimate"] = deltas[-1] if deltas else None
            logger.info(f"  Feigenbaum estimates: {deltas}")

    return result


def run_cubic_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/cubic_map",
    n_iterations: int = 40,
) -> dict:
    """Run cubic map rediscovery.

    Discovers:
    - Fixed point structure and stability
    - Odd symmetry verification
    - Bifurcation diagram and period-doubling cascade
    - Lyapunov exponent spectrum
    - Chaos onset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "cubic_map",
        "targets": {
            "fixed_points": "x*=0 always, x*=+/-sqrt(r-1) for r>1",
            "stability_origin": "|r| < 1 for x*=0 stability",
            "stability_nontrivial": "1 < r < 2 for x*=+/-sqrt(r-1) stability",
            "symmetry": "f(-x) = -f(x) (odd map)",
            "chaos_onset": "r ~ 2 (nontrivial fixed point loses stability)",
            "feigenbaum": "delta ~ 4.669",
        },
    }

    # Part 1: Fixed point analysis
    logger.info("Part 1: Fixed point analysis...")
    fp_results = analyze_fixed_points()
    results["fixed_points"] = fp_results

    # Part 2: Symmetry verification
    logger.info("Part 2: Symmetry verification...")
    sym_results = verify_symmetry(r=2.5, n_steps=200)
    results["symmetry"] = sym_results
    logger.info(
        f"  Symmetry error: {sym_results['max_symmetry_error']:.2e} "
        f"(symmetric={sym_results['symmetric']})"
    )

    # Part 3: Convergence to nontrivial fixed point for 1 < r < 2
    logger.info("Part 3: Fixed point convergence verification...")
    convergence_tests = []
    for r_test in [1.2, 1.5, 1.8]:
        x_star = np.sqrt(r_test - 1.0)
        sim = CubicMapSimulation(_make_config(r=r_test, x_0=0.3))
        sim.reset()
        for _ in range(1000):
            sim.step()
        final_x = float(sim.observe()[0])
        # Should converge to +sqrt(r-1) or -sqrt(r-1)
        error = min(abs(final_x - x_star), abs(final_x + x_star))
        convergence_tests.append({
            "r": r_test,
            "x_star": float(x_star),
            "final_x": final_x,
            "error": error,
        })
        logger.info(
            f"  r={r_test}: converged to x={final_x:.6f} "
            f"(x*=+/-{x_star:.6f}, error={error:.2e})"
        )
    results["convergence_tests"] = convergence_tests

    # Part 4: Bifurcation diagram + period detection
    logger.info("Part 4: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(n_r=500)
    results["bifurcation"] = {
        "n_r": 500,
        "r_range": [0.5, 3.0],
    }
    np.savez(
        output_path / "bifurcation_data.npz",
        r_values=bif_data["r_values"],
        periods=bif_data["periods"],
        bif_r=bif_data["bif_r"],
        bif_x=bif_data["bif_x"],
    )

    # Feigenbaum constant from period-doubling
    feig = estimate_feigenbaum(bif_data["periods"], bif_data["r_values"])
    results["feigenbaum"] = feig

    # Part 5: Lyapunov exponent spectrum
    logger.info("Part 5: Lyapunov exponent vs r...")
    lyap_data = generate_lyapunov_data(n_r=100)

    # Chaos onset (first r where lambda > 0)
    valid = np.isfinite(lyap_data["lyapunov"])
    if np.any(valid):
        valid_lyap = lyap_data["lyapunov"][valid]
        valid_r = lyap_data["r"][valid]
        chaotic = valid_lyap > 0
        if np.any(chaotic):
            r_chaos = valid_r[np.argmax(chaotic)]
            results["chaos_onset"] = {
                "r_estimate": float(r_chaos),
                "r_theory": 2.0,
            }
            logger.info(f"  Chaos onset: r ~ {r_chaos:.4f} (theory: ~2.0)")

        results["lyapunov"] = {
            "max_lyapunov": float(np.nanmax(valid_lyap)),
            "r_at_max": float(valid_r[np.nanargmax(valid_lyap)]),
        }

    # Lyapunov at nontrivial fixed point: ln|3 - 2r|
    logger.info("  Checking theoretical Lyapunov at nontrivial fixed point...")
    for r_test in [1.5, 1.8]:
        sim_test = CubicMapSimulation(_make_config(r=r_test))
        lam_num = sim_test.compute_lyapunov(n_iterations=10000, n_transient=500)
        lam_theory = np.log(abs(3.0 - 2.0 * r_test))
        logger.info(
            f"  r={r_test}: Lyapunov={lam_num:.4f} "
            f"(theory ln|3-2r|={lam_theory:.4f})"
        )

    # PySR: fit Lyapunov(r) in chaotic region
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        valid_mask = np.isfinite(lyap_data["lyapunov"])
        chaotic_region = valid_mask & (lyap_data["lyapunov"] > 0)
        if np.sum(chaotic_region) > 5:
            X = lyap_data["r"][chaotic_region].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_region]

            logger.info("  Running PySR: lambda = f(r) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["r_"],
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
        r=lyap_data["r"],
        lyapunov=lyap_data["lyapunov"],
    )

    return results
