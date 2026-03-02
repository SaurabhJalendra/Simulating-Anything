"""Tent map rediscovery.

Targets:
- Lyapunov exponent = ln(r) for r in (1, 2]
- Fixed points: x* = 0, x* = r/(r+1)
- Invariant density uniform on [0,1] at r = 2
- Direct transition to chaos (no period-doubling)
- Bifurcation diagram
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.tent_map import TentMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    r: float = 2.0,
    x_0: float = 1.0 / np.pi,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.TENT_MAP,  
        dt=1.0,
        n_steps=1000,
        parameters={"r": r, "x_0": x_0},
    )


def generate_lyapunov_data(
    n_r: int = 100,
    r_min: float = 0.1,
    r_max: float = 2.0,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs r data.

    For the tent map, the theoretical Lyapunov exponent is ln(r) when r > 1
    and the orbit is ergodic. For r <= 1, the orbit converges to a fixed point
    and the Lyapunov exponent is ln(r) < 0.
    """
    r_values = np.linspace(r_min, r_max, n_r)
    lyapunovs = []
    theory = []

    for i, r in enumerate(r_values):
        config = _make_config(r=r)
        sim = TentMapSimulation(config)
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=500)
        lyapunovs.append(lam)
        theory.append(np.log(r) if r > 0 else float("-inf"))

        if (i + 1) % 25 == 0:
            logger.info(f"  r={r:.4f}: Lyapunov={lam:.4f} (theory: ln(r)={theory[-1]:.4f})")

    return {
        "r": r_values,
        "lyapunov": np.array(lyapunovs),
        "lyapunov_theory": np.array(theory),
    }


def generate_bifurcation_data(
    n_r: int = 500,
    r_min: float = 0.0,
    r_max: float = 2.0,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram and period detection data."""
    r_values = np.linspace(r_min, r_max, n_r)

    # Bifurcation diagram points
    config = _make_config(r=2.0)
    sim = TentMapSimulation(config)
    bif_data = sim.bifurcation_diagram(r_values, n_transient=500, n_plot=50)

    # Period detection
    periods = []
    for r in r_values:
        config_r = _make_config(r=r)
        sim_r = TentMapSimulation(config_r)
        sim_r.reset()
        p = sim_r.detect_period(max_period=64)
        periods.append(p)

    return {
        "r_values": r_values,
        "periods": np.array(periods),
        "bif_r": bif_data["r"],
        "bif_x": bif_data["x"],
    }


def verify_fixed_points(
    r_values: np.ndarray | None = None,
) -> list[dict]:
    """Verify fixed point x* = r/(r+1) for a range of r values.

    The nontrivial fixed point only exists for r >= 1, where it lies on
    the right branch (x >= 0.5). For r < 1, only x*=0 exists.
    """
    if r_values is None:
        r_values = np.array([1.0, 1.2, 1.5, 1.8, 2.0])

    results = []
    for r in r_values:
        config = _make_config(r=r, x_0=0.3)
        sim = TentMapSimulation(config)
        fps = sim.find_fixed_points()

        # Nontrivial fixed point exists for r >= 1
        x_star_theory = r / (r + 1.0)
        nontrivial = fps[1] if len(fps) > 1 else None

        result = {
            "r": float(r),
            "x_star_theory": float(x_star_theory),
            "x_star_found": float(nontrivial["x"]) if nontrivial else None,
            "error": float(abs(nontrivial["x"] - x_star_theory)) if nontrivial else None,
            "stability": nontrivial["stability"] if nontrivial else None,
        }
        results.append(result)
        logger.info(
            f"  r={r:.2f}: x*={x_star_theory:.6f}, "
            f"stability={result['stability']}"
        )

    return results


def verify_invariant_density(
    n_bins: int = 50,
    n_iterations: int = 200000,
) -> dict:
    """Verify that the invariant density at r=2 is uniform on [0, 1].

    A perfectly uniform density has value 1.0 everywhere in [0, 1].
    We measure how close the empirical density is to uniform.
    """
    config = _make_config(r=2.0)
    sim = TentMapSimulation(config)
    density_data = sim.compute_invariant_density(
        n_bins=n_bins,
        n_iterations=n_iterations,
        n_transient=1000,
    )

    density = density_data["density"]
    # For a uniform distribution on [0,1], the density should be 1.0 everywhere
    mean_density = float(np.mean(density))
    std_density = float(np.std(density))
    max_deviation = float(np.max(np.abs(density - 1.0)))

    result = {
        "r": 2.0,
        "n_bins": n_bins,
        "n_iterations": n_iterations,
        "mean_density": mean_density,
        "std_density": std_density,
        "max_deviation_from_uniform": max_deviation,
        "is_approximately_uniform": max_deviation < 0.2,
    }
    logger.info(
        f"  Invariant density at r=2: mean={mean_density:.4f}, "
        f"std={std_density:.4f}, max_dev={max_deviation:.4f}"
    )
    return result


def run_tent_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/tent_map",
    n_iterations: int = 40,
) -> dict:
    """Run tent map rediscovery.

    Discovers:
    - Lyapunov exponent = ln(r) for r > 1
    - Fixed point structure
    - Invariant density uniformity at r = 2
    - Bifurcation diagram (direct chaos transition, no period-doubling)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "tent_map",
        "targets": {
            "lyapunov": "ln(r) for ergodic orbits",
            "lyapunov_r2": "ln(2) ~ 0.6931 at r=2",
            "fixed_points": "x*=0 (trivial), x*=r/(r+1) (nontrivial)",
            "invariant_density": "uniform on [0,1] at r=2",
            "chaos_onset": "r=1 (direct transition, no period-doubling)",
        },
    }

    # Part 1: Fixed point analysis
    logger.info("Part 1: Fixed point analysis...")
    fp_results = verify_fixed_points()
    results["fixed_points"] = fp_results

    # Verify convergence to fixed point for r < 1
    config_stable = _make_config(r=0.8, x_0=0.3)
    sim_stable = TentMapSimulation(config_stable)
    sim_stable.reset()
    for _ in range(1000):
        sim_stable.step()
    converged_x = float(sim_stable.observe()[0])
    results["convergence_test"] = {
        "r": 0.8,
        "final_x": converged_x,
        "expected": 0.0,
        "note": "For r < 1, orbit converges to x*=0",
    }
    logger.info(f"  r=0.8: converged to x={converged_x:.6f}")

    # Part 2: Lyapunov exponent sweep
    logger.info("Part 2: Lyapunov exponent vs r...")
    lyap_data = generate_lyapunov_data(n_r=100)

    # Verify ln(r) match
    r_gt1 = lyap_data["r"] > 1.0
    if np.any(r_gt1):
        numerical = lyap_data["lyapunov"][r_gt1]
        theoretical = lyap_data["lyapunov_theory"][r_gt1]
        correlation = float(np.corrcoef(numerical, theoretical)[0, 1])
        mean_error = float(np.mean(np.abs(numerical - theoretical)))
        results["lyapunov_verification"] = {
            "correlation_with_ln_r": correlation,
            "mean_absolute_error": mean_error,
            "n_points": int(np.sum(r_gt1)),
        }
        logger.info(
            f"  Lyapunov vs ln(r) correlation: {correlation:.6f}, "
            f"mean error: {mean_error:.6f}"
        )

    # Exact check at r=2: lambda should be ln(2)
    config_r2 = _make_config(r=2.0)
    sim_r2 = TentMapSimulation(config_r2)
    lam_r2 = sim_r2.compute_lyapunov(n_iterations=50000, n_transient=1000)
    results["lyapunov_at_r2"] = {
        "numerical": float(lam_r2),
        "theory": float(np.log(2)),
        "error": float(abs(lam_r2 - np.log(2))),
    }
    logger.info(
        f"  Lyapunov at r=2: {lam_r2:.6f} (theory: ln(2)={np.log(2):.6f})"
    )

    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "r_at_max": float(lyap_data["r"][np.argmax(lyap_data["lyapunov"])]),
    }

    # Part 3: Bifurcation diagram
    logger.info("Part 3: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(n_r=500)
    results["bifurcation"] = {
        "n_r": 500,
        "r_range": [0.0, 2.0],
    }
    np.savez(
        output_path / "bifurcation_data.npz",
        r_values=bif_data["r_values"],
        periods=bif_data["periods"],
        bif_r=bif_data["bif_r"],
        bif_x=bif_data["bif_x"],
    )

    # Part 4: Invariant density at r=2
    logger.info("Part 4: Invariant density at r=2...")
    density_result = verify_invariant_density()
    results["invariant_density"] = density_result

    # Part 5: PySR attempt on Lyapunov(r)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        chaotic_region = lyap_data["lyapunov"] > 0
        if np.sum(chaotic_region) > 5:
            X = lyap_data["r"][chaotic_region].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_region]

            logger.info("  Running PySR: lambda = f(r) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["r_"],
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
        lyapunov_theory=lyap_data["lyapunov_theory"],
    )

    return results
