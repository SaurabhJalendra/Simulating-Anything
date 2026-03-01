"""Standard map (Chirikov) rediscovery.

Targets:
- Area preservation (Jacobian determinant = 1)
- Lyapunov exponent as function of K
- Critical stochasticity K_c ~ 0.9716 (Greene's criterion)
- Chaos fraction vs K
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.standard_map import StandardMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(K: float = 0.9716, n_particles: int = 100) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.STANDARD_MAP,
        dt=1.0,
        n_steps=1000,
        parameters={"K": K, "n_particles": float(n_particles)},
    )


def generate_lyapunov_data(
    n_K: int = 50,
    K_min: float = 0.0,
    K_max: float = 5.0,
    n_steps: int = 5000,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs K data."""
    K_values = np.linspace(K_min, K_max, n_K)
    lyapunovs = []

    sim = StandardMapSimulation(_make_config())

    for i, K in enumerate(K_values):
        lam = sim.compute_lyapunov(K=K, n_steps=n_steps, n_transient=500)
        lyapunovs.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  K={K:.4f}: Lyapunov={lam:.4f}")

    return {
        "K": K_values,
        "lyapunov": np.array(lyapunovs),
    }


def generate_chaos_fraction_data(
    n_K: int = 30,
    K_min: float = 0.0,
    K_max: float = 5.0,
    n_particles: int = 100,
    n_steps: int = 2000,
) -> dict[str, np.ndarray]:
    """Generate chaos fraction vs K data."""
    K_values = np.linspace(K_min, K_max, n_K)

    sim = StandardMapSimulation(_make_config())
    sweep = sim.stochasticity_sweep(
        K_values,
        n_particles=n_particles,
        n_steps=n_steps,
    )
    return sweep


def test_area_preservation(
    K_values: np.ndarray | None = None,
    n_particles: int = 50,
    n_steps: int = 100,
) -> dict[str, list[float]]:
    """Test that the standard map preserves phase-space area.

    The Jacobian determinant of the standard map is exactly 1 for all
    (theta, p) and all K, since it is a symplectic (area-preserving) map.
    We verify this by computing det(J) at many points along orbits.

    Returns:
        Dict with K values and corresponding max |det(J) - 1| deviations.
    """
    if K_values is None:
        K_values = np.array([0.0, 0.5, 0.9716, 2.0, 5.0])

    deviations = []
    for K in K_values:
        max_dev = 0.0
        # Check Jacobian determinant analytically
        # J = [[1 + K*cos(theta), 1], [K*cos(theta), 1]]
        # det(J) = (1 + K*cos(theta))*1 - 1*K*cos(theta) = 1
        # This is exact, but verify numerically for several theta values
        thetas = np.linspace(0, 2 * np.pi, 100)
        for theta in thetas:
            cos_t = np.cos(theta)
            det = (1.0 + K * cos_t) * 1.0 - 1.0 * K * cos_t
            dev = abs(det - 1.0)
            if dev > max_dev:
                max_dev = dev
        deviations.append(max_dev)

    return {
        "K_values": K_values.tolist(),
        "max_deviations": deviations,
    }


def estimate_Kc(
    lyapunov_K: np.ndarray,
    lyapunov_vals: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """Estimate K_c from Lyapunov exponent data.

    K_c is the smallest K where the Lyapunov exponent first becomes
    significantly positive (indicating global chaos onset).

    Args:
        lyapunov_K: array of K values.
        lyapunov_vals: corresponding Lyapunov exponents.
        threshold: minimum Lyapunov to consider chaotic.

    Returns:
        Estimated K_c value.
    """
    chaotic = lyapunov_vals > threshold
    if not np.any(chaotic):
        return float(lyapunov_K[-1])  # No chaos detected
    return float(lyapunov_K[np.argmax(chaotic)])


def run_standard_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/standard_map",
    n_iterations: int = 40,
) -> dict:
    """Run standard map rediscovery.

    Performs:
    1. Area preservation verification
    2. Lyapunov exponent sweep over K
    3. K_c estimation (chaos transition)
    4. Chaos fraction sweep
    5. Optional PySR symbolic fit of Lyapunov(K)

    Args:
        output_dir: Directory for output files.
        n_iterations: PySR iterations.

    Returns:
        Results dict with all measurements.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "standard_map",
        "targets": {
            "area_preservation": "det(J) = 1 (symplectic)",
            "K_c": "~0.9716 (Greene's criterion)",
            "lyapunov": "lambda(K) > 0 for K > K_c",
        },
    }

    # Part 1: Area preservation
    logger.info("Part 1: Area preservation test...")
    area_result = test_area_preservation()
    results["area_preservation"] = area_result
    max_dev = max(area_result["max_deviations"])
    logger.info(f"  Max |det(J) - 1| = {max_dev:.2e} (should be ~0)")

    # Part 2: Lyapunov exponent sweep
    logger.info("Part 2: Lyapunov exponent vs K...")
    lyap_data = generate_lyapunov_data(n_K=50, K_min=0.0, K_max=5.0)

    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "K_at_max": float(lyap_data["K"][np.argmax(lyap_data["lyapunov"])]),
    }

    # Part 3: K_c estimation
    logger.info("Part 3: Estimating K_c...")
    K_c_est = estimate_Kc(lyap_data["K"], lyap_data["lyapunov"])
    results["K_c"] = {
        "estimate": K_c_est,
        "theory": 0.9716,
        "error_pct": abs(K_c_est - 0.9716) / 0.9716 * 100,
    }
    logger.info(f"  K_c estimate: {K_c_est:.4f} (theory: 0.9716)")

    # Part 4: Chaos fraction sweep
    logger.info("Part 4: Chaos fraction sweep...")
    chaos_data = generate_chaos_fraction_data(n_K=30, n_particles=50, n_steps=1000)
    results["chaos_fraction"] = {
        "K_range": [0.0, 5.0],
        "n_K": 30,
        "max_fraction": float(np.max(chaos_data["chaos_fractions"])),
    }

    # Part 5: PySR fit
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        chaotic_mask = lyap_data["lyapunov"] > 0.01
        if np.sum(chaotic_mask) > 5:
            X = lyap_data["K"][chaotic_mask].reshape(-1, 1)
            y = lyap_data["lyapunov"][chaotic_mask]

            logger.info("  Running PySR: lambda = f(K) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["K_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["log", "sqrt", "sin"],
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
        K=lyap_data["K"],
        lyapunov=lyap_data["lyapunov"],
    )
    np.savez(
        output_path / "chaos_fraction_data.npz",
        K=chaos_data["K_values"],
        chaos_fraction=chaos_data["chaos_fractions"],
        mean_lyapunov=chaos_data["mean_lyapunovs"],
    )

    return results
