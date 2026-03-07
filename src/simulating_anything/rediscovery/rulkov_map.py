"""Rulkov map rediscovery.

Targets:
- Regime classification: silent, spiking, bursting as function of alpha
- Spiking onset threshold: alpha_c ~ 4.0
- Firing rate vs alpha
- Bifurcation diagram in alpha
- Lyapunov exponent spectrum
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.rulkov_map import RulkovMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    alpha: float = 4.1,
    mu: float = 0.001,
    sigma: float = -1.0,
    x_0: float = -1.0,
    y_0: float = -3.0,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.RULKOV_MAP,
        dt=1.0,
        n_steps=1000,
        parameters={
            "alpha": alpha,
            "mu": mu,
            "sigma": sigma,
            "x_0": x_0,
            "y_0": y_0,
        },
    )


def generate_alpha_sweep_data(
    n_alpha: int = 40,
    alpha_min: float = 2.0,
    alpha_max: float = 6.0,
    n_steps: int = 10000,
    n_transient: int = 2000,
) -> dict[str, np.ndarray | list[str]]:
    """Sweep alpha and collect firing rates and regimes."""
    alpha_values = np.linspace(alpha_min, alpha_max, n_alpha)
    sim = RulkovMapSimulation(_make_config(alpha=4.1))
    return sim.alpha_sweep(alpha_values, n_steps, n_transient)


def generate_bifurcation_data(
    n_alpha: int = 200,
    alpha_min: float = 2.0,
    alpha_max: float = 6.0,
) -> dict[str, np.ndarray]:
    """Generate bifurcation diagram data (x vs alpha)."""
    alpha_values = np.linspace(alpha_min, alpha_max, n_alpha)
    sim = RulkovMapSimulation(_make_config(alpha=4.1))
    return sim.bifurcation_diagram(alpha_values, n_transient=2000, n_plot=200)


def generate_lyapunov_data(
    n_alpha: int = 50,
    alpha_min: float = 2.0,
    alpha_max: float = 6.0,
) -> dict[str, np.ndarray]:
    """Generate Lyapunov exponent vs alpha data."""
    alpha_values = np.linspace(alpha_min, alpha_max, n_alpha)
    lyapunovs = []

    for i, alpha in enumerate(alpha_values):
        config = _make_config(alpha=alpha)
        sim = RulkovMapSimulation(config)
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=1000)
        lyapunovs.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  alpha={alpha:.3f}: Lyapunov={lam:.4f}")

    return {
        "alpha": alpha_values,
        "lyapunov": np.array(lyapunovs),
    }


def estimate_spiking_onset(
    sweep_data: dict[str, np.ndarray | list[str]],
) -> float | None:
    """Estimate the critical alpha where spiking first appears.

    Returns the smallest alpha value where the regime is not 'silent',
    or None if all regimes are silent.
    """
    alphas = sweep_data["alpha"]
    regimes = sweep_data["regime"]
    for i, regime in enumerate(regimes):
        if regime != "silent":
            return float(alphas[i])
    return None


def run_rulkov_map_rediscovery(
    output_dir: str | Path = "output/rediscovery/rulkov_map",
    n_iterations: int = 40,
) -> dict:
    """Run Rulkov map rediscovery.

    Discovers:
    - Regime diagram (silent / spiking / bursting) vs alpha
    - Spiking onset threshold alpha_c
    - Firing rate curve f(alpha)
    - Bifurcation diagram
    - Lyapunov exponent spectrum
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "rulkov_map",
        "targets": {
            "spiking_onset": "alpha_c ~ 4.0",
            "regimes": "silent / spiking / bursting",
            "map_equations": (
                "x_{n+1} = alpha/(1+x^2) + y, "
                "y_{n+1} = y - mu*(x - sigma)"
            ),
        },
    }

    # Part 1: Alpha sweep -- firing rates and regimes
    logger.info("Part 1: Alpha sweep (firing rate + regime classification)...")
    sweep_data = generate_alpha_sweep_data(n_alpha=40)
    results["alpha_sweep"] = {
        "n_alpha": 40,
        "alpha_range": [2.0, 6.0],
        "regimes_found": list(set(sweep_data["regime"])),
        "max_firing_rate": float(np.max(sweep_data["firing_rate"])),
    }

    alpha_c = estimate_spiking_onset(sweep_data)
    if alpha_c is not None:
        results["spiking_onset"] = {
            "alpha_c_estimate": alpha_c,
            "alpha_c_theory": 4.0,
        }
        logger.info(f"  Spiking onset: alpha_c ~ {alpha_c:.3f} (theory: ~4.0)")
    else:
        results["spiking_onset"] = {"alpha_c_estimate": None}
        logger.info("  No spiking detected in sweep range")

    # Count regime types
    regime_counts: dict[str, int] = {}
    for r in sweep_data["regime"]:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    results["regime_counts"] = regime_counts
    logger.info(f"  Regime counts: {regime_counts}")

    np.savez(
        output_path / "alpha_sweep.npz",
        alpha=sweep_data["alpha"],
        firing_rate=sweep_data["firing_rate"],
    )

    # Part 2: Bifurcation diagram
    logger.info("Part 2: Bifurcation diagram...")
    bif_data = generate_bifurcation_data(n_alpha=200)
    results["bifurcation"] = {
        "n_alpha": 200,
        "alpha_range": [2.0, 6.0],
        "x_range": [float(np.min(bif_data["x"])), float(np.max(bif_data["x"]))],
    }
    np.savez(
        output_path / "bifurcation_data.npz",
        alpha=bif_data["alpha"],
        x=bif_data["x"],
    )

    # Part 3: Lyapunov exponent spectrum
    logger.info("Part 3: Lyapunov exponent vs alpha...")
    lyap_data = generate_lyapunov_data(n_alpha=50)

    # Chaos onset (first alpha where lambda > 0)
    chaotic = lyap_data["lyapunov"] > 0
    if np.any(chaotic):
        alpha_chaos = lyap_data["alpha"][np.argmax(chaotic)]
        results["chaos_onset"] = {
            "alpha_estimate": float(alpha_chaos),
        }
        logger.info(f"  Positive Lyapunov onset: alpha ~ {alpha_chaos:.3f}")

    results["lyapunov"] = {
        "max_lyapunov": float(np.max(lyap_data["lyapunov"])),
        "alpha_at_max": float(
            lyap_data["alpha"][np.argmax(lyap_data["lyapunov"])]
        ),
    }

    np.savez(
        output_path / "lyapunov_data.npz",
        alpha=lyap_data["alpha"],
        lyapunov=lyap_data["lyapunov"],
    )

    # Part 4: PySR -- fit firing rate as f(alpha) in active region
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        active = sweep_data["firing_rate"] > 0
        if np.sum(active) > 5:
            X = sweep_data["alpha"][active].reshape(-1, 1)
            y = sweep_data["firing_rate"][active]

            logger.info("  Running PySR: firing_rate = f(alpha)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["a_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "log", "exp"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["pysr_firing_rate"] = {
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
                results["pysr_firing_rate"]["best"] = best.expression
                results["pysr_firing_rate"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_firing_rate"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
