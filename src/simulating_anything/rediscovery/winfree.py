"""Winfree pulse-coupled oscillators rediscovery.

Targets:
- Synchronization transition: order parameter r(kappa) vs coupling
- Critical coupling kappa_c for coherence onset
- Comparison with Kuramoto model (Winfree has sharper transition)
- Finite-size scaling: N-dependence of transition
- Amplitude death detection at very strong coupling
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.winfree import WinfreeSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_sync_transition_data(
    n_kappa: int = 40,
    N: int = 100,
    n_trials: int = 5,
    n_transient: int = 5000,
    n_measure: int = 2000,
    dt: float = 0.01,
    delta: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate order parameter r vs coupling kappa data.

    Sweeps kappa from near zero to well above transition and measures
    steady-state order parameter for each value.

    Args:
        n_kappa: Number of kappa values to sweep.
        N: Number of oscillators.
        n_trials: Number of independent trials per kappa.
        n_transient: Transient steps to discard.
        n_measure: Steps to average over.
        dt: Integration timestep.
        delta: Frequency spread (Lorentzian half-width).

    Returns:
        Dict with 'kappa', 'r_mean', 'r_std', 'N', 'n_trials' arrays.
    """
    kappa_values = np.linspace(0.01, 1.0, n_kappa)
    r_mean = []
    r_std = []

    for i, kappa in enumerate(kappa_values):
        r_trials = []
        for trial in range(n_trials):
            config = SimulationConfig(
                domain=Domain.WINFREE,  # Placeholder domain
                dt=dt,
                n_steps=n_transient + n_measure,
                parameters={
                    "N": float(N),
                    "kappa": kappa,
                    "omega0": 1.0,
                    "delta": delta,
                },
            )
            sim = WinfreeSimulation(config)
            r = sim.measure_steady_state_r(
                n_transient_steps=n_transient,
                n_measure_steps=n_measure,
                seed=42 + trial,
            )
            r_trials.append(r)

        r_mean.append(np.mean(r_trials))
        r_std.append(np.std(r_trials))

        if (i + 1) % 10 == 0:
            logger.info(
                f"  kappa={kappa:.3f}: r={np.mean(r_trials):.4f} "
                f"+/- {np.std(r_trials):.4f}"
            )

    return {
        "kappa": kappa_values,
        "r_mean": np.array(r_mean),
        "r_std": np.array(r_std),
        "N": N,
        "n_trials": n_trials,
        "delta": delta,
    }


def generate_finite_size_data(
    N_values: list[int] | None = None,
    kappa: float = 0.5,
    n_trials: int = 10,
    dt: float = 0.01,
    delta: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate order parameter vs system size N at fixed kappa.

    Args:
        N_values: List of oscillator counts to test.
        kappa: Fixed coupling strength (should be above kappa_c).
        n_trials: Number of independent trials per N.
        dt: Integration timestep.
        delta: Frequency spread.

    Returns:
        Dict with 'N', 'r', 'kappa' arrays/values.
    """
    if N_values is None:
        N_values = [10, 20, 50, 100, 200, 500]

    all_N = []
    all_r = []

    for N in N_values:
        r_trials = []
        for trial in range(n_trials):
            config = SimulationConfig(
                domain=Domain.WINFREE,
                dt=dt,
                n_steps=8000,
                parameters={
                    "N": float(N),
                    "kappa": kappa,
                    "omega0": 1.0,
                    "delta": delta,
                },
            )
            sim = WinfreeSimulation(config)
            r = sim.measure_steady_state_r(
                n_transient_steps=5000,
                n_measure_steps=3000,
                seed=100 + trial,
            )
            r_trials.append(r)

        all_N.append(N)
        all_r.append(np.mean(r_trials))
        logger.info(f"  N={N}: r={np.mean(r_trials):.4f}")

    return {
        "N": np.array(all_N),
        "r": np.array(all_r),
        "kappa": kappa,
    }


def generate_kuramoto_comparison_data(
    n_kappa: int = 20,
    N: int = 100,
    n_trials: int = 5,
    n_transient: int = 5000,
    n_measure: int = 2000,
    dt: float = 0.01,
    delta: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate Winfree r(kappa) data for comparison with Kuramoto.

    Uses a wider kappa range to capture the full transition curve.

    Args:
        n_kappa: Number of kappa values.
        N: Number of oscillators.
        n_trials: Independent trials per kappa.
        n_transient: Transient steps.
        n_measure: Measurement steps.
        dt: Integration timestep.
        delta: Frequency spread.

    Returns:
        Dict with 'kappa', 'r_mean', 'freq_spread_mean' arrays.
    """
    kappa_values = np.linspace(0.01, 2.0, n_kappa)
    r_mean = []
    freq_spread_mean = []

    for kappa in kappa_values:
        r_trials = []
        fs_trials = []
        for trial in range(n_trials):
            config = SimulationConfig(
                domain=Domain.WINFREE,
                dt=dt,
                n_steps=n_transient + n_measure,
                parameters={
                    "N": float(N),
                    "kappa": kappa,
                    "omega0": 1.0,
                    "delta": delta,
                },
            )
            sim = WinfreeSimulation(config)
            r = sim.measure_steady_state_r(
                n_transient_steps=n_transient,
                n_measure_steps=n_measure,
                seed=200 + trial,
            )
            r_trials.append(r)

            # Also measure frequency spread
            fs = sim.measure_steady_state_freq_spread(
                n_transient_steps=n_transient,
                n_measure_steps=n_measure,
                seed=200 + trial,
            )
            fs_trials.append(fs)

        r_mean.append(np.mean(r_trials))
        freq_spread_mean.append(np.mean(fs_trials))

    return {
        "kappa": kappa_values,
        "r_mean": np.array(r_mean),
        "freq_spread_mean": np.array(freq_spread_mean),
    }


def run_winfree_rediscovery(
    output_dir: str | Path = "output/rediscovery/winfree",
    n_iterations: int = 40,
) -> dict:
    """Run Winfree model rediscovery pipeline.

    1. Sweep coupling kappa, measure order parameter r
    2. Estimate critical coupling kappa_c
    3. Finite-size scaling analysis
    4. Comparison with Kuramoto model characteristics

    Args:
        output_dir: Directory to save results.
        n_iterations: Number of PySR iterations (if PySR available).

    Returns:
        Results dict with all rediscovery data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "winfree",
        "targets": {
            "sync_transition": "r(kappa) from 0 to ~1 as kappa increases",
            "critical_coupling": "kappa_c for onset of coherence",
            "kuramoto_comparison": "Sharper transition than Kuramoto",
        },
    }

    # --- Part 1: Synchronization transition ---
    logger.info("Part 1: Synchronization transition (r vs kappa)...")
    data = generate_sync_transition_data(
        n_kappa=40, N=100, n_trials=5, dt=0.01, delta=0.1,
    )

    # Estimate kappa_c: first kappa where r > threshold
    threshold = 0.3
    above = data["r_mean"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        kappa_c_est = float(data["kappa"][max(0, idx - 1)])
        results["kappa_c_estimate"] = kappa_c_est
        logger.info(f"  kappa_c estimate: {kappa_c_est:.4f}")
    else:
        logger.warning("  Could not detect synchronization transition")

    results["sync_transition"] = {
        "n_kappa": len(data["kappa"]),
        "kappa_range": [float(data["kappa"].min()), float(data["kappa"].max())],
        "max_r": float(np.max(data["r_mean"])),
        "delta": float(data["delta"]),
    }

    # PySR: find r(kappa) relationship
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use supercritical data only
        supercritical = data["r_mean"] > 0.1
        if np.sum(supercritical) > 5:
            X = data["kappa"][supercritical].reshape(-1, 1)
            y = data["r_mean"][supercritical]

            logger.info("  Running PySR: r = f(kappa)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["kap"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square", "cos"],
                max_complexity=12,
                populations=20,
                population_size=40,
            )
            results["order_param_pysr"] = {
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
                results["order_param_pysr"]["best"] = best.expression
                results["order_param_pysr"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["order_param_pysr"] = {"error": str(e)}

    # --- Part 2: Finite-size scaling ---
    logger.info("Part 2: Finite-size scaling...")
    fs_data = generate_finite_size_data(kappa=0.5, n_trials=5, delta=0.1)
    results["finite_size"] = {
        "N_values": fs_data["N"].tolist(),
        "r_values": fs_data["r"].tolist(),
        "kappa": fs_data["kappa"],
    }

    # --- Part 3: Kuramoto comparison ---
    logger.info("Part 3: Kuramoto comparison (wider kappa range)...")
    comp_data = generate_kuramoto_comparison_data(
        n_kappa=20, N=100, n_trials=3, dt=0.01, delta=0.1,
    )
    results["kuramoto_comparison"] = {
        "kappa_range": [
            float(comp_data["kappa"].min()),
            float(comp_data["kappa"].max()),
        ],
        "max_r": float(np.max(comp_data["r_mean"])),
        "min_freq_spread": float(np.min(comp_data["freq_spread_mean"])),
    }

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "sync_transition.npz",
        kappa=data["kappa"],
        r_mean=data["r_mean"],
        r_std=data["r_std"],
    )
    np.savez(
        output_path / "finite_size.npz",
        N=fs_data["N"],
        r=fs_data["r"],
    )

    return results
