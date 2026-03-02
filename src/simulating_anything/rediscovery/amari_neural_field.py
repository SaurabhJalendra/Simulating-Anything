"""Amari neural field rediscovery.

Targets:
- Bump stability analysis: vary h (resting level), find bump existence threshold
- Mexican hat kernel: show excitation/inhibition balance
- Bump width vs sigma_e ratio
- Bump amplitude dependence on parameters
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.amari_neural_field import (
    AmariNeuralFieldSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _default_config(
    h: float = -0.5,
    sigma_e: float = 1.0,
    sigma_i: float = 3.0,
    A: float = 5.0,
    B: float = 2.0,
    dt: float = 0.02,
    n_steps: int = 3000,
    N: int = 128,
) -> SimulationConfig:
    """Build a default Amari configuration."""
    return SimulationConfig(
        domain=Domain.AMARI_NEURAL_FIELD,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "tau": 1.0,
            "A": A,
            "B": B,
            "sigma_e": sigma_e,
            "sigma_i": sigma_i,
            "beta": 50.0,
            "theta": 0.5,
            "h": h,
            "N": float(N),
            "L_half": 10.0,
        },
    )


def generate_bump_existence_data(
    n_h: int = 25,
    n_steps: int = 3000,
    dt: float = 0.02,
) -> dict[str, np.ndarray]:
    """Sweep resting level h and measure bump width and amplitude.

    The Amari model transitions from no-bump (flat) to stable bump as h
    increases past a critical threshold.
    """
    h_values = np.linspace(-2.0, 0.5, n_h)
    widths = []
    amplitudes = []

    for h_val in h_values:
        config = _default_config(h=h_val, dt=dt, n_steps=n_steps)
        sim = AmariNeuralFieldSimulation(config)
        sim.reset(seed=42)

        for _ in range(n_steps):
            sim.step()

        widths.append(sim.compute_bump_width())
        amplitudes.append(sim.compute_bump_amplitude())

    return {
        "h_values": h_values,
        "bump_widths": np.array(widths),
        "bump_amplitudes": np.array(amplitudes),
    }


def generate_sigma_ratio_data(
    n_sigma: int = 15,
    n_steps: int = 3000,
    dt: float = 0.02,
) -> dict[str, np.ndarray]:
    """Sweep excitatory spatial scale sigma_e and measure bump width.

    Bump width should scale with sigma_e when a stable bump exists.
    """
    sigma_e_values = np.linspace(0.5, 3.0, n_sigma)
    sigma_i = 3.0
    widths = []

    for sig_e in sigma_e_values:
        config = _default_config(sigma_e=sig_e, sigma_i=sigma_i, dt=dt, n_steps=n_steps)
        sim = AmariNeuralFieldSimulation(config)
        sim.reset(seed=42)

        for _ in range(n_steps):
            sim.step()

        widths.append(sim.compute_bump_width())

    return {
        "sigma_e_values": sigma_e_values,
        "sigma_ratios": sigma_e_values / sigma_i,
        "bump_widths": np.array(widths),
    }


def generate_kernel_data(
    sigma_e: float = 1.0,
    sigma_i: float = 3.0,
    A: float = 5.0,
    B: float = 2.0,
) -> dict[str, np.ndarray]:
    """Generate Mexican hat kernel profile for visualization."""
    config = _default_config(sigma_e=sigma_e, sigma_i=sigma_i, A=A, B=B)
    sim = AmariNeuralFieldSimulation(config)
    sim.reset(seed=42)

    kernel = sim.get_kernel()
    return {
        "x": sim.x.copy(),
        "kernel": kernel,
        "excitatory": A * np.exp(-np.abs(sim.x) / sigma_e),
        "inhibitory": -B * np.exp(-np.abs(sim.x) / sigma_i),
    }


def run_amari_neural_field_rediscovery(
    output_dir: str | Path = "output/rediscovery/amari_neural_field",
    n_iterations: int = 40,
) -> dict:
    """Run Amari neural field rediscovery pipeline.

    Analyzes bump formation, stability, and parameter dependence.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "amari_neural_field",
        "targets": {
            "bump_existence": "bump formation threshold in h",
            "sigma_scaling": "bump width vs sigma_e/sigma_i ratio",
            "kernel_structure": "Mexican hat connectivity",
        },
    }

    # --- Part 1: Bump existence threshold ---
    logger.info("Part 1: Bump existence threshold sweep...")
    existence_data = generate_bump_existence_data(n_h=25, n_steps=3000)

    # Find threshold: h value where bump width first becomes positive
    has_bump = existence_data["bump_widths"] > 0.5
    if np.any(has_bump):
        idx = np.argmax(has_bump)
        h_threshold = existence_data["h_values"][max(0, idx - 1)]
        results["bump_existence"] = {
            "h_threshold_estimate": float(h_threshold),
            "max_bump_width": float(np.max(existence_data["bump_widths"])),
            "max_bump_amplitude": float(np.max(existence_data["bump_amplitudes"])),
            "n_bump_states": int(np.sum(has_bump)),
        }
        logger.info(f"  Bump existence threshold: h ~ {h_threshold:.4f}")
    else:
        results["bump_existence"] = {
            "note": "No bumps detected in sweep range",
        }

    # --- Part 2: Sigma ratio scaling ---
    logger.info("Part 2: Bump width vs sigma_e scaling...")
    sigma_data = generate_sigma_ratio_data(n_sigma=15, n_steps=3000)

    valid_widths = sigma_data["bump_widths"] > 0.5
    if np.sum(valid_widths) >= 3:
        # Fit linear relationship: width = a * sigma_e + b
        sig_valid = sigma_data["sigma_e_values"][valid_widths]
        w_valid = sigma_data["bump_widths"][valid_widths]
        coeffs = np.polyfit(sig_valid, w_valid, 1)

        # Correlation
        corr = np.corrcoef(sig_valid, w_valid)[0, 1] if len(sig_valid) > 2 else 0.0

        results["sigma_scaling"] = {
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1]),
            "correlation": float(corr),
            "n_valid_points": int(np.sum(valid_widths)),
            "sigma_e_range": [
                float(sigma_data["sigma_e_values"][0]),
                float(sigma_data["sigma_e_values"][-1]),
            ],
        }
        logger.info(
            f"  Width ~ {coeffs[0]:.3f} * sigma_e + {coeffs[1]:.3f} "
            f"(R={corr:.4f})"
        )
    else:
        results["sigma_scaling"] = {
            "note": "Not enough bump states for scaling analysis",
        }

    # --- Part 3: Kernel structure ---
    logger.info("Part 3: Mexican hat kernel analysis...")
    kernel_data = generate_kernel_data()
    kernel = kernel_data["kernel"]

    # Verify Mexican hat structure: positive center, negative surround
    center_idx = len(kernel) // 2
    # For periodic grid centered at -L_half, the center (x=0) is at N//2
    # But our grid starts at -L_half, so x=0 is near the middle
    center_region = kernel[center_idx - 5:center_idx + 5]
    surround_region = np.concatenate([kernel[:10], kernel[-10:]])

    results["kernel_structure"] = {
        "center_positive": bool(np.mean(center_region) > 0),
        "surround_negative": bool(np.mean(surround_region) < 0),
        "peak_value": float(np.max(kernel)),
        "min_value": float(np.min(kernel)),
        "zero_crossing_count": int(
            np.sum(np.diff(np.sign(kernel)) != 0)
        ),
    }
    logger.info(
        f"  Kernel: peak={np.max(kernel):.4f}, "
        f"min={np.min(kernel):.4f}, "
        f"center+={np.mean(center_region) > 0}, "
        f"surround-={np.mean(surround_region) < 0}"
    )

    # --- Part 4: Bump stability check ---
    logger.info("Part 4: Bump stability analysis...")
    config = _default_config(h=-0.5, n_steps=3000)
    sim = AmariNeuralFieldSimulation(config)
    sim.reset(seed=42)

    for _ in range(3000):
        sim.step()

    stable = sim.is_bump_stable(n_check_steps=500)
    results["bump_stability"] = {
        "is_stable": bool(stable),
        "final_width": float(sim.compute_bump_width()),
        "final_amplitude": float(sim.compute_bump_amplitude()),
    }
    logger.info(f"  Bump stable: {stable}, width={sim.compute_bump_width():.3f}")

    # --- Part 5: PySR attempt for bump width scaling ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        if np.sum(valid_widths) >= 5:
            X = sigma_data["sigma_e_values"][valid_widths].reshape(-1, 1)
            y = sigma_data["bump_widths"][valid_widths]

            logger.info("  Running PySR: bump_width = f(sigma_e)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["sig_e"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square", "exp"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["pysr_width"] = {
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
                results["pysr_width"]["best"] = best.expression
                results["pysr_width"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_width"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "existence_sweep.npz",
        h_values=existence_data["h_values"],
        bump_widths=existence_data["bump_widths"],
        bump_amplitudes=existence_data["bump_amplitudes"],
    )

    np.savez(
        output_path / "sigma_sweep.npz",
        sigma_e_values=sigma_data["sigma_e_values"],
        bump_widths=sigma_data["bump_widths"],
    )

    return results
