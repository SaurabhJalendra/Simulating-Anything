"""Swift-Hohenberg rediscovery: pattern wavelength, amplitude scaling, subcritical.

Targets:
- Pattern wavelength lambda ~ 2*pi for all r > 0 (from k_c = 1)
- Amplitude scales as A ~ sqrt(r) near onset (supercritical, g=0)
- Subcritical bifurcation character for g != 0
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.swift_hohenberg import SwiftHohenbergSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_amplitude_vs_r_data(
    n_r_values: int = 20,
    r_min: float = -0.5,
    r_max: float = 2.0,
    n_steps: int = 5000,
    dt: float = 0.1,
    N: int = 256,
    L: float = 20.0 * np.pi,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Sweep control parameter r and measure saturated pattern amplitude.

    For each r value, runs the simulation to steady state and measures the
    pattern amplitude A = sqrt(2 * var(u)).

    Args:
        n_r_values: Number of r values to sweep.
        r_min: Minimum r value.
        r_max: Maximum r value.
        n_steps: Simulation steps per run.
        dt: Time step.
        N: Number of grid points.
        L: Domain length.
        seed: Random seed.

    Returns:
        Dict with 'r_values', 'amplitudes', and 'wavelengths' arrays.
    """
    r_values = np.linspace(r_min, r_max, n_r_values)
    amplitudes = np.zeros(n_r_values)
    wavelengths = np.zeros(n_r_values)

    for i, r_val in enumerate(r_values):
        config = SimulationConfig(
            domain=Domain.SWIFT_HOHENBERG,  # Placeholder until Domain.SWIFT_HOHENBERG exists
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": float(r_val),
                "g": 0.0,
                "N": float(N),
                "L": L,
                "init_amplitude": 0.01,
            },
            seed=seed,
        )
        sim = SwiftHohenbergSimulation(config)
        sim.reset(seed=seed)

        for _ in range(n_steps):
            sim.step()

        amplitudes[i] = sim.pattern_amplitude()
        wavelengths[i] = sim.pattern_wavelength()

        if (i + 1) % 5 == 0:
            logger.info(
                f"  r={r_val:.3f}: A={amplitudes[i]:.4f}, "
                f"lambda={wavelengths[i]:.4f}"
            )

    return {
        "r_values": r_values,
        "amplitudes": amplitudes,
        "wavelengths": wavelengths,
    }


def generate_wavelength_data(
    n_r_values: int = 15,
    r_min: float = 0.1,
    r_max: float = 2.0,
    n_steps: int = 5000,
    dt: float = 0.1,
    N: int = 256,
    L: float = 20.0 * np.pi,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Measure pattern wavelength for various r > 0 values.

    The dominant wavelength should be approximately 2*pi for all r > 0,
    corresponding to the critical wavenumber k_c = 1.

    Returns:
        Dict with 'r_values' and 'wavelengths' arrays.
    """
    r_values = np.linspace(r_min, r_max, n_r_values)
    wavelengths = np.zeros(n_r_values)

    for i, r_val in enumerate(r_values):
        config = SimulationConfig(
            domain=Domain.SWIFT_HOHENBERG,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": float(r_val),
                "g": 0.0,
                "N": float(N),
                "L": L,
                "init_amplitude": 0.01,
            },
            seed=seed,
        )
        sim = SwiftHohenbergSimulation(config)
        sim.reset(seed=seed)

        for _ in range(n_steps):
            sim.step()

        wavelengths[i] = sim.pattern_wavelength()

    return {
        "r_values": r_values,
        "wavelengths": wavelengths,
    }


def generate_g_sweep_data(
    n_g_values: int = 15,
    g_min: float = 0.0,
    g_max: float = 2.0,
    r_val: float = 0.5,
    n_steps: int = 5000,
    dt: float = 0.1,
    N: int = 256,
    L: float = 20.0 * np.pi,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Sweep the quadratic coupling g to detect subcritical behavior.

    For g=0 the bifurcation is supercritical (smooth onset). For g != 0 the
    bifurcation becomes subcritical (hysteresis, finite-amplitude jump).

    Returns:
        Dict with 'g_values', 'amplitudes', and 'wavelengths' arrays.
    """
    g_values = np.linspace(g_min, g_max, n_g_values)
    amplitudes = np.zeros(n_g_values)
    wavelengths = np.zeros(n_g_values)

    for i, g_val in enumerate(g_values):
        config = SimulationConfig(
            domain=Domain.SWIFT_HOHENBERG,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": r_val,
                "g": float(g_val),
                "N": float(N),
                "L": L,
                "init_amplitude": 0.01,
            },
            seed=seed,
        )
        sim = SwiftHohenbergSimulation(config)
        sim.reset(seed=seed)

        for _ in range(n_steps):
            sim.step()

        amplitudes[i] = sim.pattern_amplitude()
        wavelengths[i] = sim.pattern_wavelength()

    return {
        "g_values": g_values,
        "amplitudes": amplitudes,
        "wavelengths": wavelengths,
    }


def run_swift_hohenberg_rediscovery(
    output_dir: str | Path = "output/rediscovery/swift_hohenberg",
    n_iterations: int = 40,
) -> dict:
    """Run the full Swift-Hohenberg rediscovery.

    1. Sweep r from -0.5 to 2.0 and measure amplitude vs r
    2. Verify wavelength ~ 2*pi for all r > 0
    3. Sweep g and detect subcritical behavior
    4. PySR: find A(r) scaling, target sqrt(r) near onset

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "swift_hohenberg",
        "targets": {
            "wavelength": "lambda ~ 2*pi (k_c = 1)",
            "amplitude_scaling": "A ~ sqrt(r) near onset",
            "subcritical": "g != 0 changes bifurcation character",
        },
    }

    # --- Part 1: Amplitude vs r ---
    logger.info("Part 1: Amplitude vs control parameter r...")
    amp_data = generate_amplitude_vs_r_data(
        n_r_values=20, r_min=-0.5, r_max=2.0, n_steps=5000, dt=0.1,
    )

    r_vals = amp_data["r_values"]
    amps = amp_data["amplitudes"]

    # Theoretical: A ~ sqrt(r) for r > 0 (supercritical, g=0)
    positive_mask = r_vals > 0.05
    if np.sum(positive_mask) >= 3:
        r_pos = r_vals[positive_mask]
        a_pos = amps[positive_mask]
        sqrt_r = np.sqrt(r_pos)

        # Linear fit: A = c * sqrt(r)
        valid = (a_pos > 0) & np.isfinite(a_pos)
        if np.sum(valid) >= 3:
            c_fit = np.sum(a_pos[valid] * sqrt_r[valid]) / np.sum(sqrt_r[valid] ** 2)
            a_pred = c_fit * sqrt_r[valid]
            ss_res = np.sum((a_pos[valid] - a_pred) ** 2)
            ss_tot = np.sum((a_pos[valid] - np.mean(a_pos[valid])) ** 2)
            r2_sqrt = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            results["amplitude_scaling"] = {
                "n_points": int(np.sum(valid)),
                "coefficient": float(c_fit),
                "r_squared_sqrt_r": float(r2_sqrt),
                "theoretical_scaling": "A ~ sqrt(r)",
            }
            logger.info(
                f"  A = {c_fit:.4f} * sqrt(r), R2 = {r2_sqrt:.6f}"
            )
        else:
            results["amplitude_scaling"] = {"error": "Too few valid amplitude points"}
    else:
        results["amplitude_scaling"] = {"error": "Too few r > 0 points"}

    # For r < 0, amplitude should be near zero (decays)
    neg_mask = r_vals < -0.1
    if np.sum(neg_mask) > 0:
        max_neg_amp = float(np.max(amps[neg_mask]))
        results["negative_r"] = {
            "max_amplitude_for_r_negative": max_neg_amp,
            "decays_to_zero": bool(max_neg_amp < 0.1),
        }
        logger.info(f"  r < 0 max amplitude: {max_neg_amp:.6f}")

    # --- Part 2: Wavelength verification ---
    logger.info("Part 2: Wavelength verification...")
    wl_data = generate_wavelength_data(
        n_r_values=15, r_min=0.1, r_max=2.0, n_steps=5000, dt=0.1,
    )

    wavelengths = wl_data["wavelengths"]
    theoretical_wl = 2 * np.pi

    valid_wl = np.isfinite(wavelengths) & (wavelengths > 0)
    if np.sum(valid_wl) > 0:
        mean_wl = float(np.mean(wavelengths[valid_wl]))
        std_wl = float(np.std(wavelengths[valid_wl]))
        rel_error = abs(mean_wl - theoretical_wl) / theoretical_wl

        results["wavelength"] = {
            "n_valid": int(np.sum(valid_wl)),
            "mean_wavelength": mean_wl,
            "std_wavelength": std_wl,
            "theoretical": float(theoretical_wl),
            "relative_error": float(rel_error),
        }
        logger.info(
            f"  Mean wavelength: {mean_wl:.4f} "
            f"(theory: {theoretical_wl:.4f}, error: {rel_error:.4f})"
        )
    else:
        results["wavelength"] = {"error": "No valid wavelength measurements"}

    # --- Part 3: g sweep (subcritical behavior) ---
    logger.info("Part 3: Quadratic coupling g sweep...")
    g_data = generate_g_sweep_data(
        n_g_values=15, g_min=0.0, g_max=2.0, r_val=0.5, n_steps=5000, dt=0.1,
    )

    g_vals = g_data["g_values"]
    g_amps = g_data["amplitudes"]

    # With g > 0 the amplitude should generally be larger (subcritical effect)
    amp_g0 = float(g_amps[0]) if len(g_amps) > 0 else 0.0
    amp_g_max = float(g_amps[-1]) if len(g_amps) > 0 else 0.0

    results["g_sweep"] = {
        "n_points": len(g_vals),
        "amplitude_at_g0": amp_g0,
        "amplitude_at_g_max": amp_g_max,
        "amplitudes": [float(a) for a in g_amps],
        "g_values": [float(g) for g in g_vals],
        "amplitude_increases_with_g": bool(amp_g_max > amp_g0),
    }
    logger.info(
        f"  A(g=0)={amp_g0:.4f}, A(g={g_vals[-1]:.1f})={amp_g_max:.4f}"
    )

    # --- Part 4: PySR symbolic regression ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Fit A(r) for r > 0
        if np.sum(positive_mask) >= 5:
            r_fit = r_vals[positive_mask]
            a_fit = amps[positive_mask]
            valid_fit = (a_fit > 0) & np.isfinite(a_fit)

            if np.sum(valid_fit) >= 5:
                X = r_fit[valid_fit].reshape(-1, 1)
                y = a_fit[valid_fit]

                logger.info("Running PySR: A(r) = f(r)...")
                discoveries = run_symbolic_regression(
                    X, y,
                    variable_names=["r_"],
                    n_iterations=n_iterations,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sqrt", "square"],
                    max_complexity=10,
                    populations=15,
                    population_size=30,
                )
                results["amplitude_pysr"] = {
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
                    results["amplitude_pysr"]["best"] = best.expression
                    results["amplitude_pysr"]["best_r2"] = (
                        best.evidence.fit_r_squared
                    )
                    logger.info(
                        f"  Best: {best.expression} "
                        f"(R2={best.evidence.fit_r_squared:.6f})"
                    )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["amplitude_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
