"""BZ spiral (2D Oregonator PDE) rediscovery.

Targets:
- Spiral wave formation from broken wavefront initial conditions
- Spiral wavelength and frequency measurements
- Wavelength scaling with sqrt(D_u) and eps
- Excitable medium behavior (suprathreshold propagation)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.bz_spiral import BZSpiralSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    eps: float = 0.01,
    f: float = 1.0,
    q: float = 0.002,
    D_u: float = 1.0,
    D_v: float = 0.0,
    Nx: int = 64,
    Ny: int = 64,
    dx: float = 0.5,
    dt: float = 0.01,
    n_steps: int = 1000,
    seed: int = 42,
) -> SimulationConfig:
    """Build a SimulationConfig for BZ spiral simulation."""
    return SimulationConfig(
        domain=Domain.BZ_SPIRAL,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        parameters={
            "eps": eps,
            "f": f,
            "q": q,
            "D_u": D_u,
            "D_v": D_v,
            "Nx": float(Nx),
            "Ny": float(Ny),
            "dx": dx,
        },
    )


def generate_spiral_data(
    eps: float = 0.01,
    f: float = 1.0,
    q: float = 0.002,
    D_u: float = 1.0,
    Nx: int = 64,
    Ny: int = 64,
    dx: float = 0.5,
    dt: float = 0.01,
    n_warmup: int = 2000,
    n_measure: int = 3000,
) -> dict:
    """Generate spiral wave data for analysis.

    Runs the simulation through a warmup phase to establish the spiral,
    then measures spiral tip position, wavelength, and frequency.

    Args:
        eps: Timescale ratio.
        f: Stoichiometric factor.
        q: Excitability parameter.
        D_u: Activator diffusion coefficient.
        Nx: Grid x-dimension.
        Ny: Grid y-dimension.
        dx: Grid spacing.
        dt: Timestep.
        n_warmup: Steps to establish spiral.
        n_measure: Steps to measure properties.

    Returns:
        Dict with spiral properties.
    """
    config = _make_config(
        eps=eps, f=f, q=q, D_u=D_u, Nx=Nx, Ny=Ny, dx=dx, dt=dt,
        n_steps=n_warmup + n_measure,
    )
    sim = BZSpiralSimulation(config)
    sim.reset()

    # Warmup to establish spiral
    for _ in range(n_warmup):
        sim.step()

    # Measure spiral wavelength from FFT of u field
    u_field = sim.get_u_field()
    wavelength = _measure_wavelength(u_field, dx)

    # Measure frequency by tracking tip
    frequency = sim.compute_spiral_frequency(n_steps=n_measure)

    # Check u field has spatial structure (not uniform)
    u_range = float(np.max(u_field) - np.min(u_field))

    return {
        "eps": eps,
        "f": f,
        "q": q,
        "D_u": D_u,
        "dx": dx,
        "wavelength": wavelength,
        "frequency": frequency,
        "u_range": u_range,
        "u_mean": float(np.mean(u_field)),
        "has_pattern": u_range > 0.1,
    }


def _measure_wavelength(u_field: np.ndarray, dx: float) -> float:
    """Measure dominant spatial wavelength from 2D u field via FFT.

    Args:
        u_field: 2D activator field.
        dx: Grid spacing.

    Returns:
        Dominant wavelength in physical units, or 0.0 if no pattern.
    """
    centered = u_field - np.mean(u_field)
    if np.std(centered) < 1e-10:
        return 0.0

    power = np.abs(np.fft.fft2(centered)) ** 2
    power_shifted = np.fft.fftshift(power)

    nx, ny = u_field.shape
    cx, cy = nx // 2, ny // 2

    Y, X = np.ogrid[-cx:nx - cx, -cy:ny - cy]
    R = np.sqrt(X ** 2 + Y ** 2).astype(int)
    max_r = min(cx, cy)

    radial_power = np.zeros(max_r)
    for r_val in range(1, max_r):
        mask = R == r_val
        if np.any(mask):
            radial_power[r_val] = np.mean(power_shifted[mask])

    # Find peak wavenumber (skip DC and very low frequencies)
    if max_r > 3 and np.max(radial_power[2:]) > 0:
        peak_r = np.argmax(radial_power[2:]) + 2
        # Convert wavenumber index to physical wavelength
        domain_size = nx * dx
        wavelength = domain_size / peak_r
        return float(wavelength)

    return 0.0


def generate_eps_sweep_data(
    n_eps: int = 8,
    Nx: int = 64,
    Ny: int = 64,
    dx: float = 0.5,
    dt: float = 0.01,
    n_warmup: int = 2000,
) -> dict:
    """Sweep eps and measure spiral wavelength and frequency.

    The spiral wavelength is expected to scale with eps (faster kinetics
    lead to shorter wavelengths).

    Args:
        n_eps: Number of eps values to sweep.
        Nx: Grid x-dimension.
        Ny: Grid y-dimension.
        dx: Grid spacing.
        dt: Timestep.
        n_warmup: Steps to establish spiral.

    Returns:
        Dict with eps values, wavelengths, and frequencies.
    """
    eps_values = np.linspace(0.005, 0.04, n_eps)
    wavelengths = []
    frequencies = []

    for i, eps_val in enumerate(eps_values):
        config = _make_config(
            eps=float(eps_val), Nx=Nx, Ny=Ny, dx=dx, dt=dt,
            n_steps=n_warmup + 1000,
        )
        sim = BZSpiralSimulation(config)
        sim.reset()

        for _ in range(n_warmup):
            sim.step()

        u_field = sim.get_u_field()
        wl = _measure_wavelength(u_field, dx)
        wavelengths.append(wl)

        freq = sim.compute_spiral_frequency(n_steps=1000)
        frequencies.append(freq)

        if (i + 1) % 3 == 0:
            logger.info(
                f"  eps={eps_val:.4f}: wavelength={wl:.2f}, "
                f"frequency={freq:.4f}"
            )

    return {
        "eps": eps_values.tolist(),
        "wavelength": wavelengths,
        "frequency": frequencies,
    }


def run_bz_spiral_rediscovery(
    output_dir: str | Path = "output/rediscovery/bz_spiral",
    n_iterations: int = 40,
) -> dict:
    """Run BZ spiral rediscovery pipeline.

    1. Establish spiral wave and measure properties
    2. Sweep eps to see how spiral properties change
    3. Optionally run PySR on wavelength vs eps

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iterations (if available).

    Returns:
        Results dict with spiral measurements and scaling fits.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "bz_spiral",
        "targets": {
            "spiral_formation": "Spiral wave from broken wavefront",
            "wavelength_scaling": "Wavelength vs eps",
            "excitable_medium": "Suprathreshold propagation",
        },
    }

    # --- Part 1: Spiral formation and measurement ---
    logger.info("Part 1: Establishing spiral wave and measuring properties...")
    spiral_data = generate_spiral_data(
        eps=0.01, D_u=1.0, Nx=64, Ny=64, dx=0.5, dt=0.01,
        n_warmup=2000, n_measure=2000,
    )
    results["spiral_properties"] = spiral_data
    logger.info(
        f"  Wavelength: {spiral_data['wavelength']:.2f}, "
        f"Frequency: {spiral_data['frequency']:.4f}, "
        f"u_range: {spiral_data['u_range']:.3f}"
    )

    # --- Part 2: Eps sweep ---
    logger.info("Part 2: Sweeping eps for wavelength scaling...")
    eps_data = generate_eps_sweep_data(
        n_eps=8, Nx=64, Ny=64, dx=0.5, dt=0.01, n_warmup=2000,
    )
    results["eps_sweep"] = eps_data

    # Analyze scaling: wavelength vs sqrt(eps)
    valid = [
        i for i, wl in enumerate(eps_data["wavelength"])
        if wl > 0
    ]
    if len(valid) >= 3:
        wl_arr = np.array([eps_data["wavelength"][i] for i in valid])
        eps_arr = np.array([eps_data["eps"][i] for i in valid])
        sqrt_eps = np.sqrt(eps_arr)

        corr = float(np.corrcoef(wl_arr, sqrt_eps)[0, 1])
        results["wavelength_scaling"] = {
            "n_valid": len(valid),
            "correlation_lambda_vs_sqrt_eps": corr,
            "mean_wavelength": float(np.mean(wl_arr)),
        }
        logger.info(
            f"  Wavelength vs sqrt(eps) correlation: {corr:.4f} "
            f"({len(valid)} valid points)"
        )

        # --- Part 3: PySR fit ---
        if len(valid) >= 5:
            try:
                from simulating_anything.analysis.symbolic_regression import (
                    run_symbolic_regression,
                )

                X = eps_arr.reshape(-1, 1)
                y = wl_arr

                logger.info("  Running PySR: wavelength = g(eps)...")
                discoveries = run_symbolic_regression(
                    X, y,
                    variable_names=["eps_val"],
                    n_iterations=n_iterations,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sqrt", "square", "log"],
                    max_complexity=12,
                    populations=20,
                    population_size=40,
                )
                results["wavelength_pysr"] = {
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
                    results["wavelength_pysr"]["best"] = best.expression
                    results["wavelength_pysr"]["best_r2"] = (
                        best.evidence.fit_r_squared
                    )
                    logger.info(
                        f"  Best: {best.expression} "
                        f"(R2={best.evidence.fit_r_squared:.6f})"
                    )
            except Exception as e:
                logger.warning(f"PySR failed: {e}")
                results["wavelength_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f_out:
        json.dump(results, f_out, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
