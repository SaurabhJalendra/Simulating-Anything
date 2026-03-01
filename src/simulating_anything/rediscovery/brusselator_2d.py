"""2D Brusselator reaction-diffusion rediscovery.

Targets:
- Turing instability: patterns form when b > 1+a^2 and D_v/D_u large
- Pattern wavelength: lambda ~ 2*pi*sqrt(D_v / (b - 1 - a^2))
- Homogeneous steady state: (u*, v*) = (a, b/a)
- Pattern classification (uniform vs spots vs stripes) in parameter space
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.brusselator_2d import Brusselator2DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    dt: float = 0.01,
    n_steps: int = 5000,
    N_grid: int = 64,
    L_domain: float = 64.0,
    **params: float,
) -> SimulationConfig:
    """Build a SimulationConfig for the 2D Brusselator domain."""
    defaults = {
        "a": 4.5,
        "b": 7.0,
        "D_u": 1.0,
        "D_v": 8.0,
        "N_grid": float(N_grid),
        "L_domain": L_domain,
    }
    defaults.update(params)

    # Check CFL before creating
    D_max = max(defaults.get("D_u", 1.0), defaults.get("D_v", 8.0))
    dx = L_domain / N_grid
    if D_max > 0:
        dt_cfl = dx ** 2 / (4.0 * D_max)
        if dt > dt_cfl:
            dt = 0.9 * dt_cfl  # Back off to 90% of CFL limit

    return SimulationConfig(
        domain=Domain.BRUSSELATOR_2D,
        dt=dt,
        n_steps=n_steps,
        parameters=defaults,
    )


def generate_pattern_data(
    n_b: int = 10,
    n_Dv: int = 6,
    n_steps: int = 10000,
    dt: float = 0.01,
    N_grid: int = 32,
    L_domain: float = 32.0,
    a: float = 4.5,
) -> dict[str, np.ndarray]:
    """Sweep b and D_v to detect Turing pattern formation.

    For each (b, D_v) pair, run the simulation and measure whether spatial
    patterns have formed (via spatial heterogeneity of u).

    Args:
        n_b: Number of b values to sweep.
        n_Dv: Number of D_v values to sweep.
        n_steps: Simulation steps per run.
        dt: Timestep.
        N_grid: Spatial grid points per side.
        L_domain: Domain length.
        a: Production rate parameter.

    Returns:
        Dictionary with b, D_v, heterogeneity, and pattern classification.
    """
    # Sweep b from below to well above 1+a^2
    b_c = 1.0 + a ** 2
    b_values = np.linspace(b_c * 0.5, b_c * 1.5, n_b)
    D_u = 1.0
    D_v_values = np.logspace(0, 1.5, n_Dv)  # 1.0 to ~31.6

    all_b = []
    all_Dv = []
    all_het_u = []
    all_het_v = []
    all_patterned = []

    for i, b in enumerate(b_values):
        for j, D_v in enumerate(D_v_values):
            config = _make_config(
                dt=dt, n_steps=n_steps, N_grid=N_grid, L_domain=L_domain,
                a=a, b=b, D_u=D_u, D_v=D_v,
            )
            sim = Brusselator2DSimulation(config)
            sim.reset(seed=i * 100 + j)

            for _ in range(n_steps):
                sim.step()

            het_u = sim.spatial_heterogeneity_u

            all_b.append(b)
            all_Dv.append(D_v)
            all_het_u.append(het_u)
            all_het_v.append(sim.spatial_heterogeneity_v)
            # Pattern detected if heterogeneity exceeds threshold
            all_patterned.append(het_u > 0.05)

        if (i + 1) % 3 == 0:
            logger.info(f"  b sweep {i + 1}/{n_b} complete")

    return {
        "b": np.array(all_b),
        "D_v": np.array(all_Dv),
        "heterogeneity_u": np.array(all_het_u),
        "heterogeneity_v": np.array(all_het_v),
        "patterned": np.array(all_patterned),
        "a": a,
        "D_u": D_u,
        "b_c_theory": b_c,
    }


def generate_wavelength_data(
    n_Dv: int = 10,
    n_steps: int = 20000,
    dt: float = 0.01,
    N_grid: int = 64,
    L_domain: float = 64.0,
    a: float = 4.5,
    b: float = 25.0,
) -> dict[str, np.ndarray]:
    """Measure dominant Turing wavelength vs D_v.

    The theoretical prediction is:
        lambda ~ 2*pi*sqrt(D_v / (b - 1 - a^2))

    Args:
        n_Dv: Number of D_v values to sweep.
        n_steps: Simulation steps per run.
        dt: Timestep.
        N_grid: Spatial grid points per side.
        L_domain: Domain length.
        a: Production rate.
        b: Control parameter (must be > 1+a^2).

    Returns:
        Dictionary with D_v values, measured wavelengths, and theory.
    """
    D_u = 1.0
    D_v_values = np.linspace(4.0, 20.0, n_Dv)

    all_Dv = []
    all_wavelength = []
    all_theory = []

    gap = b - 1.0 - a ** 2
    if gap <= 0:
        raise ValueError(f"b={b} must be > 1+a^2={1+a**2} for Turing instability")

    for i, D_v in enumerate(D_v_values):
        config = _make_config(
            dt=dt, n_steps=n_steps, N_grid=N_grid, L_domain=L_domain,
            a=a, b=b, D_u=D_u, D_v=D_v,
        )
        sim = Brusselator2DSimulation(config)
        sim.reset(seed=i + 300)

        for _ in range(n_steps):
            sim.step()

        wavelength = sim.compute_pattern_wavelength()
        theory = 2.0 * np.pi * np.sqrt(D_v / gap)

        all_Dv.append(D_v)
        all_wavelength.append(wavelength)
        all_theory.append(theory)

        if (i + 1) % 3 == 0:
            logger.info(
                f"  D_v={D_v:.2f}: wavelength={wavelength:.3f}, "
                f"theory={theory:.3f}"
            )

    return {
        "D_v": np.array(all_Dv),
        "wavelength": np.array(all_wavelength),
        "wavelength_theory": np.array(all_theory),
        "D_u": D_u,
        "a": a,
        "b": b,
    }


def run_brusselator_2d_rediscovery(
    output_dir: str | Path = "output/rediscovery/brusselator_2d",
    n_iterations: int = 40,
) -> dict:
    """Run 2D Brusselator reaction-diffusion rediscovery pipeline.

    1. Sweep (b, D_v) to map the Turing pattern phase diagram.
    2. Measure wavelength vs D_v for PySR symbolic regression.
    3. Attempt PySR: wavelength = f(D_v).

    Returns:
        Results dictionary with pattern data, wavelength measurements,
        and PySR discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "brusselator_2d",
        "targets": {
            "turing_instability": "patterns when b > 1+a^2 and D_v/D_u large",
            "wavelength_scaling": "lambda ~ 2*pi*sqrt(D_v/(b-1-a^2))",
            "steady_state": "(u*, v*) = (a, b/a)",
        },
    }

    # --- Part 1: Pattern phase diagram ---
    logger.info("Part 1: 2D Turing pattern phase diagram...")
    pattern_data = generate_pattern_data(
        n_b=6, n_Dv=5, n_steps=3000, dt=0.01, N_grid=32, L_domain=32.0,
    )

    n_patterned = int(np.sum(pattern_data["patterned"]))
    n_total = len(pattern_data["patterned"])
    results["pattern_phase_diagram"] = {
        "n_runs": n_total,
        "n_patterned": n_patterned,
        "fraction_patterned": n_patterned / n_total if n_total > 0 else 0.0,
        "b_c_theory": float(pattern_data["b_c_theory"]),
        "mean_heterogeneity_u": float(np.mean(pattern_data["heterogeneity_u"])),
    }
    logger.info(
        f"  {n_patterned}/{n_total} runs showed Turing patterns "
        f"(b_c theory = {pattern_data['b_c_theory']:.2f})"
    )

    # --- Part 2: Wavelength vs D_v ---
    logger.info("Part 2: Wavelength scaling vs D_v...")
    wave_data = generate_wavelength_data(
        n_Dv=8, n_steps=10000, dt=0.01, N_grid=64, L_domain=64.0,
        a=4.5, b=25.0,
    )

    # Filter out infinite wavelengths (no pattern)
    max_wl = wave_data["D_v"].max() * 100
    valid = np.isfinite(wave_data["wavelength"]) & (
        wave_data["wavelength"] < max_wl
    )
    n_valid = int(np.sum(valid))

    if n_valid > 1:
        corr = np.corrcoef(
            wave_data["wavelength_theory"][valid],
            wave_data["wavelength"][valid],
        )[0, 1]
    else:
        corr = 0.0

    results["wavelength_data"] = {
        "n_valid": n_valid,
        "n_total": len(wave_data["D_v"]),
        "correlation_with_theory": float(corr) if np.isfinite(corr) else 0.0,
        "D_v_range": [float(wave_data["D_v"].min()), float(wave_data["D_v"].max())],
    }

    # --- Part 3: PySR wavelength scaling ---
    if n_valid >= 5:
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = wave_data["D_v"][valid].reshape(-1, 1)
            y = wave_data["wavelength"][valid]

            logger.info("Running PySR: wavelength = f(D_v)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["Dv"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=10,
                populations=15,
                population_size=30,
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
                results["wavelength_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        except Exception as e:
            logger.warning(f"PySR failed: {e}")
            results["wavelength_pysr"] = {"error": str(e)}
    else:
        logger.warning(
            f"Only {n_valid} valid wavelength measurements -- skipping PySR"
        )
        results["wavelength_pysr"] = {"error": "insufficient valid data"}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
