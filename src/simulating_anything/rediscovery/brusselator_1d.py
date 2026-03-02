"""1D Brusselator PDE rediscovery.

Targets:
- Turing instability onset: b_c = 1 + a^2 (necessary condition)
- Turing wavelength scaling: lambda_c ~ 2*pi*sqrt(D_u*D_v) / sqrt(b-1-a^2)
- Homogeneous steady state: (u*, v*) = (a, b/a)
- Pattern formation only when b > b_c AND D_v/D_u is large enough
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.brusselator_1d import Brusselator1DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    dt: float = 0.005,
    n_steps: int = 2000,
    N: int = 256,
    L: float = 20.0,
    **params: float,
) -> SimulationConfig:
    """Build a SimulationConfig for the 1D Brusselator domain."""
    defaults: dict[str, float] = {
        "a": 1.0,
        "b": 3.0,
        "D_u": 0.01,
        "D_v": 0.1,
        "N": float(N),
        "L": L,
    }
    defaults.update(params)

    # Auto-adjust dt for CFL stability
    D_max = max(defaults.get("D_u", 0.01), defaults.get("D_v", 0.1))
    dx = L / N
    if D_max > 0:
        dt_cfl = dx ** 2 / (2.0 * D_max)
        if dt > dt_cfl:
            dt = 0.9 * dt_cfl

    return SimulationConfig(
        domain=Domain.BRUSSELATOR_1D,  # placeholder -- no BRUSSELATOR_1D enum yet
        dt=dt,
        n_steps=n_steps,
        parameters=defaults,
    )


def generate_wavelength_data(
    n_Dv: int = 12,
    n_steps: int = 20000,
    dt: float = 0.005,
    N: int = 256,
    L: float = 40.0,
    a: float = 1.0,
    b: float = 3.5,
) -> dict[str, np.ndarray | float]:
    """Measure dominant Turing wavelength vs D_v (fixed D_u, a, b).

    Theoretical prediction:
        lambda_c ~ 2*pi*sqrt(D_u*D_v) / sqrt(b - 1 - a^2)

    Args:
        n_Dv: Number of D_v values to sweep.
        n_steps: Simulation steps per run (must be long enough for patterns).
        dt: Timestep (auto-adjusted for CFL).
        N: Spatial grid points.
        L: Domain length (large enough for multiple wavelengths).
        a: Production rate.
        b: Control parameter (must be > 1+a^2 for Turing instability).

    Returns:
        Dictionary with D_v values, measured wavelengths, and theory.
    """
    D_u = 0.01
    D_v_values = np.logspace(-1.5, 0.0, n_Dv)  # ~0.03 to 1.0

    b_minus_bc = b - 1.0 - a ** 2
    if b_minus_bc <= 0:
        raise ValueError(f"b={b} must be > 1+a^2={1+a**2} for Turing instability")

    all_Dv: list[float] = []
    all_wavelength: list[float] = []
    all_theory: list[float] = []

    for i, D_v in enumerate(D_v_values):
        config = _make_config(
            dt=dt, n_steps=n_steps, N=N, L=L,
            a=a, b=b, D_u=D_u, D_v=D_v,
        )
        sim = Brusselator1DSimulation(config)
        sim.reset(seed=i + 200)

        for _ in range(n_steps):
            sim.step()

        wavelength = sim.dominant_wavelength()
        theory = 2.0 * np.pi * np.sqrt(D_u * D_v) / np.sqrt(b_minus_bc)

        all_Dv.append(float(D_v))
        all_wavelength.append(wavelength)
        all_theory.append(theory)

        if (i + 1) % 4 == 0:
            logger.info(
                f"  D_v={D_v:.4f}: wavelength={wavelength:.3f}, "
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


def generate_instability_onset_data(
    n_b: int = 20,
    n_steps: int = 15000,
    dt: float = 0.005,
    N: int = 128,
    L: float = 20.0,
    a: float = 1.0,
    D_u: float = 0.01,
    D_v: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Sweep b to detect the onset of Turing instability.

    For each b value, run the simulation and measure final spatial
    heterogeneity. The transition should occur near b_c = 1 + a^2.

    Args:
        n_b: Number of b values to test.
        n_steps: Steps per run.
        dt: Timestep.
        N: Grid points.
        L: Domain length.
        a: Production rate.
        D_u: Activator diffusion.
        D_v: Inhibitor diffusion.

    Returns:
        Dictionary with b values, heterogeneity, and theoretical threshold.
    """
    b_c = 1.0 + a ** 2
    # Sweep b from below to above the threshold
    b_values = np.linspace(max(0.5, b_c - 1.5), b_c + 3.0, n_b)

    all_b: list[float] = []
    all_het: list[float] = []
    all_patterned: list[bool] = []

    for i, b in enumerate(b_values):
        config = _make_config(
            dt=dt, n_steps=n_steps, N=N, L=L,
            a=a, b=b, D_u=D_u, D_v=D_v,
        )
        sim = Brusselator1DSimulation(config)
        sim.reset(seed=i + 100)

        for _ in range(n_steps):
            sim.step()

        het = sim.spatial_heterogeneity_u
        all_b.append(float(b))
        all_het.append(het)
        all_patterned.append(het > 0.05)

        if (i + 1) % 5 == 0:
            logger.info(f"  b={b:.3f}: heterogeneity={het:.4f}")

    return {
        "b": np.array(all_b),
        "heterogeneity": np.array(all_het),
        "patterned": np.array(all_patterned),
        "b_c_theory": b_c,
        "a": a,
        "D_u": D_u,
        "D_v": D_v,
    }


def generate_pattern_phase_data(
    n_b: int = 10,
    n_Dv: int = 8,
    n_steps: int = 10000,
    dt: float = 0.005,
    N: int = 64,
    L: float = 20.0,
    a: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Sweep b and D_v to build a pattern phase diagram.

    Args:
        n_b: Number of b values.
        n_Dv: Number of D_v values.
        n_steps: Steps per simulation.
        dt: Timestep.
        N: Grid points.
        L: Domain length.
        a: Production rate.

    Returns:
        Dictionary with b, D_v arrays, heterogeneity grid, and classifications.
    """
    D_u = 0.01
    b_values = np.linspace(1.5, 5.0, n_b)
    D_v_values = np.logspace(-2, 0, n_Dv)

    all_b: list[float] = []
    all_Dv: list[float] = []
    all_het: list[float] = []
    all_patterned: list[bool] = []

    for i, b in enumerate(b_values):
        for j, D_v in enumerate(D_v_values):
            config = _make_config(
                dt=dt, n_steps=n_steps, N=N, L=L,
                a=a, b=b, D_u=D_u, D_v=D_v,
            )
            sim = Brusselator1DSimulation(config)
            sim.reset(seed=i * 100 + j)

            for _ in range(n_steps):
                sim.step()

            het = sim.spatial_heterogeneity_u
            all_b.append(float(b))
            all_Dv.append(float(D_v))
            all_het.append(het)
            all_patterned.append(het > 0.05)

        if (i + 1) % 3 == 0:
            logger.info(f"  Phase diagram: b sweep {i + 1}/{n_b}")

    return {
        "b": np.array(all_b),
        "D_v": np.array(all_Dv),
        "heterogeneity": np.array(all_het),
        "patterned": np.array(all_patterned),
        "b_c_theory": 1.0 + a ** 2,
        "a": a,
        "D_u": D_u,
    }


def run_brusselator_1d_rediscovery(
    output_dir: str | Path = "output/rediscovery/brusselator_1d",
    n_iterations: int = 50,
) -> dict:
    """Run 1D Brusselator PDE rediscovery pipeline.

    1. Measure Turing wavelength from spatial FFT across D_v sweep.
    2. Sweep b to find instability onset (b_c = 1 + a^2).
    3. Build (b, D_v) phase diagram.
    4. Attempt PySR: wavelength = f(D_v, D_u, b, a).

    Returns:
        Results dictionary with wavelength data, onset data, phase diagram,
        and PySR discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "brusselator_1d",
        "targets": {
            "turing_instability": "patterns when b > 1+a^2 and D_v/D_u large",
            "wavelength_scaling": "lambda ~ 2*pi*sqrt(D_u*D_v)/sqrt(b-1-a^2)",
            "steady_state": "(u*, v*) = (a, b/a)",
        },
    }

    # --- Part 1: Wavelength vs D_v ---
    logger.info("Part 1: Turing wavelength vs D_v...")
    wave_data = generate_wavelength_data(
        n_Dv=10, n_steps=15000, dt=0.005, N=256, L=40.0,
        a=1.0, b=3.5,
    )

    # Filter out infinite wavelengths (no pattern formed)
    max_wl = float(wave_data["D_v"][-1]) * 1000  # type: ignore[index]
    wl = wave_data["wavelength"]
    valid = np.isfinite(wl) & (wl < max_wl)
    n_valid = int(np.sum(valid))

    if n_valid > 1:
        corr = float(np.corrcoef(
            wave_data["wavelength_theory"][valid],
            wl[valid],
        )[0, 1])
    else:
        corr = 0.0

    results["wavelength_data"] = {
        "n_valid": n_valid,
        "n_total": len(wave_data["D_v"]),
        "correlation_with_theory": corr if np.isfinite(corr) else 0.0,
        "D_v_range": [
            float(wave_data["D_v"][0]),
            float(wave_data["D_v"][-1]),
        ],
    }
    logger.info(
        f"  {n_valid}/{len(wave_data['D_v'])} valid wavelength measurements, "
        f"correlation={corr:.4f}"
    )

    # --- Part 2: Instability onset b sweep ---
    logger.info("Part 2: Turing instability onset (b sweep)...")
    onset_data = generate_instability_onset_data(
        n_b=15, n_steps=10000, dt=0.005, N=128, L=20.0,
    )

    # Estimate b_c from the data: first b where heterogeneity exceeds threshold
    patterned_mask = onset_data["patterned"]
    if np.any(patterned_mask):
        b_c_estimate = float(onset_data["b"][np.argmax(patterned_mask)])
    else:
        b_c_estimate = float("nan")

    results["instability_onset"] = {
        "b_c_theory": float(onset_data["b_c_theory"]),
        "b_c_estimate": b_c_estimate,
        "n_patterned": int(np.sum(patterned_mask)),
        "n_total": len(onset_data["b"]),
    }
    logger.info(
        f"  b_c theory={onset_data['b_c_theory']:.3f}, "
        f"estimate={b_c_estimate:.3f}"
    )

    # --- Part 3: Phase diagram ---
    logger.info("Part 3: (b, D_v) phase diagram...")
    phase_data = generate_pattern_phase_data(
        n_b=6, n_Dv=5, n_steps=5000, dt=0.005, N=64, L=20.0,
    )

    n_patterned = int(np.sum(phase_data["patterned"]))
    n_total = len(phase_data["patterned"])
    results["phase_diagram"] = {
        "n_runs": n_total,
        "n_patterned": n_patterned,
        "fraction_patterned": n_patterned / n_total if n_total > 0 else 0.0,
        "b_c_theory": float(phase_data["b_c_theory"]),
    }

    # --- Part 4: PySR wavelength scaling ---
    if n_valid >= 5:
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            sqrt_DuDv = np.sqrt(
                wave_data["D_v"][valid] * float(wave_data["D_u"])
            )
            X = sqrt_DuDv.reshape(-1, 1)
            y = wl[valid]

            logger.info("Running PySR: wavelength = f(sqrt(D_u*D_v))...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["sqrtDD"],
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
