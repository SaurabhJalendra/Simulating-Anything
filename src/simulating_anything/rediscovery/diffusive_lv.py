"""Diffusive Lotka-Volterra rediscovery.

Targets:
- Traveling wave speed vs diffusion coefficient
- Spatial heterogeneity characterization
- Connection to classical LV (no diffusion limit)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.diffusive_lv import DiffusiveLotkaVolterra
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    dt: float = 0.005,
    n_steps: int = 2000,
    N_grid: int = 64,
    L_domain: float = 20.0,
    **params: float,
) -> SimulationConfig:
    """Build a SimulationConfig for the diffusive LV domain."""
    defaults = {
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 0.5,
        "delta": 0.2,
        "D_u": 0.1,
        "D_v": 0.05,
        "N_grid": float(N_grid),
        "L_domain": L_domain,
    }
    defaults.update(params)
    return SimulationConfig(
        domain=Domain.DIFFUSIVE_LV,
        dt=dt,
        n_steps=n_steps,
        parameters=defaults,
    )


def generate_wave_speed_data(
    n_D: int = 15,
    n_steps: int = 4000,
    dt: float = 0.005,
    N_grid: int = 128,
    L_domain: float = 40.0,
) -> dict[str, np.ndarray]:
    """Measure traveling wave speed vs prey diffusion coefficient.

    Strategy: Initialize with a localized prey pulse on a mostly-empty domain,
    run to observe spreading front, measure front speed from the position of
    the 50%-of-max contour over time.

    Returns:
        Dictionary with D_u values and measured wave speeds.
    """
    D_u_values = np.logspace(-2, 0, n_D)  # 0.01 to 1.0
    all_D = []
    all_speed = []

    for i, D_u in enumerate(D_u_values):
        config = _make_config(
            dt=dt,
            n_steps=n_steps,
            N_grid=N_grid,
            L_domain=L_domain,
            D_u=D_u,
            D_v=D_u * 0.5,  # Keep ratio fixed
        )
        sim = DiffusiveLotkaVolterra(config)
        sim.reset(seed=42)

        # Override initial condition: localized prey pulse in left quarter
        u_eq = sim.gamma / sim.delta
        sim._u[:] = 0.1 * u_eq
        sim._u[:N_grid // 4] = u_eq
        sim._v[:] = sim.alpha / sim.beta * 0.5
        sim._state = np.concatenate([sim._u, sim._v])

        # Record front position at two times
        t_measure_1 = n_steps // 3
        t_measure_2 = 2 * n_steps // 3

        pos_1 = None
        pos_2 = None

        for step in range(1, n_steps + 1):
            sim.step()
            if step == t_measure_1:
                threshold = 0.5 * np.max(sim._u)
                above = np.where(sim._u > threshold)[0]
                pos_1 = float(np.max(above)) * sim.dx if len(above) > 0 else 0.0
            elif step == t_measure_2:
                threshold = 0.5 * np.max(sim._u)
                above = np.where(sim._u > threshold)[0]
                pos_2 = float(np.max(above)) * sim.dx if len(above) > 0 else 0.0

        if pos_1 is not None and pos_2 is not None:
            delta_t = (t_measure_2 - t_measure_1) * dt
            speed = abs(pos_2 - pos_1) / delta_t if delta_t > 0 else 0.0
        else:
            speed = 0.0

        all_D.append(D_u)
        all_speed.append(speed)

        if (i + 1) % 5 == 0:
            logger.info(f"  D_u={D_u:.4f}: wave_speed={speed:.4f}")

    return {
        "D_u": np.array(all_D),
        "wave_speed": np.array(all_speed),
    }


def generate_spatial_pattern_data(
    n_runs: int = 10,
    n_steps: int = 5000,
    dt: float = 0.005,
    N_grid: int = 64,
    L_domain: float = 20.0,
) -> dict[str, np.ndarray]:
    """Run diffusive LV to characterize spatial heterogeneity at steady state.

    Varies D_u/D_v ratio and measures the coefficient of variation (spatial
    heterogeneity) of the final prey and predator fields.

    Returns:
        Dictionary with D_u, D_v, and heterogeneity measurements.
    """
    D_ratios = np.linspace(0.5, 5.0, n_runs)
    D_v_base = 0.05

    all_D_u = []
    all_D_v = []
    all_prey_het = []
    all_pred_het = []
    all_total_prey = []
    all_total_pred = []

    for i, ratio in enumerate(D_ratios):
        D_u = D_v_base * ratio
        config = _make_config(
            dt=dt,
            n_steps=n_steps,
            N_grid=N_grid,
            L_domain=L_domain,
            D_u=D_u,
            D_v=D_v_base,
        )
        sim = DiffusiveLotkaVolterra(config)
        sim.reset(seed=i + 100)

        for _ in range(n_steps):
            sim.step()

        prey_het = sim.spatial_heterogeneity(sim._u)
        pred_het = sim.spatial_heterogeneity(sim._v)

        all_D_u.append(D_u)
        all_D_v.append(D_v_base)
        all_prey_het.append(prey_het)
        all_pred_het.append(pred_het)
        all_total_prey.append(sim.total_prey)
        all_total_pred.append(sim.total_predator)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  D_u/D_v={ratio:.2f}: "
                f"prey_het={prey_het:.4f}, pred_het={pred_het:.4f}"
            )

    return {
        "D_u": np.array(all_D_u),
        "D_v": np.array(all_D_v),
        "prey_heterogeneity": np.array(all_prey_het),
        "pred_heterogeneity": np.array(all_pred_het),
        "total_prey": np.array(all_total_prey),
        "total_predator": np.array(all_total_pred),
    }


def run_diffusive_lv_rediscovery(
    output_dir: str | Path = "output/rediscovery/diffusive_lv",
    n_iterations: int = 40,
) -> dict:
    """Run diffusive Lotka-Volterra rediscovery.

    1. Measure traveling wave speed vs D_u
    2. Characterize spatial patterns vs D_u/D_v ratio
    3. Run PySR on wave speed ~ f(D_u)

    Returns:
        Results dictionary with all measurements and PySR discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "diffusive_lv",
        "targets": {
            "wave_speed": "c ~ 2*sqrt(alpha*D_u) (Fisher-KPP type)",
            "spatial_heterogeneity": "pattern formation from diffusion mismatch",
        },
    }

    # Part 1: Wave speed data
    logger.info("Part 1: Measuring traveling wave speed vs D_u...")
    wave_data = generate_wave_speed_data(
        n_D=15, n_steps=4000, dt=0.005, N_grid=128, L_domain=40.0,
    )

    valid = wave_data["wave_speed"] > 0.01
    n_valid = int(np.sum(valid))
    results["wave_speed_data"] = {
        "n_samples": n_valid,
        "D_u_range": [float(wave_data["D_u"].min()), float(wave_data["D_u"].max())],
        "speed_range": [
            float(wave_data["wave_speed"][valid].min()) if n_valid > 0 else 0.0,
            float(wave_data["wave_speed"][valid].max()) if n_valid > 0 else 0.0,
        ],
    }

    # PySR on wave speed
    if n_valid >= 5:
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = wave_data["D_u"][valid].reshape(-1, 1)
            y = wave_data["wave_speed"][valid]

            logger.info("Running PySR: wave_speed = f(D_u)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["D_u"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["wave_speed_pysr"] = {
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
                results["wave_speed_pysr"]["best"] = best.expression
                results["wave_speed_pysr"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        except Exception as e:
            logger.warning(f"PySR failed: {e}")
            results["wave_speed_pysr"] = {"error": str(e)}
    else:
        logger.warning(f"Only {n_valid} valid wave speed measurements -- skipping PySR")
        results["wave_speed_pysr"] = {"error": "insufficient valid data"}

    # Part 2: Spatial pattern data
    logger.info("Part 2: Characterizing spatial patterns vs D_u/D_v ratio...")
    pattern_data = generate_spatial_pattern_data(
        n_runs=10, n_steps=5000, dt=0.005, N_grid=64, L_domain=20.0,
    )

    results["spatial_pattern_data"] = {
        "n_runs": len(pattern_data["D_u"]),
        "mean_prey_heterogeneity": float(np.mean(pattern_data["prey_heterogeneity"])),
        "mean_pred_heterogeneity": float(np.mean(pattern_data["pred_heterogeneity"])),
        "mean_total_prey": float(np.mean(pattern_data["total_prey"])),
        "mean_total_predator": float(np.mean(pattern_data["total_predator"])),
    }

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
