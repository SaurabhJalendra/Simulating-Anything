"""FitzHugh-Nagumo 2D lattice rediscovery.

Targets:
- Spiral wave detection and characterization
- Synchronization order parameter vs diffusion coupling D
- Pattern classification (uniform, spiral, turbulent)
- Traveling wave speed measurement
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.fhn_lattice import FHNLattice
from simulating_anything.types.simulation import SimulationConfig

logger = logging.getLogger(__name__)

# Use FHN_SPATIAL as a stand-in Domain until FHN_LATTICE is added to the enum
_DOMAIN_KEY = "fhn_spatial"


def _make_config(
    dt: float = 0.05,
    n_steps: int = 500,
    N: int = 32,
    **params: float,
) -> SimulationConfig:
    """Build a SimulationConfig for the FHN lattice domain."""
    from simulating_anything.types.simulation import Domain

    defaults: dict[str, float] = {
        "a": 0.7,
        "b": 0.8,
        "eps": 0.08,
        "I": 0.5,
        "D": 1.0,
        "N": float(N),
    }
    defaults.update(params)

    # CFL check: dt must be < 1/(4*D) for forward Euler stability
    D_val = defaults.get("D", 1.0)
    if D_val > 0:
        dt_cfl = 1.0 / (4.0 * D_val)
        if dt > dt_cfl:
            dt = 0.9 * dt_cfl

    return SimulationConfig(
        domain=Domain.FHN_SPATIAL,
        dt=dt,
        n_steps=n_steps,
        parameters=defaults,
    )


def generate_sync_data(
    n_D: int = 15,
    n_steps: int = 1000,
    dt: float = 0.02,
    N: int = 16,
) -> dict[str, np.ndarray]:
    """Measure synchronization order parameter vs diffusion coefficient D.

    For each D value, run the simulation from a perturbed initial state
    and measure the synchronization order parameter at steady state.

    Args:
        n_D: Number of D values to sweep.
        n_steps: Simulation steps per run.
        dt: Timestep (adjusted for CFL if needed).
        N: Lattice side length.

    Returns:
        Dictionary with D values and order parameters.
    """
    D_values = np.logspace(-2, 1, n_D)  # 0.01 to 10.0
    all_D: list[float] = []
    all_order: list[float] = []
    all_std_v: list[float] = []

    for i, D_val in enumerate(D_values):
        config = _make_config(dt=dt, n_steps=n_steps, N=N, D=D_val)
        try:
            sim = FHNLattice(config)
        except ValueError:
            logger.warning(f"  D={D_val:.4f}: skipped (CFL violation)")
            continue

        sim.reset(seed=42)

        for _ in range(n_steps):
            sim.step()

        order = sim.synchronization_order_parameter
        all_D.append(D_val)
        all_order.append(order)
        all_std_v.append(sim.std_v)

        if (i + 1) % 5 == 0:
            logger.info(f"  D={D_val:.4f}: sync_order={order:.4f}")

    return {
        "D": np.array(all_D),
        "order_parameter": np.array(all_order),
        "std_v": np.array(all_std_v),
    }


def generate_pattern_data(
    n_D: int = 10,
    n_I: int = 8,
    n_steps: int = 800,
    dt: float = 0.02,
    N: int = 16,
) -> dict[str, np.ndarray]:
    """Sweep D and I to classify spatial patterns.

    For each (D, I) pair, run the simulation and classify the resulting
    pattern as uniform, wave, spiral, or turbulent.

    Args:
        n_D: Number of D values.
        n_I: Number of I (external current) values.
        n_steps: Steps per run.
        dt: Timestep.
        N: Lattice side length.

    Returns:
        Dictionary with parameter values and pattern classifications.
    """
    D_values = np.logspace(-1, 1, n_D)  # 0.1 to 10
    I_values = np.linspace(0.0, 1.5, n_I)

    all_D: list[float] = []
    all_I: list[float] = []
    all_pattern: list[str] = []
    all_std_v: list[float] = []

    for i, D_val in enumerate(D_values):
        for j, I_val in enumerate(I_values):
            config = _make_config(dt=dt, n_steps=n_steps, N=N, D=D_val, I=I_val)
            try:
                sim = FHNLattice(config)
            except ValueError:
                continue

            sim.reset(seed=i * 100 + j)

            for _ in range(n_steps):
                sim.step()

            pattern = sim.classify_pattern()
            all_D.append(D_val)
            all_I.append(I_val)
            all_pattern.append(pattern)
            all_std_v.append(sim.std_v)

        if (i + 1) % 3 == 0:
            logger.info(f"  D sweep {i + 1}/{n_D} complete")

    return {
        "D": np.array(all_D),
        "I": np.array(all_I),
        "pattern": all_pattern,
        "std_v": np.array(all_std_v),
    }


def generate_wave_speed_data(
    n_D: int = 10,
    n_steps: int = 600,
    n_measure: int = 300,
    dt: float = 0.02,
    N: int = 32,
) -> dict[str, np.ndarray]:
    """Measure traveling wave speed vs diffusion coefficient D.

    Args:
        n_D: Number of D values to sweep.
        n_steps: Warmup steps before measuring.
        n_measure: Steps used for speed measurement.
        dt: Timestep.
        N: Lattice side length.

    Returns:
        Dictionary with D values and measured wave speeds.
    """
    D_values = np.logspace(-1, 0.7, n_D)  # 0.1 to ~5
    all_D: list[float] = []
    all_speed: list[float] = []

    for i, D_val in enumerate(D_values):
        config = _make_config(dt=dt, n_steps=n_steps + n_measure, N=N, D=D_val)
        try:
            sim = FHNLattice(config)
        except ValueError:
            logger.warning(f"  D={D_val:.4f}: skipped (CFL violation)")
            continue

        sim.reset(seed=42)

        # Warmup to let waves develop
        for _ in range(n_steps):
            sim.step()

        speed = sim.measure_wave_speed(n_measure=n_measure)
        all_D.append(D_val)
        all_speed.append(speed)

        if (i + 1) % 3 == 0:
            logger.info(f"  D={D_val:.4f}: wave_speed={speed:.4f}")

    return {
        "D": np.array(all_D),
        "wave_speed": np.array(all_speed),
    }


def run_fhn_lattice_rediscovery(
    output_dir: str | Path = "output/rediscovery/fhn_lattice",
    n_iterations: int = 40,
) -> dict:
    """Run FHN 2D lattice rediscovery pipeline.

    1. Measure synchronization order parameter vs D
    2. Classify patterns across (D, I) parameter space
    3. Measure wave speed vs D
    4. Attempt PySR on wave_speed ~ f(D)

    Args:
        output_dir: Directory for saving results.
        n_iterations: PySR iteration count.

    Returns:
        Results dictionary with all measurements and discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "fhn_lattice",
        "description": "FitzHugh-Nagumo on 2D square lattice",
        "targets": {
            "sync_transition": "synchronization order parameter vs D",
            "pattern_classification": "uniform / wave / spiral / turbulent",
            "wave_speed": "traveling wave speed c ~ f(D)",
        },
    }

    # --- Part 1: Synchronization vs D ---
    logger.info("Part 1: Synchronization order parameter vs D...")
    sync_data = generate_sync_data(n_D=15, n_steps=1000, dt=0.02, N=16)

    n_points = len(sync_data["D"])
    results["sync_transition"] = {
        "n_samples": n_points,
        "D_range": [
            float(sync_data["D"].min()) if n_points > 0 else 0.0,
            float(sync_data["D"].max()) if n_points > 0 else 0.0,
        ],
        "order_param_range": [
            float(sync_data["order_parameter"].min()) if n_points > 0 else 0.0,
            float(sync_data["order_parameter"].max()) if n_points > 0 else 0.0,
        ],
        "mean_order_param": float(np.mean(sync_data["order_parameter"]))
        if n_points > 0
        else 0.0,
    }
    if n_points > 0:
        logger.info(
            f"  {n_points} D values measured, "
            f"order param range [{sync_data['order_parameter'].min():.3f}, "
            f"{sync_data['order_parameter'].max():.3f}]"
        )

    # --- Part 2: Pattern classification ---
    logger.info("Part 2: Pattern classification in (D, I) space...")
    pattern_data = generate_pattern_data(n_D=8, n_I=6, n_steps=500, dt=0.02, N=16)

    n_runs = len(pattern_data["D"])
    pattern_counts: dict[str, int] = {}
    for p in pattern_data["pattern"]:
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

    results["pattern_classification"] = {
        "n_runs": n_runs,
        "pattern_counts": pattern_counts,
        "D_range": [
            float(pattern_data["D"].min()) if n_runs > 0 else 0.0,
            float(pattern_data["D"].max()) if n_runs > 0 else 0.0,
        ],
        "I_range": [
            float(pattern_data["I"].min()) if n_runs > 0 else 0.0,
            float(pattern_data["I"].max()) if n_runs > 0 else 0.0,
        ],
    }
    logger.info(f"  {n_runs} (D,I) pairs classified: {pattern_counts}")

    # --- Part 3: Wave speed vs D ---
    logger.info("Part 3: Wave speed vs D...")
    wave_data = generate_wave_speed_data(n_D=10, n_steps=400, n_measure=200, dt=0.02, N=32)

    valid = wave_data["wave_speed"] > 0.01
    n_valid = int(np.sum(valid))
    results["wave_speed_data"] = {
        "n_valid": n_valid,
        "n_total": len(wave_data["D"]),
        "D_range": [
            float(wave_data["D"].min()) if len(wave_data["D"]) > 0 else 0.0,
            float(wave_data["D"].max()) if len(wave_data["D"]) > 0 else 0.0,
        ],
        "speed_range": [
            float(wave_data["wave_speed"][valid].min()) if n_valid > 0 else 0.0,
            float(wave_data["wave_speed"][valid].max()) if n_valid > 0 else 0.0,
        ],
    }

    # --- Part 4: PySR on wave speed ---
    if n_valid >= 5:
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = wave_data["D"][valid].reshape(-1, 1)
            y = wave_data["wave_speed"][valid]

            logger.info("Running PySR: wave_speed = f(D)...")
            discoveries = run_symbolic_regression(
                X,
                y,
                variable_names=["D_"],
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
                results["wave_speed_pysr"]["best_r2"] = best.evidence.fit_r_squared
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

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
