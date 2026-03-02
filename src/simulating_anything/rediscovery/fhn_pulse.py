"""FHN Pulse 1D rediscovery -- traveling pulse speed scaling.

Targets:
- Pulse speed c ~ sqrt(D_v) (diffusion scaling)
- Pulse speed decreases with increasing eps (slower recovery = slower pulse)
- Pulse propagation tracking across spatial domain
- Pulse width characterization
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.fhn_pulse import FHNPulseSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    dt: float | None = None,
    n_steps: int = 2000,
    N: int = 256,
    L: float = 100.0,
    **params: float,
) -> SimulationConfig:
    """Build a SimulationConfig for FHN Pulse with CFL-safe dt."""
    defaults: dict[str, float] = {
        "a": 0.7,
        "b": 0.8,
        "eps": 0.08,
        "D_v": 1.0,
        "D_w": 0.0,
        "I_stim": 0.0,
        "N": float(N),
        "L": L,
    }
    defaults.update(params)

    dx = L / N
    D_max = max(defaults["D_v"], defaults["D_w"])
    cfl_limit = dx ** 2 / (2.0 * D_max) if D_max > 0 else 1.0
    if dt is None:
        dt = 0.5 * cfl_limit  # half the CFL limit for safety

    return SimulationConfig(
        domain=Domain.FHN_PULSE,
        dt=dt,
        n_steps=n_steps,
        parameters=defaults,
    )


def generate_speed_vs_D_v_data(
    n_D: int = 12,
    n_steps: int = 3000,
    N: int = 256,
    L: float = 100.0,
) -> dict[str, np.ndarray]:
    """Measure traveling pulse speed vs activator diffusion coefficient D_v.

    Theory predicts c ~ sqrt(D_v) for FHN traveling pulses in excitable media.

    Args:
        n_D: Number of D_v values to sweep.
        n_steps: Simulation steps per measurement.
        N: Number of grid points.
        L: Domain length.

    Returns:
        Dictionary with D_v values, measured speeds, and sqrt(D_v) scaling.
    """
    D_v_values = np.linspace(0.3, 4.0, n_D)
    all_D: list[float] = []
    all_speed: list[float] = []

    for i, D_v in enumerate(D_v_values):
        try:
            config = _make_config(n_steps=n_steps, N=N, L=L, D_v=D_v)
        except ValueError:
            logger.warning(f"  D_v={D_v:.4f}: skipped (CFL violation)")
            continue

        sim = FHNPulseSimulation(config)
        sim.reset(seed=42)

        # Let pulse form
        warmup = n_steps // 3
        for _ in range(warmup):
            sim.step()

        # Measure speed
        track_steps = n_steps - warmup
        speed = sim.measure_wave_speed(n_steps=track_steps)

        all_D.append(D_v)
        all_speed.append(speed)

        if (i + 1) % 4 == 0:
            logger.info(f"  D_v={D_v:.4f}: wave_speed={speed:.4f}")

    return {
        "D_v": np.array(all_D),
        "wave_speed": np.array(all_speed),
        "sqrt_D_v": np.sqrt(np.array(all_D)),
    }


def generate_speed_vs_eps_data(
    n_eps: int = 12,
    n_steps: int = 3000,
    N: int = 256,
    L: float = 100.0,
    D_v: float = 1.0,
) -> dict[str, np.ndarray]:
    """Measure traveling pulse speed vs timescale separation eps.

    Larger eps means faster recovery, which generally slows the pulse.

    Args:
        n_eps: Number of eps values to sweep.
        n_steps: Simulation steps per measurement.
        N: Number of grid points.
        L: Domain length.
        D_v: Fixed activator diffusion.

    Returns:
        Dictionary with eps values and measured speeds.
    """
    eps_values = np.logspace(-2, -0.3, n_eps)  # 0.01 to ~0.5
    all_eps: list[float] = []
    all_speed: list[float] = []

    for i, eps in enumerate(eps_values):
        config = _make_config(n_steps=n_steps, N=N, L=L, D_v=D_v, eps=eps)
        sim = FHNPulseSimulation(config)
        sim.reset(seed=42)

        # Let pulse form
        warmup = n_steps // 3
        for _ in range(warmup):
            sim.step()

        # Measure speed
        track_steps = n_steps - warmup
        speed = sim.measure_wave_speed(n_steps=track_steps)

        all_eps.append(eps)
        all_speed.append(speed)

        if (i + 1) % 4 == 0:
            logger.info(f"  eps={eps:.4f}: wave_speed={speed:.4f}")

    return {
        "eps": np.array(all_eps),
        "wave_speed": np.array(all_speed),
    }


def generate_pulse_propagation_data(
    D_v: float = 1.0,
    n_steps: int = 2000,
    N: int = 256,
    L: float = 100.0,
    n_snapshots: int = 10,
) -> dict[str, np.ndarray]:
    """Generate snapshots of pulse propagation for visualization.

    Records the v field at evenly spaced times as the pulse travels
    across the domain.

    Args:
        D_v: Activator diffusion coefficient.
        n_steps: Total simulation steps.
        N: Number of grid points.
        L: Domain length.
        n_snapshots: Number of snapshots to capture.

    Returns:
        Dictionary with snapshots, peak positions, times.
    """
    config = _make_config(n_steps=n_steps, N=N, L=L, D_v=D_v)
    sim = FHNPulseSimulation(config)
    sim.reset(seed=42)

    snapshot_interval = max(n_steps // n_snapshots, 1)
    snapshots_v: list[np.ndarray] = []
    snapshots_w: list[np.ndarray] = []
    peak_positions: list[float] = []
    times: list[float] = []

    for step_i in range(1, n_steps + 1):
        sim.step()
        if step_i % snapshot_interval == 0:
            snapshots_v.append(sim.v_field)
            snapshots_w.append(sim.w_field)
            peak_positions.append(sim.v_max_position)
            times.append(step_i * config.dt)

    return {
        "x": sim.x.copy(),
        "snapshots_v": np.array(snapshots_v),
        "snapshots_w": np.array(snapshots_w),
        "peak_positions": np.array(peak_positions),
        "times": np.array(times),
    }


def generate_pulse_shape_data(
    D_v: float = 1.0,
    n_steps: int = 2000,
    N: int = 256,
    L: float = 100.0,
) -> dict[str, float | np.ndarray]:
    """Characterize the steady pulse shape after propagation.

    Args:
        D_v: Activator diffusion coefficient.
        n_steps: Steps to let pulse develop.
        N: Number of grid points.
        L: Domain length.

    Returns:
        Dictionary with pulse profile, width, peak value, and rest state.
    """
    config = _make_config(n_steps=n_steps, N=N, L=L, D_v=D_v)
    sim = FHNPulseSimulation(config)
    sim.reset(seed=42)

    for _ in range(n_steps):
        sim.step()

    v_rest = sim._find_rest_v()

    return {
        "x": sim.x.copy(),
        "v": sim.v_field,
        "w": sim.w_field,
        "v_rest": v_rest,
        "v_max": sim.max_v,
        "pulse_width": sim.pulse_width,
        "pulse_count": sim.pulse_count,
    }


def run_fhn_pulse_rediscovery(
    output_dir: str | Path = "output/rediscovery/fhn_pulse",
    n_iterations: int = 50,
) -> dict:
    """Run FHN Pulse traveling wave rediscovery pipeline.

    1. Measure pulse speed vs D_v (expected: c ~ sqrt(D_v))
    2. Measure pulse speed vs eps (expected: slower for larger eps)
    3. Track pulse propagation across domain
    4. Characterize pulse shape
    5. Optional PySR: speed = f(D_v)

    Args:
        output_dir: Directory for output files.
        n_iterations: PySR iteration count.

    Returns:
        Results dictionary with all measurements and discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "fhn_pulse",
        "targets": {
            "speed_vs_D_v": "c ~ sqrt(D_v) (diffusion scaling)",
            "speed_vs_eps": "c decreases with eps",
            "pulse_propagation": "traveling pulse across domain",
            "pulse_shape": "steady pulse profile characterization",
        },
    }

    # --- Part 1: Pulse speed vs D_v ---
    logger.info("Part 1: Measuring pulse speed vs D_v...")
    dv_data = generate_speed_vs_D_v_data(n_D=12, n_steps=3000, N=256, L=100.0)

    valid = dv_data["wave_speed"] > 0.01
    n_valid = int(np.sum(valid))

    results["speed_vs_D_v"] = {
        "n_samples": n_valid,
        "D_v_range": [
            float(dv_data["D_v"].min()),
            float(dv_data["D_v"].max()),
        ],
        "speed_range": [
            float(dv_data["wave_speed"][valid].min()) if n_valid > 0 else 0.0,
            float(dv_data["wave_speed"][valid].max()) if n_valid > 0 else 0.0,
        ],
    }

    # Compute correlation with sqrt(D_v)
    if n_valid >= 3:
        corr = float(np.corrcoef(
            dv_data["sqrt_D_v"][valid], dv_data["wave_speed"][valid]
        )[0, 1])
        results["speed_vs_D_v"]["correlation_with_sqrt_D_v"] = corr
        logger.info(f"  corr(speed, sqrt(D_v)) = {corr:.4f}")

    # PySR: wave_speed = f(D_v)
    if n_valid >= 5:
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = dv_data["D_v"][valid].reshape(-1, 1)
            y = dv_data["wave_speed"][valid]

            logger.info("Running PySR: wave_speed = f(D_v)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["D_v"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["speed_pysr"] = {
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
                results["speed_pysr"]["best"] = best.expression
                results["speed_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        except Exception as e:
            logger.warning(f"PySR failed: {e}")
            results["speed_pysr"] = {"error": str(e)}
    else:
        logger.warning(f"Only {n_valid} valid speed measurements -- skipping PySR")
        results["speed_pysr"] = {"error": "insufficient valid data"}

    # --- Part 2: Pulse speed vs eps ---
    logger.info("Part 2: Measuring pulse speed vs eps...")
    eps_data = generate_speed_vs_eps_data(
        n_eps=12, n_steps=3000, N=256, L=100.0,
    )

    valid_eps = eps_data["wave_speed"] > 0.01
    n_valid_eps = int(np.sum(valid_eps))
    results["speed_vs_eps"] = {
        "n_samples": n_valid_eps,
        "eps_range": [
            float(eps_data["eps"].min()),
            float(eps_data["eps"].max()),
        ],
        "speed_range": [
            float(eps_data["wave_speed"][valid_eps].min()) if n_valid_eps > 0 else 0.0,
            float(eps_data["wave_speed"][valid_eps].max()) if n_valid_eps > 0 else 0.0,
        ],
    }

    # --- Part 3: Pulse propagation ---
    logger.info("Part 3: Tracking pulse propagation...")
    prop_data = generate_pulse_propagation_data(
        D_v=1.0, n_steps=2000, N=256, L=100.0, n_snapshots=10,
    )
    results["pulse_propagation"] = {
        "n_snapshots": len(prop_data["times"]),
        "times": prop_data["times"].tolist(),
        "peak_positions": prop_data["peak_positions"].tolist(),
    }
    logger.info(f"  {len(prop_data['times'])} snapshots captured")

    # --- Part 4: Pulse shape ---
    logger.info("Part 4: Characterizing pulse shape...")
    shape_data = generate_pulse_shape_data(
        D_v=1.0, n_steps=2000, N=256, L=100.0,
    )
    results["pulse_shape"] = {
        "v_rest": float(shape_data["v_rest"]),
        "v_max": float(shape_data["v_max"]),
        "pulse_width": float(shape_data["pulse_width"]),
        "pulse_count": int(shape_data["pulse_count"]),
    }
    logger.info(
        f"  v_rest={shape_data['v_rest']:.4f}, "
        f"v_max={shape_data['v_max']:.4f}, "
        f"width={shape_data['pulse_width']:.4f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
