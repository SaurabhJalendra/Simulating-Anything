"""Oregonator 1D rediscovery -- traveling pulse speed and wave formation.

Targets:
- Pulse speed vs eps: c ~ sqrt(D_u / eps) in the sharp-front limit
- Pulse speed vs D_u: c ~ sqrt(D_u)
- Wave formation from localized stimulus
- Pulse counting after transient
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.oregonator_1d import Oregonator1DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    eps: float = 0.1,
    f: float = 1.0,
    q: float = 0.002,
    D_u: float = 1.0,
    D_v: float = 0.6,
    N: int = 200,
    L: float = 100.0,
    dt: float | None = None,
    n_steps: int = 1000,
    seed: int = 42,
) -> SimulationConfig:
    """Create a SimulationConfig for 1D Oregonator with CFL-safe dt."""
    dx = L / N
    D_max = max(D_u, D_v)
    cfl_limit = dx**2 / (4.0 * D_max) if D_max > 0 else 1.0
    if dt is None:
        dt = 0.5 * cfl_limit  # half the CFL limit for safety
    return SimulationConfig(
        domain=Domain.OREGONATOR_1D,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "eps": eps,
            "f": f,
            "q": q,
            "D_u": D_u,
            "D_v": D_v,
            "N": float(N),
            "L": L,
        },
        seed=seed,
    )


def generate_pulse_speed_vs_eps(
    n_eps: int = 10,
    N: int = 200,
    L: float = 100.0,
    warmup_steps: int = 500,
    track_steps: int = 300,
) -> dict[str, np.ndarray]:
    """Sweep eps and measure pulse speed.

    Theory predicts c ~ sqrt(D_u / eps) for sharp-front traveling pulses
    in excitable media. We measure the pulse speed for a range of eps
    values and check the scaling.

    Args:
        n_eps: Number of eps values to sweep.
        N: Grid resolution.
        L: Domain length.
        warmup_steps: Steps to let pulse form before measuring speed.
        track_steps: Steps to track pulse movement.

    Returns:
        Dict with eps_values, speeds, and theoretical c ~ 1/sqrt(eps).
    """
    eps_values = np.linspace(0.02, 0.3, n_eps)
    D_u = 1.0

    all_eps = []
    all_speeds = []

    for eps_val in eps_values:
        config = _make_config(
            eps=eps_val, D_u=D_u, N=N, L=L, n_steps=warmup_steps,
        )
        sim = Oregonator1DSimulation(config)
        sim.reset()

        # Warmup to let pulse form and propagate
        for _ in range(warmup_steps):
            sim.step()

        # Only measure if a pulse is present
        if sim.count_pulses() >= 1:
            speed = sim.measure_pulse_speed(n_steps=track_steps)
            all_eps.append(eps_val)
            all_speeds.append(speed)
            logger.debug(f"  eps={eps_val:.3f}: speed={speed:.4f}")
        else:
            logger.debug(f"  eps={eps_val:.3f}: no pulse detected")

    eps_arr = np.array(all_eps)
    speed_arr = np.array(all_speeds)

    # Theoretical scaling: c = A * sqrt(D_u / eps)
    if len(eps_arr) > 0:
        theory_scaling = np.sqrt(D_u / eps_arr)
    else:
        theory_scaling = np.array([])

    return {
        "eps": eps_arr,
        "speed": speed_arr,
        "theory_scaling": theory_scaling,
        "D_u": D_u,
    }


def generate_pulse_speed_vs_D_u(
    n_D: int = 8,
    N: int = 200,
    L: float = 100.0,
    warmup_steps: int = 500,
    track_steps: int = 300,
) -> dict[str, np.ndarray]:
    """Sweep D_u and measure pulse speed.

    Theory: c ~ sqrt(D_u / eps), so at fixed eps, c ~ sqrt(D_u).

    Args:
        n_D: Number of D_u values.
        N: Grid resolution.
        L: Domain length.
        warmup_steps: Warmup steps.
        track_steps: Tracking steps.

    Returns:
        Dict with D_u_values, speeds, sqrt(D_u) scaling.
    """
    D_u_values = np.linspace(0.3, 3.0, n_D)
    eps_fixed = 0.1

    all_D = []
    all_speeds = []

    for D_u_val in D_u_values:
        config = _make_config(
            eps=eps_fixed, D_u=D_u_val, N=N, L=L, n_steps=warmup_steps,
        )
        sim = Oregonator1DSimulation(config)
        sim.reset()

        for _ in range(warmup_steps):
            sim.step()

        if sim.count_pulses() >= 1:
            speed = sim.measure_pulse_speed(n_steps=track_steps)
            all_D.append(D_u_val)
            all_speeds.append(speed)
            logger.debug(f"  D_u={D_u_val:.3f}: speed={speed:.4f}")

    return {
        "D_u": np.array(all_D),
        "speed": np.array(all_speeds),
        "eps": eps_fixed,
    }


def generate_wave_formation_data(
    N: int = 200,
    L: float = 100.0,
    n_snapshots: int = 10,
    total_steps: int = 2000,
) -> dict[str, np.ndarray]:
    """Capture snapshots of wave formation from localized stimulus.

    Records u and v fields at evenly-spaced times to visualize
    the formation and propagation of the traveling pulse.

    Args:
        N: Grid resolution.
        L: Domain length.
        n_snapshots: Number of snapshots to capture.
        total_steps: Total simulation steps.

    Returns:
        Dict with snapshots array (n_snapshots, 2, N) and times.
    """
    config = _make_config(N=N, L=L, n_steps=total_steps)
    sim = Oregonator1DSimulation(config)
    sim.reset()

    snapshot_interval = max(total_steps // n_snapshots, 1)
    snapshots = []
    times = []

    for step_i in range(total_steps):
        sim.step()
        if (step_i + 1) % snapshot_interval == 0:
            snapshots.append(sim.observe().copy())
            times.append((step_i + 1) * config.dt)

    return {
        "snapshots": np.array(snapshots),
        "times": np.array(times),
        "x": sim.x.copy(),
    }


def run_oregonator_1d_rediscovery(
    output_dir: str | Path = "output/rediscovery/oregonator_1d",
    n_iterations: int = 40,
) -> dict:
    """Run Oregonator 1D rediscovery analysis.

    Measures pulse speed vs eps and D_u to verify the scaling
    c ~ sqrt(D_u / eps) predicted for excitable media.

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iterations (for future symbolic regression).

    Returns:
        Results dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "oregonator_1d",
        "targets": {
            "pulse_speed_eps": "Pulse speed ~ sqrt(D_u / eps)",
            "pulse_speed_D_u": "Pulse speed ~ sqrt(D_u)",
            "wave_formation": "Traveling pulse from localized stimulus",
        },
    }

    # --- Part 1: Pulse speed vs eps ---
    logger.info("Part 1: Measuring pulse speed vs eps...")
    eps_data = generate_pulse_speed_vs_eps(
        n_eps=8, N=150, L=80.0, warmup_steps=400, track_steps=200,
    )

    if len(eps_data["speed"]) >= 3:
        # Compute correlation with 1/sqrt(eps)
        inv_sqrt_eps = 1.0 / np.sqrt(eps_data["eps"])
        corr = float(np.corrcoef(inv_sqrt_eps, eps_data["speed"])[0, 1])

        results["pulse_speed_vs_eps"] = {
            "n_measurements": len(eps_data["speed"]),
            "eps_values": eps_data["eps"].tolist(),
            "speeds": eps_data["speed"].tolist(),
            "correlation_with_1_over_sqrt_eps": corr,
            "mean_speed": float(np.mean(eps_data["speed"])),
        }
        logger.info(
            f"  {len(eps_data['speed'])} measurements, "
            f"corr(speed, 1/sqrt(eps)) = {corr:.4f}"
        )

        # PySR: speed = g(eps) with D_u fixed
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            X = eps_data["eps"].reshape(-1, 1)
            y = eps_data["speed"]

            logger.info("  Running PySR: speed = g(eps)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["eps_val"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "inv", "square"],
                max_complexity=10,
                populations=20,
                population_size=40,
            )
            results["pysr_speed_eps"] = {
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
                results["pysr_speed_eps"]["best"] = best.expression
                results["pysr_speed_eps"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        except Exception as e:
            logger.warning(f"  PySR failed: {e}")
            results["pysr_speed_eps"] = {"error": str(e)}
    else:
        results["pulse_speed_vs_eps"] = {
            "n_measurements": len(eps_data["speed"]),
            "note": "Too few pulse measurements for analysis",
        }
        logger.info("  Too few pulse measurements")

    # --- Part 2: Pulse speed vs D_u ---
    logger.info("Part 2: Measuring pulse speed vs D_u...")
    D_data = generate_pulse_speed_vs_D_u(
        n_D=6, N=150, L=80.0, warmup_steps=400, track_steps=200,
    )

    if len(D_data["speed"]) >= 3:
        sqrt_D = np.sqrt(D_data["D_u"])
        corr = float(np.corrcoef(sqrt_D, D_data["speed"])[0, 1])

        results["pulse_speed_vs_D_u"] = {
            "n_measurements": len(D_data["speed"]),
            "D_u_values": D_data["D_u"].tolist(),
            "speeds": D_data["speed"].tolist(),
            "correlation_with_sqrt_D_u": corr,
        }
        logger.info(
            f"  {len(D_data['speed'])} measurements, "
            f"corr(speed, sqrt(D_u)) = {corr:.4f}"
        )
    else:
        results["pulse_speed_vs_D_u"] = {
            "n_measurements": len(D_data["speed"]),
            "note": "Too few pulse measurements",
        }

    # --- Part 3: Wave formation visualization data ---
    logger.info("Part 3: Capturing wave formation snapshots...")
    wave_data = generate_wave_formation_data(
        N=150, L=80.0, n_snapshots=8, total_steps=1000,
    )

    results["wave_formation"] = {
        "n_snapshots": len(wave_data["times"]),
        "times": wave_data["times"].tolist(),
        "max_u_per_snapshot": [
            float(np.max(wave_data["snapshots"][i, 0]))
            for i in range(len(wave_data["times"]))
        ],
    }
    logger.info(f"  Captured {len(wave_data['times'])} snapshots")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f_out:
        json.dump(results, f_out, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
