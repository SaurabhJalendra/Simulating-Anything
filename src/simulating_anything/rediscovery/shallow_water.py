"""Shallow water equations 1D rediscovery.

Targets:
- Wave speed: c = sqrt(g * h) for gravity waves
- Mass conservation: integral(h * dx) = const
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.shallow_water import ShallowWater
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_sim(
    g: float = 9.81,
    h0: float = 1.0,
    N: int = 128,
    L: float = 10.0,
    dt: float = 0.001,
    n_steps: int = 2000,
    perturbation_amplitude: float = 0.01,
) -> ShallowWater:
    """Create a ShallowWater simulation with the given parameters."""
    config = SimulationConfig(
        domain=Domain.SHALLOW_WATER,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "g": g,
            "h0": h0,
            "N": float(N),
            "L": L,
            "perturbation_amplitude": perturbation_amplitude,
        },
    )
    return ShallowWater(config)


def generate_wave_speed_data(
    n_g: int = 20,
    n_h: int = 10,
    n_steps: int = 2000,
    dt: float = 0.001,
    N: int = 256,
) -> dict[str, np.ndarray]:
    """Sweep g and h0 to measure gravity wave speed and compare to sqrt(g*h0).

    A small Gaussian bump on a flat surface splits into two counter-propagating
    waves. We track the right-traveling peak to measure the wave speed.

    Returns dict with arrays: g_values, h0_values, speed_measured, speed_theory.
    """
    L = 10.0

    all_g = []
    all_h0 = []
    all_speed_measured = []
    all_speed_theory = []

    g_values = np.linspace(2.0, 20.0, n_g)
    h0_values = np.linspace(0.5, 3.0, n_h)

    count = 0
    total = n_g * n_h
    for g_val in g_values:
        for h0_val in h0_values:
            count += 1

            # Small perturbation for linear regime
            amp = 0.01 * h0_val
            sim = _make_sim(
                g=g_val, h0=h0_val, N=N, L=L, dt=dt,
                n_steps=n_steps, perturbation_amplitude=amp,
            )
            sim.reset()

            # Let the bump split, then track the right-traveling peak
            # Wait for separation: about L/4 / c steps
            c_theory = np.sqrt(g_val * h0_val)
            sep_time = (L / 4) / c_theory
            sep_steps = max(int(sep_time / dt), 50)
            sep_steps = min(sep_steps, n_steps // 3)

            for _ in range(sep_steps):
                sim.step()

            h = sim.height_field
            # Look at the right half of domain for the right-traveling peak
            mid = N // 2
            right_half = h[mid:]
            peak1_local = int(np.argmax(right_half))
            peak1_idx = mid + peak1_local
            pos1 = sim.x[peak1_idx]
            t1 = sep_steps * dt

            # Run more steps
            more_steps = min(n_steps // 3, sep_steps)
            for _ in range(more_steps):
                sim.step()

            h = sim.height_field
            right_half = h[mid:]
            peak2_local = int(np.argmax(right_half))
            peak2_idx = mid + peak2_local
            pos2 = sim.x[peak2_idx]
            t2 = (sep_steps + more_steps) * dt

            # Account for periodic boundaries
            delta_x = pos2 - pos1
            if delta_x < -L / 2:
                delta_x += L
            elif delta_x > L / 2:
                delta_x -= L

            delta_t = t2 - t1
            if delta_t > 0 and abs(delta_x) > 1e-10:
                speed = abs(delta_x) / delta_t
            else:
                speed = float("nan")

            all_g.append(g_val)
            all_h0.append(h0_val)
            all_speed_measured.append(speed)
            all_speed_theory.append(c_theory)

            if count % 20 == 0:
                logger.info(
                    f"  [{count}/{total}] g={g_val:.2f}, h0={h0_val:.2f}: "
                    f"speed={speed:.4f}, theory={c_theory:.4f}"
                )

    return {
        "g_values": np.array(all_g),
        "h0_values": np.array(all_h0),
        "speed_measured": np.array(all_speed_measured),
        "speed_theory": np.array(all_speed_theory),
    }


def generate_mass_conservation_data(
    n_runs: int = 20,
    n_steps: int = 5000,
    dt: float = 0.001,
    N: int = 128,
) -> dict[str, np.ndarray]:
    """Verify mass conservation over multiple runs with different parameters.

    Returns dict with arrays: g_values, h0_values, mass_initial, mass_final,
    relative_drift.
    """
    L = 10.0

    g_values = np.linspace(5.0, 15.0, n_runs)
    h0_values = np.linspace(0.5, 2.0, n_runs)

    all_g = []
    all_h0 = []
    all_mass_init = []
    all_mass_final = []
    all_drift = []

    for i in range(n_runs):
        sim = _make_sim(
            g=g_values[i], h0=h0_values[i], N=N, L=L, dt=dt,
            n_steps=n_steps, perturbation_amplitude=0.05 * h0_values[i],
        )
        sim.reset()
        mass0 = sim.total_mass

        for _ in range(n_steps):
            sim.step()

        mass_f = sim.total_mass
        drift = abs(mass_f - mass0) / abs(mass0) if abs(mass0) > 1e-15 else 0.0

        all_g.append(g_values[i])
        all_h0.append(h0_values[i])
        all_mass_init.append(mass0)
        all_mass_final.append(mass_f)
        all_drift.append(drift)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  [{i + 1}/{n_runs}] g={g_values[i]:.2f}, h0={h0_values[i]:.2f}: "
                f"mass drift={drift:.2e}"
            )

    return {
        "g_values": np.array(all_g),
        "h0_values": np.array(all_h0),
        "mass_initial": np.array(all_mass_init),
        "mass_final": np.array(all_mass_final),
        "relative_drift": np.array(all_drift),
    }


def run_shallow_water_rediscovery(
    output_dir: str | Path = "output/rediscovery/shallow_water",
    n_iterations: int = 40,
) -> dict:
    """Run the full shallow water equations rediscovery.

    1. Wave speed: c = sqrt(g*h) via parameter sweep + PySR
    2. Mass conservation verification
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "shallow_water",
        "targets": {
            "wave_speed": "c = sqrt(g * h)",
            "mass_conservation": "integral(h * dx) = const",
        },
    }

    # --- Part 1: Wave speed ---
    logger.info("Part 1: Measuring wave speed vs g and h0...")
    speed_data = generate_wave_speed_data(
        n_g=15, n_h=8, n_steps=2000, dt=0.001, N=256,
    )

    valid = np.isfinite(speed_data["speed_measured"])
    if np.sum(valid) > 10:
        rel_err = (
            np.abs(
                speed_data["speed_measured"][valid]
                - speed_data["speed_theory"][valid]
            )
            / speed_data["speed_theory"][valid]
        )
        results["wave_speed_data"] = {
            "n_samples": int(np.sum(valid)),
            "mean_relative_error": float(np.mean(rel_err)),
            "correlation": float(
                np.corrcoef(
                    speed_data["speed_measured"][valid],
                    speed_data["speed_theory"][valid],
                )[0, 1]
            ),
        }
        logger.info(f"  Mean relative error: {np.mean(rel_err):.4%}")

    # PySR: wave_speed = f(g, h0)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = np.column_stack([
            speed_data["g_values"][valid],
            speed_data["h0_values"][valid],
        ])
        y = speed_data["speed_measured"][valid]

        logger.info("Running PySR: wave_speed = f(g, h0)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["g_", "h0_"],
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

    # --- Part 2: Mass conservation ---
    logger.info("Part 2: Verifying mass conservation...")
    mass_data = generate_mass_conservation_data(n_runs=20, n_steps=5000, dt=0.001)

    results["mass_conservation"] = {
        "n_runs": len(mass_data["relative_drift"]),
        "mean_drift": float(np.mean(mass_data["relative_drift"])),
        "max_drift": float(np.max(mass_data["relative_drift"])),
    }
    logger.info(f"  Mean mass drift: {np.mean(mass_data['relative_drift']):.2e}")
    logger.info(f"  Max mass drift: {np.max(mass_data['relative_drift']):.2e}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
