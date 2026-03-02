"""Advection 1D rediscovery.

Targets:
- Pulse speed equals wave speed c (PySR: speed = f(c))
- Numerical diffusion: pulse broadening over time
- Exact solution comparison: u(x,t) = u0(x - c*t)
- CFL stability threshold: Courant number must be <= 1
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.advection_1d import Advection1DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_sim(
    c: float = 1.0,
    N: int = 256,
    L: float = 10.0,
    sigma: float = 0.5,
    dt: float | None = None,
    n_steps: int = 500,
) -> Advection1DSimulation:
    """Create an Advection1DSimulation with the given parameters."""
    dx = L / N
    if dt is None:
        dt = 0.8 * dx / abs(c) if abs(c) > 0 else 0.01
    config = SimulationConfig(
        domain=Domain.ADVECTION_1D,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "c": c,
            "N": float(N),
            "L": L,
            "sigma": sigma,
        },
    )
    return Advection1DSimulation(config)


def generate_speed_data(
    n_c: int = 25,
    n_steps: int = 200,
    N: int = 256,
    L: float = 10.0,
) -> dict[str, np.ndarray]:
    """Measure pulse propagation speed vs wave speed parameter c.

    For each c value, run the simulation and track the peak position
    over time to measure the actual propagation speed.

    Returns dict with arrays: c_values, speed_measured, speed_theory.
    """
    c_values = np.linspace(0.5, 5.0, n_c)
    speed_measured = []
    speed_theory = []

    for i, c_val in enumerate(c_values):
        sim = _make_sim(c=c_val, N=N, L=L, n_steps=n_steps)
        sim.reset()

        # Track peak incrementally to handle periodic wrapping correctly.
        # Accumulate displacement over small intervals so each single-step
        # displacement is always less than L/2.
        actual_steps = min(n_steps, 200)
        total_disp = 0.0
        prev_pos = sim.peak_position
        for _ in range(actual_steps):
            sim.step()
            curr_pos = sim.peak_position
            dx = curr_pos - prev_pos
            # Correct for periodic wrapping (single-step displacement < L/2)
            if dx > L / 2:
                dx -= L
            elif dx < -L / 2:
                dx += L
            total_disp += dx
            prev_pos = curr_pos

        delta_t = actual_steps * sim.config.dt
        if delta_t > 0:
            speed = total_disp / delta_t
        else:
            speed = float("nan")

        speed_measured.append(speed)
        speed_theory.append(c_val)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  c={c_val:.2f}: speed_measured={speed:.4f}, "
                f"theory={c_val:.4f}"
            )

    return {
        "c_values": c_values,
        "speed_measured": np.array(speed_measured),
        "speed_theory": np.array(speed_theory),
    }


def generate_diffusion_data(
    n_steps_values: list[int] | None = None,
    c: float = 1.0,
    N: int = 256,
    L: float = 10.0,
    sigma0: float = 0.5,
) -> dict[str, np.ndarray]:
    """Measure numerical diffusion (pulse broadening) over time.

    The upwind scheme introduces numerical diffusion proportional to
    (1 - CFL) * c * dx / 2. The pulse width should grow over time.

    Returns dict with arrays: times, sigma_measured, sigma_initial.
    """
    if n_steps_values is None:
        n_steps_values = [0, 50, 100, 200, 300, 400, 500]

    times = []
    sigma_measured = []

    sim = _make_sim(c=c, N=N, L=L, sigma=sigma0, n_steps=max(n_steps_values))
    sim.reset()
    dt = sim.config.dt

    prev_steps = 0
    for n_s in sorted(n_steps_values):
        for _ in range(n_s - prev_steps):
            sim.step()
        prev_steps = n_s

        t = n_s * dt
        u = sim.observe()

        # Estimate pulse width as second moment around peak
        peak_idx = int(np.argmax(u))
        x_centered = sim.x - sim.x[peak_idx]
        # Wrap for periodic domain
        x_centered[x_centered > L / 2] -= L
        x_centered[x_centered < -L / 2] += L

        total = np.sum(u)
        if total > 1e-10:
            variance = np.sum(u * x_centered ** 2) / total
            sigma_est = np.sqrt(max(variance, 0.0))
        else:
            sigma_est = float("nan")

        times.append(t)
        sigma_measured.append(sigma_est)

    return {
        "times": np.array(times),
        "sigma_measured": np.array(sigma_measured),
        "sigma_initial": sigma0,
    }


def generate_exact_comparison_data(
    c: float = 1.0,
    N: int = 256,
    L: float = 10.0,
    n_steps: int = 200,
) -> dict[str, np.ndarray]:
    """Compare numerical solution with exact solution over time.

    Returns dict with arrays: times, l2_errors.
    """
    sim = _make_sim(c=c, N=N, L=L, n_steps=n_steps)
    sim.reset()
    dt = sim.config.dt

    times = []
    l2_errors = []

    sample_steps = list(range(0, n_steps + 1, max(1, n_steps // 20)))
    prev = 0
    for n_s in sample_steps:
        for _ in range(n_s - prev):
            sim.step()
        prev = n_s

        t = n_s * dt
        err = sim.l2_error(t)
        times.append(t)
        l2_errors.append(err)

    return {
        "times": np.array(times),
        "l2_errors": np.array(l2_errors),
    }


def generate_cfl_stability_data(
    n_cfl: int = 20,
    c: float = 1.0,
    N: int = 64,
    L: float = 10.0,
    n_steps: int = 200,
) -> dict[str, np.ndarray]:
    """Sweep CFL number to demonstrate stability threshold.

    CFL <= 1 should be stable; CFL > 1 should blow up.

    Returns dict with arrays: cfl_values, max_u_final, stable.
    """
    dx = L / N
    dt_critical = dx / abs(c) if abs(c) > 0 else 1.0

    # Sweep CFL from 0.1 to 1.5
    cfl_values = np.linspace(0.1, 1.5, n_cfl)
    max_u_final = []
    stable_flags = []

    for i, cfl in enumerate(cfl_values):
        dt = cfl * dx / abs(c) if abs(c) > 0 else 0.01

        config = SimulationConfig(
            domain=Domain.ADVECTION_1D,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "c": c,
                "N": float(N),
                "L": L,
                "sigma": 0.5,
            },
        )
        sim = Advection1DSimulation(config)
        # Bypass CFL override to test instability
        sim.config = config
        sim.courant = cfl
        sim.reset()

        is_stable = True
        for _ in range(n_steps):
            sim.step()
            if np.any(np.isnan(sim.observe())) or np.any(np.isinf(sim.observe())):
                is_stable = False
                break
            if np.max(np.abs(sim.observe())) > 100.0:
                is_stable = False
                break

        max_val = float(np.max(np.abs(sim.observe()))) if is_stable else float("inf")
        max_u_final.append(max_val)
        stable_flags.append(is_stable)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  CFL={cfl:.2f}: stable={is_stable}, max_u={max_val:.4f}"
            )

    return {
        "cfl_values": cfl_values,
        "max_u_final": np.array(max_u_final),
        "stable": np.array(stable_flags),
        "dt_critical": dt_critical,
    }


def run_advection_1d_rediscovery(
    output_dir: str | Path = "output/rediscovery/advection_1d",
    n_iterations: int = 50,
) -> dict:
    """Run the full 1D advection equation rediscovery.

    1. Pulse speed = c (PySR: speed = f(c))
    2. Numerical diffusion measurement
    3. Comparison with exact solution
    4. CFL stability threshold
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "advection_1d",
        "targets": {
            "pulse_speed": "speed = c",
            "exact_solution": "u(x,t) = u0(x - c*t)",
            "cfl_stability": "stable iff CFL = |c|*dt/dx <= 1",
        },
    }

    # --- Part 1: Pulse speed vs c ---
    logger.info("Part 1: Measuring pulse speed vs wave speed c...")
    speed_data = generate_speed_data(n_c=25, n_steps=200)

    valid = np.isfinite(speed_data["speed_measured"])
    if np.sum(valid) > 5:
        rel_err = (
            np.abs(
                speed_data["speed_measured"][valid]
                - speed_data["speed_theory"][valid]
            )
            / np.abs(speed_data["speed_theory"][valid])
        )
        corr = float(
            np.corrcoef(
                speed_data["speed_measured"][valid],
                speed_data["speed_theory"][valid],
            )[0, 1]
        )
        results["speed_data"] = {
            "n_samples": int(np.sum(valid)),
            "mean_relative_error": float(np.mean(rel_err)),
            "correlation": corr,
        }
        logger.info(
            f"  {int(np.sum(valid))} samples, "
            f"mean rel error: {np.mean(rel_err):.4%}, "
            f"correlation: {corr:.6f}"
        )

    # PySR: speed = f(c)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = speed_data["c_values"][valid].reshape(-1, 1)
        y = speed_data["speed_measured"][valid]

        logger.info("Running PySR: speed = f(c)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["c_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt"],
            max_complexity=6,
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

    # --- Part 2: Numerical diffusion ---
    logger.info("Part 2: Measuring numerical diffusion...")
    diff_data = generate_diffusion_data()

    valid_diff = np.isfinite(diff_data["sigma_measured"])
    if np.sum(valid_diff) > 2:
        results["diffusion_data"] = {
            "n_samples": int(np.sum(valid_diff)),
            "sigma_initial": float(diff_data["sigma_initial"]),
            "sigma_final": float(diff_data["sigma_measured"][-1]),
            "broadening_ratio": float(
                diff_data["sigma_measured"][-1] / diff_data["sigma_initial"]
            ),
        }
        logger.info(
            f"  Initial sigma: {diff_data['sigma_initial']:.4f}, "
            f"Final sigma: {diff_data['sigma_measured'][-1]:.4f}"
        )

    # --- Part 3: Exact solution comparison ---
    logger.info("Part 3: Comparison with exact solution...")
    exact_data = generate_exact_comparison_data()

    results["exact_comparison"] = {
        "n_time_points": len(exact_data["times"]),
        "max_l2_error": float(np.max(exact_data["l2_errors"])),
        "mean_l2_error": float(np.mean(exact_data["l2_errors"])),
    }
    logger.info(f"  Max L2 error: {np.max(exact_data['l2_errors']):.6f}")

    # --- Part 4: CFL stability sweep ---
    logger.info("Part 4: CFL stability sweep...")
    cfl_data = generate_cfl_stability_data(n_cfl=20)

    n_stable = int(np.sum(cfl_data["stable"]))
    n_total = len(cfl_data["stable"])

    # Find transition point
    if n_stable < n_total:
        transition_idx = np.argmin(cfl_data["stable"])
        cfl_critical = float(cfl_data["cfl_values"][transition_idx])
    else:
        cfl_critical = float(cfl_data["cfl_values"][-1])

    results["cfl_stability"] = {
        "n_stable": n_stable,
        "n_total": n_total,
        "cfl_critical_measured": cfl_critical,
        "cfl_critical_theory": 1.0,
    }
    logger.info(
        f"  {n_stable}/{n_total} stable, "
        f"critical CFL ~ {cfl_critical:.2f} (theory: 1.0)"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
