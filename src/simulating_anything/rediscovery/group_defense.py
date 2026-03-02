"""Group defense predator-prey rediscovery.

Targets:
- Non-monotone functional response: f(N) = a*N/(1 + b*N + c*N^2)
- Peak at N_max = 1/sqrt(c)
- ODE recovery via SINDy
- Group defense effect: predation decreases at high prey density
- Bifurcation analysis: sweep c to find transitions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.group_defense import GroupDefenseSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use DUFFING as placeholder domain enum
_DOMAIN = Domain.GROUP_DEFENSE


def _make_config(
    r: float = 1.0,
    K: float = 15.0,
    a: float = 1.5,
    b: float = 0.1,
    c: float = 0.05,
    e: float = 0.5,
    d: float = 0.3,
    N_0: float = 5.0,
    P_0: float = 2.0,
    dt: float = 0.01,
    n_steps: int = 500,
) -> SimulationConfig:
    """Build a SimulationConfig for the group defense model."""
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "b": b, "c": c,
            "e": e, "d": d, "N_0": N_0, "P_0": P_0,
        },
    )


def generate_trajectory_data(
    r: float = 1.0,
    K: float = 15.0,
    a: float = 1.5,
    b: float = 0.1,
    c: float = 0.05,
    e: float = 0.5,
    d: float = 0.3,
    N_0: float = 5.0,
    P_0: float = 2.0,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Generate a single long trajectory for SINDy ODE recovery.

    Returns:
        Dict with states array, time, and parameters.
    """
    config = _make_config(
        r=r, K=K, a=a, b=b, c=c, e=e, d=d,
        N_0=N_0, P_0=P_0, dt=dt, n_steps=n_steps,
    )

    sim = GroupDefenseSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "N": states_arr[:, 0],
        "P": states_arr[:, 1],
        "dt": dt,
        "r": r, "K": K, "a": a, "b": b, "c": c, "e": e, "d": d,
    }


def generate_functional_response_data(
    N_max: float = 30.0,
    n_points: int = 200,
    a: float = 1.5,
    b: float = 0.1,
    c: float = 0.05,
) -> dict[str, np.ndarray | float]:
    """Generate functional response curve f(N) for various N values.

    The Andrews functional response f(N) = a*N/(1 + b*N + c*N^2)
    should show a peak at N = 1/sqrt(c) and then decline.

    Args:
        N_max: Maximum prey density to evaluate.
        n_points: Number of evaluation points.
        a: Attack rate.
        b: Handling time coefficient.
        c: Group defense coefficient.

    Returns:
        Dict with N values, f(N) values, theoretical peak location.
    """
    N_values = np.linspace(0.01, N_max, n_points)
    f_values = a * N_values / (1.0 + b * N_values + c * N_values**2)

    N_peak_theory = 1.0 / np.sqrt(c)
    f_peak_theory = a / (b + 2.0 * np.sqrt(c))

    # Find empirical peak
    peak_idx = np.argmax(f_values)
    N_peak_empirical = N_values[peak_idx]
    f_peak_empirical = f_values[peak_idx]

    return {
        "N_values": N_values,
        "f_values": f_values,
        "N_peak_theory": N_peak_theory,
        "f_peak_theory": f_peak_theory,
        "N_peak_empirical": N_peak_empirical,
        "f_peak_empirical": f_peak_empirical,
        "a": a,
        "b": b,
        "c": c,
    }


def generate_defense_sweep_data(
    n_c_values: int = 20,
    c_min: float = 0.001,
    c_max: float = 0.2,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep group defense coefficient c and track population dynamics.

    For each c value, runs a long trajectory and measures:
    - Time-averaged prey and predator populations
    - Oscillation amplitude (max - min in second half)
    - Peak location of the functional response

    Args:
        n_c_values: Number of c values to sweep.
        c_min: Minimum group defense coefficient.
        c_max: Maximum group defense coefficient.
        n_steps: Simulation steps per run.
        dt: Timestep.

    Returns:
        Dict with c values, averages, amplitudes, and peak info.
    """
    c_values = np.linspace(c_min, c_max, n_c_values)

    all_c = []
    all_N_avg = []
    all_P_avg = []
    all_N_amp = []
    all_P_amp = []
    all_N_peak = []

    for i, c_val in enumerate(c_values):
        config = _make_config(c=float(c_val), dt=dt, n_steps=n_steps)
        sim = GroupDefenseSimulation(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)
        half = n_steps // 2
        N_half = trajectory[half:, 0]
        P_half = trajectory[half:, 1]

        all_c.append(float(c_val))
        all_N_avg.append(float(np.mean(N_half)))
        all_P_avg.append(float(np.mean(P_half)))
        all_N_amp.append(float(np.max(N_half) - np.min(N_half)))
        all_P_amp.append(float(np.max(P_half) - np.min(P_half)))
        all_N_peak.append(1.0 / np.sqrt(c_val))

        if (i + 1) % 5 == 0:
            logger.info(f"  Defense sweep: {i + 1}/{n_c_values} c values")

    return {
        "c": np.array(all_c),
        "N_avg": np.array(all_N_avg),
        "P_avg": np.array(all_P_avg),
        "N_amplitude": np.array(all_N_amp),
        "P_amplitude": np.array(all_P_amp),
        "N_peak": np.array(all_N_peak),
    }


def generate_peak_location_data(
    n_c_values: int = 25,
    c_min: float = 0.005,
    c_max: float = 0.5,
) -> dict[str, np.ndarray]:
    """Generate data for PySR to recover N_max = 1/sqrt(c).

    For each c value, computes the empirical peak of the functional response
    and the theoretical value 1/sqrt(c).

    Args:
        n_c_values: Number of c values.
        c_min: Minimum c.
        c_max: Maximum c.

    Returns:
        Dict with c values, empirical peaks, and theoretical peaks.
    """
    c_values = np.linspace(c_min, c_max, n_c_values)
    N_peak_theory = 1.0 / np.sqrt(c_values)

    # Empirical peak: evaluate f(N) over fine grid and find argmax
    N_peak_empirical = np.zeros(n_c_values)
    for i, c_val in enumerate(c_values):
        N_test = np.linspace(0.01, 50.0, 5000)
        f_test = 1.5 * N_test / (1.0 + 0.1 * N_test + c_val * N_test**2)
        N_peak_empirical[i] = N_test[np.argmax(f_test)]

    return {
        "c_values": c_values,
        "N_peak_theory": N_peak_theory,
        "N_peak_empirical": N_peak_empirical,
    }


def run_group_defense_rediscovery(
    output_dir: str | Path = "output/rediscovery/group_defense",
    n_iterations: int = 40,
    n_c_values: int = 20,
) -> dict:
    """Run the full group defense predator-prey rediscovery pipeline.

    1. Verify the non-monotone functional response shape
    2. Recover peak location N_max = 1/sqrt(c) via PySR
    3. SINDy ODE recovery
    4. Sweep c to detect bifurcation transitions

    Args:
        output_dir: Directory for output files.
        n_iterations: Number of PySR iterations.
        n_c_values: Number of c values for parameter sweep.

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "group_defense",
        "targets": {
            "functional_response": "f(N) = a*N/(1 + b*N + c*N^2)",
            "peak_location": "N_max = 1/sqrt(c)",
            "ode_N": "dN/dt = r*N*(1-N/K) - a*N*P/(1+b*N+c*N^2)",
            "ode_P": "dP/dt = e*a*N*P/(1+b*N+c*N^2) - d*P",
        },
    }

    # --- Part 1: Functional response verification ---
    logger.info("Part 1: Verifying non-monotone functional response...")
    fr_data = generate_functional_response_data()
    results["functional_response"] = {
        "N_peak_theory": float(fr_data["N_peak_theory"]),
        "N_peak_empirical": float(fr_data["N_peak_empirical"]),
        "f_peak_theory": float(fr_data["f_peak_theory"]),
        "f_peak_empirical": float(fr_data["f_peak_empirical"]),
        "peak_error_pct": float(
            abs(fr_data["N_peak_empirical"] - fr_data["N_peak_theory"])
            / fr_data["N_peak_theory"] * 100
        ),
        "is_non_monotone": bool(
            fr_data["f_values"][-1] < fr_data["f_peak_empirical"]
        ),
    }
    logger.info(
        f"  Peak: theory N={fr_data['N_peak_theory']:.3f}, "
        f"empirical N={fr_data['N_peak_empirical']:.3f}"
    )
    logger.info(
        f"  Non-monotone: {results['functional_response']['is_non_monotone']}"
    )

    # --- Part 2: PySR peak location recovery ---
    logger.info("Part 2: PySR recovery of N_max = 1/sqrt(c)...")
    peak_data = generate_peak_location_data(n_c_values=25)

    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X_peak = peak_data["c_values"].reshape(-1, 1)
        y_peak = peak_data["N_peak_empirical"]

        pysr_discoveries = run_symbolic_regression(
            X_peak,
            y_peak,
            variable_names=["c_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square", "inv"],
            max_complexity=10,
            populations=20,
            population_size=40,
        )

        results["pysr_peak"] = {
            "n_discoveries": len(pysr_discoveries),
            "discoveries": [
                {"expression": disc.expression, "r_squared": disc.evidence.fit_r_squared}
                for disc in pysr_discoveries[:5]
            ],
        }
        if pysr_discoveries:
            best = pysr_discoveries[0]
            results["pysr_peak"]["best"] = best.expression
            results["pysr_peak"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  PySR peak: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as exc:
        logger.warning(f"PySR failed: {exc}")
        results["pysr_peak"] = {"error": str(exc)}

    # --- Part 3: SINDy ODE recovery ---
    logger.info("Part 3: SINDy ODE recovery for group defense model...")
    traj_data = generate_trajectory_data(n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            traj_data["states"],
            dt=0.005,
            feature_names=["N", "P"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": disc.expression, "r_squared": disc.evidence.fit_r_squared}
                for disc in sindy_discoveries
            ],
        }
        if sindy_discoveries:
            best = sindy_discoveries[0]
            results["sindy_ode"]["best"] = best.expression
            results["sindy_ode"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  SINDy best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
        for disc in sindy_discoveries:
            logger.info(f"  SINDy: {disc.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 4: Group defense parameter sweep ---
    logger.info("Part 4: Sweeping group defense coefficient c...")
    sweep_data = generate_defense_sweep_data(
        n_c_values=n_c_values, c_min=0.001, c_max=0.2,
        n_steps=30000, dt=0.01,
    )
    results["defense_sweep"] = {
        "n_c_values": n_c_values,
        "c_range": [float(sweep_data["c"][0]), float(sweep_data["c"][-1])],
        "max_N_amplitude": float(np.max(sweep_data["N_amplitude"])),
        "max_P_amplitude": float(np.max(sweep_data["P_amplitude"])),
        "mean_N": float(np.mean(sweep_data["N_avg"])),
        "mean_P": float(np.mean(sweep_data["P_avg"])),
    }
    logger.info(
        f"  Max prey amplitude: {np.max(sweep_data['N_amplitude']):.4f}"
    )

    # --- Part 5: Equilibria computation ---
    logger.info("Part 5: Computing equilibria...")
    config = _make_config()
    sim = GroupDefenseSimulation(config)
    equilibria = sim.find_equilibria()
    results["equilibria"] = equilibria
    for eq in equilibria:
        logger.info(f"  {eq['type']}: N={eq['N']:.4f}, P={eq['P']:.4f}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "trajectory_data.npz",
        states=traj_data["states"],
    )
    np.savez(
        output_path / "sweep_data.npz",
        **{k: v for k, v in sweep_data.items()},
    )

    return results
