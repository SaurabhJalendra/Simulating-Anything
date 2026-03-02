"""Double well potential rediscovery.

Targets:
- ODE: dv/dt = x - x^3 - gamma*v  (via SINDy)
- Energy conservation (gamma=0 case)
- Well frequency: omega = sqrt(2) for small oscillations about x = +/-1
- Barrier height: V(0) - V(+/-1) = 1/4
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.double_well import DoubleWellSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


_DOMAIN = Domain.DOUBLE_WELL


def generate_ode_data(
    gamma: float = 0.5,
    n_steps: int = 10000,
    dt: float = 0.005,
    x_0: float = 0.5,
    v_0: float = 0.3,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses a deterministic trajectory (sigma=0) with moderate damping so
    that SINDy can cleanly identify the ODE coefficients.

    Returns:
        Dictionary with time, states, and parameter values.
    """
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "gamma": gamma,
            "sigma": 0.0,
            "x_0": x_0,
            "v_0": v_0,
        },
    )
    sim = DoubleWellSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    times = np.arange(n_steps + 1) * dt

    return {
        "time": times,
        "states": states,
        "x": states[:, 0],
        "v": states[:, 1],
        "gamma": gamma,
        "dt": dt,
    }


def generate_energy_conservation_data(
    n_trajectories: int = 50,
    n_steps: int = 10000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Track total energy for the conservative system (gamma=0, sigma=0).

    Energy should be conserved to within numerical tolerance for RK4.

    Returns:
        Dictionary with energy drift statistics.
    """
    rng = np.random.default_rng(42)

    all_final_drift: list[float] = []
    all_max_drift: list[float] = []
    all_mean_energy: list[float] = []

    for i in range(n_trajectories):
        x_0 = rng.uniform(-1.5, 1.5)
        v_0 = rng.uniform(-1.0, 1.0)

        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "gamma": 0.0,
                "sigma": 0.0,
                "x_0": x_0,
                "v_0": v_0,
            },
        )
        sim = DoubleWellSimulation(config)
        sim.reset()

        E0 = sim.total_energy
        energies = [E0]
        for _ in range(n_steps):
            sim.step()
            energies.append(sim.total_energy)

        energies_arr = np.array(energies)
        drift = np.abs(energies_arr - E0) / max(abs(E0), 1e-10)

        all_final_drift.append(float(drift[-1]))
        all_max_drift.append(float(np.max(drift)))
        all_mean_energy.append(float(np.mean(energies_arr)))

        if (i + 1) % 10 == 0:
            logger.info(
                f"  Energy check {i + 1}/{n_trajectories}: "
                f"max drift = {np.max(drift):.2e}"
            )

    return {
        "n_trajectories": n_trajectories,
        "final_drift": np.array(all_final_drift),
        "max_drift": np.array(all_max_drift),
        "mean_energy": np.array(all_mean_energy),
    }


def generate_well_frequency_data(
    n_samples: int = 30,
    n_steps: int = 20000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Measure oscillation frequency about the right well minimum x=+1.

    For small oscillations, V''(1) = 2, so omega = sqrt(2) and T = 2*pi/sqrt(2).

    Returns:
        Dictionary with measured and theoretical frequencies.
    """
    amplitudes: list[float] = []
    omega_measured_list: list[float] = []
    omega_theory = np.sqrt(2.0)
    T_theory = 2 * np.pi / omega_theory

    for i in range(n_samples):
        # Small perturbation from x = 1.0
        amp = 0.01 + 0.02 * i / n_samples
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "gamma": 0.0,
                "sigma": 0.0,
                "x_0": 1.0 + amp,
                "v_0": 0.0,
            },
        )
        sim = DoubleWellSimulation(config)
        sim.reset()

        # Collect x - 1.0 displacements
        prev_dx = sim.observe()[0] - 1.0
        crossings: list[float] = []
        for step_idx in range(n_steps):
            state = sim.step()
            dx = state[0] - 1.0
            if prev_dx < 0 and dx >= 0:
                frac = -prev_dx / (dx - prev_dx)
                crossings.append((step_idx + frac) * dt)
            prev_dx = dx

        if len(crossings) >= 3:
            periods = np.diff(crossings)
            T_meas = float(np.median(periods))
            omega_meas = 2 * np.pi / T_meas
            amplitudes.append(amp)
            omega_measured_list.append(omega_meas)

        if (i + 1) % 10 == 0:
            logger.info(f"  Frequency measurement {i + 1}/{n_samples}")

    return {
        "amplitudes": np.array(amplitudes),
        "omega_measured": np.array(omega_measured_list),
        "omega_theory": omega_theory,
        "T_theory": T_theory,
    }


def generate_barrier_verification_data(
    n_energies: int = 20,
    n_steps: int = 50000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Verify barrier height by checking which initial energies allow crossing.

    Start particle at x = 1.0 (right well) with varying velocities.
    If E > V(0) = 0, the particle can cross the barrier.

    Returns:
        Dictionary with initial energies and whether crossing occurred.
    """
    v_values = np.linspace(0.1, 1.5, n_energies)
    crossed: list[bool] = []
    initial_energies: list[float] = []

    for v_0 in v_values:
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "gamma": 0.0,
                "sigma": 0.0,
                "x_0": 1.0,
                "v_0": v_0,
            },
        )
        sim = DoubleWellSimulation(config)
        sim.reset()
        initial_energies.append(sim.total_energy)

        did_cross = False
        for _ in range(n_steps):
            sim.step()
            if sim.observe()[0] < 0:
                did_cross = True
                break

        crossed.append(did_cross)

    return {
        "v_0": v_values,
        "initial_energy": np.array(initial_energies),
        "crossed": np.array(crossed),
        "barrier_energy": 0.0,  # V(0) = 0
    }


def run_double_well_rediscovery(
    output_dir: str | Path = "output/rediscovery/double_well",
    n_iterations: int = 40,
) -> dict:
    """Run the full double well potential rediscovery pipeline.

    1. SINDy ODE recovery: dv/dt = x - x^3 - gamma*v
    2. Energy conservation verification (gamma=0)
    3. Well frequency measurement: omega = sqrt(2)
    4. Barrier height verification
    5. Barrier crossing verification

    Args:
        output_dir: Directory to save results.
        n_iterations: Number of PySR iterations.

    Returns:
        Results dict with all discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "double_well",
        "targets": {
            "ode": "dv/dt = x - x^3 - gamma*v",
            "energy_conservation": "H = v^2/2 + x^4/4 - x^2/2 = const (gamma=0)",
            "well_frequency": "omega = sqrt(2)",
            "barrier_height": "Delta V = 1/4",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery (gamma=0.5)...")
    data = generate_ode_data(gamma=0.5, n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=data["dt"],
            feature_names=["x", "v"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries[:5]
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
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Energy conservation ---
    logger.info("Part 2: Energy conservation verification (gamma=0)...")
    energy_data = generate_energy_conservation_data(
        n_trajectories=50, n_steps=10000, dt=0.001
    )

    results["energy_conservation"] = {
        "n_trajectories": energy_data["n_trajectories"],
        "mean_final_drift": float(np.mean(energy_data["final_drift"])),
        "max_final_drift": float(np.max(energy_data["final_drift"])),
        "mean_max_drift": float(np.mean(energy_data["max_drift"])),
    }
    logger.info(
        f"  Energy drift: mean={np.mean(energy_data['final_drift']):.2e}, "
        f"max={np.max(energy_data['final_drift']):.2e}"
    )

    # --- Part 3: Well frequency ---
    logger.info("Part 3: Well frequency measurement (omega = sqrt(2))...")
    freq_data = generate_well_frequency_data(n_samples=30, n_steps=20000, dt=0.001)

    omega_theory = freq_data["omega_theory"]
    if len(freq_data["omega_measured"]) > 0:
        rel_errors = (
            np.abs(freq_data["omega_measured"] - omega_theory) / omega_theory
        )
        results["well_frequency"] = {
            "omega_theory": float(omega_theory),
            "n_measurements": len(freq_data["omega_measured"]),
            "omega_mean": float(np.mean(freq_data["omega_measured"])),
            "mean_relative_error": float(np.mean(rel_errors)),
            "max_relative_error": float(np.max(rel_errors)),
        }
        logger.info(
            f"  omega: theory={omega_theory:.4f}, "
            f"measured={np.mean(freq_data['omega_measured']):.4f}, "
            f"error={np.mean(rel_errors):.4%}"
        )
    else:
        results["well_frequency"] = {"error": "No oscillations detected"}

    # --- Part 4: Barrier verification ---
    logger.info("Part 4: Barrier crossing verification...")
    barrier_data = generate_barrier_verification_data(
        n_energies=20, n_steps=50000, dt=0.005
    )

    # The critical energy is E = 0 (= V(0))
    # Particles with E > 0 should cross, E < 0 should not
    e_crit_theory = 0.0
    crossed = barrier_data["crossed"]
    energies = barrier_data["initial_energy"]

    if np.any(crossed) and np.any(~crossed):
        # Find the boundary
        first_cross_idx = int(np.argmax(crossed))
        e_critical_est = float(energies[first_cross_idx])
    else:
        e_critical_est = float("nan")

    results["barrier_crossing"] = {
        "n_energies": len(energies),
        "n_crossed": int(np.sum(crossed)),
        "n_trapped": int(np.sum(~crossed)),
        "barrier_energy_theory": e_crit_theory,
        "barrier_energy_estimate": e_critical_est,
        "barrier_height": float(DoubleWellSimulation.barrier_height()),
    }
    logger.info(
        f"  Barrier: theory E_c=0, estimate={e_critical_est:.4f}, "
        f"height={DoubleWellSimulation.barrier_height():.4f}"
    )

    # --- Part 5: PySR well frequency ---
    if len(freq_data["omega_measured"]) >= 5:
        try:
            from simulating_anything.analysis.symbolic_regression import (
                run_symbolic_regression,
            )

            # omega should be constant ~ sqrt(2), independent of amplitude
            # Use amplitude as variable to show omega is constant
            X = freq_data["amplitudes"].reshape(-1, 1)
            y = freq_data["omega_measured"]

            logger.info("  Running PySR: omega = f(amplitude)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["amp"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=8,
                populations=20,
                population_size=40,
            )
            results["frequency_pysr"] = {
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
                results["frequency_pysr"]["best"] = best.expression
                results["frequency_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        except Exception as e:
            logger.warning(f"PySR failed: {e}")
            results["frequency_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "energy_data.npz",
        final_drift=energy_data["final_drift"],
        max_drift=energy_data["max_drift"],
    )
    np.savez(
        output_path / "frequency_data.npz",
        amplitudes=freq_data["amplitudes"],
        omega_measured=freq_data["omega_measured"],
    )
    np.savez(
        output_path / "barrier_data.npz",
        v_0=barrier_data["v_0"],
        initial_energy=barrier_data["initial_energy"],
        crossed=barrier_data["crossed"],
    )

    return results
