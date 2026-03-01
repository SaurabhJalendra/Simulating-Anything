"""Spring-mass chain rediscovery.

Targets:
- Dispersion relation: omega(k) = 2*sqrt(K/m)*|sin(k*a/2)|
- Speed of sound: c = a*sqrt(K/m)
- Energy conservation in Hamiltonian system
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.spring_mass_chain import SpringMassChain
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_chain(
    N: int = 20,
    K: float = 4.0,
    m: float = 1.0,
    a: float = 1.0,
    mode: int = 1,
    amplitude: float = 0.1,
    dt: float = 0.001,
    n_steps: int = 10000,
) -> SpringMassChain:
    """Create a SpringMassChain simulation with the given parameters."""
    config = SimulationConfig(
        domain=Domain.SPRING_MASS_CHAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "N": float(N),
            "K": K,
            "m": m,
            "a": a,
            "mode": float(mode),
            "amplitude": amplitude,
        },
    )
    return SpringMassChain(config)


def generate_dispersion_data(
    N: int = 20,
    K: float = 4.0,
    m: float = 1.0,
    a: float = 1.0,
    dt: float = 0.001,
    n_steps: int = 50000,
) -> dict[str, np.ndarray]:
    """Measure normal mode frequencies by exciting each mode and finding period.

    For each mode n=1..N, excite a pure mode, run the simulation, and measure
    the oscillation frequency from zero crossings of the first mass displacement.
    Compare to the theoretical dispersion relation.

    Returns dict with arrays: mode_n, k_wave, omega_measured, omega_theory.
    """
    all_n = []
    all_k = []
    all_omega_measured = []
    all_omega_theory = []

    theoretical = 2.0 * np.sqrt(K / m) * np.sin(
        np.arange(1, N + 1) * np.pi / (2 * (N + 1))
    )

    for mode_n in range(1, N + 1):
        sim = _make_chain(
            N=N, K=K, m=m, a=a, mode=mode_n,
            amplitude=0.1, dt=dt, n_steps=n_steps,
        )
        sim.reset()

        # Track displacement of mass with largest mode amplitude
        mode_shape = sim.normal_mode_shape(mode_n)
        track_idx = int(np.argmax(np.abs(mode_shape)))

        positions = [sim.observe()[track_idx]]
        for _ in range(n_steps):
            state = sim.step()
            positions.append(state[track_idx])

        positions = np.array(positions)

        # Find positive-going zero crossings
        crossings = []
        for j in range(1, len(positions)):
            if positions[j - 1] < 0 and positions[j] >= 0:
                frac = -positions[j - 1] / (positions[j] - positions[j - 1])
                crossings.append((j - 1 + frac) * dt)

        if len(crossings) >= 3:
            periods = np.diff(crossings)
            T_measured = float(np.median(periods))
            omega_measured = 2 * np.pi / T_measured
        else:
            omega_measured = float("nan")

        k_wave = mode_n * np.pi / ((N + 1) * a)
        omega_theory = theoretical[mode_n - 1]

        all_n.append(mode_n)
        all_k.append(k_wave)
        all_omega_measured.append(omega_measured)
        all_omega_theory.append(omega_theory)

        if mode_n % 5 == 0:
            logger.info(
                f"  Mode {mode_n}/{N}: omega_measured={omega_measured:.4f}, "
                f"omega_theory={omega_theory:.4f}"
            )

    return {
        "mode_n": np.array(all_n),
        "k_wave": np.array(all_k),
        "omega_measured": np.array(all_omega_measured),
        "omega_theory": np.array(all_omega_theory),
    }


def generate_speed_of_sound_data(
    n_K_values: int = 25,
    N: int = 20,
    m: float = 1.0,
    a: float = 1.0,
    dt: float = 0.001,
    n_steps: int = 30000,
) -> dict[str, np.ndarray]:
    """Sweep spring constant K and measure wave propagation speed.

    The speed of sound c = a*sqrt(K/m) is the long-wavelength limit of the
    group velocity. We measure it from the lowest mode frequency:
    c_measured = omega_1 / k_1, where k_1 = pi/((N+1)*a).

    Returns dict with arrays: K_values, c_measured, c_theory.
    """
    K_values = np.linspace(0.5, 20.0, n_K_values)
    c_measured_list = []
    c_theory_list = []

    k1 = np.pi / ((N + 1) * a)  # Wave number of first mode

    for K_val in K_values:
        sim = _make_chain(
            N=N, K=K_val, m=m, a=a, mode=1,
            amplitude=0.1, dt=dt, n_steps=n_steps,
        )
        sim.reset()

        # Track a mass near chain center
        track_idx = N // 2
        positions = [sim.observe()[track_idx]]
        for _ in range(n_steps):
            state = sim.step()
            positions.append(state[track_idx])

        positions = np.array(positions)

        # Measure frequency from zero crossings
        crossings = []
        for j in range(1, len(positions)):
            if positions[j - 1] < 0 and positions[j] >= 0:
                frac = -positions[j - 1] / (positions[j] - positions[j - 1])
                crossings.append((j - 1 + frac) * dt)

        if len(crossings) >= 3:
            periods = np.diff(crossings)
            T = float(np.median(periods))
            omega1 = 2 * np.pi / T
            # Phase velocity for mode 1: c = omega / k
            c_meas = omega1 / k1
        else:
            c_meas = float("nan")

        c_theory = a * np.sqrt(K_val / m)
        c_measured_list.append(c_meas)
        c_theory_list.append(c_theory)

    return {
        "K_values": K_values,
        "c_measured": np.array(c_measured_list),
        "c_theory": np.array(c_theory_list),
    }


def generate_energy_conservation_data(
    n_trajectories: int = 10,
    N: int = 20,
    K: float = 4.0,
    m: float = 1.0,
    dt: float = 0.001,
    n_steps: int = 10000,
) -> dict[str, np.ndarray]:
    """Run multiple trajectories and verify energy conservation.

    Returns dict with arrays: E_initial, E_final, relative_drift.
    """
    E_initial = []
    E_final = []
    rel_drift = []

    for mode_n in range(1, min(n_trajectories + 1, N + 1)):
        sim = _make_chain(
            N=N, K=K, m=m, mode=mode_n,
            amplitude=0.5, dt=dt, n_steps=n_steps,
        )
        sim.reset()
        E0 = sim.total_energy

        for _ in range(n_steps):
            sim.step()

        Ef = sim.total_energy
        drift = abs(Ef - E0) / abs(E0) if E0 > 0 else 0.0

        E_initial.append(E0)
        E_final.append(Ef)
        rel_drift.append(drift)

    return {
        "E_initial": np.array(E_initial),
        "E_final": np.array(E_final),
        "relative_drift": np.array(rel_drift),
    }


def run_spring_mass_chain_rediscovery(
    output_dir: str | Path = "output/rediscovery/spring_mass_chain",
    n_iterations: int = 40,
) -> dict:
    """Run the full spring-mass chain rediscovery.

    1. Dispersion relation: omega(k) = 2*sqrt(K/m)*|sin(k*a/2)| via PySR
    2. Speed of sound: c = a*sqrt(K/m) via PySR
    3. Energy conservation verification
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "spring_mass_chain",
        "targets": {
            "dispersion": "omega = 2*sqrt(K/m)*|sin(k*a/2)|",
            "speed_of_sound": "c = a*sqrt(K/m)",
            "energy_conservation": "dE/dt = 0",
        },
    }

    # --- Part 1: Dispersion relation ---
    logger.info("Part 1: Measuring normal mode frequencies...")
    disp_data = generate_dispersion_data(
        N=20, K=4.0, m=1.0, a=1.0, dt=0.001, n_steps=50000,
    )

    valid = np.isfinite(disp_data["omega_measured"])
    if np.sum(valid) > 0:
        rel_error = (
            np.abs(
                disp_data["omega_measured"][valid]
                - disp_data["omega_theory"][valid]
            )
            / disp_data["omega_theory"][valid]
        )
        results["dispersion_accuracy"] = {
            "n_modes": int(np.sum(valid)),
            "mean_relative_error": float(np.mean(rel_error)),
            "max_relative_error": float(np.max(rel_error)),
        }
        logger.info(
            f"  Dispersion: {np.sum(valid)} modes, "
            f"mean error = {np.mean(rel_error):.4%}"
        )

    # PySR on omega vs sin(k*a/2) to recover the prefactor 2*sqrt(K/m)
    # Use k_wave as input and omega as output
    valid_mask = np.isfinite(disp_data["omega_measured"])
    k_vals = disp_data["k_wave"][valid_mask]
    omega_vals = disp_data["omega_measured"][valid_mask]

    if len(k_vals) >= 5:
        logger.info(
            f"  Running PySR for omega = f(k) with "
            f"{n_iterations} iterations..."
        )
        X = k_vals.reshape(-1, 1)
        y = omega_vals

        discoveries = run_symbolic_regression(
            X,
            y,
            variable_names=["k_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "sqrt", "square", "abs"],
            max_complexity=15,
            populations=20,
            population_size=40,
        )

        results["dispersion_pysr"] = {
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
            results["dispersion_pysr"]["best"] = best.expression
            results["dispersion_pysr"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )

    # --- Part 2: Speed of sound ---
    logger.info("Part 2: Speed of sound c = a*sqrt(K/m)...")
    sos_data = generate_speed_of_sound_data(
        n_K_values=25, N=20, m=1.0, a=1.0, dt=0.001, n_steps=30000,
    )

    valid_sos = np.isfinite(sos_data["c_measured"])
    if np.sum(valid_sos) > 0:
        sos_err = (
            np.abs(
                sos_data["c_measured"][valid_sos]
                - sos_data["c_theory"][valid_sos]
            )
            / sos_data["c_theory"][valid_sos]
        )
        results["speed_of_sound_accuracy"] = {
            "n_samples": int(np.sum(valid_sos)),
            "mean_relative_error": float(np.mean(sos_err)),
        }
        logger.info(
            f"  Speed of sound: {np.sum(valid_sos)} samples, "
            f"mean error = {np.mean(sos_err):.4%}"
        )

    # PySR: c = f(K) with m=1, a=1
    K_valid = sos_data["K_values"][valid_sos]
    c_valid = sos_data["c_measured"][valid_sos]

    if len(K_valid) >= 5:
        logger.info(
            f"  Running PySR for c = f(K) with "
            f"{n_iterations} iterations..."
        )
        X_sos = K_valid.reshape(-1, 1)
        y_sos = c_valid

        sos_discoveries = run_symbolic_regression(
            X_sos,
            y_sos,
            variable_names=["K_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=10,
            populations=20,
            population_size=40,
        )

        results["speed_of_sound_pysr"] = {
            "n_discoveries": len(sos_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sos_discoveries[:5]
            ],
        }
        if sos_discoveries:
            best = sos_discoveries[0]
            results["speed_of_sound_pysr"]["best"] = best.expression
            results["speed_of_sound_pysr"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )

    # --- Part 3: Energy conservation ---
    logger.info("Part 3: Energy conservation verification...")
    energy_data = generate_energy_conservation_data(
        n_trajectories=10, N=20, K=4.0, m=1.0, dt=0.001, n_steps=10000,
    )
    results["energy_conservation"] = {
        "n_trajectories": len(energy_data["relative_drift"]),
        "mean_relative_drift": float(
            np.mean(energy_data["relative_drift"])
        ),
        "max_relative_drift": float(
            np.max(energy_data["relative_drift"])
        ),
    }
    logger.info(
        f"  Energy: mean drift = "
        f"{np.mean(energy_data['relative_drift']):.2e}, "
        f"max drift = {np.max(energy_data['relative_drift']):.2e}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save raw data
    np.savez(
        output_path / "dispersion_data.npz",
        mode_n=disp_data["mode_n"],
        k_wave=disp_data["k_wave"],
        omega_measured=disp_data["omega_measured"],
        omega_theory=disp_data["omega_theory"],
    )
    np.savez(
        output_path / "speed_of_sound_data.npz",
        K_values=sos_data["K_values"],
        c_measured=sos_data["c_measured"],
        c_theory=sos_data["c_theory"],
    )

    return results
