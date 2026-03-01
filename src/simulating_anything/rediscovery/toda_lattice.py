"""Toda lattice rediscovery.

Targets:
- Energy conservation: E = sum(0.5*p_i^2 + exp(-(x_{i+1}-x_i))) = const
- Momentum conservation: sum(p_i) = const (periodic BC)
- Soliton speed measurement
- Harmonic limit: small amplitudes reproduce spring-mass chain behavior
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.toda_lattice import TodaLattice
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_lattice(
    N: int = 8,
    a: float = 1.0,
    mode: int = 1,
    amplitude: float = 0.1,
    dt: float = 0.001,
    n_steps: int = 10000,
) -> TodaLattice:
    """Create a TodaLattice simulation with the given parameters."""
    config = SimulationConfig(
        domain=Domain.TODA_LATTICE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "N": float(N),
            "a": a,
            "mode": float(mode),
            "amplitude": amplitude,
        },
    )
    return TodaLattice(config)


def generate_energy_conservation_data(
    n_trajectories: int = 10,
    N: int = 8,
    a: float = 1.0,
    dt: float = 0.001,
    n_steps: int = 10000,
) -> dict[str, np.ndarray]:
    """Run multiple trajectories and verify energy conservation.

    Each trajectory uses a different initial mode excitation.

    Returns dict with arrays: E_initial, E_final, relative_drift.
    """
    E_initial = []
    E_final = []
    rel_drift = []

    for mode_n in range(1, min(n_trajectories + 1, N)):
        sim = _make_lattice(
            N=N, a=a, mode=mode_n,
            amplitude=0.3, dt=dt, n_steps=n_steps,
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


def generate_soliton_data(
    N: int = 64,
    a: float = 1.0,
    n_amplitudes: int = 15,
    dt: float = 0.001,
    n_steps: int = 5000,
) -> dict[str, np.ndarray]:
    """Measure soliton propagation speed for different amplitudes.

    A soliton in the Toda lattice has speed c related to its amplitude A by:
        c = sinh(kappa) / kappa, where A ~ exp(-kappa) - related

    For small amplitudes, c -> 1 (sound speed = sqrt(a)).
    For large amplitudes, solitons travel faster than sound.

    We initialize a localized pulse and track its peak position over time
    to measure propagation speed.

    Returns dict with arrays: amplitudes, speeds, sound_speed.
    """
    amplitudes = np.linspace(0.05, 2.0, n_amplitudes)
    speeds = []
    sound_speed = np.sqrt(a)

    for amp in amplitudes:
        # Create a localized initial pulse at the center of the chain
        config = SimulationConfig(
            domain=Domain.TODA_LATTICE,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "N": float(N),
                "a": a,
                "mode": 0.0,  # Will override manually
                "amplitude": 0.0,
            },
        )
        sim = TodaLattice(config)
        sim.reset()

        # Set up localized pulse: compress one bond
        x = np.zeros(N, dtype=np.float64)
        p = np.zeros(N, dtype=np.float64)
        center = N // 4
        # Give a momentum kick to one particle
        p[center] = amp
        sim._state = np.concatenate([x, p])

        # Track the peak momentum position over time
        peak_positions = []
        peak_times = []
        for step_i in range(n_steps):
            sim.step()
            momenta = sim.momenta
            peak_idx = int(np.argmax(np.abs(momenta)))
            # Unwrap periodic boundary to track continuous motion
            peak_positions.append(peak_idx)
            peak_times.append((step_i + 1) * dt)

        # Estimate speed from peak motion over last half of simulation
        half = n_steps // 2
        positions = np.array(peak_positions[half:])
        times = np.array(peak_times[half:])

        # Unwrap positions for periodic boundary
        unwrapped = np.unwrap(positions * 2 * np.pi / N) * N / (2 * np.pi)

        if len(unwrapped) > 10:
            # Linear fit to get speed
            coeffs = np.polyfit(times, unwrapped, 1)
            speed = abs(coeffs[0])
        else:
            speed = float("nan")

        speeds.append(speed)

    return {
        "amplitudes": amplitudes,
        "speeds": np.array(speeds),
        "sound_speed": sound_speed,
    }


def generate_harmonic_limit_data(
    N: int = 8,
    a: float = 1.0,
    n_amplitudes: int = 10,
    dt: float = 0.001,
    n_steps: int = 20000,
) -> dict[str, np.ndarray]:
    """Verify that small amplitudes reproduce harmonic chain frequencies.

    In the harmonic limit, the Toda lattice with coupling a has the same
    normal mode spectrum as a spring-mass chain with spring constant K = a.
    For periodic BC: omega_n = 2*sqrt(a)*|sin(pi*n/N)|.

    Measure the oscillation frequency of mode 1 at different amplitudes.
    At small amplitudes it should match the harmonic prediction; at large
    amplitudes nonlinear corrections shift the frequency.

    Returns dict with: amplitudes, omega_measured, omega_harmonic.
    """
    amplitudes = np.logspace(-3, 0, n_amplitudes)
    omega_measured = []
    omega_harmonic = 2.0 * np.sqrt(a) * np.sin(np.pi / N)

    for amp in amplitudes:
        sim = _make_lattice(
            N=N, a=a, mode=1, amplitude=amp, dt=dt, n_steps=n_steps,
        )
        sim.reset()

        # Track displacement of particle 0
        positions = [sim.observe()[0]]
        for _ in range(n_steps):
            state = sim.step()
            positions.append(state[0])

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
            omega = 2 * np.pi / T_measured
        else:
            omega = float("nan")

        omega_measured.append(omega)

    return {
        "amplitudes": amplitudes,
        "omega_measured": np.array(omega_measured),
        "omega_harmonic": omega_harmonic,
    }


def run_toda_lattice_rediscovery(
    output_dir: str | Path = "output/rediscovery/toda_lattice",
    n_iterations: int = 40,
) -> dict:
    """Run the full Toda lattice rediscovery.

    1. Energy conservation verification
    2. Soliton speed measurement
    3. Harmonic limit frequency comparison
    4. PySR on energy conservation (E vs state variables)
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "toda_lattice",
        "targets": {
            "energy_conservation": "dE/dt = 0",
            "momentum_conservation": "sum(p_i) = const",
            "harmonic_limit": "omega_1 -> 2*sqrt(a)*sin(pi/N) as amp -> 0",
            "soliton_speed": "c > sqrt(a) for large amplitude",
        },
    }

    # --- Part 1: Energy conservation ---
    logger.info("Part 1: Energy conservation verification...")
    energy_data = generate_energy_conservation_data(
        n_trajectories=7, N=8, a=1.0, dt=0.001, n_steps=10000,
    )
    results["energy_conservation"] = {
        "n_trajectories": len(energy_data["relative_drift"]),
        "mean_final_drift": float(np.mean(energy_data["relative_drift"])),
        "max_final_drift": float(np.max(energy_data["relative_drift"])),
        "all_drifts": energy_data["relative_drift"].tolist(),
    }
    logger.info(
        f"  Energy: mean drift = {np.mean(energy_data['relative_drift']):.2e}, "
        f"max drift = {np.max(energy_data['relative_drift']):.2e}"
    )

    # --- Part 2: Soliton speeds ---
    logger.info("Part 2: Soliton speed measurement...")
    soliton_data = generate_soliton_data(
        N=64, a=1.0, n_amplitudes=15, dt=0.001, n_steps=5000,
    )
    valid_speeds = np.isfinite(soliton_data["speeds"])
    results["soliton_speed"] = {
        "n_amplitudes": len(soliton_data["amplitudes"]),
        "sound_speed": float(soliton_data["sound_speed"]),
        "n_valid": int(np.sum(valid_speeds)),
    }
    if np.sum(valid_speeds) > 0:
        results["soliton_speed"]["min_speed"] = float(
            np.min(soliton_data["speeds"][valid_speeds])
        )
        results["soliton_speed"]["max_speed"] = float(
            np.max(soliton_data["speeds"][valid_speeds])
        )
    logger.info(
        f"  Soliton: {np.sum(valid_speeds)} valid speeds, "
        f"sound speed = {soliton_data['sound_speed']:.4f}"
    )

    # --- Part 3: Harmonic limit ---
    logger.info("Part 3: Harmonic limit frequency comparison...")
    harmonic_data = generate_harmonic_limit_data(
        N=8, a=1.0, n_amplitudes=10, dt=0.001, n_steps=20000,
    )
    omega_theory = harmonic_data["omega_harmonic"]
    valid_omega = np.isfinite(harmonic_data["omega_measured"])
    if np.sum(valid_omega) > 0:
        # Check smallest-amplitude measurements
        small_amp_mask = (
            valid_omega & (harmonic_data["amplitudes"] < 0.01)
        )
        if np.sum(small_amp_mask) > 0:
            small_omega = harmonic_data["omega_measured"][small_amp_mask]
            rel_err = np.abs(small_omega - omega_theory) / omega_theory
            results["harmonic_limit"] = {
                "omega_theory": float(omega_theory),
                "omega_measured_small_amp": small_omega.tolist(),
                "mean_relative_error": float(np.mean(rel_err)),
            }
            logger.info(
                f"  Harmonic limit: omega_theory={omega_theory:.4f}, "
                f"mean error (small amp) = {np.mean(rel_err):.4%}"
            )
        else:
            results["harmonic_limit"] = {
                "omega_theory": float(omega_theory),
                "note": "No valid small-amplitude measurements",
            }

    # --- Part 4: PySR on energy components ---
    logger.info("Part 4: PySR on energy decay rate vs coupling...")
    # Sweep coupling 'a' and measure energy for a fixed initial condition
    a_values = np.linspace(0.5, 5.0, 25)
    energies = []
    for a_val in a_values:
        sim = _make_lattice(N=8, a=a_val, mode=1, amplitude=0.1, dt=0.001)
        sim.reset()
        energies.append(sim.total_energy)

    a_arr = a_values.reshape(-1, 1)
    e_arr = np.array(energies)

    logger.info(
        f"  Running PySR for E = f(a) with {n_iterations} iterations..."
    )
    discoveries = run_symbolic_regression(
        a_arr,
        e_arr,
        variable_names=["a_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "sqrt", "square", "sin", "cos"],
        max_complexity=12,
        populations=20,
        population_size=40,
    )

    results["energy_pysr"] = {
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
        results["energy_pysr"]["best"] = best.expression
        results["energy_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best: {best.expression} "
            f"(R2={best.evidence.fit_r_squared:.6f})"
        )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save raw data
    np.savez(
        output_path / "energy_conservation_data.npz",
        E_initial=energy_data["E_initial"],
        E_final=energy_data["E_final"],
        relative_drift=energy_data["relative_drift"],
    )
    np.savez(
        output_path / "soliton_data.npz",
        amplitudes=soliton_data["amplitudes"],
        speeds=soliton_data["speeds"],
    )
    np.savez(
        output_path / "harmonic_limit_data.npz",
        amplitudes=harmonic_data["amplitudes"],
        omega_measured=harmonic_data["omega_measured"],
    )

    return results
