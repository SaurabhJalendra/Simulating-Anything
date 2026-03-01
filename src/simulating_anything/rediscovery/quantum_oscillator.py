"""Quantum harmonic oscillator rediscovery.

Targets:
- Energy spectrum: E_n = hbar*omega*(n + 1/2)
- Ground state energy: E_0 = 0.5*hbar*omega
- Coherent state oscillation: <x>(t) = x_0*cos(omega*t)
- PySR recovery of E_0 = 0.5*omega (with hbar=1)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.quantum_oscillator import QuantumHarmonicOscillator
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_energy_spectrum_data(
    n_states: int = 10,
    omega: float = 1.0,
    hbar: float = 1.0,
    m: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute energy levels for n=0..n_states-1 by preparing eigenstates.

    Verifies E_n = hbar*omega*(n + 0.5) for each level.
    """
    all_n = []
    all_E_measured = []
    all_E_theory = []

    config = SimulationConfig(
        domain=Domain.QUANTUM_OSCILLATOR,
        dt=0.01,
        n_steps=10,
        parameters={
            "m": m, "omega": omega, "hbar": hbar,
            "N": 256.0, "x_max": 15.0, "x_0": 0.0,
        },
    )
    sim = QuantumHarmonicOscillator(config)
    sim.reset()

    for n in range(n_states):
        E_measured = sim.measure_energy_from_eigenstate(n)
        E_theory = hbar * omega * (n + 0.5)

        all_n.append(n)
        all_E_measured.append(E_measured)
        all_E_theory.append(E_theory)

        logger.info(
            f"  n={n}: E_measured={E_measured:.6f}, E_theory={E_theory:.6f}, "
            f"error={abs(E_measured - E_theory):.2e}"
        )

    return {
        "n": np.array(all_n),
        "E_measured": np.array(all_E_measured),
        "E_theory": np.array(all_E_theory),
        "omega": omega,
        "hbar": hbar,
        "m": m,
    }


def generate_coherent_state_data(
    n_points: int = 20,
    omega: float = 1.0,
    x_0: float = 2.0,
    total_periods: float = 3.0,
) -> dict[str, np.ndarray]:
    """Track coherent state oscillation and verify <x>(t) = x_0*cos(omega*t).

    A coherent state displaced by x_0 oscillates classically with frequency omega.
    """
    T = 2.0 * np.pi / omega
    total_time = total_periods * T
    dt = 0.005
    n_steps = int(total_time / dt)

    config = SimulationConfig(
        domain=Domain.QUANTUM_OSCILLATOR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "m": 1.0, "omega": omega, "hbar": 1.0,
            "N": 256.0, "x_max": 15.0, "x_0": x_0, "p_0": 0.0,
        },
    )
    sim = QuantumHarmonicOscillator(config)
    sim.reset()

    # Sample at n_points evenly spaced times
    sample_interval = max(1, n_steps // n_points)
    times = []
    x_expect = []
    x_theory = []
    norms = []

    times.append(0.0)
    x_expect.append(sim.position_expectation)
    x_theory.append(x_0 * np.cos(omega * 0.0))
    norms.append(sim.norm)

    for step in range(1, n_steps + 1):
        sim.step()
        if step % sample_interval == 0:
            t = step * dt
            times.append(t)
            x_expect.append(sim.position_expectation)
            x_theory.append(x_0 * np.cos(omega * t))
            norms.append(sim.norm)

    return {
        "times": np.array(times),
        "x_expectation": np.array(x_expect),
        "x_theory": np.array(x_theory),
        "norms": np.array(norms),
        "omega": omega,
        "x_0": x_0,
    }


def generate_omega_sweep_data(
    n_points: int = 15,
    hbar: float = 1.0,
    m: float = 1.0,
) -> dict[str, np.ndarray]:
    """Sweep omega and measure ground state energy E_0.

    Verifies E_0 = 0.5 * hbar * omega across different frequencies.
    """
    omega_values = np.linspace(0.5, 5.0, n_points)
    all_omega = []
    all_E0_measured = []
    all_E0_theory = []

    for i, omega in enumerate(omega_values):
        config = SimulationConfig(
            domain=Domain.QUANTUM_OSCILLATOR,
            dt=0.01,
            n_steps=10,
            parameters={
                "m": m, "omega": omega, "hbar": hbar,
                "N": 256.0, "x_max": 15.0, "x_0": 0.0,
            },
        )
        sim = QuantumHarmonicOscillator(config)
        sim.reset()

        # Prepare ground state and measure energy
        E0 = sim.measure_energy_from_eigenstate(0)
        E0_theory = 0.5 * hbar * omega

        all_omega.append(omega)
        all_E0_measured.append(E0)
        all_E0_theory.append(E0_theory)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  omega={omega:.2f}: E0={E0:.6f}, theory={E0_theory:.6f}"
            )

    return {
        "omega": np.array(all_omega),
        "E0_measured": np.array(all_E0_measured),
        "E0_theory": np.array(all_E0_theory),
        "hbar": hbar,
        "m": m,
    }


def run_quantum_oscillator_rediscovery(
    output_dir: str | Path = "output/rediscovery/quantum_oscillator",
    n_iterations: int = 40,
) -> dict:
    """Run the full quantum harmonic oscillator rediscovery.

    1. Energy spectrum: verify E_n = hbar*omega*(n+0.5)
    2. Coherent state: verify <x>(t) = x_0*cos(omega*t)
    3. Omega sweep + PySR: recover E_0 = 0.5*omega (with hbar=1)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "quantum_oscillator",
        "targets": {
            "energy_spectrum": "E_n = hbar*omega*(n + 0.5)",
            "ground_state": "E_0 = 0.5*hbar*omega",
            "coherent_oscillation": "<x>(t) = x_0*cos(omega*t)",
        },
    }

    # --- Part 1: Energy spectrum verification ---
    logger.info("Part 1: Energy spectrum E_n = hbar*omega*(n + 0.5)...")
    spectrum_data = generate_energy_spectrum_data(n_states=10)

    rel_errors = np.abs(
        spectrum_data["E_measured"] - spectrum_data["E_theory"]
    ) / spectrum_data["E_theory"]
    results["energy_spectrum"] = {
        "n_states": len(spectrum_data["n"]),
        "E_measured": spectrum_data["E_measured"].tolist(),
        "E_theory": spectrum_data["E_theory"].tolist(),
        "mean_relative_error": float(np.mean(rel_errors)),
        "max_relative_error": float(np.max(rel_errors)),
    }
    logger.info(f"  Mean relative error: {np.mean(rel_errors):.4%}")

    # --- Part 2: Coherent state oscillation ---
    logger.info("Part 2: Coherent state oscillation <x>(t) = x_0*cos(omega*t)...")
    coherent_data = generate_coherent_state_data(n_points=50)

    x_err = np.abs(coherent_data["x_expectation"] - coherent_data["x_theory"])
    norm_drift = np.abs(coherent_data["norms"] - 1.0)
    results["coherent_state"] = {
        "n_samples": len(coherent_data["times"]),
        "mean_position_error": float(np.mean(x_err)),
        "max_position_error": float(np.max(x_err)),
        "mean_norm_drift": float(np.mean(norm_drift)),
        "max_norm_drift": float(np.max(norm_drift)),
    }
    logger.info(f"  Mean |<x> - x_0*cos(wt)| = {np.mean(x_err):.6f}")
    logger.info(f"  Max norm drift = {np.max(norm_drift):.2e}")

    # --- Part 3: Omega sweep + PySR ---
    logger.info("Part 3: Omega sweep, PySR for E_0 = f(omega)...")
    sweep_data = generate_omega_sweep_data(n_points=15)

    sweep_rel_err = np.abs(
        sweep_data["E0_measured"] - sweep_data["E0_theory"]
    ) / sweep_data["E0_theory"]
    results["omega_sweep"] = {
        "n_points": len(sweep_data["omega"]),
        "mean_relative_error": float(np.mean(sweep_rel_err)),
        "correlation": float(
            np.corrcoef(sweep_data["E0_measured"], sweep_data["E0_theory"])[0, 1]
        ),
    }
    logger.info(f"  Correlation with theory: {results['omega_sweep']['correlation']:.6f}")

    # PySR: E_0 = f(omega) -- should recover E_0 = 0.5 * omega
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = sweep_data["omega"].reshape(-1, 1)
        y = sweep_data["E0_measured"]

        logger.info(f"  Running PySR for E_0 = f(omega) with {n_iterations} iterations...")
        discoveries = run_symbolic_regression(
            X,
            y,
            variable_names=["w"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt"],
            max_complexity=10,
            populations=15,
            population_size=30,
        )

        results["ground_state_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["ground_state_pysr"]["best"] = best.expression
            results["ground_state_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["ground_state_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "spectrum_data.npz",
        n=spectrum_data["n"],
        E_measured=spectrum_data["E_measured"],
        E_theory=spectrum_data["E_theory"],
    )
    np.savez(
        output_path / "coherent_data.npz",
        times=coherent_data["times"],
        x_expectation=coherent_data["x_expectation"],
        x_theory=coherent_data["x_theory"],
        norms=coherent_data["norms"],
    )
    np.savez(
        output_path / "omega_sweep_data.npz",
        omega=sweep_data["omega"],
        E0_measured=sweep_data["E0_measured"],
        E0_theory=sweep_data["E0_theory"],
    )

    return results
