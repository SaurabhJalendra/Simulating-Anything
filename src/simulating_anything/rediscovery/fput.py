"""FPUT lattice rediscovery.

Targets:
- FPUT recurrence: energy returns to initial mode (the FPUT paradox)
- Energy conservation: symplectic Verlet preserves Hamiltonian
- Mode energy sharing: only a few modes participate (no thermalization)
- Alpha vs beta model comparison: different nonlinearity types
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.fput import FPUTSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_fput(
    N: int = 32,
    k: float = 1.0,
    alpha: float = 0.25,
    beta: float = 0.0,
    mode: int = 1,
    amplitude: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 10000,
) -> FPUTSimulation:
    """Create an FPUTSimulation with the given parameters."""
    config = SimulationConfig(
        domain=Domain.FPUT,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "N": float(N),
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "mode": float(mode),
            "amplitude": amplitude,
        },
    )
    return FPUTSimulation(config)


def generate_energy_conservation_data(
    n_trajectories: int = 5,
    N: int = 32,
    alpha: float = 0.25,
    dt: float = 0.01,
    n_steps: int = 50000,
) -> dict[str, np.ndarray]:
    """Run trajectories and verify energy conservation with Verlet integrator.

    Each trajectory uses a different initial mode excitation.

    Returns:
        Dict with arrays: E_initial, E_final, relative_drift.
    """
    E_initial = []
    E_final = []
    rel_drift = []

    for mode_n in range(1, min(n_trajectories + 1, N)):
        sim = _make_fput(
            N=N, alpha=alpha, mode=mode_n,
            amplitude=1.0, dt=dt, n_steps=n_steps,
        )
        sim.reset()
        E0 = sim.compute_total_energy()

        for _ in range(n_steps):
            sim.step()

        Ef = sim.compute_total_energy()
        drift = abs(Ef - E0) / max(abs(E0), 1e-15)

        E_initial.append(E0)
        E_final.append(Ef)
        rel_drift.append(drift)

    return {
        "E_initial": np.array(E_initial),
        "E_final": np.array(E_final),
        "relative_drift": np.array(rel_drift),
    }


def generate_recurrence_data(
    N: int = 32,
    alpha: float = 0.25,
    amplitude: float = 1.0,
    dt: float = 0.05,
    n_steps: int = 100000,
    sample_interval: int = 100,
) -> dict[str, np.ndarray]:
    """Track mode energies over time to detect FPUT recurrence.

    The FPUT paradox: when energy is initially placed in mode 1,
    it spreads to a few low modes but then returns (nearly) to mode 1.

    Args:
        N: number of particles.
        alpha: quadratic nonlinearity.
        amplitude: initial excitation amplitude.
        dt: timestep.
        n_steps: total number of steps.
        sample_interval: save mode energies every this many steps.

    Returns:
        Dict with:
            times: array of sample times
            mode_energies: (n_samples, N) array of mode energies
            mode1_fraction: fraction of energy in mode 1 at each sample
            recurrence_detected: whether mode 1 recovers > 90% of initial
    """
    sim = _make_fput(
        N=N, alpha=alpha, mode=1, amplitude=amplitude, dt=dt, n_steps=n_steps,
    )
    sim.reset()

    times = []
    mode_energies_list = []

    # Record initial
    me = sim.compute_mode_energies()
    times.append(0.0)
    mode_energies_list.append(me.copy())

    for step_i in range(1, n_steps + 1):
        sim.step()
        if step_i % sample_interval == 0:
            me = sim.compute_mode_energies()
            times.append(step_i * dt)
            mode_energies_list.append(me.copy())

    times = np.array(times)
    mode_energies = np.array(mode_energies_list)

    # Mode 1 fraction of total energy
    total_mode_energy = np.sum(mode_energies, axis=1)
    # Avoid division by zero
    safe_total = np.maximum(total_mode_energy, 1e-15)
    mode1_fraction = mode_energies[:, 0] / safe_total

    # Detect recurrence: mode1 fraction drops below 0.5 and later recovers above 0.9
    dropped = False
    recurrence_detected = False
    for frac in mode1_fraction:
        if frac < 0.5:
            dropped = True
        if dropped and frac > 0.9:
            recurrence_detected = True
            break

    return {
        "times": times,
        "mode_energies": mode_energies,
        "mode1_fraction": mode1_fraction,
        "recurrence_detected": recurrence_detected,
    }


def generate_alpha_vs_beta_data(
    N: int = 32,
    amplitude: float = 1.0,
    dt: float = 0.05,
    n_steps: int = 50000,
    sample_interval: int = 100,
) -> dict[str, dict]:
    """Compare alpha (quadratic) and beta (cubic) FPUT models.

    Both are initialized with mode 1 excitation of the same amplitude.

    Returns:
        Dict with "alpha_model" and "beta_model" sub-dicts, each containing
        mode1_fraction timeseries and energy conservation metrics.
    """
    results = {}

    for label, alpha_val, beta_val in [
        ("alpha_model", 0.25, 0.0),
        ("beta_model", 0.0, 0.25),
    ]:
        sim = _make_fput(
            N=N, alpha=alpha_val, beta=beta_val,
            mode=1, amplitude=amplitude, dt=dt, n_steps=n_steps,
        )
        sim.reset()
        E0 = sim.compute_total_energy()

        mode1_fractions = []
        me = sim.compute_mode_energies()
        total_me = max(np.sum(me), 1e-15)
        mode1_fractions.append(float(me[0] / total_me))

        for step_i in range(1, n_steps + 1):
            sim.step()
            if step_i % sample_interval == 0:
                me = sim.compute_mode_energies()
                total_me = max(np.sum(me), 1e-15)
                mode1_fractions.append(float(me[0] / total_me))

        Ef = sim.compute_total_energy()
        drift = abs(Ef - E0) / max(abs(E0), 1e-15)

        results[label] = {
            "alpha": alpha_val,
            "beta": beta_val,
            "E_initial": float(E0),
            "E_final": float(Ef),
            "relative_drift": float(drift),
            "mode1_fraction_initial": mode1_fractions[0],
            "mode1_fraction_final": mode1_fractions[-1],
            "mode1_fraction_min": float(np.min(mode1_fractions)),
            "n_samples": len(mode1_fractions),
        }

    return results


def run_fput_rediscovery(
    output_dir: str | Path = "output/rediscovery/fput",
    n_iterations: int = 40,
) -> dict:
    """Run the full FPUT lattice rediscovery.

    1. Energy conservation verification (symplectic Verlet)
    2. FPUT recurrence detection (mode energy returns to mode 1)
    3. Alpha vs beta model comparison
    4. PySR on mode energy fraction vs parameters
    """
    from simulating_anything.analysis.symbolic_regression import (
        run_symbolic_regression,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "fput",
        "targets": {
            "energy_conservation": "dE/dt = 0 (symplectic Verlet)",
            "fput_recurrence": "Mode 1 energy returns after spreading",
            "no_thermalization": "Energy stays in low modes",
            "alpha_vs_beta": "Quadratic vs cubic nonlinearity comparison",
        },
    }

    # --- Part 1: Energy conservation ---
    logger.info("Part 1: Energy conservation verification...")
    energy_data = generate_energy_conservation_data(
        n_trajectories=5, N=32, alpha=0.25, dt=0.01, n_steps=50000,
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

    # --- Part 2: FPUT recurrence ---
    logger.info("Part 2: FPUT recurrence detection...")
    recurrence_data = generate_recurrence_data(
        N=32, alpha=0.25, amplitude=1.0, dt=0.05, n_steps=100000,
        sample_interval=100,
    )
    results["fput_recurrence"] = {
        "recurrence_detected": bool(recurrence_data["recurrence_detected"]),
        "n_time_samples": len(recurrence_data["times"]),
        "mode1_fraction_initial": float(recurrence_data["mode1_fraction"][0]),
        "mode1_fraction_min": float(np.min(recurrence_data["mode1_fraction"])),
        "mode1_fraction_final": float(recurrence_data["mode1_fraction"][-1]),
    }
    logger.info(
        f"  Recurrence detected: {recurrence_data['recurrence_detected']}, "
        f"mode1 fraction: "
        f"initial={recurrence_data['mode1_fraction'][0]:.4f}, "
        f"min={np.min(recurrence_data['mode1_fraction']):.4f}, "
        f"final={recurrence_data['mode1_fraction'][-1]:.4f}"
    )

    # --- Part 3: Alpha vs beta comparison ---
    logger.info("Part 3: Alpha vs beta model comparison...")
    ab_data = generate_alpha_vs_beta_data(
        N=32, amplitude=1.0, dt=0.05, n_steps=50000, sample_interval=100,
    )
    results["alpha_vs_beta"] = ab_data
    for label in ["alpha_model", "beta_model"]:
        d = ab_data[label]
        logger.info(
            f"  {label}: E_drift={d['relative_drift']:.2e}, "
            f"mode1_min={d['mode1_fraction_min']:.4f}"
        )

    # --- Part 4: PySR on mode energy fraction vs alpha ---
    logger.info("Part 4: PySR on minimum mode-1 fraction vs alpha...")
    alpha_values = np.linspace(0.05, 0.5, 20)
    mode1_min_fracs = []

    for a_val in alpha_values:
        sim = _make_fput(
            N=32, alpha=a_val, mode=1, amplitude=1.0,
            dt=0.05, n_steps=30000,
        )
        sim.reset()
        me = sim.compute_mode_energies()
        total_me = max(np.sum(me), 1e-15)
        min_frac = me[0] / total_me

        for step_i in range(1, 30001):
            sim.step()
            if step_i % 200 == 0:
                me = sim.compute_mode_energies()
                total_me = max(np.sum(me), 1e-15)
                frac = me[0] / total_me
                if frac < min_frac:
                    min_frac = frac

        mode1_min_fracs.append(float(min_frac))

    a_arr = alpha_values.reshape(-1, 1)
    frac_arr = np.array(mode1_min_fracs)

    logger.info(
        f"  Running PySR for mode1_min = f(alpha) with {n_iterations} iterations..."
    )
    discoveries = run_symbolic_regression(
        a_arr,
        frac_arr,
        variable_names=["alpha_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "sqrt", "square", "sin", "cos"],
        max_complexity=12,
        populations=20,
        population_size=40,
    )

    results["mode1_pysr"] = {
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
        results["mode1_pysr"]["best"] = best.expression
        results["mode1_pysr"]["best_r2"] = best.evidence.fit_r_squared
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
        output_path / "recurrence_data.npz",
        times=recurrence_data["times"],
        mode1_fraction=recurrence_data["mode1_fraction"],
    )

    return results
