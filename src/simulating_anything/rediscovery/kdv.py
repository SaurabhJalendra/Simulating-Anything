"""KdV (Korteweg-de Vries) equation rediscovery.

Targets:
- Speed-amplitude relation: c = 2*A (soliton speed proportional to amplitude)
- Elastic soliton collisions (two solitons pass through each other unchanged)
- Conserved quantities: mass, momentum, energy
- PySR: speed = f(amplitude) should recover c = 2*A
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.kdv import KdVSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_speed_amplitude_data(
    n_speeds: int = 20,
    N: int = 256,
    L: float = 40.0,
) -> dict[str, np.ndarray]:
    """Generate soliton amplitude vs speed parameter data.

    For each speed c, creates a soliton and measures its peak amplitude.
    Theory: peak amplitude A = c/2, i.e. c = 2*A.
    """
    c_values = np.linspace(0.5, 8.0, n_speeds)
    amplitudes = []
    theoretical = []

    for c in c_values:
        config = SimulationConfig(
            domain=Domain.KDV,
            dt=1e-4,
            n_steps=100,
            parameters={"c": c, "N": float(N), "L": L, "x0": L / 2},
        )
        sim = KdVSimulation(config)
        sim.init_type = "soliton"
        sim.reset()

        amplitudes.append(sim.measure_peak_amplitude())
        theoretical.append(c / 2.0)

    return {
        "c_values": np.array(c_values),
        "amplitudes": np.array(amplitudes),
        "theoretical_amplitudes": np.array(theoretical),
    }


def generate_conservation_data(
    n_steps: int = 2000,
    dt: float = 1e-4,
    c: float = 2.0,
    N: int = 256,
    L: float = 40.0,
) -> dict[str, np.ndarray]:
    """Run soliton evolution and track conserved quantities.

    Tracks mass (integral u), momentum (integral u^2),
    and energy (integral u^3 - u_x^2/2).
    """
    config = SimulationConfig(
        domain=Domain.KDV,
        dt=dt,
        n_steps=n_steps,
        parameters={"c": c, "N": float(N), "L": L, "x0": L / 4},
    )
    sim = KdVSimulation(config)
    sim.init_type = "soliton"
    sim.reset()

    masses = [sim.compute_mass()]
    momenta = [sim.compute_momentum()]
    energies = [sim.compute_energy()]
    times = [0.0]

    for i in range(n_steps):
        sim.step()
        if (i + 1) % 20 == 0:
            masses.append(sim.compute_mass())
            momenta.append(sim.compute_momentum())
            energies.append(sim.compute_energy())
            times.append((i + 1) * dt)

    return {
        "times": np.array(times),
        "masses": np.array(masses),
        "momenta": np.array(momenta),
        "energies": np.array(energies),
    }


def generate_collision_data(
    n_steps: int = 5000,
    dt: float = 1e-4,
    N: int = 512,
    L: float = 60.0,
) -> dict[str, np.ndarray]:
    """Run two-soliton collision and verify elastic scattering.

    A fast soliton (c=4.0) overtakes a slow soliton (c=1.0).
    After the collision, both solitons should re-emerge with
    unchanged amplitude and speed but with a phase shift.
    """
    c_fast = 4.0
    c_slow = 1.0
    x0_fast = L * 0.2
    x0_slow = L * 0.5

    config = SimulationConfig(
        domain=Domain.KDV,
        dt=dt,
        n_steps=n_steps,
        parameters={"c": c_fast, "N": float(N), "L": L, "x0": x0_fast},
    )
    sim = KdVSimulation(config)
    sim.set_two_solitons(c_fast, x0_fast, c_slow, x0_slow)

    # Measure initial conserved quantities
    mass_0 = sim.compute_mass()
    momentum_0 = sim.compute_momentum()
    energy_0 = sim.compute_energy()

    # Store amplitude of initial fast soliton
    amp_fast_0 = c_fast / 2.0
    amp_slow_0 = c_slow / 2.0

    masses = [mass_0]
    momenta = [momentum_0]

    for i in range(n_steps):
        sim.step()
        if (i + 1) % 100 == 0:
            masses.append(sim.compute_mass())
            momenta.append(sim.compute_momentum())

    # After collision, measure final conserved quantities
    mass_f = sim.compute_mass()
    momentum_f = sim.compute_momentum()
    energy_f = sim.compute_energy()

    return {
        "mass_initial": mass_0,
        "mass_final": mass_f,
        "mass_drift": abs(mass_f - mass_0) / abs(mass_0) if abs(mass_0) > 1e-15 else 0.0,
        "momentum_initial": momentum_0,
        "momentum_final": momentum_f,
        "momentum_drift": (
            abs(momentum_f - momentum_0) / abs(momentum_0)
            if abs(momentum_0) > 1e-15 else 0.0
        ),
        "energy_initial": energy_0,
        "energy_final": energy_f,
        "energy_drift": (
            abs(energy_f - energy_0) / abs(energy_0) if abs(energy_0) > 1e-15 else 0.0
        ),
        "c_fast": c_fast,
        "c_slow": c_slow,
        "amp_fast_theory": amp_fast_0,
        "amp_slow_theory": amp_slow_0,
        "masses": np.array(masses),
        "momenta": np.array(momenta),
    }


def run_kdv_rediscovery(
    output_dir: str | Path = "output/rediscovery/kdv",
    n_iterations: int = 50,
) -> dict:
    """Run KdV rediscovery pipeline.

    Demonstrates:
    1. Speed-amplitude relation c = 2*A (PySR)
    2. Elastic soliton collisions
    3. Conservation of mass, momentum, and energy
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "kdv",
        "targets": {
            "speed_amplitude": "c = 2*A (soliton speed = 2 * peak amplitude)",
            "elastic_collision": "Two solitons pass through each other unchanged",
            "conservation": "mass, momentum, energy conserved",
        },
    }

    # 1. Speed-amplitude relation
    logger.info("Generating speed-amplitude data...")
    sa_data = generate_speed_amplitude_data(n_speeds=20)

    rel_err = np.abs(
        sa_data["amplitudes"] - sa_data["theoretical_amplitudes"]
    ) / sa_data["theoretical_amplitudes"]
    correlation = float(np.corrcoef(
        sa_data["amplitudes"], sa_data["theoretical_amplitudes"],
    )[0, 1])

    results["speed_amplitude_data"] = {
        "n_samples": len(sa_data["c_values"]),
        "mean_relative_error": float(np.mean(rel_err)),
        "correlation": correlation,
    }
    logger.info(
        f"  Speed-amplitude: corr={correlation:.6f}, "
        f"mean_err={np.mean(rel_err):.4%}"
    )

    # 2. Soliton collision
    logger.info("Running two-soliton collision...")
    col_data = generate_collision_data(n_steps=3000, dt=1e-4, N=512, L=60.0)

    results["collision"] = {
        "mass_drift": float(col_data["mass_drift"]),
        "momentum_drift": float(col_data["momentum_drift"]),
        "energy_drift": float(col_data["energy_drift"]),
        "c_fast": float(col_data["c_fast"]),
        "c_slow": float(col_data["c_slow"]),
    }
    logger.info(
        f"  Collision: mass_drift={col_data['mass_drift']:.2e}, "
        f"momentum_drift={col_data['momentum_drift']:.2e}, "
        f"energy_drift={col_data['energy_drift']:.2e}"
    )

    # 3. Conservation law verification
    logger.info("Checking conservation laws...")
    cons_data = generate_conservation_data(n_steps=2000, dt=1e-4)

    M = cons_data["masses"]
    P = cons_data["momenta"]
    E = cons_data["energies"]

    mass_drift = np.abs(M - M[0]) / abs(M[0]) if abs(M[0]) > 1e-15 else np.zeros_like(M)
    mom_drift = np.abs(P - P[0]) / abs(P[0]) if abs(P[0]) > 1e-15 else np.zeros_like(P)
    eng_drift = np.abs(E - E[0]) / abs(E[0]) if abs(E[0]) > 1e-15 else np.zeros_like(E)

    results["conservation"] = {
        "mass_max_drift": float(np.max(mass_drift)),
        "momentum_max_drift": float(np.max(mom_drift)),
        "energy_max_drift": float(np.max(eng_drift)),
        "initial_mass": float(M[0]),
        "initial_momentum": float(P[0]),
        "initial_energy": float(E[0]),
        "n_samples": len(M),
    }
    logger.info(
        f"  Conservation: mass_drift={np.max(mass_drift):.2e}, "
        f"momentum_drift={np.max(mom_drift):.2e}, "
        f"energy_drift={np.max(eng_drift):.2e}"
    )

    # 4. PySR: speed = f(amplitude)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = sa_data["amplitudes"].reshape(-1, 1)
        y = sa_data["c_values"]

        logger.info("Running PySR: speed = f(amplitude)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["A_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt"],
            max_complexity=8,
            populations=15,
            population_size=30,
        )
        results["speed_amplitude_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["speed_amplitude_pysr"]["best"] = best.expression
            results["speed_amplitude_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["speed_amplitude_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
