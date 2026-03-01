"""Sine-Gordon equation rediscovery.

Targets:
- Lorentz contraction: kink width ~ sqrt(1 - v^2/c^2)
- Energy conservation: symplectic integrator preserves energy
- Topological charge conservation: Q = integer for kinks
- PySR: width = f(velocity) should recover relativistic contraction
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sine_gordon import SineGordonSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_lorentz_contraction_data(
    n_velocities: int = 30,
    c: float = 1.0,
    N: int = 512,
    L: float = 80.0,
) -> dict[str, np.ndarray]:
    """Generate kink width vs velocity data for Lorentz contraction.

    Measures kink width at various velocities and compares to
    the theoretical prediction: width(v) = width(0) * sqrt(1 - v^2/c^2).
    """
    velocities = np.linspace(0.0, 0.9 * c, n_velocities)

    config = SimulationConfig(
        domain=Domain.SINE_GORDON,
        dt=0.01,
        n_steps=100,
        parameters={"c": c, "N": float(N), "L": L},
    )
    sim = SineGordonSimulation(config)
    sim.reset()

    data = sim.measure_lorentz_contraction(velocities)

    return {
        "velocities": data["velocities"],
        "measured_widths": data["measured_widths"],
        "theoretical_widths": data["theoretical_widths"],
        "rest_width": data["rest_width"],
        "c": c,
    }


def generate_energy_conservation_data(
    n_steps: int = 5000,
    dt: float = 0.005,
    c: float = 1.0,
    N: int = 256,
    L: float = 40.0,
) -> dict[str, np.ndarray]:
    """Run kink evolution and track energy conservation."""
    config = SimulationConfig(
        domain=Domain.SINE_GORDON,
        dt=dt,
        n_steps=n_steps,
        parameters={"c": c, "N": float(N), "L": L},
    )
    sim = SineGordonSimulation(config)
    sim.init_type = "kink"
    sim.reset()

    energies = [sim.compute_energy()]
    charges = [sim.compute_topological_charge()]

    for i in range(n_steps):
        sim.step()
        if (i + 1) % 50 == 0:
            energies.append(sim.compute_energy())
            charges.append(sim.compute_topological_charge())

    return {
        "energies": np.array(energies),
        "charges": np.array(charges),
        "n_steps": n_steps,
        "dt": dt,
    }


def generate_velocity_sweep_data(
    n_velocities: int = 25,
    c: float = 1.0,
    N: int = 512,
    L: float = 80.0,
) -> dict[str, np.ndarray]:
    """Generate kink energy vs velocity data.

    Theory: E(v) = 8*c / sqrt(1 - v^2/c^2).
    """
    velocities = np.linspace(0.0, 0.9 * c, n_velocities)

    config = SimulationConfig(
        domain=Domain.SINE_GORDON,
        dt=0.01,
        n_steps=100,
        parameters={"c": c, "N": float(N), "L": L},
    )
    sim = SineGordonSimulation(config)
    sim.reset()

    data = sim.kink_velocity_sweep(velocities)
    theoretical_energies = np.array([
        SineGordonSimulation.analytical_kink_energy(c=c, v=v) for v in velocities
    ])

    return {
        "velocities": data["velocities"],
        "measured_energies": data["energies"],
        "theoretical_energies": theoretical_energies,
        "widths": data["widths"],
        "c": c,
    }


def run_sine_gordon_rediscovery(
    output_dir: str | Path = "output/rediscovery/sine_gordon",
    n_iterations: int = 40,
) -> dict:
    """Run Sine-Gordon rediscovery pipeline.

    Demonstrates:
    1. Lorentz contraction of kink width
    2. Energy conservation
    3. Topological charge conservation
    4. PySR: width = f(v) recovers relativistic factor
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sine_gordon",
        "targets": {
            "lorentz_contraction": "width(v) = width(0) * sqrt(1 - v^2/c^2)",
            "energy_conservation": "E = const (symplectic integrator)",
            "topological_charge": "Q = integer (kink = 1, antikink = -1)",
        },
    }

    # 1. Lorentz contraction data
    logger.info("Generating Lorentz contraction data...")
    lc_data = generate_lorentz_contraction_data(n_velocities=30)

    valid = lc_data["theoretical_widths"] > 1e-6
    if np.sum(valid) > 3:
        rel_err = np.abs(
            lc_data["measured_widths"][valid] - lc_data["theoretical_widths"][valid]
        ) / lc_data["theoretical_widths"][valid]
        correlation = float(np.corrcoef(
            lc_data["measured_widths"][valid],
            lc_data["theoretical_widths"][valid],
        )[0, 1])
        results["lorentz_contraction"] = {
            "n_samples": int(np.sum(valid)),
            "mean_relative_error": float(np.mean(rel_err)),
            "correlation": correlation,
            "rest_width": float(lc_data["rest_width"]),
        }
        logger.info(
            f"  Lorentz contraction: corr={correlation:.6f}, "
            f"mean_err={np.mean(rel_err):.4%}"
        )

    # 2. Energy conservation
    logger.info("Checking energy conservation...")
    ec_data = generate_energy_conservation_data(n_steps=5000, dt=0.005)
    E = ec_data["energies"]
    E0 = E[0]
    drift = np.abs(E - E0) / E0
    results["energy_conservation"] = {
        "initial_energy": float(E0),
        "max_drift": float(np.max(drift)),
        "mean_drift": float(np.mean(drift)),
        "final_drift": float(drift[-1]),
        "n_samples": len(E),
    }
    logger.info(f"  Energy drift: max={np.max(drift):.2e}, mean={np.mean(drift):.2e}")

    # 3. Topological charge conservation
    Q = ec_data["charges"]
    q_drift = np.abs(Q - Q[0])
    results["topological_charge"] = {
        "initial_charge": float(Q[0]),
        "max_drift": float(np.max(q_drift)),
        "final_charge": float(Q[-1]),
    }
    logger.info(f"  Topological charge: Q(0)={Q[0]:.4f}, max_drift={np.max(q_drift):.4e}")

    # 4. PySR: width = f(v, c)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Build dataset: width vs v for PySR
        v_arr = lc_data["velocities"][valid]
        w_arr = lc_data["measured_widths"][valid]
        # Normalize widths by rest width
        w_norm = w_arr / lc_data["rest_width"]

        # Also use v^2/c^2 as a feature for easier discovery
        v_sq_over_c_sq = (v_arr / lc_data["c"]) ** 2
        X = v_sq_over_c_sq.reshape(-1, 1)
        y = w_norm

        logger.info("Running PySR: normalized_width = f(v2_c2)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["v2_c2"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square"],
            max_complexity=10,
            populations=15,
            population_size=30,
        )
        results["lorentz_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["lorentz_pysr"]["best"] = best.expression
            results["lorentz_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["lorentz_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
