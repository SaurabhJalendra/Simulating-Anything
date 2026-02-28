"""Navier-Stokes 2D rediscovery: viscous decay rate and energy spectrum.

Targets:
- Viscous decay rate: E(t) = E_0 * exp(-2*nu*k^2*t) for Taylor-Green vortex
- Energy spectrum scaling in decaying 2D turbulence
- Enstrophy decay rate
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.navier_stokes import NavierStokes2DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_decay_rate_data(
    n_samples: int = 30,
    n_steps: int = 500,
    dt: float = 0.01,
    N: int = 64,
) -> dict[str, np.ndarray]:
    """Generate energy decay data for various viscosity values.

    Measures energy decay rate for Taylor-Green vortex at different nu values.
    For each nu, the decay rate should be 2*nu*k^2.
    """
    rng = np.random.default_rng(42)
    nu_values = np.logspace(-3, -1, n_samples)

    all_nu = []
    all_decay_rate = []
    all_E_0 = []
    all_E_final = []

    L = 2 * np.pi
    k = 2 * np.pi / L  # Fundamental wavenumber

    for i, nu in enumerate(nu_values):
        config = SimulationConfig(
            domain=Domain.NAVIER_STOKES_2D,
            dt=dt,
            n_steps=n_steps,
            parameters={"nu": nu, "N": N, "L": L, "init_amplitude": 1.0},
        )
        sim = NavierStokes2DSimulation(config)
        sim.reset()

        E_0 = sim.kinetic_energy
        energies = [E_0]

        for _ in range(n_steps):
            sim.step()
            energies.append(sim.kinetic_energy)

        E_final = energies[-1]
        t_total = n_steps * dt

        # Measure decay rate from exponential fit: E(t) = E_0 * exp(-lambda * t)
        if E_final > 1e-15 and E_0 > 1e-15:
            decay_rate = -np.log(E_final / E_0) / t_total
        else:
            decay_rate = np.inf

        all_nu.append(nu)
        all_decay_rate.append(decay_rate)
        all_E_0.append(E_0)
        all_E_final.append(E_final)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  nu={nu:.4f}: decay_rate={decay_rate:.4f}, "
                f"theory={2 * nu * k ** 2:.4f}"
            )

    return {
        "nu": np.array(all_nu),
        "decay_rate": np.array(all_decay_rate),
        "E_0": np.array(all_E_0),
        "E_final": np.array(all_E_final),
        "k_fundamental": k,
        "L": L,
    }


def generate_energy_timeseries(
    nu: float = 0.01,
    n_steps: int = 1000,
    dt: float = 0.01,
    N: int = 64,
) -> dict[str, np.ndarray]:
    """Generate a full energy time series for one viscosity value."""
    config = SimulationConfig(
        domain=Domain.NAVIER_STOKES_2D,
        dt=dt,
        n_steps=n_steps,
        parameters={"nu": nu, "N": N, "L": 2 * np.pi},
    )
    sim = NavierStokes2DSimulation(config)
    sim.reset()

    times = [0.0]
    energies = [sim.kinetic_energy]
    enstrophies = [sim.enstrophy]

    for step in range(n_steps):
        sim.step()
        times.append((step + 1) * dt)
        energies.append(sim.kinetic_energy)
        enstrophies.append(sim.enstrophy)

    return {
        "time": np.array(times),
        "energy": np.array(energies),
        "enstrophy": np.array(enstrophies),
        "nu": nu,
        "dt": dt,
        "N": N,
    }


def run_navier_stokes_rediscovery(
    output_dir: str | Path = "output/rediscovery/navier_stokes",
    n_iterations: int = 40,
) -> dict:
    """Run the full Navier-Stokes 2D rediscovery.

    1. Generate decay rate data for various viscosities
    2. Run PySR to find decay_rate = f(nu)
    3. Generate energy time series for analysis
    4. Compare with analytical solution

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "navier_stokes_2d",
        "targets": {
            "decay_rate": "lambda = 2 * nu * |k|^2 = 4*nu for Taylor-Green mode (1,1)",
            "energy_decay": "E(t) = E_0 * exp(-4*nu*t) for Taylor-Green mode (1,1)",
        },
    }

    # --- Part 1: Decay rate vs viscosity ---
    logger.info("Part 1: Generating decay rate data...")
    data = generate_decay_rate_data(n_samples=30, n_steps=500, dt=0.01, N=64)

    # Theoretical decay rate for Taylor-Green vortex with mode (1,1):
    # |k|^2 = kx^2 + ky^2 = 1 + 1 = 2, so decay_rate = 2 * nu * |k|^2 = 4 * nu
    k = data["k_fundamental"]
    k_sq_total = 2 * k**2  # kx^2 + ky^2 for mode (1,1)
    theory_rate = 2 * data["nu"] * k_sq_total

    # Filter valid data (finite decay rates)
    valid = np.isfinite(data["decay_rate"]) & (data["decay_rate"] > 0)
    if np.sum(valid) < 5:
        logger.warning("Too few valid data points for PySR")
    else:
        rel_err = np.abs(data["decay_rate"][valid] - theory_rate[valid]) / theory_rate[valid]
        results["decay_rate_data"] = {
            "n_samples": int(np.sum(valid)),
            "mean_relative_error": float(np.mean(rel_err)),
            "correlation": float(np.corrcoef(data["decay_rate"][valid], theory_rate[valid])[0, 1]),
        }
        logger.info(f"  Mean relative error vs theory: {np.mean(rel_err):.2%}")
        logger.info(f"  Correlation: {results['decay_rate_data']['correlation']:.6f}")

    # PySR: find decay_rate = f(nu)
    try:
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        X = data["nu"][valid].reshape(-1, 1)
        y = data["decay_rate"][valid]

        logger.info(f"  Running PySR: decay_rate = f(nu)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["nu"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square"],
            max_complexity=10,
            populations=20,
            population_size=40,
        )
        results["decay_rate_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["decay_rate_pysr"]["best"] = best.expression
            results["decay_rate_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})")
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["decay_rate_pysr"] = {"error": str(e)}

    # --- Part 2: Energy time series ---
    logger.info("Part 2: Generating energy time series...")
    ts_data = generate_energy_timeseries(nu=0.01, n_steps=500, dt=0.01, N=64)

    # Compare with analytical solution: E(t) = E_0 * exp(-2*nu*|k|^2*t), |k|^2=2
    E_theory = ts_data["energy"][0] * np.exp(-2 * 0.01 * k_sq_total * ts_data["time"])
    rel_err_ts = np.abs(ts_data["energy"] - E_theory) / np.maximum(E_theory, 1e-15)

    results["energy_timeseries"] = {
        "nu": 0.01,
        "n_steps": 500,
        "E_initial": float(ts_data["energy"][0]),
        "E_final": float(ts_data["energy"][-1]),
        "mean_relative_error_vs_theory": float(np.mean(rel_err_ts[1:])),
    }
    logger.info(f"  E_0 = {ts_data['energy'][0]:.6f}")
    logger.info(f"  E_final = {ts_data['energy'][-1]:.6f}")
    logger.info(f"  Mean error vs analytical: {np.mean(rel_err_ts[1:]):.2%}")

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "decay_rate_data.npz",
        **{k_: v for k_, v in data.items() if isinstance(v, np.ndarray)},
    )
    np.savez(
        output_path / "energy_timeseries.npz",
        **{k_: v for k_, v in ts_data.items() if isinstance(v, np.ndarray)},
    )

    return results
