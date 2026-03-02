"""1D viscous Burgers equation rediscovery.

Targets:
- Shock formation time vs initial amplitude: t_s ~ 1 / (A * k)
- Cole-Hopf analytical solution comparison for sin(x) IC
- Viscosity sweep: nu -> 0 gives sharper gradients
- Energy dissipation rate: dE/dt = -nu * integral(u_x^2) dx
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.burgers_1d import Burgers1DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_sim(
    nu: float = 0.01,
    N: int = 256,
    L: float = 2 * np.pi,
    amplitude: float = 1.0,
    dt: float = 0.001,
    n_steps: int = 1000,
) -> Burgers1DSimulation:
    """Create a Burgers1DSimulation with the given parameters."""
    config = SimulationConfig(
        domain=Domain.BURGERS_1D,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "nu": nu,
            "N": float(N),
            "L": L,
            "amplitude": amplitude,
        },
    )
    return Burgers1DSimulation(config)


def generate_shock_formation_data(
    n_amp: int = 20,
    N: int = 512,
    dt: float = 0.0005,
    nu: float = 0.001,
) -> dict[str, np.ndarray]:
    """Measure shock formation time vs initial amplitude.

    For inviscid Burgers with u(x,0) = A*sin(x) on [0, 2*pi],
    the shock forms at t_s = 1/(A*k) = 1/A (since k=1).

    With small viscosity, the maximum gradient peaks near t_s
    before viscous smoothing takes over. We detect the time of
    maximum gradient as a proxy for the shock formation time.

    Returns dict with arrays: amplitudes, t_shock_measured, t_shock_theory.
    """
    L = 2 * np.pi
    amplitudes = np.linspace(0.5, 5.0, n_amp)
    t_shock_measured = []
    t_shock_theory = []

    for i, amp in enumerate(amplitudes):
        # Inviscid shock time
        k0 = 2 * np.pi / L
        t_theory = 1.0 / (amp * k0)

        # Run long enough to capture the shock formation
        t_max = 3.0 * t_theory
        n_steps = max(int(t_max / dt), 500)

        sim = _make_sim(nu=nu, N=N, L=L, amplitude=amp, dt=dt, n_steps=n_steps)
        sim.reset()

        max_grad = 0.0
        t_max_grad = 0.0

        for step in range(n_steps):
            sim.step()
            grad = sim.max_gradient
            if grad > max_grad:
                max_grad = grad
                t_max_grad = (step + 1) * dt

        t_shock_measured.append(t_max_grad)
        t_shock_theory.append(t_theory)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  A={amp:.2f}: t_shock_measured={t_max_grad:.4f}, "
                f"theory={t_theory:.4f}"
            )

    return {
        "amplitudes": amplitudes,
        "t_shock_measured": np.array(t_shock_measured),
        "t_shock_theory": np.array(t_shock_theory),
    }


def generate_cole_hopf_comparison(
    nu: float = 0.05,
    N: int = 256,
    dt: float = 0.001,
    n_snapshots: int = 10,
    t_final: float = 2.0,
) -> dict[str, object]:
    """Compare simulation with the Cole-Hopf exact solution at multiple times.

    Returns dict with arrays: times, l2_errors, linf_errors.
    """
    sim = _make_sim(nu=nu, N=N, dt=dt, n_steps=int(t_final / dt))
    sim.reset()

    snapshot_interval = max(1, int(t_final / dt) // n_snapshots)
    times = []
    l2_errors = []
    linf_errors = []

    step = 0
    for _ in range(int(t_final / dt)):
        sim.step()
        step += 1

        if step % snapshot_interval == 0:
            t = step * dt
            try:
                u_exact = sim.cole_hopf_exact(t)
                u_sim = sim.observe()
                l2 = float(np.sqrt(np.sum((u_sim - u_exact) ** 2) * sim.dx))
                linf = float(np.max(np.abs(u_sim - u_exact)))
                times.append(t)
                l2_errors.append(l2)
                linf_errors.append(linf)
            except Exception as e:
                logger.warning(f"Cole-Hopf failed at t={t}: {e}")

    return {
        "times": np.array(times),
        "l2_errors": np.array(l2_errors),
        "linf_errors": np.array(linf_errors),
        "nu": nu,
        "N": N,
    }


def generate_viscosity_sweep_data(
    n_nu: int = 20,
    N: int = 512,
    dt: float = 0.0002,
    t_eval: float = 0.8,
) -> dict[str, np.ndarray]:
    """Sweep viscosity and measure max gradient at a fixed time.

    As nu -> 0, the gradient at the shock location grows, showing
    the transition from smooth decay to near-inviscid shock behavior.

    Returns dict with arrays: nu_values, max_gradients, energies.
    """
    nu_values = np.logspace(-3, -0.5, n_nu)
    max_gradients = []
    energies = []

    for i, nu_val in enumerate(nu_values):
        n_steps = int(t_eval / dt)
        sim = _make_sim(nu=nu_val, N=N, dt=dt, n_steps=n_steps)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        max_gradients.append(sim.max_gradient)
        energies.append(sim.total_energy)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  nu={nu_val:.4f}: max_grad={sim.max_gradient:.2f}, "
                f"energy={sim.total_energy:.6f}"
            )

    return {
        "nu_values": nu_values,
        "max_gradients": np.array(max_gradients),
        "energies": np.array(energies),
        "t_eval": t_eval,
    }


def generate_energy_dissipation_data(
    nu: float = 0.05,
    N: int = 256,
    dt: float = 0.001,
    n_steps: int = 3000,
    sample_interval: int = 10,
) -> dict[str, np.ndarray]:
    """Track energy and its dissipation rate over time.

    Verifies dE/dt = -nu * integral(u_x^2) dx.

    Returns dict with arrays: times, energies, dissipation_rates,
        numerical_dEdt.
    """
    sim = _make_sim(nu=nu, N=N, dt=dt, n_steps=n_steps)
    sim.reset()

    times = []
    energies = []
    dissipation_rates = []

    for step in range(n_steps):
        sim.step()
        if (step + 1) % sample_interval == 0:
            times.append((step + 1) * dt)
            energies.append(sim.total_energy)
            dissipation_rates.append(sim.energy_dissipation_rate)

    times = np.array(times)
    energies = np.array(energies)
    dissipation_rates = np.array(dissipation_rates)

    # Numerical dE/dt from finite differences of the energy time series
    numerical_dEdt = np.gradient(energies, times)

    return {
        "times": times,
        "energies": energies,
        "dissipation_rates": dissipation_rates,
        "numerical_dEdt": numerical_dEdt,
        "nu": nu,
    }


def run_burgers_1d_rediscovery(
    output_dir: str | Path = "output/rediscovery/burgers_1d",
    n_iterations: int = 50,
) -> dict:
    """Run the full 1D Burgers equation rediscovery.

    1. Shock formation time vs amplitude: t_s ~ 1/(A*k)
    2. Cole-Hopf exact solution comparison
    3. Viscosity sweep: nu -> 0 sharpens gradients
    4. Energy dissipation verification: dE/dt = -nu * integral(u_x^2)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "burgers_1d",
        "targets": {
            "shock_time": "t_s = 1 / (A * k) for inviscid limit",
            "cole_hopf": "exact solution via Cole-Hopf transform",
            "viscosity_scaling": "sharper gradients as nu -> 0",
            "energy_dissipation": "dE/dt = -nu * integral(u_x^2) dx",
        },
    }

    # --- Part 1: Shock formation time ---
    logger.info("Part 1: Shock formation time vs amplitude...")
    shock_data = generate_shock_formation_data(
        n_amp=20, N=512, dt=0.0005, nu=0.001,
    )

    valid = np.isfinite(shock_data["t_shock_measured"])
    if np.sum(valid) > 5:
        corr = float(np.corrcoef(
            shock_data["t_shock_measured"][valid],
            shock_data["t_shock_theory"][valid],
        )[0, 1])
        rel_err = np.abs(
            shock_data["t_shock_measured"][valid]
            - shock_data["t_shock_theory"][valid]
        ) / shock_data["t_shock_theory"][valid]
        results["shock_time_data"] = {
            "n_samples": int(np.sum(valid)),
            "correlation": corr,
            "mean_relative_error": float(np.mean(rel_err)),
        }
        logger.info(f"  Correlation: {corr:.4f}")
        logger.info(f"  Mean relative error: {np.mean(rel_err):.4%}")

    # PySR: t_shock = f(amplitude)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X = shock_data["amplitudes"][valid].reshape(-1, 1)
        y = shock_data["t_shock_measured"][valid]

        logger.info("Running PySR: t_shock = f(A)...")
        discoveries = run_symbolic_regression(
            X, y,
            variable_names=["A_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "inv"],
            max_complexity=8,
            populations=15,
            population_size=30,
        )
        results["shock_time_pysr"] = {
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
            results["shock_time_pysr"]["best"] = best.expression
            results["shock_time_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["shock_time_pysr"] = {"error": str(e)}

    # --- Part 2: Cole-Hopf comparison ---
    logger.info("Part 2: Cole-Hopf exact solution comparison...")
    try:
        cole_hopf = generate_cole_hopf_comparison(
            nu=0.05, N=256, dt=0.001, n_snapshots=10, t_final=2.0,
        )
        if len(cole_hopf["l2_errors"]) > 0:
            results["cole_hopf_data"] = {
                "n_snapshots": len(cole_hopf["l2_errors"]),
                "mean_l2_error": float(np.mean(cole_hopf["l2_errors"])),
                "max_l2_error": float(np.max(cole_hopf["l2_errors"])),
                "mean_linf_error": float(np.mean(cole_hopf["linf_errors"])),
                "nu": cole_hopf["nu"],
            }
            logger.info(
                f"  Mean L2 error: {np.mean(cole_hopf['l2_errors']):.6f}"
            )
    except Exception as e:
        logger.warning(f"Cole-Hopf comparison failed: {e}")
        results["cole_hopf_data"] = {"error": str(e)}

    # --- Part 3: Viscosity sweep ---
    logger.info("Part 3: Viscosity sweep (nu -> 0 sharpens gradients)...")
    visc_data = generate_viscosity_sweep_data(
        n_nu=20, N=512, dt=0.0002, t_eval=0.8,
    )

    # Check that lower viscosity -> higher gradients (monotonic trend)
    max_grads = visc_data["max_gradients"]
    nu_vals = visc_data["nu_values"]
    if len(max_grads) > 5:
        # Correlation between log(nu) and log(max_grad) should be negative
        valid_g = (max_grads > 0) & np.isfinite(max_grads)
        if np.sum(valid_g) > 5:
            log_corr = float(np.corrcoef(
                np.log(nu_vals[valid_g]),
                np.log(max_grads[valid_g]),
            )[0, 1])
            results["viscosity_sweep_data"] = {
                "n_samples": int(np.sum(valid_g)),
                "log_nu_vs_log_grad_correlation": log_corr,
                "min_nu": float(nu_vals[0]),
                "max_nu": float(nu_vals[-1]),
                "gradient_range": [
                    float(np.min(max_grads[valid_g])),
                    float(np.max(max_grads[valid_g])),
                ],
            }
            logger.info(f"  log(nu)-log(grad) correlation: {log_corr:.4f}")

    # --- Part 4: Energy dissipation ---
    logger.info("Part 4: Energy dissipation verification...")
    energy_data = generate_energy_dissipation_data(
        nu=0.05, N=256, dt=0.001, n_steps=3000, sample_interval=10,
    )

    # Compare numerical dE/dt with -nu * enstrophy
    predicted = -energy_data["dissipation_rates"]
    measured = energy_data["numerical_dEdt"]

    # Skip initial and final points (finite-diff boundary effects)
    interior = slice(2, -2)
    if len(predicted[interior]) > 5:
        corr = float(np.corrcoef(
            predicted[interior], measured[interior]
        )[0, 1])
        rel_err = np.abs(predicted[interior] - measured[interior]) / (
            np.abs(predicted[interior]) + 1e-15
        )
        valid_diss = np.isfinite(rel_err) & (np.abs(predicted[interior]) > 1e-10)
        results["energy_dissipation_data"] = {
            "correlation": corr,
            "mean_relative_error": float(np.mean(rel_err[valid_diss]))
            if np.sum(valid_diss) > 0 else float("nan"),
            "energy_initial": float(energy_data["energies"][0]),
            "energy_final": float(energy_data["energies"][-1]),
        }
        logger.info(f"  dE/dt correlation: {corr:.4f}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
