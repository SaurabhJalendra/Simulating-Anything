"""Cart-pole rediscovery.

Targets:
- Small-angle frequency: omega = sqrt(g*(M+m) / (M*L))
- Energy conservation: E = const for frictionless, unforced system
- SINDy recovery of linearized equations of motion
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.cart_pole import CartPole
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_linearized_frequency_data(
    n_samples: int = 200,
    n_steps: int = 20000,
    dt: float = 0.0005,
) -> dict[str, np.ndarray]:
    """Sweep m, M, L and measure small-oscillation frequency.

    For small displacement near the hanging equilibrium (theta=pi), the pendulum
    oscillates with frequency omega = sqrt(g*(M+m)/(M*L)).
    We measure the period from zero crossings of (theta - pi).
    """
    rng = np.random.default_rng(42)

    all_M = []
    all_m = []
    all_L = []
    all_omega_measured = []
    all_omega_theory = []

    for i in range(n_samples):
        M_val = rng.uniform(0.5, 5.0)
        m_val = rng.uniform(0.05, 1.0)
        L_val = rng.uniform(0.3, 2.0)
        g_val = 9.81

        config = SimulationConfig(
            domain=Domain.CART_POLE,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "M": M_val, "m": m_val, "L": L_val, "g": g_val,
                "mu_c": 0.0, "mu_p": 0.0, "F": 0.0,
                "x_0": 0.0, "x_dot_0": 0.0,
                "theta_0": np.pi + 0.02, "theta_dot_0": 0.0,
            },
        )
        sim = CartPole(config)
        sim.reset()

        # Collect theta over time
        theta_history = [sim.observe()[2]]
        for _ in range(n_steps):
            state = sim.step()
            theta_history.append(state[2])

        theta_arr = np.array(theta_history)
        # Track deviation from the stable hanging equilibrium at theta=pi
        delta = theta_arr - np.pi

        # Find period from positive-going zero crossings of the deviation
        crossings = []
        for j in range(1, len(delta)):
            if delta[j - 1] < 0 and delta[j] >= 0:
                frac = -delta[j - 1] / (delta[j] - delta[j - 1])
                crossings.append((j - 1 + frac) * dt)

        if len(crossings) >= 3:
            periods = np.diff(crossings)
            T_measured = float(np.median(periods))
            omega_measured = 2 * np.pi / T_measured
            omega_theory = np.sqrt(g_val * (M_val + m_val) / (M_val * L_val))

            all_M.append(M_val)
            all_m.append(m_val)
            all_L.append(L_val)
            all_omega_measured.append(omega_measured)
            all_omega_theory.append(omega_theory)

        if (i + 1) % 50 == 0:
            logger.info(f"  Frequency measurement {i + 1}/{n_samples}")

    return {
        "M": np.array(all_M),
        "m": np.array(all_m),
        "L": np.array(all_L),
        "omega_measured": np.array(all_omega_measured),
        "omega_theory": np.array(all_omega_theory),
    }


def generate_energy_conservation_data(
    n_trajectories: int = 50,
    n_steps: int = 10000,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Track total energy over time for frictionless, unforced cart-pole.

    Energy should be conserved to within numerical tolerance.
    """
    rng = np.random.default_rng(42)

    all_final_drift = []
    all_max_drift = []
    all_mean_energy = []

    for i in range(n_trajectories):
        M_val = rng.uniform(0.5, 5.0)
        m_val = rng.uniform(0.05, 1.0)
        L_val = rng.uniform(0.3, 2.0)
        theta_0 = rng.uniform(-0.5, 0.5)
        theta_dot_0 = rng.uniform(-1.0, 1.0)
        x_dot_0 = rng.uniform(-0.5, 0.5)

        config = SimulationConfig(
            domain=Domain.CART_POLE,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "M": M_val, "m": m_val, "L": L_val, "g": 9.81,
                "mu_c": 0.0, "mu_p": 0.0, "F": 0.0,
                "x_0": 0.0, "x_dot_0": x_dot_0,
                "theta_0": theta_0, "theta_dot_0": theta_dot_0,
            },
        )
        sim = CartPole(config)
        sim.reset()

        E0 = sim.total_energy
        energies = [E0]
        for _ in range(n_steps):
            sim.step()
            energies.append(sim.total_energy)

        energies = np.array(energies)
        drift = np.abs(energies - E0) / max(abs(E0), 1e-10)

        all_final_drift.append(drift[-1])
        all_max_drift.append(np.max(drift))
        all_mean_energy.append(np.mean(energies))

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


def run_cart_pole_rediscovery(
    output_dir: str | Path = "output/rediscovery/cart_pole",
    n_iterations: int = 40,
) -> dict:
    """Run the full cart-pole rediscovery.

    1. Energy conservation verification
    2. Small-angle frequency: omega = sqrt(g*(M+m)/(M*L)) via PySR
    3. Report results
    """
    from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "cart_pole",
        "targets": {
            "energy_conservation": "E(t) = E(0) for all t (no friction/force)",
            "frequency": "omega = sqrt(g*(M+m) / (M*L))",
        },
    }

    # --- Part 1: Energy conservation ---
    logger.info("Part 1: Energy conservation verification...")
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

    # --- Part 2: Small-angle frequency ---
    logger.info("Part 2: Small-angle frequency measurement...")
    freq_data = generate_linearized_frequency_data(
        n_samples=200, n_steps=20000, dt=0.0005
    )

    rel_error = (
        np.abs(freq_data["omega_measured"] - freq_data["omega_theory"])
        / freq_data["omega_theory"]
    )
    results["frequency_accuracy"] = {
        "n_samples": len(freq_data["M"]),
        "mean_relative_error": float(np.mean(rel_error)),
        "max_relative_error": float(np.max(rel_error)),
    }
    logger.info(f"  Frequency accuracy: mean error = {np.mean(rel_error):.4%}")

    # PySR: omega = f(M, m, L)
    logger.info(
        f"  Running PySR for omega = f(M, m, L) with {n_iterations} iterations..."
    )
    X = np.column_stack([freq_data["M"], freq_data["m"], freq_data["L"]])
    y = freq_data["omega_measured"]

    discoveries = run_symbolic_regression(
        X,
        y,
        variable_names=["M_", "m_", "L_"],
        n_iterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "square"],
        max_complexity=15,
        populations=20,
        population_size=40,
    )

    results["frequency_pysr"] = {
        "n_discoveries": len(discoveries),
        "discoveries": [
            {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
            for d in discoveries[:5]
        ],
    }
    if discoveries:
        best = discoveries[0]
        results["frequency_pysr"]["best"] = best.expression
        results["frequency_pysr"]["best_r2"] = best.evidence.fit_r_squared
        logger.info(
            f"  Best: {best.expression} (R2={best.evidence.fit_r_squared:.6f})"
        )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "frequency_data.npz",
        M=freq_data["M"],
        m=freq_data["m"],
        L=freq_data["L"],
        omega_measured=freq_data["omega_measured"],
        omega_theory=freq_data["omega_theory"],
    )

    return results
