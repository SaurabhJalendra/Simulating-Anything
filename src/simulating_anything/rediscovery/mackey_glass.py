"""Mackey-Glass delay differential equation rediscovery.

Targets:
- Equilibrium x* = (beta/gamma - 1)^(1/n) verification
- Chaos onset as function of delay tau
- Lyapunov exponent vs tau sweep
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.mackey_glass import MackeyGlassSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_equilibrium_data(
    n_beta: int = 15,
    n_steps: int = 50000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep beta values and verify convergence to x* = (beta/gamma - 1)^(1/n).

    Uses small tau (tau=2) to ensure monotone convergence to equilibrium.
    """
    beta_values = np.linspace(0.15, 0.5, n_beta)
    gamma = 0.1
    n_exp = 10.0
    measured_eq = []
    theory_eq = []

    for beta_val in beta_values:
        config = SimulationConfig(
            domain=Domain.MACKEY_GLASS,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta_val,
                "gamma": gamma,
                "tau": 2.0,  # Small tau for convergence
                "n": n_exp,
                "x_0": 0.9,
            },
        )
        sim = MackeyGlassSimulation(config)
        sim.reset()

        # Run to steady state
        for _ in range(n_steps):
            sim.step()

        # Average over last 1000 steps to get equilibrium
        x_vals = []
        for _ in range(1000):
            state = sim.step()
            x_vals.append(state[0])

        measured_eq.append(float(np.mean(x_vals)))

        # Theoretical equilibrium
        ratio = beta_val / gamma
        x_star = (ratio - 1.0) ** (1.0 / n_exp) if ratio > 1.0 else 0.0
        theory_eq.append(x_star)

    return {
        "beta": beta_values,
        "measured_equilibrium": np.array(measured_eq),
        "theory_equilibrium": np.array(theory_eq),
        "gamma": gamma,
        "n": n_exp,
    }


def generate_tau_sweep_data(
    n_tau: int = 25,
    tau_min: float = 2.0,
    tau_max: float = 30.0,
    n_steps: int = 50000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep tau to map the transition from convergence to chaos.

    Returns amplitude range and Lyapunov exponent for each tau value.
    """
    tau_values = np.linspace(tau_min, tau_max, n_tau)
    amplitudes = []
    lyapunovs = []

    for i, tau_val in enumerate(tau_values):
        config = SimulationConfig(
            domain=Domain.MACKEY_GLASS,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": 0.2,
                "gamma": 0.1,
                "tau": tau_val,
                "n": 10.0,
                "x_0": 0.9,
            },
        )
        sim = MackeyGlassSimulation(config)
        sim.reset()

        # Transient
        for _ in range(20000):
            sim.step()

        # Collect post-transient values
        x_vals = []
        for _ in range(n_steps):
            state = sim.step()
            x_vals.append(state[0])

        x_arr = np.array(x_vals)
        amplitudes.append(float(np.max(x_arr) - np.min(x_arr)))

        # Lyapunov exponent
        lam = sim.estimate_lyapunov(n_steps=min(n_steps, 50000))
        lyapunovs.append(lam)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  tau={tau_val:.1f}: amplitude={amplitudes[-1]:.4f}, "
                f"Lyapunov={lyapunovs[-1]:.4f}"
            )

    return {
        "tau": tau_values,
        "amplitude": np.array(amplitudes),
        "lyapunov": np.array(lyapunovs),
    }


def run_mackey_glass_rediscovery(
    output_dir: str | Path = "output/rediscovery/mackey_glass",
    n_iterations: int = 40,
) -> dict:
    """Run the full Mackey-Glass DDE rediscovery.

    1. Verify equilibrium x* = (beta/gamma - 1)^(1/n)
    2. Sweep tau to map chaos onset
    3. Identify critical tau where Lyapunov becomes positive

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "mackey_glass",
        "targets": {
            "equilibrium": "x* = (beta/gamma - 1)^(1/n)",
            "chaos_onset": "tau_c where Lyapunov > 0",
            "tau_regimes": "convergence (tau<4), oscillation (4<tau<13), chaos (tau>17)",
        },
    }

    # --- Part 1: Equilibrium verification ---
    logger.info("Part 1: Equilibrium verification (beta sweep at tau=2)...")
    eq_data = generate_equilibrium_data(n_beta=15, n_steps=50000, dt=0.1)

    # Compute agreement
    measured = eq_data["measured_equilibrium"]
    theory = eq_data["theory_equilibrium"]
    rel_errors = np.abs(measured - theory) / np.maximum(theory, 1e-12)
    mean_rel_error = float(np.mean(rel_errors))

    results["equilibrium"] = {
        "n_beta_values": len(eq_data["beta"]),
        "mean_relative_error": mean_rel_error,
        "max_relative_error": float(np.max(rel_errors)),
        "theory_formula": "x* = (beta/gamma - 1)^(1/n)",
        "gamma": eq_data["gamma"],
        "n": eq_data["n"],
    }
    logger.info(f"  Equilibrium mean relative error: {mean_rel_error:.6f}")

    # --- Part 2: Tau sweep ---
    logger.info("Part 2: Tau sweep to map chaos onset...")
    tau_data = generate_tau_sweep_data(n_tau=25, tau_min=2.0, tau_max=30.0)

    # Find chaos onset (first tau where Lyapunov > 0)
    lyap = tau_data["lyapunov"]
    tau_arr = tau_data["tau"]
    positive_mask = lyap > 0
    if np.any(positive_mask):
        tau_c_approx = float(tau_arr[np.argmax(positive_mask)])
        results["chaos_onset"] = {
            "tau_c_estimate": tau_c_approx,
            "tau_c_note": "First tau with positive Lyapunov exponent",
        }
        logger.info(f"  Chaos onset: tau ~ {tau_c_approx:.1f}")
    else:
        results["chaos_onset"] = {"tau_c_estimate": None, "note": "No chaos detected"}

    # Classify regimes
    n_convergent = int(np.sum(tau_data["amplitude"] < 0.01))
    n_oscillatory = int(np.sum(
        (tau_data["amplitude"] >= 0.01) & (lyap <= 0)
    ))
    n_chaotic = int(np.sum(lyap > 0))
    results["tau_sweep"] = {
        "n_tau_values": len(tau_arr),
        "tau_range": [float(tau_arr[0]), float(tau_arr[-1])],
        "n_convergent": n_convergent,
        "n_oscillatory": n_oscillatory,
        "n_chaotic": n_chaotic,
        "max_lyapunov": float(np.max(lyap)),
        "max_amplitude": float(np.max(tau_data["amplitude"])),
    }

    # --- Part 3: Equilibrium value at default parameters ---
    config_default = SimulationConfig(
        domain=Domain.MACKEY_GLASS,
        dt=0.1,
        n_steps=1000,
        parameters={"beta": 0.2, "gamma": 0.1, "tau": 17.0, "n": 10.0, "x_0": 0.9},
    )
    sim_default = MackeyGlassSimulation(config_default)
    sim_default.reset()
    x_star = sim_default.equilibrium
    results["default_equilibrium"] = {
        "x_star": x_star,
        "beta": 0.2,
        "gamma": 0.1,
        "n": 10.0,
        "note": "(beta/gamma - 1)^(1/n) = (2 - 1)^(1/10) = 1.0",
    }
    logger.info(f"  Default equilibrium x* = {x_star:.6f}")

    # --- Part 4: PySR on Lyapunov vs tau (optional) ---
    try:
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        chaotic_mask = lyap > 0
        if np.sum(chaotic_mask) > 5:
            X = tau_arr[chaotic_mask].reshape(-1, 1)
            y = lyap[chaotic_mask]

            logger.info("  Running PySR: Lyapunov = f(tau) for chaotic region...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["tau"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["log", "sqrt"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["lyapunov_pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["lyapunov_pysr"]["best"] = best.expression
                results["lyapunov_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["lyapunov_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "equilibrium_data.npz",
        beta=eq_data["beta"],
        measured=eq_data["measured_equilibrium"],
        theory=eq_data["theory_equilibrium"],
    )
    np.savez(
        output_path / "tau_sweep.npz",
        tau=tau_data["tau"],
        amplitude=tau_data["amplitude"],
        lyapunov=tau_data["lyapunov"],
    )

    return results
