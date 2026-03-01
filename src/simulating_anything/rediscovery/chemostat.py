"""Chemostat rediscovery.

Targets:
- ODE: dS/dt = D*(S_in-S) - mu_max*S/(K_s+S)*X/Y_xs,
       dX/dt = mu_max*S/(K_s+S)*X - D*X  (via SINDy)
- Washout bifurcation: D_c = mu_max * S_in / (K_s + S_in)
- Steady-state substrate: S* = K_s * D / (mu_max - D)
- Monod growth rate: mu(S) = mu_max * S / (K_s + S)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.chemostat import Chemostat
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_steady_state_data(
    n_D: int = 30,
    dt: float = 0.01,
    n_settle: int = 50000,
) -> dict[str, np.ndarray]:
    """Sweep dilution rate D and measure steady-state S and X.

    Runs the chemostat to steady state for each D value below washout,
    collecting (D, S*, X*) tuples for symbolic regression.
    """
    # Compute washout D for default parameters
    mu_max, K_s, S_in = 0.5, 2.0, 10.0
    D_c = mu_max * S_in / (K_s + S_in)

    D_values = np.linspace(0.02, D_c * 1.2, n_D)
    S_steady = []
    X_steady = []

    for i, D in enumerate(D_values):
        config = SimulationConfig(
            domain=Domain.CHEMOSTAT,
            dt=dt,
            n_steps=100,
            parameters={
                "D": D, "S_in": S_in, "mu_max": mu_max,
                "K_s": K_s, "Y_xs": 0.5, "S_0": 5.0, "X_0": 1.0,
            },
        )
        sim = Chemostat(config)
        sim.reset()

        # Run to steady state
        for _ in range(n_settle):
            sim.step()

        S, X = sim.observe()
        S_steady.append(S)
        X_steady.append(X)

        if (i + 1) % 10 == 0:
            logger.info(f"  D={D:.4f}: S*={S:.4f}, X*={X:.4f}")

    return {
        "D": D_values,
        "S_steady": np.array(S_steady),
        "X_steady": np.array(X_steady),
        "D_c_theory": D_c,
        "mu_max": mu_max,
        "K_s": K_s,
        "S_in": S_in,
        "Y_xs": 0.5,
    }


def generate_washout_data(
    n_D: int = 40,
    dt: float = 0.01,
    n_settle: int = 50000,
) -> dict[str, np.ndarray]:
    """Generate data showing washout transition.

    Sweeps D from well below to well above D_c, measuring final biomass.
    """
    mu_max, K_s, S_in, Y_xs = 0.5, 2.0, 10.0, 0.5
    D_c = mu_max * S_in / (K_s + S_in)

    D_values = np.linspace(0.01, 0.6, n_D)
    X_final = []

    for D in D_values:
        config = SimulationConfig(
            domain=Domain.CHEMOSTAT,
            dt=dt,
            n_steps=100,
            parameters={
                "D": D, "S_in": S_in, "mu_max": mu_max,
                "K_s": K_s, "Y_xs": Y_xs, "S_0": 5.0, "X_0": 1.0,
            },
        )
        sim = Chemostat(config)
        sim.reset()

        for _ in range(n_settle):
            sim.step()

        _, X = sim.observe()
        X_final.append(X)

    return {
        "D": D_values,
        "X_final": np.array(X_final),
        "D_c_theory": D_c,
    }


def generate_trajectory_data(
    D: float = 0.1,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.CHEMOSTAT,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "D": D, "S_in": 10.0, "mu_max": 0.5,
            "K_s": 2.0, "Y_xs": 0.5, "S_0": 5.0, "X_0": 1.0,
        },
    )
    sim = Chemostat(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "S": states[:, 0],
        "X": states[:, 1],
        "D": D,
    }


def run_chemostat_rediscovery(
    output_dir: str | Path = "output/rediscovery/chemostat",
    n_iterations: int = 40,
) -> dict:
    """Run chemostat rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "chemostat",
        "targets": {
            "ode": "dS/dt=D*(S_in-S)-mu*X/Y, dX/dt=mu*X-D*X, mu=mu_max*S/(K_s+S)",
            "washout": "D_c = mu_max*S_in/(K_s+S_in)",
            "steady_state": "S* = K_s*D/(mu_max-D), X* = Y*(S_in-S*)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at D=0.1...")
    data = generate_trajectory_data(D=0.1, n_steps=10000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.01,
            feature_names=["S", "X"],
            threshold=0.05,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries[:5]
            ],
        }
        if sindy_discoveries:
            best = sindy_discoveries[0]
            results["sindy_ode"]["best"] = best.expression
            results["sindy_ode"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  SINDy best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Steady-state vs D ---
    logger.info("Part 2: Steady-state substrate S* vs D...")
    ss_data = generate_steady_state_data(n_D=25, dt=0.01, n_settle=30000)

    # Compare simulated S* to theory: S* = K_s*D/(mu_max - D)
    D_vals = ss_data["D"]
    D_c = ss_data["D_c_theory"]
    below_washout = D_vals < D_c * 0.95  # avoid near-bifurcation instability

    if np.sum(below_washout) > 3:
        S_theory = ss_data["K_s"] * D_vals[below_washout] / (
            ss_data["mu_max"] - D_vals[below_washout]
        )
        S_sim = ss_data["S_steady"][below_washout]
        rel_errors = np.abs(S_sim - S_theory) / (S_theory + 1e-10)
        results["steady_state"] = {
            "mean_relative_error": float(np.mean(rel_errors)),
            "max_relative_error": float(np.max(rel_errors)),
            "n_points": int(np.sum(below_washout)),
        }
        logger.info(f"  S* mean rel error: {np.mean(rel_errors):.6f}")

    # PySR: find S* = f(D)
    try:
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        if np.sum(below_washout) > 5:
            X_reg = D_vals[below_washout].reshape(-1, 1)
            y_reg = ss_data["S_steady"][below_washout]

            logger.info("  Running PySR: S* = f(D)...")
            discoveries = run_symbolic_regression(
                X_reg, y_reg,
                variable_names=["D_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square"],
                max_complexity=10,
                populations=20,
                population_size=40,
            )
            results["steady_state_pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["steady_state_pysr"]["best"] = best.expression
                results["steady_state_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["steady_state_pysr"] = {"error": str(e)}

    # --- Part 3: Washout bifurcation ---
    logger.info("Part 3: Washout bifurcation...")
    wo_data = generate_washout_data(n_D=30, dt=0.01, n_settle=30000)

    # Detect washout: first D where X < threshold
    threshold = 0.01
    washed_out = wo_data["X_final"] < threshold
    if np.any(washed_out):
        idx = np.argmax(washed_out)
        D_c_est = wo_data["D"][max(0, idx - 1)]
        D_c_theory = wo_data["D_c_theory"]
        results["washout"] = {
            "D_c_estimate": float(D_c_est),
            "D_c_theory": float(D_c_theory),
            "relative_error": float(abs(D_c_est - D_c_theory) / D_c_theory),
        }
        logger.info(f"  D_c estimate: {D_c_est:.4f} (theory: {D_c_theory:.4f})")

    # PySR: find D_c = f(S_in) by sweeping S_in
    try:
        from simulating_anything.analysis.symbolic_regression import run_symbolic_regression

        S_in_values = np.linspace(2.0, 20.0, 15)
        D_c_measured = []
        mu_max_val, K_s_val = 0.5, 2.0

        for S_in_val in S_in_values:
            # Sweep D near D_c for this S_in
            D_sweep = np.linspace(0.01, mu_max_val * 0.98, 25)
            X_finals = []
            for D_val in D_sweep:
                config = SimulationConfig(
                    domain=Domain.CHEMOSTAT,
                    dt=0.01,
                    n_steps=100,
                    parameters={
                        "D": D_val, "S_in": S_in_val, "mu_max": mu_max_val,
                        "K_s": K_s_val, "Y_xs": 0.5, "S_0": S_in_val / 2, "X_0": 1.0,
                    },
                )
                sim = Chemostat(config)
                sim.reset()
                for _ in range(20000):
                    sim.step()
                _, X = sim.observe()
                X_finals.append(X)

            X_finals = np.array(X_finals)
            washed = X_finals < 0.01
            if np.any(washed):
                idx = np.argmax(washed)
                D_c_measured.append(D_sweep[max(0, idx - 1)])
            else:
                D_c_measured.append(np.nan)

        D_c_measured = np.array(D_c_measured)
        valid = np.isfinite(D_c_measured)
        if np.sum(valid) > 5:
            X_reg = S_in_values[valid].reshape(-1, 1)
            y_reg = D_c_measured[valid]

            logger.info("  Running PySR: D_c = f(S_in)...")
            discoveries = run_symbolic_regression(
                X_reg, y_reg,
                variable_names=["S_in"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=[],
                max_complexity=8,
                populations=20,
                population_size=40,
            )
            results["washout_pysr"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["washout_pysr"]["best"] = best.expression
                results["washout_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["washout_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
