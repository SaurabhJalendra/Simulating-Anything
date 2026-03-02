"""Zombie SIR model rediscovery (Munz et al. 2009).

Targets:
- Zombie reproduction number R0_z = beta/alpha
- Alpha threshold for human survival
- Outbreak dynamics: peak zombie timing, final populations
- SINDy recovery of 4-compartment ODEs
- Resurrection effect on outbreak severity
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.zombie_sir import ZombieSIRSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Placeholder domain -- will be replaced with Domain.ZOMBIE_SIR
_DOMAIN = Domain.ZOMBIE_SIR


def generate_ode_data(
    n_steps: int = 10000,
    dt: float = 0.1,
    **param_overrides: float,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Args:
        n_steps: Number of integration steps.
        dt: Timestep.
        **param_overrides: Override default parameters.

    Returns:
        Dict with time, states, S, I, Z, R arrays.
    """
    params: dict[str, float] = {
        "beta": 0.0095, "alpha": 0.005, "zeta": 0.0001,
        "delta": 0.0001, "rho": 0.5, "Pi": 0.0, "N": 500.0,
        "S_0": 499.0, "I_0": 0.0, "Z_0": 1.0, "R_0_init": 0.0,
    }
    params.update(param_overrides)

    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )
    sim = ZombieSIRSimulation(config)
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
        "I": states[:, 1],
        "Z": states[:, 2],
        "R": states[:, 3],
    }


def generate_alpha_sweep(
    n_alpha: int = 30,
    n_steps: int = 20000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep kill rate alpha to find human survival threshold.

    Args:
        n_alpha: Number of alpha values to test.
        n_steps: Steps per simulation.
        dt: Timestep.

    Returns:
        Dict with alpha_values, final_S, final_Z, survived, R0_z arrays.
    """
    alpha_values = np.linspace(0.001, 0.05, n_alpha)
    final_S = []
    final_Z = []
    survived = []
    R0_z = []

    beta = 0.0095

    for i, alpha_val in enumerate(alpha_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": beta, "alpha": alpha_val, "zeta": 0.0001,
                "delta": 0.0001, "rho": 0.5, "Pi": 0.0, "N": 500.0,
                "S_0": 499.0, "I_0": 0.0, "Z_0": 1.0, "R_0_init": 0.0,
            },
        )
        sim = ZombieSIRSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        state = sim.observe()
        final_S.append(state[0])
        final_Z.append(state[2])
        survived.append(state[0] > 5.0)  # >1% of N=500
        R0_z.append(beta / alpha_val)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  alpha={alpha_val:.4f}: S={state[0]:.1f}, "
                f"Z={state[2]:.1f}, R0_z={beta / alpha_val:.2f}"
            )

    return {
        "alpha_values": alpha_values,
        "final_S": np.array(final_S),
        "final_Z": np.array(final_Z),
        "survived": np.array(survived),
        "R0_z": np.array(R0_z),
    }


def generate_resurrection_sweep(
    n_zeta: int = 20,
    n_steps: int = 20000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep resurrection rate zeta to measure its effect on outbreak severity.

    Args:
        n_zeta: Number of zeta values.
        n_steps: Steps per simulation.
        dt: Timestep.

    Returns:
        Dict with zeta_values, final_S, final_Z, peak_Z arrays.
    """
    zeta_values = np.linspace(0.0, 0.01, n_zeta)
    final_S = []
    final_Z = []
    peak_Z = []

    for i, zeta_val in enumerate(zeta_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "beta": 0.0095, "alpha": 0.005, "zeta": zeta_val,
                "delta": 0.0001, "rho": 0.5, "Pi": 0.0, "N": 500.0,
                "S_0": 499.0, "I_0": 0.0, "Z_0": 1.0, "R_0_init": 0.0,
            },
        )
        sim = ZombieSIRSimulation(config)
        sim.reset()

        max_Z = 0.0
        for _ in range(n_steps):
            sim.step()
            z_now = sim.observe()[2]
            if z_now > max_Z:
                max_Z = z_now

        state = sim.observe()
        final_S.append(state[0])
        final_Z.append(state[2])
        peak_Z.append(max_Z)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  zeta={zeta_val:.5f}: S={state[0]:.1f}, "
                f"Z={state[2]:.1f}, peak_Z={max_Z:.1f}"
            )

    return {
        "zeta_values": zeta_values,
        "final_S": np.array(final_S),
        "final_Z": np.array(final_Z),
        "peak_Z": np.array(peak_Z),
    }


def run_zombie_sir_rediscovery(
    output_dir: str | Path = "output/rediscovery/zombie_sir",
    n_iterations: int = 40,
) -> dict:
    """Run zombie SIR model rediscovery pipeline.

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iterations.

    Returns:
        Results dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "zombie_sir",
        "targets": {
            "R0_zombie": "beta / alpha",
            "ode_S": "dS/dt = -beta*S*Z + Pi",
            "ode_I": "dI/dt = beta*S*Z - rho*I - delta*I",
            "ode_Z": "dZ/dt = rho*I + zeta*R - alpha*S*Z",
            "ode_R": "dR/dt = delta*I + alpha*S*Z - zeta*R",
            "survival_threshold": "alpha > beta for human survival",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery...")
    data = generate_ode_data(n_steps=5000, dt=0.1)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.1,
            feature_names=["S", "I", "Z", "R"],
            threshold=0.001,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries[:8]
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

    # --- Part 2: Alpha sweep for survival threshold ---
    logger.info("Part 2: Alpha sweep for survival threshold...")
    alpha_data = generate_alpha_sweep(n_alpha=30, n_steps=20000, dt=0.1)

    # Find survival threshold: first alpha where humans survive
    survived = alpha_data["survived"]
    if np.any(survived):
        idx = np.argmax(survived)
        alpha_c_est = alpha_data["alpha_values"][idx]
        r0_z_at_threshold = 0.0095 / alpha_c_est
        results["survival_threshold"] = {
            "alpha_c_estimate": float(alpha_c_est),
            "R0_z_at_threshold": float(r0_z_at_threshold),
            "n_survived": int(np.sum(survived)),
            "n_total": len(survived),
        }
        logger.info(
            f"  Survival threshold: alpha ~ {alpha_c_est:.4f} "
            f"(R0_z = {r0_z_at_threshold:.2f})"
        )
    else:
        results["survival_threshold"] = {
            "note": "No survival detected -- zombies always win"
        }

    # --- Part 3: Resurrection effect ---
    logger.info("Part 3: Resurrection rate sweep...")
    resurrection_data = generate_resurrection_sweep(
        n_zeta=20, n_steps=20000, dt=0.1,
    )

    # Higher zeta should make outbreak worse (more zombies)
    finite_mask = np.isfinite(resurrection_data["final_Z"])
    if np.sum(finite_mask) > 3:
        zeta_fin = resurrection_data["zeta_values"][finite_mask]
        z_fin = resurrection_data["final_Z"][finite_mask]
        if np.std(zeta_fin) > 0 and np.std(z_fin) > 0:
            corr = float(np.corrcoef(zeta_fin, z_fin)[0, 1])
        else:
            corr = 0.0
        results["resurrection_effect"] = {
            "correlation_zeta_vs_final_Z": corr,
            "min_final_Z": float(np.min(z_fin)),
            "max_final_Z": float(np.max(z_fin)),
            "note": (
                "Positive correlation: higher resurrection -> more zombies"
            ),
        }
        logger.info(f"  Correlation(zeta, final_Z) = {corr:.4f}")

    # --- Part 4: PySR for R0_zombie ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use alpha sweep: final_S = f(alpha) or survival = f(beta/alpha)
        X = alpha_data["alpha_values"].reshape(-1, 1)
        y = alpha_data["final_S"]

        valid = np.isfinite(y) & (y > 0)
        if np.sum(valid) > 5:
            logger.info("  Running PySR: final_S = f(alpha)...")
            discoveries = run_symbolic_regression(
                X[valid], y[valid],
                variable_names=["a_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square", "exp"],
                max_complexity=12,
                populations=15,
                population_size=30,
            )
            results["pysr_survival"] = {
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
                results["pysr_survival"]["best"] = best.expression
                results["pysr_survival"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_survival"] = {"error": str(e)}

    # --- Part 5: Equilibrium analysis ---
    logger.info("Part 5: Equilibrium analysis...")
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=0.1,
        n_steps=1000,
        parameters={
            "beta": 0.0095, "alpha": 0.005, "zeta": 0.0001,
            "delta": 0.0001, "rho": 0.5, "Pi": 0.0, "N": 500.0,
            "S_0": 499.0, "I_0": 0.0, "Z_0": 1.0, "R_0_init": 0.0,
        },
    )
    sim = ZombieSIRSimulation(config)
    sim.reset()

    r0_z = sim.compute_basic_reproduction()
    equilibria = sim.compute_equilibria()
    results["equilibria"] = {
        "R0_zombie": float(r0_z),
        "R0_formula": "beta / alpha",
        "disease_free": equilibria["disease_free"].tolist(),
        "doomsday": equilibria["doomsday"].tolist(),
    }
    logger.info(f"  R0_zombie = beta/alpha = {r0_z:.4f}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
