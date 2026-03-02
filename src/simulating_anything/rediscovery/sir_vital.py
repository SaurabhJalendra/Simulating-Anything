"""SIR with vital dynamics rediscovery.

Targets:
- R0 = beta / (gamma + mu): basic reproduction number
- Endemic equilibrium: S* = 1/R0, I* = mu*(R0-1)/beta
- Transcritical bifurcation at R0 = 1 (beta sweep)
- Damped oscillation period near endemic equilibrium
- ODE recovery via SINDy
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sir_vital import SIRVitalSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    beta: float = 0.4,
    gamma: float = 0.1,
    mu: float = 0.01,
    S_0: float = 0.99,
    I_0: float = 0.01,
    R_0_init: float = 0.0,
    dt: float = 0.1,
    n_steps: int = 5000,
) -> SimulationConfig:
    """Create a SimulationConfig for the SIR vital dynamics model."""
    return SimulationConfig(
        domain=Domain.SIR_VITAL,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": beta,
            "gamma": gamma,
            "mu": mu,
            "S_0": S_0,
            "I_0": I_0,
            "R_0_init": R_0_init,
        },
    )


def generate_r0_sweep_data(
    n_samples: int = 200,
    n_steps: int = 50000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep beta and gamma to study R0 = beta/(gamma+mu).

    For each parameter set, runs to steady state and records final I
    (endemic level or zero for disease-free).
    """
    rng = np.random.default_rng(42)

    all_beta = []
    all_gamma = []
    all_mu = []
    all_R0 = []
    all_I_final = []
    all_S_final = []

    mu = 0.01  # fixed birth/death rate

    for i in range(n_samples):
        beta = rng.uniform(0.05, 0.8)
        gamma = rng.uniform(0.02, 0.3)

        config = _make_config(beta=beta, gamma=gamma, mu=mu, dt=dt, n_steps=n_steps)
        sim = SIRVitalSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        state = sim.observe()

        all_beta.append(beta)
        all_gamma.append(gamma)
        all_mu.append(mu)
        all_R0.append(beta / (gamma + mu))
        all_I_final.append(state[1])
        all_S_final.append(state[0])

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} SIR-vital trajectories")

    return {
        "beta": np.array(all_beta),
        "gamma": np.array(all_gamma),
        "mu": np.array(all_mu),
        "R0": np.array(all_R0),
        "I_final": np.array(all_I_final),
        "S_final": np.array(all_S_final),
    }


def generate_beta_bifurcation_data(
    n_beta: int = 40,
    n_steps: int = 100000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep beta across the R0=1 bifurcation point.

    The transcritical bifurcation occurs at beta_c = gamma + mu.
    Below this, I -> 0 (disease-free). Above, I -> I* > 0 (endemic).
    """
    gamma, mu = 0.1, 0.01
    beta_c = gamma + mu  # = 0.11

    beta_values = np.linspace(0.02, 0.5, n_beta)
    I_final = []
    S_final = []

    for beta in beta_values:
        config = _make_config(beta=beta, gamma=gamma, mu=mu, dt=dt, n_steps=n_steps)
        sim = SIRVitalSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        state = sim.observe()
        I_final.append(state[1])
        S_final.append(state[0])

    return {
        "beta": beta_values,
        "I_final": np.array(I_final),
        "S_final": np.array(S_final),
        "beta_c_theory": beta_c,
        "gamma": gamma,
        "mu": mu,
    }


def generate_endemic_equilibrium_data(
    n_samples: int = 30,
    n_steps: int = 100000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Verify endemic equilibrium I* = mu*(R0-1)/beta numerically.

    Only samples parameters where R0 > 1 so endemic equilibrium exists.
    """
    rng = np.random.default_rng(123)

    all_beta = []
    all_gamma = []
    all_mu = []
    all_R0 = []
    all_I_sim = []
    all_I_theory = []
    all_S_sim = []
    all_S_theory = []

    for i in range(n_samples):
        # Ensure R0 > 1 by picking beta > gamma + mu
        gamma = rng.uniform(0.05, 0.2)
        mu = rng.uniform(0.005, 0.03)
        beta = rng.uniform(gamma + mu + 0.05, 0.8)

        r0 = beta / (gamma + mu)
        I_star = mu * (r0 - 1.0) / beta
        S_star = 1.0 / r0

        config = _make_config(beta=beta, gamma=gamma, mu=mu, dt=dt, n_steps=n_steps)
        sim = SIRVitalSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        state = sim.observe()

        all_beta.append(beta)
        all_gamma.append(gamma)
        all_mu.append(mu)
        all_R0.append(r0)
        all_I_sim.append(state[1])
        all_I_theory.append(I_star)
        all_S_sim.append(state[0])
        all_S_theory.append(S_star)

        if (i + 1) % 10 == 0:
            logger.info(f"  Endemic eq: {i + 1}/{n_samples} done")

    return {
        "beta": np.array(all_beta),
        "gamma": np.array(all_gamma),
        "mu": np.array(all_mu),
        "R0": np.array(all_R0),
        "I_sim": np.array(all_I_sim),
        "I_theory": np.array(all_I_theory),
        "S_sim": np.array(all_S_sim),
        "S_theory": np.array(all_S_theory),
    }


def generate_trajectory_data(
    beta: float = 0.4,
    gamma: float = 0.1,
    mu: float = 0.01,
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate a single trajectory for SINDy ODE recovery."""
    config = _make_config(beta=beta, gamma=gamma, mu=mu, dt=dt, n_steps=n_steps)
    sim = SIRVitalSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "beta": beta,
        "gamma": gamma,
        "mu": mu,
    }


def measure_damped_oscillation_period(
    beta: float = 0.4,
    gamma: float = 0.1,
    mu: float = 0.01,
    dt: float = 0.1,
    n_steps: int = 100000,
) -> float | None:
    """Measure the period of damped oscillations near endemic equilibrium.

    Near the endemic equilibrium, perturbations oscillate with an approximate
    angular frequency omega ~ sqrt(mu * beta * I*) where I* is the endemic level.

    Returns:
        Oscillation period, or None if no oscillations detected.
    """
    config = _make_config(beta=beta, gamma=gamma, mu=mu, dt=dt, n_steps=n_steps)
    sim = SIRVitalSimulation(config)
    sim.reset()

    # Skip initial transient (first large epidemic wave)
    transient = min(n_steps // 2, 50000)
    for _ in range(transient):
        sim.step()

    # Record I values after transient
    I_values = []
    for _ in range(n_steps - transient):
        sim.step()
        I_values.append(sim.observe()[1])

    I_values = np.array(I_values)

    # Detect oscillation peaks
    peaks = []
    for i in range(1, len(I_values) - 1):
        if I_values[i] > I_values[i - 1] and I_values[i] > I_values[i + 1]:
            peaks.append(i)

    if len(peaks) < 2:
        return None

    periods = np.diff(peaks) * dt
    return float(np.median(periods))


def run_sir_vital_rediscovery(
    output_dir: str | Path = "output/rediscovery/sir_vital",
    n_iterations: int = 40,
    n_samples: int = 100,
) -> dict:
    """Run the full SIR vital dynamics rediscovery pipeline.

    1. Sweep beta/gamma to map R0 and endemic I* levels
    2. Verify endemic equilibrium analytically vs numerically
    3. Detect transcritical bifurcation at R0 = 1
    4. Measure damped oscillation period
    5. SINDy ODE recovery
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sir_vital",
        "targets": {
            "R0": "beta / (gamma + mu)",
            "endemic_S": "S* = 1/R0 = (gamma+mu)/beta",
            "endemic_I": "I* = mu*(R0-1)/beta",
            "ode_S": "dS/dt = mu - beta*S*I - mu*S",
            "ode_I": "dI/dt = beta*S*I - gamma*I - mu*I",
            "ode_R": "dR/dt = gamma*I - mu*R",
        },
    }

    # --- Part 1: R0 sweep ---
    logger.info("Part 1: R0 parameter sweep...")
    sweep_data = generate_r0_sweep_data(n_samples=n_samples, n_steps=50000, dt=0.1)

    # Correlation between R0 and endemic I level
    mask_endemic = sweep_data["R0"] > 1.0
    if np.sum(mask_endemic) > 5:
        corr = np.corrcoef(
            sweep_data["R0"][mask_endemic], sweep_data["I_final"][mask_endemic]
        )[0, 1]
        results["r0_sweep"] = {
            "n_endemic": int(np.sum(mask_endemic)),
            "n_disease_free": int(np.sum(~mask_endemic)),
            "R0_I_correlation": float(corr),
        }
        logger.info(
            f"  {np.sum(mask_endemic)}/{n_samples} endemic, "
            f"R0-I correlation: {corr:.4f}"
        )

    # --- Part 2: Endemic equilibrium verification ---
    logger.info("Part 2: Endemic equilibrium verification...")
    eq_data = generate_endemic_equilibrium_data(n_samples=30, n_steps=100000, dt=0.1)

    I_rel_err = np.abs(eq_data["I_sim"] - eq_data["I_theory"]) / (
        eq_data["I_theory"] + 1e-10
    )
    S_rel_err = np.abs(eq_data["S_sim"] - eq_data["S_theory"]) / (
        eq_data["S_theory"] + 1e-10
    )
    results["endemic_equilibrium"] = {
        "I_mean_rel_error": float(np.mean(I_rel_err)),
        "I_max_rel_error": float(np.max(I_rel_err)),
        "S_mean_rel_error": float(np.mean(S_rel_err)),
        "S_max_rel_error": float(np.max(S_rel_err)),
        "n_points": len(eq_data["R0"]),
    }
    logger.info(
        f"  I* mean rel error: {np.mean(I_rel_err):.6f}, "
        f"S* mean rel error: {np.mean(S_rel_err):.6f}"
    )

    # --- Part 3: Transcritical bifurcation ---
    logger.info("Part 3: Transcritical bifurcation (beta sweep)...")
    bif_data = generate_beta_bifurcation_data(n_beta=40, n_steps=100000, dt=0.1)

    # Detect bifurcation: first beta where I_final > threshold
    threshold = 0.001
    above = bif_data["I_final"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        beta_c_est = bif_data["beta"][max(0, idx - 1)]
        beta_c_theory = bif_data["beta_c_theory"]
        results["bifurcation"] = {
            "beta_c_estimate": float(beta_c_est),
            "beta_c_theory": float(beta_c_theory),
            "relative_error": float(abs(beta_c_est - beta_c_theory) / beta_c_theory),
        }
        logger.info(
            f"  beta_c estimate: {beta_c_est:.4f} (theory: {beta_c_theory:.4f})"
        )

    # --- Part 4: Damped oscillation period ---
    logger.info("Part 4: Damped oscillation period...")
    period = measure_damped_oscillation_period(
        beta=0.4, gamma=0.1, mu=0.01, dt=0.1, n_steps=200000,
    )
    if period is not None:
        results["oscillation_period"] = float(period)
        logger.info(f"  Damped oscillation period: {period:.2f}")
    else:
        results["oscillation_period"] = None
        logger.info("  No oscillations detected (already at equilibrium)")

    # --- Part 5: SINDy ODE recovery ---
    logger.info("Part 5: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        traj_data = generate_trajectory_data(
            beta=0.4, gamma=0.1, mu=0.01, n_steps=5000, dt=0.1,
        )
        sindy_discoveries = run_sindy(
            traj_data["states"],
            dt=traj_data["time"][1] - traj_data["time"][0],
            feature_names=["S", "I", "R"],
            threshold=0.005,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries
            ],
            "true_beta": traj_data["beta"],
            "true_gamma": traj_data["gamma"],
            "true_mu": traj_data["mu"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 6: PySR for R0 ---
    logger.info("Part 6: PySR for R0 = f(beta, gamma, mu)...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use the R0 sweep data (endemic cases only)
        if np.sum(mask_endemic) > 10:
            X = np.column_stack([
                sweep_data["beta"][mask_endemic],
                sweep_data["gamma"][mask_endemic],
                sweep_data["mu"][mask_endemic],
            ])
            y = sweep_data["R0"][mask_endemic]

            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["b_", "g_", "m_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=[],
                max_complexity=10,
                populations=20,
                population_size=40,
            )
            results["R0_pysr"] = {
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
                results["R0_pysr"]["best"] = best.expression
                results["R0_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best R0: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["R0_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
