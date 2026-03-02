"""SIR with vaccination rediscovery.

Targets:
- R_0 = beta / (gamma + mu) (basic reproduction number)
- Herd immunity threshold: p_c = 1 - 1/R_0
- Critical vaccination rate: nu_c = mu * (R_0 - 1)
- R_eff = R_0 * mu / (nu + mu) (effective reproduction number)
- SINDy recovery of the SIR-vaccination ODEs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sir_vaccination import (
    SIRVaccinationSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.SIR_VACCINATION


def _make_config(
    beta: float = 0.3,
    gamma: float = 0.1,
    mu: float = 0.01,
    nu: float = 0.0,
    N: float = 1000.0,
    S_0: float | None = None,
    I_0: float | None = None,
    dt: float = 0.1,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Create a SimulationConfig for SIR-vaccination."""
    params: dict[str, float] = {
        "beta": beta,
        "gamma": gamma,
        "mu": mu,
        "nu": nu,
        "N": N,
    }
    if S_0 is not None:
        params["S_0"] = S_0
    if I_0 is not None:
        params["I_0"] = I_0
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


def generate_vaccination_sweep_data(
    n_nu: int = 30,
    beta: float = 0.3,
    gamma: float = 0.1,
    mu: float = 0.01,
    N: float = 1000.0,
    n_steps: int = 50000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep vaccination rate nu and measure long-term endemic infected.

    For each nu, run until steady state and record the final I/N.
    This traces the transition from endemic to disease-free as nu increases.
    """
    r0 = beta / (gamma + mu)
    nu_c_theory = mu * (r0 - 1.0) if r0 > 1 else 0.0
    nu_max = nu_c_theory * 2.5 if nu_c_theory > 0 else 0.1
    nu_values = np.linspace(0.0, nu_max, n_nu)

    final_I_frac = []
    final_S_frac = []
    r_eff_values = []

    for i, nu in enumerate(nu_values):
        config = _make_config(
            beta=beta, gamma=gamma, mu=mu, nu=nu, N=N,
            S_0=N * 0.99, I_0=N * 0.01,
            dt=dt, n_steps=n_steps,
        )
        sim = SIRVaccinationSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        state = sim.observe()
        total = state.sum()
        final_I_frac.append(float(state[1] / total) if total > 0 else 0.0)
        final_S_frac.append(float(state[0] / total) if total > 0 else 0.0)
        r_eff_values.append(sim.compute_r_eff())

        if (i + 1) % 10 == 0:
            logger.info(
                f"  nu={nu:.4f}: I/N={final_I_frac[-1]:.6f}, "
                f"R_eff={r_eff_values[-1]:.4f}"
            )

    return {
        "nu": nu_values,
        "final_I_frac": np.array(final_I_frac),
        "final_S_frac": np.array(final_S_frac),
        "r_eff": np.array(r_eff_values),
        "nu_c_theory": nu_c_theory,
        "r0": r0,
        "beta": beta,
        "gamma": gamma,
        "mu": mu,
    }


def generate_r0_sweep_data(
    n_samples: int = 150,
    n_steps: int = 50000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep beta, gamma, mu to measure R_0 and herd immunity threshold.

    For each parameter combination, compute theoretical R_0 and herd
    immunity threshold, then verify against simulation behavior.
    """
    rng = np.random.default_rng(42)

    all_beta = []
    all_gamma = []
    all_mu = []
    all_r0 = []
    all_pc = []
    all_epidemic = []
    all_peak_I = []

    for i in range(n_samples):
        beta = rng.uniform(0.1, 0.8)
        gamma = rng.uniform(0.02, 0.3)
        mu = rng.uniform(0.001, 0.05)

        r0 = beta / (gamma + mu)
        pc = max(0.0, 1.0 - 1.0 / r0) if r0 > 1 else 0.0

        config = _make_config(
            beta=beta, gamma=gamma, mu=mu, nu=0.0, N=1000.0,
            S_0=990.0, I_0=10.0,
            dt=dt, n_steps=n_steps,
        )
        sim = SIRVaccinationSimulation(config)
        sim.reset()

        peak_I = 0.0
        for _ in range(n_steps):
            state = sim.step()
            if state[1] > peak_I:
                peak_I = state[1]

        epidemic = peak_I > 20.0  # Significant epidemic

        all_beta.append(beta)
        all_gamma.append(gamma)
        all_mu.append(mu)
        all_r0.append(r0)
        all_pc.append(pc)
        all_epidemic.append(epidemic)
        all_peak_I.append(peak_I)

        if (i + 1) % 50 == 0:
            logger.info(
                f"  Generated {i + 1}/{n_samples} R0 sweep trajectories"
            )

    return {
        "beta": np.array(all_beta),
        "gamma": np.array(all_gamma),
        "mu": np.array(all_mu),
        "R0": np.array(all_r0),
        "p_c": np.array(all_pc),
        "epidemic": np.array(all_epidemic),
        "peak_I": np.array(all_peak_I),
    }


def generate_ode_data(
    beta: float = 0.3,
    gamma: float = 0.1,
    mu: float = 0.01,
    nu: float = 0.02,
    N: float = 1000.0,
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate a single trajectory for SINDy ODE recovery."""
    config = _make_config(
        beta=beta, gamma=gamma, mu=mu, nu=nu, N=N,
        S_0=N * 0.8, I_0=N * 0.05,
        dt=dt, n_steps=n_steps,
    )
    sim = SIRVaccinationSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "beta": beta,
        "gamma": gamma,
        "mu": mu,
        "nu": nu,
        "N": N,
    }


def run_sir_vaccination_rediscovery(
    output_dir: str | Path = "output/rediscovery/sir_vaccination",
    n_iterations: int = 40,
) -> dict:
    """Run the SIR-vaccination rediscovery pipeline.

    Discovers:
    1. R_0 = beta / (gamma + mu) via PySR
    2. Herd immunity threshold p_c = 1 - 1/R_0
    3. Critical vaccination rate nu_c = mu*(R_0 - 1)
    4. ODE recovery via SINDy
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "sir_vaccination",
        "targets": {
            "R0": "beta / (gamma + mu)",
            "herd_immunity": "p_c = 1 - 1/R_0",
            "critical_nu": "nu_c = mu * (R_0 - 1)",
            "ode_S": "dS/dt = mu*N - beta*S*I/N - nu*S - mu*S",
            "ode_I": "dI/dt = beta*S*I/N - gamma*I - mu*I",
            "ode_R": "dR/dt = gamma*I + nu*S - mu*R",
        },
    }

    # --- Part 1: R_0 rediscovery via PySR ---
    logger.info("Part 1: R_0 = beta/(gamma+mu) via parameter sweep...")
    r0_data = generate_r0_sweep_data(
        n_samples=150, n_steps=20000, dt=0.1,
    )

    # PySR: predict R0 from beta, gamma, mu
    mask = r0_data["R0"] > 0.5  # All valid points
    X = np.column_stack([
        r0_data["beta"][mask],
        r0_data["gamma"][mask],
        r0_data["mu"][mask],
    ])
    y_r0 = r0_data["R0"][mask]

    results["r0_data"] = {
        "n_samples": int(mask.sum()),
        "r0_range": [float(y_r0.min()), float(y_r0.max())],
    }

    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        logger.info("  Running PySR for R0 = f(beta, gamma, mu)...")
        r0_discoveries = run_symbolic_regression(
            X, y_r0,
            variable_names=["b_", "g_", "m_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            max_complexity=10,
            populations=20,
            population_size=40,
        )
        results["R0_pysr"] = {
            "n_discoveries": len(r0_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in r0_discoveries[:5]
            ],
        }
        if r0_discoveries:
            best = r0_discoveries[0]
            results["R0_pysr"]["best"] = best.expression
            results["R0_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best R0: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR R0 failed: {e}")
        results["R0_pysr"] = {"error": str(e)}

    # --- Part 2: Vaccination sweep and herd immunity ---
    logger.info(
        "Part 2: Vaccination sweep for herd immunity threshold..."
    )
    vacc_data = generate_vaccination_sweep_data(
        n_nu=30, beta=0.3, gamma=0.1, mu=0.01,
        N=1000.0, n_steps=30000, dt=0.1,
    )

    # Detect critical nu: first nu where I/N drops to ~0
    threshold = 1e-4
    endemic_mask = vacc_data["final_I_frac"] > threshold
    if np.any(~endemic_mask):
        idx = np.argmax(~endemic_mask)
        nu_c_est = float(vacc_data["nu"][max(0, idx - 1)])
    else:
        nu_c_est = float(vacc_data["nu"][-1])

    nu_c_theory = vacc_data["nu_c_theory"]
    results["vaccination_sweep"] = {
        "n_points": len(vacc_data["nu"]),
        "nu_c_estimate": nu_c_est,
        "nu_c_theory": nu_c_theory,
        "relative_error": float(
            abs(nu_c_est - nu_c_theory) / nu_c_theory
        ) if nu_c_theory > 0 else 0.0,
        "r0": vacc_data["r0"],
        "herd_threshold_theory": float(
            1.0 - 1.0 / vacc_data["r0"]
        ) if vacc_data["r0"] > 1 else 0.0,
    }
    logger.info(
        f"  nu_c estimate: {nu_c_est:.4f} "
        f"(theory: {nu_c_theory:.4f})"
    )

    # --- Part 3: Herd immunity threshold via PySR ---
    logger.info("Part 3: Herd immunity threshold p_c = f(R0)...")
    mask_epidemic = r0_data["R0"] > 1.2
    if np.sum(mask_epidemic) > 10:
        X_pc = r0_data["R0"][mask_epidemic].reshape(-1, 1)
        y_pc = r0_data["p_c"][mask_epidemic]

        try:
            pc_discoveries = run_symbolic_regression(
                X_pc, y_pc,
                variable_names=["R0_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=[],
                max_complexity=8,
                populations=20,
                population_size=40,
            )
            results["pc_pysr"] = {
                "n_discoveries": len(pc_discoveries),
                "discoveries": [
                    {
                        "expression": d.expression,
                        "r_squared": d.evidence.fit_r_squared,
                    }
                    for d in pc_discoveries[:5]
                ],
            }
            if pc_discoveries:
                best = pc_discoveries[0]
                results["pc_pysr"]["best"] = best.expression
                results["pc_pysr"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best p_c: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        except Exception as e:
            logger.warning(f"PySR p_c failed: {e}")
            results["pc_pysr"] = {"error": str(e)}

    # --- Part 4: SINDy ODE recovery ---
    logger.info("Part 4: SINDy ODE recovery...")
    ode_data = generate_ode_data(
        beta=0.3, gamma=0.1, mu=0.01, nu=0.02,
        N=1000.0, n_steps=5000, dt=0.1,
    )

    try:
        from simulating_anything.analysis.equation_discovery import (
            run_sindy,
        )

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["S", "I", "R"],
            threshold=0.01,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries[:5]
            ],
        }
        if sindy_discoveries:
            best = sindy_discoveries[0]
            results["sindy_ode"]["best"] = best.expression
            results["sindy_ode"]["best_r2"] = (
                best.evidence.fit_r_squared
            )
            logger.info(
                f"  SINDy best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 5: Equilibrium verification ---
    logger.info("Part 5: Equilibrium verification...")
    config_eq = _make_config(
        beta=0.3, gamma=0.1, mu=0.01, nu=0.0, N=1000.0,
    )
    sim_eq = SIRVaccinationSimulation(config_eq)
    r0_val = sim_eq.compute_r0()
    dfe = sim_eq.disease_free_equilibrium()
    ee = sim_eq.endemic_equilibrium()
    pc = sim_eq.herd_immunity_threshold()
    nu_c = sim_eq.critical_vaccination_rate()

    results["equilibrium_verification"] = {
        "R0": r0_val,
        "DFE": [float(x) for x in dfe],
        "endemic_eq": (
            [float(x) for x in ee] if ee is not None else None
        ),
        "herd_threshold": pc,
        "critical_nu": nu_c,
    }
    logger.info(
        f"  R0={r0_val:.4f}, p_c={pc:.4f}, nu_c={nu_c:.6f}"
    )
    if ee is not None:
        logger.info(
            f"  Endemic eq: S*={ee[0]:.1f}, "
            f"I*={ee[1]:.1f}, R*={ee[2]:.1f}"
        )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save sweep data
    np.savez(
        output_path / "vaccination_sweep.npz",
        **{
            k: v
            for k, v in vacc_data.items()
            if isinstance(v, np.ndarray)
        },
    )

    return results
