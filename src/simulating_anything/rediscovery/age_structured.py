"""Age-structured predator-prey rediscovery.

Targets:
- Equilibrium age ratios: N_j/N_a = b_N / (mu_N + m_j_N),
                           P_j/P_a = d_P / mu_P
- Maturation rate effects on population dynamics and stability
- Oscillation/stability transitions as maturation rate varies
- ODE recovery via SINDy
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.age_structured import AgeStructuredSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    b_N: float = 2.0,
    mu_N: float = 0.5,
    m_j_N: float = 0.1,
    d_N: float = 0.1,
    a: float = 0.02,
    h: float = 0.01,
    e: float = 0.5,
    mu_P: float = 0.3,
    m_j_P: float = 0.15,
    d_P: float = 0.2,
    N_j_0: float = 30.0,
    N_a_0: float = 50.0,
    P_j_0: float = 5.0,
    P_a_0: float = 10.0,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Generate a single long trajectory for SINDy ODE recovery.

    Returns dict with states array, time, and parameters.
    """
    config = SimulationConfig(
        domain=Domain.AGE_STRUCTURED,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "b_N": b_N, "mu_N": mu_N, "m_j_N": m_j_N, "d_N": d_N,
            "a": a, "h": h, "e": e,
            "mu_P": mu_P, "m_j_P": m_j_P, "d_P": d_P,
            "N_j_0": N_j_0, "N_a_0": N_a_0, "P_j_0": P_j_0, "P_a_0": P_a_0,
        },
    )

    sim = AgeStructuredSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "N_j": states_arr[:, 0],
        "N_a": states_arr[:, 1],
        "P_j": states_arr[:, 2],
        "P_a": states_arr[:, 3],
        "dt": dt,
        "b_N": b_N, "mu_N": mu_N, "m_j_N": m_j_N, "d_N": d_N,
        "a": a, "h": h, "e": e,
        "mu_P": mu_P, "m_j_P": m_j_P, "d_P": d_P,
    }


def generate_maturation_sweep_data(
    n_mu_N: int = 30,
    mu_N_min: float = 0.1,
    mu_N_max: float = 2.0,
    n_steps: int = 20000,
    n_transient: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep prey maturation rate and measure equilibrium age ratios.

    Returns dict with mu_N values, mean populations, age ratios, and amplitudes.
    """
    mu_N_vals = np.linspace(mu_N_min, mu_N_max, n_mu_N)

    config = SimulationConfig(
        domain=Domain.AGE_STRUCTURED,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "b_N": 2.0, "mu_N": 0.5, "m_j_N": 0.1, "d_N": 0.1,
            "a": 0.02, "h": 0.01, "e": 0.5,
            "mu_P": 0.3, "m_j_P": 0.15, "d_P": 0.2,
            "N_j_0": 30.0, "N_a_0": 50.0, "P_j_0": 5.0, "P_a_0": 10.0,
        },
    )

    sim = AgeStructuredSimulation(config)
    result = sim.maturation_sweep(mu_N_vals, n_steps=n_steps, n_transient=n_transient)

    # Add theoretical ratios for comparison
    theoretical_prey_ratio = []
    for mu_N_val in mu_N_vals:
        # N_j/N_a = b_N / (mu_N + m_j_N) at predator-free equilibrium
        theoretical_prey_ratio.append(2.0 / (mu_N_val + 0.1))

    result["theoretical_prey_ratio"] = np.array(theoretical_prey_ratio)
    return result


def generate_equilibrium_data(
    n_samples: int = 50,
    n_steps: int = 30000,
    n_transient: int = 15000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep parameters and measure time-averaged populations and age ratios.

    Returns dict with parameter arrays and measured averages.
    """
    rng = np.random.default_rng(42)
    n_measure = n_steps - n_transient

    all_mu_N, all_mu_P, all_b_N = [], [], []
    all_prey_ratio, all_pred_ratio = [], []
    all_N_j_avg, all_N_a_avg, all_P_j_avg, all_P_a_avg = [], [], [], []

    for i in range(n_samples):
        b_N = rng.uniform(1.0, 4.0)
        mu_N = rng.uniform(0.2, 1.5)
        mu_P = rng.uniform(0.1, 0.8)
        m_j_N = rng.uniform(0.05, 0.3)
        m_j_P = rng.uniform(0.05, 0.3)

        config = SimulationConfig(
            domain=Domain.AGE_STRUCTURED,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "b_N": b_N, "mu_N": mu_N, "m_j_N": m_j_N, "d_N": 0.1,
                "a": 0.02, "h": 0.01, "e": 0.5,
                "mu_P": mu_P, "m_j_P": m_j_P, "d_P": 0.2,
                "N_j_0": 30.0, "N_a_0": 50.0, "P_j_0": 5.0, "P_a_0": 10.0,
            },
        )

        sim = AgeStructuredSimulation(config)
        sim.reset()

        # Transient
        for _ in range(n_transient):
            sim.step()

        # Measure
        N_j_vals, N_a_vals, P_j_vals, P_a_vals = [], [], [], []
        for _ in range(n_measure):
            state = sim.step()
            N_j_vals.append(state[0])
            N_a_vals.append(state[1])
            P_j_vals.append(state[2])
            P_a_vals.append(state[3])

        N_j_mean = np.mean(N_j_vals)
        N_a_mean = np.mean(N_a_vals)
        P_j_mean = np.mean(P_j_vals)
        P_a_mean = np.mean(P_a_vals)

        all_mu_N.append(mu_N)
        all_mu_P.append(mu_P)
        all_b_N.append(b_N)
        all_N_j_avg.append(N_j_mean)
        all_N_a_avg.append(N_a_mean)
        all_P_j_avg.append(P_j_mean)
        all_P_a_avg.append(P_a_mean)
        all_prey_ratio.append(
            N_j_mean / N_a_mean if N_a_mean > 1e-10 else 0.0
        )
        all_pred_ratio.append(
            P_j_mean / P_a_mean if P_a_mean > 1e-10 else 0.0
        )

        if (i + 1) % 10 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} trajectories")

    return {
        "mu_N": np.array(all_mu_N),
        "mu_P": np.array(all_mu_P),
        "b_N": np.array(all_b_N),
        "N_j_avg": np.array(all_N_j_avg),
        "N_a_avg": np.array(all_N_a_avg),
        "P_j_avg": np.array(all_P_j_avg),
        "P_a_avg": np.array(all_P_a_avg),
        "prey_ratio": np.array(all_prey_ratio),
        "predator_ratio": np.array(all_pred_ratio),
    }


def run_age_structured_rediscovery(
    output_dir: str | Path = "output/rediscovery/age_structured",
    n_iterations: int = 50,
    n_equilibrium_samples: int = 50,
) -> dict:
    """Run the full age-structured predator-prey rediscovery.

    1. Generate a trajectory for SINDy ODE recovery
    2. Sweep maturation rate to study juvenile-adult dynamics
    3. Measure equilibrium age ratios N_j/N_a, P_j/P_a
    4. Detect oscillation/stability transitions

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "age_structured",
        "targets": {
            "prey_age_ratio": "N_j/N_a = b_N / (mu_N + m_j_N)",
            "predator_age_ratio": "P_j/P_a = d_P / mu_P",
            "ode_N_j": "dN_j/dt = b_N*N_a - mu_N*N_j - m_j_N*N_j",
            "ode_N_a": "dN_a/dt = mu_N*N_j - d_N*N_a - a*N_a*P_a/(1+h*N_a)",
            "ode_P_j": "dP_j/dt = e*a*N_a*P_a/(1+h*N_a) - mu_P*P_j - m_j_P*P_j",
            "ode_P_a": "dP_a/dt = mu_P*P_j - d_P*P_a",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for age-structured predator-prey...")
    data = generate_trajectory_data(n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["N_j", "N_a", "P_j", "P_a"],
            threshold=0.05,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries
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
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: Maturation rate sweep ---
    logger.info("Part 2: Maturation rate sweep...")
    sweep_data = generate_maturation_sweep_data(
        n_mu_N=30, mu_N_min=0.1, mu_N_max=2.0,
        n_steps=20000, n_transient=10000,
    )

    # Identify stability transitions (oscillation onset)
    oscillating = sweep_data["amplitude_N"] > 1.0
    n_oscillating = int(np.sum(oscillating))

    results["maturation_sweep"] = {
        "n_points": len(sweep_data["mu_N"]),
        "mu_N_range": [float(sweep_data["mu_N"].min()),
                       float(sweep_data["mu_N"].max())],
        "n_oscillating": n_oscillating,
        "n_stable": len(sweep_data["mu_N"]) - n_oscillating,
        "prey_ratio_mean": float(np.mean(sweep_data["prey_ratio"])),
        "predator_ratio_mean": float(np.mean(sweep_data["predator_ratio"])),
    }

    # --- Part 3: Equilibrium data ---
    logger.info("Part 3: Equilibrium data across parameter sweeps...")
    eq_data = generate_equilibrium_data(
        n_samples=n_equilibrium_samples, n_steps=30000, n_transient=15000,
    )

    results["equilibrium_data"] = {
        "n_samples": len(eq_data["mu_N"]),
        "prey_ratio_mean": float(np.mean(eq_data["prey_ratio"])),
        "predator_ratio_mean": float(np.mean(eq_data["predator_ratio"])),
        "N_j_avg_mean": float(np.mean(eq_data["N_j_avg"])),
        "N_a_avg_mean": float(np.mean(eq_data["N_a_avg"])),
        "P_j_avg_mean": float(np.mean(eq_data["P_j_avg"])),
        "P_a_avg_mean": float(np.mean(eq_data["P_a_avg"])),
    }

    logger.info(
        f"  Mean prey ratio: {np.mean(eq_data['prey_ratio']):.3f}, "
        f"predator ratio: {np.mean(eq_data['predator_ratio']):.3f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "trajectory_data.npz",
        states=data["states"],
    )
    np.savez(
        output_path / "maturation_sweep.npz",
        **{k: v for k, v in sweep_data.items()},
    )
    np.savez(
        output_path / "equilibrium_data.npz",
        **{k: v for k, v in eq_data.items()},
    )

    return results
