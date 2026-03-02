"""Predator-prey with refuge rediscovery.

Targets:
- Equilibrium: N* = delta/(beta*alpha*(1-m)), P* = r*(1-N*/K)/(alpha*(1-m))
- Refuge effect: increasing m raises prey equilibrium, lowers predator
- Stabilization: increasing m reduces oscillation amplitude
- ODE recovery via SINDy: dN/dt = r*N*(1-N/K) - alpha*(1-m)*N*P
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.prey_refuge import PreyRefugeSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use DUFFING as placeholder domain enum value
_DOMAIN = Domain.PREY_REFUGE


def _make_config(
    r: float = 1.0,
    K: float = 10.0,
    alpha: float = 0.5,
    m: float = 0.3,
    beta: float = 0.4,
    delta: float = 0.2,
    N_0: float = 2.0,
    P_0: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 2000,
) -> SimulationConfig:
    """Build a SimulationConfig for the prey-refuge model."""
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "alpha": alpha, "m": m,
            "beta": beta, "delta": delta,
            "N_0": N_0, "P_0": P_0,
        },
    )


def generate_trajectory_data(
    r: float = 1.0,
    K: float = 10.0,
    alpha: float = 0.5,
    m: float = 0.3,
    beta: float = 0.4,
    delta: float = 0.2,
    N_0: float = 2.0,
    P_0: float = 1.0,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Generate a single long trajectory for SINDy ODE recovery.

    Returns dict with states array, time, and parameters.
    """
    config = _make_config(
        r=r, K=K, alpha=alpha, m=m, beta=beta, delta=delta,
        N_0=N_0, P_0=P_0, dt=dt, n_steps=n_steps,
    )

    sim = PreyRefugeSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "N": states_arr[:, 0],
        "P": states_arr[:, 1],
        "dt": dt,
        "r": r, "K": K, "alpha": alpha, "m": m,
        "beta": beta, "delta": delta,
    }


def generate_refuge_sweep_data(
    n_m_values: int = 20,
    m_min: float = 0.0,
    m_max: float = 0.9,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep refuge fraction m and measure equilibrium + stability.

    For each m, runs a long trajectory and measures:
    - Time-averaged prey and predator populations
    - Oscillation amplitude (max - min in second half)
    - Theoretical equilibrium values

    Returns dict with m values, averages, amplitudes, and theory.
    """
    m_values = np.linspace(m_min, m_max, n_m_values)

    all_m = []
    all_N_avg = []
    all_P_avg = []
    all_N_amp = []
    all_P_amp = []
    all_N_star = []
    all_P_star = []

    for i, m_val in enumerate(m_values):
        config = _make_config(
            m=float(m_val), dt=dt, n_steps=n_steps,
            N_0=2.0, P_0=1.0,
        )

        sim = PreyRefugeSimulation(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)

        # Use second half to avoid transients
        half = n_steps // 2
        N_half = trajectory[half:, 0]
        P_half = trajectory[half:, 1]

        all_m.append(float(m_val))
        all_N_avg.append(float(np.mean(N_half)))
        all_P_avg.append(float(np.mean(P_half)))
        all_N_amp.append(float(np.max(N_half) - np.min(N_half)))
        all_P_amp.append(float(np.max(P_half) - np.min(P_half)))

        # Theoretical equilibrium
        try:
            N_star, P_star = sim.equilibrium()
            all_N_star.append(N_star)
            all_P_star.append(P_star)
        except ValueError:
            all_N_star.append(float("nan"))
            all_P_star.append(float("nan"))

        if (i + 1) % 5 == 0:
            logger.info(f"  Refuge sweep: {i + 1}/{n_m_values} m values")

    return {
        "m": np.array(all_m),
        "N_avg": np.array(all_N_avg),
        "P_avg": np.array(all_P_avg),
        "N_amplitude": np.array(all_N_amp),
        "P_amplitude": np.array(all_P_amp),
        "N_star_theory": np.array(all_N_star),
        "P_star_theory": np.array(all_P_star),
    }


def run_prey_refuge_rediscovery(
    output_dir: str | Path = "output/rediscovery/prey_refuge",
    n_iterations: int = 40,
    n_m_values: int = 20,
) -> dict:
    """Run the full prey-refuge rediscovery pipeline.

    1. Generate a trajectory for SINDy ODE recovery
    2. Sweep refuge fraction m to show stabilization effect
    3. Verify equilibrium predictions match simulation
    4. Show prey increase / predator decrease with refuge

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "prey_refuge",
        "targets": {
            "prey_ode": "dN/dt = r*N*(1-N/K) - alpha*(1-m)*N*P",
            "pred_ode": "dP/dt = beta*alpha*(1-m)*N*P - delta*P",
            "N_star": "delta / (beta*alpha*(1-m))",
            "P_star": "r*(1-N*/K) / (alpha*(1-m))",
            "stabilization": "increasing m reduces oscillation amplitude",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for prey-refuge model...")
    data = generate_trajectory_data(
        r=1.0, K=10.0, alpha=0.5, m=0.3, beta=0.4, delta=0.2,
        n_steps=10000, dt=0.005,
    )

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["N", "P"],
            threshold=0.05,
            poly_degree=3,
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
        for disc in sindy_discoveries:
            logger.info(f"  SINDy: {disc.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: Refuge sweep ---
    logger.info("Part 2: Refuge fraction sweep...")
    sweep_data = generate_refuge_sweep_data(
        n_m_values=n_m_values, m_min=0.0, m_max=0.9,
        n_steps=20000, dt=0.01,
    )

    # Check that prey equilibrium increases with m
    valid = ~np.isnan(sweep_data["N_star_theory"])
    if np.sum(valid) >= 2:
        N_stars = sweep_data["N_star_theory"][valid]
        prey_increasing = bool(np.all(np.diff(N_stars) > 0))
    else:
        prey_increasing = False

    # Check that oscillation amplitude decreases with m
    amps = sweep_data["N_amplitude"]
    amp_decreasing = bool(amps[-1] < amps[0]) if len(amps) >= 2 else False

    # Correlation between simulated averages and theory
    valid_mask = valid & (sweep_data["N_avg"] > 0.01)
    if np.sum(valid_mask) >= 3:
        corr_N = float(np.corrcoef(
            sweep_data["N_avg"][valid_mask],
            sweep_data["N_star_theory"][valid_mask],
        )[0, 1])
    else:
        corr_N = float("nan")

    results["refuge_sweep"] = {
        "n_m_values": n_m_values,
        "m_range": [float(sweep_data["m"][0]), float(sweep_data["m"][-1])],
        "prey_eq_increases_with_m": prey_increasing,
        "amplitude_decreases_with_m": amp_decreasing,
        "N_avg_theory_correlation": corr_N,
        "max_N_amplitude": float(np.max(sweep_data["N_amplitude"])),
        "min_N_amplitude": float(np.min(sweep_data["N_amplitude"])),
        "N_star_at_m0": float(sweep_data["N_star_theory"][0])
        if valid[0] else None,
        "N_star_at_m_max": float(sweep_data["N_star_theory"][-1])
        if valid[-1] else None,
    }

    logger.info(f"  Prey eq. increases with m: {prey_increasing}")
    logger.info(f"  Amplitude decreases with m: {amp_decreasing}")
    logger.info(f"  N_avg vs theory correlation: {corr_N:.4f}")

    # --- Part 3: Equilibrium verification ---
    logger.info("Part 3: Equilibrium verification...")
    config_eq = _make_config(m=0.3, n_steps=50000, dt=0.01)
    sim_eq = PreyRefugeSimulation(config_eq)
    try:
        N_star, P_star = sim_eq.equilibrium()
        sim_eq.reset()
        for _ in range(50000):
            sim_eq.step()
        N_final, P_final = sim_eq.observe()

        # For stable equilibria, simulation should converge
        stable = sim_eq.is_stable()
        if stable:
            N_err = abs(N_final - N_star) / max(N_star, 1e-10)
            P_err = abs(P_final - P_star) / max(P_star, 1e-10)
        else:
            # For unstable equilibria, check time-average
            sim_eq2 = PreyRefugeSimulation(config_eq)
            sim_eq2.reset()
            states_eq = []
            for _ in range(50000):
                sim_eq2.step()
                states_eq.append(sim_eq2.observe().copy())
            traj_eq = np.array(states_eq)
            half = len(traj_eq) // 2
            N_err = abs(np.mean(traj_eq[half:, 0]) - N_star) / max(N_star, 1e-10)
            P_err = abs(np.mean(traj_eq[half:, 1]) - P_star) / max(P_star, 1e-10)

        results["equilibrium_verification"] = {
            "N_star_theory": N_star,
            "P_star_theory": P_star,
            "is_stable": stable,
            "N_error_relative": float(N_err),
            "P_error_relative": float(P_err),
            "effective_predation": sim_eq.effective_predation_rate(),
        }
        logger.info(f"  N* = {N_star:.4f}, P* = {P_star:.4f}, stable={stable}")
        logger.info(f"  Relative errors: N={N_err:.4f}, P={P_err:.4f}")
    except ValueError as exc:
        results["equilibrium_verification"] = {"error": str(exc)}
        logger.warning(f"  Equilibrium does not exist: {exc}")

    # --- Part 4: No-refuge comparison ---
    logger.info("Part 4: No-refuge (m=0) vs high-refuge (m=0.8) comparison...")
    comparisons = {}
    for m_val, label in [(0.0, "no_refuge"), (0.8, "high_refuge")]:
        config_cmp = _make_config(m=m_val, n_steps=20000, dt=0.01)
        sim_cmp = PreyRefugeSimulation(config_cmp)
        sim_cmp.reset()

        states_cmp = []
        for _ in range(20000):
            sim_cmp.step()
            states_cmp.append(sim_cmp.observe().copy())

        traj_cmp = np.array(states_cmp)
        half = len(traj_cmp) // 2
        comparisons[label] = {
            "m": m_val,
            "N_avg": float(np.mean(traj_cmp[half:, 0])),
            "P_avg": float(np.mean(traj_cmp[half:, 1])),
            "N_amplitude": float(
                np.max(traj_cmp[half:, 0]) - np.min(traj_cmp[half:, 0])
            ),
            "P_amplitude": float(
                np.max(traj_cmp[half:, 1]) - np.min(traj_cmp[half:, 1])
            ),
        }
        logger.info(
            f"  m={m_val}: N_avg={comparisons[label]['N_avg']:.3f}, "
            f"N_amp={comparisons[label]['N_amplitude']:.3f}"
        )

    results["comparison"] = comparisons

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
        output_path / "refuge_sweep_data.npz",
        **{k: v for k, v in sweep_data.items()},
    )

    return results
