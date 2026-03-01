"""Predator-Prey-Mutualist ODE rediscovery.

Targets:
- ODE recovery via SINDy:
    dx/dt = r*x*(1 - x/K) - a*x*y/(1 + b*x) + m*x*z/(1 + n*z)
    dy/dt = -d*y + e*a*x*y/(1 + b*x)
    dz/dt = s*z*(1 - z/C) + p*x*z/(1 + n*z)
- Mutualism stabilization: higher m reduces oscillation amplitude
- Equilibrium analysis: multiple coexistence equilibria
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.predator_prey_mutualist import (
    PredatorPreyMutualistSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    r: float = 1.0,
    K: float = 10.0,
    a: float = 1.0,
    b: float = 0.1,
    m: float = 0.5,
    n: float = 0.2,
    d: float = 0.4,
    e: float = 0.6,
    s: float = 0.8,
    C: float = 8.0,
    p: float = 0.3,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Generate a single long trajectory for SINDy ODE recovery.

    Returns dict with states array, time, and parameters.
    """
    config = SimulationConfig(
        domain=Domain.PREDATOR_PREY_MUTUALIST,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "b": b, "m": m, "n": n,
            "d": d, "e": e, "s": s, "C": C, "p": p,
            "x_0": 5.0, "y_0": 2.0, "z_0": 3.0,
        },
    )

    sim = PredatorPreyMutualistSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "x": states_arr[:, 0],
        "y": states_arr[:, 1],
        "z": states_arr[:, 2],
        "dt": dt,
        "r": r, "K": K, "a": a, "b": b, "m": m, "n": n,
        "d": d, "e": e, "s": s, "C": C, "p": p,
    }


def generate_mutualism_sweep(
    n_m: int = 20,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep mutualism strength m and measure oscillation amplitude.

    Varying m demonstrates that mutualism stabilizes predator-prey oscillations.
    Higher m leads to smaller oscillation amplitudes in prey/predator.

    Returns dict with m values and measured amplitudes/means.
    """
    m_values = np.linspace(0.0, 2.0, n_m)

    all_m = []
    all_prey_amp = []
    all_pred_amp = []
    all_prey_mean = []
    all_pred_mean = []
    all_mutualist_mean = []

    for i, m_val in enumerate(m_values):
        config = SimulationConfig(
            domain=Domain.PREDATOR_PREY_MUTUALIST,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 1.0, "K": 10.0, "a": 1.0, "b": 0.1,
                "m": float(m_val), "n": 0.2,
                "d": 0.4, "e": 0.6, "s": 0.8, "C": 8.0, "p": 0.3,
                "x_0": 5.0, "y_0": 2.0, "z_0": 3.0,
            },
        )

        sim = PredatorPreyMutualistSimulation(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            states.append(sim.step().copy())

        trajectory = np.array(states)

        # Skip initial transient (first 50%)
        skip = n_steps // 2
        steady = trajectory[skip:]

        prey = steady[:, 0]
        pred = steady[:, 1]
        mutualist = steady[:, 2]

        prey_amp = float(np.max(prey) - np.min(prey))
        pred_amp = float(np.max(pred) - np.min(pred))

        all_m.append(float(m_val))
        all_prey_amp.append(prey_amp)
        all_pred_amp.append(pred_amp)
        all_prey_mean.append(float(np.mean(prey)))
        all_pred_mean.append(float(np.mean(pred)))
        all_mutualist_mean.append(float(np.mean(mutualist)))

        if (i + 1) % 5 == 0:
            logger.info(
                f"  m={m_val:.2f}: prey_amp={prey_amp:.3f}, "
                f"pred_amp={pred_amp:.3f}"
            )

    return {
        "m": np.array(all_m),
        "prey_amplitude": np.array(all_prey_amp),
        "pred_amplitude": np.array(all_pred_amp),
        "prey_mean": np.array(all_prey_mean),
        "pred_mean": np.array(all_pred_mean),
        "mutualist_mean": np.array(all_mutualist_mean),
    }


def generate_equilibrium_data(
    n_initial: int = 20,
) -> dict[str, list]:
    """Find equilibria and classify their stability.

    Returns dict with equilibrium points and eigenvalue information.
    """
    config = SimulationConfig(
        domain=Domain.PREDATOR_PREY_MUTUALIST,
        dt=0.01,
        n_steps=1000,
        parameters={
            "r": 1.0, "K": 10.0, "a": 1.0, "b": 0.1,
            "m": 0.5, "n": 0.2, "d": 0.4, "e": 0.6,
            "s": 0.8, "C": 8.0, "p": 0.3,
        },
    )

    sim = PredatorPreyMutualistSimulation(config)
    sim.reset()

    equilibria = sim.find_equilibria(n_initial=n_initial)

    eq_data: list[dict] = []
    for eq in equilibria:
        eigs = sim.stability_eigenvalues(eq)
        max_real = float(np.max(np.real(eigs)))
        stable = max_real < 0

        eq_data.append({
            "point": eq.tolist(),
            "eigenvalues_real": np.real(eigs).tolist(),
            "eigenvalues_imag": np.imag(eigs).tolist(),
            "max_real_eigenvalue": max_real,
            "is_stable": stable,
        })

    return {
        "n_equilibria": len(equilibria),
        "equilibria": eq_data,
    }


def run_predator_prey_mutualist_rediscovery(
    output_dir: str | Path = "output/rediscovery/predator_prey_mutualist",
    n_iterations: int = 40,
    n_mutualism_sweep: int = 20,
) -> dict:
    """Run the full Predator-Prey-Mutualist rediscovery.

    1. Generate a trajectory for SINDy ODE recovery
    2. Run SINDy to recover the three coupled ODEs
    3. Sweep mutualism strength to show stabilization effect
    4. Find and classify equilibria

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "predator_prey_mutualist",
        "targets": {
            "ode_x": "dx/dt = r*x*(1-x/K) - a*x*y/(1+b*x) + m*x*z/(1+n*z)",
            "ode_y": "dy/dt = -d*y + e*a*x*y/(1+b*x)",
            "ode_z": "dz/dt = s*z*(1-z/C) + p*x*z/(1+n*z)",
            "stabilization": "Mutualism (m) reduces oscillation amplitude",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for predator-prey-mutualist...")
    data = generate_trajectory_data(n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["x", "y", "z"],
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

    # --- Part 2: Mutualism stabilization sweep ---
    logger.info("Part 2: Sweeping mutualism strength m...")
    sweep_data = generate_mutualism_sweep(
        n_m=n_mutualism_sweep, n_steps=20000, dt=0.01,
    )

    # Compute correlation: does higher m reduce prey amplitude?
    m_arr = sweep_data["m"]
    prey_amp = sweep_data["prey_amplitude"]
    if len(m_arr) > 2 and np.std(prey_amp) > 1e-10:
        corr = float(np.corrcoef(m_arr, prey_amp)[0, 1])
    else:
        corr = 0.0

    results["mutualism_sweep"] = {
        "n_points": len(m_arr),
        "m_range": [float(m_arr[0]), float(m_arr[-1])],
        "prey_amplitude_range": [float(np.min(prey_amp)), float(np.max(prey_amp))],
        "m_vs_prey_amplitude_correlation": corr,
        "stabilization_detected": corr < -0.3,
    }
    logger.info(
        f"  Mutualism-amplitude correlation: {corr:.3f} "
        f"(stabilization={'YES' if corr < -0.3 else 'NO'})"
    )

    # --- Part 3: Equilibrium analysis ---
    logger.info("Part 3: Equilibrium analysis...")
    eq_data = generate_equilibrium_data(n_initial=30)

    results["equilibrium_analysis"] = eq_data
    n_stable = sum(1 for e in eq_data["equilibria"] if e["is_stable"])
    logger.info(
        f"  Found {eq_data['n_equilibria']} equilibria, "
        f"{n_stable} stable"
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
        output_path / "mutualism_sweep.npz",
        **{k: v for k, v in sweep_data.items()},
    )

    return results
