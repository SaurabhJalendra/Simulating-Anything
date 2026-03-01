"""Allee effect predator-prey rediscovery.

Targets:
- Bistability: two ICs -> two different outcomes (extinction vs coexistence)
- Allee threshold effect: prey declines when N < A
- Critical predator density for prey extinction
- Allee growth function shape: r*N*(N/A - 1)*(1 - N/K)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.allee_predator_prey import (
    AlleePredatorPreySimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use ALLEE_PREDATOR_PREY domain enum
_DOMAIN = Domain.ALLEE_PREDATOR_PREY


def _make_config(
    N_0: float = 50.0,
    P_0: float = 5.0,
    r: float = 1.0,
    A: float = 10.0,
    K: float = 100.0,
    a: float = 0.01,
    h: float = 0.1,
    e: float = 0.5,
    m: float = 0.3,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Build a SimulationConfig for the Allee predator-prey model."""
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "A": A, "K": K,
            "a": a, "h": h, "e": e, "m": m,
            "N_0": N_0, "P_0": P_0,
        },
    )


def generate_bistability_data(
    N_above: float = 50.0,
    N_below: float = 5.0,
    P_0: float = 5.0,
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Demonstrate bistability: two ICs lead to different outcomes.

    Runs one trajectory starting above the Allee threshold (N > A)
    and one starting below (N < A), both with the same predator density.

    Returns:
        Dict with trajectories for both initial conditions.
    """
    results = {}

    for label, N_init in [("above", N_above), ("below", N_below)]:
        config = _make_config(N_0=N_init, P_0=P_0, dt=dt, n_steps=n_steps)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        traj = np.array(states)
        results[f"{label}_trajectory"] = traj
        results[f"{label}_final_N"] = float(traj[-1, 0])
        results[f"{label}_final_P"] = float(traj[-1, 1])
        results[f"{label}_N_0"] = N_init

        logger.info(
            f"  IC N={N_init}: final N={traj[-1, 0]:.2f}, P={traj[-1, 1]:.2f}"
        )

    return results


def generate_allee_sweep_data(
    A_values: np.ndarray | None = None,
    N_0: float = 50.0,
    P_0: float = 5.0,
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep Allee threshold A and measure extinction probability.

    For each A, runs the simulation and checks if prey survives.

    Returns:
        Dict with A_values and extinction outcomes.
    """
    if A_values is None:
        A_values = np.linspace(5.0, 60.0, 20)

    final_N = np.zeros(len(A_values))
    final_P = np.zeros(len(A_values))
    extinct = np.zeros(len(A_values), dtype=bool)

    for i, A_val in enumerate(A_values):
        config = _make_config(
            N_0=N_0, P_0=P_0, A=A_val, dt=dt, n_steps=n_steps,
        )
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()
            N, P = sim.observe()
            if N < 0.1:
                extinct[i] = True
                break

        final_N[i] = sim.observe()[0]
        final_P[i] = sim.observe()[1]

    return {
        "A_values": A_values,
        "final_N": final_N,
        "final_P": final_P,
        "extinct": extinct,
    }


def generate_predator_impact_data(
    P0_values: np.ndarray | None = None,
    N_0: float = 50.0,
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep initial predator density and find critical P for prey extinction.

    Returns:
        Dict with P0_values, final populations, and critical P estimate.
    """
    if P0_values is None:
        P0_values = np.linspace(0.0, 50.0, 25)

    final_N = np.zeros(len(P0_values))
    final_P = np.zeros(len(P0_values))
    extinct = np.zeros(len(P0_values), dtype=bool)

    for i, P0 in enumerate(P0_values):
        config = _make_config(N_0=N_0, P_0=P0, dt=dt, n_steps=n_steps)
        sim = AlleePredatorPreySimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()
            N, P = sim.observe()
            if N < 0.1:
                extinct[i] = True
                break

        final_N[i] = sim.observe()[0]
        final_P[i] = sim.observe()[1]

    # Estimate critical P: first P0 where extinction occurs
    critical_P = float("nan")
    if np.any(extinct):
        idx = np.argmax(extinct)
        if idx > 0:
            critical_P = 0.5 * (P0_values[idx - 1] + P0_values[idx])
        else:
            critical_P = P0_values[0]

    return {
        "P0_values": P0_values,
        "final_N": final_N,
        "final_P": final_P,
        "extinct": extinct,
        "critical_P": critical_P,
    }


def generate_ode_data(
    N_0: float = 50.0,
    P_0: float = 5.0,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate a single trajectory for SINDy ODE recovery.

    Returns:
        Dict with states array, time, and parameters.
    """
    config = _make_config(N_0=N_0, P_0=P_0, dt=dt, n_steps=n_steps)
    sim = AlleePredatorPreySimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "states": states,
        "time": np.arange(n_steps + 1) * dt,
        "dt": dt,
        "N": states[:, 0],
        "P": states[:, 1],
    }


def run_allee_predator_prey_rediscovery(
    output_dir: str | Path = "output/rediscovery/allee_predator_prey",
    n_iterations: int = 40,
) -> dict:
    """Run the full Allee predator-prey rediscovery pipeline.

    1. Demonstrate bistability: above-A vs below-A initial conditions
    2. Sweep Allee threshold A, measure extinction boundary
    3. Sweep predator density, find critical P for prey extinction
    4. Compute equilibria and verify dynamics

    Args:
        output_dir: Directory for output files.
        n_iterations: Number of PySR iterations (for future symbolic regression).

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "allee_predator_prey",
        "targets": {
            "allee_effect": "Prey declines when N < A",
            "bistability": "Two stable states: extinction vs coexistence",
            "critical_P": "Above this predator density, prey goes extinct",
            "prey_ode": "dN/dt = r*N*(N/A-1)*(1-N/K) - a*N*P/(1+h*a*N)",
            "pred_ode": "dP/dt = e*a*N*P/(1+h*a*N) - m*P",
        },
    }

    # --- Part 1: Bistability demonstration ---
    logger.info("Part 1: Bistability demonstration...")
    bistab = generate_bistability_data(
        N_above=50.0, N_below=5.0, P_0=5.0,
        n_steps=50000, dt=0.01,
    )
    results["bistability"] = {
        "above_N0": bistab["above_N_0"],
        "above_final_N": bistab["above_final_N"],
        "above_final_P": bistab["above_final_P"],
        "below_N0": bistab["below_N_0"],
        "below_final_N": bistab["below_final_N"],
        "below_final_P": bistab["below_final_P"],
        "is_bistable": (
            bistab["above_final_N"] > 1.0 and bistab["below_final_N"] < 1.0
        ),
    }
    logger.info(
        f"  Above A: N0={bistab['above_N_0']}, "
        f"final N={bistab['above_final_N']:.2f}"
    )
    logger.info(
        f"  Below A: N0={bistab['below_N_0']}, "
        f"final N={bistab['below_final_N']:.2f}"
    )

    # --- Part 2: Allee threshold sweep ---
    logger.info("Part 2: Allee threshold sweep...")
    allee_sweep = generate_allee_sweep_data(
        A_values=np.linspace(5.0, 60.0, 20),
        N_0=50.0, P_0=5.0,
        n_steps=50000, dt=0.01,
    )
    n_extinct = int(np.sum(allee_sweep["extinct"]))
    results["allee_sweep"] = {
        "n_A_values": len(allee_sweep["A_values"]),
        "n_extinct": n_extinct,
        "n_survive": len(allee_sweep["A_values"]) - n_extinct,
        "A_range": [float(allee_sweep["A_values"][0]),
                     float(allee_sweep["A_values"][-1])],
    }
    logger.info(
        f"  A sweep: {n_extinct} extinct, "
        f"{len(allee_sweep['A_values']) - n_extinct} survive"
    )

    # --- Part 3: Predator impact sweep ---
    logger.info("Part 3: Predator density sweep...")
    pred_sweep = generate_predator_impact_data(
        P0_values=np.linspace(0.0, 50.0, 25),
        N_0=50.0,
        n_steps=50000, dt=0.01,
    )
    results["predator_impact"] = {
        "n_P_values": len(pred_sweep["P0_values"]),
        "critical_P": float(pred_sweep["critical_P"]),
        "n_extinct": int(np.sum(pred_sweep["extinct"])),
    }
    logger.info(f"  Critical P estimate: {pred_sweep['critical_P']:.2f}")

    # --- Part 4: Equilibria ---
    logger.info("Part 4: Computing equilibria...")
    config = _make_config()
    sim = AlleePredatorPreySimulation(config)
    equilibria = sim.find_equilibria()
    results["equilibria"] = equilibria
    for eq in equilibria:
        logger.info(f"  {eq['type']}: N={eq['N']:.4f}, P={eq['P']:.4f}")

    # --- Part 5: Allee growth function verification ---
    logger.info("Part 5: Allee growth function verification...")
    N_test = np.linspace(0, 120, 200)
    growth_test = sim.allee_growth(N_test)
    # Verify zeros at N=0, N=A, N=K
    zero_at_0 = abs(sim.allee_growth(0.0)) < 1e-10
    zero_at_A = abs(sim.allee_growth(sim.A)) < 1e-10
    zero_at_K = abs(sim.allee_growth(sim.K)) < 1e-10
    # Verify sign: negative between 0 and A, positive between A and K
    N_mid_low = 0.5 * sim.A  # Between 0 and A
    N_mid_high = 0.5 * (sim.A + sim.K)  # Between A and K
    neg_below_A = sim.allee_growth(N_mid_low) < 0
    pos_above_A = sim.allee_growth(N_mid_high) > 0

    results["allee_growth"] = {
        "zero_at_0": bool(zero_at_0),
        "zero_at_A": bool(zero_at_A),
        "zero_at_K": bool(zero_at_K),
        "negative_below_A": bool(neg_below_A),
        "positive_above_A": bool(pos_above_A),
        "max_growth_rate": float(np.max(growth_test)),
    }
    logger.info(f"  Zeros at 0,A,K: {zero_at_0}, {zero_at_A}, {zero_at_K}")
    logger.info(f"  Negative below A: {neg_below_A}, Positive above A: {pos_above_A}")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
