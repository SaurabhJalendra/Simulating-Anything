"""Double Allee predator-prey rediscovery.

Targets:
- Four-state bistability: coexistence, prey-only, predator-only, extinction
- Prey Allee threshold: prey declines when N < A_N
- Predator Allee threshold: predator declines when P < A_P (without prey)
- Basin of attraction mapping in (N, P) space
- Equilibria verification
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.allee_two import AlleeTwoSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_DOMAIN = Domain.ALLEE_TWO


def _make_config(
    N_0: float = 50.0,
    P_0: float = 10.0,
    r_N: float = 1.0,
    K_N: float = 100.0,
    A_N: float = 5.0,
    r_P: float = 0.5,
    K_P: float = 50.0,
    A_P: float = 3.0,
    a: float = 0.01,
    h: float = 0.1,
    e: float = 0.5,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Build a SimulationConfig for the double Allee model."""
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r_N": r_N, "K_N": K_N, "A_N": A_N,
            "r_P": r_P, "K_P": K_P, "A_P": A_P,
            "a": a, "h": h, "e": e,
            "N_0": N_0, "P_0": P_0,
        },
    )


def generate_basin_data(
    N_range: np.ndarray | None = None,
    P_range: np.ndarray | None = None,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep initial conditions to map basins of attraction.

    Classifies each (N_0, P_0) outcome as extinction, prey-only,
    predator-only, or coexistence.

    Returns:
        Dict with grid data, final populations, and outcome codes.
    """
    if N_range is None:
        N_range = np.linspace(1.0, 80.0, 12)
    if P_range is None:
        P_range = np.linspace(0.5, 30.0, 12)

    nN, nP = len(N_range), len(P_range)
    final_N = np.zeros((nN, nP))
    final_P = np.zeros((nN, nP))
    outcome = np.zeros((nN, nP), dtype=int)

    for i, N0 in enumerate(N_range):
        for j, P0 in enumerate(P_range):
            config = _make_config(N_0=N0, P_0=P0, dt=dt, n_steps=n_steps)
            sim = AlleeTwoSimulation(config)
            sim.reset()
            for _ in range(n_steps):
                sim.step()
            N_f, P_f = sim.observe()
            final_N[i, j] = N_f
            final_P[i, j] = P_f

            if N_f < 0.1 and P_f < 0.1:
                outcome[i, j] = 0
            elif N_f >= 0.1 and P_f < 0.1:
                outcome[i, j] = 1
            elif N_f < 0.1 and P_f >= 0.1:
                outcome[i, j] = 2
            else:
                outcome[i, j] = 3

    return {
        "N_range": N_range,
        "P_range": P_range,
        "final_N": final_N,
        "final_P": final_P,
        "outcome": outcome,
    }


def generate_prey_allee_sweep_data(
    A_N_values: np.ndarray | None = None,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep prey Allee threshold and measure outcomes."""
    if A_N_values is None:
        A_N_values = np.linspace(2.0, 50.0, 20)

    final_N = np.zeros(len(A_N_values))
    final_P = np.zeros(len(A_N_values))
    prey_survives = np.zeros(len(A_N_values), dtype=bool)

    for i, A_N_val in enumerate(A_N_values):
        config = _make_config(A_N=A_N_val, dt=dt, n_steps=n_steps)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(n_steps):
            sim.step()
        final_N[i] = sim.observe()[0]
        final_P[i] = sim.observe()[1]
        prey_survives[i] = final_N[i] > 0.1

    return {
        "A_N_values": A_N_values,
        "final_N": final_N,
        "final_P": final_P,
        "prey_survives": prey_survives,
    }


def generate_predator_allee_sweep_data(
    A_P_values: np.ndarray | None = None,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep predator Allee threshold and measure outcomes."""
    if A_P_values is None:
        A_P_values = np.linspace(1.0, 30.0, 20)

    final_N = np.zeros(len(A_P_values))
    final_P = np.zeros(len(A_P_values))
    pred_survives = np.zeros(len(A_P_values), dtype=bool)

    for i, A_P_val in enumerate(A_P_values):
        config = _make_config(A_P=A_P_val, dt=dt, n_steps=n_steps)
        sim = AlleeTwoSimulation(config)
        sim.reset()
        for _ in range(n_steps):
            sim.step()
        final_N[i] = sim.observe()[0]
        final_P[i] = sim.observe()[1]
        pred_survives[i] = final_P[i] > 0.1

    return {
        "A_P_values": A_P_values,
        "final_N": final_N,
        "final_P": final_P,
        "pred_survives": pred_survives,
    }


def generate_ode_data(
    N_0: float = 50.0,
    P_0: float = 10.0,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate a single trajectory for SINDy ODE recovery."""
    config = _make_config(N_0=N_0, P_0=P_0, dt=dt, n_steps=n_steps)
    sim = AlleeTwoSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "states": states_arr,
        "time": np.arange(n_steps + 1) * dt,
        "dt": dt,
        "N": states_arr[:, 0],
        "P": states_arr[:, 1],
    }


def generate_bistability_data(
    N_above: float = 50.0,
    N_below: float = 3.0,
    P_above: float = 10.0,
    P_below: float = 1.0,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict:
    """Demonstrate four-state bistability from different IC regimes.

    Regimes:
    - Both above: N > A_N, P > A_P -> coexistence
    - Prey below: N < A_N, P > A_P -> predator-only
    - Pred below: N > A_N, P < A_P -> prey-only
    - Both below: N < A_N, P < A_P -> mutual extinction

    Returns:
        Dict with trajectories and final states for all four regimes.
    """
    regimes = {
        "both_above": (N_above, P_above),
        "prey_below": (N_below, P_above),
        "pred_below": (N_above, P_below),
        "both_below": (N_below, P_below),
    }

    results: dict = {}
    for label, (N_init, P_init) in regimes.items():
        config = _make_config(N_0=N_init, P_0=P_init, dt=dt, n_steps=n_steps)
        sim = AlleeTwoSimulation(config)
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
        results[f"{label}_P_0"] = P_init

        logger.info(
            f"  IC N={N_init}, P={P_init}: "
            f"final N={traj[-1, 0]:.2f}, P={traj[-1, 1]:.2f}"
        )

    return results


def run_allee_two_rediscovery(
    output_dir: str | Path = "output/rediscovery/allee_two",
    n_iterations: int = 50,
) -> dict:
    """Run the full double Allee predator-prey rediscovery pipeline.

    1. Demonstrate four-state bistability
    2. Sweep prey Allee threshold A_N
    3. Sweep predator Allee threshold A_P
    4. Map basins of attraction
    5. Compute equilibria
    6. Growth function verification

    Args:
        output_dir: Directory for output files.
        n_iterations: Number of PySR iterations (for future symbolic regression).

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "allee_two",
        "targets": {
            "prey_allee": "Prey declines when N < A_N",
            "pred_allee": "Predator declines when P < A_P (without prey)",
            "bistability": "Four stable states from both Allee thresholds",
            "prey_ode": "dN/dt = r_N*N*(N/A_N-1)*(1-N/K_N) - a*N*P/(1+h*N)",
            "pred_ode": "dP/dt = r_P*P*(P/A_P-1)*(1-P/K_P) + e*a*N*P/(1+h*N)",
        },
    }

    # --- Part 1: Four-state bistability ---
    logger.info("Part 1: Four-state bistability...")
    bistab = generate_bistability_data(
        N_above=50.0, N_below=3.0,
        P_above=10.0, P_below=1.0,
        n_steps=30000, dt=0.01,
    )
    results["bistability"] = {
        "both_above_final": {
            "N": bistab["both_above_final_N"],
            "P": bistab["both_above_final_P"],
        },
        "prey_below_final": {
            "N": bistab["prey_below_final_N"],
            "P": bistab["prey_below_final_P"],
        },
        "pred_below_final": {
            "N": bistab["pred_below_final_N"],
            "P": bistab["pred_below_final_P"],
        },
        "both_below_final": {
            "N": bistab["both_below_final_N"],
            "P": bistab["both_below_final_P"],
        },
    }

    # --- Part 2: Prey Allee threshold sweep ---
    logger.info("Part 2: Prey Allee threshold sweep...")
    prey_sweep = generate_prey_allee_sweep_data(
        A_N_values=np.linspace(2.0, 50.0, 20),
        n_steps=30000, dt=0.01,
    )
    n_survive = int(np.sum(prey_sweep["prey_survives"]))
    results["prey_allee_sweep"] = {
        "n_A_N_values": len(prey_sweep["A_N_values"]),
        "n_prey_survive": n_survive,
        "n_prey_extinct": len(prey_sweep["A_N_values"]) - n_survive,
    }
    logger.info(f"  Prey sweep: {n_survive} survive")

    # --- Part 3: Predator Allee threshold sweep ---
    logger.info("Part 3: Predator Allee threshold sweep...")
    pred_sweep = generate_predator_allee_sweep_data(
        A_P_values=np.linspace(1.0, 30.0, 20),
        n_steps=30000, dt=0.01,
    )
    n_pred_survive = int(np.sum(pred_sweep["pred_survives"]))
    results["pred_allee_sweep"] = {
        "n_A_P_values": len(pred_sweep["A_P_values"]),
        "n_pred_survive": n_pred_survive,
        "n_pred_extinct": len(pred_sweep["A_P_values"]) - n_pred_survive,
    }
    logger.info(f"  Pred sweep: {n_pred_survive} survive")

    # --- Part 4: Basin of attraction ---
    logger.info("Part 4: Basin of attraction mapping...")
    basins = generate_basin_data(
        N_range=np.linspace(1.0, 80.0, 10),
        P_range=np.linspace(0.5, 30.0, 10),
        n_steps=30000, dt=0.01,
    )
    outcome_counts = {
        "extinction": int(np.sum(basins["outcome"] == 0)),
        "prey_only": int(np.sum(basins["outcome"] == 1)),
        "predator_only": int(np.sum(basins["outcome"] == 2)),
        "coexistence": int(np.sum(basins["outcome"] == 3)),
    }
    results["basins"] = {
        "grid_size": [len(basins["N_range"]), len(basins["P_range"])],
        "outcome_counts": outcome_counts,
    }
    logger.info(f"  Basin outcomes: {outcome_counts}")

    # --- Part 5: Equilibria ---
    logger.info("Part 5: Computing equilibria...")
    config = _make_config()
    sim = AlleeTwoSimulation(config)
    equilibria = sim.find_equilibria()
    results["equilibria"] = equilibria
    for eq in equilibria:
        logger.info(f"  {eq['type']}: N={eq['N']:.4f}, P={eq['P']:.4f}")

    # --- Part 6: Growth function verification ---
    logger.info("Part 6: Growth function verification...")
    prey_zero_0 = abs(sim.prey_allee_growth(0.0)) < 1e-10
    prey_zero_AN = abs(sim.prey_allee_growth(sim.A_N)) < 1e-10
    prey_zero_KN = abs(sim.prey_allee_growth(sim.K_N)) < 1e-10
    pred_zero_0 = abs(sim.predator_allee_growth(0.0)) < 1e-10
    pred_zero_AP = abs(sim.predator_allee_growth(sim.A_P)) < 1e-10
    pred_zero_KP = abs(sim.predator_allee_growth(sim.K_P)) < 1e-10

    results["growth_verification"] = {
        "prey_zero_at_0": bool(prey_zero_0),
        "prey_zero_at_AN": bool(prey_zero_AN),
        "prey_zero_at_KN": bool(prey_zero_KN),
        "pred_zero_at_0": bool(pred_zero_0),
        "pred_zero_at_AP": bool(pred_zero_AP),
        "pred_zero_at_KP": bool(pred_zero_KP),
    }

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
