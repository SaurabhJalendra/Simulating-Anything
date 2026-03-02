"""Predator-Two-Prey rediscovery.

Targets:
- ODE recovery via SINDy:
    dx1/dt = r1*x1*(1 - x1/K1) - a1*x1*y
    dx2/dt = r2*x2*(1 - x2/K2) - a2*x2*y
    dy/dt  = -d*y + b1*x1*y + b2*x2*y
- Interior coexistence equilibrium
- Apparent competition: prey exclusion via shared predator
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.predator_two_prey import (
    PredatorTwoPreySimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    r1: float = 1.0,
    r2: float = 0.8,
    K1: float = 10.0,
    K2: float = 8.0,
    a1: float = 0.5,
    a2: float = 0.4,
    b1: float = 0.2,
    b2: float = 0.15,
    d: float = 0.6,
    x1_0: float = 5.0,
    x2_0: float = 4.0,
    y_0: float = 2.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Build a SimulationConfig for the predator-two-prey model."""
    return SimulationConfig(
        domain=Domain.PREDATOR_TWO_PREY,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r1": r1, "r2": r2, "K1": K1, "K2": K2,
            "a1": a1, "a2": a2, "b1": b1, "b2": b2, "d": d,
            "x1_0": x1_0, "x2_0": x2_0, "y_0": y_0,
        },
    )


def generate_trajectory_data(
    n_steps: int = 10000,
    dt: float = 0.005,
    **kwargs: float,
) -> dict[str, np.ndarray | float]:
    """Generate a single long trajectory for SINDy ODE recovery.

    Returns dict with states array, time, and parameters.
    """
    config = _make_config(dt=dt, n_steps=n_steps, **kwargs)
    sim = PredatorTwoPreySimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "x1": states_arr[:, 0],
        "x2": states_arr[:, 1],
        "y": states_arr[:, 2],
        "dt": dt,
    }


def generate_coexistence_sweep_data(
    n_samples: int = 50,
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep parameters to find coexistence vs exclusion regions.

    Varies predation rates and predator death rate, tracking which
    species survive at steady state.

    Returns dict with parameter arrays and survival outcomes.
    """
    rng = np.random.default_rng(42)

    all_a1 = []
    all_a2 = []
    all_d = []
    all_n_surviving = []
    all_x1_final = []
    all_x2_final = []
    all_y_final = []

    for i in range(n_samples):
        a1 = rng.uniform(0.2, 0.8)
        a2 = rng.uniform(0.2, 0.8)
        d_val = rng.uniform(0.3, 1.0)

        config = _make_config(
            a1=a1, a2=a2, d=d_val, dt=dt, n_steps=n_steps,
        )
        sim = PredatorTwoPreySimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        state = sim.observe()
        all_a1.append(a1)
        all_a2.append(a2)
        all_d.append(d_val)
        all_n_surviving.append(sim.n_surviving())
        all_x1_final.append(state[0])
        all_x2_final.append(state[1])
        all_y_final.append(state[2])

        if (i + 1) % 10 == 0:
            logger.info(
                f"  Sample {i + 1}/{n_samples}: "
                f"a1={a1:.3f}, a2={a2:.3f}, d={d_val:.3f}, "
                f"surviving={all_n_surviving[-1]}"
            )

    return {
        "a1": np.array(all_a1),
        "a2": np.array(all_a2),
        "d": np.array(all_d),
        "n_surviving": np.array(all_n_surviving),
        "x1_final": np.array(all_x1_final),
        "x2_final": np.array(all_x2_final),
        "y_final": np.array(all_y_final),
    }


def generate_apparent_competition_data(
    n_points: int = 30,
    n_steps: int = 80000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep predator conversion rate b2 and track prey 2 exclusion.

    As b2 increases, the predator benefits more from prey 2, which
    increases predator density, which in turn suppresses prey 1
    (apparent competition). This can lead to prey 1 exclusion.

    Returns dict with b2 values and population outcomes.
    """
    b2_values = np.linspace(0.05, 0.5, n_points)
    x1_finals = []
    x2_finals = []
    y_finals = []

    for b2 in b2_values:
        config = _make_config(b2=b2, dt=dt, n_steps=n_steps)
        sim = PredatorTwoPreySimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        state = sim.observe()
        x1_finals.append(state[0])
        x2_finals.append(state[1])
        y_finals.append(state[2])

    return {
        "b2": b2_values,
        "x1_final": np.array(x1_finals),
        "x2_final": np.array(x2_finals),
        "y_final": np.array(y_finals),
    }


def run_predator_two_prey_rediscovery(
    output_dir: str | Path = "output/rediscovery/predator_two_prey",
    n_iterations: int = 40,
) -> dict:
    """Run the full predator-two-prey rediscovery pipeline.

    1. Generate trajectory for SINDy ODE recovery
    2. Run SINDy to recover the three coupled ODEs
    3. Sweep parameters for coexistence vs exclusion
    4. Analyze apparent competition dynamics

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "predator_two_prey",
        "targets": {
            "ode_x1": "dx1/dt = r1*x1*(1 - x1/K1) - a1*x1*y",
            "ode_x2": "dx2/dt = r2*x2*(1 - x2/K2) - a2*x2*y",
            "ode_y": "dy/dt = -d*y + b1*x1*y + b2*x2*y",
            "coexistence": "Interior fixed point with all species positive",
            "apparent_competition": "Prey exclusion via shared predator",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for predator-two-prey...")
    traj_data = generate_trajectory_data(n_steps=20000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            traj_data["states"],
            dt=0.005,
            feature_names=["x1", "x2", "y"],
            threshold=0.05,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
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
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Equilibrium analysis ---
    logger.info("Part 2: Fixed point and equilibrium analysis...")
    config_default = _make_config(dt=0.01, n_steps=100000)
    sim = PredatorTwoPreySimulation(config_default)
    sim.reset()

    fps = sim.fixed_points()
    results["fixed_points"] = {
        "count": len(fps),
        "points": [fp.tolist() for fp in fps],
    }

    # Run to steady state and compare
    for _ in range(100000):
        sim.step()
    final_state = sim.observe().copy()
    results["steady_state"] = {
        "x1": float(final_state[0]),
        "x2": float(final_state[1]),
        "y": float(final_state[2]),
        "n_surviving": sim.n_surviving(),
        "is_coexisting": bool(sim.is_coexisting),
    }
    logger.info(
        f"  Steady state: x1={final_state[0]:.3f}, "
        f"x2={final_state[1]:.3f}, y={final_state[2]:.3f}"
    )

    # --- Part 3: Coexistence sweep ---
    logger.info("Part 3: Coexistence parameter sweep...")
    sweep_data = generate_coexistence_sweep_data(
        n_samples=40, n_steps=50000, dt=0.01,
    )

    n_coexist = int(np.sum(sweep_data["n_surviving"] == 3))
    results["coexistence_sweep"] = {
        "n_samples": len(sweep_data["a1"]),
        "n_coexisting": n_coexist,
        "fraction_coexisting": float(n_coexist / len(sweep_data["a1"])),
        "mean_survivors": float(np.mean(sweep_data["n_surviving"])),
    }
    logger.info(
        f"  Coexistence: {n_coexist}/{len(sweep_data['a1'])} parameter sets"
    )

    # --- Part 4: Apparent competition sweep ---
    logger.info("Part 4: Apparent competition (b2 sweep)...")
    ac_data = generate_apparent_competition_data(
        n_points=25, n_steps=60000, dt=0.01,
    )

    results["apparent_competition"] = {
        "b2_range": [float(ac_data["b2"][0]), float(ac_data["b2"][-1])],
        "x1_at_low_b2": float(ac_data["x1_final"][0]),
        "x1_at_high_b2": float(ac_data["x1_final"][-1]),
        "x2_at_low_b2": float(ac_data["x2_final"][0]),
        "x2_at_high_b2": float(ac_data["x2_final"][-1]),
        "y_at_low_b2": float(ac_data["y_final"][0]),
        "y_at_high_b2": float(ac_data["y_final"][-1]),
    }
    logger.info(
        f"  Low b2: x1={ac_data['x1_final'][0]:.3f}, "
        f"x2={ac_data['x2_final'][0]:.3f}"
    )
    logger.info(
        f"  High b2: x1={ac_data['x1_final'][-1]:.3f}, "
        f"x2={ac_data['x2_final'][-1]:.3f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "trajectory_data.npz",
        states=traj_data["states"],
    )
    np.savez(
        output_path / "coexistence_sweep.npz",
        **{k: v for k, v in sweep_data.items()},
    )
    np.savez(
        output_path / "apparent_competition.npz",
        **{k: v for k, v in ac_data.items()},
    )

    return results
