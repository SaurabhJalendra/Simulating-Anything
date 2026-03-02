"""Stommel box model rediscovery.

Targets:
- ODE recovery via SINDy: dT/dt = eta1 - T*(1+|T-S|), dS/dt = eta2*eta3 - S*(eta3+|T-S|)
- Bistability: two stable equilibria (thermal vs haline driven)
- Saddle-node bifurcation and hysteresis as eta1 varies
- Flow rate characterization: q = |T - S|
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.stommel import StommelSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    eta1: float = 3.0,
    eta2: float = 1.0,
    eta3: float = 0.3,
    n_steps: int = 10000,
    dt: float = 0.01,
    T_0: float = 1.0,
    S_0: float = 0.5,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.STOMMEL,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "eta1": eta1, "eta2": eta2, "eta3": eta3,
            "T_0": T_0, "S_0": S_0,
        },
    )
    sim = StommelSimulation(config)
    sim.reset(seed=42)

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    times = np.arange(n_steps + 1) * dt

    return {
        "time": times,
        "states": states,
        "T": states[:, 0],
        "S": states[:, 1],
        "eta1": eta1,
        "eta2": eta2,
        "eta3": eta3,
        "dt": dt,
    }


def generate_hysteresis_data(
    eta2: float = 1.0,
    eta3: float = 0.3,
    n_eta1: int = 40,
    eta1_range: tuple[float, float] = (0.5, 5.0),
    dt: float = 0.01,
    n_steps: int = 10000,
) -> dict[str, np.ndarray]:
    """Sweep eta1 from two different ICs to capture hysteresis."""
    config = SimulationConfig(
        domain=Domain.STOMMEL,
        dt=dt,
        n_steps=1000,
        parameters={"eta1": 3.0, "eta2": eta2, "eta3": eta3},
    )
    sim = StommelSimulation(config)
    result = sim.eta1_sweep(
        eta1_range=eta1_range,
        n_eta=n_eta1,
        n_steps=n_steps,
    )

    return result


def generate_equilibria_data(
    eta1_values: np.ndarray | None = None,
    eta2: float = 1.0,
    eta3: float = 0.3,
) -> dict[str, list]:
    """Find equilibria at multiple eta1 values to map bifurcation diagram."""
    if eta1_values is None:
        eta1_values = np.linspace(0.5, 5.0, 30)

    all_equilibria = []
    eta1_list = []
    T_eq_list = []
    S_eq_list = []
    types_list = []

    for eta1 in eta1_values:
        config = SimulationConfig(
            domain=Domain.STOMMEL,
            dt=0.01,
            n_steps=1000,
            parameters={"eta1": float(eta1), "eta2": eta2, "eta3": eta3},
        )
        sim = StommelSimulation(config)
        sim.reset(seed=42)
        eqs = sim.find_equilibria(n_guesses=80)

        for T_eq, S_eq in eqs:
            eta1_list.append(float(eta1))
            T_eq_list.append(T_eq)
            S_eq_list.append(S_eq)
            types_list.append(sim.classify_equilibrium(T_eq, S_eq))
            all_equilibria.append((float(eta1), T_eq, S_eq))

    return {
        "eta1": eta1_list,
        "T_eq": T_eq_list,
        "S_eq": S_eq_list,
        "type": types_list,
        "n_equilibria_by_eta1": {
            float(e): sum(1 for x in eta1_list if abs(x - e) < 1e-8)
            for e in eta1_values
        },
    }


def run_stommel_rediscovery(
    output_dir: str | Path = "output/rediscovery/stommel",
    n_iterations: int = 40,
) -> dict:
    """Run Stommel box model rediscovery pipeline.

    1. Find equilibria and classify (thermal vs haline)
    2. Hysteresis loop: sweep eta1 from two ICs
    3. Detect saddle-node bifurcation points
    4. Flow rate characterization
    5. SINDy ODE recovery
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "stommel",
        "targets": {
            "ode_T": "dT/dt = eta1 - T*(1 + |T-S|)",
            "ode_S": "dS/dt = eta2*eta3 - S*(eta3 + |T-S|)",
            "bistability": "thermal-driven (T>S) vs salinity-driven (S>T)",
            "hysteresis": "saddle-node bifurcation as eta1 varies",
            "flow_rate": "q = |T - S|",
        },
    }

    # --- Part 1: Equilibria at default parameters ---
    logger.info("Part 1: Finding equilibria at eta1=3.0, eta2=1.0, eta3=0.3...")
    config = SimulationConfig(
        domain=Domain.STOMMEL,
        dt=0.01,
        n_steps=1000,
        parameters={"eta1": 3.0, "eta2": 1.0, "eta3": 0.3},
    )
    sim = StommelSimulation(config)
    sim.reset(seed=42)
    equilibria = sim.find_equilibria(n_guesses=100)

    results["equilibria"] = {
        "n_found": len(equilibria),
        "points": [
            {
                "T": T, "S": S,
                "type": sim.classify_equilibrium(T, S),
                "flow_rate": abs(T - S),
            }
            for T, S in equilibria
        ],
    }
    for T, S in equilibria:
        etype = sim.classify_equilibrium(T, S)
        logger.info(f"  Equilibrium: T={T:.4f}, S={S:.4f}, type={etype}, q={abs(T-S):.4f}")

    # --- Part 2: Hysteresis loop ---
    logger.info("Part 2: Hysteresis sweep (eta1 from 0.5 to 5.0)...")
    hyst_data = generate_hysteresis_data(
        n_eta1=40,
        eta1_range=(0.5, 5.0),
        dt=0.01,
        n_steps=15000,
    )

    # Detect hysteresis: where thermal and haline ICs converge to different states
    T_diff = np.abs(hyst_data["T_thermal"] - hyst_data["T_haline"])
    hysteresis_mask = T_diff > 0.1
    n_hysteresis = int(np.sum(hysteresis_mask))

    results["hysteresis"] = {
        "n_eta1_points": len(hyst_data["eta1"]),
        "n_bistable_points": n_hysteresis,
        "bistable_fraction": float(n_hysteresis / len(hyst_data["eta1"])),
    }

    if n_hysteresis > 0:
        bistable_eta1 = hyst_data["eta1"][hysteresis_mask]
        results["hysteresis"]["eta1_bistable_range"] = [
            float(bistable_eta1.min()),
            float(bistable_eta1.max()),
        ]
        logger.info(
            f"  Bistable region: eta1 in [{bistable_eta1.min():.3f}, "
            f"{bistable_eta1.max():.3f}]"
        )

    # Detect saddle-node bifurcation: where branches merge/disappear
    # Forward: thermal branch disappears at lower saddle-node
    # Reverse: haline branch disappears at upper saddle-node
    q_thermal = hyst_data["q_thermal"]
    q_haline = hyst_data["q_haline"]

    results["flow_rates"] = {
        "mean_q_thermal": float(np.mean(q_thermal)),
        "mean_q_haline": float(np.mean(q_haline)),
        "max_q_thermal": float(np.max(q_thermal)),
        "max_q_haline": float(np.max(q_haline)),
    }

    # --- Part 3: Equilibria bifurcation diagram ---
    logger.info("Part 3: Equilibria bifurcation diagram...")
    eq_data = generate_equilibria_data(
        eta1_values=np.linspace(0.5, 5.0, 30),
        eta2=1.0,
        eta3=0.3,
    )

    # Count how many equilibria exist at each eta1
    n_eq_counts = eq_data["n_equilibria_by_eta1"]
    multi_eq = {k: v for k, v in n_eq_counts.items() if v >= 2}

    results["bifurcation_diagram"] = {
        "n_eta1_points": 30,
        "total_equilibria_found": len(eq_data["eta1"]),
        "n_eta1_with_multiple_eq": len(multi_eq),
        "n_thermal_equilibria": sum(1 for t in eq_data["type"] if t == "thermal"),
        "n_haline_equilibria": sum(1 for t in eq_data["type"] if t == "haline"),
    }
    logger.info(
        f"  {len(multi_eq)} eta1 values with multiple equilibria "
        f"(out of 30)"
    )

    # --- Part 4: SINDy ODE recovery ---
    logger.info("Part 4: SINDy ODE recovery...")

    # Generate data on limit cycle / near equilibrium with rich dynamics
    data = generate_ode_data(eta1=3.0, eta2=1.0, eta3=0.3, n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=data["dt"],
            feature_names=["T", "S"],
            threshold=0.05,
            poly_degree=3,
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

    # --- Part 5: PySR flow rate characterization ---
    logger.info("Part 5: PySR flow rate law...")

    # Generate flow rate data from sweeps
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use multiple trajectories at different parameters
        T_vals = []
        S_vals = []
        q_vals = []

        for eta1 in np.linspace(1.0, 5.0, 10):
            odata = generate_ode_data(
                eta1=eta1, n_steps=2000, dt=0.01,
                T_0=2.0, S_0=0.5,
            )
            # Sample every 50th point after transient
            T_traj = odata["T"][500::50]
            S_traj = odata["S"][500::50]
            T_vals.extend(T_traj.tolist())
            S_vals.extend(S_traj.tolist())
            q_vals.extend(np.abs(T_traj - S_traj).tolist())

        T_arr = np.array(T_vals)
        S_arr = np.array(S_vals)
        q_arr = np.array(q_vals)

        X = np.column_stack([T_arr, S_arr])
        logger.info(f"  Running PySR on {len(q_arr)} flow rate samples...")

        discoveries = run_symbolic_regression(
            X, q_arr,
            variable_names=["T_", "S_"],
            n_iterations=n_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["abs"],
            max_complexity=8,
            populations=20,
            population_size=40,
        )
        results["flow_rate_pysr"] = {
            "n_discoveries": len(discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in discoveries[:5]
            ],
        }
        if discoveries:
            best = discoveries[0]
            results["flow_rate_pysr"]["best"] = best.expression
            results["flow_rate_pysr"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  Best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["flow_rate_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "hysteresis_data.npz",
        eta1=hyst_data["eta1"],
        T_thermal=hyst_data["T_thermal"],
        S_thermal=hyst_data["S_thermal"],
        T_haline=hyst_data["T_haline"],
        S_haline=hyst_data["S_haline"],
    )

    return results
