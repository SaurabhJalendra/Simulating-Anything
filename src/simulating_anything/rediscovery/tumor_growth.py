"""Tumor-immune interaction model rediscovery.

Targets:
- Elimination vs escape threshold in immune killing rate a
- Dormancy equilibrium conditions (stable coexistence)
- Saddle-node bifurcation in immune parameters
- ODE recovery via SINDy
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.tumor_growth import TumorGrowthSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    n_steps: int = 10000,
    dt: float = 0.1,
) -> dict[str, np.ndarray | float]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses default parameters that produce dormancy (coexistence) behavior
    for interesting dynamics.
    """
    config = SimulationConfig(
        domain=Domain.TUMOR_GROWTH,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": 0.5,
            "K": 1e6,
            "a": 1e-7,
            "s": 1e4,
            "p": 0.1,
            "g": 1e5,
            "d": 0.03,
            "a_I": 1e-7,
            "T_0": 1e4,
            "I_0": 1e5,
        },
    )
    sim = TumorGrowthSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "T": states_arr[:, 0],
        "I": states_arr[:, 1],
        "dt": dt,
    }


def generate_immune_sweep_data(
    n_a: int = 30,
    n_steps: int = 5000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep immune killing rate a to find elimination/escape threshold.

    Returns final tumor size and classification for each value of a.
    """
    a_values = np.logspace(-9, -5, n_a)
    final_T = []
    final_I = []
    outcomes = []

    for i, a_val in enumerate(a_values):
        config = SimulationConfig(
            domain=Domain.TUMOR_GROWTH,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 0.5,
                "K": 1e6,
                "a": a_val,
                "s": 1e4,
                "p": 0.1,
                "g": 1e5,
                "d": 0.03,
                "a_I": 1e-7,
                "T_0": 1e4,
                "I_0": 1e5,
            },
        )
        sim = TumorGrowthSimulation(config)
        outcome = sim.classify_outcome(n_steps=n_steps)
        final_state = sim.observe()

        final_T.append(final_state[0])
        final_I.append(final_state[1])
        outcomes.append(outcome)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  a={a_val:.2e}: T_final={final_state[0]:.1f}, "
                f"outcome={outcome}"
            )

    return {
        "a": a_values,
        "final_T": np.array(final_T),
        "final_I": np.array(final_I),
        "outcomes": outcomes,
    }


def generate_dormancy_data(
    n_samples: int = 30,
    n_steps: int = 8000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep parameters to find dormancy (coexistence) conditions.

    Varies immune source rate s and killing rate a to map the dormancy region.
    """
    rng = np.random.default_rng(42)

    all_a = []
    all_s = []
    all_final_T = []
    all_final_I = []
    all_outcome = []

    for i in range(n_samples):
        a_val = 10 ** rng.uniform(-8, -6)
        s_val = 10 ** rng.uniform(3, 5)

        config = SimulationConfig(
            domain=Domain.TUMOR_GROWTH,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 0.5,
                "K": 1e6,
                "a": a_val,
                "s": s_val,
                "p": 0.1,
                "g": 1e5,
                "d": 0.03,
                "a_I": 1e-7,
                "T_0": 1e4,
                "I_0": 1e5,
            },
        )
        sim = TumorGrowthSimulation(config)
        outcome = sim.classify_outcome(n_steps=n_steps)
        final_state = sim.observe()

        all_a.append(a_val)
        all_s.append(s_val)
        all_final_T.append(final_state[0])
        all_final_I.append(final_state[1])
        all_outcome.append(outcome)

        if (i + 1) % 10 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} dormancy sweep points")

    return {
        "a": np.array(all_a),
        "s": np.array(all_s),
        "final_T": np.array(all_final_T),
        "final_I": np.array(all_final_I),
        "outcomes": all_outcome,
    }


def generate_bifurcation_data(
    n_points: int = 40,
    n_steps: int = 8000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Sweep immune source rate s to trace a saddle-node bifurcation.

    At low s, tumor escapes. As s increases, a saddle-node bifurcation
    creates dormancy (coexistence) and eventually elimination.
    """
    s_values = np.logspace(2, 6, n_points)
    final_T = []
    final_I = []
    outcomes = []

    for i, s_val in enumerate(s_values):
        config = SimulationConfig(
            domain=Domain.TUMOR_GROWTH,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 0.5,
                "K": 1e6,
                "a": 1e-7,
                "s": s_val,
                "p": 0.1,
                "g": 1e5,
                "d": 0.03,
                "a_I": 1e-7,
                "T_0": 1e4,
                "I_0": 1e5,
            },
        )
        sim = TumorGrowthSimulation(config)
        outcome = sim.classify_outcome(n_steps=n_steps)
        final_state = sim.observe()

        final_T.append(final_state[0])
        final_I.append(final_state[1])
        outcomes.append(outcome)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  s={s_val:.1f}: T_final={final_state[0]:.1f}, "
                f"outcome={outcome}"
            )

    return {
        "s": s_values,
        "final_T": np.array(final_T),
        "final_I": np.array(final_I),
        "outcomes": outcomes,
    }


def run_tumor_growth_rediscovery(
    output_dir: str | Path = "output/rediscovery/tumor_growth",
    n_iterations: int = 50,
) -> dict:
    """Run the full tumor-immune interaction rediscovery pipeline.

    1. Sweep immune killing rate a to find elimination vs escape threshold
    2. Map dormancy (coexistence) conditions in (a, s) space
    3. Trace saddle-node bifurcation in immune source rate s
    4. SINDy ODE recovery
    5. PySR: predict final tumor size from immune parameters
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "tumor_growth",
        "targets": {
            "ode_T": "dT/dt = r*T*(1 - T/K) - a*T*I",
            "ode_I": "dI/dt = s + p*T*I/(g+T) - d*I - a_I*T*I",
            "elimination_threshold": "immune killing rate a above which tumor is eliminated",
            "dormancy": "stable coexistence equilibrium",
            "bifurcation": "saddle-node in immune source rate s",
        },
    }

    # --- Part 1: Immune killing rate sweep ---
    logger.info("Part 1: Immune killing rate sweep...")
    sweep_data = generate_immune_sweep_data(n_a=30, n_steps=5000, dt=0.1)

    n_elimination = sum(1 for o in sweep_data["outcomes"] if o == "elimination")
    n_dormancy = sum(1 for o in sweep_data["outcomes"] if o == "dormancy")
    n_escape = sum(1 for o in sweep_data["outcomes"] if o == "escape")

    # Find threshold: first a value where outcome switches from escape to non-escape
    threshold_a = None
    for i in range(1, len(sweep_data["outcomes"])):
        if sweep_data["outcomes"][i - 1] == "escape" and sweep_data["outcomes"][i] != "escape":
            threshold_a = float(sweep_data["a"][i])
            break

    results["immune_sweep"] = {
        "n_elimination": n_elimination,
        "n_dormancy": n_dormancy,
        "n_escape": n_escape,
        "threshold_a": threshold_a,
        "a_range": [float(sweep_data["a"].min()), float(sweep_data["a"].max())],
    }
    logger.info(
        f"  Outcomes: {n_elimination} elimination, "
        f"{n_dormancy} dormancy, {n_escape} escape"
    )
    if threshold_a is not None:
        logger.info(f"  Escape/non-escape threshold: a ~ {threshold_a:.2e}")

    # --- Part 2: Dormancy conditions ---
    logger.info("Part 2: Dormancy region mapping...")
    dorm_data = generate_dormancy_data(n_samples=30, n_steps=8000, dt=0.1)

    n_dorm = sum(1 for o in dorm_data["outcomes"] if o == "dormancy")
    results["dormancy"] = {
        "n_dormancy_points": n_dorm,
        "n_total": len(dorm_data["outcomes"]),
        "fraction": float(n_dorm / len(dorm_data["outcomes"])) if dorm_data["outcomes"] else 0,
    }

    # Record dormancy equilibrium values
    dorm_mask = [o == "dormancy" for o in dorm_data["outcomes"]]
    if any(dorm_mask):
        dorm_T = dorm_data["final_T"][dorm_mask]
        dorm_I = dorm_data["final_I"][dorm_mask]
        results["dormancy"]["T_range"] = [float(dorm_T.min()), float(dorm_T.max())]
        results["dormancy"]["I_range"] = [float(dorm_I.min()), float(dorm_I.max())]
        logger.info(
            f"  Dormancy: T in [{dorm_T.min():.0f}, {dorm_T.max():.0f}], "
            f"I in [{dorm_I.min():.0f}, {dorm_I.max():.0f}]"
        )

    # --- Part 3: Saddle-node bifurcation in s ---
    logger.info("Part 3: Bifurcation in immune source rate s...")
    bif_data = generate_bifurcation_data(n_points=40, n_steps=8000, dt=0.1)

    n_elim = sum(1 for o in bif_data["outcomes"] if o == "elimination")
    n_dorm_bif = sum(1 for o in bif_data["outcomes"] if o == "dormancy")
    n_esc = sum(1 for o in bif_data["outcomes"] if o == "escape")

    # Find critical s values
    s_crit_escape = None
    s_crit_elim = None
    for i in range(1, len(bif_data["outcomes"])):
        if bif_data["outcomes"][i - 1] == "escape" and bif_data["outcomes"][i] != "escape":
            s_crit_escape = float(bif_data["s"][i])
        prev_not_elim = bif_data["outcomes"][i - 1] != "elimination"
        if bif_data["outcomes"][i] == "elimination" and prev_not_elim:
            s_crit_elim = float(bif_data["s"][i])

    results["bifurcation"] = {
        "n_elimination": n_elim,
        "n_dormancy": n_dorm_bif,
        "n_escape": n_esc,
        "s_critical_escape": s_crit_escape,
        "s_critical_elimination": s_crit_elim,
        "s_range": [float(bif_data["s"].min()), float(bif_data["s"].max())],
    }
    logger.info(
        f"  Bifurcation: {n_elim} elimination, "
        f"{n_dorm_bif} dormancy, {n_esc} escape"
    )

    # --- Part 4: SINDy ODE recovery ---
    logger.info("Part 4: SINDy ODE recovery...")
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        ode_data = generate_ode_data(n_steps=10000, dt=0.1)
        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["T", "I_"],
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
                for d in sindy_discoveries
            ],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression} (R2={d.evidence.fit_r_squared:.6f})")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 5: PySR symbolic regression ---
    logger.info("Part 5: PySR symbolic regression...")
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use immune sweep data: predict final T from a
        valid = np.isfinite(sweep_data["final_T"]) & (sweep_data["final_T"] > 0)
        if np.sum(valid) > 5:
            X = np.log10(sweep_data["a"][valid]).reshape(-1, 1)
            y = np.log10(sweep_data["final_T"][valid] + 1.0)

            logger.info("  Running PySR: log10(T_final) = f(log10(a))...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["log_a"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square", "sqrt", "exp"],
                max_complexity=15,
                populations=20,
                population_size=40,
            )
            results["pysr_tumor_size"] = {
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
                results["pysr_tumor_size"]["best"] = best.expression
                results["pysr_tumor_size"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_tumor_size"] = {"error": str(e)}

    # --- Part 6: Tumor-free equilibrium verification ---
    logger.info("Part 6: Tumor-free equilibrium verification...")
    config = SimulationConfig(
        domain=Domain.TUMOR_GROWTH,
        dt=0.1,
        n_steps=1000,
        parameters={
            "r": 0.5, "K": 1e6, "a": 1e-7, "s": 1e4,
            "p": 0.1, "g": 1e5, "d": 0.03, "a_I": 1e-7,
            "T_0": 1e4, "I_0": 1e5,
        },
    )
    sim = TumorGrowthSimulation(config)
    sim.reset()
    tfe = sim.tumor_free_equilibrium
    dy_at_tfe = sim._derivatives(np.array(tfe))
    results["tumor_free_eq"] = {
        "T_star": tfe[0],
        "I_star": tfe[1],
        "derivative_at_eq": [float(dy_at_tfe[0]), float(dy_at_tfe[1])],
        "derivative_norm": float(np.linalg.norm(dy_at_tfe)),
    }
    logger.info(
        f"  Tumor-free equilibrium: T*={tfe[0]:.1f}, I*={tfe[1]:.1f}, "
        f"|f(eq)|={np.linalg.norm(dy_at_tfe):.2e}"
    )

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
