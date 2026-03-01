"""Eco-epidemiological model rediscovery.

Targets:
- ODE recovery via SINDy: 3-variable eco-epidemic system
- Disease invasion threshold: R0 = beta*S*/(d + a2*P*)
- Predators as biological disease control: sweeping predator mortality
- Disease prevalence vs beta sweep
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.eco_epidemic import EcoEpidemicSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use ECO_EPIDEMIC domain enum
_DOMAIN = Domain.ECO_EPIDEMIC


def generate_ode_data(
    n_steps: int = 20000,
    dt: float = 0.01,
    **param_overrides: float,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Args:
        n_steps: Number of integration steps.
        dt: Timestep.
        **param_overrides: Override default parameters.

    Returns:
        Dict with time, states, S, I, P arrays.
    """
    params = {
        "r": 1.0, "K": 100.0, "beta": 0.01,
        "a1": 0.1, "a2": 0.3, "h1": 0.1, "h2": 0.1,
        "e1": 0.5, "e2": 0.3, "d_disease": 0.2, "m": 0.3,
        "S_0": 50.0, "I_0": 10.0, "P_0": 5.0,
    }
    params.update(param_overrides)

    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )
    sim = EcoEpidemicSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "S": states[:, 0],
        "I": states[:, 1],
        "P": states[:, 2],
    }


def generate_beta_sweep(
    n_beta: int = 25,
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep disease transmission rate beta.

    Args:
        n_beta: Number of beta values.
        n_steps: Steps per simulation.
        dt: Timestep.

    Returns:
        Dict with beta_values, disease_prevalence, S, I, P arrays.
    """
    beta_values = np.linspace(0.0, 0.05, n_beta)
    prevalence = []
    S_arr = []
    I_arr = []
    P_arr = []

    for i, beta_val in enumerate(beta_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 1.0, "K": 100.0, "beta": beta_val,
                "a1": 0.1, "a2": 0.3, "h1": 0.1, "h2": 0.1,
                "e1": 0.5, "e2": 0.3, "d_disease": 0.2, "m": 0.3,
                "S_0": 50.0, "I_0": 10.0, "P_0": 5.0,
            },
        )
        sim = EcoEpidemicSimulation(config)
        sim.reset()

        # Run to steady state
        for _ in range(n_steps):
            sim.step()

        # Average over last 10% for steady-state estimate
        tail_steps = max(1, int(n_steps * 0.1))
        tail_states = []
        for _ in range(tail_steps):
            sim.step()
            tail_states.append(sim.observe().copy())

        tail_states = np.array(tail_states)
        S_avg = np.mean(tail_states[:, 0])
        I_avg = np.mean(tail_states[:, 1])
        P_avg = np.mean(tail_states[:, 2])

        total_prey = S_avg + I_avg
        prev = I_avg / total_prey if total_prey > 1e-10 else 0.0

        prevalence.append(prev)
        S_arr.append(S_avg)
        I_arr.append(I_avg)
        P_arr.append(P_avg)

        if (i + 1) % 5 == 0:
            logger.info(f"  beta={beta_val:.4f}: prevalence={prev:.4f}")

    return {
        "beta_values": beta_values,
        "disease_prevalence": np.array(prevalence),
        "S": np.array(S_arr),
        "I": np.array(I_arr),
        "P": np.array(P_arr),
    }


def generate_predator_control_sweep(
    n_m: int = 20,
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep predator mortality m to show biological control effect.

    Args:
        n_m: Number of m values.
        n_steps: Steps per simulation.
        dt: Timestep.

    Returns:
        Dict with m_values, disease_prevalence, predator_pop arrays.
    """
    m_values = np.linspace(0.1, 1.0, n_m)
    prevalence = []
    pred_pop = []
    S_arr = []
    I_arr = []
    P_arr = []

    for i, m_val in enumerate(m_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "r": 1.0, "K": 100.0, "beta": 0.01,
                "a1": 0.1, "a2": 0.3, "h1": 0.1, "h2": 0.1,
                "e1": 0.5, "e2": 0.3, "d_disease": 0.2, "m": m_val,
                "S_0": 50.0, "I_0": 10.0, "P_0": 5.0,
            },
        )
        sim = EcoEpidemicSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        tail_steps = max(1, int(n_steps * 0.1))
        tail_states = []
        for _ in range(tail_steps):
            sim.step()
            tail_states.append(sim.observe().copy())

        tail_states = np.array(tail_states)
        S_avg = np.mean(tail_states[:, 0])
        I_avg = np.mean(tail_states[:, 1])
        P_avg = np.mean(tail_states[:, 2])

        total_prey = S_avg + I_avg
        prev = I_avg / total_prey if total_prey > 1e-10 else 0.0

        prevalence.append(prev)
        pred_pop.append(P_avg)
        S_arr.append(S_avg)
        I_arr.append(I_avg)
        P_arr.append(P_avg)

        if (i + 1) % 5 == 0:
            logger.info(f"  m={m_val:.3f}: prevalence={prev:.4f}, P={P_avg:.2f}")

    return {
        "m_values": m_values,
        "disease_prevalence": np.array(prevalence),
        "predator_pop": np.array(pred_pop),
        "S": np.array(S_arr),
        "I": np.array(I_arr),
        "P": np.array(P_arr),
    }


def run_eco_epidemic_rediscovery(
    output_dir: str | Path = "output/rediscovery/eco_epidemic",
    n_iterations: int = 40,
) -> dict:
    """Run eco-epidemiological model rediscovery pipeline.

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iterations.

    Returns:
        Results dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "eco_epidemic",
        "targets": {
            "ode": "dS/dt = rS(1-(S+I)/K) - beta*SI - a1*SP/(1+h1*a1*S), etc.",
            "R0": "beta*K/d (without predators)",
            "biological_control": "Predators reduce disease prevalence",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery...")
    data = generate_ode_data(n_steps=20000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["S", "I", "P"],
            threshold=0.05,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries[:6]
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

    # --- Part 2: Disease invasion threshold (beta sweep) ---
    logger.info("Part 2: Disease invasion threshold (beta sweep)...")
    beta_data = generate_beta_sweep(n_beta=25, n_steps=30000, dt=0.01)

    # Find disease invasion point: first beta where prevalence > threshold
    invasion_threshold = 0.01
    above = beta_data["disease_prevalence"] > invasion_threshold
    if np.any(above):
        idx = np.argmax(above)
        beta_c_est = beta_data["beta_values"][max(0, idx - 1)]
        results["disease_invasion"] = {
            "beta_c_estimate": float(beta_c_est),
            "R0_theory_no_pred": float(beta_c_est * 100.0 / 0.2),
            "n_endemic_points": int(np.sum(above)),
        }
        logger.info(f"  Disease invasion at beta ~ {beta_c_est:.4f}")
    else:
        results["disease_invasion"] = {"note": "No disease invasion detected"}

    # --- Part 3: Predator biological control effect ---
    logger.info("Part 3: Predator biological control sweep...")
    control_data = generate_predator_control_sweep(
        n_m=20, n_steps=30000, dt=0.01,
    )

    # Correlation: higher m (weaker predators) -> higher prevalence?
    finite_mask = np.isfinite(control_data["disease_prevalence"])
    if np.sum(finite_mask) > 3:
        m_fin = control_data["m_values"][finite_mask]
        prev_fin = control_data["disease_prevalence"][finite_mask]
        if np.std(m_fin) > 0 and np.std(prev_fin) > 0:
            corr = np.corrcoef(m_fin, prev_fin)[0, 1]
        else:
            corr = 0.0
        results["biological_control"] = {
            "correlation_m_vs_prevalence": float(corr),
            "min_prevalence": float(np.min(prev_fin)),
            "max_prevalence": float(np.max(prev_fin)),
            "note": (
                "Positive correlation means weaker predators -> more disease "
                "(predators act as biological control)"
            ),
        }
        logger.info(f"  Correlation(m, prevalence) = {corr:.4f}")

    # --- Part 4: PySR for R0 or disease prevalence ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use beta sweep data: prevalence = f(beta)
        valid = beta_data["disease_prevalence"] > 0.001
        if np.sum(valid) > 5:
            X = beta_data["beta_values"][valid].reshape(-1, 1)
            y = beta_data["disease_prevalence"][valid]

            logger.info("  Running PySR: prevalence = f(beta)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["b_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["prevalence_pysr"] = {
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
                results["prevalence_pysr"]["best"] = best.expression
                results["prevalence_pysr"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["prevalence_pysr"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
