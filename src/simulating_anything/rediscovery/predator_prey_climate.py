"""Rediscovery analysis for the novel Predator-Prey-Climate coupled system.

Sweeps coupling strength to discover:
1. How climate oscillations modulate population dynamics
2. SINDy ODE recovery for the coupled 4D system
3. Bifurcation from stable equilibrium to oscillations as coupling increases
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def run_predator_prey_climate_rediscovery(
    n_coupling_points: int = 25,
    pysr_iterations: int = 40,
) -> dict:
    """Run coupling sweep and equation discovery.

    Args:
        n_coupling_points: Number of coupling values to sweep.
        pysr_iterations: PySR iterations.

    Returns:
        Dict with all results.
    """
    from simulating_anything.simulation.predator_prey_climate import (
        PredatorPreyClimateSimulation,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    logger.info("=== Predator-Prey-Climate Novel Discovery ===")

    # --- 1. Coupling sweep ---
    couplings = np.linspace(0.0, 0.5, n_coupling_points)
    mean_preys = []
    mean_preds = []
    std_preys = []
    std_preds = []
    mean_temps = []

    for c in couplings:
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=5000,
            parameters={"coupling_TK": float(c)},
        )
        sim = PredatorPreyClimateSimulation(config)
        sim.reset(seed=0)

        states = []
        for _ in range(5000):
            sim.step()
            states.append(sim.observe().copy())

        states = np.array(states)
        # Use last 2000 steps for statistics (after transient)
        tail = states[-2000:]
        mean_preys.append(float(np.mean(tail[:, 0])))
        mean_preds.append(float(np.mean(tail[:, 1])))
        std_preys.append(float(np.std(tail[:, 0])))
        std_preds.append(float(np.std(tail[:, 1])))
        mean_temps.append(float(np.mean(tail[:, 2])))

    logger.info(f"Coupling sweep: {n_coupling_points} points")
    logger.info(f"  Prey std range: [{min(std_preys):.3f}, {max(std_preys):.3f}]")

    # --- 2. Detect bifurcation: coupling value where oscillation amplitude jumps ---
    bifurcation_idx = 0
    for i in range(1, len(std_preys)):
        if std_preys[i] > 2 * std_preys[0] + 0.01:
            bifurcation_idx = i
            break
    coupling_bifurcation = float(couplings[bifurcation_idx])
    logger.info(f"  Bifurcation at coupling ~ {coupling_bifurcation:.3f}")

    # --- 3. SINDy ODE recovery ---
    sindy_results = {}
    try:
        import pysindy as ps

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=10000,
            parameters={"coupling_TK": 0.2},
        )
        sim = PredatorPreyClimateSimulation(config)
        sim.reset(seed=0)

        states = [sim.observe().copy()]
        for _ in range(10000):
            sim.step()
            states.append(sim.observe().copy())
        data = np.array(states)

        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.01),
            feature_library=ps.PolynomialLibrary(degree=2),
        )
        dXdt = np.gradient(data, 0.01, axis=0)
        model.fit(data, t=0.01, x_dot=dXdt, feature_names=["N", "P", "T", "S"])

        equations = []
        r2_scores = []
        for i, name in enumerate(["N", "P", "T", "S"]):
            eq = model.equations(precision=4)[i]
            equations.append(f"d({name})/dt = {eq}")
        x_dot_pred = model.predict(data)
        for i in range(4):
            ss_res = np.sum((dXdt[:, i] - x_dot_pred[:, i]) ** 2)
            ss_tot = np.sum((dXdt[:, i] - np.mean(dXdt[:, i])) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            r2_scores.append(float(r2))

        sindy_results = {
            "equations": equations,
            "r2_scores": r2_scores,
            "mean_r2": float(np.mean(r2_scores)),
        }
        logger.info(f"SINDy mean R²: {sindy_results['mean_r2']:.4f}")
        for eq in equations:
            logger.info(f"  {eq}")
    except ImportError:
        logger.warning("PySINDy not available")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")

    # --- 4. PySR: prey amplitude vs coupling ---
    pysr_results = {}
    try:
        from pysr import PySRRegressor

        X = couplings.reshape(-1, 1)
        y = np.array(std_preys)

        model = PySRRegressor(
            niterations=pysr_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "exp"],
            maxsize=15,
            populations=30,
            verbosity=0,
        )
        model.fit(X, y, variable_names=["c_"])
        best = model.get_best()
        pysr_results = {
            "equation": str(best["equation"]),
            "r2": float(best["score"]) if "score" in best else None,
            "complexity": int(best["complexity"]),
        }
        logger.info(f"PySR prey amplitude: {pysr_results['equation']}")
    except ImportError:
        logger.warning("PySR not available")
    except Exception as e:
        logger.warning(f"PySR failed: {e}")

    # --- 5. Correlation analysis ---
    # Strong coupling: how correlated are temperature and prey?
    config = SimulationConfig(
        domain=Domain.CUSTOM, dt=0.01, n_steps=10000,
        parameters={"coupling_TK": 0.3},
    )
    sim = PredatorPreyClimateSimulation(config)
    sim.reset(seed=0)
    states = []
    for _ in range(10000):
        sim.step()
        states.append(sim.observe().copy())
    states = np.array(states[-5000:])

    corr_TN = float(np.corrcoef(states[:, 2], states[:, 0])[0, 1])
    corr_TP = float(np.corrcoef(states[:, 2], states[:, 1])[0, 1])
    logger.info(f"  corr(T, N) = {corr_TN:.3f}")
    logger.info(f"  corr(T, P) = {corr_TP:.3f}")

    results = {
        "domain": "predator_prey_climate",
        "type": "novel_coupled",
        "dimensions": 4,
        "coupling_sweep": {
            "couplings": couplings.tolist(),
            "mean_prey": mean_preys,
            "std_prey": std_preys,
            "mean_predator": mean_preds,
            "std_predator": std_preds,
            "mean_temperature": mean_temps,
        },
        "bifurcation_coupling": coupling_bifurcation,
        "sindy": sindy_results,
        "pysr": pysr_results,
        "correlations": {
            "corr_T_N": corr_TN,
            "corr_T_P": corr_TP,
        },
        "best_r2": sindy_results.get("mean_r2", 0.0),
    }

    return results
