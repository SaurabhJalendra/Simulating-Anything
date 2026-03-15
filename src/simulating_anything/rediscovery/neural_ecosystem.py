"""Rediscovery analysis for the novel Neural-Ecosystem coupled system.

Sweeps coupling strength to discover:
1. How neural foraging drive modulates predator-prey oscillations
2. SINDy ODE recovery for the coupled 4D system
3. Phase synchronization between neural oscillations and population cycles
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def run_neural_ecosystem_rediscovery(
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
    from simulating_anything.simulation.neural_ecosystem import (
        NeuralEcosystemSimulation,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    logger.info("=== Neural-Ecosystem Novel Discovery ===")

    # --- 1. Coupling sweep (E -> predation) ---
    couplings = np.linspace(0.0, 0.5, n_coupling_points)
    mean_E = []
    mean_prey = []
    std_prey = []
    std_E = []
    prey_period = []

    for c in couplings:
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=10000,
            parameters={"coupling_EN": float(c), "coupling_NE": 0.1},
        )
        sim = NeuralEcosystemSimulation(config)
        sim.reset(seed=0)

        states = []
        for _ in range(10000):
            sim.step()
            states.append(sim.observe().copy())

        states = np.array(states)
        tail = states[-5000:]
        mean_E.append(float(np.mean(tail[:, 0])))
        mean_prey.append(float(np.mean(tail[:, 2])))
        std_prey.append(float(np.std(tail[:, 2])))
        std_E.append(float(np.std(tail[:, 0])))

        # Estimate period via zero-crossings
        prey_centered = tail[:, 2] - np.mean(tail[:, 2])
        crossings = np.where(np.diff(np.sign(prey_centered)))[0]
        if len(crossings) >= 2:
            periods = np.diff(crossings) * 0.01 * 2
            prey_period.append(float(np.mean(periods)))
        else:
            prey_period.append(0.0)

    logger.info(f"Coupling sweep: {n_coupling_points} points")
    logger.info(f"  Prey std range: [{min(std_prey):.3f}, {max(std_prey):.3f}]")

    # --- 2. SINDy ODE recovery ---
    sindy_results = {}
    try:
        import pysindy as ps

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=10000,
            parameters={"coupling_EN": 0.2, "coupling_NE": 0.1},
        )
        sim = NeuralEcosystemSimulation(config)
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
        model.fit(data, t=0.01, x_dot=dXdt, feature_names=["E", "In", "N", "P"])

        equations = []
        r2_scores = []
        for i, name in enumerate(["E", "In", "N", "P"]):
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

    # --- 3. PySR: prey period vs coupling ---
    pysr_results = {}
    try:
        from pysr import PySRRegressor

        valid = [(c, p) for c, p in zip(couplings, prey_period) if p > 0]
        if len(valid) >= 5:
            X = np.array([v[0] for v in valid]).reshape(-1, 1)
            y = np.array([v[1] for v in valid])

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
            logger.info(f"PySR period: {pysr_results['equation']}")
    except ImportError:
        logger.warning("PySR not available")
    except Exception as e:
        logger.warning(f"PySR failed: {e}")

    # --- 4. Phase synchronization analysis ---
    config = SimulationConfig(
        domain=Domain.CUSTOM, dt=0.01, n_steps=10000,
        parameters={"coupling_EN": 0.2, "coupling_NE": 0.1},
    )
    sim = NeuralEcosystemSimulation(config)
    sim.reset(seed=0)
    states = []
    for _ in range(10000):
        sim.step()
        states.append(sim.observe().copy())
    states = np.array(states[-5000:])

    corr_EN = float(np.corrcoef(states[:, 0], states[:, 2])[0, 1])
    corr_EP = float(np.corrcoef(states[:, 0], states[:, 3])[0, 1])
    logger.info(f"  corr(E, N) = {corr_EN:.3f}")
    logger.info(f"  corr(E, P) = {corr_EP:.3f}")

    results = {
        "domain": "neural_ecosystem",
        "type": "novel_coupled",
        "dimensions": 4,
        "coupling_sweep": {
            "couplings": couplings.tolist(),
            "mean_excitatory": mean_E,
            "std_excitatory": std_E,
            "mean_prey": mean_prey,
            "std_prey": std_prey,
            "prey_period": prey_period,
        },
        "sindy": sindy_results,
        "pysr": pysr_results,
        "correlations": {
            "corr_E_N": corr_EN,
            "corr_E_P": corr_EP,
        },
        "best_r2": sindy_results.get("mean_r2", 0.0),
    }

    return results
