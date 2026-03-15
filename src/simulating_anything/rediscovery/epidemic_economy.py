"""Rediscovery analysis for the novel Epidemic-Economy coupled system.

Sweeps coupling strength to discover:
1. How economic cycles amplify/dampen epidemics
2. SINDy ODE recovery for the coupled 4D system
3. Critical coupling where boom-bust-epidemic cycles emerge
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def run_epidemic_economy_rediscovery(
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
    from simulating_anything.simulation.epidemic_economy import (
        EpidemicEconomySimulation,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    logger.info("=== Epidemic-Economy Novel Discovery ===")

    # --- 1. Coupling sweep (u -> S transmission) ---
    couplings = np.linspace(0.0, 1.0, n_coupling_points)
    peak_infected = []
    final_susceptible = []
    mean_employment = []
    mean_wage = []

    for c in couplings:
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=3000,
            parameters={"coupling_uS": float(c)},
        )
        sim = EpidemicEconomySimulation(config)
        sim.reset(seed=0)

        states = []
        for _ in range(3000):
            sim.step()
            states.append(sim.observe().copy())

        states = np.array(states)
        peak_infected.append(float(np.max(states[:, 1])))
        final_susceptible.append(float(states[-1, 0]))
        mean_employment.append(float(np.mean(states[-1000:, 3])))
        mean_wage.append(float(np.mean(states[-1000:, 2])))

    logger.info(f"Coupling sweep: {n_coupling_points} points")
    logger.info(f"  Peak infected range: [{min(peak_infected):.3f}, {max(peak_infected):.3f}]")

    # --- 2. SINDy ODE recovery ---
    sindy_results = {}
    try:
        import pysindy as ps

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=5000,
            parameters={"coupling_uS": 0.3, "coupling_Iu": 0.5},
        )
        sim = EpidemicEconomySimulation(config)
        sim.reset(seed=0)

        states = [sim.observe().copy()]
        for _ in range(5000):
            sim.step()
            states.append(sim.observe().copy())
        data = np.array(states)

        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.005),
            feature_library=ps.PolynomialLibrary(degree=2),
        )
        dXdt = np.gradient(data, 0.1, axis=0)
        model.fit(data, t=0.1, x_dot=dXdt, feature_names=["S", "I", "w", "u"])

        equations = []
        r2_scores = []
        for i, name in enumerate(["S", "I", "w", "u"]):
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

    # --- 3. PySR: peak infected vs coupling ---
    pysr_results = {}
    try:
        from pysr import PySRRegressor

        X = couplings.reshape(-1, 1)
        y = np.array(peak_infected)

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
        logger.info(f"PySR peak I: {pysr_results['equation']}")
    except ImportError:
        logger.warning("PySR not available")
    except Exception as e:
        logger.warning(f"PySR failed: {e}")

    # --- 4. Economy-epidemic correlation ---
    config = SimulationConfig(
        domain=Domain.CUSTOM, dt=0.1, n_steps=5000,
        parameters={"coupling_uS": 0.3, "coupling_Iu": 0.5},
    )
    sim = EpidemicEconomySimulation(config)
    sim.reset(seed=0)
    states = []
    for _ in range(5000):
        sim.step()
        states.append(sim.observe().copy())
    states = np.array(states[-2000:])

    corr_uI = float(np.corrcoef(states[:, 3], states[:, 1])[0, 1])
    corr_wS = float(np.corrcoef(states[:, 2], states[:, 0])[0, 1])
    logger.info(f"  corr(u, I) = {corr_uI:.3f}")
    logger.info(f"  corr(w, S) = {corr_wS:.3f}")

    results = {
        "domain": "epidemic_economy",
        "type": "novel_coupled",
        "dimensions": 4,
        "coupling_sweep": {
            "couplings": couplings.tolist(),
            "peak_infected": peak_infected,
            "final_susceptible": final_susceptible,
            "mean_employment": mean_employment,
            "mean_wage": mean_wage,
        },
        "sindy": sindy_results,
        "pysr": pysr_results,
        "correlations": {
            "corr_u_I": corr_uI,
            "corr_w_S": corr_wS,
        },
        "best_r2": sindy_results.get("mean_r2", 0.0),
    }

    return results
