"""Rediscovery for novel Prey-Disease-Predator eco-epidemic system."""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def run_prey_disease_predator_rediscovery(n_coupling_points=25, pysr_iterations=40):
    from simulating_anything.simulation.prey_disease_predator import PreyDiseasePredatorSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    logger.info("=== Prey-Disease-Predator Novel Discovery ===")

    # Sweep selective predation advantage (a_i / a_s ratio)
    a_i_values = np.linspace(0.3, 1.5, n_coupling_points)
    mean_prey_total = []
    mean_prevalence = []
    mean_predator = []

    for ai in a_i_values:
        config = SimulationConfig(domain=Domain.CUSTOM, dt=0.01, n_steps=10000,
                                  parameters={"a_i": float(ai), "a_s": 0.3})
        sim = PreyDiseasePredatorSimulation(config)
        sim.reset(seed=0)
        states = []
        for _ in range(10000):
            sim.step()
            states.append(sim.observe().copy())
        tail = np.array(states[-5000:])
        mean_prey_total.append(float(np.mean(tail[:, 0] + tail[:, 1])))
        mean_prevalence.append(float(np.mean(tail[:, 3])))
        mean_predator.append(float(np.mean(tail[:, 2])))

    logger.info(f"  Prevalence range: [{min(mean_prevalence):.3f}, {max(mean_prevalence):.3f}]")

    sindy_results = {}
    try:
        import pysindy as ps
        config = SimulationConfig(domain=Domain.CUSTOM, dt=0.01, n_steps=10000,
                                  parameters={"a_i": 0.6, "a_s": 0.3})
        sim = PreyDiseasePredatorSimulation(config)
        sim.reset(seed=0)
        states = [sim.observe().copy()]
        for _ in range(10000):
            sim.step()
            states.append(sim.observe().copy())
        data = np.array(states)
        dXdt = np.gradient(data, 0.01, axis=0)
        model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.005),
                         feature_library=ps.PolynomialLibrary(degree=2))
        model.fit(data, t=0.01, x_dot=dXdt, feature_names=["Xs", "Xi", "Y", "Z"])
        equations = [f"d({n})/dt = {model.equations(precision=4)[i]}"
                     for i, n in enumerate(["Xs", "Xi", "Y", "Z"])]
        x_dot_pred = model.predict(data)
        r2_scores = []
        for i in range(4):
            ss_res = np.sum((dXdt[:, i] - x_dot_pred[:, i]) ** 2)
            ss_tot = np.sum((dXdt[:, i] - np.mean(dXdt[:, i])) ** 2)
            r2_scores.append(float(1.0 - ss_res / max(ss_tot, 1e-10)))
        sindy_results = {"equations": equations, "r2_scores": r2_scores,
                         "mean_r2": float(np.mean(r2_scores))}
        logger.info(f"SINDy mean R²: {sindy_results['mean_r2']:.4f}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")

    return {
        "domain": "prey_disease_predator", "type": "novel_coupled", "dimensions": 4,
        "selective_predation_sweep": {
            "a_i_values": a_i_values.tolist(),
            "mean_prey_total": mean_prey_total,
            "mean_prevalence": mean_prevalence,
            "mean_predator": mean_predator,
        },
        "sindy": sindy_results,
        "best_r2": sindy_results.get("mean_r2", 0.0),
    }
