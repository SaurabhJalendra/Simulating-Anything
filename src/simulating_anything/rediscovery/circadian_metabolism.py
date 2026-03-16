"""Rediscovery for novel Circadian-Metabolism coupled system."""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def run_circadian_metabolism_rediscovery(n_coupling_points=25, pysr_iterations=40):
    from simulating_anything.simulation.circadian_metabolism import CircadianMetabolismSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    logger.info("=== Circadian-Metabolism Novel Discovery ===")

    couplings = np.linspace(0.0, 1.0, n_coupling_points)
    mean_mRNA = []
    std_mRNA = []
    mean_substrate = []

    for c in couplings:
        config = SimulationConfig(domain=Domain.CUSTOM, dt=0.1, n_steps=5000, parameters={"coupling_SM": float(c)})
        sim = CircadianMetabolismSimulation(config)
        sim.reset(seed=0)
        states = []
        for _ in range(5000):
            sim.step()
            states.append(sim.observe().copy())
        tail = np.array(states[-2000:])
        mean_mRNA.append(float(np.mean(tail[:, 0])))
        std_mRNA.append(float(np.std(tail[:, 0])))
        mean_substrate.append(float(np.mean(tail[:, 3])))

    sindy_results = {}
    try:
        import pysindy as ps
        config = SimulationConfig(domain=Domain.CUSTOM, dt=0.1, n_steps=10000, parameters={"coupling_PE": 0.3, "coupling_SM": 0.2})
        sim = CircadianMetabolismSimulation(config)
        sim.reset(seed=0)
        states = [sim.observe().copy()]
        for _ in range(10000):
            sim.step()
            states.append(sim.observe().copy())
        data = np.array(states)
        dXdt = np.gradient(data, 0.1, axis=0)
        model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.005), feature_library=ps.PolynomialLibrary(degree=2))
        model.fit(data, t=0.1, x_dot=dXdt, feature_names=["M", "Pc", "E", "S"])
        equations = [f"d({n})/dt = {model.equations(precision=4)[i]}" for i, n in enumerate(["M", "Pc", "E", "S"])]
        x_dot_pred = model.predict(data)
        r2_scores = []
        for i in range(4):
            ss_res = np.sum((dXdt[:, i] - x_dot_pred[:, i]) ** 2)
            ss_tot = np.sum((dXdt[:, i] - np.mean(dXdt[:, i])) ** 2)
            r2_scores.append(float(1.0 - ss_res / max(ss_tot, 1e-10)))
        sindy_results = {"equations": equations, "r2_scores": r2_scores, "mean_r2": float(np.mean(r2_scores))}
        logger.info(f"SINDy mean R²: {sindy_results['mean_r2']:.4f}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")

    return {
        "domain": "circadian_metabolism", "type": "novel_coupled", "dimensions": 4,
        "coupling_sweep": {"couplings": couplings.tolist(), "mean_mRNA": mean_mRNA, "std_mRNA": std_mRNA, "mean_substrate": mean_substrate},
        "sindy": sindy_results, "best_r2": sindy_results.get("mean_r2", 0.0),
    }
