"""Rediscovery for Immune-Virus system."""
from __future__ import annotations
import logging, numpy as np
logger = logging.getLogger(__name__)

def run_immune_virus_rediscovery(n_coupling_points=25, pysr_iterations=40):
    from simulating_anything.simulation.immune_virus import ImmuneVirusSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig
    sindy_results = {}
    try:
        import pysindy as ps
        config = SimulationConfig(domain=Domain.CUSTOM, dt=0.01, n_steps=10000, parameters={})
        sim = ImmuneVirusSimulation(config)
        sim.reset(seed=0)
        states = [sim.observe().copy()]
        for _ in range(10000): sim.step(); states.append(sim.observe().copy())
        data = np.array(states)
        dXdt = np.gradient(data, 0.01, axis=0)
        model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.01), feature_library=ps.PolynomialLibrary(degree=2))
        model.fit(data, t=0.01, x_dot=dXdt, feature_names=["V", "Tc", "A", "Ic"])
        equations = [f"d({n})/dt = {model.equations(precision=4)[i]}" for i, n in enumerate(["V", "Tc", "A", "Ic"])]
        x_dot_pred = model.predict(data)
        r2_scores = [float(1.0 - np.sum((dXdt[:,i]-x_dot_pred[:,i])**2)/max(np.sum((dXdt[:,i]-np.mean(dXdt[:,i]))**2),1e-10)) for i in range(4)]
        sindy_results = {"equations": equations, "r2_scores": r2_scores, "mean_r2": float(np.mean(r2_scores))}
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
    return {"domain": "immune_virus", "type": "novel_coupled", "dimensions": 4,
            "sindy": sindy_results, "best_r2": sindy_results.get("mean_r2", 0.0)}
