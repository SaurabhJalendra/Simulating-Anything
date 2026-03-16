"""Rediscovery for novel Neuron-Astrocyte system."""
from __future__ import annotations
import logging, numpy as np
logger = logging.getLogger(__name__)

def run_neuron_astrocyte_rediscovery(n_coupling_points=25, pysr_iterations=40):
    from simulating_anything.simulation.neuron_astrocyte import NeuronAstrocyteSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig
    logger.info("=== Neuron-Astrocyte Novel Discovery ===")
    couplings = np.linspace(0.0, 0.5, n_coupling_points)
    mean_Ca, mean_v = [], []
    for c in couplings:
        config = SimulationConfig(domain=Domain.CUSTOM, dt=0.1, n_steps=5000, parameters={"coupling_an": float(c)})
        sim = NeuronAstrocyteSimulation(config)
        sim.reset(seed=0)
        states = []
        for _ in range(5000): sim.step(); states.append(sim.observe().copy())
        tail = np.array(states[-2000:])
        mean_Ca.append(float(np.mean(tail[:, 2])))
        mean_v.append(float(np.mean(tail[:, 0])))
    sindy_results = {}
    try:
        import pysindy as ps
        config = SimulationConfig(domain=Domain.CUSTOM, dt=0.1, n_steps=10000, parameters={"coupling_na": 0.2, "coupling_an": 0.1})
        sim = NeuronAstrocyteSimulation(config)
        sim.reset(seed=0)
        states = [sim.observe().copy()]
        for _ in range(10000): sim.step(); states.append(sim.observe().copy())
        data = np.array(states)
        dXdt = np.gradient(data, 0.1, axis=0)
        model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.005), feature_library=ps.PolynomialLibrary(degree=2))
        model.fit(data, t=0.1, x_dot=dXdt, feature_names=["v", "w", "Ca", "IP3"])
        equations = [f"d({n})/dt = {model.equations(precision=4)[i]}" for i, n in enumerate(["v", "w", "Ca", "IP3"])]
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
    corr = float(np.corrcoef(np.array(mean_Ca), np.array(mean_v))[0, 1]) if len(mean_Ca) > 1 else 0.0
    return {"domain": "neuron_astrocyte", "type": "novel_coupled", "dimensions": 4,
            "sindy": sindy_results, "correlations": {"corr_Ca_v": corr},
            "best_r2": sindy_results.get("mean_r2", 0.0)}
