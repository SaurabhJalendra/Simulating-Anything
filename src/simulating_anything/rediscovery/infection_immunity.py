"""Rediscovery for novel Infection-Immunity system."""
from __future__ import annotations
import logging, numpy as np
logger = logging.getLogger(__name__)

def run_infection_immunity_rediscovery(n_coupling_points=25, pysr_iterations=40):
    from simulating_anything.simulation.infection_immunity import InfectionImmunitySimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig
    logger.info("=== Infection-Immunity Novel Discovery ===")
    couplings = np.linspace(0.0, 5.0, n_coupling_points)
    endemic_I, endemic_M = [], []
    for c in couplings:
        config = SimulationConfig(domain=Domain.CUSTOM, dt=0.1, n_steps=10000, parameters={"coupling_MR": float(c)})
        sim = InfectionImmunitySimulation(config)
        sim.reset(seed=0)
        states = []
        for _ in range(10000): sim.step(); states.append(sim.observe().copy())
        tail = np.array(states[-3000:])
        endemic_I.append(float(np.mean(tail[:, 1])))
        endemic_M.append(float(np.mean(tail[:, 3])))
    sindy_results = {}
    try:
        import pysindy as ps
        config = SimulationConfig(domain=Domain.CUSTOM, dt=0.1, n_steps=10000, parameters={"coupling_MR": 2.0})
        sim = InfectionImmunitySimulation(config)
        sim.reset(seed=0)
        states = [sim.observe().copy()]
        for _ in range(10000): sim.step(); states.append(sim.observe().copy())
        data = np.array(states)
        dXdt = np.gradient(data, 0.1, axis=0)
        model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.003), feature_library=ps.PolynomialLibrary(degree=2))
        model.fit(data, t=0.1, x_dot=dXdt, feature_names=["S", "I", "R", "M"])
        equations = [f"d({n})/dt = {model.equations(precision=4)[i]}" for i, n in enumerate(["S", "I", "R", "M"])]
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
    return {"domain": "infection_immunity", "type": "novel_coupled", "dimensions": 4,
            "coupling_sweep": {"couplings": couplings.tolist(), "endemic_I": endemic_I, "endemic_M": endemic_M},
            "sindy": sindy_results, "best_r2": sindy_results.get("mean_r2", 0.0)}
