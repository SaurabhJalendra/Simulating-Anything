"""Rediscovery for novel Battery-Thermal system."""
from __future__ import annotations
import logging, numpy as np
logger = logging.getLogger(__name__)

def run_battery_thermal_rediscovery(n_coupling_points=25, pysr_iterations=40):
    from simulating_anything.simulation.battery_thermal import BatteryThermalSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig
    logger.info("=== Battery-Thermal Novel Discovery ===")
    sindy_results = {}
    try:
        import pysindy as ps
        config = SimulationConfig(domain=Domain.CUSTOM, dt=0.1, n_steps=10000, parameters={})
        sim = BatteryThermalSimulation(config)
        sim.reset(seed=0)
        states = [sim.observe().copy()]
        for _ in range(10000): sim.step(); states.append(sim.observe().copy())
        data = np.array(states)
        dXdt = np.gradient(data, 0.1, axis=0)
        model = ps.SINDy(optimizer=ps.STLSQ(threshold=0.0001), feature_library=ps.PolynomialLibrary(degree=2))
        model.fit(data, t=0.1, x_dot=dXdt, feature_names=["Voc", "Vrc", "T", "SOC"])
        equations = [f"d({n})/dt = {model.equations(precision=4)[i]}" for i, n in enumerate(["Voc", "Vrc", "T", "SOC"])]
        x_dot_pred = model.predict(data)
        r2_scores = []
        for i in range(4):
            ss_res = np.sum((dXdt[:, i] - x_dot_pred[:, i]) ** 2)
            ss_tot = np.sum((dXdt[:, i] - np.mean(dXdt[:, i])) ** 2)
            r2_scores.append(float(1.0 - ss_res / max(ss_tot, 1e-10)))
        sindy_results = {"equations": equations, "r2_scores": r2_scores, "mean_r2": float(np.mean(r2_scores))}
        logger.info(f"SINDy mean R2: {sindy_results['mean_r2']:.4f}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
    return {"domain": "battery_thermal", "type": "novel_coupled", "dimensions": 4,
            "sindy": sindy_results, "best_r2": sindy_results.get("mean_r2", 0.0)}
