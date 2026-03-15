"""Rediscovery analysis for the novel Plankton-Ocean coupled system."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def run_plankton_ocean_rediscovery(
    n_mixing_points: int = 25,
    pysr_iterations: int = 40,
) -> dict:
    """Run mixing rate sweep and equation discovery."""
    from simulating_anything.simulation.plankton_ocean import PlanktonOceanSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    logger.info("=== Plankton-Ocean Novel Discovery ===")

    mixings = np.linspace(0.01, 0.2, n_mixing_points)
    mean_P = []
    std_P = []
    mean_N = []
    bloom_size = []

    for w in mixings:
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=10000,
            parameters={"w_mix": float(w)},
        )
        sim = PlanktonOceanSimulation(config)
        sim.reset(seed=0)
        states = []
        for _ in range(10000):
            sim.step()
            states.append(sim.observe().copy())
        states = np.array(states)
        tail = states[-5000:]
        mean_P.append(float(np.mean(tail[:, 0])))
        std_P.append(float(np.std(tail[:, 0])))
        mean_N.append(float(np.mean(tail[:, 2])))
        bloom_size.append(float(np.max(tail[:, 0])))

    logger.info(f"Mixing sweep: {n_mixing_points} points")
    logger.info(f"  Mean P range: [{min(mean_P):.3f}, {max(mean_P):.3f}]")

    sindy_results = {}
    try:
        import pysindy as ps

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=10000,
            parameters={"w_mix": 0.05},
        )
        sim = PlanktonOceanSimulation(config)
        sim.reset(seed=0)
        states = [sim.observe().copy()]
        for _ in range(10000):
            sim.step()
            states.append(sim.observe().copy())
        data = np.array(states)

        dXdt = np.gradient(data, 0.1, axis=0)
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.005),
            feature_library=ps.PolynomialLibrary(degree=2),
        )
        model.fit(data, t=0.1, x_dot=dXdt, feature_names=["P", "Z", "N", "D"])

        equations = []
        r2_scores = []
        for i, name in enumerate(["P", "Z", "N", "D"]):
            eq = model.equations(precision=4)[i]
            equations.append(f"d({name})/dt = {eq}")

        x_dot_pred = model.predict(data)
        for i in range(4):
            ss_res = np.sum((dXdt[:, i] - x_dot_pred[:, i]) ** 2)
            ss_tot = np.sum((dXdt[:, i] - np.mean(dXdt[:, i])) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
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

    results = {
        "domain": "plankton_ocean",
        "type": "novel_coupled",
        "dimensions": 4,
        "mixing_sweep": {
            "mixing_rates": mixings.tolist(),
            "mean_phyto": mean_P,
            "std_phyto": std_P,
            "mean_nutrient": mean_N,
            "bloom_size": bloom_size,
        },
        "sindy": sindy_results,
        "best_r2": sindy_results.get("mean_r2", 0.0),
    }
    return results
