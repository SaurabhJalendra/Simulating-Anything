"""Rediscovery analysis for the novel Gene-Regulation-Metabolism system."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def run_gene_metabolism_rediscovery(
    n_coupling_points: int = 25,
    pysr_iterations: int = 40,
) -> dict:
    """Run coupling sweep and equation discovery."""
    from simulating_anything.simulation.gene_metabolism import (
        GeneMetabolismSimulation,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    logger.info("=== Gene-Metabolism Novel Discovery ===")

    couplings = np.linspace(0.0, 1.0, n_coupling_points)
    mean_g1 = []
    mean_g2 = []
    std_g1 = []
    mean_m2 = []

    for c in couplings:
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=5000,
            parameters={"coupling_mg": float(c)},
        )
        sim = GeneMetabolismSimulation(config)
        sim.reset(seed=0)
        states = []
        for _ in range(5000):
            sim.step()
            states.append(sim.observe().copy())
        states = np.array(states)
        tail = states[-2000:]
        mean_g1.append(float(np.mean(tail[:, 0])))
        mean_g2.append(float(np.mean(tail[:, 1])))
        std_g1.append(float(np.std(tail[:, 0])))
        mean_m2.append(float(np.mean(tail[:, 2])))

    logger.info(f"Coupling sweep: {n_coupling_points} points")
    logger.info(f"  g1 std range: [{min(std_g1):.3f}, {max(std_g1):.3f}]")

    # SINDy
    sindy_results = {}
    try:
        import pysindy as ps

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.01, n_steps=10000,
            parameters={"coupling_mg": 0.3, "coupling_gm": 0.5},
        )
        sim = GeneMetabolismSimulation(config)
        sim.reset(seed=0)
        states = [sim.observe().copy()]
        for _ in range(10000):
            sim.step()
            states.append(sim.observe().copy())
        data = np.array(states)

        dXdt = np.gradient(data, 0.01, axis=0)
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.01),
            feature_library=ps.PolynomialLibrary(degree=2),
        )
        model.fit(data, t=0.01, x_dot=dXdt, feature_names=["g1", "g2", "m1", "m2"])

        equations = []
        r2_scores = []
        for i, name in enumerate(["g1", "g2", "m1", "m2"]):
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

    # Correlation
    config = SimulationConfig(
        domain=Domain.CUSTOM, dt=0.01, n_steps=10000,
        parameters={"coupling_mg": 0.3},
    )
    sim = GeneMetabolismSimulation(config)
    sim.reset(seed=0)
    states = []
    for _ in range(10000):
        sim.step()
        states.append(sim.observe().copy())
    states = np.array(states[-5000:])

    corr_g1m2 = float(np.corrcoef(states[:, 0], states[:, 3])[0, 1])
    logger.info(f"  corr(g1, m2) = {corr_g1m2:.3f}")

    results = {
        "domain": "gene_metabolism",
        "type": "novel_coupled",
        "dimensions": 4,
        "coupling_sweep": {
            "couplings": couplings.tolist(),
            "mean_g1": mean_g1,
            "std_g1": std_g1,
            "mean_g2": mean_g2,
            "mean_m2": mean_m2,
        },
        "sindy": sindy_results,
        "correlations": {"corr_g1_m2": corr_g1m2},
        "best_r2": sindy_results.get("mean_r2", 0.0),
    }
    return results
