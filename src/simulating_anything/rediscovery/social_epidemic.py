"""Rediscovery analysis for the novel Social-Epidemic coupled system."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def run_social_epidemic_rediscovery(
    n_coupling_points: int = 25,
    pysr_iterations: int = 40,
) -> dict:
    """Run coupling sweep and equation discovery."""
    from simulating_anything.simulation.social_epidemic import SocialEpidemicSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    logger.info("=== Social-Epidemic Novel Discovery ===")

    couplings = np.linspace(0.0, 3.0, n_coupling_points)
    peak_infected = []
    final_opinion = []
    final_vax_coverage = []

    for c in couplings:
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=5000,
            parameters={"coupling_IS": float(c)},
        )
        sim = SocialEpidemicSimulation(config)
        sim.reset(seed=0)
        states = []
        for _ in range(5000):
            sim.step()
            states.append(sim.observe().copy())
        states = np.array(states)
        peak_infected.append(float(np.max(states[:, 3])))
        final_opinion.append(float(states[-1, 0]))
        # Vaccination coverage approximated by 1 - S - I at end
        final_vax_coverage.append(float(1.0 - states[-1, 2] - states[-1, 3]))

    logger.info(f"Coupling sweep: {n_coupling_points} points")
    logger.info(f"  Peak I range: [{min(peak_infected):.3f}, {max(peak_infected):.3f}]")
    logger.info(f"  Final opinion range: [{min(final_opinion):.3f}, {max(final_opinion):.3f}]")

    sindy_results = {}
    try:
        import pysindy as ps

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=5000,
            parameters={"coupling_IS": 1.0},
        )
        sim = SocialEpidemicSimulation(config)
        sim.reset(seed=0)
        states = [sim.observe().copy()]
        for _ in range(5000):
            sim.step()
            states.append(sim.observe().copy())
        data = np.array(states)

        dXdt = np.gradient(data, 0.1, axis=0)
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.005),
            feature_library=ps.PolynomialLibrary(degree=2),
        )
        model.fit(data, t=0.1, x_dot=dXdt, feature_names=["x", "sig", "S", "I"])

        equations = []
        r2_scores = []
        for i, name in enumerate(["x", "sig", "S", "I"]):
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

    # Correlation: does disease fear change opinions?
    config = SimulationConfig(
        domain=Domain.CUSTOM, dt=0.1, n_steps=5000,
        parameters={"coupling_IS": 1.0},
    )
    sim = SocialEpidemicSimulation(config)
    sim.reset(seed=0)
    states = []
    for _ in range(5000):
        sim.step()
        states.append(sim.observe().copy())
    states = np.array(states)

    corr_xI = float(np.corrcoef(states[:, 0], states[:, 3])[0, 1])
    logger.info(f"  corr(opinion, infected) = {corr_xI:.3f}")

    results = {
        "domain": "social_epidemic",
        "type": "novel_coupled",
        "dimensions": 4,
        "coupling_sweep": {
            "couplings": couplings.tolist(),
            "peak_infected": peak_infected,
            "final_opinion": final_opinion,
            "final_vax_coverage": final_vax_coverage,
        },
        "sindy": sindy_results,
        "correlations": {"corr_opinion_infected": corr_xI},
        "best_r2": sindy_results.get("mean_r2", 0.0),
    }
    return results
