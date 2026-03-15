"""Rediscovery analysis for the novel Tumor-Immune coupled system.

Sweeps immune evasion coupling to discover:
1. How immunosuppressive cytokines enable tumor escape
2. SINDy ODE recovery for the coupled 4D system
3. Critical coupling for tumor escape vs immune control
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def run_tumor_immune_rediscovery(
    n_coupling_points: int = 25,
    pysr_iterations: int = 40,
) -> dict:
    """Run immune evasion sweep and equation discovery."""
    from simulating_anything.simulation.tumor_immune import TumorImmuneSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    logger.info("=== Tumor-Immune Novel Discovery ===")

    # --- 1. Coupling sweep (cytokine immunosuppression) ---
    couplings = np.linspace(0.0, 0.2, n_coupling_points)
    final_tumor = []
    max_tumor = []
    final_immune = []

    for c in couplings:
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=5000,
            parameters={"coupling_ct": float(c)},
        )
        sim = TumorImmuneSimulation(config)
        sim.reset(seed=0)

        states = []
        for _ in range(5000):
            sim.step()
            states.append(sim.observe().copy())

        states = np.array(states)
        final_tumor.append(float(states[-1, 0]))
        max_tumor.append(float(np.max(states[:, 0])))
        final_immune.append(float(states[-1, 1] + states[-1, 2]))

    logger.info(f"Coupling sweep: {n_coupling_points} points")
    logger.info(f"  Final tumor range: [{min(final_tumor):.2f}, {max(final_tumor):.2f}]")

    # Detect escape threshold
    escape_idx = 0
    baseline_tumor = final_tumor[0]
    for i in range(1, len(final_tumor)):
        if final_tumor[i] > 2 * baseline_tumor + 1.0:
            escape_idx = i
            break
    escape_coupling = float(couplings[escape_idx]) if escape_idx > 0 else float("nan")
    logger.info(f"  Tumor escape at coupling ~ {escape_coupling:.3f}")

    # --- 2. SINDy ODE recovery ---
    sindy_results = {}
    try:
        import pysindy as ps

        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.1, n_steps=10000,
            parameters={"coupling_ct": 0.05},
        )
        sim = TumorImmuneSimulation(config)
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
        model.fit(data, t=0.1, x_dot=dXdt, feature_names=["T", "N", "I_c", "C"])

        equations = []
        r2_scores = []
        for i, name in enumerate(["T", "N", "I_c", "C"]):
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
        "domain": "tumor_immune",
        "type": "novel_coupled",
        "dimensions": 4,
        "coupling_sweep": {
            "couplings": couplings.tolist(),
            "final_tumor": final_tumor,
            "max_tumor": max_tumor,
            "final_immune": final_immune,
        },
        "escape_coupling": escape_coupling,
        "sindy": sindy_results,
        "best_r2": sindy_results.get("mean_r2", 0.0),
    }

    return results
