"""Aizawa attractor rediscovery.

Targets:
- SINDy recovery of Aizawa ODEs
- Lyapunov exponent estimation (positive for classic parameters)
- Attractor geometry: mushroom shape, bounded trajectory
- Parameter sweep for chaos characterization
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.aizawa import AizawaSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 0.95,
    b: float = 0.7,
    c: float = 0.6,
    d: float = 3.5,
    e: float = 0.25,
    f: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate a single Aizawa trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.AIZAWA,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a, "b": b, "c": c, "d": d, "e": e, "f": f,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        },
    )
    sim = AizawaSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "a": a, "b": b, "c": c, "d": d, "e": e, "f": f,
    }


def generate_lyapunov_sweep_data(
    n_a: int = 25,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep parameter a to characterize the chaos transition.

    The parameter a controls the z-dynamics growth rate. Varying it
    reveals transitions between periodic and chaotic regimes.
    """
    a_values = np.linspace(0.5, 1.2, n_a)
    lyapunov_exps = []

    for i, a_val in enumerate(a_values):
        config = SimulationConfig(
            domain=Domain.AIZAWA,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a_val, "b": 0.7, "c": 0.6,
                "d": 3.5, "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim = AizawaSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  a={a_val:.3f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def generate_attractor_statistics(
    dt: float = 0.01,
) -> dict[str, float]:
    """Compute attractor statistics at classic parameters.

    Measures the bounding box and radial extent of the mushroom attractor.
    """
    config = SimulationConfig(
        domain=Domain.AIZAWA,
        dt=dt,
        n_steps=20000,
        parameters={
            "a": 0.95, "b": 0.7, "c": 0.6,
            "d": 3.5, "e": 0.25, "f": 0.1,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        },
    )
    sim = AizawaSimulation(config)
    stats = sim.compute_trajectory_statistics(
        n_steps=15000, n_transient=5000
    )
    return stats


def run_aizawa_rediscovery(
    output_dir: str | Path = "output/rediscovery/aizawa",
    n_iterations: int = 40,
) -> dict:
    """Run the full Aizawa attractor rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep parameter a to characterize chaos transition
    3. Compute Lyapunov exponent at classic parameters
    4. Verify fixed points
    5. Compute attractor statistics (mushroom geometry)

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "aizawa",
        "targets": {
            "ode_x": "dx/dt = (z - b)*x - d*y",
            "ode_y": "dy/dt = d*x + (z - b)*y",
            "ode_z": "dz/dt = c + a*z - z^3/3 - (x^2+y^2)*(1+e*z) + f*z*x^3",
            "classic_params": "a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Aizawa trajectory for SINDy...")
    ode_data = generate_trajectory_data(n_steps=10000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x", "y", "z"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries
            ],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: Lyapunov sweep over parameter a ---
    logger.info("Part 2: Sweeping parameter a for chaos characterization...")
    lyap_data = generate_lyapunov_sweep_data(n_a=25, n_steps=30000, dt=0.01)

    lam = lyap_data["lyapunov_exponent"]
    a_vals = lyap_data["a"]
    n_chaotic = int(np.sum(lam > 0.01))
    n_stable = int(np.sum(lam < -0.01))

    results["a_sweep"] = {
        "n_a_values": len(a_vals),
        "n_chaotic": n_chaotic,
        "n_stable": n_stable,
        "a_range": [float(a_vals[0]), float(a_vals[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }
    logger.info(f"  {n_chaotic} chaotic, {n_stable} stable regimes found")

    # --- Part 3: Lyapunov at classic parameters ---
    logger.info("Part 3: Lyapunov exponent at classic parameters...")
    config_classic = SimulationConfig(
        domain=Domain.AIZAWA,
        dt=0.01,
        n_steps=50000,
        parameters={
            "a": 0.95, "b": 0.7, "c": 0.6,
            "d": 3.5, "e": 0.25, "f": 0.1,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        },
    )
    sim_classic = AizawaSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.01)

    results["classic_parameters"] = {
        "a": 0.95, "b": 0.7, "c": 0.6,
        "d": 3.5, "e": 0.25, "f": 0.1,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Aizawa Lyapunov: {lam_classic:.4f}")

    # --- Part 4: Fixed points ---
    logger.info("Part 4: Computing fixed points...")
    sim_fp = AizawaSimulation(config_classic)
    sim_fp.reset()
    fps = sim_fp.fixed_points
    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": [fp.tolist() for fp in fps],
    }
    logger.info(f"  Fixed points: {len(fps)} found")
    for i, fp in enumerate(fps):
        derivs = sim_fp._derivatives(fp)
        logger.info(
            f"    FP{i+1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}], "
            f"|deriv|={np.linalg.norm(derivs):.2e}"
        )

    # --- Part 5: Attractor statistics ---
    logger.info("Part 5: Computing attractor statistics...")
    stats = generate_attractor_statistics(dt=0.01)
    results["attractor_statistics"] = stats
    logger.info(
        f"  Attractor extent: x=[{stats['x_min']:.2f}, {stats['x_max']:.2f}], "
        f"z=[{stats['z_min']:.2f}, {stats['z_max']:.2f}], r_max={stats['r_max']:.2f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "ode_data.npz",
        states=ode_data["states"],
    )
    np.savez(
        output_path / "lyapunov_sweep.npz",
        a=lyap_data["a"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )

    return results
