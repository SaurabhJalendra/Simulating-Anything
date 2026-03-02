"""Sakarya attractor rediscovery.

Targets:
- SINDy recovery of Sakarya ODEs:
    dx/dt = -x + y + y*z
    dy/dt = -x - y + a*x*z
    dz/dt = z - b*x*y
- Lyapunov exponent estimation (positive for chaotic regime)
- Chaos transition as parameter a varies
- Fixed point computation and verification
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sakarya import SakaryaSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 0.4,
    b: float = 0.3,
) -> dict[str, np.ndarray]:
    """Generate a single Sakarya trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.SAKARYA,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "x_0": 1.0,
            "y_0": -1.0,
            "z_0": 1.0,
        },
    )
    sim = SakaryaSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "a": a,
        "b": b,
    }


def generate_lyapunov_vs_a_data(
    n_a: int = 30,
    n_steps: int = 30000,
    dt: float = 0.01,
    b: float = 0.3,
) -> dict[str, np.ndarray]:
    """Sweep parameter a to map the chaos transition.

    Focuses on a in [0.1, 1.0] to capture the transition behavior.
    """
    a_values = np.linspace(0.1, 1.0, n_a)
    lyapunov_exps = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=Domain.SAKARYA,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a,
                "b": b,
                "x_0": 1.0,
                "y_0": -1.0,
                "z_0": 1.0,
            },
        )
        sim = SakaryaSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  a={a:.4f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_sakarya_rediscovery(
    output_dir: str | Path = "output/rediscovery/sakarya",
    n_iterations: int = 40,
) -> dict:
    """Run the full Sakarya system rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep a to map chaos transition (Lyapunov exponent)
    3. Compute Lyapunov at standard parameters
    4. Verify fixed points

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "sakarya",
        "targets": {
            "ode_x": "dx/dt = -x + y + y*z",
            "ode_y": "dy/dt = -x - y + a*x*z",
            "ode_z": "dz/dt = z - b*x*y",
            "chaos_regime": "a=0.4, b=0.3 produces bounded chaos",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Sakarya trajectory for SINDy...")
    ode_data = generate_trajectory_data(n_steps=10000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x", "y", "z"],
            threshold=0.05,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries
            ],
            "true_a": ode_data["a"],
            "true_b": ode_data["b"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Lyapunov vs a sweep ---
    logger.info("Part 2: Mapping chaos transition (a sweep)...")
    lyap_data = generate_lyapunov_vs_a_data(n_a=30, n_steps=30000, dt=0.01)

    lam = lyap_data["lyapunov_exponent"]
    a_vals = lyap_data["a"]

    n_chaotic = int(np.sum(lam > 0.01))
    n_stable = int(np.sum(lam < -0.01))

    results["chaos_transition"] = {
        "n_a_values": len(a_vals),
        "n_chaotic": n_chaotic,
        "n_stable": n_stable,
        "a_range": [float(a_vals[0]), float(a_vals[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }
    logger.info(f"  Found {n_chaotic} chaotic, {n_stable} stable regimes")

    # --- Part 3: Lyapunov at classic parameters ---
    logger.info("Part 3: Lyapunov exponent at classic parameters...")
    config_classic = SimulationConfig(
        domain=Domain.SAKARYA,
        dt=0.01,
        n_steps=50000,
        parameters={"a": 0.4, "b": 0.3, "x_0": 1.0, "y_0": -1.0, "z_0": 1.0},
    )
    sim_classic = SakaryaSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.01)

    results["classic_parameters"] = {
        "a": 0.4,
        "b": 0.3,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Sakarya Lyapunov: {lam_classic:.4f}")

    # --- Part 4: Fixed points ---
    sim_fp = SakaryaSimulation(config_classic)
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

    # --- Part 5: Trajectory statistics ---
    logger.info("Part 5: Trajectory statistics at chaotic parameters...")
    config_stats = SimulationConfig(
        domain=Domain.SAKARYA,
        dt=0.01,
        n_steps=20000,
        parameters={"a": 0.4, "b": 0.3, "x_0": 1.0, "y_0": -1.0, "z_0": 1.0},
    )
    sim_stats = SakaryaSimulation(config_stats)
    traj_stats = sim_stats.compute_trajectory_statistics(
        n_steps=15000, n_transient=5000
    )
    results["trajectory_statistics"] = traj_stats

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "ode_data.npz",
        states=ode_data["states"],
    )
    np.savez(
        output_path / "lyapunov_vs_a.npz",
        a=lyap_data["a"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )

    return results
