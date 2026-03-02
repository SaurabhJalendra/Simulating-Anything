"""Dadras chaotic attractor rediscovery.

Targets:
- SINDy recovery of Dadras ODEs: x'=y-a*x+b*y*z, y'=c*y-x*z+z, z'=d*x*y-e*z
- Lyapunov exponent estimation (positive for chaotic regime)
- Parameter sweep mapping chaos boundary (vary a or e)
- Fixed point analysis
- Attractor statistics and boundedness verification
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.dadras import DadrasSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use CHAOTIC_ODE as the domain enum placeholder until Domain.DADRAS is added
_DADRAS_DOMAIN = Domain.CHAOTIC_ODE


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.005,
    a: float = 3.0,
    b: float = 2.7,
    c: float = 1.7,
    d: float = 2.0,
    e: float = 9.0,
) -> dict[str, np.ndarray]:
    """Generate a single Dadras trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=_DADRAS_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "x_0": 1.0,
            "y_0": 1.0,
            "z_0": 0.0,
        },
    )
    sim = DadrasSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
    }


def generate_lyapunov_vs_a_data(
    n_a: int = 30,
    n_steps: int = 30000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep parameter a to map the transition to/from chaos.

    For the Dadras system, a controls the linear damping on x.
    Increasing a increases dissipation, potentially suppressing chaos.
    """
    a_values = np.linspace(1.0, 6.0, n_a)
    lyapunov_exps = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0,
                "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
            },
        )
        sim = DadrasSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  a={a:.2f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def generate_lyapunov_vs_e_data(
    n_e: int = 30,
    n_steps: int = 30000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep parameter e to map the transition to/from chaos.

    For the Dadras system, e controls the linear damping on z.
    Increasing e increases dissipation, potentially suppressing chaos.
    """
    e_values = np.linspace(4.0, 15.0, n_e)
    lyapunov_exps = []

    for i, e_val in enumerate(e_values):
        config = SimulationConfig(
            domain=_DADRAS_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": e_val,
                "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
            },
        )
        sim = DadrasSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  e={e_val:.2f}: Lyapunov={lam:.4f}")

    return {
        "e": e_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_dadras_rediscovery(
    output_dir: str | Path = "output/rediscovery/dadras",
    n_iterations: int = 40,
) -> dict:
    """Run the full Dadras attractor rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep a to map chaos transition (Lyapunov exponent)
    3. Sweep e to map chaos transition
    4. Compute Lyapunov at classic parameters
    5. Fixed point analysis
    6. Attractor statistics

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "dadras",
        "targets": {
            "ode_x": "dx/dt = y - a*x + b*y*z",
            "ode_y": "dy/dt = c*y - x*z + z",
            "ode_z": "dz/dt = d*x*y - e*z",
            "chaos_regime": "a=3, b=2.7, c=1.7, d=2, e=9 (classic chaotic)",
            "divergence": "div = -a + c - e = -10.3 (dissipative)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Dadras trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.005)

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
            "true_c": ode_data["c"],
            "true_d": ode_data["d"],
            "true_e": ode_data["e"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: a-parameter sweep ---
    logger.info("Part 2: Mapping chaos transition (a sweep)...")
    a_data = generate_lyapunov_vs_a_data(n_a=30, n_steps=30000, dt=0.005)

    lam_a = a_data["lyapunov_exponent"]
    a_vals = a_data["a"]
    n_chaotic_a = int(np.sum(lam_a > 0.01))
    n_stable_a = int(np.sum(lam_a < -0.01))

    results["a_sweep"] = {
        "n_a_values": len(a_vals),
        "n_chaotic": n_chaotic_a,
        "n_stable": n_stable_a,
        "a_range": [float(a_vals[0]), float(a_vals[-1])],
        "max_lyapunov": float(np.max(lam_a)),
        "min_lyapunov": float(np.min(lam_a)),
    }
    logger.info(f"  a sweep: {n_chaotic_a} chaotic, {n_stable_a} stable")

    # --- Part 3: e-parameter sweep ---
    logger.info("Part 3: Mapping chaos transition (e sweep)...")
    e_data = generate_lyapunov_vs_e_data(n_e=30, n_steps=30000, dt=0.005)

    lam_e = e_data["lyapunov_exponent"]
    e_vals = e_data["e"]
    n_chaotic_e = int(np.sum(lam_e > 0.01))
    n_stable_e = int(np.sum(lam_e < -0.01))

    results["e_sweep"] = {
        "n_e_values": len(e_vals),
        "n_chaotic": n_chaotic_e,
        "n_stable": n_stable_e,
        "e_range": [float(e_vals[0]), float(e_vals[-1])],
        "max_lyapunov": float(np.max(lam_e)),
        "min_lyapunov": float(np.min(lam_e)),
    }
    logger.info(f"  e sweep: {n_chaotic_e} chaotic, {n_stable_e} stable")

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info("Part 4: Lyapunov exponent at classic chaotic parameters...")
    config_classic = SimulationConfig(
        domain=_DADRAS_DOMAIN,
        dt=0.005,
        n_steps=50000,
        parameters={"a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0},
    )
    sim_classic = DadrasSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "a": 3.0,
        "b": 2.7,
        "c": 1.7,
        "d": 2.0,
        "e": 9.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
        "divergence": -3.0 + 1.7 - 9.0,
    }
    logger.info(f"  Classic Dadras Lyapunov: {lam_classic:.4f}")

    # --- Part 5: Fixed points ---
    sim_fp = DadrasSimulation(config_classic)
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

    # --- Part 6: Attractor statistics ---
    logger.info("Part 6: Computing attractor statistics...")
    config_stats = SimulationConfig(
        domain=_DADRAS_DOMAIN,
        dt=0.005,
        n_steps=20000,
        parameters={"a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0},
    )
    sim_stats = DadrasSimulation(config_stats)
    traj_stats = sim_stats.compute_trajectory_statistics(
        n_steps=15000, n_transient=5000
    )
    results["trajectory_statistics"] = traj_stats
    logger.info(
        f"  x: mean={traj_stats['x_mean']:.3f}, std={traj_stats['x_std']:.3f}"
    )
    logger.info(
        f"  y: mean={traj_stats['y_mean']:.3f}, std={traj_stats['y_std']:.3f}"
    )
    logger.info(
        f"  z: mean={traj_stats['z_mean']:.3f}, std={traj_stats['z_std']:.3f}"
    )

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
        a=a_data["a"],
        lyapunov_exponent=a_data["lyapunov_exponent"],
    )
    np.savez(
        output_path / "lyapunov_vs_e.npz",
        e=e_data["e"],
        lyapunov_exponent=e_data["lyapunov_exponent"],
    )

    return results
