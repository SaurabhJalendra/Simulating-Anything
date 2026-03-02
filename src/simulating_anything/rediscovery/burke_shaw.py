"""Burke-Shaw chaotic system rediscovery.

Targets:
- SINDy recovery of Burke-Shaw ODEs: x'=-s(x+y), y'=-y-sxz, z'=sxy+v
- Lyapunov exponent estimation for chaos confirmation
- Fixed point analysis and verification
- Parameter sweep of s for chaos transition
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.burke_shaw import BurkeShawSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    n_steps: int = 5000,
    dt: float = 0.005,
    s: float = 10.0,
    v: float = 4.272,
) -> dict[str, np.ndarray]:
    """Generate a single Burke-Shaw trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.BURKE_SHAW,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "s": s,
            "v": v,
            "x_0": 1.0,
            "y_0": 0.0,
            "z_0": 0.0,
        },
    )
    sim = BurkeShawSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "s": s,
        "v": v,
    }


def generate_lyapunov_vs_s_data(
    n_s: int = 30,
    n_steps: int = 30000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep parameter s to map the chaos transition.

    Focuses on s in [1.0, 15.0] to capture the transition to chaos.
    """
    s_values = np.linspace(1.0, 15.0, n_s)
    lyapunov_exps = []

    for i, s in enumerate(s_values):
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=dt,
            n_steps=n_steps,
            parameters={"s": s, "v": 4.272, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = BurkeShawSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  s={s:.2f}: Lyapunov={lam:.4f}")

    return {
        "s": s_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def generate_fixed_point_data(
    s_values: np.ndarray | None = None,
) -> dict[str, list]:
    """Compute fixed points at several s values.

    Fixed points are at x = +/- sqrt(v/s), y = -x, z = 1/s.
    As s increases, the fixed points move closer to the origin.
    """
    if s_values is None:
        s_values = np.array([2.0, 5.0, 8.0, 10.0, 13.0])

    results = []
    for s in s_values:
        config = SimulationConfig(
            domain=Domain.BURKE_SHAW,
            dt=0.005,
            n_steps=100,
            parameters={"s": s, "v": 4.272},
        )
        sim = BurkeShawSimulation(config)
        sim.reset()
        fps = sim.fixed_points
        results.append({
            "s": float(s),
            "n_fixed_points": len(fps),
            "fixed_points": [fp.tolist() for fp in fps],
        })
        logger.info(f"  s={s:.2f}: {len(fps)} fixed points found")

    return {"data": results, "s_values": s_values.tolist()}


def run_burke_shaw_rediscovery(
    output_dir: str | Path = "output/rediscovery/burke_shaw",
    n_iterations: int = 40,
) -> dict:
    """Run the full Burke-Shaw system rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep s to map chaos transition (Lyapunov exponent)
    3. Verify fixed points analytically
    4. Compute Lyapunov at standard parameters
    5. Trajectory statistics on the attractor

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "burke_shaw",
        "targets": {
            "ode_x": "dx/dt = -s*(x + y)",
            "ode_y": "dy/dt = -y - s*x*z",
            "ode_z": "dz/dt = s*x*y + v",
            "classic_params": "s=10.0, v=4.272",
            "fixed_points": "x=+/-sqrt(v/s), y=-x, z=1/s",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Burke-Shaw trajectory for SINDy...")
    ode_data = generate_trajectory_data(n_steps=10000, dt=0.005)

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
            "true_s": ode_data["s"],
            "true_v": ode_data["v"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Lyapunov vs s sweep ---
    logger.info("Part 2: Mapping chaos transition (s sweep)...")
    lyap_data = generate_lyapunov_vs_s_data(n_s=30, n_steps=30000, dt=0.005)

    # Find approximate chaos onset (first positive Lyapunov)
    lam = lyap_data["lyapunov_exponent"]
    s_vals = lyap_data["s"]
    n_chaotic = int(np.sum(lam > 0.01))
    n_stable = int(np.sum(lam < -0.01))

    results["chaos_transition"] = {
        "n_s_values": len(s_vals),
        "n_chaotic": n_chaotic,
        "n_stable": n_stable,
        "s_range": [float(s_vals[0]), float(s_vals[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }

    # Detect zero crossings for chaos onset
    zero_crossings = []
    for j in range(len(lam) - 1):
        if lam[j] <= 0 and lam[j + 1] > 0:
            frac = -lam[j] / (lam[j + 1] - lam[j])
            s_cross = s_vals[j] + frac * (s_vals[j + 1] - s_vals[j])
            zero_crossings.append(float(s_cross))
    results["chaos_transition"]["zero_crossings"] = zero_crossings

    if zero_crossings:
        logger.info(f"  Lyapunov zero crossings at s = {zero_crossings}")

    # --- Part 3: Fixed point analysis ---
    logger.info("Part 3: Computing fixed points...")
    fp_data = generate_fixed_point_data()
    results["fixed_points"] = fp_data

    # Verify fixed points at classic parameters
    config_classic = SimulationConfig(
        domain=Domain.BURKE_SHAW,
        dt=0.005,
        n_steps=10000,
        parameters={"s": 10.0, "v": 4.272},
    )
    sim_fp = BurkeShawSimulation(config_classic)
    sim_fp.reset()
    fps = sim_fp.fixed_points
    fp_derivs = []
    for fp in fps:
        derivs = sim_fp._derivatives(fp)
        fp_derivs.append(float(np.linalg.norm(derivs)))
    results["fixed_point_verification"] = {
        "n_fixed_points": len(fps),
        "points": [fp.tolist() for fp in fps],
        "derivative_norms": fp_derivs,
    }
    for i, fp in enumerate(fps):
        logger.info(
            f"  FP{i+1}: [{fp[0]:.6f}, {fp[1]:.6f}, {fp[2]:.6f}], "
            f"|deriv|={fp_derivs[i]:.2e}"
        )

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info("Part 4: Lyapunov exponent at classic chaotic parameters...")
    config_lyap = SimulationConfig(
        domain=Domain.BURKE_SHAW,
        dt=0.005,
        n_steps=50000,
        parameters={"s": 10.0, "v": 4.272},
    )
    sim_lyap = BurkeShawSimulation(config_lyap)
    sim_lyap.reset()
    for _ in range(10000):
        sim_lyap.step()
    lam_classic = sim_lyap.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "s": 10.0,
        "v": 4.272,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Burke-Shaw Lyapunov: {lam_classic:.4f}")

    # --- Part 5: Trajectory statistics ---
    logger.info("Part 5: Trajectory statistics on the attractor...")
    sim_stats = BurkeShawSimulation(config_lyap)
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
        output_path / "lyapunov_vs_s.npz",
        s=lyap_data["s"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )

    return results
