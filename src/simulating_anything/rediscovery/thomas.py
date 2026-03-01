"""Thomas cyclically symmetric attractor rediscovery.

Targets:
- SINDy recovery of Thomas ODEs: x'=sin(y)-b*x, y'=sin(z)-b*y, z'=sin(x)-b*z
- Critical dissipation b_c ~ 0.208186 for chaos onset
- Cyclic symmetry verification: (x,y,z) -> (y,z,x)
- Lyapunov exponent as a function of b
- Fixed points at (x,x,x) where sin(x) = b*x
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.thomas import ThomasSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    b: float = 0.208186,
) -> dict[str, np.ndarray]:
    """Generate a single Thomas trajectory for SINDy ODE recovery.

    Uses parameters in the chaotic regime by default.
    """
    config = SimulationConfig(
        domain=Domain.THOMAS,
        dt=dt,
        n_steps=n_steps,
        parameters={"b": b, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
    )
    sim = ThomasSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "b": b,
    }


def generate_lyapunov_vs_b_data(
    n_b: int = 30,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep dissipation b to map the transition from chaos to fixed point.

    Focuses on b in [0.05, 0.40] to capture the critical transition near
    b_c ~ 0.208186.
    """
    b_values = np.linspace(0.05, 0.40, n_b)
    lyapunov_exps = []

    for i, b in enumerate(b_values):
        config = SimulationConfig(
            domain=Domain.THOMAS,
            dt=dt,
            n_steps=n_steps,
            parameters={"b": b, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = ThomasSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  b={b:.4f}: Lyapunov={lam:.4f}")

    return {
        "b": b_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def generate_fixed_point_data(
    b_values: np.ndarray | None = None,
) -> dict[str, list]:
    """Compute fixed points at several dissipation values.

    Fixed points satisfy sin(x) = b*x with x = y = z.
    For small b, there are many fixed points; for large b, only the origin.
    """
    if b_values is None:
        b_values = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])

    results = []
    for b in b_values:
        config = SimulationConfig(
            domain=Domain.THOMAS,
            dt=0.01,
            n_steps=100,
            parameters={"b": b},
        )
        sim = ThomasSimulation(config)
        sim.reset()
        fps = sim.find_fixed_points()
        results.append({
            "b": float(b),
            "n_fixed_points": len(fps),
            "fixed_points": [fp.tolist() for fp in fps],
        })
        logger.info(f"  b={b:.2f}: {len(fps)} fixed points found")

    return {"data": results, "b_values": b_values.tolist()}


def run_thomas_rediscovery(
    output_dir: str | Path = "output/rediscovery/thomas",
    n_iterations: int = 40,
) -> dict:
    """Run the full Thomas system rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep b to map chaos transition (Lyapunov exponent)
    3. Verify cyclic symmetry
    4. Compute fixed points at several b values
    5. Compute Lyapunov at the critical dissipation

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "thomas",
        "targets": {
            "ode_x": "dx/dt = sin(y) - b*x",
            "ode_y": "dy/dt = sin(z) - b*y",
            "ode_z": "dz/dt = sin(x) - b*z",
            "critical_b": "b_c ~ 0.208186",
            "symmetry": "(x,y,z) -> (y,z,x) cyclic symmetry",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Thomas trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=5000, dt=0.01, b=0.18)

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
            "true_b": ode_data["b"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Lyapunov vs b sweep ---
    logger.info("Part 2: Mapping chaos transition (b sweep)...")
    lyap_data = generate_lyapunov_vs_b_data(n_b=30, n_steps=30000, dt=0.01)

    # Find approximate critical b (last positive -> negative Lyapunov)
    lam = lyap_data["lyapunov_exponent"]
    b_vals = lyap_data["b"]
    zero_crossings = []
    for j in range(len(lam) - 1):
        if lam[j] > 0 and lam[j + 1] <= 0:
            frac = lam[j] / (lam[j] - lam[j + 1])
            b_cross = b_vals[j] + frac * (b_vals[j + 1] - b_vals[j])
            zero_crossings.append(float(b_cross))

    n_chaotic = int(np.sum(lam > 0.01))
    n_stable = int(np.sum(lam < -0.01))

    results["chaos_transition"] = {
        "n_b_values": len(b_vals),
        "n_chaotic": n_chaotic,
        "n_stable": n_stable,
        "b_range": [float(b_vals[0]), float(b_vals[-1])],
        "zero_crossings": zero_crossings,
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }
    if zero_crossings:
        logger.info(
            f"  Lyapunov zero crossings at b = {zero_crossings}"
        )
        logger.info(
            f"  First crossing: {zero_crossings[0]:.4f} (true: ~0.2082)"
        )

    # --- Part 3: Cyclic symmetry verification ---
    logger.info("Part 3: Verifying cyclic symmetry...")
    config_sym = SimulationConfig(
        domain=Domain.THOMAS,
        dt=0.01,
        n_steps=1000,
        parameters={"b": 0.18, "x_0": 1.0, "y_0": 0.5, "z_0": -0.3},
    )
    sim_sym = ThomasSimulation(config_sym)
    sim_sym.reset()
    sym_result = sim_sym.verify_cyclic_symmetry(n_steps=1000)

    results["cyclic_symmetry"] = {
        "max_deviation": sym_result["max_deviation"],
        "mean_deviation": sym_result["mean_deviation"],
        "verified": sym_result["max_deviation"] < 1e-10,
    }
    logger.info(
        f"  Symmetry max deviation: {sym_result['max_deviation']:.2e}"
    )

    # --- Part 4: Fixed point analysis ---
    logger.info("Part 4: Computing fixed points...")
    fp_data = generate_fixed_point_data()
    results["fixed_points"] = fp_data

    # --- Part 5: Lyapunov at critical b ---
    logger.info("Part 5: Lyapunov at critical dissipation...")
    config_critical = SimulationConfig(
        domain=Domain.THOMAS,
        dt=0.01,
        n_steps=50000,
        parameters={"b": 0.208186, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
    )
    sim_critical = ThomasSimulation(config_critical)
    sim_critical.reset()
    for _ in range(10000):
        sim_critical.step()
    lam_critical = sim_critical.estimate_lyapunov(n_steps=50000, dt=0.01)

    results["critical_parameters"] = {
        "b": 0.208186,
        "lyapunov_exponent": float(lam_critical),
        "near_zero": bool(abs(lam_critical) < 0.1),
    }
    logger.info(
        f"  Critical b Lyapunov: {lam_critical:.4f} (should be near 0)"
    )

    # Trajectory statistics at chaotic parameters
    config_chaotic = SimulationConfig(
        domain=Domain.THOMAS,
        dt=0.01,
        n_steps=20000,
        parameters={"b": 0.18, "x_0": 1.0, "y_0": 0.0, "z_0": 0.0},
    )
    sim_stats = ThomasSimulation(config_chaotic)
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
        output_path / "lyapunov_vs_b.npz",
        b=lyap_data["b"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )

    return results
