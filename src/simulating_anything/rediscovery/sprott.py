"""Sprott system rediscovery.

Targets:
- SINDy recovery of Sprott-B ODEs: dx=yz, dy=x-y, dz=1-xy
- Positive Lyapunov exponent confirming chaos
- Attractor characterization (bounded strange attractor)
- Comparison of Lyapunov exponents across Sprott variants
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.sprott import SprottSimulation

logger = logging.getLogger(__name__)


def generate_ode_data(
    system: str = "B",
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate a single Sprott trajectory for SINDy ODE recovery.

    Uses small dt and many steps for accurate derivative estimation.
    """
    sim = SprottSimulation.create(system=system, dt=dt, n_steps=n_steps)
    sim.reset()

    # Skip transient to land on the attractor
    for _ in range(2000):
        sim.step()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "system": system,
    }


def generate_lyapunov_comparison(
    systems: list[str] | None = None,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, float | list]:
    """Compute Lyapunov exponents for multiple Sprott systems.

    This characterizes which systems are chaotic (positive Lyapunov)
    and allows comparison of their chaoticity.
    """
    if systems is None:
        systems = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    lyapunov_exps = {}

    for system in systems:
        try:
            sim = SprottSimulation.create(system=system, dt=dt, n_steps=n_steps)
            sim.reset()

            # Skip transient
            for _ in range(5000):
                sim.step()

            # Check if trajectory is bounded before computing Lyapunov
            state = sim.observe()
            if not np.all(np.isfinite(state)) or np.linalg.norm(state) > 1e6:
                logger.warning(f"  Sprott-{system}: trajectory diverged, skipping")
                lyapunov_exps[system] = float("nan")
                continue

            lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
            lyapunov_exps[system] = float(lam)
            logger.info(f"  Sprott-{system}: Lyapunov = {lam:.4f}")
        except Exception as e:
            logger.warning(f"  Sprott-{system} failed: {e}")
            lyapunov_exps[system] = float("nan")

    return {
        "systems": systems,
        "lyapunov_exponents": lyapunov_exps,
    }


def characterize_attractor(
    system: str = "B",
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, float]:
    """Characterize the strange attractor of a Sprott system.

    Computes bounding box, mean, std, and confirms boundedness.
    """
    sim = SprottSimulation.create(system=system, dt=dt, n_steps=n_steps)
    sim.reset()

    # Skip transient
    for _ in range(5000):
        sim.step()

    # Collect attractor points
    states = []
    for _ in range(n_steps):
        states.append(sim.step().copy())

    states = np.array(states)

    return {
        "system": system,
        "x_range": [float(np.min(states[:, 0])), float(np.max(states[:, 0]))],
        "y_range": [float(np.min(states[:, 1])), float(np.max(states[:, 1]))],
        "z_range": [float(np.min(states[:, 2])), float(np.max(states[:, 2]))],
        "mean": states.mean(axis=0).tolist(),
        "std": states.std(axis=0).tolist(),
        "is_bounded": bool(np.all(np.isfinite(states))),
        "max_norm": float(np.max(np.linalg.norm(states, axis=1))),
    }


def run_sprott_rediscovery(
    output_dir: str | Path = "output/rediscovery/sprott",
    n_iterations: int = 40,
) -> dict:
    """Run the full Sprott system rediscovery.

    1. Generate Sprott-B trajectory for SINDy ODE recovery
    2. Compute Lyapunov exponent for Sprott-B (verify chaos)
    3. Characterize the Sprott-B attractor
    4. Compare Lyapunov exponents across Sprott systems A-J
    5. Verify fixed points of Sprott-B

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "sprott",
        "targets": {
            "ode_x": "dx/dt = y*z",
            "ode_y": "dy/dt = x - y",
            "ode_z": "dz/dt = 1 - x*y",
            "chaos": "Positive Lyapunov exponent for Sprott-B",
            "attractor": "Bounded strange attractor",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Sprott-B trajectory for SINDy...")
    ode_data = generate_ode_data(system="B", n_steps=10000, dt=0.01)

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
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Lyapunov exponent for Sprott-B ---
    logger.info("Part 2: Computing Sprott-B Lyapunov exponent...")
    sim_b = SprottSimulation.create(system="B", dt=0.01, n_steps=50000)
    sim_b.reset()

    # Skip transient
    for _ in range(10000):
        sim_b.step()

    lam_b = sim_b.estimate_lyapunov(n_steps=50000, dt=0.01)
    results["sprott_b_lyapunov"] = {
        "lyapunov_exponent": float(lam_b),
        "is_chaotic": bool(lam_b > 0),
    }
    logger.info(f"  Sprott-B Lyapunov: {lam_b:.4f} (chaotic: {lam_b > 0})")

    # --- Part 3: Attractor characterization ---
    logger.info("Part 3: Characterizing Sprott-B attractor...")
    attractor = characterize_attractor(system="B", n_steps=50000, dt=0.01)
    results["attractor"] = attractor
    logger.info(
        f"  Attractor bounded: {attractor['is_bounded']}, "
        f"max_norm: {attractor['max_norm']:.2f}"
    )

    # --- Part 4: Cross-system Lyapunov comparison ---
    logger.info("Part 4: Comparing Lyapunov exponents across Sprott systems...")
    comparison = generate_lyapunov_comparison(
        systems=["A", "B", "C", "D", "E"],
        n_steps=20000,
        dt=0.01,
    )
    results["cross_system_comparison"] = comparison

    n_chaotic = sum(
        1 for v in comparison["lyapunov_exponents"].values()
        if not np.isnan(v) and v > 0
    )
    logger.info(f"  {n_chaotic}/{len(comparison['systems'])} systems are chaotic")

    # --- Part 5: Fixed points ---
    logger.info("Part 5: Verifying Sprott-B fixed points...")
    sim_fp = SprottSimulation.create(system="B", dt=0.01)
    sim_fp.reset()
    fps = sim_fp.sprott_b_fixed_points

    fp_results = []
    for fp in fps:
        derivs = sim_fp._derivatives(fp)
        deriv_norm = float(np.linalg.norm(derivs))
        fp_results.append({
            "point": fp.tolist(),
            "derivative_norm": deriv_norm,
        })
        logger.info(
            f"  FP: [{fp[0]:.1f}, {fp[1]:.1f}, {fp[2]:.1f}], "
            f"|deriv|={deriv_norm:.2e}"
        )

    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": fp_results,
    }

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

    return results
