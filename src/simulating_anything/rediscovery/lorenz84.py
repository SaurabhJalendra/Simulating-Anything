"""Lorenz-84 atmospheric model rediscovery.

Targets:
- Hadley circulation fixed point x* = F (for G=0, low forcing)
- Chaos onset as F increases (Lyapunov exponent transition)
- SINDy recovery of Lorenz-84 ODEs
- Quasi-periodic route to chaos characterization
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.lorenz84 import Lorenz84Simulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use LORENZ_84 domain enum
_DOMAIN = Domain.LORENZ_84


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 0.25,
    b: float = 4.0,
    F: float = 8.0,
    G: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate a single Lorenz-84 trajectory for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a, "b": b, "F": F, "G": G,
            "x_0": 1.0, "y_0": 0.5, "z_0": 0.5,
        },
    )
    sim = Lorenz84Simulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "a": a,
        "b": b,
        "F": F,
        "G": G,
    }


def generate_chaos_transition_data(
    n_F: int = 40,
    n_steps: int = 20000,
    dt: float = 0.01,
    a: float = 0.25,
    b: float = 4.0,
    G: float = 1.0,
) -> dict[str, np.ndarray]:
    """Sweep F to map the transition to chaos in the Lorenz-84 model.

    For each F value, compute the Lyapunov exponent and measure trajectory
    statistics (amplitude, variance).
    """
    F_values = np.linspace(1.0, 10.0, n_F)
    lyapunov_exps = []
    x_means = []
    x_stds = []
    attractor_types = []

    for i, F in enumerate(F_values):
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a, "b": b, "F": F, "G": G,
                "x_0": 1.0, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim = Lorenz84Simulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        # Estimate Lyapunov exponent
        lam = sim.compute_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        # Collect trajectory statistics
        x_vals = []
        for _ in range(5000):
            state = sim.step()
            x_vals.append(state[0])

        x_means.append(np.mean(x_vals))
        x_stds.append(np.std(x_vals))

        # Classify attractor
        if lam < -0.01:
            atype = "fixed_point"
        elif lam < 0.01:
            atype = "periodic_or_quasiperiodic"
        else:
            atype = "chaotic"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(f"  F={F:.2f}: Lyapunov={lam:.4f}, type={atype}")

    return {
        "F": F_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "x_mean": np.array(x_means),
        "x_std": np.array(x_stds),
        "attractor_type": np.array(attractor_types),
    }


def generate_hadley_verification_data(
    n_F: int = 20,
    n_steps: int = 10000,
    dt: float = 0.01,
    a: float = 0.25,
    b: float = 4.0,
) -> dict[str, np.ndarray]:
    """Verify the Hadley fixed point x* = F for G=0 across F values.

    The Hadley fixed point (F, 0, 0) is only linearly stable for F < 1
    (the eigenvalues of the Jacobian at this point have real part F-1).
    For F >= 1, the system undergoes a Hopf bifurcation and converges to
    a different attractor.

    We sweep F in [0.1, 0.95] where the fixed point is stable.
    """
    F_values = np.linspace(0.1, 0.95, n_F)
    x_final = []
    y_final = []
    z_final = []

    for F in F_values:
        config = SimulationConfig(
            domain=_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a, "b": b, "F": F, "G": 0.0,
                "x_0": F + 0.1, "y_0": 0.1, "z_0": 0.1,
            },
        )
        sim = Lorenz84Simulation(config)
        sim.reset()

        # Run to settle to fixed point
        for _ in range(n_steps):
            sim.step()

        state = sim.observe()
        x_final.append(state[0])
        y_final.append(state[1])
        z_final.append(state[2])

    return {
        "F": F_values,
        "x_final": np.array(x_final),
        "y_final": np.array(y_final),
        "z_final": np.array(z_final),
    }


def run_lorenz84_rediscovery(
    output_dir: str | Path = "output/rediscovery/lorenz84",
    n_iterations: int = 40,
) -> dict:
    """Run the full Lorenz-84 rediscovery pipeline.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep F to map chaos transition (Lyapunov exponent)
    3. Verify Hadley fixed point x* = F for G=0
    4. Compute Lyapunov at classic chaotic parameters

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "lorenz_84",
        "targets": {
            "ode_x": "dx/dt = -y^2 - z^2 - a*x + a*F",
            "ode_y": "dy/dt = x*y - b*x*z - y + G",
            "ode_z": "dz/dt = b*x*y + x*z - z",
            "hadley_fp": "x* = F, y* = 0, z* = 0 (for G=0)",
            "chaos_onset": "Positive Lyapunov exponent at large F",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Lorenz-84 trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=5000, dt=0.01)

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
            "true_F": ode_data["F"],
            "true_G": ode_data["G"],
        }
        if sindy_discoveries:
            best = sindy_discoveries[0]
            results["sindy_ode"]["best"] = best.expression
            results["sindy_ode"]["best_r2"] = best.evidence.fit_r_squared
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Chaos transition sweep ---
    logger.info("Part 2: Mapping chaos transition (F sweep)...")
    chaos_data = generate_chaos_transition_data(n_F=40, n_steps=20000, dt=0.01)

    n_chaotic = int(np.sum(chaos_data["attractor_type"] == "chaotic"))
    n_fixed = int(np.sum(chaos_data["attractor_type"] == "fixed_point"))
    n_periodic = int(
        np.sum(chaos_data["attractor_type"] == "periodic_or_quasiperiodic")
    )
    logger.info(
        f"  Found {n_chaotic} chaotic, {n_fixed} fixed-point, "
        f"{n_periodic} periodic/QP regimes"
    )

    results["chaos_transition"] = {
        "n_F_values": len(chaos_data["F"]),
        "n_chaotic": n_chaotic,
        "n_fixed_point": n_fixed,
        "n_periodic": n_periodic,
        "F_range": [float(chaos_data["F"][0]), float(chaos_data["F"][-1])],
    }

    # Find approximate critical F for chaos onset
    mask_positive = chaos_data["lyapunov_exponent"] > 0.01
    if np.any(mask_positive):
        F_c_approx = float(chaos_data["F"][np.argmax(mask_positive)])
        results["chaos_transition"]["F_c_approx"] = F_c_approx
        logger.info(f"  Approximate chaos onset F_c: {F_c_approx:.2f}")

    # --- Part 3: Hadley fixed point verification ---
    logger.info("Part 3: Verifying Hadley fixed point x* = F for G=0...")
    hadley_data = generate_hadley_verification_data(n_F=20, n_steps=10000, dt=0.01)

    # Compare x_final to F
    errors = np.abs(hadley_data["x_final"] - hadley_data["F"])
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    # Check y and z are near zero
    y_max = float(np.max(np.abs(hadley_data["y_final"])))
    z_max = float(np.max(np.abs(hadley_data["z_final"])))

    results["hadley_verification"] = {
        "n_F_values": len(hadley_data["F"]),
        "F_range": [float(hadley_data["F"][0]), float(hadley_data["F"][-1])],
        "mean_error_x_vs_F": mean_error,
        "max_error_x_vs_F": max_error,
        "max_abs_y": y_max,
        "max_abs_z": z_max,
        "verified": mean_error < 0.1,
    }
    logger.info(
        f"  Hadley FP: mean |x*-F| = {mean_error:.6f}, "
        f"max |y| = {y_max:.6f}, max |z| = {z_max:.6f}"
    )

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info("Part 4: Lyapunov exponent at classic chaotic parameters...")
    config_classic = SimulationConfig(
        domain=_DOMAIN,
        dt=0.005,
        n_steps=50000,
        parameters={
            "a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0,
        },
    )
    sim_classic = Lorenz84Simulation(config_classic)
    sim_classic.reset()

    # Skip transient
    for _ in range(10000):
        sim_classic.step()

    lam_classic = sim_classic.compute_lyapunov(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "a": 0.25,
        "b": 4.0,
        "F": 8.0,
        "G": 1.0,
        "lyapunov_exponent": float(lam_classic),
        "is_chaotic": lam_classic > 0.01,
    }
    logger.info(f"  Classic Lorenz-84 Lyapunov: {lam_classic:.4f}")

    # Fixed points
    sim_fp = Lorenz84Simulation(config_classic)
    sim_fp.reset()
    fps = sim_fp.find_fixed_points()
    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": [fp.tolist() for fp in fps],
    }
    logger.info(f"  Found {len(fps)} fixed points")

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data arrays
    np.savez(
        output_path / "ode_data.npz",
        states=ode_data["states"],
    )
    np.savez(
        output_path / "chaos_transition.npz",
        F=chaos_data["F"],
        lyapunov_exponent=chaos_data["lyapunov_exponent"],
        x_mean=chaos_data["x_mean"],
        x_std=chaos_data["x_std"],
    )
    np.savez(
        output_path / "hadley_data.npz",
        F=hadley_data["F"],
        x_final=hadley_data["x_final"],
        y_final=hadley_data["y_final"],
        z_final=hadley_data["z_final"],
    )

    return results
