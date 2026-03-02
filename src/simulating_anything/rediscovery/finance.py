"""Chaotic financial system rediscovery.

Targets:
- SINDy recovery of financial ODEs:
    dx/dt = z + (y - a)*x
    dy/dt = 1 - b*y - x^2
    dz/dt = -x - c*z
- Lyapunov exponent estimation (positive for chaotic regime)
- Parameter sweep mapping market stability transition (vary a)
- Fixed point analysis and eigenvalue classification
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.finance import FinanceSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use CHAOTIC_ODE as the domain enum placeholder until Domain.FINANCE is added
_FINANCE_DOMAIN = Domain.CHAOTIC_ODE


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.005,
    a: float = 1.0,
    b: float = 0.1,
    c: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate a single financial system trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=_FINANCE_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "c": c,
            "x_0": 2.0,
            "y_0": 3.0,
            "z_0": 2.0,
        },
    )
    sim = FinanceSimulation(config)
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
    }


def generate_lyapunov_vs_a_data(
    n_a: int = 30,
    n_steps: int = 30000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep savings rate a to map the stability transition.

    For the financial system, a controls the savings rate.
    Increasing a generally stabilizes the system, potentially
    suppressing chaos beyond a critical threshold.
    """
    a_values = np.linspace(0.1, 3.0, n_a)
    lyapunov_exps = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": a, "b": 0.1, "c": 1.0,
                "x_0": 2.0, "y_0": 3.0, "z_0": 2.0,
            },
        )
        sim = FinanceSimulation(config)
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


def generate_lyapunov_vs_b_data(
    n_b: int = 30,
    n_steps: int = 30000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep cost per investment b to map the stability transition.

    For the financial system, b controls the investment cost.
    Varying b changes the damping on the investment demand variable.
    """
    b_values = np.linspace(0.01, 0.5, n_b)
    lyapunov_exps = []

    for i, b_val in enumerate(b_values):
        config = SimulationConfig(
            domain=_FINANCE_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": 1.0, "b": b_val, "c": 1.0,
                "x_0": 2.0, "y_0": 3.0, "z_0": 2.0,
            },
        )
        sim = FinanceSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  b={b_val:.3f}: Lyapunov={lam:.4f}")

    return {
        "b": b_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_finance_rediscovery(
    output_dir: str | Path = "output/rediscovery/finance",
    n_iterations: int = 40,
) -> dict:
    """Run the full chaotic financial system rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep a to map market stability transition (Lyapunov exponent)
    3. Sweep b to map investment cost transition
    4. Compute Lyapunov at classic parameters
    5. Fixed point analysis with eigenvalue classification
    6. Attractor statistics

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "finance",
        "targets": {
            "ode_x": "dx/dt = z + (y - a)*x",
            "ode_y": "dy/dt = 1 - b*y - x^2",
            "ode_z": "dz/dt = -x - c*z",
            "chaos_regime": "a=1.0, b=0.1, c=1.0 (classic chaotic)",
            "divergence": "div = -a - b - c (dissipative for positive params)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating financial system trajectory for SINDy...")
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
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # --- Part 2: a-parameter sweep ---
    logger.info("Part 2: Mapping market stability transition (a sweep)...")
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

    # Detect stability transition
    mask_positive = lam_a > 0.01
    if np.any(mask_positive) and np.any(~mask_positive):
        # Find last chaotic a value as approximate transition
        chaotic_indices = np.where(mask_positive)[0]
        a_transition_approx = float(a_vals[chaotic_indices[-1]])
        results["a_sweep"]["a_transition_approx"] = a_transition_approx
        logger.info(f"  Approximate chaos-stability transition a: {a_transition_approx:.2f}")

    # --- Part 3: b-parameter sweep ---
    logger.info("Part 3: Mapping investment cost transition (b sweep)...")
    b_data = generate_lyapunov_vs_b_data(n_b=30, n_steps=30000, dt=0.005)

    lam_b = b_data["lyapunov_exponent"]
    b_vals = b_data["b"]
    n_chaotic_b = int(np.sum(lam_b > 0.01))
    n_stable_b = int(np.sum(lam_b < -0.01))

    results["b_sweep"] = {
        "n_b_values": len(b_vals),
        "n_chaotic": n_chaotic_b,
        "n_stable": n_stable_b,
        "b_range": [float(b_vals[0]), float(b_vals[-1])],
        "max_lyapunov": float(np.max(lam_b)),
        "min_lyapunov": float(np.min(lam_b)),
    }
    logger.info(f"  b sweep: {n_chaotic_b} chaotic, {n_stable_b} stable")

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info("Part 4: Lyapunov exponent at classic chaotic parameters...")
    config_classic = SimulationConfig(
        domain=_FINANCE_DOMAIN,
        dt=0.005,
        n_steps=50000,
        parameters={"a": 1.0, "b": 0.1, "c": 1.0},
    )
    sim_classic = FinanceSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "a": 1.0,
        "b": 0.1,
        "c": 1.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
        "divergence": -1.0 - 0.1 - 1.0,
    }
    logger.info(f"  Classic finance Lyapunov: {lam_classic:.4f}")

    # Also test alternative chaotic parameters
    config_alt = SimulationConfig(
        domain=_FINANCE_DOMAIN,
        dt=0.005,
        n_steps=50000,
        parameters={"a": 0.9, "b": 0.2, "c": 1.2},
    )
    sim_alt = FinanceSimulation(config_alt)
    sim_alt.reset()
    for _ in range(10000):
        sim_alt.step()
    lam_alt = sim_alt.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["alt_parameters"] = {
        "a": 0.9,
        "b": 0.2,
        "c": 1.2,
        "lyapunov_exponent": float(lam_alt),
        "positive": bool(lam_alt > 0),
    }
    logger.info(f"  Alt finance Lyapunov (a=0.9,b=0.2,c=1.2): {lam_alt:.4f}")

    # --- Part 5: Fixed points and eigenvalues ---
    logger.info("Part 5: Fixed point analysis...")
    sim_fp = FinanceSimulation(config_classic)
    sim_fp.reset()
    fps = sim_fp.fixed_points
    eig_results = sim_fp.eigenvalues_at_fixed_points()

    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": [fp.tolist() for fp in fps],
        "eigenvalue_analysis": [
            {
                "point": r["fixed_point"],
                "max_real_part": r["max_real_part"],
                "stability": r["stability"],
            }
            for r in eig_results
        ],
    }
    logger.info(f"  Fixed points: {len(fps)} found")
    for i, fp in enumerate(fps):
        derivs = sim_fp._derivatives(fp)
        logger.info(
            f"    FP{i+1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}], "
            f"|deriv|={np.linalg.norm(derivs):.2e}, "
            f"stability={eig_results[i]['stability']}"
        )

    # --- Part 6: Attractor statistics ---
    logger.info("Part 6: Computing attractor statistics...")
    config_stats = SimulationConfig(
        domain=_FINANCE_DOMAIN,
        dt=0.005,
        n_steps=20000,
        parameters={"a": 1.0, "b": 0.1, "c": 1.0},
    )
    sim_stats = FinanceSimulation(config_stats)
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
        output_path / "lyapunov_vs_b.npz",
        b=b_data["b"],
        lyapunov_exponent=b_data["lyapunov_exponent"],
    )

    return results
