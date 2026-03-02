"""Qi 4D chaotic system rediscovery.

Targets:
- SINDy recovery of 4D ODEs:
    dx/dt = a*(y-x) + y*z
    dy/dt = c*x - y - x*z
    dz/dt = x*y - b*z
    dw/dt = -d*w + x*z
- Lyapunov exponent estimation (positive for chaotic regime)
- d-parameter sweep: mapping chaos transitions
- Fixed point analysis and verification
- Attractor statistics and boundedness
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.qi import QiSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use CHAOTIC_ODE as the domain enum placeholder until Domain.QI is added
_QI_DOMAIN = Domain.CHAOTIC_ODE


def generate_trajectory_data(
    n_steps: int = 10000,
    dt: float = 0.001,
    a: float = 10.0,
    b: float = 8.0 / 3.0,
    c: float = 28.0,
    d: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Generate a single 4D chaotic trajectory for SINDy ODE recovery.

    Skips a transient period before recording to ensure the system
    is on the attractor.
    """
    config = SimulationConfig(
        domain=_QI_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a, "b": b, "c": c, "d": d,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        },
    )
    sim = QiSimulation(config)
    sim.reset()

    # Skip transient before recording
    for _ in range(5000):
        sim.step()

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
    }


def generate_d_sweep_data(
    n_d_values: int = 20,
    dt: float = 0.001,
    d_min: float = 0.1,
    d_max: float = 5.0,
) -> dict[str, np.ndarray]:
    """Sweep parameter d to map chaos transitions.

    For a=10, b=8/3, c=28:
    - The d parameter controls the damping of the w variable.
    - Varying d changes the coupling strength and can induce transitions
      between stable, periodic, and chaotic regimes.
    """
    d_values = np.linspace(d_min, d_max, n_d_values)
    lyapunov_exps = []
    max_amplitudes = []
    w_amplitudes = []

    for i, d_val in enumerate(d_values):
        config = SimulationConfig(
            domain=_QI_DOMAIN,
            dt=dt,
            n_steps=1000,
            parameters={
                "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": d_val,
                "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
            },
        )
        sim = QiSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        # Estimate Lyapunov exponent
        lam = sim.estimate_lyapunov(n_steps=20000, dt=dt)
        lyapunov_exps.append(lam)

        # Measure attractor amplitude
        sim.reset()
        for _ in range(5000):
            sim.step()
        norms = []
        w_vals = []
        for _ in range(5000):
            state = sim.step()
            norms.append(np.linalg.norm(state))
            w_vals.append(abs(state[3]))
        max_amplitudes.append(np.max(norms))
        w_amplitudes.append(np.max(w_vals))

        if (i + 1) % 5 == 0:
            logger.info(
                f"  d={d_val:.2f}: Lyapunov={lam:.4f}, "
                f"max_amp={max_amplitudes[-1]:.1f}, max_|w|={w_amplitudes[-1]:.2f}"
            )

    return {
        "d": d_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "w_amplitude": np.array(w_amplitudes),
    }


def run_qi_rediscovery(
    output_dir: str | Path = "output/rediscovery/qi",
    n_iterations: int = 40,
) -> dict:
    """Run the full Qi 4D chaotic system rediscovery.

    1. Generate trajectory for SINDy ODE recovery (4 equations)
    2. Compute Lyapunov exponent at classic parameters
    3. Compute full Lyapunov spectrum and Kaplan-Yorke dimension
    4. Fixed point analysis
    5. d-parameter sweep mapping chaos transitions
    6. Attractor statistics

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "qi",
        "targets": {
            "ode_x": "dx/dt = a*(y-x) + y*z",
            "ode_y": "dy/dt = c*x - y - x*z",
            "ode_z": "dz/dt = x*y - b*z",
            "ode_w": "dw/dt = -d*w + x*z",
            "chaos_regime": "positive Lyapunov for classic parameters",
            "params": "a=10, b=8/3, c=28, d=1",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating 4D Qi trajectory for SINDy...")
    ode_data = generate_trajectory_data(n_steps=10000, dt=0.001)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x", "y", "z", "w"],
            threshold=0.05,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries
            ],
            "true_a": ode_data["a"],
            "true_b": ode_data["b"],
            "true_c": ode_data["c"],
            "true_d": ode_data["d"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Lyapunov exponent at classic parameters ---
    logger.info("Part 2: Lyapunov exponent at classic parameters...")
    config_std = SimulationConfig(
        domain=_QI_DOMAIN,
        dt=0.001,
        n_steps=1000,
        parameters={
            "a": 10.0, "b": 8.0 / 3.0, "c": 28.0, "d": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        },
    )
    sim_std = QiSimulation(config_std)
    sim_std.reset()
    for _ in range(10000):
        sim_std.step()
    lam_std = sim_std.estimate_lyapunov(n_steps=50000, dt=0.001)

    results["standard_parameters"] = {
        "a": 10.0,
        "b": 8.0 / 3.0,
        "c": 28.0,
        "d": 1.0,
        "lyapunov_exponent": float(lam_std),
        "positive": bool(lam_std > 0),
    }
    logger.info(f"  Qi system Lyapunov: {lam_std:.4f}")

    # --- Part 3: Lyapunov spectrum + Kaplan-Yorke dimension ---
    logger.info("Part 3: Computing full Lyapunov spectrum...")
    sim_spectrum = QiSimulation(config_std)
    sim_spectrum.reset()

    spectrum = sim_spectrum.estimate_lyapunov_spectrum(
        n_steps=60000, dt=0.001, n_transient=10000
    )
    n_positive = int(np.sum(spectrum > 0.001))
    d_ky = sim_spectrum.kaplan_yorke_dimension(spectrum=spectrum)

    results["lyapunov_spectrum"] = {
        "exponents": spectrum.tolist(),
        "n_positive": n_positive,
        "sum_exponents": float(np.sum(spectrum)),
    }
    results["kaplan_yorke"] = {
        "dimension": d_ky,
    }
    logger.info(f"  Lyapunov spectrum: {spectrum}")
    logger.info(f"  Positive exponents: {n_positive}")
    logger.info(f"  Kaplan-Yorke dimension: {d_ky:.3f}")

    # --- Part 4: Fixed point analysis ---
    logger.info("Part 4: Fixed point analysis...")
    sim_fp = QiSimulation(config_std)
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
            f"    FP{i+1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}, {fp[3]:.4f}], "
            f"|deriv|={np.linalg.norm(derivs):.2e}"
        )

    # --- Part 5: d-parameter sweep ---
    logger.info("Part 5: Sweeping d parameter for chaos transitions...")
    sweep_data = generate_d_sweep_data(n_d_values=15, dt=0.001)

    n_chaotic = int(np.sum(sweep_data["lyapunov_exponent"] > 0.01))
    n_stable = int(np.sum(sweep_data["lyapunov_exponent"] <= 0.01))
    logger.info(f"  Found {n_chaotic} chaotic, {n_stable} non-chaotic regimes")

    results["d_sweep"] = {
        "n_d_values": len(sweep_data["d"]),
        "n_chaotic": n_chaotic,
        "n_stable": n_stable,
        "d_range": [float(sweep_data["d"][0]), float(sweep_data["d"][-1])],
        "max_lyapunov": float(np.max(sweep_data["lyapunov_exponent"])),
        "min_lyapunov": float(np.min(sweep_data["lyapunov_exponent"])),
    }

    # --- Part 6: Attractor statistics ---
    logger.info("Part 6: Attractor analysis...")
    sim_att = QiSimulation(config_std)
    stats = sim_att.compute_trajectory_statistics(n_steps=10000, n_transient=5000)
    results["attractor_statistics"] = stats
    logger.info(f"  x_range: {stats['x_range']:.2f}, z_range: {stats['z_range']:.2f}")

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
        output_path / "d_sweep.npz",
        d=sweep_data["d"],
        lyapunov_exponent=sweep_data["lyapunov_exponent"],
        max_amplitude=sweep_data["max_amplitude"],
        w_amplitude=sweep_data["w_amplitude"],
    )

    return results
