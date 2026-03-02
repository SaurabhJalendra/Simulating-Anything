"""Rossler hyperchaotic system rediscovery.

Targets:
- Two positive Lyapunov exponents (hyperchaos verification)
- Lyapunov spectrum computation and Kaplan-Yorke dimension
- SINDy recovery of 4D ODEs
- Attractor analysis: boundedness, state statistics
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.rossler_hyperchaos import (
    RosslerHyperchaosSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    n_steps: int = 10000,
    dt: float = 0.005,
    a: float = 0.25,
    b: float = 3.0,
    c: float = 0.5,
    d: float = 0.05,
) -> dict[str, np.ndarray | float]:
    """Generate a single 4D hyperchaotic trajectory for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.ROSSLER_HYPERCHAOS,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a, "b": b, "c": c, "d": d,
            "x_0": -10.0, "y_0": -6.0, "z_0": 0.0, "w_0": 10.0,
        },
    )
    sim = RosslerHyperchaosSimulation(config)
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


def generate_lyapunov_data(
    n_d_values: int = 15,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep parameter d to map transition from chaos to hyperchaos.

    For a=0.25, b=3.0, c=0.5:
    - d=0: standard 3D Rossler behavior (at most 1 positive LE)
    - d>~0.03: second Lyapunov exponent becomes positive (hyperchaos)
    """
    d_values = np.linspace(0.0, 0.15, n_d_values)
    max_le = []
    second_le = []
    n_positive_list = []

    for i, d_val in enumerate(d_values):
        config = SimulationConfig(
            domain=Domain.ROSSLER_HYPERCHAOS,
            dt=dt,
            n_steps=1000,
            parameters={
                "a": 0.25, "b": 3.0, "c": 0.5, "d": d_val,
                "x_0": -10.0, "y_0": -6.0, "z_0": 0.0, "w_0": 10.0,
            },
        )
        sim = RosslerHyperchaosSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        spectrum = sim.estimate_lyapunov_spectrum(n_steps=40000, dt=dt)
        max_le.append(float(spectrum[0]))
        second_le.append(float(spectrum[1]))
        n_positive = int(np.sum(spectrum > 0.001))
        n_positive_list.append(n_positive)

        if (i + 1) % 5 == 0:
            logger.info(
                f"  d={d_val:.3f}: LE1={spectrum[0]:.4f}, "
                f"LE2={spectrum[1]:.4f}, n_positive={n_positive}"
            )

    return {
        "d": d_values,
        "max_lyapunov": np.array(max_le),
        "second_lyapunov": np.array(second_le),
        "n_positive": np.array(n_positive_list),
    }


def run_rossler_hyperchaos_rediscovery(
    output_dir: str | Path = "output/rediscovery/rossler_hyperchaos",
    n_iterations: int = 40,
) -> dict:
    """Run the full Rossler hyperchaotic system rediscovery.

    1. Generate trajectory for SINDy ODE recovery
    2. Compute full Lyapunov spectrum at classic parameters
    3. Verify hyperchaos (two positive exponents)
    4. Compute Kaplan-Yorke dimension
    5. Sweep d parameter to map chaos-to-hyperchaos transition

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "rossler_hyperchaos",
        "targets": {
            "ode_x": "dx/dt = -(y + z)",
            "ode_y": "dy/dt = x + a*y + w",
            "ode_z": "dz/dt = b + x*z",
            "ode_w": "dw/dt = -c*z + d*w",
            "hyperchaos": "Two positive Lyapunov exponents",
            "params": "a=0.25, b=3.0, c=0.5, d=0.05",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating 4D hyperchaotic trajectory for SINDy...")
    ode_data = generate_trajectory_data(n_steps=10000, dt=0.005)

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

    # --- Part 2: Lyapunov spectrum at classic parameters ---
    logger.info("Part 2: Computing full Lyapunov spectrum...")
    config_classic = SimulationConfig(
        domain=Domain.ROSSLER_HYPERCHAOS,
        dt=0.005,
        n_steps=1000,
        parameters={
            "a": 0.25, "b": 3.0, "c": 0.5, "d": 0.05,
            "x_0": -10.0, "y_0": -6.0, "z_0": 0.0, "w_0": 10.0,
        },
    )
    sim_classic = RosslerHyperchaosSimulation(config_classic)
    sim_classic.reset()

    spectrum = sim_classic.estimate_lyapunov_spectrum(
        n_steps=60000, dt=0.005, n_transient=10000
    )
    n_positive = int(np.sum(spectrum > 0.001))

    results["lyapunov_spectrum"] = {
        "exponents": spectrum.tolist(),
        "n_positive": n_positive,
        "is_hyperchaotic": bool(n_positive >= 2),
        "sum_exponents": float(np.sum(spectrum)),
    }
    logger.info(f"  Lyapunov spectrum: {spectrum}")
    logger.info(f"  Positive exponents: {n_positive}")
    logger.info(f"  Hyperchaotic: {n_positive >= 2}")

    # --- Part 3: Kaplan-Yorke dimension ---
    logger.info("Part 3: Kaplan-Yorke dimension...")
    d_ky = sim_classic.kaplan_yorke_dimension(spectrum=spectrum)
    results["kaplan_yorke"] = {
        "dimension": d_ky,
        "expected_range": [3.0, 4.0],  # should be > 3 for hyperchaos
    }
    logger.info(f"  Kaplan-Yorke dimension: {d_ky:.3f}")

    # --- Part 4: Attractor statistics ---
    logger.info("Part 4: Attractor analysis...")
    sim_att = RosslerHyperchaosSimulation(config_classic)
    sim_att.reset()

    # Skip transient
    for _ in range(10000):
        sim_att.step()

    # Collect attractor data
    att_states = []
    for _ in range(20000):
        att_states.append(sim_att.step().copy())
    att_states = np.array(att_states)

    results["attractor"] = {
        "is_bounded": bool(np.all(np.isfinite(att_states))),
        "state_mean": att_states.mean(axis=0).tolist(),
        "state_std": att_states.std(axis=0).tolist(),
        "state_min": att_states.min(axis=0).tolist(),
        "state_max": att_states.max(axis=0).tolist(),
    }
    logger.info(f"  Bounded: {results['attractor']['is_bounded']}")
    logger.info(f"  State ranges: min={att_states.min(axis=0)}, max={att_states.max(axis=0)}")

    # --- Part 5: d-parameter sweep (chaos to hyperchaos transition) ---
    logger.info("Part 5: Sweeping d parameter for hyperchaos transition...")
    sweep_data = generate_lyapunov_data(n_d_values=10, dt=0.005)

    # Find approximate transition point where second LE becomes positive
    mask_hyper = sweep_data["n_positive"] >= 2
    if np.any(mask_hyper):
        d_transition = float(sweep_data["d"][np.argmax(mask_hyper)])
    else:
        d_transition = None

    results["d_sweep"] = {
        "n_d_values": len(sweep_data["d"]),
        "d_range": [float(sweep_data["d"][0]), float(sweep_data["d"][-1])],
        "n_hyperchaotic": int(np.sum(mask_hyper)),
        "d_transition_approx": d_transition,
    }
    logger.info(f"  Hyperchaotic regimes: {int(np.sum(mask_hyper))} / {len(sweep_data['d'])}")
    if d_transition is not None:
        logger.info(f"  Approximate hyperchaos onset d: {d_transition:.3f}")

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
        output_path / "sweep_data.npz",
        d=sweep_data["d"],
        max_lyapunov=sweep_data["max_lyapunov"],
        second_lyapunov=sweep_data["second_lyapunov"],
        n_positive=sweep_data["n_positive"],
    )

    return results
