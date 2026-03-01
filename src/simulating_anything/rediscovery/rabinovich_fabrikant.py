"""Rabinovich-Fabrikant system rediscovery.

Targets:
- SINDy recovery of RF ODEs:
    dx/dt = y(z - 1 + x^2) + gamma*x
    dy/dt = x(3z + 1 - x^2) + gamma*y
    dz/dt = -2z(alpha + xy)
- Lyapunov exponent (positive for chaotic parameters)
- Chaos transition as gamma varies
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.rabinovich_fabrikant import (
    RabinovichFabrikantSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    n_steps: int = 5000,
    dt: float = 0.005,
    alpha: float = 1.1,
    gamma: float = 0.87,
) -> dict[str, np.ndarray]:
    """Generate a single Rabinovich-Fabrikant trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.RABINOVICH_FABRIKANT,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha": alpha,
            "gamma": gamma,
            "x_0": -1.0,
            "y_0": 0.0,
            "z_0": 0.5,
        },
    )
    sim = RabinovichFabrikantSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "alpha": alpha,
        "gamma": gamma,
    }


def generate_gamma_sweep_data(
    n_gamma: int = 30,
    n_steps: int = 20000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep gamma to map chaos transitions.

    For alpha=1.1, chaotic behavior emerges for certain gamma ranges.
    """
    gamma_values = np.linspace(0.05, 1.2, n_gamma)
    lyapunov_exps = []
    max_amplitudes = []
    attractor_types = []

    for i, gamma in enumerate(gamma_values):
        config = SimulationConfig(
            domain=Domain.RABINOVICH_FABRIKANT,
            dt=dt,
            n_steps=n_steps,
            parameters={"alpha": 1.1, "gamma": gamma},
        )
        sim = RabinovichFabrikantSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        # Estimate Lyapunov exponent
        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        # Run more steps to measure amplitude
        x_vals = []
        for _ in range(5000):
            state = sim.step()
            x_vals.append(state[0])

        max_amp = np.max(np.abs(x_vals))
        max_amplitudes.append(max_amp)

        # Classify attractor
        if lam > 0.5:
            atype = "chaotic"
        elif lam > 0.01:
            atype = "weakly_chaotic"
        elif max_amp < 0.01:
            atype = "fixed_point"
        else:
            atype = "periodic_or_quasiperiodic"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  gamma={gamma:.3f}: Lyapunov={lam:.3f}, type={atype}"
            )

    return {
        "gamma": gamma_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "attractor_type": np.array(attractor_types),
    }


def run_rabinovich_fabrikant_rediscovery(
    output_dir: str | Path = "output/rediscovery/rabinovich_fabrikant",
    n_iterations: int = 40,
) -> dict:
    """Run the full Rabinovich-Fabrikant system rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep gamma to map chaos transitions
    3. Compute Lyapunov at standard chaotic parameters
    4. Verify attractor boundedness

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "rabinovich_fabrikant",
        "targets": {
            "ode_x": "dx/dt = y(z - 1 + x^2) + gamma*x",
            "ode_y": "dy/dt = x(3z + 1 - x^2) + gamma*y",
            "ode_z": "dz/dt = -2z(alpha + xy)",
            "chaos_regime": "alpha=1.1, gamma=0.87",
            "lyapunov_chaotic": "lambda > 0 for chaotic parameters",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating RF trajectory for SINDy...")
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
            "true_alpha": ode_data["alpha"],
            "true_gamma": ode_data["gamma"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Gamma sweep for chaos transition ---
    logger.info("Part 2: Mapping chaos transition (gamma sweep)...")
    sweep_data = generate_gamma_sweep_data(n_gamma=30, n_steps=20000, dt=0.005)

    n_chaotic = int(np.sum(
        (sweep_data["attractor_type"] == "chaotic")
        | (sweep_data["attractor_type"] == "weakly_chaotic")
    ))
    n_periodic = int(np.sum(
        sweep_data["attractor_type"] == "periodic_or_quasiperiodic"
    ))
    n_fixed = int(np.sum(sweep_data["attractor_type"] == "fixed_point"))
    logger.info(
        f"  Found {n_chaotic} chaotic, {n_periodic} periodic, {n_fixed} fixed"
    )

    results["gamma_sweep"] = {
        "n_gamma_values": len(sweep_data["gamma"]),
        "n_chaotic": n_chaotic,
        "n_periodic": n_periodic,
        "n_fixed_point": n_fixed,
        "gamma_range": [
            float(sweep_data["gamma"][0]),
            float(sweep_data["gamma"][-1]),
        ],
    }

    # --- Part 3: Lyapunov at classic chaotic parameters ---
    logger.info("Part 3: Lyapunov exponent at classic chaotic parameters...")
    config_classic = SimulationConfig(
        domain=Domain.RABINOVICH_FABRIKANT,
        dt=0.005,
        n_steps=50000,
        parameters={"alpha": 1.1, "gamma": 0.87},
    )
    sim_classic = RabinovichFabrikantSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "alpha": 1.1,
        "gamma": 0.87,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic RF Lyapunov: {lam_classic:.4f}")

    # --- Part 4: Attractor boundedness verification ---
    logger.info("Part 4: Verifying attractor boundedness...")
    config_bounded = SimulationConfig(
        domain=Domain.RABINOVICH_FABRIKANT,
        dt=0.005,
        n_steps=20000,
        parameters={"alpha": 1.1, "gamma": 0.87},
    )
    sim_bounded = RabinovichFabrikantSimulation(config_bounded)
    sim_bounded.reset()
    states_list = []
    for _ in range(20000):
        states_list.append(sim_bounded.step().copy())
    states_arr = np.array(states_list)

    results["attractor_bounds"] = {
        "x_range": [float(np.min(states_arr[:, 0])), float(np.max(states_arr[:, 0]))],
        "y_range": [float(np.min(states_arr[:, 1])), float(np.max(states_arr[:, 1]))],
        "z_range": [float(np.min(states_arr[:, 2])), float(np.max(states_arr[:, 2]))],
        "all_finite": bool(np.all(np.isfinite(states_arr))),
        "max_norm": float(np.max(np.linalg.norm(states_arr, axis=1))),
    }
    logger.info(
        f"  Max norm: {results['attractor_bounds']['max_norm']:.2f}, "
        f"all finite: {results['attractor_bounds']['all_finite']}"
    )

    # Fixed points
    sim_fp = RabinovichFabrikantSimulation(config_classic)
    sim_fp.reset()
    fps = sim_fp.fixed_points
    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": [fp.tolist() for fp in fps],
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
    np.savez(
        output_path / "gamma_sweep.npz",
        **{k: v for k, v in sweep_data.items() if isinstance(v, np.ndarray)},
    )

    return results
