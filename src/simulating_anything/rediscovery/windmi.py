"""WINDMI solar wind-magnetosphere-ionosphere model rediscovery.

Targets:
- SINDy recovery of WINDMI ODEs: x'=y, y'=z, z'=-a*z - y + b - exp(x)
- Lyapunov exponent estimation for chaos detection
- b-parameter sweep for substorm threshold mapping
- Fixed point verification: x* = ln(b)
- Jerk equation form recovery
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.windmi import WindmiSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 0.7,
    b: float = 2.5,
) -> dict[str, np.ndarray]:
    """Generate a single WINDMI trajectory for SINDy ODE recovery.

    Uses classic chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.WINDMI,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "x_0": 0.1,
            "y_0": 0.0,
            "z_0": 0.0,
        },
    )
    sim = WindmiSimulation(config)
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


def generate_substorm_sweep_data(
    n_b: int = 30,
    n_steps: int = 20000,
    dt: float = 0.01,
    a: float = 0.7,
) -> dict[str, np.ndarray]:
    """Sweep b (solar wind input) to map the substorm transition.

    For a=0.7, lower b values produce stable or periodic behavior
    while higher b values produce chaotic substorm-like dynamics.

    Args:
        n_b: Number of b values to sweep.
        n_steps: Steps for Lyapunov estimation.
        dt: Integration timestep.
        a: Damping coefficient (fixed during sweep).

    Returns:
        Dict with b values, Lyapunov exponents, max amplitudes,
        and attractor type classifications.
    """
    b_values = np.linspace(0.5, 5.0, n_b)
    lyapunov_exps = []
    max_amplitudes = []
    attractor_types = []

    for i, b in enumerate(b_values):
        config = SimulationConfig(
            domain=Domain.WINDMI,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": a, "b": b, "x_0": 0.1},
        )
        sim = WindmiSimulation(config)
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
        if lam > 0.01:
            atype = "chaotic"
        elif lam > -0.01:
            atype = "weakly_chaotic"
        else:
            atype = "periodic_or_fixed"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  b={b:.2f}: Lyapunov={lam:.4f}, type={atype}"
            )

    return {
        "b": b_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "attractor_type": np.array(attractor_types),
    }


def run_windmi_rediscovery(
    output_dir: str | Path = "output/rediscovery/windmi",
    n_iterations: int = 40,
) -> dict:
    """Run the full WINDMI system rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep b to map substorm transition (Lyapunov exponent)
    3. Compute Lyapunov at classic chaotic parameters
    4. Verify fixed point (x* = ln(b))
    5. Verify jerk equation form

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "windmi",
        "targets": {
            "ode_x": "dx/dt = y",
            "ode_y": "dy/dt = z",
            "ode_z": "dz/dt = -a*z - y + b - exp(x)",
            "nonlinearity": "exp(x) magnetospheric current response",
            "jerk_form": "x''' + a*x'' + x' = b - exp(x)",
            "fixed_point": "x* = ln(b), y* = 0, z* = 0",
            "chaos_regime": "a=0.7, b=2.5",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating WINDMI trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=5000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import (
            run_sindy,
        )

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
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
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

    # --- Part 2: Substorm transition sweep ---
    logger.info("Part 2: Mapping substorm transition (b sweep)...")
    sweep_data = generate_substorm_sweep_data(
        n_b=30, n_steps=20000, dt=0.01
    )

    n_chaotic = int(
        np.sum(sweep_data["attractor_type"] == "chaotic")
    )
    n_periodic = int(
        np.sum(sweep_data["attractor_type"] == "periodic_or_fixed")
    )
    logger.info(
        f"  Found {n_chaotic} chaotic, {n_periodic} periodic regimes"
    )

    results["substorm_transition"] = {
        "n_b_values": len(sweep_data["b"]),
        "n_chaotic": n_chaotic,
        "n_periodic": n_periodic,
        "b_range": [
            float(sweep_data["b"][0]),
            float(sweep_data["b"][-1]),
        ],
    }

    # Find approximate substorm threshold (first chaotic b value)
    mask_chaotic = sweep_data["lyapunov_exponent"] > 0.01
    if np.any(mask_chaotic):
        b_c_approx = float(
            sweep_data["b"][np.argmax(mask_chaotic)]
        )
        results["substorm_transition"]["b_c_approx"] = b_c_approx
        logger.info(
            f"  Approximate substorm onset b: {b_c_approx:.2f}"
        )

    # --- Part 3: Lyapunov at classic chaotic parameters ---
    logger.info(
        "Part 3: Lyapunov exponent at classic chaotic parameters..."
    )
    config_classic = SimulationConfig(
        domain=Domain.WINDMI,
        dt=0.01,
        n_steps=50000,
        parameters={"a": 0.7, "b": 2.5, "x_0": 0.1},
    )
    sim_classic = WindmiSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(
        n_steps=50000, dt=0.01
    )

    results["classic_parameters"] = {
        "a": 0.7,
        "b": 2.5,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic WINDMI Lyapunov: {lam_classic:.4f}")

    # --- Part 4: Fixed point verification ---
    logger.info("Part 4: Fixed point verification...")
    sim_fp = WindmiSimulation(config_classic)
    sim_fp.reset()
    fps = sim_fp.fixed_points
    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": [fp.tolist() for fp in fps],
        "expected_x_star": float(np.log(2.5)),
    }
    for i, fp in enumerate(fps):
        derivs = sim_fp._derivatives(fp)
        logger.info(
            f"  FP{i+1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}], "
            f"|deriv|={np.linalg.norm(derivs):.2e}"
        )

    # --- Part 5: Jerk equation verification ---
    logger.info("Part 5: Jerk equation verification...")
    sim_jerk = WindmiSimulation(config_classic)
    sim_jerk.reset()
    for _ in range(5000):
        sim_jerk.step()
    jerk_val = sim_jerk.compute_jerk(sim_jerk._state)
    dz_val = sim_jerk._derivatives(sim_jerk._state)[2]
    results["jerk_verification"] = {
        "jerk_equals_dz": bool(np.isclose(jerk_val, dz_val)),
        "jerk_value": float(jerk_val),
        "dz_value": float(dz_val),
    }
    logger.info(
        f"  Jerk = dz/dt check: {np.isclose(jerk_val, dz_val)}"
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
        output_path / "substorm_sweep.npz",
        b=sweep_data["b"],
        lyapunov_exponent=sweep_data["lyapunov_exponent"],
        max_amplitude=sweep_data["max_amplitude"],
    )

    return results
