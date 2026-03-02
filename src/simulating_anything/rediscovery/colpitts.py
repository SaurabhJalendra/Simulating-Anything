"""Colpitts oscillator rediscovery.

Targets:
- SINDy recovery of Colpitts ODEs: x'=y, y'=z, z'=-g_d*z-y+V_cc-Q*max(0,x)
- Lyapunov exponent estimation for chaos detection
- Q sweep for chaos transition mapping (Q_c ~ 7)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.colpitts import ColpittsSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    Q: float = 8.0,
    g_d: float = 0.3,
    V_cc: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate a single Colpitts trajectory for SINDy ODE recovery.

    Uses classic chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.COLPITTS,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "Q": Q,
            "g_d": g_d,
            "V_cc": V_cc,
            "x_0": 0.1,
            "y_0": 0.0,
            "z_0": 0.0,
        },
    )
    sim = ColpittsSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "Q": Q,
        "g_d": g_d,
        "V_cc": V_cc,
    }


def generate_chaos_sweep_data(
    n_Q: int = 30,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep Q (transistor gain) to map the transition to chaos.

    For g_d=0.3, V_cc=1.0, lower Q is periodic and higher Q is chaotic.
    Chaos onset is around Q_c ~ 7.
    """
    Q_values = np.linspace(3.0, 12.0, n_Q)
    lyapunov_exps = []
    max_amplitudes = []
    attractor_types = []

    for i, Q in enumerate(Q_values):
        config = SimulationConfig(
            domain=Domain.COLPITTS,
            dt=dt,
            n_steps=n_steps,
            parameters={"Q": Q, "g_d": 0.3, "V_cc": 1.0, "x_0": 0.1},
        )
        sim = ColpittsSimulation(config)
        sim.reset()

        # Run to skip transient
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
                f"  Q={Q:.2f}: Lyapunov={lam:.4f}, type={atype}"
            )

    return {
        "Q": Q_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "attractor_type": np.array(attractor_types),
    }


def run_colpitts_rediscovery(
    output_dir: str | Path = "output/rediscovery/colpitts",
    n_iterations: int = 40,
) -> dict:
    """Run the full Colpitts oscillator rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep Q to map chaos transition (Lyapunov exponent)
    3. Compute Lyapunov at classic chaotic parameters
    4. Verify fixed point

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "colpitts",
        "targets": {
            "ode_x": "dx/dt = y",
            "ode_y": "dy/dt = z",
            "ode_z": "dz/dt = -g_d*z - y + V_cc - Q*max(0,x)",
            "nonlinearity": "max(0, x) piecewise-linear BJT model",
            "chaos_regime": "Q=8.0, g_d=0.3, V_cc=1.0",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Colpitts trajectory for SINDy...")
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
            "true_Q": ode_data["Q"],
            "true_g_d": ode_data["g_d"],
            "true_V_cc": ode_data["V_cc"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Chaos transition sweep ---
    logger.info("Part 2: Mapping chaos transition (Q sweep)...")
    chaos_data = generate_chaos_sweep_data(
        n_Q=30, n_steps=20000, dt=0.01
    )

    n_chaotic = int(
        np.sum(chaos_data["attractor_type"] == "chaotic")
    )
    n_periodic = int(
        np.sum(chaos_data["attractor_type"] == "periodic_or_fixed")
    )
    logger.info(
        f"  Found {n_chaotic} chaotic, {n_periodic} periodic regimes"
    )

    results["chaos_transition"] = {
        "n_Q_values": len(chaos_data["Q"]),
        "n_chaotic": n_chaotic,
        "n_periodic": n_periodic,
        "Q_range": [
            float(chaos_data["Q"][0]),
            float(chaos_data["Q"][-1]),
        ],
    }

    # Find approximate transition Q (first chaotic value)
    mask_chaotic = chaos_data["lyapunov_exponent"] > 0.01
    if np.any(mask_chaotic):
        Q_c_approx = float(
            chaos_data["Q"][np.argmax(mask_chaotic)]
        )
        results["chaos_transition"]["Q_c_approx"] = Q_c_approx
        logger.info(
            f"  Approximate chaos onset Q: {Q_c_approx:.2f}"
        )

    # --- Part 3: Lyapunov at classic chaotic parameters ---
    logger.info(
        "Part 3: Lyapunov exponent at classic chaotic parameters..."
    )
    config_classic = SimulationConfig(
        domain=Domain.COLPITTS,
        dt=0.01,
        n_steps=50000,
        parameters={"Q": 8.0, "g_d": 0.3, "V_cc": 1.0, "x_0": 0.1},
    )
    sim_classic = ColpittsSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(
        n_steps=50000, dt=0.01
    )

    results["classic_parameters"] = {
        "Q": 8.0,
        "g_d": 0.3,
        "V_cc": 1.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Colpitts Lyapunov: {lam_classic:.4f}")

    # --- Part 4: Fixed point verification ---
    sim_fp = ColpittsSimulation(config_classic)
    sim_fp.reset()
    fps = sim_fp.fixed_points
    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": [fp.tolist() for fp in fps],
    }
    for i, fp in enumerate(fps):
        derivs = sim_fp._derivatives(fp)
        logger.info(
            f"  FP{i+1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}], "
            f"|deriv|={np.linalg.norm(derivs):.2e}"
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
        output_path / "chaos_sweep.npz",
        Q=chaos_data["Q"],
        lyapunov_exponent=chaos_data["lyapunov_exponent"],
        max_amplitude=chaos_data["max_amplitude"],
    )

    return results
