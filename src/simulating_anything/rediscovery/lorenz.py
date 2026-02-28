"""Lorenz system rediscovery.

Targets:
- SINDy recovery of Lorenz ODEs: x'=sigma*(y-x), y'=x*(rho-z)-y, z'=x*y-beta*z
- Critical rho value for onset of chaos (rho_c ~ 24.74)
- Lyapunov exponent as a function of rho
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.lorenz import LorenzSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> dict[str, np.ndarray]:
    """Generate a single Lorenz trajectory for SINDy ODE recovery.

    Uses classic chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.LORENZ_ATTRACTOR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "sigma": sigma,
            "rho": rho,
            "beta": beta,
            "x_0": 1.0,
            "y_0": 1.0,
            "z_0": 1.0,
        },
    )
    sim = LorenzSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "sigma": sigma,
        "rho": rho,
        "beta": beta,
    }


def generate_chaos_transition_data(
    n_rho: int = 50,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep rho to map the transition to chaos.

    For each rho value, compute the Lyapunov exponent and classify
    the attractor type (fixed point, periodic, chaotic).
    """
    rho_values = np.linspace(0.5, 35.0, n_rho)
    lyapunov_exps = []
    attractor_types = []
    max_amplitudes = []

    for i, rho in enumerate(rho_values):
        config = SimulationConfig(
            domain=Domain.LORENZ_ATTRACTOR,
            dt=dt,
            n_steps=n_steps,
            parameters={"sigma": 10.0, "rho": rho, "beta": 8.0 / 3.0},
        )
        sim = LorenzSimulation(config)
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
        if rho < 1.0:
            atype = "origin"
        elif lam < 0.01:
            atype = "fixed_point"
        elif lam > 0.5:
            atype = "chaotic"
        else:
            atype = "periodic_or_transient"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(f"  rho={rho:.1f}: Lyapunov={lam:.3f}, type={atype}")

    return {
        "rho": rho_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "attractor_type": np.array(attractor_types),
    }


def generate_lyapunov_vs_rho_data(
    n_rho: int = 30,
    n_steps: int = 30000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Fine sweep of Lyapunov exponent near the chaos transition.

    Focuses on rho in [20, 30] to find critical rho ~ 24.74.
    """
    rho_values = np.linspace(20.0, 30.0, n_rho)
    lyapunov_exps = []

    for i, rho in enumerate(rho_values):
        config = SimulationConfig(
            domain=Domain.LORENZ_ATTRACTOR,
            dt=dt,
            n_steps=n_steps,
            parameters={"sigma": 10.0, "rho": rho, "beta": 8.0 / 3.0},
        )
        sim = LorenzSimulation(config)
        sim.reset()

        # Transient
        for _ in range(10000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  rho={rho:.2f}: Lyapunov={lam:.4f}")

    return {
        "rho": rho_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_lorenz_rediscovery(
    output_dir: str | Path = "output/rediscovery/lorenz",
    n_iterations: int = 40,
) -> dict:
    """Run the full Lorenz system rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep rho to map chaos transition (Lyapunov exponent)
    3. Fine sweep near critical rho
    4. Run PySR on Lyapunov vs rho data

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "lorenz",
        "targets": {
            "ode_x": "dx/dt = sigma*(y-x)",
            "ode_y": "dy/dt = x*(rho-z) - y",
            "ode_z": "dz/dt = x*y - beta*z",
            "chaos_onset": "rho_c ~ 24.74",
            "lyapunov_chaotic": "lambda ~ 0.9 for sigma=10, rho=28, beta=8/3",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Lorenz trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=5000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x", "y", "z"],
            threshold=0.1,
            poly_degree=2,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries
            ],
            "true_sigma": ode_data["sigma"],
            "true_rho": ode_data["rho"],
            "true_beta": ode_data["beta"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Chaos transition sweep ---
    logger.info("Part 2: Mapping chaos transition (rho sweep)...")
    chaos_data = generate_chaos_transition_data(n_rho=50, n_steps=20000, dt=0.01)

    n_chaotic = np.sum(chaos_data["attractor_type"] == "chaotic")
    n_fixed = np.sum(chaos_data["attractor_type"] == "fixed_point")
    logger.info(f"  Found {n_chaotic} chaotic, {n_fixed} fixed-point regimes")

    results["chaos_transition"] = {
        "n_rho_values": len(chaos_data["rho"]),
        "n_chaotic": int(n_chaotic),
        "n_fixed_point": int(n_fixed),
        "rho_range": [float(chaos_data["rho"][0]), float(chaos_data["rho"][-1])],
    }

    # Find approximate critical rho (first positive Lyapunov)
    mask_positive = chaos_data["lyapunov_exponent"] > 0.1
    if np.any(mask_positive):
        rho_c_approx = float(chaos_data["rho"][np.argmax(mask_positive)])
        results["chaos_transition"]["rho_c_approx"] = rho_c_approx
        logger.info(f"  Approximate critical rho: {rho_c_approx:.1f} (true: ~24.74)")

    # --- Part 3: Fine Lyapunov sweep ---
    logger.info("Part 3: Fine Lyapunov exponent sweep near transition...")
    fine_data = generate_lyapunov_vs_rho_data(n_rho=30, n_steps=30000, dt=0.005)

    # Find zero crossing of Lyapunov exponent
    lam = fine_data["lyapunov_exponent"]
    rho_fine = fine_data["rho"]
    zero_crossings = []
    for j in range(len(lam) - 1):
        if lam[j] <= 0 < lam[j + 1]:
            # Linear interpolation
            frac = -lam[j] / (lam[j + 1] - lam[j])
            rho_cross = rho_fine[j] + frac * (rho_fine[j + 1] - rho_fine[j])
            zero_crossings.append(float(rho_cross))

    results["lyapunov_analysis"] = {
        "n_points": len(rho_fine),
        "rho_range": [float(rho_fine[0]), float(rho_fine[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
        "zero_crossings": zero_crossings,
    }
    if zero_crossings:
        logger.info(f"  Lyapunov zero crossings at rho = {zero_crossings}")
        logger.info(f"  First crossing: {zero_crossings[0]:.2f} (true: ~24.74)")

    # --- Part 4: Lyapunov exponent analysis ---
    # Compute Lyapunov at the classic chaotic parameters
    config_classic = SimulationConfig(
        domain=Domain.LORENZ_ATTRACTOR,
        dt=0.005,
        n_steps=50000,
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    )
    sim_classic = LorenzSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "sigma": 10.0,
        "rho": 28.0,
        "beta": 8.0 / 3.0,
        "lyapunov_exponent": float(lam_classic),
        "lyapunov_known": 0.9056,  # Literature value
        "relative_error": float(abs(lam_classic - 0.9056) / 0.9056),
    }
    logger.info(f"  Classic Lorenz Lyapunov: {lam_classic:.4f} (known: 0.9056)")

    # Fixed points
    sim_fp = LorenzSimulation(config_classic)
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
        output_path / "chaos_transition.npz",
        **{k: v for k, v in chaos_data.items() if isinstance(v, np.ndarray)},
    )
    np.savez(
        output_path / "lyapunov_fine.npz",
        rho=fine_data["rho"],
        lyapunov_exponent=fine_data["lyapunov_exponent"],
    )

    return results
