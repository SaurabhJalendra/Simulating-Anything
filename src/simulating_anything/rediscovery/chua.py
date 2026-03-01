"""Chua's circuit rediscovery.

Targets:
- SINDy recovery of Chua ODEs: x'=alpha*(y-x-f(x)), y'=x-y+z, z'=-beta*y
- Period-doubling route to chaos as alpha varies
- Lyapunov exponent as a function of alpha
- Three fixed points and their stability
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.chua import ChuaCircuit
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    n_steps: int = 10000,
    dt: float = 0.001,
    alpha: float = 15.6,
    beta: float = 28.0,
    m0: float = -1.143,
    m1: float = -0.714,
) -> dict[str, np.ndarray]:
    """Generate a single Chua circuit trajectory for SINDy ODE recovery.

    Uses classic double-scroll parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.CHUA,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha": alpha,
            "beta": beta,
            "m0": m0,
            "m1": m1,
            "x_0": 0.1,
            "y_0": 0.0,
            "z_0": 0.0,
        },
    )
    sim = ChuaCircuit(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "alpha": alpha,
        "beta": beta,
        "m0": m0,
        "m1": m1,
    }


def generate_attractor_data(
    n_alpha: int = 30,
    dt: float = 0.001,
) -> dict[str, np.ndarray]:
    """Sweep alpha to map the route to chaos.

    For the classic Chua circuit with m0=-1.143, m1=-0.714, beta=28:
    - alpha ~ 8: periodic orbit
    - alpha ~ 12: period-doubling
    - alpha ~ 15.6: double-scroll chaos
    """
    alpha_values = np.linspace(7.0, 18.0, n_alpha)
    lyapunov_exps = []

    for i, alpha in enumerate(alpha_values):
        config = SimulationConfig(
            domain=Domain.CHUA,
            dt=dt,
            n_steps=1000,
            parameters={
                "alpha": alpha,
                "beta": 28.0,
                "m0": -1.143,
                "m1": -0.714,
                "x_0": 0.1,
                "y_0": 0.0,
                "z_0": 0.0,
            },
        )
        sim = ChuaCircuit(config)
        sim.reset()

        # Skip transient
        for _ in range(10000):
            sim.step()

        # Estimate Lyapunov exponent
        lam = sim.estimate_lyapunov(n_steps=30000, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  alpha={alpha:.2f}: Lyapunov={lam:.4f}"
            )

    return {
        "alpha": alpha_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_chua_rediscovery(
    output_dir: str | Path = "output/rediscovery/chua",
    n_iterations: int = 40,
) -> dict:
    """Run the full Chua circuit rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep alpha to map the route to chaos
    3. Compute Lyapunov at standard parameters
    4. Verify fixed points

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "chua",
        "targets": {
            "ode_x": "dx/dt = alpha*(y - x - f(x))",
            "ode_y": "dy/dt = x - y + z",
            "ode_z": "dz/dt = -beta*y",
            "chaos_regime": "alpha ~ 15.6 with beta=28, m0=-1.143, m1=-0.714",
            "lyapunov_chaotic": "lambda ~ 0.3 for standard params",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Chua trajectory for SINDy...")
    ode_data = generate_trajectory_data(n_steps=20000, dt=0.001)

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
            "true_beta": ode_data["beta"],
            "true_m0": ode_data["m0"],
            "true_m1": ode_data["m1"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Alpha sweep for chaos transition ---
    logger.info("Part 2: Mapping chaos transition (alpha sweep)...")
    attractor_data = generate_attractor_data(n_alpha=30, dt=0.001)

    n_chaotic = int(np.sum(attractor_data["lyapunov_exponent"] > 0.01))
    n_periodic = int(np.sum(attractor_data["lyapunov_exponent"] <= 0.01))
    logger.info(f"  Found {n_chaotic} chaotic, {n_periodic} periodic regimes")

    results["chaos_transition"] = {
        "n_alpha_values": len(attractor_data["alpha"]),
        "n_chaotic": n_chaotic,
        "n_periodic": n_periodic,
        "alpha_range": [
            float(attractor_data["alpha"][0]),
            float(attractor_data["alpha"][-1]),
        ],
    }

    # Find approximate chaos onset
    mask_positive = attractor_data["lyapunov_exponent"] > 0.01
    if np.any(mask_positive):
        alpha_chaos_approx = float(
            attractor_data["alpha"][np.argmax(mask_positive)]
        )
        results["chaos_transition"]["alpha_chaos_approx"] = alpha_chaos_approx
        logger.info(f"  Approximate chaos onset alpha: {alpha_chaos_approx:.2f}")

    # --- Part 3: Lyapunov at classic parameters ---
    logger.info("Part 3: Lyapunov exponent at classic chaotic parameters...")
    config_classic = SimulationConfig(
        domain=Domain.CHUA,
        dt=0.001,
        n_steps=50000,
        parameters={
            "alpha": 15.6,
            "beta": 28.0,
            "m0": -1.143,
            "m1": -0.714,
            "x_0": 0.1,
            "y_0": 0.0,
            "z_0": 0.0,
        },
    )
    sim_classic = ChuaCircuit(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.001)

    results["classic_parameters"] = {
        "alpha": 15.6,
        "beta": 28.0,
        "m0": -1.143,
        "m1": -0.714,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Chua Lyapunov: {lam_classic:.4f}")

    # --- Part 4: Fixed points ---
    sim_fp = ChuaCircuit(config_classic)
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
            f"    FP{i + 1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}], "
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
        output_path / "attractor_data.npz",
        alpha=attractor_data["alpha"],
        lyapunov_exponent=attractor_data["lyapunov_exponent"],
    )

    return results
