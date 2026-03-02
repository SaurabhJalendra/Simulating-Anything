"""Lorenz-Stenflo system rediscovery.

Targets:
- SINDy recovery of 4 ODEs:
    dx/dt = sigma*(y-x) + s*w
    dy/dt = r*x - y - x*z
    dz/dt = x*y - b*z
    dw/dt = -x - sigma*w
- Lyapunov exponent estimation (positive for chaotic regime)
- s-parameter sweep: transition from Lorenz-like chaos to hyperchaos
- Fixed point analysis and verification
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.lorenz_stenflo import LorenzStenfloSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    n_steps: int = 5000,
    dt: float = 0.005,
    sigma: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0,
    s: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate a single Lorenz-Stenflo trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.LORENZ_STENFLO,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "sigma": sigma,
            "r": r,
            "b": b,
            "s": s,
            "x_0": 1.0,
            "y_0": 0.0,
            "z_0": 0.0,
            "w_0": 0.0,
        },
    )
    sim = LorenzStenfloSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "sigma": sigma,
        "r": r,
        "b": b,
        "s": s,
    }


def generate_s_sweep_data(
    n_s: int = 30,
    dt: float = 0.005,
    s_min: float = 0.0,
    s_max: float = 10.0,
) -> dict[str, np.ndarray]:
    """Sweep Stenflo parameter s to map transition from Lorenz to hyperchaos.

    For s=0, system reduces to classic Lorenz (3D chaos).
    As s increases, fourth dimension couples and can produce hyperchaos.
    """
    s_values = np.linspace(s_min, s_max, n_s)
    lyapunov_exps = []
    max_amplitudes = []
    w_amplitudes = []

    for i, s_val in enumerate(s_values):
        config = SimulationConfig(
            domain=Domain.LORENZ_STENFLO,
            dt=dt,
            n_steps=1000,
            parameters={
                "sigma": 10.0,
                "r": 28.0,
                "b": 8.0 / 3.0,
                "s": s_val,
                "x_0": 1.0,
                "y_0": 0.0,
                "z_0": 0.0,
                "w_0": 0.0,
            },
        )
        sim = LorenzStenfloSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        # Estimate Lyapunov exponent
        lam = sim.estimate_lyapunov(n_steps=20000, dt=dt)
        lyapunov_exps.append(lam)

        # Measure amplitude
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

        if (i + 1) % 10 == 0:
            logger.info(
                f"  s={s_val:.2f}: Lyapunov={lam:.4f}, "
                f"max_amp={max_amplitudes[-1]:.1f}, max_|w|={w_amplitudes[-1]:.2f}"
            )

    return {
        "s": s_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "w_amplitude": np.array(w_amplitudes),
    }


def run_lorenz_stenflo_rediscovery(
    output_dir: str | Path = "output/rediscovery/lorenz_stenflo",
    n_iterations: int = 40,
) -> dict:
    """Run the full Lorenz-Stenflo system rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery (4 equations)
    2. Sweep s to map transition from Lorenz-like chaos to hyperchaos
    3. Compute Lyapunov at standard parameters
    4. Verify fixed points
    5. Verify reduction to Lorenz when s=0

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "lorenz_stenflo",
        "targets": {
            "ode_x": "dx/dt = sigma*(y-x) + s*w",
            "ode_y": "dy/dt = r*x - y - x*z",
            "ode_z": "dz/dt = x*y - b*z",
            "ode_w": "dw/dt = -x - sigma*w",
            "lorenz_reduction": "s=0 recovers classic Lorenz",
            "chaos_regime": "positive Lyapunov for standard parameters",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Lorenz-Stenflo trajectory for SINDy...")
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
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                for d in sindy_discoveries
            ],
            "true_sigma": ode_data["sigma"],
            "true_r": ode_data["r"],
            "true_b": ode_data["b"],
            "true_s": ode_data["s"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: s-parameter sweep ---
    logger.info("Part 2: Stenflo parameter sweep (s=0 to s=10)...")
    sweep_data = generate_s_sweep_data(n_s=30, dt=0.005)

    n_chaotic = int(np.sum(sweep_data["lyapunov_exponent"] > 0.01))
    n_stable = int(np.sum(sweep_data["lyapunov_exponent"] <= 0.01))
    logger.info(f"  Found {n_chaotic} chaotic, {n_stable} non-chaotic regimes")

    results["s_sweep"] = {
        "n_s_values": len(sweep_data["s"]),
        "n_chaotic": n_chaotic,
        "n_stable": n_stable,
        "s_range": [float(sweep_data["s"][0]), float(sweep_data["s"][-1])],
        "max_lyapunov": float(np.max(sweep_data["lyapunov_exponent"])),
        "min_lyapunov": float(np.min(sweep_data["lyapunov_exponent"])),
    }

    # Check w-amplitude growth with s
    if len(sweep_data["s"]) > 1:
        w_at_s0 = float(sweep_data["w_amplitude"][0])
        w_at_smax = float(sweep_data["w_amplitude"][-1])
        results["s_sweep"]["w_amplitude_s0"] = w_at_s0
        results["s_sweep"]["w_amplitude_smax"] = w_at_smax
        logger.info(
            f"  w amplitude: s=0 -> {w_at_s0:.4f}, "
            f"s={sweep_data['s'][-1]:.1f} -> {w_at_smax:.2f}"
        )

    # --- Part 3: Lyapunov at standard parameters ---
    logger.info("Part 3: Lyapunov exponent at standard parameters...")
    config_std = SimulationConfig(
        domain=Domain.LORENZ_STENFLO,
        dt=0.005,
        n_steps=50000,
        parameters={
            "sigma": 10.0,
            "r": 28.0,
            "b": 8.0 / 3.0,
            "s": 1.0,
        },
    )
    sim_std = LorenzStenfloSimulation(config_std)
    sim_std.reset()
    for _ in range(10000):
        sim_std.step()
    lam_std = sim_std.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["standard_parameters"] = {
        "sigma": 10.0,
        "r": 28.0,
        "b": 8.0 / 3.0,
        "s": 1.0,
        "lyapunov_exponent": float(lam_std),
        "positive": bool(lam_std > 0),
    }
    logger.info(f"  Standard Lorenz-Stenflo Lyapunov: {lam_std:.4f}")

    # --- Part 4: Fixed points ---
    logger.info("Part 4: Fixed point analysis...")
    sim_fp = LorenzStenfloSimulation(config_std)
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

    # --- Part 5: Lorenz reduction check ---
    logger.info("Part 5: Verifying reduction to Lorenz when s=0...")
    reduction = sim_std.reduces_to_lorenz(n_steps=5000)
    results["lorenz_reduction"] = {
        "max_w": reduction["max_w"],
        "max_lorenz_deviation": reduction["max_lorenz_deviation"],
        "reduces_correctly": bool(
            reduction["max_w"] < 1e-10
            and reduction["max_lorenz_deviation"] < 1e-10
        ),
    }
    logger.info(
        f"  max |w| with s=0: {reduction['max_w']:.2e}, "
        f"max Lorenz deviation: {reduction['max_lorenz_deviation']:.2e}"
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
        output_path / "s_sweep.npz",
        s=sweep_data["s"],
        lyapunov_exponent=sweep_data["lyapunov_exponent"],
        max_amplitude=sweep_data["max_amplitude"],
        w_amplitude=sweep_data["w_amplitude"],
    )

    return results
