"""Wang chaotic attractor rediscovery.

Targets:
- SINDy recovery of Wang ODEs: x'=x-a*y, y'=-b*y+x*z, z'=-c*z+d*x*y
- Lyapunov exponent estimation (positive for chaotic regime)
- Fixed point analysis (origin + two symmetric)
- Parameter sweep mapping chaos transition
- Divergence verification: div = 1 - b - c
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.wang import WangSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_WANG_DOMAIN = Domain.WANG


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 0.7,
    d: float = 0.5,
) -> dict[str, np.ndarray]:
    """Generate a single Wang trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=_WANG_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "x_0": 0.1,
            "y_0": 0.2,
            "z_0": 0.3,
        },
    )
    sim = WangSimulation(config)
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
        "d": d,
    }


def generate_lyapunov_sweep_data(
    n_points: int = 30,
    n_steps: int = 30000,
    dt: float = 0.01,
    param_name: str = "d",
    param_range: tuple[float, float] = (0.1, 1.5),
) -> dict[str, np.ndarray]:
    """Sweep a parameter to map Lyapunov exponent variation.

    By default sweeps d, the nonlinear coupling parameter, which controls
    the transition to chaos in the Wang system.
    """
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    lyapunov_exps = []

    defaults = {"a": 1.0, "b": 1.0, "c": 0.7, "d": 0.5}

    for i, pval in enumerate(param_values):
        params = dict(defaults)
        params[param_name] = pval
        config = SimulationConfig(
            domain=_WANG_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters=params,
        )
        sim = WangSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  {param_name}={pval:.3f}: Lyapunov={lam:.4f}")

    return {
        "param_name": param_name,
        "param_values": param_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_wang_rediscovery(
    output_dir: str | Path = "output/rediscovery/wang",
    n_iterations: int = 40,
) -> dict:
    """Run the full Wang attractor rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Lyapunov exponent sweep (d parameter)
    3. Lyapunov at classic parameters
    4. Fixed point analysis
    5. Divergence verification

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "wang",
        "targets": {
            "ode_x": "dx/dt = x - a*y",
            "ode_y": "dy/dt = -b*y + x*z",
            "ode_z": "dz/dt = -c*z + d*x*y",
            "divergence": "div = 1 - b - c",
            "fixed_points": "origin + two symmetric at x=+/-sqrt(b*c/d)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Wang trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.01)

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
            "true_d": ode_data["d"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Lyapunov sweep ---
    logger.info("Part 2: Lyapunov exponent sweep (d parameter)...")
    sweep_data = generate_lyapunov_sweep_data(
        n_points=30, n_steps=30000, dt=0.01
    )

    lam_arr = sweep_data["lyapunov_exponent"]
    n_chaotic = int(np.sum(lam_arr > 0.1))
    results["lyapunov_sweep"] = {
        "param_name": sweep_data["param_name"],
        "n_points": len(sweep_data["param_values"]),
        "n_chaotic": n_chaotic,
        "max_lyapunov": float(np.max(lam_arr)),
        "min_lyapunov": float(np.min(lam_arr)),
    }

    # --- Part 3: Lyapunov at classic parameters ---
    logger.info("Part 3: Lyapunov exponent at classic parameters...")
    config_classic = SimulationConfig(
        domain=_WANG_DOMAIN,
        dt=0.01,
        n_steps=50000,
        parameters={"a": 1.0, "b": 1.0, "c": 0.7, "d": 0.5},
    )
    sim_classic = WangSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.01)

    results["classic_parameters"] = {
        "a": 1.0,
        "b": 1.0,
        "c": 0.7,
        "d": 0.5,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Wang Lyapunov: {lam_classic:.4f}")

    # --- Part 4: Fixed points ---
    sim_fp = WangSimulation(config_classic)
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
            f"    FP{i+1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}], "
            f"|deriv|={np.linalg.norm(derivs):.2e}"
        )

    # --- Part 5: Divergence verification ---
    div = sim_fp.compute_divergence()
    expected_div = 1.0 - 1.0 - 0.7  # = -0.7
    results["divergence"] = {
        "computed": float(div),
        "expected": float(expected_div),
        "match": bool(np.isclose(div, expected_div)),
        "dissipative": bool(div < 0),
    }
    logger.info(f"  Divergence: {div:.4f} (expected {expected_div:.4f})")

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
        output_path / "lyapunov_sweep.npz",
        param_values=sweep_data["param_values"],
        lyapunov_exponent=sweep_data["lyapunov_exponent"],
    )

    return results
