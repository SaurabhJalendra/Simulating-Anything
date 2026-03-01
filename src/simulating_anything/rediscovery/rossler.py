"""Rossler system rediscovery.

Targets:
- SINDy recovery of Rossler ODEs: x'=-y-z, y'=x+a*y, z'=b+z*(x-c)
- Period-doubling route to chaos as c increases
- Lyapunov exponent as a function of c
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.rossler import RosslerSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_trajectory_data(
    n_steps: int = 5000,
    dt: float = 0.005,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
) -> dict[str, np.ndarray]:
    """Generate a single Rossler trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default.
    """
    config = SimulationConfig(
        domain=Domain.ROSSLER,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "c": c,
            "x_0": 1.0,
            "y_0": 1.0,
            "z_0": 0.0,
        },
    )
    sim = RosslerSimulation(config)
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


def generate_period_data(
    n_c: int = 30,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep c to map period-doubling route to chaos.

    For a=0.2, b=0.2:
    - c ~ 3: period-1 oscillation
    - c ~ 4: period-2
    - c ~ 5.7: chaos
    """
    c_values = np.linspace(2.0, 6.5, n_c)
    periods = []
    lyapunov_exps = []

    for i, c in enumerate(c_values):
        config = SimulationConfig(
            domain=Domain.ROSSLER,
            dt=dt,
            n_steps=1000,
            parameters={"a": 0.2, "b": 0.2, "c": c},
        )
        sim = RosslerSimulation(config)
        sim.reset()

        # Measure period
        T = sim.measure_period(n_transient=5000, n_measure=20000)
        periods.append(T)

        # Estimate Lyapunov exponent
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  c={c:.2f}: period={T:.2f}, Lyapunov={lam:.4f}"
            )

    return {
        "c": c_values,
        "period": np.array(periods),
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_rossler_rediscovery(
    output_dir: str | Path = "output/rediscovery/rossler",
    n_iterations: int = 40,
) -> dict:
    """Run the full Rossler system rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep c to map period-doubling route to chaos
    3. Compute Lyapunov at standard parameters
    4. Verify fixed points

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "rossler",
        "targets": {
            "ode_x": "dx/dt = -y - z",
            "ode_y": "dy/dt = x + a*y",
            "ode_z": "dz/dt = b + z*(x - c)",
            "chaos_regime": "c ~ 5.7 with a=0.2, b=0.2",
            "lyapunov_chaotic": "lambda ~ 0.07 for standard params",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Rossler trajectory for SINDy...")
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
            "true_a": ode_data["a"],
            "true_b": ode_data["b"],
            "true_c": ode_data["c"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Period-doubling sweep ---
    logger.info("Part 2: Mapping period-doubling route to chaos (c sweep)...")
    period_data = generate_period_data(n_c=30, dt=0.005)

    # Classify regimes
    n_chaotic = int(np.sum(period_data["lyapunov_exponent"] > 0.01))
    n_periodic = int(np.sum(period_data["lyapunov_exponent"] <= 0.01))
    logger.info(f"  Found {n_chaotic} chaotic, {n_periodic} periodic regimes")

    results["period_doubling"] = {
        "n_c_values": len(period_data["c"]),
        "n_chaotic": n_chaotic,
        "n_periodic": n_periodic,
        "c_range": [float(period_data["c"][0]), float(period_data["c"][-1])],
    }

    # Find approximate chaos onset (first positive Lyapunov)
    mask_positive = period_data["lyapunov_exponent"] > 0.01
    if np.any(mask_positive):
        c_chaos_approx = float(period_data["c"][np.argmax(mask_positive)])
        results["period_doubling"]["c_chaos_approx"] = c_chaos_approx
        logger.info(f"  Approximate chaos onset c: {c_chaos_approx:.2f}")

    # --- Part 3: Lyapunov at classic parameters ---
    logger.info("Part 3: Lyapunov exponent at classic chaotic parameters...")
    config_classic = SimulationConfig(
        domain=Domain.ROSSLER,
        dt=0.005,
        n_steps=50000,
        parameters={"a": 0.2, "b": 0.2, "c": 5.7},
    )
    sim_classic = RosslerSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.005)

    results["classic_parameters"] = {
        "a": 0.2,
        "b": 0.2,
        "c": 5.7,
        "lyapunov_exponent": float(lam_classic),
        "lyapunov_known": 0.07,
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Rossler Lyapunov: {lam_classic:.4f} (known: ~0.07)")

    # --- Part 4: Fixed points ---
    sim_fp = RosslerSimulation(config_classic)
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
        output_path / "period_data.npz",
        c=period_data["c"],
        period=period_data["period"],
        lyapunov_exponent=period_data["lyapunov_exponent"],
    )

    return results
