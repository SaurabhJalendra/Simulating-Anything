"""Arneodo attractor rediscovery.

Targets:
- SINDy recovery of Arneodo ODEs: x'=y, y'=z, z'=-a*x-b*y-z+d*x^3
- Lyapunov exponent estimation (positive for chaotic regime)
- d-parameter sweep mapping period-doubling cascade to chaos
- Fixed point analysis (origin + two symmetric)
- Constant divergence = -1 verification
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.arneodo import ArneodoSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

_ARNEODO_DOMAIN = Domain.ARNEODO


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 5.5,
    b: float = 3.5,
    d: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate a single Arneodo trajectory for SINDy ODE recovery.

    Uses standard chaotic parameters by default. The cubic nonlinearity
    requires poly_degree >= 3 for SINDy to recover the x^3 term.
    """
    config = SimulationConfig(
        domain=_ARNEODO_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "d": d,
            "x_0": 0.2,
            "y_0": 0.2,
            "z_0": 0.2,
        },
    )
    sim = ArneodoSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "a": a,
        "b": b,
        "d": d,
    }


def generate_chaos_transition_data(
    n_d: int = 30,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep d to map the period-doubling cascade to chaos.

    For the Arneodo system with a=5.5, b=3.5, increasing d drives
    the system through period-doubling bifurcations into chaos.
    """
    d_values = np.linspace(0.5, 3.0, n_d)
    lyapunov_exps = []
    attractor_types = []
    max_amplitudes = []

    for i, d_val in enumerate(d_values):
        config = SimulationConfig(
            domain=_ARNEODO_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": 5.5, "b": 3.5, "d": d_val},
        )
        sim = ArneodoSimulation(config)
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
        if lam > 0.5:
            atype = "chaotic"
        elif lam < -0.1:
            atype = "fixed_point"
        else:
            atype = "periodic_or_transient"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(f"  d={d_val:.2f}: Lyapunov={lam:.3f}, type={atype}")

    return {
        "d": d_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "attractor_type": np.array(attractor_types),
    }


def generate_lyapunov_vs_d_data(
    n_d: int = 30,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Fine sweep of Lyapunov exponent as a function of d.

    Focuses on d in [0.5, 3.0] to capture the chaos onset region.
    """
    d_values = np.linspace(0.5, 3.0, n_d)
    lyapunov_exps = []

    for i, d_val in enumerate(d_values):
        config = SimulationConfig(
            domain=_ARNEODO_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": 5.5, "b": 3.5, "d": d_val},
        )
        sim = ArneodoSimulation(config)
        sim.reset()

        # Transient
        for _ in range(10000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  d={d_val:.2f}: Lyapunov={lam:.4f}")

    return {
        "d": d_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_arneodo_rediscovery(
    output_dir: str | Path = "output/rediscovery/arneodo",
    n_iterations: int = 40,
) -> dict:
    """Run the full Arneodo attractor rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep d to map period-doubling cascade (Lyapunov exponent)
    3. Fine Lyapunov sweep
    4. Fixed point analysis
    5. Divergence verification (constant = -1)

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "arneodo",
        "targets": {
            "ode_x": "dx/dt = y",
            "ode_y": "dy/dt = z",
            "ode_z": "dz/dt = -a*x - b*y - z + d*x^3",
            "jerk_form": "x''' + x'' + b*x' + a*x = d*x^3",
            "divergence": "-1 (constant, uniformly dissipative)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Arneodo trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x", "y", "z"],
            threshold=0.1,
            poly_degree=3,
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
            "true_d": ode_data["d"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Chaos transition sweep ---
    logger.info("Part 2: Mapping period-doubling cascade (d sweep)...")
    chaos_data = generate_chaos_transition_data(
        n_d=30, n_steps=20000, dt=0.01
    )

    n_chaotic = int(np.sum(chaos_data["attractor_type"] == "chaotic"))
    n_fixed = int(np.sum(chaos_data["attractor_type"] == "fixed_point"))
    logger.info(
        f"  Found {n_chaotic} chaotic, {n_fixed} fixed-point regimes"
    )

    results["chaos_transition"] = {
        "n_d_values": len(chaos_data["d"]),
        "n_chaotic": n_chaotic,
        "n_fixed_point": n_fixed,
        "d_range": [float(chaos_data["d"][0]), float(chaos_data["d"][-1])],
    }

    # --- Part 3: Fine Lyapunov sweep ---
    logger.info("Part 3: Fine Lyapunov exponent sweep...")
    fine_data = generate_lyapunov_vs_d_data(
        n_d=30, n_steps=30000, dt=0.01
    )

    lam = fine_data["lyapunov_exponent"]
    d_fine = fine_data["d"]

    results["lyapunov_analysis"] = {
        "n_points": len(d_fine),
        "d_range": [float(d_fine[0]), float(d_fine[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info("Part 4: Lyapunov exponent at classic parameters...")
    config_classic = SimulationConfig(
        domain=_ARNEODO_DOMAIN,
        dt=0.01,
        n_steps=50000,
        parameters={"a": 5.5, "b": 3.5, "d": 1.0},
    )
    sim_classic = ArneodoSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.01)

    results["classic_parameters"] = {
        "a": 5.5,
        "b": 3.5,
        "d": 1.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Arneodo Lyapunov: {lam_classic:.4f}")

    # --- Part 5: Fixed points ---
    sim_fp = ArneodoSimulation(config_classic)
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

    # --- Part 6: Divergence verification ---
    logger.info("Part 6: Verifying constant divergence = -1...")
    test_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([-5.0, 10.0, -7.0]),
    ]
    divs = [sim_fp.compute_divergence(pt) for pt in test_points]
    results["divergence"] = {
        "values": divs,
        "all_equal_minus_one": all(d == -1.0 for d in divs),
        "theory": -1.0,
    }
    logger.info(f"  Divergence values: {divs} (all should be -1.0)")

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
        **{
            k: v
            for k, v in chaos_data.items()
            if isinstance(v, np.ndarray)
        },
    )
    np.savez(
        output_path / "lyapunov_fine.npz",
        d=fine_data["d"],
        lyapunov_exponent=fine_data["lyapunov_exponent"],
    )

    return results
