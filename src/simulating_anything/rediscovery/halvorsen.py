"""Halvorsen attractor rediscovery.

Targets:
- SINDy recovery of Halvorsen ODEs:
    x' = -a*x - 4*y - 4*z - y^2
    y' = -a*y - 4*z - 4*x - z^2
    z' = -a*z - 4*x - 4*y - x^2
- Positive Lyapunov exponent confirming chaos (a ~ 1.89)
- Cyclic symmetry (x, y, z) -> (y, z, x) verification
- a-parameter sweep mapping chaos boundary (a ~ 1.4 to 2.0)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.halvorsen import HalvorsenSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use CHAOTIC_ODE as the domain enum placeholder until Domain.HALVORSEN is added
_HALVORSEN_DOMAIN = Domain.CHAOTIC_ODE


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 1.89,
) -> dict[str, np.ndarray]:
    """Generate a single Halvorsen trajectory for SINDy ODE recovery.

    Uses the classic chaotic parameter a = 1.89 by default.
    """
    config = SimulationConfig(
        domain=_HALVORSEN_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={"a": a, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
    )
    sim = HalvorsenSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "a": a,
    }


def generate_lyapunov_vs_a_data(
    n_a: int = 30,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep parameter a to map the chaos transition.

    Focuses on a in [1.0, 2.5] to capture the chaotic regime around a ~ 1.89.
    """
    a_values = np.linspace(1.0, 2.5, n_a)
    lyapunov_exps = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": a, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = HalvorsenSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  a={a:.4f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def generate_chaos_transition_data(
    n_a: int = 30,
    n_steps: int = 20000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep a to classify attractor type (chaotic, periodic, fixed point).

    Covers a range that includes the chaos onset and the high-dissipation
    fixed-point regime.
    """
    a_values = np.linspace(1.0, 3.0, n_a)
    lyapunov_exps = []
    attractor_types = []
    max_amplitudes = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=_HALVORSEN_DOMAIN,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": a, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
        )
        sim = HalvorsenSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        # Estimate Lyapunov exponent
        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        # Measure amplitude
        x_vals = []
        for _ in range(5000):
            state = sim.step()
            x_vals.append(np.linalg.norm(state))
        max_amplitudes.append(np.max(x_vals))

        # Classify
        if lam > 0.01:
            atype = "chaotic"
        elif lam < -0.01:
            atype = "fixed_point"
        else:
            atype = "periodic_or_marginal"
        attractor_types.append(atype)

        if (i + 1) % 10 == 0:
            logger.info(f"  a={a:.2f}: Lyapunov={lam:.3f}, type={atype}")

    return {
        "a": a_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
        "max_amplitude": np.array(max_amplitudes),
        "attractor_type": np.array(attractor_types),
    }


def run_halvorsen_rediscovery(
    output_dir: str | Path = "output/rediscovery/halvorsen",
    n_iterations: int = 40,
) -> dict:
    """Run the full Halvorsen attractor rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep a to map chaos transition (Lyapunov exponent)
    3. Verify cyclic symmetry
    4. Compute Lyapunov at classic parameters
    5. Trajectory statistics on the attractor

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "halvorsen",
        "targets": {
            "ode_x": "dx/dt = -a*x - 4*y - 4*z - y^2",
            "ode_y": "dy/dt = -a*y - 4*z - 4*x - z^2",
            "ode_z": "dz/dt = -a*z - 4*x - 4*y - x^2",
            "classic_a": 1.89,
            "symmetry": "(x, y, z) -> (y, z, x) cyclic symmetry",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Halvorsen trajectory for SINDy...")
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
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Lyapunov vs a sweep ---
    logger.info("Part 2: Mapping chaos transition (a sweep)...")
    lyap_data = generate_lyapunov_vs_a_data(n_a=30, n_steps=30000, dt=0.01)

    # Find approximate chaos boundary (last positive -> negative Lyapunov)
    lam = lyap_data["lyapunov_exponent"]
    a_vals = lyap_data["a"]
    zero_crossings = []
    for j in range(len(lam) - 1):
        if lam[j] > 0 and lam[j + 1] <= 0:
            frac = lam[j] / (lam[j] - lam[j + 1])
            a_cross = a_vals[j] + frac * (a_vals[j + 1] - a_vals[j])
            zero_crossings.append(float(a_cross))

    n_chaotic = int(np.sum(lam > 0.01))
    n_stable = int(np.sum(lam < -0.01))

    results["chaos_transition"] = {
        "n_a_values": len(a_vals),
        "n_chaotic": n_chaotic,
        "n_stable": n_stable,
        "a_range": [float(a_vals[0]), float(a_vals[-1])],
        "zero_crossings": zero_crossings,
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }
    if zero_crossings:
        logger.info(f"  Lyapunov zero crossings at a = {zero_crossings}")

    # --- Part 3: Cyclic symmetry verification ---
    logger.info("Part 3: Verifying cyclic symmetry...")
    config_sym = SimulationConfig(
        domain=_HALVORSEN_DOMAIN,
        dt=0.01,
        n_steps=1000,
        parameters={"a": 1.89, "x_0": -5.0, "y_0": 1.0, "z_0": -1.0},
    )
    sim_sym = HalvorsenSimulation(config_sym)
    sim_sym.reset()
    sym_result = sim_sym.cyclic_symmetry_check(n_steps=1000)

    results["cyclic_symmetry"] = {
        "max_deviation": sym_result["max_deviation"],
        "mean_deviation": sym_result["mean_deviation"],
        "verified": sym_result["max_deviation"] < 1e-10,
    }
    logger.info(
        f"  Symmetry max deviation: {sym_result['max_deviation']:.2e}"
    )

    # --- Part 4: Lyapunov at classic a ---
    logger.info("Part 4: Lyapunov at classic a=1.89...")
    config_classic = SimulationConfig(
        domain=_HALVORSEN_DOMAIN,
        dt=0.01,
        n_steps=50000,
        parameters={"a": 1.89, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
    )
    sim_classic = HalvorsenSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.01)

    results["classic_parameters"] = {
        "a": 1.89,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Halvorsen Lyapunov: {lam_classic:.4f}")

    # --- Part 5: Trajectory statistics ---
    logger.info("Part 5: Trajectory statistics on the attractor...")
    config_stats = SimulationConfig(
        domain=_HALVORSEN_DOMAIN,
        dt=0.01,
        n_steps=20000,
        parameters={"a": 1.89, "x_0": -5.0, "y_0": 0.0, "z_0": 0.0},
    )
    sim_stats = HalvorsenSimulation(config_stats)
    traj_stats = sim_stats.compute_trajectory_statistics(
        n_steps=15000, n_transient=5000
    )
    results["trajectory_statistics"] = traj_stats

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
        output_path / "lyapunov_vs_a.npz",
        a=lyap_data["a"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )

    return results
