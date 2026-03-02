"""Nose-Hoover thermostat rediscovery.

Targets:
- SINDy recovery of Nose-Hoover ODEs: x'=y, y'=-x+y*z, z'=a-y^2
- Positive Lyapunov exponent at a=1.0 (chaos)
- Time-averaged divergence near zero (measure preservation)
- Temperature equilibration: <y^2> ~ a
- Parameter sweep: chaos as a function of a
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.nose_hoover import NoseHooverSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    a: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate a single Nose-Hoover trajectory for SINDy ODE recovery.

    Uses chaotic parameters by default (a=1.0).
    """
    config = SimulationConfig(
        domain=Domain.NOSE_HOOVER,
        dt=dt,
        n_steps=n_steps,
        parameters={"a": a, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
    )
    sim = NoseHooverSimulation(config)
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
    """Sweep parameter a to map chaotic vs periodic behavior.

    The Nose-Hoover system shows different dynamics depending on a:
    - a=1.0: chaotic (classic value)
    - Very small a: quasiperiodic or periodic
    - Large a: complex dynamics
    """
    a_values = np.linspace(0.1, 3.0, n_a)
    lyapunov_exps = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=dt,
            n_steps=n_steps,
            parameters={"a": a, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
        )
        sim = NoseHooverSimulation(config)
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


def generate_temperature_data(
    a_values: np.ndarray | None = None,
    n_steps: int = 20000,
    n_transient: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Verify thermostat property: <y^2> ~ a across parameter values.

    The Nose-Hoover thermostat equilibrates kinetic energy y^2 to
    the target value a on the attractor.
    """
    if a_values is None:
        a_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    mean_y2_values = []

    for a in a_values:
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=dt,
            n_steps=n_steps + n_transient,
            parameters={"a": a, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
        )
        sim = NoseHooverSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(n_transient):
            sim.step()

        y2_sum = 0.0
        for _ in range(n_steps):
            state = sim.step()
            y2_sum += state[1] ** 2

        mean_y2_values.append(y2_sum / n_steps)

    return {
        "a": a_values,
        "mean_y_squared": np.array(mean_y2_values),
    }


def run_nose_hoover_rediscovery(
    output_dir: str | Path = "output/rediscovery/nose_hoover",
    n_iterations: int = 40,
) -> dict:
    """Run the full Nose-Hoover thermostat rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep a to map chaos transition (Lyapunov exponent)
    3. Verify measure preservation (time-averaged divergence ~ 0)
    4. Verify temperature equilibration (<y^2> ~ a)
    5. Compute Lyapunov at classic a=1.0

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "nose_hoover",
        "targets": {
            "ode_x": "dx/dt = y",
            "ode_y": "dy/dt = -x + y*z",
            "ode_z": "dz/dt = a - y^2",
            "chaos_regime": "a = 1.0 (classic chaotic parameter)",
            "thermostat": "<y^2> = a (temperature equilibration)",
            "measure": "time-averaged div(F) = <z> ~ 0",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Nose-Hoover trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.01, a=1.0)

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
    logger.info("Part 2: Mapping chaos as a function of a...")
    lyap_data = generate_lyapunov_vs_a_data(n_a=30, n_steps=30000, dt=0.01)

    lam = lyap_data["lyapunov_exponent"]
    a_vals = lyap_data["a"]
    n_chaotic = int(np.sum(lam > 0.01))
    n_stable = int(np.sum(lam < -0.01))

    results["chaos_sweep"] = {
        "n_a_values": len(a_vals),
        "n_chaotic": n_chaotic,
        "n_stable": n_stable,
        "a_range": [float(a_vals[0]), float(a_vals[-1])],
        "max_lyapunov": float(np.max(lam)),
        "min_lyapunov": float(np.min(lam)),
    }
    logger.info(
        f"  Found {n_chaotic} chaotic, {n_stable} stable out of {len(a_vals)}"
    )

    # --- Part 3: Measure preservation ---
    logger.info("Part 3: Verifying measure preservation...")
    config_vol = SimulationConfig(
        domain=Domain.NOSE_HOOVER,
        dt=0.01,
        n_steps=20000,
        parameters={"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
    )
    sim_vol = NoseHooverSimulation(config_vol)
    sim_vol.reset()
    vol_result = sim_vol.check_volume_preservation(
        n_steps=15000, n_transient=5000
    )
    results["volume_preservation"] = vol_result
    logger.info(
        f"  Mean divergence: {vol_result['mean_divergence']:.4f} "
        f"(should be near 0)"
    )

    # --- Part 4: Temperature equilibration ---
    logger.info("Part 4: Verifying temperature equilibration...")
    temp_data = generate_temperature_data()
    results["temperature_equilibration"] = {
        "a_values": temp_data["a"].tolist(),
        "mean_y_squared": temp_data["mean_y_squared"].tolist(),
        "correlation": float(np.corrcoef(
            temp_data["a"], temp_data["mean_y_squared"]
        )[0, 1]),
    }
    logger.info(
        f"  <y^2> vs a correlation: "
        f"{results['temperature_equilibration']['correlation']:.4f}"
    )

    # --- Part 5: Lyapunov at classic a=1.0 ---
    logger.info("Part 5: Lyapunov exponent at classic a=1.0...")
    config_classic = SimulationConfig(
        domain=Domain.NOSE_HOOVER,
        dt=0.01,
        n_steps=50000,
        parameters={"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
    )
    sim_classic = NoseHooverSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.01)

    results["classic_parameters"] = {
        "a": 1.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Nose-Hoover Lyapunov: {lam_classic:.4f}")

    # Trajectory statistics at chaotic parameters
    config_stats = SimulationConfig(
        domain=Domain.NOSE_HOOVER,
        dt=0.01,
        n_steps=20000,
        parameters={"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
    )
    sim_stats = NoseHooverSimulation(config_stats)
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
