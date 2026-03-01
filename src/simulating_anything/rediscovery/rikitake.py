"""Rikitake dynamo rediscovery.

Targets:
- SINDy recovery of Rikitake ODEs: x'=-mu*x+z*y, y'=-mu*y+(z-a)*x, z'=1-x*y
- Chaotic polarity reversals and sensitivity to asymmetry parameter a
- Lyapunov exponent as a function of a
- Fixed point verification
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.rikitake import RikitakeSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    n_steps: int = 10000,
    dt: float = 0.01,
    mu: float = 1.0,
    a: float = 5.0,
) -> dict[str, np.ndarray]:
    """Generate a single Rikitake trajectory for SINDy ODE recovery.

    Uses default chaotic parameters.
    """
    config = SimulationConfig(
        domain=Domain.RIKITAKE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "mu": mu,
            "a": a,
            "x_0": 1.0,
            "y_0": 1.0,
            "z_0": 0.0,
        },
    )
    sim = RikitakeSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        states.append(sim.step().copy())

    return {
        "states": np.array(states),
        "dt": dt,
        "mu": mu,
        "a": a,
    }


def generate_reversal_data(
    n_a: int = 30,
    n_transient: int = 5000,
    n_measure: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep asymmetry parameter a to study polarity reversal behavior.

    For each a, count the number of polarity reversals and compute
    the mean interval between reversals.
    """
    a_values = np.linspace(1.0, 10.0, n_a)
    n_reversals = []
    mean_intervals = []
    x_stds = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=Domain.RIKITAKE,
            dt=dt,
            n_steps=1000,
            parameters={"mu": 1.0, "a": a, "x_0": 1.0, "y_0": 1.0, "z_0": 0.0},
        )
        sim = RikitakeSimulation(config)
        sim.reset()

        reversal_info = sim.count_reversals(
            n_transient=n_transient,
            n_measure=n_measure,
        )
        n_reversals.append(reversal_info["n_reversals"])
        mean_intervals.append(reversal_info["mean_interval"])
        x_stds.append(reversal_info["x_std"])

        if (i + 1) % 10 == 0:
            logger.info(
                f"  a={a:.2f}: reversals={reversal_info['n_reversals']}, "
                f"mean_interval={reversal_info['mean_interval']:.2f}"
            )

    return {
        "a": a_values,
        "n_reversals": np.array(n_reversals),
        "mean_interval": np.array(mean_intervals),
        "x_std": np.array(x_stds),
    }


def generate_lyapunov_vs_a_data(
    n_a: int = 30,
    n_steps: int = 30000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep a to compute the Lyapunov exponent across parameter space.

    Maps the chaos transition as a function of the asymmetry parameter.
    """
    a_values = np.linspace(1.0, 10.0, n_a)
    lyapunov_exps = []

    for i, a in enumerate(a_values):
        config = SimulationConfig(
            domain=Domain.RIKITAKE,
            dt=dt,
            n_steps=n_steps,
            parameters={"mu": 1.0, "a": a, "x_0": 1.0, "y_0": 1.0, "z_0": 0.0},
        )
        sim = RikitakeSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(5000):
            sim.step()

        lam = sim.estimate_lyapunov(n_steps=n_steps, dt=dt)
        lyapunov_exps.append(lam)

        if (i + 1) % 10 == 0:
            logger.info(f"  a={a:.2f}: Lyapunov={lam:.4f}")

    return {
        "a": a_values,
        "lyapunov_exponent": np.array(lyapunov_exps),
    }


def run_rikitake_rediscovery(
    output_dir: str | Path = "output/rediscovery/rikitake",
    n_iterations: int = 40,
) -> dict:
    """Run the full Rikitake dynamo rediscovery.

    1. Generate chaotic trajectory for SINDy ODE recovery
    2. Sweep a to map reversal statistics
    3. Sweep a for Lyapunov exponent
    4. Verify fixed points

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "rikitake",
        "targets": {
            "ode_x": "dx/dt = -mu*x + z*y",
            "ode_y": "dy/dt = -mu*y + (z - a)*x",
            "ode_z": "dz/dt = 1 - x*y",
            "chaotic_reversals": "Polarity reversals in x",
            "fixed_points": "Two symmetric equilibria",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: Generating Rikitake trajectory for SINDy...")
    ode_data = generate_ode_data(n_steps=10000, dt=0.01)

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
            "true_mu": ode_data["mu"],
            "true_a": ode_data["a"],
        }
        for d in sindy_discoveries:
            logger.info(f"  SINDy: {d.expression}")
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Reversal statistics ---
    logger.info("Part 2: Mapping polarity reversal behavior (a sweep)...")
    reversal_data = generate_reversal_data(n_a=30, n_transient=5000, n_measure=50000)

    total_reversals = int(np.sum(reversal_data["n_reversals"]))
    mean_rev = float(np.mean(reversal_data["n_reversals"]))
    logger.info(
        f"  Total reversals across sweep: {total_reversals}, "
        f"mean per a value: {mean_rev:.1f}"
    )

    results["reversal_statistics"] = {
        "n_a_values": len(reversal_data["a"]),
        "a_range": [float(reversal_data["a"][0]), float(reversal_data["a"][-1])],
        "total_reversals": total_reversals,
        "mean_reversals_per_a": mean_rev,
        "max_reversals": int(np.max(reversal_data["n_reversals"])),
        "min_reversals": int(np.min(reversal_data["n_reversals"])),
    }

    # --- Part 3: Lyapunov exponent sweep ---
    logger.info("Part 3: Computing Lyapunov exponent vs a...")
    lyap_data = generate_lyapunov_vs_a_data(n_a=30, n_steps=30000, dt=0.01)

    n_chaotic = int(np.sum(lyap_data["lyapunov_exponent"] > 0.01))
    max_lyap = float(np.max(lyap_data["lyapunov_exponent"]))
    logger.info(
        f"  {n_chaotic}/{len(lyap_data['a'])} chaotic regimes, "
        f"max Lyapunov: {max_lyap:.4f}"
    )

    results["lyapunov_sweep"] = {
        "n_a_values": len(lyap_data["a"]),
        "a_range": [float(lyap_data["a"][0]), float(lyap_data["a"][-1])],
        "n_chaotic": n_chaotic,
        "max_lyapunov": max_lyap,
        "mean_lyapunov": float(np.mean(lyap_data["lyapunov_exponent"])),
    }

    # --- Part 4: Lyapunov at classic parameters ---
    logger.info("Part 4: Lyapunov exponent at classic parameters (mu=1, a=5)...")
    config_classic = SimulationConfig(
        domain=Domain.RIKITAKE,
        dt=0.01,
        n_steps=50000,
        parameters={"mu": 1.0, "a": 5.0},
    )
    sim_classic = RikitakeSimulation(config_classic)
    sim_classic.reset()
    for _ in range(10000):
        sim_classic.step()
    lam_classic = sim_classic.estimate_lyapunov(n_steps=50000, dt=0.01)

    results["classic_parameters"] = {
        "mu": 1.0,
        "a": 5.0,
        "lyapunov_exponent": float(lam_classic),
        "positive": bool(lam_classic > 0),
    }
    logger.info(f"  Classic Rikitake Lyapunov: {lam_classic:.4f}")

    # --- Part 5: Fixed points ---
    logger.info("Part 5: Verifying fixed points...")
    sim_fp = RikitakeSimulation(config_classic)
    sim_fp.reset()
    fps = sim_fp.fixed_points
    fp_data = []
    for i, fp in enumerate(fps):
        derivs = sim_fp._derivatives(fp)
        norm = float(np.linalg.norm(derivs))
        fp_data.append({
            "point": fp.tolist(),
            "deriv_norm": norm,
        })
        logger.info(
            f"  FP{i+1}: [{fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}], "
            f"|deriv|={norm:.2e}"
        )

    results["fixed_points"] = {
        "n_fixed_points": len(fps),
        "points": fp_data,
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
        output_path / "reversal_data.npz",
        a=reversal_data["a"],
        n_reversals=reversal_data["n_reversals"],
        mean_interval=reversal_data["mean_interval"],
    )
    np.savez(
        output_path / "lyapunov_data.npz",
        a=lyap_data["a"],
        lyapunov_exponent=lyap_data["lyapunov_exponent"],
    )

    return results
