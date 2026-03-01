"""FitzHugh-Rinzel rediscovery.

Targets:
- ODE recovery via SINDy: dv/dt = v - v^3/3 - w + y + I_ext,
  dw/dt = delta*(a + v - b*w), dy/dt = mu*(c - v - d*y)
- Bursting dynamics: clusters of spikes with quiescent intervals
- Burst statistics as a function of mu (ultraslow timescale)
- FHN limit: mu=0 gives standard FitzHugh-Nagumo behavior
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.fitzhugh_rinzel import FitzHughRinzelSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    I_ext: float = 0.3,
    mu: float = 0.0001,
    n_steps: int = 20000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses default FitzHugh-Rinzel parameters with specified I_ext and mu.
    Discards initial transient before recording.

    Args:
        I_ext: External current.
        mu: Ultraslow timescale parameter.
        n_steps: Number of integration steps to record.
        dt: Timestep.

    Returns:
        Dict with time, states, v, w, y arrays and parameter values.
    """
    config = SimulationConfig(
        domain=Domain.FITZHUGH_RINZEL,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": 0.7, "b": 0.8, "c": -0.775, "d": 1.0,
            "delta": 0.08, "mu": mu, "I_ext": I_ext,
            "v_0": -1.0, "w_0": -0.5, "y_0": 0.0,
        },
    )
    sim = FitzHughRinzelSimulation(config)
    sim.reset()

    # Skip transient
    for _ in range(5000):
        sim.step()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "v": states[:, 0],
        "w": states[:, 1],
        "y": states[:, 2],
        "I_ext": I_ext,
        "mu": mu,
    }


def generate_mu_sweep(
    n_mu: int = 20,
    dt: float = 0.1,
    n_steps: int = 200000,
    transient: int = 50000,
) -> dict[str, list]:
    """Sweep mu and measure burst statistics at each value.

    Args:
        n_mu: Number of mu values.
        dt: Timestep.
        n_steps: Simulation steps per mu value.
        transient: Transient steps to discard.

    Returns:
        Dict with mu, n_bursts, burst_frequency lists.
    """
    mu_values = np.logspace(-5, -2, n_mu)
    config = SimulationConfig(
        domain=Domain.FITZHUGH_RINZEL,
        dt=dt,
        n_steps=1000,
        parameters={
            "a": 0.7, "b": 0.8, "c": -0.775, "d": 1.0,
            "delta": 0.08, "mu": 0.0001, "I_ext": 0.3,
            "v_0": -1.0, "w_0": -0.5, "y_0": 0.0,
        },
    )
    sim = FitzHughRinzelSimulation(config)
    result = sim.mu_sweep(
        mu_values, n_steps=n_steps, transient=transient
    )

    for i, mu_val in enumerate(mu_values):
        if (i + 1) % 5 == 0:
            logger.info(
                f"  mu={mu_val:.6f}: "
                f"bursts={result['n_bursts'][i]}, "
                f"freq={result['burst_frequency'][i]:.6f}"
            )

    return result


def run_fitzhugh_rinzel_rediscovery(
    output_dir: str | Path = "output/rediscovery/fitzhugh_rinzel",
    n_iterations: int = 40,
) -> dict:
    """Run FitzHugh-Rinzel rediscovery pipeline.

    Steps:
    1. SINDy ODE recovery from trajectory data
    2. Burst statistics measurement
    3. mu sweep to map burst frequency vs ultraslow timescale

    Args:
        output_dir: Directory for saving results.
        n_iterations: PySR iterations (if used).

    Returns:
        Results dict with all discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "fitzhugh_rinzel",
        "targets": {
            "ode": (
                "dv/dt=v-v^3/3-w+y+I, "
                "dw/dt=delta*(a+v-b*w), "
                "dy/dt=mu*(c-v-d*y)"
            ),
            "bursting": "clusters of spikes with quiescent intervals",
            "mu_sweep": "burst frequency vs ultraslow timescale",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at I_ext=0.3, mu=0.0001...")
    data = generate_ode_data(I_ext=0.3, mu=0.0001, n_steps=40000, dt=0.05)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.05,
            feature_names=["v", "w", "y"],
            threshold=0.01,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries[:5]
            ],
        }
        if sindy_discoveries:
            best = sindy_discoveries[0]
            results["sindy_ode"]["best"] = best.expression
            results["sindy_ode"]["best_r2"] = best.evidence.fit_r_squared
            logger.info(
                f"  SINDy best: {best.expression} "
                f"(R2={best.evidence.fit_r_squared:.6f})"
            )
    except Exception as e:
        logger.warning(f"SINDy failed: {e}")
        results["sindy_ode"] = {"error": str(e)}

    # --- Part 2: Burst statistics ---
    logger.info("Part 2: Burst statistics...")
    config = SimulationConfig(
        domain=Domain.FITZHUGH_RINZEL,
        dt=0.1,
        n_steps=1000,
        parameters={
            "a": 0.7, "b": 0.8, "c": -0.775, "d": 1.0,
            "delta": 0.08, "mu": 0.0001, "I_ext": 0.3,
            "v_0": -1.0, "w_0": -0.5, "y_0": 0.0,
        },
    )
    sim = FitzHughRinzelSimulation(config)
    burst_stats = sim.measure_burst_statistics(
        n_steps=200000, transient=50000, min_gap_steps=2000
    )
    results["burst_statistics"] = burst_stats
    logger.info(f"  Bursts detected: {burst_stats['n_bursts']}")
    logger.info(
        f"  Spikes/burst: {burst_stats['spikes_per_burst']:.1f}"
    )
    logger.info(
        f"  Interburst interval: {burst_stats['interburst_interval']:.1f}"
    )

    # --- Part 3: mu sweep ---
    logger.info("Part 3: mu sweep...")
    mu_sweep = generate_mu_sweep(n_mu=15, dt=0.1, n_steps=100000, transient=20000)
    results["mu_sweep"] = mu_sweep

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
