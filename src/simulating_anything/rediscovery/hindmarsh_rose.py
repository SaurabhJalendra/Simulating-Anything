"""Hindmarsh-Rose rediscovery.

Targets:
- Behavior transitions: quiescent -> spiking -> bursting -> continuous spiking
- Burst structure: spikes per burst as a function of I_ext
- ODE recovery via SINDy: dx/dt = y - a*x^3 + b*x^2 - z + I_ext
- Slow-fast separation: z changes much slower than x, y
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.hindmarsh_rose import HindmarshRoseSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    I_ext: float = 3.25,
    n_steps: int = 20000,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses default Hindmarsh-Rose parameters with specified I_ext.
    Discards initial transient before recording.

    Args:
        I_ext: External current.
        n_steps: Number of integration steps to record.
        dt: Timestep.

    Returns:
        Dict with time, states, x, y, z arrays and I_ext value.
    """
    config = SimulationConfig(
        domain=Domain.HINDMARSH_ROSE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": 1.0, "b": 3.0, "c": 1.0, "d": 5.0,
            "r": 0.001, "s": 4.0, "x_rest": -1.6, "I_ext": I_ext,
            "x_0": -1.5, "y_0": -10.0, "z_0": 2.0,
        },
    )
    sim = HindmarshRoseSimulation(config)
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
        "x": states[:, 0],
        "y": states[:, 1],
        "z": states[:, 2],
        "I_ext": I_ext,
    }


def generate_behavior_sweep(
    n_I: int = 30,
    dt: float = 0.05,
    t_max: int = 10000,
    transient: int = 3000,
) -> dict[str, list]:
    """Sweep I_ext and classify behavior at each value.

    Args:
        n_I: Number of I_ext values.
        dt: Timestep.
        t_max: Simulation steps for classification.
        transient: Transient steps to discard.

    Returns:
        Dict with I_ext, behavior, spikes_per_burst, n_bursts lists.
    """
    I_values = np.linspace(0.0, 5.0, n_I)
    behaviors = []
    spikes_per_burst = []
    n_bursts_list = []

    for i, i_val in enumerate(I_values):
        config = SimulationConfig(
            domain=Domain.HINDMARSH_ROSE,
            dt=dt,
            n_steps=1000,
            parameters={
                "a": 1.0, "b": 3.0, "c": 1.0, "d": 5.0,
                "r": 0.001, "s": 4.0, "x_rest": -1.6, "I_ext": float(i_val),
                "x_0": -1.5, "y_0": -10.0, "z_0": 2.0,
            },
        )
        sim = HindmarshRoseSimulation(config)
        behavior = sim.classify_behavior(t_max=t_max, transient=transient)
        behaviors.append(behavior)

        # Get burst data
        sim.reset()
        for _ in range(transient):
            sim.step()
        x_trace = np.zeros(t_max)
        for j in range(t_max):
            sim.step()
            x_trace[j] = sim.observe()[0]

        bursts = sim.detect_bursts(x_trace, threshold=0.0)
        n_bursts_list.append(len(bursts))
        if len(bursts) > 0:
            mean_spb = np.mean([len(b) for b in bursts])
            spikes_per_burst.append(float(mean_spb))
        else:
            spikes_per_burst.append(0.0)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  I_ext={i_val:.3f}: {behavior}, "
                f"bursts={len(bursts)}, spk/burst={spikes_per_burst[-1]:.1f}"
            )

    return {
        "I_ext": list(I_values),
        "behavior": behaviors,
        "spikes_per_burst": spikes_per_burst,
        "n_bursts": n_bursts_list,
    }


def run_hindmarsh_rose_rediscovery(
    output_dir: str | Path = "output/rediscovery/hindmarsh_rose",
    n_iterations: int = 40,
) -> dict:
    """Run Hindmarsh-Rose rediscovery pipeline.

    Steps:
    1. SINDy ODE recovery from trajectory data
    2. I_ext sweep to map behavior transitions
    3. Identify quiescent/spiking/bursting thresholds

    Args:
        output_dir: Directory for saving results.
        n_iterations: PySR iterations (if used).

    Returns:
        Results dict with all discoveries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "hindmarsh_rose",
        "targets": {
            "ode": "dx/dt=y-ax^3+bx^2-z+I, dy/dt=c-dx^2-y, dz/dt=r(s(x-x_r)-z)",
            "behavior_transitions": "quiescent -> spiking -> bursting",
            "slow_fast": "z changes much slower than x, y (r << 1)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at I_ext=3.25...")
    data = generate_ode_data(I_ext=3.25, n_steps=40000, dt=0.02)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.02,
            feature_names=["x", "y", "z"],
            threshold=0.01,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
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

    # --- Part 2: Behavior sweep ---
    logger.info("Part 2: I_ext behavior sweep...")
    sweep = generate_behavior_sweep(n_I=30, dt=0.05, t_max=10000, transient=3000)

    behavior_counts = {}
    for b in sweep["behavior"]:
        behavior_counts[b] = behavior_counts.get(b, 0) + 1

    # Find transition thresholds
    transitions = {}
    prev_b = sweep["behavior"][0]
    for i in range(1, len(sweep["behavior"])):
        cur_b = sweep["behavior"][i]
        if cur_b != prev_b:
            key = f"{prev_b}_to_{cur_b}"
            transitions[key] = float(sweep["I_ext"][i])
        prev_b = cur_b

    results["behavior_sweep"] = {
        "n_I_values": len(sweep["I_ext"]),
        "behavior_counts": behavior_counts,
        "transitions": transitions,
        "I_ext": sweep["I_ext"],
        "behaviors": sweep["behavior"],
        "spikes_per_burst": sweep["spikes_per_burst"],
    }
    logger.info(f"  Behavior counts: {behavior_counts}")
    logger.info(f"  Transitions: {transitions}")

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
