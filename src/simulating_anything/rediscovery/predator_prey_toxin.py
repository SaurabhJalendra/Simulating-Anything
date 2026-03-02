"""Predator-prey with toxic defense rediscovery.

Targets:
- Toxin effect: increasing c reduces predation efficiency at high N
- c=0 reduces to standard Holling Type II (Rosenzweig-MacArthur)
- Enrichment effect: increasing r changes system stability
- Equilibrium and oscillation detection across parameter sweeps
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.predator_prey_toxin import (
    PredatorPreyToxinSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)

# Use DUFFING as placeholder domain enum
_DOMAIN = Domain.PREDATOR_PREY_TOXIN


def _make_config(
    N_0: float = 50.0,
    P_0: float = 10.0,
    r: float = 1.0,
    K: float = 100.0,
    a: float = 0.5,
    h: float = 0.02,
    e: float = 0.4,
    d: float = 0.2,
    c: float = 0.001,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Build a SimulationConfig for the toxin predator-prey model."""
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "h": h, "e": e, "d": d, "c": c,
            "N_0": N_0, "P_0": P_0,
        },
    )


def generate_toxin_sweep_data(
    n_c_values: int = 20,
    c_min: float = 0.0,
    c_max: float = 0.01,
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep toxin strength c and measure equilibrium/oscillation.

    For each c value, runs a long trajectory and records:
    - Time-averaged prey and predator in second half
    - Oscillation amplitude
    - Whether the system is oscillatory

    Returns dict with c_values, averages, amplitudes, and oscillatory flags.
    """
    c_values = np.linspace(c_min, c_max, n_c_values)

    all_c = []
    all_N_avg = []
    all_P_avg = []
    all_N_amp = []
    all_P_amp = []
    all_oscillatory = []

    for i, c_val in enumerate(c_values):
        config = _make_config(c=float(c_val), dt=dt, n_steps=n_steps)
        sim = PredatorPreyToxinSimulation(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)

        # Use second half to avoid transients
        half = n_steps // 2
        N_half = trajectory[half:, 0]
        P_half = trajectory[half:, 1]

        all_c.append(float(c_val))
        all_N_avg.append(float(np.mean(N_half)))
        all_P_avg.append(float(np.mean(P_half)))
        all_N_amp.append(float(np.max(N_half) - np.min(N_half)))
        all_P_amp.append(float(np.max(P_half) - np.min(P_half)))
        all_oscillatory.append(bool(sim.is_oscillatory(trajectory)))

        if (i + 1) % 5 == 0:
            logger.info(f"  Toxin sweep: {i + 1}/{n_c_values} c values")

    return {
        "c_values": np.array(all_c),
        "N_avg": np.array(all_N_avg),
        "P_avg": np.array(all_P_avg),
        "N_amplitude": np.array(all_N_amp),
        "P_amplitude": np.array(all_P_amp),
        "oscillatory": np.array(all_oscillatory),
    }


def generate_enrichment_data(
    n_r_values: int = 15,
    r_min: float = 0.5,
    r_max: float = 3.0,
    n_steps: int = 50000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep prey growth rate r and measure enrichment effect.

    Higher r can destabilize the system (paradox of enrichment analog).

    Returns dict with r_values, averages, amplitudes, and oscillatory flags.
    """
    r_values = np.linspace(r_min, r_max, n_r_values)

    all_r = []
    all_N_avg = []
    all_P_avg = []
    all_N_amp = []
    all_oscillatory = []

    for i, r_val in enumerate(r_values):
        config = _make_config(r=float(r_val), dt=dt, n_steps=n_steps)
        sim = PredatorPreyToxinSimulation(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(n_steps):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)

        half = n_steps // 2
        N_half = trajectory[half:, 0]
        P_half = trajectory[half:, 1]

        all_r.append(float(r_val))
        all_N_avg.append(float(np.mean(N_half)))
        all_P_avg.append(float(np.mean(P_half)))
        all_N_amp.append(float(np.max(N_half) - np.min(N_half)))
        all_oscillatory.append(bool(sim.is_oscillatory(trajectory)))

        if (i + 1) % 5 == 0:
            logger.info(f"  Enrichment sweep: {i + 1}/{n_r_values} r values")

    return {
        "r_values": np.array(all_r),
        "N_avg": np.array(all_N_avg),
        "P_avg": np.array(all_P_avg),
        "N_amplitude": np.array(all_N_amp),
        "oscillatory": np.array(all_oscillatory),
    }


def generate_trajectory_data(
    N_0: float = 50.0,
    P_0: float = 10.0,
    c: float = 0.001,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Generate a single trajectory for SINDy ODE recovery.

    Returns dict with states array, time, and parameters.
    """
    config = _make_config(N_0=N_0, P_0=P_0, c=c, dt=dt, n_steps=n_steps)
    sim = PredatorPreyToxinSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "N": states_arr[:, 0],
        "P": states_arr[:, 1],
        "dt": dt,
        "c": c,
    }


def generate_functional_response_comparison(
    N_values: np.ndarray | None = None,
    c_values: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compare functional response with and without toxin.

    For multiple c values, compute the functional response curve
    as a function of N, showing how toxin suppresses consumption
    at high prey density.

    Returns dict with N_values and response curves.
    """
    if N_values is None:
        N_values = np.linspace(0, 200, 100)
    if c_values is None:
        c_values = np.array([0.0, 0.0005, 0.001, 0.002, 0.005])

    config = _make_config()
    sim = PredatorPreyToxinSimulation(config)

    # Holling Type II baseline (c=0)
    holling2 = sim.functional_response_holling2(N_values)

    responses = {}
    for c_val in c_values:
        # Temporarily set c to compute response
        original_c = sim.c
        sim.c = float(c_val)
        responses[f"c_{c_val:.4f}"] = sim.functional_response(N_values).copy()
        sim.c = original_c

    return {
        "N_values": N_values,
        "holling2": holling2,
        "c_values": c_values,
        **responses,
    }


def run_predator_prey_toxin_rediscovery(
    output_dir: str | Path = "output/rediscovery/predator_prey_toxin",
    n_iterations: int = 40,
) -> dict:
    """Run the full predator-prey toxin rediscovery pipeline.

    1. Sweep toxin strength c from 0 to 0.01 (20 values)
    2. Sweep growth rate r from 0.5 to 3.0 (15 values)
    3. Compare functional response with and without toxin
    4. Compute equilibria and verify dynamics

    Args:
        output_dir: Directory for output files.
        n_iterations: Number of PySR iterations (for future use).

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "predator_prey_toxin",
        "targets": {
            "toxin_effect": "c*N^2 in denominator reduces predation at high N",
            "holling_limit": "c=0 reduces to Holling Type II",
            "prey_ode": "dN/dt = r*N*(1-N/K) - a*N*P/(1+a*h*N+c*N^2)",
            "pred_ode": "dP/dt = e*a*N*P/(1+a*h*N+c*N^2) - d*P",
        },
    }

    # --- Part 1: Toxin sweep ---
    logger.info("Part 1: Toxin strength sweep...")
    toxin_data = generate_toxin_sweep_data(
        n_c_values=20, c_min=0.0, c_max=0.01,
        n_steps=50000, dt=0.01,
    )
    n_oscillatory = int(np.sum(toxin_data["oscillatory"]))
    results["toxin_sweep"] = {
        "n_c_values": len(toxin_data["c_values"]),
        "c_range": [float(toxin_data["c_values"][0]),
                     float(toxin_data["c_values"][-1])],
        "n_oscillatory": n_oscillatory,
        "n_stable": len(toxin_data["c_values"]) - n_oscillatory,
        "max_N_amplitude": float(np.max(toxin_data["N_amplitude"])),
        "N_avg_at_c0": float(toxin_data["N_avg"][0]),
        "N_avg_at_cmax": float(toxin_data["N_avg"][-1]),
    }
    logger.info(
        f"  Toxin sweep: {n_oscillatory} oscillatory, "
        f"{len(toxin_data['c_values']) - n_oscillatory} stable"
    )

    # --- Part 2: Enrichment sweep ---
    logger.info("Part 2: Enrichment (r) sweep...")
    enrichment_data = generate_enrichment_data(
        n_r_values=15, r_min=0.5, r_max=3.0,
        n_steps=50000, dt=0.01,
    )
    n_osc_r = int(np.sum(enrichment_data["oscillatory"]))
    results["enrichment_sweep"] = {
        "n_r_values": len(enrichment_data["r_values"]),
        "r_range": [float(enrichment_data["r_values"][0]),
                     float(enrichment_data["r_values"][-1])],
        "n_oscillatory": n_osc_r,
        "max_N_amplitude": float(np.max(enrichment_data["N_amplitude"])),
    }
    logger.info(
        f"  Enrichment sweep: {n_osc_r} oscillatory out of "
        f"{len(enrichment_data['r_values'])}"
    )

    # --- Part 3: Functional response comparison ---
    logger.info("Part 3: Functional response comparison...")
    generate_functional_response_comparison()
    # Check that toxin reduces response at high N
    N_high = 150.0
    config_base = _make_config(c=0.0)
    sim_base = PredatorPreyToxinSimulation(config_base)
    fr_c0 = sim_base.functional_response(N_high)
    config_toxin = _make_config(c=0.005)
    sim_toxin = PredatorPreyToxinSimulation(config_toxin)
    fr_c005 = sim_toxin.functional_response(N_high)

    results["functional_response"] = {
        "fr_at_N150_c0": float(fr_c0),
        "fr_at_N150_c005": float(fr_c005),
        "toxin_reduces_predation": bool(fr_c005 < fr_c0),
        "reduction_fraction": float(1.0 - fr_c005 / fr_c0) if fr_c0 > 0 else 0.0,
    }
    logger.info(
        f"  f(150) at c=0: {fr_c0:.4f}, c=0.005: {fr_c005:.4f} "
        f"(reduction: {(1 - fr_c005 / fr_c0) * 100:.1f}%)"
    )

    # --- Part 4: Equilibria ---
    logger.info("Part 4: Computing equilibria...")
    config = _make_config()
    sim = PredatorPreyToxinSimulation(config)
    equilibria = sim.find_equilibria()
    results["equilibria"] = equilibria
    for eq in equilibria:
        logger.info(f"  {eq['type']}: N={eq['N']:.4f}, P={eq['P']:.4f}")

    # --- Part 5: SINDy ODE recovery ---
    logger.info("Part 5: Trajectory generation for ODE recovery...")
    traj_data = generate_trajectory_data(
        N_0=50.0, P_0=10.0, c=0.001,
        n_steps=10000, dt=0.005,
    )
    results["trajectory"] = {
        "n_points": len(traj_data["time"]),
        "dt": traj_data["dt"],
        "final_N": float(traj_data["N"][-1]),
        "final_P": float(traj_data["P"][-1]),
    }

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            traj_data["states"],
            dt=0.005,
            feature_names=["N", "P"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {"expression": disc.expression,
                 "r_squared": disc.evidence.fit_r_squared}
                for disc in sindy_discoveries
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
    except Exception as exc:
        logger.warning(f"SINDy failed: {exc}")
        results["sindy_ode"] = {"error": str(exc)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save sweep data
    np.savez(
        output_path / "toxin_sweep_data.npz",
        **{k: v for k, v in toxin_data.items()},
    )
    np.savez(
        output_path / "enrichment_data.npz",
        **{k: v for k, v in enrichment_data.items()},
    )

    return results
