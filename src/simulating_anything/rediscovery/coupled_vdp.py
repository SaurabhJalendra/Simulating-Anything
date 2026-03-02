"""Coupled Van der Pol oscillators rediscovery.

Targets:
- Synchronization transition: sync error vs coupling strength k
- In-phase vs anti-phase mode selection depending on initial conditions
- ODE recovery via SINDy for the coupled system
- Period and amplitude invariance under coupling
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.coupled_vdp import CoupledVdPSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    mu: float = 1.0,
    k: float = 0.5,
    n_steps: int = 5000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Runs the coupled system and records the full state trajectory for
    use with sparse identification of nonlinear dynamics.

    Args:
        mu: Nonlinearity parameter.
        k: Coupling strength.
        n_steps: Number of integration steps.
        dt: Timestep.

    Returns:
        Dict with 'time', 'states', and individual variable arrays.
    """
    config = SimulationConfig(
        domain=Domain.COUPLED_VDP,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "mu": mu, "k": k,
            "x1_0": 0.5, "y1_0": 0.0,
            "x2_0": -0.3, "y2_0": 0.1,
        },
    )
    sim = CoupledVdPSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    times = np.arange(n_steps + 1) * dt

    return {
        "time": times,
        "states": states_arr,
        "x1": states_arr[:, 0],
        "y1": states_arr[:, 1],
        "x2": states_arr[:, 2],
        "y2": states_arr[:, 3],
        "mu": mu,
        "k": k,
        "dt": dt,
    }


def generate_coupling_sweep_data(
    n_k: int = 25,
    k_min: float = 0.0,
    k_max: float = 5.0,
    mu: float = 1.0,
    n_steps: int = 8000,
    n_transient: int = 5000,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep coupling strength and measure steady-state sync metrics.

    For each k value, runs the coupled system from anti-phase initial
    conditions, discards transient, and measures the time-averaged
    synchronization error and phase difference.

    Args:
        n_k: Number of coupling values to test.
        k_min: Minimum coupling strength.
        k_max: Maximum coupling strength.
        mu: Nonlinearity parameter.
        n_steps: Measurement steps after transient.
        n_transient: Transient steps to discard.
        dt: Timestep.

    Returns:
        Dict with 'k', 'mean_error', and 'mean_phase_diff' arrays.
    """
    k_values = np.linspace(k_min, k_max, n_k)
    mean_errors = np.empty(n_k)
    mean_phases = np.empty(n_k)

    for i, k_val in enumerate(k_values):
        config = SimulationConfig(
            domain=Domain.COUPLED_VDP,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "mu": mu, "k": float(k_val),
                "x1_0": 2.0, "y1_0": 0.0,
                "x2_0": -2.0, "y2_0": 0.0,
            },
        )
        sim = CoupledVdPSimulation(config)
        sim.reset()

        for _ in range(n_transient):
            sim.step()

        error_sum = 0.0
        phase_sum = 0.0
        for _ in range(n_steps):
            sim.step()
            error_sum += sim.compute_sync_error()
            phase_sum += sim.compute_phase_difference()

        mean_errors[i] = error_sum / n_steps
        mean_phases[i] = phase_sum / n_steps

        if (i + 1) % 5 == 0:
            logger.info(
                f"  k={k_val:.2f}: mean_error={mean_errors[i]:.4f}, "
                f"phase_diff={mean_phases[i]:.4f}"
            )

    return {
        "k": k_values,
        "mean_error": mean_errors,
        "mean_phase_diff": mean_phases,
    }


def generate_mode_selection_data(
    k: float = 0.5,
    mu: float = 1.0,
    n_steps: int = 15000,
    dt: float = 0.005,
) -> dict[str, dict]:
    """Compare in-phase vs anti-phase outcomes from different ICs.

    Runs two simulations at the same coupling -- one starting in-phase,
    the other starting anti-phase -- and measures the final phase
    relationship.

    Args:
        k: Coupling strength.
        mu: Nonlinearity parameter.
        n_steps: Total integration steps.
        dt: Timestep.

    Returns:
        Dict with 'in_phase_ic' and 'anti_phase_ic' result sub-dicts.
    """
    results = {}

    # In-phase initial conditions
    config_ip = SimulationConfig(
        domain=Domain.COUPLED_VDP,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "mu": mu, "k": k,
            "x1_0": 2.0, "y1_0": 0.0,
            "x2_0": 2.0, "y2_0": 0.0,
        },
    )
    sim_ip = CoupledVdPSimulation(config_ip)
    sim_ip.reset()
    for _ in range(n_steps):
        sim_ip.step()
    results["in_phase_ic"] = {
        "final_sync_error": sim_ip.compute_sync_error(),
        "final_phase_diff": sim_ip.compute_phase_difference(),
        "is_in_phase": sim_ip.is_in_phase(threshold=0.5),
        "is_anti_phase": sim_ip.is_anti_phase(threshold=0.5),
    }

    # Anti-phase initial conditions
    config_ap = SimulationConfig(
        domain=Domain.COUPLED_VDP,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "mu": mu, "k": k,
            "x1_0": 2.0, "y1_0": 0.0,
            "x2_0": -2.0, "y2_0": 0.0,
        },
    )
    sim_ap = CoupledVdPSimulation(config_ap)
    sim_ap.reset()
    for _ in range(n_steps):
        sim_ap.step()
    results["anti_phase_ic"] = {
        "final_sync_error": sim_ap.compute_sync_error(),
        "final_phase_diff": sim_ap.compute_phase_difference(),
        "is_in_phase": sim_ap.is_in_phase(threshold=0.5),
        "is_anti_phase": sim_ap.is_anti_phase(threshold=0.5),
    }

    return results


def run_coupled_vdp_rediscovery(
    output_dir: str | Path = "output/rediscovery/coupled_vdp",
    n_iterations: int = 40,
) -> dict:
    """Run the full coupled Van der Pol rediscovery pipeline.

    1. Generate trajectory for SINDy ODE recovery
    2. Sweep coupling strength k to find synchronization transition
    3. Analyze in-phase vs anti-phase mode selection
    4. Measure period and amplitude under coupling

    Args:
        output_dir: Directory for saving results and data files.
        n_iterations: Number of PySR iterations.

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "coupled_vdp",
        "targets": {
            "ode": (
                "dx1/dt=y1, dy1/dt=mu*(1-x1^2)*y1 - x1 + k*(x2-x1), "
                "dx2/dt=y2, dy2/dt=mu*(1-x2^2)*y2 - x2 + k*(x1-x2)"
            ),
            "sync_transition": "Sync error decreases with increasing k",
            "mode_selection": "In-phase vs anti-phase depending on ICs",
            "amplitude": "A ~ 2 (limit cycle amplitude preserved under coupling)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for coupled VdP (mu=1, k=0.5)...")
    ode_data = generate_ode_data(mu=1.0, k=0.5, n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["x1", "y1", "x2", "y2"],
            threshold=0.05,
            poly_degree=3,
        )
        results["sindy_ode"] = {
            "n_discoveries": len(sindy_discoveries),
            "discoveries": [
                {
                    "expression": d.expression,
                    "r_squared": d.evidence.fit_r_squared,
                }
                for d in sindy_discoveries[:8]
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

    # --- Part 2: Coupling strength sweep ---
    logger.info("Part 2: Sweeping coupling strength k...")
    sweep_data = generate_coupling_sweep_data(
        n_k=25, k_min=0.0, k_max=5.0,
        mu=1.0, n_steps=8000, n_transient=5000, dt=0.005,
    )

    results["coupling_sweep"] = {
        "n_k_values": len(sweep_data["k"]),
        "k_range": [float(sweep_data["k"][0]), float(sweep_data["k"][-1])],
        "min_error": float(np.min(sweep_data["mean_error"])),
        "max_error": float(np.max(sweep_data["mean_error"])),
        "error_at_k0": float(sweep_data["mean_error"][0]),
        "error_at_kmax": float(sweep_data["mean_error"][-1]),
    }

    # Check that error generally decreases with coupling
    low_k_error = np.mean(sweep_data["mean_error"][:5])
    high_k_error = np.mean(sweep_data["mean_error"][-5:])
    results["coupling_sweep"]["low_k_mean_error"] = float(low_k_error)
    results["coupling_sweep"]["high_k_mean_error"] = float(high_k_error)
    results["coupling_sweep"]["error_decreases"] = bool(
        high_k_error < low_k_error
    )

    # --- Part 3: Mode selection analysis ---
    logger.info("Part 3: Analyzing in-phase vs anti-phase mode selection...")
    mode_data = generate_mode_selection_data(k=0.5, mu=1.0)

    results["mode_selection"] = {
        "k": 0.5,
        "mu": 1.0,
        "in_phase_ic": {
            k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
            for k, v in mode_data["in_phase_ic"].items()
        },
        "anti_phase_ic": {
            k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
            for k, v in mode_data["anti_phase_ic"].items()
        },
    }

    logger.info(
        f"  In-phase IC -> in_phase={mode_data['in_phase_ic']['is_in_phase']}, "
        f"anti_phase={mode_data['in_phase_ic']['is_anti_phase']}"
    )
    logger.info(
        f"  Anti-phase IC -> in_phase={mode_data['anti_phase_ic']['is_in_phase']}, "
        f"anti_phase={mode_data['anti_phase_ic']['is_anti_phase']}"
    )

    # --- Part 4: Period and amplitude measurement ---
    logger.info("Part 4: Measuring period and amplitude under coupling...")
    k_test_values = [0.0, 0.1, 0.5, 1.0, 2.0]
    amp_results = []

    for k_val in k_test_values:
        config = SimulationConfig(
            domain=Domain.COUPLED_VDP,
            dt=0.005,
            n_steps=1000,
            parameters={
                "mu": 1.0, "k": k_val,
                "x1_0": 0.1, "y1_0": 0.0,
                "x2_0": -0.1, "y2_0": 0.0,
            },
        )
        sim = CoupledVdPSimulation(config)
        sim.reset()
        a1, a2 = sim.measure_amplitudes(n_periods=3)
        amp_results.append({
            "k": k_val,
            "amplitude_1": a1,
            "amplitude_2": a2,
        })
        logger.info(f"  k={k_val:.1f}: A1={a1:.4f}, A2={a2:.4f}")

    results["amplitude_vs_k"] = amp_results

    # --- Save results ---
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "coupling_sweep.npz",
        k=sweep_data["k"],
        mean_error=sweep_data["mean_error"],
        mean_phase_diff=sweep_data["mean_phase_diff"],
    )

    return results
