"""FitzHugh-Nagumo coupled pair rediscovery.

Targets:
- Synchronization transition: sync error vs coupling strength g
- Critical coupling g_c for sync onset
- Phase-locking under heterogeneous drive (I1 != I2)
- ODE recovery via SINDy for the coupled system
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.fhn_coupled_pair import FHNCoupledPairSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    g: float = 0.1,
    I1: float = 0.5,
    I2: float = 0.5,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Runs the coupled system and records the full state trajectory for
    use with sparse identification of nonlinear dynamics.

    Args:
        g: Coupling strength.
        I1: External current to neuron 1.
        I2: External current to neuron 2.
        n_steps: Number of integration steps.
        dt: Timestep.

    Returns:
        Dict with 'time', 'states', and individual variable arrays.
    """
    config = SimulationConfig(
        domain=Domain.FHN_COUPLED_PAIR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "g": g, "I1": I1, "I2": I2,
            "v1_0": -1.0, "w1_0": -0.5,
            "v2_0": 0.5, "w2_0": 0.0,
        },
    )
    sim = FHNCoupledPairSimulation(config)
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
        "v1": states_arr[:, 0],
        "w1": states_arr[:, 1],
        "v2": states_arr[:, 2],
        "w2": states_arr[:, 3],
        "g": g,
        "I1": I1,
        "I2": I2,
        "dt": dt,
    }


def generate_coupling_sweep_data(
    n_g: int = 25,
    g_min: float = 0.0,
    g_max: float = 1.0,
    I1: float = 0.5,
    I2: float = 0.5,
    n_steps: int = 8000,
    n_transient: int = 5000,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Sweep coupling strength and measure steady-state sync metrics.

    For each g value, runs the coupled system from different initial
    conditions, discards transient, and measures the time-averaged
    synchronization error and phase difference.

    Args:
        n_g: Number of coupling values to test.
        g_min: Minimum coupling strength.
        g_max: Maximum coupling strength.
        I1: External current to neuron 1.
        I2: External current to neuron 2.
        n_steps: Measurement steps after transient.
        n_transient: Transient steps to discard.
        dt: Timestep.

    Returns:
        Dict with 'g', 'mean_error', and 'mean_phase_diff' arrays.
    """
    g_values = np.linspace(g_min, g_max, n_g)
    mean_errors = np.empty(n_g)
    mean_phases = np.empty(n_g)

    for i, g_val in enumerate(g_values):
        config = SimulationConfig(
            domain=Domain.FHN_COUPLED_PAIR,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "g": float(g_val),
                "I1": I1, "I2": I2,
                "v1_0": -1.0, "w1_0": -0.5,
                "v2_0": 0.5, "w2_0": 0.0,
            },
        )
        sim = FHNCoupledPairSimulation(config)
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
                f"  g={g_val:.3f}: mean_error={mean_errors[i]:.4f}, "
                f"phase_diff={mean_phases[i]:.4f}"
            )

    return {
        "g": g_values,
        "mean_error": mean_errors,
        "mean_phase_diff": mean_phases,
    }


def generate_heterogeneity_data(
    g: float = 0.2,
    I1: float = 0.5,
    n_I2: int = 20,
    I2_min: float = 0.3,
    I2_max: float = 0.7,
    n_steps: int = 8000,
    n_transient: int = 5000,
    dt: float = 0.05,
) -> dict[str, np.ndarray]:
    """Sweep I2 while keeping I1 fixed to measure sync robustness.

    Tests how heterogeneous input currents affect synchronization
    at a given coupling strength.

    Args:
        g: Coupling strength.
        I1: Fixed external current to neuron 1.
        n_I2: Number of I2 values to test.
        I2_min: Minimum I2 value.
        I2_max: Maximum I2 value.
        n_steps: Measurement steps after transient.
        n_transient: Transient steps to discard.
        dt: Timestep.

    Returns:
        Dict with 'I2', 'mean_error', 'delta_I', and 'mean_phase_diff'.
    """
    I2_values = np.linspace(I2_min, I2_max, n_I2)
    mean_errors = np.empty(n_I2)
    mean_phases = np.empty(n_I2)

    for i, I2_val in enumerate(I2_values):
        config = SimulationConfig(
            domain=Domain.FHN_COUPLED_PAIR,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "g": g, "I1": I1, "I2": float(I2_val),
                "v1_0": -1.0, "w1_0": -0.5,
                "v2_0": 0.5, "w2_0": 0.0,
            },
        )
        sim = FHNCoupledPairSimulation(config)
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

    return {
        "I2": I2_values,
        "delta_I": I2_values - I1,
        "mean_error": mean_errors,
        "mean_phase_diff": mean_phases,
    }


def _detect_critical_coupling(
    g_values: np.ndarray,
    mean_errors: np.ndarray,
    threshold_fraction: float = 0.1,
) -> float:
    """Estimate critical coupling g_c where sync error drops below threshold.

    Uses the error at g=0 as reference and finds the first g where
    the error drops below threshold_fraction of the uncoupled error.

    Args:
        g_values: Array of coupling strength values.
        mean_errors: Corresponding mean sync errors.
        threshold_fraction: Fraction of g=0 error to use as threshold.

    Returns:
        Estimated critical coupling g_c, or NaN if not detected.
    """
    if len(g_values) < 2:
        return float("nan")

    error_at_zero = mean_errors[0]
    if error_at_zero < 1e-10:
        return 0.0

    threshold = threshold_fraction * error_at_zero
    for i in range(len(g_values)):
        if mean_errors[i] < threshold:
            if i == 0:
                return float(g_values[0])
            # Linear interpolation between i-1 and i
            g_prev = g_values[i - 1]
            g_curr = g_values[i]
            e_prev = mean_errors[i - 1]
            e_curr = mean_errors[i]
            frac = (threshold - e_prev) / (e_curr - e_prev + 1e-30)
            return float(g_prev + frac * (g_curr - g_prev))

    return float("nan")


def run_fhn_coupled_pair_rediscovery(
    output_dir: str | Path = "output/rediscovery/fhn_coupled_pair",
    n_iterations: int = 40,
) -> dict:
    """Run the full FHN coupled pair rediscovery pipeline.

    1. Generate trajectory for SINDy ODE recovery
    2. Sweep coupling strength g to find synchronization transition
    3. Detect critical coupling g_c
    4. Measure phase-locking ratio
    5. Test heterogeneity robustness (vary I2)

    Args:
        output_dir: Directory for saving results and data files.
        n_iterations: Number of PySR iterations (unused, kept for API compat).

    Returns:
        Dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "fhn_coupled_pair",
        "targets": {
            "ode": (
                "dv1/dt=v1-v1^3/3-w1+I1+g*(v2-v1), dw1/dt=eps*(v1+a-b*w1), "
                "dv2/dt=v2-v2^3/3-w2+I2+g*(v1-v2), dw2/dt=eps*(v2+a-b*w2)"
            ),
            "sync_transition": "Sync error decreases with increasing g",
            "critical_coupling": "g_c for synchronization onset",
            "heterogeneity": "Sync robustness under I1 != I2",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for coupled FHN (g=0.1)...")
    ode_data = generate_ode_data(g=0.1, n_steps=10000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            ode_data["states"],
            dt=ode_data["dt"],
            feature_names=["v1", "w1", "v2", "w2"],
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
    logger.info("Part 2: Sweeping coupling strength g...")
    sweep_data = generate_coupling_sweep_data(
        n_g=25, g_min=0.0, g_max=1.0,
        n_steps=8000, n_transient=5000, dt=0.05,
    )

    results["coupling_sweep"] = {
        "n_g_values": len(sweep_data["g"]),
        "g_range": [float(sweep_data["g"][0]), float(sweep_data["g"][-1])],
        "min_error": float(np.min(sweep_data["mean_error"])),
        "max_error": float(np.max(sweep_data["mean_error"])),
        "error_at_g0": float(sweep_data["mean_error"][0]),
        "error_at_gmax": float(sweep_data["mean_error"][-1]),
    }

    # Check that error generally decreases with coupling
    low_g_error = np.mean(sweep_data["mean_error"][:5])
    high_g_error = np.mean(sweep_data["mean_error"][-5:])
    results["coupling_sweep"]["low_g_mean_error"] = float(low_g_error)
    results["coupling_sweep"]["high_g_mean_error"] = float(high_g_error)
    results["coupling_sweep"]["error_decreases"] = bool(
        high_g_error < low_g_error
    )

    # --- Part 3: Critical coupling detection ---
    logger.info("Part 3: Detecting critical coupling g_c...")
    g_c = _detect_critical_coupling(
        sweep_data["g"], sweep_data["mean_error"],
    )
    results["critical_coupling"] = {
        "g_c": float(g_c) if np.isfinite(g_c) else None,
        "method": "10% of uncoupled error threshold",
    }
    logger.info(f"  Estimated g_c = {g_c:.4f}")

    # --- Part 4: Phase-locking ratio ---
    logger.info("Part 4: Measuring phase-locking ratio...")
    n_locked = 0
    total = len(sweep_data["g"])
    for i in range(total):
        dphi = sweep_data["mean_phase_diff"][i]
        # Phase-locked if mean phase diff is near 0, pi, or 2pi
        is_locked = (
            dphi < 0.3
            or dphi > (2 * np.pi - 0.3)
            or abs(dphi - np.pi) < 0.3
        )
        if is_locked:
            n_locked += 1

    results["phase_locking"] = {
        "n_locked": n_locked,
        "total": total,
        "locking_ratio": float(n_locked / total) if total > 0 else 0.0,
    }
    logger.info(f"  Phase-locked: {n_locked}/{total}")

    # --- Part 5: Heterogeneity robustness ---
    logger.info("Part 5: Testing heterogeneity robustness (varying I2)...")
    hetero_data = generate_heterogeneity_data(
        g=0.2, I1=0.5,
        n_I2=20, I2_min=0.3, I2_max=0.7,
        n_steps=6000, n_transient=4000, dt=0.05,
    )

    # Sync error should increase with |I1 - I2|
    delta_abs = np.abs(hetero_data["delta_I"])
    center_idx = np.argmin(delta_abs)
    edge_errors = np.mean([
        hetero_data["mean_error"][0],
        hetero_data["mean_error"][-1],
    ])
    center_error = hetero_data["mean_error"][center_idx]

    results["heterogeneity"] = {
        "g": 0.2,
        "I1": 0.5,
        "I2_range": [float(hetero_data["I2"][0]), float(hetero_data["I2"][-1])],
        "center_error": float(center_error),
        "edge_error": float(edge_errors),
        "error_increases_with_mismatch": bool(edge_errors > center_error),
    }
    logger.info(
        f"  Center error (I1~I2): {center_error:.4f}, "
        f"Edge error (max |dI|): {edge_errors:.4f}"
    )

    # --- Save results ---
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "coupling_sweep.npz",
        g=sweep_data["g"],
        mean_error=sweep_data["mean_error"],
        mean_phase_diff=sweep_data["mean_phase_diff"],
    )
    np.savez(
        output_path / "heterogeneity.npz",
        I2=hetero_data["I2"],
        delta_I=hetero_data["delta_I"],
        mean_error=hetero_data["mean_error"],
        mean_phase_diff=hetero_data["mean_phase_diff"],
    )

    return results
