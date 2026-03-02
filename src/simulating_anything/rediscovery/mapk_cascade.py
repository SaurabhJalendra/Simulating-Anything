"""MAPK cascade rediscovery: ultrasensitivity, amplification, and ODE recovery.

Targets:
- Ultrasensitive dose-response curve (Hill coefficient n_H >> 1 for small K)
- Signal amplification ratio X3/X1
- Steady-state response X3_ss(V1) via PySR
- ODE recovery via SINDy
- Effect of Michaelis constant K on switch-like behavior
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.mapk_cascade import MAPKCascadeSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(
    V1: float = 1.0,
    V2: float = 0.3,
    k3: float = 1.0,
    V4: float = 0.3,
    k5: float = 1.0,
    V6: float = 0.3,
    K1: float = 0.01,
    K2: float = 0.01,
    K3: float = 0.01,
    K4: float = 0.01,
    K5: float = 0.01,
    K6: float = 0.01,
    dt: float = 0.01,
    n_steps: int = 2000,
) -> SimulationConfig:
    """Create a SimulationConfig for the MAPK cascade."""
    return SimulationConfig(
        domain=Domain.MAPK_CASCADE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "V1": V1, "V2": V2, "k3": k3, "V4": V4, "k5": k5, "V6": V6,
            "K1": K1, "K2": K2, "K3": K3, "K4": K4, "K5": K5, "K6": K6,
        },
    )


def generate_ode_data(
    V1: float = 1.0,
    K: float = 0.01,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Args:
        V1: Input signal strength.
        K: Michaelis constant (same for all tiers).
        n_steps: Number of integration steps.
        dt: Timestep.

    Returns:
        Dictionary with time, states, X1, X2, X3 arrays and parameters.
    """
    config = _make_config(
        V1=V1, K1=K, K2=K, K3=K, K4=K, K5=K, K6=K, dt=dt, n_steps=n_steps,
    )
    sim = MAPKCascadeSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states_arr = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states_arr,
        "X1": states_arr[:, 0],
        "X2": states_arr[:, 1],
        "X3": states_arr[:, 2],
        "V1": V1,
        "K": K,
        "dt": dt,
    }


def generate_dose_response_data(
    n_V1: int = 40,
    V1_range: tuple[float, float] = (0.0, 2.0),
    K: float = 0.01,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep V1 and measure steady-state X3 (dose-response curve).

    Args:
        n_V1: Number of V1 values in sweep.
        V1_range: (min, max) for signal strength.
        K: Michaelis constant (same for all tiers).
        n_steps: Steps to reach steady state per V1.
        dt: Timestep.

    Returns:
        Dictionary with V1, X1_ss, X2_ss, X3_ss, amplification arrays.
    """
    V1_values = np.linspace(V1_range[0], V1_range[1], n_V1)
    X1_ss = np.zeros(n_V1)
    X2_ss = np.zeros(n_V1)
    X3_ss = np.zeros(n_V1)

    for i, V1 in enumerate(V1_values):
        config = _make_config(
            V1=V1, K1=K, K2=K, K3=K, K4=K, K5=K, K6=K, dt=dt, n_steps=n_steps,
        )
        sim = MAPKCascadeSimulation(config)
        sim.reset()

        for _ in range(n_steps):
            sim.step()

        state = sim.observe()
        X1_ss[i] = state[0]
        X2_ss[i] = state[1]
        X3_ss[i] = state[2]

    # Amplification where X1 is nonzero
    amplification = np.zeros(n_V1)
    mask = X1_ss > 0.01
    amplification[mask] = X3_ss[mask] / X1_ss[mask]

    return {
        "V1": V1_values,
        "X1_ss": X1_ss,
        "X2_ss": X2_ss,
        "X3_ss": X3_ss,
        "amplification": amplification,
        "K": K,
    }


def generate_K_sweep_data(
    n_K: int = 10,
    K_range: tuple[float, float] = (0.001, 0.5),
    V1: float = 1.0,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep Michaelis constant K and measure Hill coefficient.

    Smaller K should produce steeper (more switch-like) responses.

    Args:
        n_K: Number of K values.
        K_range: (min, max) for Michaelis constant sweep.
        V1: Fixed input signal (used during dose-response).
        n_steps: Steps to reach steady state.
        dt: Timestep.

    Returns:
        Dictionary with K_values and hill_coefficients arrays.
    """
    K_values = np.logspace(
        np.log10(K_range[0]), np.log10(K_range[1]), n_K,
    )
    hill_coefficients = np.zeros(n_K)

    for i, K in enumerate(K_values):
        config = _make_config(
            V1=V1, K1=K, K2=K, K3=K, K4=K, K5=K, K6=K,
            dt=dt, n_steps=n_steps,
        )
        sim = MAPKCascadeSimulation(config)
        result = sim.dose_response(
            V1_range=(0.0, 2.0), n_points=40, n_steps=n_steps,
        )
        hill_coefficients[i] = result["hill_coefficient"]

    return {
        "K_values": K_values,
        "hill_coefficients": hill_coefficients,
    }


def run_mapk_cascade_rediscovery(
    output_dir: str | Path = "output/rediscovery/mapk_cascade",
    n_iterations: int = 40,
) -> dict:
    """Run MAPK cascade rediscovery pipeline.

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iteration count.

    Returns:
        Results dictionary with all rediscovery findings.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "mapk_cascade",
        "targets": {
            "ultrasensitivity": "Hill coefficient n_H >> 1 for small K",
            "amplification": "X3/X1 ratio through cascade",
            "dose_response": "Steady-state X3 vs V1",
            "ode": "Three-tier Goldbeter-Koshland cascade",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery for MAPK cascade...")
    data = generate_ode_data(V1=1.0, K=0.01, n_steps=5000, dt=0.01)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.01,
            feature_names=["X1", "X2", "X3"],
            threshold=0.01,
            poly_degree=2,
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

    # --- Part 2: Dose-response curve ---
    logger.info("Part 2: Dose-response sweep (V1 vs X3_ss)...")
    dr_data = generate_dose_response_data(n_V1=40, K=0.01, n_steps=10000)

    results["dose_response"] = {
        "n_points": len(dr_data["V1"]),
        "X3_max": float(np.max(dr_data["X3_ss"])),
        "X3_min": float(np.min(dr_data["X3_ss"])),
        "mean_amplification": float(
            np.mean(dr_data["amplification"][dr_data["amplification"] > 0])
        ) if np.any(dr_data["amplification"] > 0) else 0.0,
    }

    # Estimate Hill coefficient from dose-response
    config_hill = _make_config(V1=1.0, dt=0.01)
    sim_hill = MAPKCascadeSimulation(config_hill)
    hill_result = sim_hill.dose_response(
        V1_range=(0.0, 2.0), n_points=60, n_steps=10000,
    )
    results["dose_response"]["hill_coefficient"] = float(
        hill_result["hill_coefficient"]
    )
    logger.info(
        f"  Hill coefficient: {hill_result['hill_coefficient']:.2f}"
    )

    # PySR: fit X3_ss = f(V1)
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use data where V1 > 0 and X3 varies
        mask = dr_data["V1"] > 0.01
        X = dr_data["V1"][mask].reshape(-1, 1)
        y = dr_data["X3_ss"][mask]

        if len(y) > 5 and np.std(y) > 0.01:
            logger.info("  Running PySR: X3_ss = f(V1)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["V1_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square", "sqrt"],
                max_complexity=12,
                populations=20,
                population_size=40,
            )
            results["pysr_dose_response"] = {
                "n_discoveries": len(discoveries),
                "discoveries": [
                    {
                        "expression": d.expression,
                        "r_squared": d.evidence.fit_r_squared,
                    }
                    for d in discoveries[:5]
                ],
            }
            if discoveries:
                best = discoveries[0]
                results["pysr_dose_response"]["best"] = best.expression
                results["pysr_dose_response"]["best_r2"] = (
                    best.evidence.fit_r_squared
                )
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR dose-response failed: {e}")
        results["pysr_dose_response"] = {"error": str(e)}

    # --- Part 3: Michaelis constant K sweep ---
    logger.info("Part 3: K sweep for ultrasensitivity...")
    try:
        K_data = generate_K_sweep_data(n_K=8, K_range=(0.001, 0.5))

        results["K_sweep"] = {
            "n_points": len(K_data["K_values"]),
            "K_values": [float(k) for k in K_data["K_values"]],
            "hill_coefficients": [float(h) for h in K_data["hill_coefficients"]],
            "max_hill": float(np.max(K_data["hill_coefficients"])),
            "min_hill": float(np.min(K_data["hill_coefficients"])),
        }

        # Verify: small K should give higher Hill coefficient
        small_K_mask = K_data["K_values"] < 0.01
        large_K_mask = K_data["K_values"] > 0.1
        if np.any(small_K_mask) and np.any(large_K_mask):
            mean_small = float(np.mean(K_data["hill_coefficients"][small_K_mask]))
            mean_large = float(np.mean(K_data["hill_coefficients"][large_K_mask]))
            results["K_sweep"]["small_K_mean_hill"] = mean_small
            results["K_sweep"]["large_K_mean_hill"] = mean_large
            results["K_sweep"]["small_K_steeper"] = bool(mean_small > mean_large)

        logger.info(
            f"  K sweep: max Hill={np.max(K_data['hill_coefficients']):.2f}, "
            f"min Hill={np.min(K_data['hill_coefficients']):.2f}"
        )
    except Exception as e:
        logger.warning(f"K sweep failed: {e}")
        results["K_sweep"] = {"error": str(e)}

    # --- Part 4: Signal amplification ---
    logger.info("Part 4: Signal amplification analysis...")
    config_amp = _make_config(V1=0.5, dt=0.01, n_steps=10000)
    sim_amp = MAPKCascadeSimulation(config_amp)
    sim_amp.reset()
    for _ in range(10000):
        sim_amp.step()
    amp_ratio = sim_amp.signal_amplification()
    state_ss = sim_amp.observe()

    results["amplification"] = {
        "V1": 0.5,
        "X1_ss": float(state_ss[0]),
        "X2_ss": float(state_ss[1]),
        "X3_ss": float(state_ss[2]),
        "amplification_ratio": amp_ratio,
    }
    logger.info(
        f"  Amplification at V1=0.5: X3/X1 = {amp_ratio:.4f}"
    )

    # --- Part 5: Cascade temporal ordering ---
    logger.info("Part 5: Cascade temporal ordering...")
    config_time = _make_config(V1=1.5, dt=0.01, n_steps=2000)
    sim_time = MAPKCascadeSimulation(config_time)
    sim_time.reset()

    # Find half-activation times for each tier
    half_times = [float("inf"), float("inf"), float("inf")]
    for step_i in range(1, 2001):
        sim_time.step()
        state = sim_time.observe()
        t = step_i * 0.01
        for j in range(3):
            if half_times[j] == float("inf") and state[j] >= 0.5:
                half_times[j] = t

    results["temporal_ordering"] = {
        "half_activation_times": half_times,
        "X1_first": bool(half_times[0] <= half_times[1]),
        "X2_before_X3": bool(half_times[1] <= half_times[2]),
        "cascade_ordered": bool(
            half_times[0] <= half_times[1] <= half_times[2]
        ),
    }
    logger.info(
        f"  Half-activation times: X1={half_times[0]:.3f}, "
        f"X2={half_times[1]:.3f}, X3={half_times[2]:.3f}"
    )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return results
