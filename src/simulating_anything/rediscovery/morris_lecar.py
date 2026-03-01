"""Morris-Lecar rediscovery.

Targets:
- ODE: C*dV/dt = I_ext - g_L*(V-V_L) - g_Ca*m_ss(V)*(V-V_Ca) - g_K*w*(V-V_K)
        dw/dt = phi*(w_ss(V)-w)/tau_w(V)  (via SINDy)
- f-I curve: firing frequency as a function of input current I_ext
- Excitability type: Type I (saddle-node) vs Type II (Hopf) classification
- Nullcline structure and equilibrium point
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.morris_lecar import MorrisLecarSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    I_ext: float = 100.0,
    n_steps: int = 10000,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses I_ext=100 to ensure oscillatory regime for clean ODE fitting.
    """
    config = SimulationConfig(
        domain=Domain.MORRIS_LECAR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "C": 20.0, "g_L": 2.0, "g_Ca": 4.0, "g_K": 8.0,
            "V_L": -60.0, "V_Ca": 120.0, "V_K": -84.0,
            "V1": -1.2, "V2": 18.0, "V3": 2.0, "V4": 30.0,
            "phi": 0.04, "I_ext": I_ext,
        },
    )
    sim = MorrisLecarSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "V": states[:, 0],
        "w": states[:, 1],
        "I_ext": I_ext,
    }


def generate_fi_curve(
    n_I: int = 30,
    I_min: float = 30.0,
    I_max: float = 200.0,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate firing frequency vs input current (f-I curve).

    The Morris-Lecar model requires larger currents than FHN (mV scale).
    """
    I_values = np.linspace(I_min, I_max, n_I)
    frequencies = []

    for i, I_ext in enumerate(I_values):
        config = SimulationConfig(
            domain=Domain.MORRIS_LECAR,
            dt=dt,
            n_steps=1000,
            parameters={
                "C": 20.0, "g_L": 2.0, "g_Ca": 4.0, "g_K": 8.0,
                "V_L": -60.0, "V_Ca": 120.0, "V_K": -84.0,
                "V1": -1.2, "V2": 18.0, "V3": 2.0, "V4": 30.0,
                "phi": 0.04, "I_ext": I_ext,
            },
        )
        sim = MorrisLecarSimulation(config)
        sim.reset()
        freq = sim.measure_firing_frequency(n_spikes=5)
        frequencies.append(freq)

        if (i + 1) % 10 == 0:
            logger.info(f"  I_ext={I_ext:.1f}: freq={freq:.2f} Hz")

    return {
        "I": I_values,
        "frequency": np.array(frequencies),
    }


def classify_excitability(
    dt: float = 0.1,
) -> dict[str, object]:
    """Classify excitability type by examining f-I curve near threshold.

    Type I (saddle-node): frequency rises continuously from zero at onset.
    Type II (Hopf): frequency jumps to a finite value at onset.
    """
    # Sweep with fine resolution near onset
    I_values = np.linspace(30.0, 120.0, 50)

    config_template = {
        "C": 20.0, "g_L": 2.0, "g_Ca": 4.0, "g_K": 8.0,
        "V_L": -60.0, "V_Ca": 120.0, "V_K": -84.0,
        "V1": -1.2, "V2": 18.0, "V3": 2.0, "V4": 30.0,
        "phi": 0.04,
    }

    frequencies = []
    for I_ext in I_values:
        params = {**config_template, "I_ext": float(I_ext)}
        config = SimulationConfig(
            domain=Domain.MORRIS_LECAR,
            dt=dt,
            n_steps=1000,
            parameters=params,
        )
        sim = MorrisLecarSimulation(config)
        sim.reset()
        freq = sim.measure_firing_frequency(n_spikes=5)
        frequencies.append(freq)

    frequencies = np.array(frequencies)
    firing = frequencies > 1.0  # Hz threshold

    if not np.any(firing):
        return {
            "type": "quiescent",
            "I_threshold": None,
            "onset_frequency": None,
        }

    # Find onset
    onset_idx = int(np.argmax(firing))
    I_threshold = float(I_values[onset_idx])
    onset_freq = float(frequencies[onset_idx])

    # Type I: onset freq near 0; Type II: onset freq is finite jump
    # Heuristic: if onset freq > 20 Hz, likely Type II (Hopf)
    exc_type = "Type II" if onset_freq > 20.0 else "Type I"

    return {
        "type": exc_type,
        "I_threshold": I_threshold,
        "onset_frequency": onset_freq,
        "I_values": I_values.tolist(),
        "frequencies": frequencies.tolist(),
    }


def run_morris_lecar_rediscovery(
    output_dir: str | Path = "output/rediscovery/morris_lecar",
    n_iterations: int = 40,
) -> dict:
    """Run Morris-Lecar rediscovery pipeline.

    Args:
        output_dir: Directory to save results.
        n_iterations: PySR iteration count.

    Returns:
        Results dict with SINDy ODE, f-I curve, and excitability classification.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "morris_lecar",
        "targets": {
            "ode": "C*dV/dt = I_ext - g_L*(V-V_L) - g_Ca*m_ss(V)*(V-V_Ca) - g_K*w*(V-V_K)",
            "fi_curve": "firing frequency vs input current",
            "excitability": "Type I (saddle-node) vs Type II (Hopf)",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at I_ext=100...")
    data = generate_ode_data(I_ext=100.0, n_steps=20000, dt=0.1)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.1,
            feature_names=["V", "w"],
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

    # --- Part 2: f-I curve ---
    logger.info("Part 2: f-I curve...")
    fi_data = generate_fi_curve(n_I=30, dt=0.1)

    # Find critical current (onset of firing)
    firing = fi_data["frequency"] > 1.0
    if np.any(firing):
        I_c_idx = int(np.argmax(firing))
        I_c = float(fi_data["I"][I_c_idx])
        results["fi_curve"] = {
            "I_critical": I_c,
            "max_frequency": float(np.max(fi_data["frequency"])),
            "n_oscillatory": int(np.sum(firing)),
        }
        logger.info(f"  Critical current: I_c ~ {I_c:.1f} uA/cm^2")
        logger.info(f"  Max frequency: {np.max(fi_data['frequency']):.2f} Hz")
    else:
        results["fi_curve"] = {"note": "No oscillations detected in range"}

    # PySR: find f(I) for I > I_c
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        if np.any(firing):
            X = fi_data["I"][firing].reshape(-1, 1)
            y = fi_data["frequency"][firing]

            if len(X) > 3:
                logger.info("  Running PySR: freq = f(I_ext)...")
                discoveries = run_symbolic_regression(
                    X, y,
                    variable_names=["I_ext"],
                    n_iterations=n_iterations,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sqrt", "square"],
                    max_complexity=10,
                    populations=15,
                    population_size=30,
                )
                results["fi_pysr"] = {
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
                    results["fi_pysr"]["best"] = best.expression
                    results["fi_pysr"]["best_r2"] = best.evidence.fit_r_squared
                    logger.info(
                        f"  Best: {best.expression} "
                        f"(R2={best.evidence.fit_r_squared:.6f})"
                    )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["fi_pysr"] = {"error": str(e)}

    # --- Part 3: Excitability classification ---
    logger.info("Part 3: Excitability type classification...")
    exc_result = classify_excitability(dt=0.1)
    results["excitability"] = {
        "type": exc_result["type"],
        "I_threshold": exc_result["I_threshold"],
        "onset_frequency": exc_result["onset_frequency"],
    }
    logger.info(f"  Excitability: {exc_result['type']}")
    logger.info(f"  Threshold: I_ext ~ {exc_result['I_threshold']}")
    logger.info(f"  Onset freq: {exc_result['onset_frequency']}")

    # --- Part 4: Nullclines and equilibrium ---
    logger.info("Part 4: Nullclines and equilibrium...")
    config = SimulationConfig(
        domain=Domain.MORRIS_LECAR,
        dt=0.1,
        n_steps=1000,
        parameters={"I_ext": 0.0},
    )
    sim = MorrisLecarSimulation(config)
    V_eq, w_eq = sim.find_equilibrium()
    results["equilibrium"] = {
        "V_eq": V_eq,
        "w_eq": w_eq,
    }
    logger.info(f"  Equilibrium at I=0: V={V_eq:.2f} mV, w={w_eq:.4f}")

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "fi_curve.npz",
        I=fi_data["I"],
        frequency=fi_data["frequency"],
    )

    return results
