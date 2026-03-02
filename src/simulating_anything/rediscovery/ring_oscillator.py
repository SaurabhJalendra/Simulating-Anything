"""Ring oscillator rediscovery.

Targets:
- Oscillation onset at gain g > 1 (loop gain instability)
- Period T ~ 6*tau (tau = 1 RC time constant)
- 3-phase symmetry: 120-degree phase shifts between stages
- Amplitude saturation at tanh(g) for large g
- ODE recovery via SINDy
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.ring_oscillator import RingOscillatorSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    gain: float = 5.0,
    n_steps: int = 5000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery."""
    config = SimulationConfig(
        domain=Domain.RING_OSCILLATOR,  # placeholder until RING_OSCILLATOR added to enum
        dt=dt,
        n_steps=n_steps,
        parameters={
            "gain": gain,
            "x1_0": 0.01,
            "x2_0": 0.0,
            "x3_0": 0.0,
        },
    )
    sim = RingOscillatorSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    times = np.arange(n_steps + 1) * dt

    return {
        "time": times,
        "states": states,
        "x1": states[:, 0],
        "x2": states[:, 1],
        "x3": states[:, 2],
        "gain": gain,
        "dt": dt,
    }


def generate_gain_sweep_data(
    n_gains: int = 30,
    dt: float = 0.01,
    n_steps: int = 5000,
) -> dict[str, np.ndarray]:
    """Sweep gain and measure amplitude and period for each value."""
    config = SimulationConfig(
        domain=Domain.RING_OSCILLATOR,
        dt=dt,
        n_steps=n_steps,
        parameters={"gain": 5.0, "x1_0": 0.01, "x2_0": 0.0, "x3_0": 0.0},
    )
    sim = RingOscillatorSimulation(config)
    sim.reset()

    result = sim.gain_sweep(
        gain_range=(0.5, 10.0), n_gains=n_gains, n_steps=n_steps
    )
    return result


def generate_phase_symmetry_data(
    gain: float = 5.0,
    dt: float = 0.005,
) -> dict[str, float]:
    """Verify 3-phase symmetry by measuring phase shifts."""
    config = SimulationConfig(
        domain=Domain.RING_OSCILLATOR,
        dt=dt,
        n_steps=10000,
        parameters={"gain": gain, "x1_0": 0.01, "x2_0": 0.0, "x3_0": 0.0},
    )
    sim = RingOscillatorSimulation(config)
    sim.reset()

    shift_12, shift_23 = sim.compute_phase_shifts(
        n_transient=3000, n_measure=7000
    )

    return {
        "shift_12_degrees": shift_12,
        "shift_23_degrees": shift_23,
        "ideal_shift": 120.0,
        "error_12": abs(shift_12 - 120.0),
        "error_23": abs(shift_23 - 120.0),
    }


def run_ring_oscillator_rediscovery(
    output_dir: str | Path = "output/rediscovery/ring_oscillator",
    n_iterations: int = 40,
) -> dict:
    """Run ring oscillator rediscovery pipeline.

    1. Gain sweep: amplitude and period vs gain
    2. Oscillation onset detection (gain_c ~ 1)
    3. 3-phase symmetry verification
    4. SINDy ODE recovery

    Args:
        output_dir: Directory for output files.
        n_iterations: PySR iterations (if used).

    Returns:
        Results dict with all rediscovery outcomes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "ring_oscillator",
        "targets": {
            "ode": "dx_i/dt = -x_i + tanh(g*(-x_{i-1}))",
            "oscillation_onset": "g_c ~ 1 (sufficient loop gain)",
            "period": "T ~ 6*tau (tau=1)",
            "phase_symmetry": "120-degree phase shifts (Z3 symmetry)",
            "amplitude": "saturates at tanh(g) for large g",
        },
    }

    # --- Part 1: Gain sweep ---
    logger.info("Part 1: Gain sweep (amplitude and period vs gain)...")
    sweep_data = generate_gain_sweep_data(n_gains=30, dt=0.01, n_steps=6000)

    gains = sweep_data["gain"]
    amps = sweep_data["amplitude"]
    periods = sweep_data["period"]

    # Detect oscillation onset: first gain where amplitude > threshold
    amp_threshold = 0.05
    oscillating = amps > amp_threshold
    if np.any(oscillating):
        idx = np.argmax(oscillating)
        gain_c_est = float(gains[max(0, idx - 1)])
    else:
        gain_c_est = float("nan")

    results["gain_sweep"] = {
        "n_gains": len(gains),
        "gain_range": [float(gains.min()), float(gains.max())],
        "gain_c_estimate": gain_c_est,
        "gain_c_theory": 1.0,
        "gain_c_error": abs(gain_c_est - 1.0) if np.isfinite(gain_c_est) else None,
    }

    # Period analysis for oscillating regime
    osc_mask = (amps > 0.1) & np.isfinite(periods) & (periods < 100)
    if np.any(osc_mask):
        mean_period = float(np.mean(periods[osc_mask]))
        results["period_data"] = {
            "n_oscillating": int(np.sum(osc_mask)),
            "mean_period": mean_period,
            "theoretical_period": 6.0,
            "relative_error": abs(mean_period - 6.0) / 6.0,
        }
        logger.info(
            f"  Mean period = {mean_period:.3f} "
            f"(theory ~ 6.0, error = {abs(mean_period - 6.0) / 6.0:.1%})"
        )

    # Amplitude saturation for large gain
    large_g_mask = gains > 5.0
    if np.any(large_g_mask):
        mean_amp_large = float(np.mean(amps[large_g_mask]))
        results["amplitude_saturation"] = {
            "mean_amplitude_large_g": mean_amp_large,
            "theoretical_saturation": 1.0,
            "note": "For large g, amplitude -> tanh(g) -> 1",
        }
        logger.info(f"  Large-gain amplitude: {mean_amp_large:.4f} (theory ~ 1.0)")

    logger.info(f"  Oscillation onset: g_c ~ {gain_c_est:.3f} (theory: 1.0)")

    # --- Part 2: 3-phase symmetry ---
    logger.info("Part 2: Verifying 3-phase symmetry...")
    phase_data = generate_phase_symmetry_data(gain=5.0, dt=0.005)

    results["phase_symmetry"] = {
        "shift_12_degrees": phase_data["shift_12_degrees"],
        "shift_23_degrees": phase_data["shift_23_degrees"],
        "ideal_shift": 120.0,
        "error_12_degrees": phase_data["error_12"],
        "error_23_degrees": phase_data["error_23"],
    }
    logger.info(
        f"  Phase shifts: 1->2 = {phase_data['shift_12_degrees']:.1f} deg, "
        f"2->3 = {phase_data['shift_23_degrees']:.1f} deg (ideal: 120 deg)"
    )

    # --- Part 3: Amplitude symmetry ---
    logger.info("Part 3: Amplitude symmetry across stages...")
    config = SimulationConfig(
        domain=Domain.RING_OSCILLATOR,
        dt=0.01,
        n_steps=8000,
        parameters={"gain": 5.0, "x1_0": 0.01, "x2_0": 0.0, "x3_0": 0.0},
    )
    sim = RingOscillatorSimulation(config)
    sim.reset()
    a1, a2, a3 = sim.measure_amplitudes_all(n_transient=3000, n_measure=5000)
    results["amplitude_symmetry"] = {
        "amp1": a1,
        "amp2": a2,
        "amp3": a3,
        "max_spread": max(a1, a2, a3) - min(a1, a2, a3),
    }
    logger.info(f"  Amplitudes: [{a1:.4f}, {a2:.4f}, {a3:.4f}]")

    # --- Part 4: SINDy ODE recovery ---
    logger.info("Part 4: SINDy ODE recovery at gain=5.0...")
    data = generate_ode_data(gain=5.0, n_steps=10000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=data["dt"],
            feature_names=["x1", "x2", "x3"],
            threshold=0.05,
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

    # --- Save results ---
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "gain_sweep.npz",
        gain=sweep_data["gain"],
        amplitude=sweep_data["amplitude"],
        period=sweep_data["period"],
    )

    return results
