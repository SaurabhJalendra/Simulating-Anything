"""Wilson-Cowan neural population model rediscovery.

Targets:
- ODE: tau_E*dE/dt = -E + S(w_EE*E - w_EI*I + I_ext_E)  (via SINDy)
- Hopf bifurcation: oscillation onset as function of I_ext_E
- E-I oscillation frequency and amplitude
- Nullcline structure and fixed point analysis
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.wilson_cowan import WilsonCowanSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_ode_data(
    I_ext_E: float = 1.5,
    n_steps: int = 10000,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Generate trajectory data for SINDy ODE recovery.

    Uses default Wilson-Cowan parameters with specified external input.
    """
    config = SimulationConfig(
        domain=Domain.WILSON_COWAN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "tau_E": 1.0, "tau_I": 2.0,
            "w_EE": 16.0, "w_EI": 12.0,
            "w_IE": 15.0, "w_II": 3.0,
            "a": 1.3, "theta": 4.0,
            "I_ext_E": I_ext_E, "I_ext_I": 0.0,
            "E_0": 0.1, "I_0": 0.05,
        },
    )
    sim = WilsonCowanSimulation(config)
    sim.reset()

    states = [sim.observe().copy()]
    for _ in range(n_steps):
        sim.step()
        states.append(sim.observe().copy())

    states = np.array(states)
    return {
        "time": np.arange(n_steps + 1) * dt,
        "states": states,
        "E": states[:, 0],
        "I": states[:, 1],
        "I_ext_E": I_ext_E,
    }


def generate_hopf_data(
    n_I_ext: int = 25,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Sweep I_ext_E and measure oscillation amplitude to find Hopf bifurcation."""
    I_ext_values = np.linspace(0.0, 3.0, n_I_ext)
    amplitudes = []

    for i, I_ext in enumerate(I_ext_values):
        config = SimulationConfig(
            domain=Domain.WILSON_COWAN,
            dt=dt,
            n_steps=1000,
            parameters={
                "tau_E": 1.0, "tau_I": 2.0,
                "w_EE": 16.0, "w_EI": 12.0,
                "w_IE": 15.0, "w_II": 3.0,
                "a": 1.3, "theta": 4.0,
                "I_ext_E": I_ext, "I_ext_I": 0.0,
                "E_0": 0.1, "I_0": 0.05,
            },
        )
        sim = WilsonCowanSimulation(config)
        sim.reset()

        # Transient
        for _ in range(int(200 / dt)):
            sim.step()

        # Measure amplitude of E
        E_vals = []
        for _ in range(int(100 / dt)):
            sim.step()
            E_vals.append(sim.observe()[0])

        amp = max(E_vals) - min(E_vals)
        amplitudes.append(amp)

        if (i + 1) % 5 == 0:
            logger.info(f"  I_ext_E={I_ext:.3f}: amplitude={amp:.4f}")

    return {
        "I_ext": I_ext_values,
        "amplitude": np.array(amplitudes),
    }


def generate_frequency_data(
    n_I_ext: int = 20,
    dt: float = 0.01,
) -> dict[str, np.ndarray]:
    """Measure oscillation frequency across I_ext_E sweep."""
    I_ext_values = np.linspace(0.5, 3.0, n_I_ext)
    frequencies = []

    for I_ext in I_ext_values:
        config = SimulationConfig(
            domain=Domain.WILSON_COWAN,
            dt=dt,
            n_steps=1000,
            parameters={
                "tau_E": 1.0, "tau_I": 2.0,
                "w_EE": 16.0, "w_EI": 12.0,
                "w_IE": 15.0, "w_II": 3.0,
                "a": 1.3, "theta": 4.0,
                "I_ext_E": I_ext, "I_ext_I": 0.0,
                "E_0": 0.1, "I_0": 0.05,
            },
        )
        sim = WilsonCowanSimulation(config)
        spectrum = sim.frequency_spectrum(n_steps=8000)
        frequencies.append(spectrum["peak_freq"])

    return {
        "I_ext": I_ext_values,
        "frequency": np.array(frequencies),
    }


def run_wilson_cowan_rediscovery(
    output_dir: str | Path = "output/rediscovery/wilson_cowan",
    n_iterations: int = 40,
) -> dict:
    """Run Wilson-Cowan rediscovery pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "wilson_cowan",
        "targets": {
            "ode": "tau_E*dE/dt = -E + S(w_EE*E - w_EI*I + I_ext_E)",
            "hopf": "oscillation onset vs I_ext_E",
            "frequency": "E-I oscillation frequency",
            "nullclines": "E and I nullcline intersection",
        },
    }

    # --- Part 1: SINDy ODE recovery ---
    logger.info("Part 1: SINDy ODE recovery at I_ext_E=1.5...")
    data = generate_ode_data(I_ext_E=1.5, n_steps=20000, dt=0.005)

    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        sindy_discoveries = run_sindy(
            data["states"],
            dt=0.005,
            feature_names=["E", "I"],
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

    # --- Part 2: Hopf bifurcation ---
    logger.info("Part 2: Hopf bifurcation sweep...")
    hopf_data = generate_hopf_data(n_I_ext=25, dt=0.005)

    # Estimate bifurcation point
    threshold = 0.05
    above = hopf_data["amplitude"] > threshold
    if np.any(above):
        idx = np.argmax(above)
        I_ext_c = hopf_data["I_ext"][max(0, idx - 1)]
        results["hopf_bifurcation"] = {
            "I_ext_c_estimate": float(I_ext_c),
            "max_amplitude": float(np.max(hopf_data["amplitude"])),
        }
        logger.info(f"  Hopf bifurcation at I_ext_E ~ {I_ext_c:.4f}")
    else:
        results["hopf_bifurcation"] = {"note": "No oscillations detected"}

    # --- Part 3: Fixed points and eigenvalues ---
    logger.info("Part 3: Fixed points and eigenvalues...")
    config = SimulationConfig(
        domain=Domain.WILSON_COWAN,
        dt=0.01,
        n_steps=1000,
        parameters={
            "tau_E": 1.0, "tau_I": 2.0,
            "w_EE": 16.0, "w_EI": 12.0,
            "w_IE": 15.0, "w_II": 3.0,
            "a": 1.3, "theta": 4.0,
            "I_ext_E": 1.5, "I_ext_I": 0.0,
        },
    )
    sim = WilsonCowanSimulation(config)
    fps = sim.find_fixed_points()
    results["fixed_points"] = {
        "n_found": len(fps),
        "points": [{"E": float(fp[0]), "I": float(fp[1])} for fp in fps],
    }
    for fp in fps:
        eigs = sim.compute_eigenvalues(fp)
        logger.info(
            f"  FP at ({fp[0]:.4f}, {fp[1]:.4f}), "
            f"eigenvalues: {eigs[0]:.4f}, {eigs[1]:.4f}"
        )
        # Record eigenvalues for first fixed point
        if "eigenvalues" not in results["fixed_points"]:
            results["fixed_points"]["eigenvalues"] = [
                str(eigs[0]), str(eigs[1])
            ]
            results["fixed_points"]["oscillatory"] = bool(
                np.any(np.abs(np.imag(eigs)) > 1e-6)
            )

    # --- Part 4: Frequency analysis ---
    logger.info("Part 4: Frequency vs I_ext_E...")
    freq_data = generate_frequency_data(n_I_ext=20, dt=0.005)
    results["frequency"] = {
        "I_ext_range": [float(freq_data["I_ext"][0]),
                        float(freq_data["I_ext"][-1])],
        "max_frequency": float(np.max(freq_data["frequency"])),
        "mean_frequency": float(np.mean(
            freq_data["frequency"][freq_data["frequency"] > 0.1]
        )) if np.any(freq_data["frequency"] > 0.1) else 0.0,
    }
    logger.info(f"  Max frequency: {np.max(freq_data['frequency']):.4f}")

    # --- Part 5: Nullclines ---
    logger.info("Part 5: Nullcline computation...")
    null_data = sim.nullclines(n_points=100)
    n_valid_E = int(np.sum(np.isfinite(null_data["E_null_I"])))
    n_valid_I = int(np.sum(np.isfinite(null_data["I_null_I"])))
    results["nullclines"] = {
        "n_E_nullcline_points": n_valid_E,
        "n_I_nullcline_points": n_valid_I,
    }
    logger.info(
        f"  E-nullcline: {n_valid_E} valid points, "
        f"I-nullcline: {n_valid_I} valid points"
    )

    # PySR: try to fit I_ext_c as function of tau ratio
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Sweep tau_E/tau_I ratio and find bifurcation point
        tau_ratios = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0])
        I_ext_c_values = []
        for ratio in tau_ratios:
            tau_E_val = ratio
            config_sweep = SimulationConfig(
                domain=Domain.WILSON_COWAN,
                dt=0.005,
                n_steps=1000,
                parameters={
                    "tau_E": tau_E_val, "tau_I": 2.0,
                    "w_EE": 16.0, "w_EI": 12.0,
                    "w_IE": 15.0, "w_II": 3.0,
                    "a": 1.3, "theta": 4.0,
                    "I_ext_E": 0.0, "I_ext_I": 0.0,
                    "E_0": 0.1, "I_0": 0.05,
                },
            )
            sim_sweep = WilsonCowanSimulation(config_sweep)
            sweep_result = sim_sweep.hopf_bifurcation_sweep(
                I_ext_values=np.linspace(0.0, 3.0, 20),
                n_test_steps=3000,
            )
            above_sweep = sweep_result["amplitude"] > 0.05
            if np.any(above_sweep):
                idx_sweep = np.argmax(above_sweep)
                I_ext_c_values.append(
                    sweep_result["I_ext"][max(0, idx_sweep - 1)]
                )
            else:
                I_ext_c_values.append(np.nan)

        I_ext_c_arr = np.array(I_ext_c_values)
        valid = np.isfinite(I_ext_c_arr)
        if np.sum(valid) > 3:
            X = tau_ratios[valid].reshape(-1, 1)
            y = I_ext_c_arr[valid]

            logger.info("  Running PySR: I_ext_c = f(tau_ratio)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["tau_r"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=10,
                populations=15,
                population_size=30,
            )
            results["pysr_hopf"] = {
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
                results["pysr_hopf"]["best"] = best.expression
                results["pysr_hopf"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["pysr_hopf"] = {"error": str(e)}

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "hopf_sweep.npz",
        I_ext=hopf_data["I_ext"],
        amplitude=hopf_data["amplitude"],
    )

    return results
