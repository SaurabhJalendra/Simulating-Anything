"""Rediscovery: Stochastic resonance in a bistable system.

Targets:
1. SNR peaks at optimal noise intensity D_opt
2. Kramers rate: r_K ~ exp(-Delta_V / D)
3. Residence time distribution
4. Signal amplification factor vs noise
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run_stochastic_resonance_rediscovery(
    output_dir: str | Path = "output/rediscovery/stochastic_resonance",
    n_iterations: int = 40,
) -> dict:
    """Run stochastic resonance analysis."""
    from simulating_anything.simulation.stochastic_resonance import (
        StochasticResonanceSimulation,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"domain": "stochastic_resonance", "targets": []}

    # Part 1: SNR vs noise intensity sweep
    logger.info("Part 1: SNR vs noise intensity sweep...")
    noise_values = np.logspace(-2, 1, 30)
    snr_values = []
    amplification = []
    signal_omega = 0.1
    signal_amp = 0.3
    n_steps = 50000
    dt = 0.01

    for D in noise_values:
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=dt, n_steps=n_steps,
            parameters={
                "gamma": 0.5, "signal_amplitude": signal_amp,
                "signal_omega": signal_omega, "noise_intensity": float(D),
            },
        )
        sim = StochasticResonanceSimulation(config)
        sim.reset(seed=42)

        positions = []
        for _ in range(n_steps):
            sim.step()
            positions.append(sim.observe()[0])

        positions = np.array(positions)

        # Compute power spectrum
        fft = np.fft.rfft(positions)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(positions), d=dt)

        # Find signal peak
        signal_idx = np.argmin(np.abs(freqs - signal_omega / (2 * np.pi)))
        signal_power = power[max(0, signal_idx - 2):signal_idx + 3].max()

        # Noise floor (excluding signal peak)
        mask = np.ones(len(power), dtype=bool)
        mask[max(0, signal_idx - 5):signal_idx + 6] = False
        noise_floor = np.mean(power[mask]) if mask.sum() > 0 else 1e-10

        snr = float(signal_power / max(noise_floor, 1e-10))
        snr_values.append(snr)

        # Signal amplification: correlation with pure signal
        t_arr = np.arange(n_steps) * dt
        pure_signal = signal_amp * np.sin(signal_omega * t_arr)
        corr = float(np.abs(np.corrcoef(positions, pure_signal)[0, 1]))
        amplification.append(corr)

    snr_arr = np.array(snr_values)
    best_idx = np.argmax(snr_arr)
    D_opt = float(noise_values[best_idx])
    logger.info(f"  Optimal noise D_opt = {D_opt:.4f}, SNR = {snr_arr[best_idx]:.2f}")

    results["snr_sweep"] = {
        "noise_values": noise_values.tolist(),
        "snr_values": [float(s) for s in snr_values],
        "amplification": [float(a) for a in amplification],
        "D_opt": D_opt,
        "max_snr": float(snr_arr[best_idx]),
    }

    # Part 2: Kramers escape rate vs noise
    logger.info("Part 2: Kramers escape rate measurement...")
    escape_rates = []
    noise_for_kramers = np.logspace(-1, 0.5, 15)
    delta_V = 0.25  # Barrier height for V(x) = x^4/4 - x^2/2

    for D in noise_for_kramers:
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=dt, n_steps=100000,
            parameters={
                "gamma": 0.5, "signal_amplitude": 0.0,
                "signal_omega": 0.0, "noise_intensity": float(D),
            },
        )
        sim = StochasticResonanceSimulation(config)
        sim.reset(seed=42)

        crossings = 0
        prev_x = sim.observe()[0]
        for _ in range(100000):
            sim.step()
            x = sim.observe()[0]
            if prev_x * x < 0:  # Zero crossing
                crossings += 1
            prev_x = x

        rate = crossings / (100000 * dt)  # Crossings per unit time
        escape_rates.append(float(rate))
        logger.info(f"  D={D:.3f}: escape_rate={rate:.4f}")

    results["kramers"] = {
        "noise_values": noise_for_kramers.tolist(),
        "escape_rates": escape_rates,
        "delta_V": delta_V,
    }

    # Part 3: PySR on SNR vs noise
    logger.info("Part 3: PySR for SNR = f(D)...")
    try:
        from pysr import PySRRegressor

        valid = [(n, s) for n, s in zip(noise_values, snr_values) if np.isfinite(s) and s > 0]
        if len(valid) > 10:
            D_arr = np.array([v[0] for v in valid]).reshape(-1, 1)
            snr_log = np.log10(np.array([v[1] for v in valid]) + 1)

            model = PySRRegressor(
                niterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "log", "sqrt", "inv"],
                maxsize=15,
                populations=15,
                progress=False,
                verbosity=0,
            )
            model.fit(D_arr, snr_log, variable_names=["D"])

            best_eq = str(model.sympy())
            r2 = float(model.score(D_arr, snr_log))
            logger.info(f"  PySR SNR: {best_eq} (R2={r2:.6f})")
            results["pysr_snr"] = {"equation": best_eq, "r2": r2}
        else:
            logger.warning("  Not enough valid SNR data for PySR")
    except Exception as e:
        logger.warning(f"  PySR failed: {e}")

    # Part 4: PySR on Kramers rate vs noise
    logger.info("Part 4: PySR for escape_rate = f(D)...")
    try:
        from pysr import PySRRegressor

        valid_kr = [(n, r) for n, r in zip(noise_for_kramers, escape_rates)
                    if np.isfinite(r) and r > 0]
        if len(valid_kr) > 5:
            D_kr = np.array([v[0] for v in valid_kr]).reshape(-1, 1)
            rates = np.array([v[1] for v in valid_kr])

            model = PySRRegressor(
                niterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "log", "sqrt", "inv"],
                maxsize=12,
                populations=15,
                progress=False,
                verbosity=0,
            )
            model.fit(D_kr, rates, variable_names=["D"])

            best_eq = str(model.sympy())
            r2 = float(model.score(D_kr, rates))
            logger.info(f"  PySR Kramers: {best_eq} (R2={r2:.6f})")
            results["pysr_kramers"] = {"equation": best_eq, "r2": r2}
        else:
            logger.warning("  Not enough valid Kramers data for PySR")
    except Exception as e:
        logger.warning(f"  PySR Kramers failed: {e}")

    # Find best R2
    best_r2 = None
    for key in ["pysr_snr", "pysr_kramers"]:
        if key in results and "r2" in results[key]:
            r2 = results[key]["r2"]
            if best_r2 is None or r2 > best_r2:
                best_r2 = r2
    if best_r2 is not None:
        results["best_r2"] = best_r2

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_dir / 'results.json'}")

    return results
