"""Rayleigh-Benard convection rediscovery.

Targets:
- Critical Rayleigh number Ra_c ~ 657.5 (free-slip BCs)
- Nusselt number Nu(Ra) scaling above onset
- Convection roll wavelength ~ 2*H at onset
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.rayleigh_benard import (
    RA_CRITICAL_FREE_SLIP,
    RayleighBenardSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_onset_data(
    n_Ra: int = 30,
    n_steps: int = 8000,
    dt: float = 5e-5,
    Nx: int = 64,
    Nz: int = 32,
) -> dict[str, np.ndarray]:
    """Generate convection amplitude vs Ra data to identify onset.

    Sweeps Ra from below to above Ra_c and measures steady-state
    convection amplitude and Nusselt number.
    """
    Ra_values = np.concatenate([
        np.linspace(100, 600, 8),
        np.linspace(620, 700, 8),
        np.linspace(750, 2000, 8),
        np.linspace(2500, 5000, 6),
    ])
    Ra_values = np.sort(Ra_values)[:n_Ra]

    all_Ra = []
    all_amp = []
    all_nu = []

    for i, Ra in enumerate(Ra_values):
        config = SimulationConfig(
            domain=Domain.RAYLEIGH_BENARD,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "Ra": Ra, "Pr": 1.0,
                "Lx": 2.0, "H": 1.0,
                "Nx": float(Nx), "Nz": float(Nz),
                "perturbation_amp": 0.01,
            },
        )
        sim = RayleighBenardSimulation(config)
        sim.reset(seed=42)

        # Run to steady state
        for _ in range(n_steps):
            sim.step()

        amp = sim.convection_amplitude()
        nu_ = sim.compute_nusselt()

        all_Ra.append(Ra)
        all_amp.append(amp)
        all_nu.append(nu_)

        if (i + 1) % 10 == 0:
            logger.info(f"  Ra={Ra:.0f}: amp={amp:.6f}, Nu={nu_:.4f}")

    return {
        "Ra": np.array(all_Ra),
        "amplitude": np.array(all_amp),
        "nusselt": np.array(all_nu),
    }


def find_critical_ra(
    Ra_values: np.ndarray,
    amplitudes: np.ndarray,
    threshold_fraction: float = 0.01,
) -> float:
    """Estimate critical Ra from amplitude data.

    Uses interpolation to find the Ra where amplitude first exceeds
    a threshold (fraction of max amplitude).
    """
    max_amp = np.max(amplitudes)
    if max_amp < 1e-12:
        return float("inf")

    threshold = threshold_fraction * max_amp
    above = amplitudes > threshold

    if not np.any(above):
        return float("inf")

    # First index above threshold
    idx = np.argmax(above)
    if idx == 0:
        return float(Ra_values[0])

    # Linear interpolation between last-below and first-above
    Ra_below = Ra_values[idx - 1]
    Ra_above = Ra_values[idx]
    amp_below = amplitudes[idx - 1]
    amp_above = amplitudes[idx]

    if amp_above - amp_below < 1e-15:
        return float(Ra_above)

    frac = (threshold - amp_below) / (amp_above - amp_below)
    return float(Ra_below + frac * (Ra_above - Ra_below))


def run_rayleigh_benard_rediscovery(
    output_dir: str | Path = "output/rediscovery/rayleigh_benard",
    n_iterations: int = 40,
) -> dict:
    """Run the full Rayleigh-Benard rediscovery.

    1. Sweep Ra to identify convection onset (Ra_c)
    2. Measure Nu(Ra) scaling above onset
    3. Run PySR to find Nu(Ra) relationship

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": "rayleigh_benard",
        "targets": {
            "critical_Ra": f"Ra_c ~ {RA_CRITICAL_FREE_SLIP:.1f} (free-slip)",
            "nusselt_scaling": "Nu(Ra) > 1 above onset",
            "roll_wavelength": "lambda ~ 2*H at onset",
        },
    }

    # --- Part 1: Convection onset sweep ---
    logger.info("Part 1: Sweeping Ra for convection onset...")
    data = generate_onset_data(n_Ra=25, n_steps=6000, dt=5e-5, Nx=64, Nz=32)

    # Find critical Ra
    Ra_c_est = find_critical_ra(data["Ra"], data["amplitude"])
    Ra_c_theory = RA_CRITICAL_FREE_SLIP
    Ra_c_error = abs(Ra_c_est - Ra_c_theory) / Ra_c_theory

    results["critical_Ra"] = {
        "estimate": Ra_c_est,
        "theory": Ra_c_theory,
        "relative_error": Ra_c_error,
    }
    logger.info(f"  Ra_c estimate: {Ra_c_est:.1f} (theory: {Ra_c_theory:.1f})")
    logger.info(f"  Relative error: {Ra_c_error:.2%}")

    # --- Part 2: Nusselt number data ---
    logger.info("Part 2: Analyzing Nusselt number scaling...")
    above_onset = data["Ra"] > Ra_c_est
    if np.sum(above_onset) >= 3:
        Ra_above = data["Ra"][above_onset]
        Nu_above = data["nusselt"][above_onset]

        results["nusselt_data"] = {
            "n_points_above_onset": int(np.sum(above_onset)),
            "Nu_range": [float(np.min(Nu_above)), float(np.max(Nu_above))],
            "Ra_range": [float(np.min(Ra_above)), float(np.max(Ra_above))],
        }

        # Check that Nu increases with Ra
        if len(Ra_above) >= 3:
            corr = np.corrcoef(Ra_above, Nu_above)[0, 1]
            results["nusselt_data"]["correlation_with_Ra"] = float(corr)
            logger.info(f"  Nu-Ra correlation: {corr:.4f}")

    # --- Part 3: PySR on Nu(Ra) ---
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        if np.sum(above_onset) >= 5:
            X = data["Ra"][above_onset].reshape(-1, 1)
            y = data["nusselt"][above_onset]

            logger.info("  Running PySR: Nu = f(Ra)...")
            discoveries = run_symbolic_regression(
                X, y,
                variable_names=["Ra"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square"],
                max_complexity=12,
                populations=20,
                population_size=40,
            )
            results["nusselt_pysr"] = {
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
                results["nusselt_pysr"]["best"] = best.expression
                results["nusselt_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["nusselt_pysr"] = {"error": str(e)}

    # Save
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    np.savez(
        output_path / "onset_data.npz",
        **{k: v for k, v in data.items() if isinstance(v, np.ndarray)},
    )

    return results
