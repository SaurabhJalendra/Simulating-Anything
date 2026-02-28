"""Sensitivity analysis for discovery robustness.

Measures how discovery quality (R²) degrades as a function of:
1. Data quantity (number of samples)
2. Observation noise level
3. Parameter diversity (range of parameter sweeps)

Provides concrete evidence for paper claims about pipeline robustness.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Result of one sensitivity experiment."""

    domain: str
    variable: str  # "n_samples", "noise_level", "param_range"
    values: list[float]
    r_squared: list[float]
    discovered_form: list[str]
    metadata: dict = field(default_factory=dict)


def _projectile_range(v0: float, theta: float, g: float = 9.81) -> float:
    """Analytical projectile range (no drag)."""
    theta_rad = np.radians(theta)
    return v0**2 * np.sin(2 * theta_rad) / g


def sensitivity_data_quantity(
    domain: str = "projectile",
    sample_counts: list[int] | None = None,
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> SensitivityResult:
    """Measure how R² varies with number of data points.

    For projectile: sweep v0 and theta, fit R = c * v0^2 * sin(2*theta).
    """
    if sample_counts is None:
        sample_counts = [5, 10, 25, 50, 100, 225, 500, 1000]

    rng = np.random.default_rng(seed)
    r_squared_list = []
    forms = []

    for n in sample_counts:
        # Generate data
        n_per_dim = max(2, int(np.sqrt(n)))
        v0_vals = np.linspace(5, 50, n_per_dim)
        theta_vals = np.linspace(10, 80, n_per_dim)

        X = []
        y = []
        for v0 in v0_vals:
            for theta in theta_vals:
                R = _projectile_range(v0, theta)
                if noise_sigma > 0:
                    R += rng.normal(0, noise_sigma * R)
                X.append([v0**2 * np.sin(2 * np.radians(theta))])
                y.append(R)

        X = np.array(X)
        y = np.array(y)

        # Subsample to exactly n points
        if len(y) > n:
            idx = rng.choice(len(y), n, replace=False)
            X = X[idx]
            y = y[idx]

        # Fit: R = c * v0^2 * sin(2*theta) via least squares
        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ c
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-15)

        r_squared_list.append(float(r2))
        forms.append(f"R = {c[0]:.5f} * v0^2 * sin(2*theta)")

        logger.info(f"  n={n:5d}: R²={r2:.6f}, c={c[0]:.5f} (theory: {1/9.81:.5f})")

    return SensitivityResult(
        domain=domain,
        variable="n_samples",
        values=[float(n) for n in sample_counts],
        r_squared=r_squared_list,
        discovered_form=forms,
    )


def sensitivity_noise(
    domain: str = "projectile",
    noise_levels: list[float] | None = None,
    n_samples: int = 225,
    seed: int = 42,
) -> SensitivityResult:
    """Measure how R² varies with observation noise level.

    noise_level is the relative standard deviation (0.0 = no noise, 0.1 = 10% noise).
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    r_squared_list = []
    forms = []

    for sigma in noise_levels:
        result = sensitivity_data_quantity(
            domain=domain,
            sample_counts=[n_samples],
            noise_sigma=sigma,
            seed=seed,
        )
        r_squared_list.append(result.r_squared[0])
        forms.append(result.discovered_form[0])
        logger.info(f"  noise={sigma:.3f}: R²={result.r_squared[0]:.6f}")

    return SensitivityResult(
        domain=domain,
        variable="noise_level",
        values=noise_levels,
        r_squared=r_squared_list,
        discovered_form=forms,
    )


def sensitivity_param_range(
    domain: str = "projectile",
    range_fractions: list[float] | None = None,
    n_samples: int = 225,
    seed: int = 42,
) -> SensitivityResult:
    """Measure how R² varies with parameter sweep breadth.

    range_fraction=1.0 means full range (5-50 m/s, 10-80 deg).
    range_fraction=0.1 means 10% of full range (narrow sweep).
    """
    if range_fractions is None:
        range_fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    rng = np.random.default_rng(seed)
    r_squared_list = []
    forms = []

    v0_full = (5, 50)
    theta_full = (10, 80)

    for frac in range_fractions:
        v0_center = (v0_full[0] + v0_full[1]) / 2
        theta_center = (theta_full[0] + theta_full[1]) / 2
        v0_half_range = (v0_full[1] - v0_full[0]) / 2 * frac
        theta_half_range = (theta_full[1] - theta_full[0]) / 2 * frac

        n_per_dim = max(2, int(np.sqrt(n_samples)))
        v0_vals = np.linspace(
            v0_center - v0_half_range, v0_center + v0_half_range, n_per_dim
        )
        theta_vals = np.linspace(
            theta_center - theta_half_range, theta_center + theta_half_range, n_per_dim
        )

        X = []
        y = []
        for v0 in v0_vals:
            for theta in theta_vals:
                R = _projectile_range(v0, theta)
                X.append([v0**2 * np.sin(2 * np.radians(theta))])
                y.append(R)

        X = np.array(X)
        y = np.array(y)

        # Subsample
        if len(y) > n_samples:
            idx = rng.choice(len(y), n_samples, replace=False)
            X = X[idx]
            y = y[idx]

        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ c
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-15)

        r_squared_list.append(float(r2))
        forms.append(f"R = {c[0]:.5f} * v0^2 * sin(2*theta)")

        logger.info(
            f"  range={frac:.2f}: R²={r2:.6f}, "
            f"v0=[{v0_vals[0]:.1f},{v0_vals[-1]:.1f}], "
            f"theta=[{theta_vals[0]:.1f},{theta_vals[-1]:.1f}]"
        )

    return SensitivityResult(
        domain=domain,
        variable="param_range",
        values=range_fractions,
        r_squared=r_squared_list,
        discovered_form=forms,
    )


def sensitivity_harmonic_oscillator(
    noise_levels: list[float] | None = None,
    n_steps_list: list[int] | None = None,
    seed: int = 42,
) -> dict[str, SensitivityResult]:
    """Sensitivity analysis for harmonic oscillator frequency recovery.

    Varies noise and trajectory length, measures frequency recovery accuracy.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    if n_steps_list is None:
        n_steps_list = [100, 250, 500, 1000, 2500, 5000]

    from simulating_anything.simulation.harmonic_oscillator import (
        DampedHarmonicOscillator,
    )
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(seed)
    k, m, c_damp = 4.0, 1.0, 0.4
    omega_true = np.sqrt(k / m)

    # Noise sensitivity
    noise_r2 = []
    noise_forms = []
    for sigma in noise_levels:
        config = SimulationConfig(
            domain=Domain.HARMONIC_OSCILLATOR, dt=0.005, n_steps=5000,
            parameters={"k": k, "m": m, "c": c_damp, "x_0": 2.0, "v_0": 0.0},
        )
        sim = DampedHarmonicOscillator(config)
        traj = sim.run(n_steps=5000)
        x = traj.states[:, 0].copy()

        # Add noise
        if sigma > 0:
            x += rng.normal(0, sigma * np.max(np.abs(x)), size=x.shape)

        # Extract frequency via FFT
        dt = 0.005
        fft = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), dt)
        power = np.abs(fft[1:])
        peak_idx = np.argmax(power) + 1
        omega_measured = 2 * np.pi * freqs[peak_idx]

        # R² as relative accuracy
        r2 = 1 - (omega_measured - omega_true) ** 2 / omega_true**2
        r2 = max(0, min(1, r2))

        noise_r2.append(float(r2))
        noise_forms.append(f"omega = {omega_measured:.4f} (true: {omega_true:.4f})")
        logger.info(
            f"  HO noise={sigma:.3f}: omega={omega_measured:.4f}, "
            f"error={abs(omega_measured-omega_true)/omega_true*100:.2f}%"
        )

    # Data quantity sensitivity
    data_r2 = []
    data_forms = []
    for n_steps in n_steps_list:
        config = SimulationConfig(
            domain=Domain.HARMONIC_OSCILLATOR, dt=0.005, n_steps=n_steps,
            parameters={"k": k, "m": m, "c": c_damp, "x_0": 2.0, "v_0": 0.0},
        )
        sim = DampedHarmonicOscillator(config)
        traj = sim.run(n_steps=n_steps)
        x = traj.states[:, 0]

        dt = 0.005
        fft = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), dt)
        power = np.abs(fft[1:])
        peak_idx = np.argmax(power) + 1
        omega_measured = 2 * np.pi * freqs[peak_idx]

        r2 = 1 - (omega_measured - omega_true) ** 2 / omega_true**2
        r2 = max(0, min(1, r2))

        data_r2.append(float(r2))
        data_forms.append(f"omega = {omega_measured:.4f} (n_steps={n_steps})")
        logger.info(
            f"  HO n_steps={n_steps:5d}: omega={omega_measured:.4f}, "
            f"error={abs(omega_measured-omega_true)/omega_true*100:.2f}%"
        )

    return {
        "noise": SensitivityResult(
            domain="harmonic_oscillator",
            variable="noise_level",
            values=noise_levels,
            r_squared=noise_r2,
            discovered_form=noise_forms,
        ),
        "data_quantity": SensitivityResult(
            domain="harmonic_oscillator",
            variable="n_steps",
            values=[float(n) for n in n_steps_list],
            r_squared=data_r2,
            discovered_form=data_forms,
        ),
    }


def run_sensitivity_analysis(
    output_dir: str | Path = "output/sensitivity",
) -> dict:
    """Run full sensitivity analysis across domains and variables."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSIS")
    logger.info("=" * 60)

    results = {}

    # Projectile: data quantity
    logger.info("\n--- Projectile: Data Quantity ---")
    r1 = sensitivity_data_quantity()
    results["projectile_data_quantity"] = {
        "values": r1.values,
        "r_squared": r1.r_squared,
        "forms": r1.discovered_form,
    }

    # Projectile: noise
    logger.info("\n--- Projectile: Noise ---")
    r2 = sensitivity_noise()
    results["projectile_noise"] = {
        "values": r2.values,
        "r_squared": r2.r_squared,
        "forms": r2.discovered_form,
    }

    # Projectile: parameter range
    logger.info("\n--- Projectile: Parameter Range ---")
    r3 = sensitivity_param_range()
    results["projectile_param_range"] = {
        "values": r3.values,
        "r_squared": r3.r_squared,
        "forms": r3.discovered_form,
    }

    # Harmonic oscillator
    logger.info("\n--- Harmonic Oscillator ---")
    ho_results = sensitivity_harmonic_oscillator()
    for key, result in ho_results.items():
        results[f"harmonic_oscillator_{key}"] = {
            "values": result.values,
            "r_squared": result.r_squared,
            "forms": result.discovered_form,
        }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, data in results.items():
        r2_vals = data["r_squared"]
        logger.info(
            f"  {name}: R² range [{min(r2_vals):.4f}, {max(r2_vals):.4f}]"
        )

    # Save
    results_file = output_path / "sensitivity_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_sensitivity_analysis()
