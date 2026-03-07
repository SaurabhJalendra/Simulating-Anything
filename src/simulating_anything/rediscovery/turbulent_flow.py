"""2D turbulent flow rediscovery / analysis.

Targets:
- Energy spectrum scaling E(k) ~ k^(-3) (2D enstrophy cascade)
- Enstrophy decay rate vs viscosity
- Reynolds number dependence on viscosity
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.turbulent_flow import TurbulentFlow2DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def generate_energy_spectrum_data(
    nu: float = 1e-3,
    N: int = 64,
    n_warmup: int = 200,
    n_average: int = 50,
    dt: float = 0.005,
) -> dict[str, np.ndarray | float]:
    """Run simulation to statistical steady state and measure energy spectrum.

    Warms up the flow, then averages the energy spectrum over several snapshots.

    Args:
        nu: Kinematic viscosity.
        N: Grid resolution.
        n_warmup: Warmup steps to reach steady state.
        n_average: Number of snapshots to average spectrum over.
        dt: Timestep.

    Returns:
        Dict with wavenumber bins, averaged spectrum, and fit parameters.
    """
    config = SimulationConfig(
        domain=Domain.TURBULENT_FLOW,
        dt=dt,
        n_steps=1,
        parameters={
            "nu": nu,
            "N": float(N),
            "forcing_amplitude": 0.5,
            "forcing_k": 4.0,
        },
    )
    sim = TurbulentFlow2DSimulation(config)
    sim.reset(seed=42)

    # Warmup to statistical steady state
    for _ in range(n_warmup):
        sim.step()

    # Collect spectra
    all_spectra = []
    for _ in range(n_average):
        for _ in range(5):  # Step between snapshots
            sim.step()
        k_bins, E_k = sim.energy_spectrum()
        all_spectra.append(E_k)

    avg_spectrum = np.mean(all_spectra, axis=0)

    # Fit power law in inertial range (exclude forcing and dissipation scales)
    k_min_fit = 6
    k_max_fit = N // 4
    mask = (k_bins >= k_min_fit) & (k_bins <= k_max_fit) & (avg_spectrum > 0)

    slope = np.nan
    intercept = np.nan
    if np.sum(mask) >= 3:
        log_k = np.log(k_bins[mask])
        log_E = np.log(avg_spectrum[mask])
        coeffs = np.polyfit(log_k, log_E, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

    return {
        "k_bins": k_bins,
        "E_k": avg_spectrum,
        "slope": float(slope),
        "intercept": float(intercept),
        "nu": nu,
        "Re": sim.reynolds_number(),
    }


def generate_enstrophy_decay_data(
    n_nu_values: int = 10,
    N: int = 64,
    n_warmup: int = 200,
    n_measure: int = 100,
    dt: float = 0.005,
) -> dict[str, np.ndarray]:
    """Sweep viscosity and measure enstrophy decay rate.

    For each viscosity, run to steady state, then track enstrophy
    evolution to characterize dissipation.

    Args:
        n_nu_values: Number of viscosity values to sweep.
        N: Grid resolution.
        n_warmup: Warmup steps.
        n_measure: Steps for enstrophy measurement.
        dt: Timestep.

    Returns:
        Dict with nu values, enstrophy decay rates, and Reynolds numbers.
    """
    nu_values = np.logspace(-4, -2, n_nu_values)
    decay_rates = []
    re_values = []
    enstrophy_means = []

    for i, nu in enumerate(nu_values):
        config = SimulationConfig(
            domain=Domain.TURBULENT_FLOW,
            dt=dt,
            n_steps=1,
            parameters={
                "nu": nu,
                "N": float(N),
                "forcing_amplitude": 0.5,
                "forcing_k": 4.0,
            },
        )
        sim = TurbulentFlow2DSimulation(config)
        sim.reset(seed=42)

        # Warmup
        for _ in range(n_warmup):
            sim.step()

        re_values.append(sim.reynolds_number())

        # Measure enstrophy over time
        enstrophies = []
        for _ in range(n_measure):
            sim.step()
            enstrophies.append(sim.total_enstrophy())

        enstrophies_arr = np.array(enstrophies)
        enstrophy_means.append(float(np.mean(enstrophies_arr)))

        # Estimate decay rate from linear fit of log(enstrophy) vs time
        times = np.arange(n_measure) * dt
        if np.all(enstrophies_arr > 0):
            log_Z = np.log(enstrophies_arr)
            coeffs = np.polyfit(times, log_Z, 1)
            decay_rates.append(coeffs[0])
        else:
            decay_rates.append(np.nan)

        if (i + 1) % 5 == 0:
            logger.info(f"  Viscosity sweep {i + 1}/{n_nu_values}")

    return {
        "nu_values": nu_values,
        "decay_rates": np.array(decay_rates),
        "Re_values": np.array(re_values),
        "enstrophy_means": np.array(enstrophy_means),
    }


def generate_spectrum_vs_nu_data(
    n_nu_values: int = 8,
    N: int = 64,
    n_warmup: int = 200,
    n_average: int = 30,
    dt: float = 0.005,
) -> dict[str, list]:
    """Collect energy spectrum slopes at different viscosities.

    Args:
        n_nu_values: Number of viscosity values.
        N: Grid resolution.
        n_warmup: Warmup steps.
        n_average: Snapshots for averaging.
        dt: Timestep.

    Returns:
        Dict with list of slopes and Re per viscosity.
    """
    nu_values = np.logspace(-4, -2, n_nu_values)
    slopes = []
    re_list = []

    for i, nu in enumerate(nu_values):
        data = generate_energy_spectrum_data(
            nu=nu, N=N, n_warmup=n_warmup, n_average=n_average, dt=dt,
        )
        slopes.append(data["slope"])
        re_list.append(data["Re"])

        if (i + 1) % 4 == 0:
            logger.info(
                f"  Spectrum {i + 1}/{n_nu_values}: "
                f"nu={nu:.1e}, slope={data['slope']:.2f}"
            )

    return {
        "nu_values": nu_values.tolist(),
        "slopes": slopes,
        "Re_values": re_list,
    }


def run_turbulent_flow_analysis(
    output_dir: str | Path = "output/rediscovery/turbulent_flow",
    n_iterations: int = 40,
) -> dict:
    """Run the full 2D turbulent flow analysis.

    1. Measure energy spectrum and fit power law (target: E(k) ~ k^-3)
    2. Sweep viscosity, measure enstrophy decay rate
    3. Collect spectral slopes at multiple viscosities

    Returns dict with all results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "domain": "turbulent_flow",
        "targets": {
            "energy_spectrum": "E(k) ~ k^(-3) in enstrophy cascade range",
            "enstrophy_decay": "Enstrophy decay rate depends on nu",
        },
    }

    # --- Part 1: Energy spectrum power law ---
    logger.info("Part 1: Measuring energy spectrum at reference viscosity...")
    spec_data = generate_energy_spectrum_data(
        nu=1e-3, N=64, n_warmup=200, n_average=50, dt=0.005,
    )
    results["energy_spectrum"] = {
        "slope": spec_data["slope"],
        "target_slope": -3.0,
        "slope_error": abs(spec_data["slope"] - (-3.0)),
        "Re": spec_data["Re"],
        "nu": spec_data["nu"],
    }
    logger.info(
        f"  Spectral slope: {spec_data['slope']:.3f} (theory: -3.0)"
    )
    logger.info(f"  Reynolds number: {spec_data['Re']:.1f}")

    # PySR: E(k) = f(k) -- fit power law
    logger.info(
        f"  Running PySR for E(k) = f(k) with {n_iterations} iterations..."
    )
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        # Use wavenumbers in the cascade range with positive energy
        k_bins = spec_data["k_bins"]
        E_k = spec_data["E_k"]
        valid = (k_bins >= 3) & (k_bins <= 30) & (E_k > 0)

        if np.sum(valid) >= 5:
            X = k_bins[valid].reshape(-1, 1)
            y = E_k[valid]

            discoveries = run_symbolic_regression(
                X,
                y,
                variable_names=["k_"],
                n_iterations=n_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "square", "cube", "log"],
                max_complexity=12,
                populations=20,
                population_size=40,
            )

            results["spectrum_pysr"] = {
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
                results["spectrum_pysr"]["best"] = best.expression
                results["spectrum_pysr"]["best_r2"] = best.evidence.fit_r_squared
                logger.info(
                    f"  Best: {best.expression} "
                    f"(R2={best.evidence.fit_r_squared:.6f})"
                )
        else:
            results["spectrum_pysr"] = {"error": "Insufficient valid data points"}

    except Exception as e:
        logger.warning(f"PySR failed: {e}")
        results["spectrum_pysr"] = {"error": str(e)}

    # --- Part 2: Enstrophy decay vs viscosity ---
    logger.info("Part 2: Sweeping viscosity for enstrophy decay...")
    enstrophy_data = generate_enstrophy_decay_data(
        n_nu_values=10, N=64, n_warmup=200, n_measure=100, dt=0.005,
    )

    valid_decay = np.isfinite(enstrophy_data["decay_rates"])
    results["enstrophy_decay"] = {
        "n_nu_values": len(enstrophy_data["nu_values"]),
        "n_valid": int(np.sum(valid_decay)),
        "decay_rate_range": [
            float(np.nanmin(enstrophy_data["decay_rates"])),
            float(np.nanmax(enstrophy_data["decay_rates"])),
        ],
        "Re_range": [
            float(np.min(enstrophy_data["Re_values"])),
            float(np.max(enstrophy_data["Re_values"])),
        ],
    }
    logger.info(
        f"  Re range: [{np.min(enstrophy_data['Re_values']):.0f}, "
        f"{np.max(enstrophy_data['Re_values']):.0f}]"
    )

    # --- Part 3: Spectral slopes across viscosities ---
    logger.info("Part 3: Collecting spectra at multiple viscosities...")
    multi_spec = generate_spectrum_vs_nu_data(
        n_nu_values=8, N=64, n_warmup=200, n_average=30, dt=0.005,
    )

    valid_slopes = [s for s in multi_spec["slopes"] if np.isfinite(s)]
    results["multi_viscosity_spectra"] = {
        "n_viscosities": len(multi_spec["nu_values"]),
        "slopes": multi_spec["slopes"],
        "mean_slope": float(np.mean(valid_slopes)) if valid_slopes else float("nan"),
        "std_slope": float(np.std(valid_slopes)) if valid_slopes else float("nan"),
    }
    if valid_slopes:
        logger.info(
            f"  Mean slope: {np.mean(valid_slopes):.3f} +/- "
            f"{np.std(valid_slopes):.3f}"
        )

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # Save data
    np.savez(
        output_path / "spectrum_data.npz",
        k_bins=spec_data["k_bins"],
        E_k=spec_data["E_k"],
    )
    np.savez(
        output_path / "enstrophy_data.npz",
        nu_values=enstrophy_data["nu_values"],
        decay_rates=enstrophy_data["decay_rates"],
        Re_values=enstrophy_data["Re_values"],
    )

    return results
