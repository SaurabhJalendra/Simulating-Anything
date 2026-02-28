"""Gray-Scott reaction-diffusion pattern analysis.

Targets:
- Turing instability boundary in (f, k) parameter space
- Dominant wavelength scaling: lambda ~ sqrt(D/k)
- Phase diagram showing pattern types (spots, stripes, waves, uniform)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.reaction_diffusion import GrayScottSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def compute_dominant_wavelength(field: np.ndarray, domain_size: float) -> float:
    """Compute dominant spatial wavelength from 2D field using FFT.

    Args:
        field: 2D array (nx, ny).
        domain_size: Physical size of the domain.

    Returns:
        Dominant wavelength (physical units), or 0 if no pattern.
    """
    # Remove mean
    field_centered = field - np.mean(field)

    # 2D FFT
    fft2 = np.fft.fft2(field_centered)
    power = np.abs(fft2) ** 2

    nx, ny = field.shape
    freq_x = np.fft.fftfreq(nx, d=domain_size / nx)
    freq_y = np.fft.fftfreq(ny, d=domain_size / ny)
    kx, ky = np.meshgrid(freq_x, freq_y, indexing="ij")
    k_mag = np.sqrt(kx**2 + ky**2)

    # Exclude DC component
    power[0, 0] = 0

    # Find peak wavenumber
    peak_idx = np.unravel_index(np.argmax(power), power.shape)
    k_peak = k_mag[peak_idx]

    if k_peak > 0:
        return 1.0 / k_peak
    return 0.0


def compute_pattern_energy(field: np.ndarray) -> float:
    """Compute pattern energy (variance of field) as measure of pattern strength."""
    return float(np.var(field))


def classify_pattern(v_field: np.ndarray, threshold: float = 1e-4) -> str:
    """Classify pattern type from the v concentration field.

    Returns one of: 'uniform', 'spots', 'stripes', 'complex'.
    """
    energy = compute_pattern_energy(v_field)
    if energy < threshold:
        return "uniform"

    # Use FFT to distinguish patterns
    fft2 = np.fft.fft2(v_field - np.mean(v_field))
    power = np.abs(fft2) ** 2
    power[0, 0] = 0

    # Compute radial power spectrum
    nx, ny = v_field.shape
    cx, cy = nx // 2, ny // 2
    Y, X = np.ogrid[-cx:nx - cx, -cy:ny - cy]
    R = np.sqrt(X**2 + Y**2).astype(int)

    power_shifted = np.fft.fftshift(power)
    max_r = min(cx, cy)
    radial_profile = np.zeros(max_r)
    for r in range(max_r):
        mask = R == r
        if np.any(mask):
            radial_profile[r] = np.mean(power_shifted[mask])

    # Peak sharpness indicates pattern regularity
    if radial_profile.max() > 0:
        peak_r = np.argmax(radial_profile[1:]) + 1
        peak_val = radial_profile[peak_r]
        mean_val = np.mean(radial_profile[1:])

        if peak_val > 10 * mean_val:
            # Check angular distribution at peak radius
            # Spots: isotropic, Stripes: anisotropic
            ring_mask = (R >= peak_r - 1) & (R <= peak_r + 1)
            angular_power = power_shifted[ring_mask]
            angular_cv = np.std(angular_power) / (np.mean(angular_power) + 1e-10)

            if angular_cv > 1.5:
                return "stripes"
            return "spots"

    return "complex"


def run_gray_scott_analysis(
    output_dir: str | Path = "output/rediscovery/gray_scott",
    grid_size: int = 128,
    n_steps: int = 10000,
) -> dict:
    """Run Gray-Scott pattern analysis across parameter space.

    1. Scan (f, k) parameter space
    2. For each combination, run simulation and analyze patterns
    3. Compute dominant wavelengths
    4. Build phase diagram
    5. Test wavelength scaling relationships

    Returns dict with phase diagram, wavelength data, and analysis.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    domain_size = 2.5
    dx = domain_size / grid_size
    D_u, D_v = 0.16, 0.08
    # CFL-safe dt
    dt = min(0.5 * dx**2 / (4 * D_u), 0.0005)

    # Scan (f, k) parameter space -- known interesting region
    f_values = np.linspace(0.01, 0.07, 13)
    k_values = np.linspace(0.04, 0.07, 13)

    phase_diagram = []
    wavelength_data = []

    total = len(f_values) * len(k_values)
    count = 0

    for f_val in f_values:
        for k_val in k_values:
            count += 1
            config = SimulationConfig(
                domain=Domain.REACTION_DIFFUSION,
                grid_resolution=(grid_size, grid_size),
                domain_size=(domain_size, domain_size),
                dt=dt,
                n_steps=n_steps,
                parameters={
                    "D_u": D_u,
                    "D_v": D_v,
                    "f": float(f_val),
                    "k": float(k_val),
                },
                seed=42,
            )

            sim = GrayScottSimulation(config)
            state = sim.reset()

            # Run simulation
            for _ in range(n_steps):
                state = sim.step()

            v_field = state[:, :, 1]  # v concentration
            energy = compute_pattern_energy(v_field)
            pattern_type = classify_pattern(v_field)
            wavelength = compute_dominant_wavelength(v_field, domain_size)

            phase_diagram.append({
                "f": float(f_val),
                "k": float(k_val),
                "pattern_type": pattern_type,
                "energy": energy,
                "wavelength": wavelength,
            })

            if pattern_type != "uniform" and wavelength > 0:
                wavelength_data.append({
                    "f": float(f_val),
                    "k": float(k_val),
                    "D_v": D_v,
                    "wavelength": wavelength,
                })

            if count % 20 == 0 or count == total:
                logger.info(f"  Scanned {count}/{total} parameter combinations")

    # Analyze wavelength scaling
    scaling_results = {}
    if wavelength_data:
        wl_arr = np.array([d["wavelength"] for d in wavelength_data])
        k_arr = np.array([d["k"] for d in wavelength_data])
        f_arr = np.array([d["f"] for d in wavelength_data])
        Dv_arr = np.array([d["D_v"] for d in wavelength_data])

        # Test lambda ~ sqrt(D_v / k)
        predicted_scale = np.sqrt(Dv_arr / k_arr)
        corr_Dv_k = np.corrcoef(wl_arr, predicted_scale)[0, 1] if len(wl_arr) > 2 else 0.0

        # Try PySR on wavelength data if enough points
        scaling_results = {
            "n_patterned_points": len(wavelength_data),
            "correlation_lambda_vs_sqrt_Dv_over_k": float(corr_Dv_k),
            "mean_wavelength": float(np.mean(wl_arr)),
            "std_wavelength": float(np.std(wl_arr)),
        }

        if len(wavelength_data) >= 10:
            try:
                from simulating_anything.analysis.symbolic_regression import (
                    run_symbolic_regression,
                )

                X_wl = np.column_stack([f_arr, k_arr, Dv_arr])
                discoveries = run_symbolic_regression(
                    X_wl,
                    wl_arr,
                    variable_names=["f", "k", "D_v"],
                    n_iterations=30,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sqrt", "square"],
                    max_complexity=15,
                )
                scaling_results["pysr_discoveries"] = [
                    {"expression": d.expression, "r_squared": d.evidence.fit_r_squared}
                    for d in discoveries[:5]
                ]
                if discoveries:
                    scaling_results["best_scaling_equation"] = discoveries[0].expression
                    scaling_results["best_scaling_r2"] = discoveries[0].evidence.fit_r_squared
            except Exception as e:
                logger.warning(f"PySR wavelength analysis failed: {e}")

    # Count pattern types
    type_counts = {}
    for p in phase_diagram:
        t = p["pattern_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    # Find instability boundary
    boundary_points = []
    for i, p in enumerate(phase_diagram):
        if p["pattern_type"] != "uniform":
            # Check if any neighbor is uniform
            for j, q in enumerate(phase_diagram):
                if (
                    q["pattern_type"] == "uniform"
                    and abs(p["f"] - q["f"]) <= (f_values[1] - f_values[0]) * 1.1
                    and abs(p["k"] - q["k"]) <= (k_values[1] - k_values[0]) * 1.1
                ):
                    boundary_points.append({
                        "f": (p["f"] + q["f"]) / 2,
                        "k": (p["k"] + q["k"]) / 2,
                    })
                    break

    results = {
        "domain": "gray_scott",
        "grid_size": grid_size,
        "n_steps": n_steps,
        "dt": dt,
        "D_u": D_u,
        "D_v": D_v,
        "f_range": [float(f_values[0]), float(f_values[-1])],
        "k_range": [float(k_values[0]), float(k_values[-1])],
        "n_parameter_combinations": total,
        "pattern_type_counts": type_counts,
        "n_boundary_points": len(boundary_points),
        "boundary_points": boundary_points[:20],
        "scaling_analysis": scaling_results,
        "phase_diagram": phase_diagram,
    }

    # Save results
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    logger.info(f"Pattern types found: {type_counts}")
    logger.info(f"Instability boundary: {len(boundary_points)} points")
    if scaling_results:
        logger.info(
            f"Wavelength correlation with sqrt(D_v/k): "
            f"{scaling_results.get('correlation_lambda_vs_sqrt_Dv_over_k', 'N/A'):.4f}"
        )

    return results
