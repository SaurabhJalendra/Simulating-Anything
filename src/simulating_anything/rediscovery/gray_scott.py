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
    """Compute dominant spatial wavelength from 2D field using radial power spectrum.

    Args:
        field: 2D array (nx, ny).
        domain_size: Physical size of the domain.

    Returns:
        Dominant wavelength (in grid units), or 0 if no pattern.
    """
    # Remove mean
    field_centered = field - np.mean(field)
    if np.std(field_centered) < 1e-10:
        return 0.0

    # 2D FFT and power spectrum
    fft2 = np.fft.fft2(field_centered)
    power = np.abs(fft2) ** 2
    power_shifted = np.fft.fftshift(power)

    nx, ny = field.shape
    cx, cy = nx // 2, ny // 2

    # Compute radial power spectrum (average power at each wavenumber)
    Y, X = np.ogrid[-cx:nx - cx, -cy:ny - cy]
    R = np.sqrt(X**2 + Y**2).astype(int)
    max_r = min(cx, cy)

    radial_power = np.zeros(max_r)
    for r_val in range(1, max_r):  # Skip DC (r=0)
        mask = R == r_val
        if np.any(mask):
            radial_power[r_val] = np.mean(power_shifted[mask])

    # Find peak wavenumber (skip very low frequencies r < 2)
    if np.max(radial_power[2:]) > 0:
        peak_r = np.argmax(radial_power[2:]) + 2
        # Convert wavenumber index to wavelength: lambda = N / k
        wavelength = domain_size / peak_r
        return float(wavelength)

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


def _run_gray_scott_jax(
    u_init: np.ndarray,
    v_init: np.ndarray,
    D_u: float,
    D_v: float,
    f: float,
    k: float,
    dt: float,
    dx: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Gray-Scott simulation entirely in JAX for speed.

    Uses unscaled discrete Laplacian (Karl Sims convention): the diffusion
    coefficient directly controls the mixing rate per timestep without
    normalizing by grid spacing. This is the standard convention for
    Gray-Scott pattern formation studies.
    """
    try:
        import jax
        import jax.numpy as jnp

        def step_fn(carry, _):
            u, v = carry
            # Unscaled discrete Laplacian (no /dx^2)
            lap_u = (jnp.roll(u, 1, 0) + jnp.roll(u, -1, 0)
                     + jnp.roll(u, 1, 1) + jnp.roll(u, -1, 1) - 4.0 * u)
            lap_v = (jnp.roll(v, 1, 0) + jnp.roll(v, -1, 0)
                     + jnp.roll(v, 1, 1) + jnp.roll(v, -1, 1) - 4.0 * v)
            uvv = u * v * v
            u_new = u + dt * (D_u * lap_u - uvv + f * (1.0 - u))
            v_new = v + dt * (D_v * lap_v + uvv - (f + k) * v)
            return (u_new, v_new), None

        u_jax = jnp.array(u_init)
        v_jax = jnp.array(v_init)
        (u_final, v_final), _ = jax.lax.scan(step_fn, (u_jax, v_jax), None, length=n_steps)
        return np.asarray(u_final), np.asarray(v_final)

    except ImportError:
        u, v = u_init.copy(), v_init.copy()
        for _ in range(n_steps):
            lap_u = (np.roll(u, 1, 0) + np.roll(u, -1, 0)
                     + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u)
            lap_v = (np.roll(v, 1, 0) + np.roll(v, -1, 0)
                     + np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4 * v)
            uvv = u * v * v
            u = u + dt * (D_u * lap_u - uvv + f * (1 - u))
            v = v + dt * (D_v * lap_v + uvv - (f + k) * v)
        return u, v


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

    # Standard Gray-Scott parameters (Karl Sims / web convention)
    # Unscaled discrete Laplacian: D controls mixing rate per timestep
    # D_u = 0.16, D_v = 0.08 are standard values
    # CFL: dt < 1/(4*D_u) = 1.5625 for stability, so dt=1.0 is fine
    # Patterns develop in ~5000-20000 timesteps
    D_u, D_v = 0.16, 0.08
    dx = 1.0
    domain_size = float(grid_size)
    dt = 1.0  # Stable: dt < 1/(4*D_u) = 1.5625

    # Scan (f, k) parameter space -- Pearson's known interesting region
    # Key regimes: spots (~0.035, 0.065), stripes (~0.04, 0.06),
    #              waves (~0.014, 0.045), mitosis (~0.028, 0.062)
    f_values = np.linspace(0.01, 0.06, 11)
    k_values = np.linspace(0.04, 0.07, 11)

    phase_diagram = []
    wavelength_data = []

    total = len(f_values) * len(k_values)
    count = 0

    # Generate initial condition once: u=1, v=0 with random square seed
    rng = np.random.default_rng(42)
    u_init = np.ones((grid_size, grid_size), dtype=np.float64)
    v_init = np.zeros((grid_size, grid_size), dtype=np.float64)
    cx, cy = grid_size // 2, grid_size // 2
    r = max(grid_size // 10, 2)
    u_init[cx - r:cx + r, cy - r:cy + r] = 0.50
    v_init[cx - r:cx + r, cy - r:cy + r] = 0.25
    # Small random perturbation across whole domain to break symmetry
    u_init += 0.05 * rng.standard_normal(u_init.shape)
    v_init += 0.05 * rng.standard_normal(v_init.shape)
    v_init = np.clip(v_init, 0, 1)

    for f_val in f_values:
        for k_val in k_values:
            count += 1

            # Run using fast JAX lax.scan
            u_final, v_final = _run_gray_scott_jax(
                u_init, v_init, D_u, D_v, float(f_val), float(k_val),
                dt, dx, n_steps
            )

            v_field = v_final
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

    # --- Part 2: Wavelength vs D_v scaling ---
    # Fix (f, k) in spots regime and vary D_v to measure wavelength dependence
    logger.info("Running D_v variation for wavelength scaling...")
    dv_wavelength_data = []
    f_fixed, k_fixed = 0.035, 0.065  # Standard spots regime
    D_v_values = np.linspace(0.03, 0.12, 15)
    for dv_val in D_v_values:
        u_f, v_f = _run_gray_scott_jax(
            u_init, v_init, D_u, float(dv_val), f_fixed, k_fixed,
            dt, dx, n_steps
        )
        wl = compute_dominant_wavelength(v_f, domain_size)
        energy = compute_pattern_energy(v_f)
        ptype = classify_pattern(v_f)
        if ptype != "uniform" and wl > 0 and wl < domain_size * 0.9:
            dv_wavelength_data.append({
                "D_v": float(dv_val),
                "wavelength": wl,
                "energy": energy,
            })
    logger.info(f"  D_v scan: {len(dv_wavelength_data)} patterned points from {len(D_v_values)} D_v values")

    # Analyze wavelength scaling
    scaling_results = {}
    use_data = dv_wavelength_data if dv_wavelength_data else wavelength_data
    if use_data:
        wl_arr = np.array([d["wavelength"] for d in use_data])
        Dv_arr = np.array([d["D_v"] for d in use_data])

        if len(use_data) == len(dv_wavelength_data) and len(dv_wavelength_data) > 2:
            # For D_v variation: test lambda ~ sqrt(D_v)
            predicted_scale = np.sqrt(Dv_arr)
            corr_sqrt_Dv = float(np.corrcoef(wl_arr, predicted_scale)[0, 1])
        else:
            corr_sqrt_Dv = float("nan")

        scaling_results = {
            "n_patterned_points": len(use_data),
            "correlation_lambda_vs_sqrt_Dv": corr_sqrt_Dv,
            "mean_wavelength": float(np.mean(wl_arr)),
            "std_wavelength": float(np.std(wl_arr)),
            "dv_wavelength_pairs": [
                {"D_v": d["D_v"], "wavelength": d["wavelength"]}
                for d in dv_wavelength_data
            ],
        }

        if len(use_data) >= 5:
            try:
                from simulating_anything.analysis.symbolic_regression import (
                    run_symbolic_regression,
                )

                X_wl = Dv_arr.reshape(-1, 1)
                discoveries = run_symbolic_regression(
                    X_wl,
                    wl_arr,
                    variable_names=["Dv"],
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
        corr = scaling_results.get("correlation_lambda_vs_sqrt_Dv", None)
        if corr is not None and not (isinstance(corr, float) and corr != corr):
            logger.info(f"Wavelength correlation with sqrt(D_v): {corr:.4f}")
        logger.info(f"Patterned points: {scaling_results.get('n_patterned_points', 0)}")

    return results
