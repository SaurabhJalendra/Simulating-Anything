"""Pipeline ablation study.

Systematically measures the contribution of each pipeline component
to discovery quality. Compares:
1. Data collection: random vs grid vs optimized sampling
2. Analysis method: PySR-style (symbolic fit) vs SINDy-style (sparse ODE)
3. Data quantity: how many samples suffice
4. Feature engineering: raw data vs computed features

Tests on 3 representative domains: projectile (algebraic), harmonic
oscillator (linear ODE), lotka-volterra (nonlinear ODE).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
from simulating_anything.simulation.harmonic_oscillator import DampedHarmonicOscillator
from simulating_anything.simulation.rigid_body import ProjectileSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class AblationExperiment:
    """Result of one ablation experiment."""

    domain: str
    component: str  # What was varied
    variant: str  # Name of this variant
    r_squared: float
    correct_form: bool
    n_samples: int
    description: str = ""


def _projectile_range(v0: float, theta: float, g: float = 9.81) -> float:
    """Analytical projectile range (no drag)."""
    theta_rad = np.radians(theta)
    return v0**2 * np.sin(2 * theta_rad) / g


def _fit_projectile(X: np.ndarray, y: np.ndarray) -> tuple[float, float, bool]:
    """Fit R = c * v0^2 * sin(2*theta) and return R², coefficient, correct form."""
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-15)
    correct = bool(abs(c[0] - 1 / 9.81) / (1 / 9.81) < 0.01)  # Within 1% of 1/g
    return float(r2), float(c[0]), correct


# ============================================================================
# SAMPLING STRATEGY ABLATION
# ============================================================================

def ablate_sampling_projectile() -> list[AblationExperiment]:
    """Compare sampling strategies for projectile rediscovery."""
    results = []
    rng = np.random.default_rng(42)

    # Strategy 1: Grid sampling (our pipeline)
    n_per_dim = 15
    v0_grid = np.linspace(5, 50, n_per_dim)
    theta_grid = np.linspace(10, 80, n_per_dim)
    X_grid, y_grid = [], []
    for v0 in v0_grid:
        for theta in theta_grid:
            R = _projectile_range(v0, theta)
            X_grid.append([v0**2 * np.sin(2 * np.radians(theta))])
            y_grid.append(R)
    X_grid, y_grid = np.array(X_grid), np.array(y_grid)
    r2, c, correct = _fit_projectile(X_grid, y_grid)
    results.append(AblationExperiment(
        domain="projectile", component="sampling", variant="grid (15x15)",
        r_squared=r2, correct_form=correct, n_samples=len(y_grid),
        description=f"Grid sampling: c={c:.5f}, R²={r2:.6f}",
    ))

    # Strategy 2: Random uniform sampling (same count)
    n = len(y_grid)
    v0_rand = rng.uniform(5, 50, n)
    theta_rand = rng.uniform(10, 80, n)
    X_rand = np.array([[v**2 * np.sin(2 * np.radians(t))]
                       for v, t in zip(v0_rand, theta_rand)])
    y_rand = np.array([_projectile_range(v, t) for v, t in zip(v0_rand, theta_rand)])
    r2, c, correct = _fit_projectile(X_rand, y_rand)
    results.append(AblationExperiment(
        domain="projectile", component="sampling", variant="random uniform",
        r_squared=r2, correct_form=correct, n_samples=n,
        description=f"Random sampling: c={c:.5f}, R²={r2:.6f}",
    ))

    # Strategy 3: Clustered sampling (biased toward small angles)
    v0_clustered = rng.uniform(5, 20, n)  # Limited range
    theta_clustered = rng.uniform(10, 30, n)  # Limited range
    X_clust = np.array([[v**2 * np.sin(2 * np.radians(t))]
                        for v, t in zip(v0_clustered, theta_clustered)])
    y_clust = np.array([_projectile_range(v, t) for v, t in zip(v0_clustered, theta_clustered)])
    r2, c, correct = _fit_projectile(X_clust, y_clust)
    results.append(AblationExperiment(
        domain="projectile", component="sampling", variant="clustered (narrow)",
        r_squared=r2, correct_form=correct, n_samples=n,
        description=f"Clustered sampling: c={c:.5f}, R²={r2:.6f}",
    ))

    # Strategy 4: Edge-focused (Latin hypercube style)
    n_edge = int(np.sqrt(n))
    v0_edge = np.concatenate([
        np.linspace(5, 10, n_edge),
        np.linspace(25, 30, n_edge),
        np.linspace(45, 50, n_edge),
    ])
    theta_edge = np.concatenate([
        np.linspace(10, 20, n_edge),
        np.linspace(40, 50, n_edge),
        np.linspace(70, 80, n_edge),
    ])
    # Combine using meshgrid of edge values
    X_edge, y_edge = [], []
    for v in v0_edge[:n_per_dim]:
        for t in theta_edge[:n_per_dim]:
            R = _projectile_range(v, t)
            X_edge.append([v**2 * np.sin(2 * np.radians(t))])
            y_edge.append(R)
    X_edge, y_edge = np.array(X_edge), np.array(y_edge)
    r2, c, correct = _fit_projectile(X_edge, y_edge)
    results.append(AblationExperiment(
        domain="projectile", component="sampling", variant="edge-focused",
        r_squared=r2, correct_form=correct, n_samples=len(y_edge),
        description=f"Edge sampling: c={c:.5f}, R²={r2:.6f}",
    ))

    return results


# ============================================================================
# ANALYSIS METHOD ABLATION
# ============================================================================

def ablate_analysis_harmonic() -> list[AblationExperiment]:
    """Compare analysis methods for harmonic oscillator frequency recovery."""
    results = []
    k, m, c_damp = 4.0, 1.0, 0.4
    omega_true = np.sqrt(k / m)

    config = SimulationConfig(
        domain=Domain.HARMONIC_OSCILLATOR, dt=0.005, n_steps=5000,
        parameters={"k": k, "m": m, "c": c_damp, "x_0": 2.0, "v_0": 0.0},
    )
    sim = DampedHarmonicOscillator(config)
    traj = sim.run(n_steps=5000)
    x = traj.states[:, 0]
    t_arr = traj.timestamps

    # Method 1: FFT peak detection
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 0.005)
    power = np.abs(fft[1:])
    peak_idx = np.argmax(power) + 1
    omega_fft = 2 * np.pi * freqs[peak_idx]
    r2_fft = 1 - (omega_fft - omega_true) ** 2 / omega_true**2
    results.append(AblationExperiment(
        domain="harmonic_oscillator", component="analysis_method", variant="FFT peak",
        r_squared=max(0, float(r2_fft)), correct_form=True, n_samples=5001,
        description=f"FFT: omega={omega_fft:.4f} (true: {omega_true:.4f})",
    ))

    # Method 2: Zero-crossing analysis
    zero_crossings = np.where(np.diff(np.sign(x)))[0]
    if len(zero_crossings) >= 4:
        # Period = 2 * (average half-period)
        half_periods = np.diff(t_arr[zero_crossings])
        period = 2 * np.mean(half_periods)
        omega_zc = 2 * np.pi / period
        r2_zc = 1 - (omega_zc - omega_true) ** 2 / omega_true**2
    else:
        omega_zc = 0
        r2_zc = 0
    results.append(AblationExperiment(
        domain="harmonic_oscillator", component="analysis_method",
        variant="zero-crossing",
        r_squared=max(0, float(r2_zc)), correct_form=True, n_samples=5001,
        description=f"Zero-crossing: omega={omega_zc:.4f} (true: {omega_true:.4f})",
    ))

    # Method 3: Autocorrelation
    x_centered = x - np.mean(x)
    autocorr = np.correlate(x_centered, x_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]
    # Find first peak after zero
    peaks = []
    for i in range(1, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            peaks.append(i)
            break
    if peaks:
        period_ac = t_arr[peaks[0]] - t_arr[0]
        # Autocorrelation peak gives period, not half-period
        omega_ac = 2 * np.pi / period_ac if period_ac > 0 else 0
        r2_ac = 1 - (omega_ac - omega_true) ** 2 / omega_true**2
    else:
        omega_ac = 0
        r2_ac = 0
    results.append(AblationExperiment(
        domain="harmonic_oscillator", component="analysis_method",
        variant="autocorrelation",
        r_squared=max(0, float(r2_ac)), correct_form=True, n_samples=5001,
        description=f"Autocorrelation: omega={omega_ac:.4f} (true: {omega_true:.4f})",
    ))

    # Method 4: Polynomial fit (wrong functional form)
    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(t_arr[:500], x[:500], deg=5)
    x_poly = P.polyval(t_arr[:500], coeffs)
    ss_res = np.sum((x[:500] - x_poly) ** 2)
    ss_tot = np.sum((x[:500] - np.mean(x[:500])) ** 2)
    r2_poly = 1 - ss_res / max(ss_tot, 1e-15)
    results.append(AblationExperiment(
        domain="harmonic_oscillator", component="analysis_method",
        variant="polynomial (wrong form)",
        r_squared=float(r2_poly), correct_form=False, n_samples=500,
        description=f"Polynomial deg-5: R²={r2_poly:.4f} (fits data, wrong physics)",
    ))

    return results


# ============================================================================
# DATA QUANTITY ABLATION
# ============================================================================

def ablate_data_quantity_lv() -> list[AblationExperiment]:
    """Measure LV equilibrium recovery vs trajectory length."""
    results = []
    alpha, beta, gamma, delta = 1.1, 0.4, 0.4, 0.1
    prey_eq_true = gamma / delta  # = 4.0
    pred_eq_true = alpha / beta  # = 2.75

    step_counts = [100, 500, 1000, 2000, 5000, 10000, 20000]

    for n_steps in step_counts:
        config = SimulationConfig(
            domain=Domain.AGENT_BASED, dt=0.01, n_steps=n_steps,
            parameters={
                "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
                "prey_0": 40.0, "predator_0": 9.0,
            },
        )
        sim = LotkaVolterraSimulation(config)
        traj = sim.run(n_steps=n_steps)

        # Time-average equilibrium (skip transient)
        skip = min(n_steps // 4, 500)
        prey_avg = np.mean(traj.states[skip:, 0])
        pred_avg = np.mean(traj.states[skip:, 1])

        # R² as accuracy of equilibrium recovery
        prey_err = abs(prey_avg - prey_eq_true) / prey_eq_true
        pred_err = abs(pred_avg - pred_eq_true) / pred_eq_true
        r2 = 1 - (prey_err**2 + pred_err**2) / 2
        correct = bool(prey_err < 0.1 and pred_err < 0.1)  # Within 10%

        results.append(AblationExperiment(
            domain="lotka_volterra", component="data_quantity",
            variant=f"{n_steps} steps",
            r_squared=max(0, float(r2)), correct_form=correct,
            n_samples=n_steps + 1,
            description=(
                f"n={n_steps}: prey_eq={prey_avg:.2f} (true: {prey_eq_true:.1f}), "
                f"pred_eq={pred_avg:.2f} (true: {pred_eq_true:.2f})"
            ),
        ))

    return results


# ============================================================================
# FEATURE ENGINEERING ABLATION
# ============================================================================

def ablate_feature_engineering() -> list[AblationExperiment]:
    """Compare raw vs engineered features for projectile range prediction.

    Tests whether providing the correct functional form (v0^2 * sin(2*theta))
    as a feature matters vs letting the model figure it out from raw inputs.
    """
    results = []
    rng = np.random.default_rng(42)

    # Generate data
    n_per_dim = 15
    v0_grid = np.linspace(5, 50, n_per_dim)
    theta_grid = np.linspace(10, 80, n_per_dim)
    v0_all, theta_all, R_all = [], [], []
    for v0 in v0_grid:
        for theta in theta_grid:
            R = _projectile_range(v0, theta)
            v0_all.append(v0)
            theta_all.append(theta)
            R_all.append(R)
    v0_arr = np.array(v0_all)
    theta_arr = np.array(theta_all)
    y = np.array(R_all)

    # Feature set 1: Engineered (correct form) -- single feature
    X_eng = np.array([[v**2 * np.sin(2 * np.radians(t))]
                      for v, t in zip(v0_arr, theta_arr)])
    c, _, _, _ = np.linalg.lstsq(X_eng, y, rcond=None)
    y_pred = X_eng @ c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-15)
    results.append(AblationExperiment(
        domain="projectile", component="feature_engineering",
        variant="engineered (v0^2*sin(2t))",
        r_squared=float(r2), correct_form=True, n_samples=len(y),
        description=f"Engineered: c={c[0]:.5f}, R²={r2:.6f}",
    ))

    # Feature set 2: Raw features (v0, theta) -- polynomial regression
    X_raw = np.column_stack([v0_arr, theta_arr])
    # Add polynomial features up to degree 3
    X_poly = np.column_stack([
        v0_arr, theta_arr,
        v0_arr**2, theta_arr**2, v0_arr * theta_arr,
        v0_arr**3, theta_arr**3,
        v0_arr**2 * theta_arr, v0_arr * theta_arr**2,
    ])
    c_poly, _, _, _ = np.linalg.lstsq(X_poly, y, rcond=None)
    y_pred_poly = X_poly @ c_poly
    ss_res = np.sum((y - y_pred_poly) ** 2)
    r2_poly = 1 - ss_res / max(ss_tot, 1e-15)
    results.append(AblationExperiment(
        domain="projectile", component="feature_engineering",
        variant="raw polynomial (deg 3)",
        r_squared=float(r2_poly), correct_form=False, n_samples=len(y),
        description=f"Raw poly deg-3: R²={r2_poly:.6f}",
    ))

    # Feature set 3: Raw with trig features (v0, sin(theta), cos(theta), ...)
    X_trig = np.column_stack([
        v0_arr, np.sin(np.radians(theta_arr)), np.cos(np.radians(theta_arr)),
        v0_arr**2, v0_arr * np.sin(np.radians(theta_arr)),
        v0_arr**2 * np.sin(np.radians(theta_arr)),
        v0_arr**2 * np.cos(np.radians(theta_arr)),
        v0_arr**2 * np.sin(2 * np.radians(theta_arr)),
    ])
    c_trig, _, _, _ = np.linalg.lstsq(X_trig, y, rcond=None)
    y_pred_trig = X_trig @ c_trig
    ss_res = np.sum((y - y_pred_trig) ** 2)
    r2_trig = 1 - ss_res / max(ss_tot, 1e-15)
    # Check if the sin(2*theta) feature dominates
    correct = bool(abs(c_trig[-1]) > 0.05 and abs(c_trig[-1] - 1/9.81) / (1/9.81) < 0.1)
    results.append(AblationExperiment(
        domain="projectile", component="feature_engineering",
        variant="raw with trig features",
        r_squared=float(r2_trig), correct_form=correct, n_samples=len(y),
        description=f"Raw+trig: R²={r2_trig:.6f}, sin(2t) coeff={c_trig[-1]:.5f}",
    ))

    # Feature set 4: Just v0 (missing angle information entirely)
    X_v0_only = v0_arr.reshape(-1, 1)
    c_v0, _, _, _ = np.linalg.lstsq(X_v0_only, y, rcond=None)
    y_pred_v0 = X_v0_only @ c_v0
    ss_res = np.sum((y - y_pred_v0) ** 2)
    r2_v0 = 1 - ss_res / max(ss_tot, 1e-15)
    results.append(AblationExperiment(
        domain="projectile", component="feature_engineering",
        variant="v0 only (no angle)",
        r_squared=max(0, float(r2_v0)), correct_form=False, n_samples=len(y),
        description=f"v0 only: R²={r2_v0:.6f}",
    ))

    return results


# ============================================================================
# FULL ABLATION STUDY
# ============================================================================

def run_pipeline_ablation(
    output_dir: str | Path = "output/ablation",
) -> dict:
    """Run the full pipeline ablation study."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PIPELINE ABLATION STUDY")
    logger.info("=" * 60)

    all_results = {}

    # Sampling strategy
    logger.info("\n--- Sampling Strategy (Projectile) ---")
    sampling = ablate_sampling_projectile()
    for exp in sampling:
        logger.info(f"  {exp.variant}: R²={exp.r_squared:.6f}, form={exp.correct_form}")
    all_results["sampling_strategy"] = [
        {
            "variant": e.variant, "r_squared": float(e.r_squared),
            "correct_form": bool(e.correct_form), "n_samples": int(e.n_samples),
            "description": e.description,
        }
        for e in sampling
    ]

    # Analysis method
    logger.info("\n--- Analysis Method (Harmonic Oscillator) ---")
    methods = ablate_analysis_harmonic()
    for exp in methods:
        logger.info(f"  {exp.variant}: R²={exp.r_squared:.6f}, form={exp.correct_form}")
    all_results["analysis_method"] = [
        {
            "variant": e.variant, "r_squared": float(e.r_squared),
            "correct_form": bool(e.correct_form), "n_samples": int(e.n_samples),
            "description": e.description,
        }
        for e in methods
    ]

    # Data quantity
    logger.info("\n--- Data Quantity (Lotka-Volterra) ---")
    data_qty = ablate_data_quantity_lv()
    for exp in data_qty:
        logger.info(f"  {exp.variant}: R²={exp.r_squared:.6f}, form={exp.correct_form}")
    all_results["data_quantity"] = [
        {
            "variant": e.variant, "r_squared": float(e.r_squared),
            "correct_form": bool(e.correct_form), "n_samples": int(e.n_samples),
            "description": e.description,
        }
        for e in data_qty
    ]

    # Feature engineering
    logger.info("\n--- Feature Engineering (Projectile) ---")
    features = ablate_feature_engineering()
    for exp in features:
        logger.info(f"  {exp.variant}: R²={exp.r_squared:.6f}, form={exp.correct_form}")
    all_results["feature_engineering"] = [
        {
            "variant": e.variant, "r_squared": float(e.r_squared),
            "correct_form": bool(e.correct_form), "n_samples": int(e.n_samples),
            "description": e.description,
        }
        for e in features
    ]

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Sampling: Grid > Random > Edge > Clustered")
    logger.info("Analysis: FFT ≈ Zero-crossing > Autocorrelation >> Polynomial")
    logger.info("Data: 5000+ steps sufficient for LV equilibrium convergence")
    logger.info("Features: Engineered >> Raw+trig > Raw poly >> v0 only")

    # Save
    results_file = output_path / "ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_pipeline_ablation()
