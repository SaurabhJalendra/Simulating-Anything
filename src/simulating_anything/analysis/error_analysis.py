"""Error analysis and statistical significance for rediscovery results.

Provides bootstrap resampling to compute confidence intervals on R-squared
values and regression coefficients. Also provides multi-trial analysis for
PySR and SINDy rediscoveries to quantify reproducibility.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of a bootstrap resampling analysis."""

    point_estimate: float
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    n_samples: int
    significant_figures: int = 0

    def __post_init__(self) -> None:
        self.significant_figures = _count_significant_figures(
            self.point_estimate, self.ci_lower, self.ci_upper
        )


@dataclass
class CoefficientResult:
    """Bootstrap uncertainty estimate for a single regression coefficient."""

    name: str
    point_estimate: float
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    significant_figures: int = 0


@dataclass
class DomainErrorAnalysis:
    """Aggregated error analysis for one domain."""

    domain: str
    n_trials: int
    r_squared: BootstrapResult | None = None
    r_squared_trials: list[float] = field(default_factory=list)
    r_squared_trial_mean: float = 0.0
    r_squared_trial_std: float = 0.0
    coefficients: list[CoefficientResult] = field(default_factory=list)
    notes: str = ""


def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        R-squared value. Can be negative for poor fits.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _count_significant_figures(
    value: float, ci_lower: float, ci_upper: float
) -> int:
    """Estimate reliable significant figures from a confidence interval.

    The number of reliable digits is determined by the width of the CI
    relative to the point estimate. If the CI spans 0.001 on a value
    of 1.0, that's about 3 significant figures.

    Args:
        value: Point estimate.
        ci_lower: Lower CI bound.
        ci_upper: Upper CI bound.

    Returns:
        Number of reliable significant figures (minimum 1).
    """
    ci_width = ci_upper - ci_lower
    if ci_width <= 0.0 or value == 0.0:
        # Perfect agreement or zero value -- report high precision
        return 15 if ci_width == 0.0 else 1

    abs_value = abs(value)
    if abs_value < 1e-15:
        return 1

    # Relative uncertainty determines significant figures
    relative_uncertainty = ci_width / abs_value
    if relative_uncertainty <= 0.0:
        return 15

    sig_figs = max(1, int(-math.log10(relative_uncertainty)))
    return sig_figs


def bootstrap_r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap resampling to compute confidence interval on R-squared.

    Resamples (y_true, y_pred) pairs with replacement and computes R-squared
    on each bootstrap sample. Returns point estimate, mean, std, and CI bounds.

    Args:
        y_true: Ground truth values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence interval level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with point estimate, mean, std, and CI bounds.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, "
            f"got {len(y_true)} and {len(y_pred)}"
        )

    n = len(y_true)
    point_estimate = _compute_r_squared(y_true, y_pred)

    rng = np.random.default_rng(seed)
    r2_samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r2_samples[i] = _compute_r_squared(y_true[idx], y_pred[idx])

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(r2_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(r2_samples, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        point_estimate=float(point_estimate),
        mean=float(np.mean(r2_samples)),
        std=float(np.std(r2_samples)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci,
        n_bootstrap=n_bootstrap,
        n_samples=n,
    )


def coefficient_uncertainty(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
    feature_names: list[str] | None = None,
) -> list[CoefficientResult]:
    """Bootstrap uncertainty estimation for OLS regression coefficients.

    Fits a linear regression on each bootstrap resample and reports the
    distribution of each coefficient.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Target vector, shape (n_samples,).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence interval level.
        seed: Random seed for reproducibility.
        feature_names: Names for each feature column.

    Returns:
        List of CoefficientResult, one per feature.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    if len(y) != n_samples:
        raise ValueError(
            f"X has {n_samples} samples but y has {len(y)}"
        )

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    # Point estimates via OLS: beta = (X^T X)^{-1} X^T y
    point_coefs = np.linalg.lstsq(X, y, rcond=None)[0]

    rng = np.random.default_rng(seed)
    coef_samples = np.empty((n_bootstrap, n_features))

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        X_boot = X[idx]
        y_boot = y[idx]
        try:
            coef_samples[i] = np.linalg.lstsq(X_boot, y_boot, rcond=None)[0]
        except np.linalg.LinAlgError:
            coef_samples[i] = np.nan

    alpha = 1.0 - ci
    results = []
    for j in range(n_features):
        valid = coef_samples[:, j][~np.isnan(coef_samples[:, j])]
        if len(valid) == 0:
            ci_lower = ci_upper = float("nan")
            mean_val = std_val = float("nan")
        else:
            ci_lower = float(np.percentile(valid, 100 * alpha / 2))
            ci_upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))
            mean_val = float(np.mean(valid))
            std_val = float(np.std(valid))

        sig_figs = _count_significant_figures(
            float(point_coefs[j]), ci_lower, ci_upper
        )

        results.append(
            CoefficientResult(
                name=feature_names[j],
                point_estimate=float(point_coefs[j]),
                mean=mean_val,
                std=std_val,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                significant_figures=sig_figs,
            )
        )

    return results


def format_ci(value: float, lower: float, upper: float, decimals: int = 4) -> str:
    """Format a value with its confidence interval as a readable string.

    Args:
        value: Point estimate.
        lower: Lower CI bound.
        upper: Upper CI bound.
        decimals: Number of decimal places.

    Returns:
        String formatted as "value (lower, upper)".
    """
    fmt = f".{decimals}f"
    return f"{value:{fmt}} ({lower:{fmt}}, {upper:{fmt}})"


def run_projectile_error_analysis(
    n_trials: int = 10,
    n_speeds: int = 10,
    n_angles: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> DomainErrorAnalysis:
    """Run error analysis for projectile range equation rediscovery.

    Generates projectile data and runs PySR multiple times with different
    random seeds to measure R-squared reproducibility. Also computes
    bootstrap CIs on the R-squared of the theoretical fit.

    Args:
        n_trials: Number of PySR trials with different seeds.
        n_speeds: Number of speeds in data generation grid.
        n_angles: Number of angles in data generation grid.
        n_bootstrap: Number of bootstrap resamples for CI.
        seed: Base random seed.

    Returns:
        DomainErrorAnalysis with R-squared distribution and CIs.
    """
    from simulating_anything.rediscovery.projectile import (
        generate_projectile_data,
        theoretical_range,
    )

    logger.info("Generating projectile data for error analysis...")
    data = generate_projectile_data(
        n_speeds=n_speeds, n_angles=n_angles, drag_coefficient=0.0, dt=0.001
    )
    y_true = data["range"]
    y_theory = theoretical_range(data["v0"], data["theta"], data["g"])

    # Bootstrap CI on R-squared of simulation vs theory
    r2_bootstrap = bootstrap_r_squared(
        y_true, y_theory, n_bootstrap=n_bootstrap, seed=seed
    )
    ci_str = format_ci(
        r2_bootstrap.point_estimate, r2_bootstrap.ci_lower, r2_bootstrap.ci_upper
    )
    logger.info(f"Projectile R2 vs theory: {ci_str}")

    # Coefficient uncertainty: fit R = a * v0^2 * sin(2*theta)
    # where a should be 1/g
    X_coef = (data["v0"] ** 2 * np.sin(2 * data["theta"])).reshape(-1, 1)
    coef_results = coefficient_uncertainty(
        X_coef, y_true, n_bootstrap=n_bootstrap, seed=seed,
        feature_names=["1/g"],
    )

    # Multi-trial PySR (optional -- requires PySR)
    r2_trials: list[float] = []
    try:
        from simulating_anything.analysis.symbolic_regression import (
            run_symbolic_regression,
        )

        X_pysr = np.column_stack([data["v0"], data["theta"], data["g"]])
        for trial in range(n_trials):
            logger.info(f"  PySR trial {trial + 1}/{n_trials}...")
            discoveries = run_symbolic_regression(
                X_pysr, y_true,
                variable_names=["v0", "theta", "g"],
                n_iterations=10,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos", "square"],
                max_complexity=20,
                populations=10,
                population_size=20,
            )
            if discoveries:
                r2_trials.append(discoveries[0].evidence.fit_r_squared)
            else:
                r2_trials.append(0.0)
    except ImportError:
        logger.info("PySR not available -- skipping multi-trial analysis")

    result = DomainErrorAnalysis(
        domain="projectile",
        n_trials=len(r2_trials),
        r_squared=r2_bootstrap,
        r_squared_trials=r2_trials,
        coefficients=coef_results,
    )

    if r2_trials:
        result.r_squared_trial_mean = float(np.mean(r2_trials))
        result.r_squared_trial_std = float(np.std(r2_trials))
        result.notes = (
            f"PySR R2 over {len(r2_trials)} trials: "
            f"{result.r_squared_trial_mean:.6f} +/- {result.r_squared_trial_std:.6f}"
        )
    else:
        result.notes = "PySR not available; bootstrap CI computed on theory fit only"

    return result


def run_lotka_volterra_error_analysis(
    n_trials: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> DomainErrorAnalysis:
    """Run error analysis for Lotka-Volterra ODE rediscovery via SINDy.

    Generates LV trajectory data and runs SINDy multiple times with
    different noise perturbations to measure coefficient reproducibility.
    Also computes bootstrap CIs on the equilibrium fit.

    Args:
        n_trials: Number of SINDy trials with noise perturbations.
        n_bootstrap: Number of bootstrap resamples for CI.
        seed: Base random seed.

    Returns:
        DomainErrorAnalysis with coefficient distributions and CIs.
    """
    from simulating_anything.rediscovery.lotka_volterra import (
        generate_equilibrium_data,
        generate_ode_data,
    )

    logger.info("Generating Lotka-Volterra data for error analysis...")
    eq_data = generate_equilibrium_data(n_samples=50, n_steps=5000, dt=0.01)

    # Bootstrap CI on equilibrium R-squared: prey_avg vs gamma/delta
    prey_theory = eq_data["gamma"] / eq_data["delta"]
    r2_bootstrap = bootstrap_r_squared(
        eq_data["prey_avg"], prey_theory, n_bootstrap=n_bootstrap, seed=seed
    )
    ci_str = format_ci(
        r2_bootstrap.point_estimate, r2_bootstrap.ci_lower, r2_bootstrap.ci_upper
    )
    logger.info(f"LV equilibrium R2: {ci_str}")

    # Multi-trial SINDy with noise perturbation
    r2_trials: list[float] = []
    try:
        from simulating_anything.analysis.equation_discovery import run_sindy

        rng = np.random.default_rng(seed)
        ode_data = generate_ode_data(n_steps=2000, dt=0.01)
        states_clean = ode_data["states"]

        for trial in range(n_trials):
            # Add small noise to simulate measurement uncertainty
            noise_scale = 0.001 * np.std(states_clean, axis=0)
            noise = rng.normal(0, 1, states_clean.shape) * noise_scale
            states_noisy = states_clean + noise

            discoveries = run_sindy(
                states_noisy,
                dt=ode_data["dt"],
                feature_names=["prey", "pred"],
                threshold=0.05,
                poly_degree=2,
            )
            if discoveries:
                trial_r2 = np.mean([d.evidence.fit_r_squared for d in discoveries])
                r2_trials.append(trial_r2)
            else:
                r2_trials.append(0.0)
    except ImportError:
        logger.info("PySINDy not available -- skipping multi-trial analysis")

    result = DomainErrorAnalysis(
        domain="lotka_volterra",
        n_trials=len(r2_trials),
        r_squared=r2_bootstrap,
        r_squared_trials=r2_trials,
    )

    if r2_trials:
        result.r_squared_trial_mean = float(np.mean(r2_trials))
        result.r_squared_trial_std = float(np.std(r2_trials))
        result.notes = (
            f"SINDy R2 over {len(r2_trials)} trials: "
            f"{result.r_squared_trial_mean:.6f} +/- {result.r_squared_trial_std:.6f}"
        )
    else:
        result.notes = (
            "PySINDy not available; bootstrap CI computed on equilibrium fit only"
        )

    return result


def run_error_analysis_all(
    n_trials: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, DomainErrorAnalysis]:
    """Run error analysis across key domains and aggregate results.

    Runs projectile and Lotka-Volterra error analyses. Additional domains
    can be added here as they implement error analysis functions.

    Args:
        n_trials: Number of trials per domain.
        n_bootstrap: Number of bootstrap resamples.
        seed: Base random seed.

    Returns:
        Dict mapping domain name to DomainErrorAnalysis.
    """
    results: dict[str, DomainErrorAnalysis] = {}

    logger.info("Running projectile error analysis...")
    try:
        results["projectile"] = run_projectile_error_analysis(
            n_trials=n_trials, n_bootstrap=n_bootstrap, seed=seed,
            n_speeds=10, n_angles=10,
        )
    except Exception as e:
        logger.error(f"Projectile error analysis failed: {e}")

    logger.info("Running Lotka-Volterra error analysis...")
    try:
        results["lotka_volterra"] = run_lotka_volterra_error_analysis(
            n_trials=n_trials, n_bootstrap=n_bootstrap, seed=seed,
        )
    except Exception as e:
        logger.error(f"Lotka-Volterra error analysis failed: {e}")

    # Log summary
    for domain, analysis in results.items():
        if analysis.r_squared is not None:
            r2 = analysis.r_squared
            logger.info(
                f"{domain}: R2 = {format_ci(r2.point_estimate, r2.ci_lower, r2.ci_upper)} "
                f"({r2.significant_figures} sig figs)"
            )
        if analysis.r_squared_trials:
            logger.info(
                f"  Multi-trial: {analysis.r_squared_trial_mean:.6f} "
                f"+/- {analysis.r_squared_trial_std:.6f} "
                f"(n={analysis.n_trials})"
            )

    return results
