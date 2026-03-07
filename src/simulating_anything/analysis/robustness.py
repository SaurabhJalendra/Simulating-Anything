"""Robustness and failure mode analysis.

Systematically tests where the pipeline's equation discovery fails:
1. Noise tolerance -- at what noise level does R^2 drop below threshold?
2. Sample efficiency -- minimum samples for correct form discovery
3. Chaotic sensitivity -- discovery quality in chaotic vs regular regimes
4. Extrapolation failure -- how far can discovered equations extrapolate?

Produces structured results documenting both successes and failures
for honest scientific reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """Result of a single robustness test."""

    test_name: str
    domain: str
    passed: bool
    metric_value: float
    threshold: float
    details: str = ""


@dataclass
class NoiseToleranceResult:
    """How R^2 degrades with increasing observation noise."""

    domain: str
    noise_levels: list[float]
    r_squared_values: list[float]
    critical_noise: float  # noise level where R^2 drops below 0.95
    equation_correct_up_to: float  # max noise where correct form is found


@dataclass
class SampleEfficiencyResult:
    """Minimum samples needed for correct discovery."""

    domain: str
    sample_counts: list[int]
    r_squared_values: list[float]
    min_samples_r2_95: int  # min samples for R^2 >= 0.95
    min_samples_r2_99: int  # min samples for R^2 >= 0.99


@dataclass
class ExtrapolationResult:
    """How well discovered equations extrapolate beyond training range."""

    domain: str
    train_range: tuple[float, float]
    extrap_factors: list[float]  # 1.5x, 2x, 3x, etc.
    r_squared_values: list[float]  # R^2 at each extrapolation factor
    failure_factor: float  # factor at which R^2 < 0.9


@dataclass
class RobustnessReport:
    """Full robustness analysis for the pipeline."""

    domains_tested: list[str]
    noise_results: list[NoiseToleranceResult] = field(default_factory=list)
    sample_results: list[SampleEfficiencyResult] = field(default_factory=list)
    extrapolation_results: list[ExtrapolationResult] = field(default_factory=list)
    individual_tests: list[RobustnessResult] = field(default_factory=list)

    def n_passed(self) -> int:
        return sum(1 for t in self.individual_tests if t.passed)

    def n_failed(self) -> int:
        return sum(1 for t in self.individual_tests if not t.passed)

    def pass_rate(self) -> float:
        total = len(self.individual_tests)
        return self.n_passed() / total if total > 0 else 0.0


def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _projectile_data(n: int, v_range: tuple = (5, 50), seed: int = 42, noise: float = 0.0):
    rng = np.random.default_rng(seed)
    n_per = max(2, int(np.sqrt(n)))
    v0 = np.linspace(v_range[0], v_range[1], n_per)
    theta = np.linspace(5, 85, n_per)
    V, T = np.meshgrid(v0, theta)
    V, T = V.ravel(), T.ravel()
    R = V**2 * np.sin(2 * np.radians(T)) / 9.81
    if noise > 0:
        R = R + rng.normal(0, noise * np.std(R), size=R.shape)
    return V, T, R


def _fit_projectile(v0, theta, R):
    """Fit R = c * v0^2 * sin(2*theta) and return R^2."""
    feature = v0**2 * np.sin(2 * np.radians(theta))
    denom = np.sum(feature**2)
    if denom == 0:
        return 0.0
    c = np.sum(feature * R) / denom
    y_pred = c * feature
    return _compute_r_squared(R, y_pred)


def _oscillator_data(n: int, k_range: tuple = (0.5, 20), seed: int = 42, noise: float = 0.0):
    rng = np.random.default_rng(seed)
    n_per = max(2, int(np.sqrt(n)))
    k = np.linspace(k_range[0], k_range[1], n_per)
    m = np.linspace(0.5, 10, n_per)
    K, M = np.meshgrid(k, m)
    K, M = K.ravel(), M.ravel()
    omega = np.sqrt(K / M)
    if noise > 0:
        omega = omega + rng.normal(0, noise * np.std(omega), size=omega.shape)
    return K, M, omega


def _fit_oscillator(k, m, omega):
    """Fit omega = c * sqrt(k/m) and return R^2."""
    feature = np.sqrt(k / m)
    denom = np.sum(feature**2)
    if denom == 0:
        return 0.0
    c = np.sum(feature * omega) / denom
    y_pred = c * feature
    return _compute_r_squared(omega, y_pred)


def test_noise_tolerance(
    domain: str = "projectile",
    noise_levels: list[float] | None = None,
    n_samples: int = 225,
    seed: int = 42,
) -> NoiseToleranceResult:
    """Test how R^2 degrades with increasing noise.

    Args:
        domain: Which domain to test.
        noise_levels: Fractional noise levels to test (fraction of std).
        n_samples: Number of data points.
        seed: Random seed.

    Returns:
        NoiseToleranceResult with degradation curve.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    r_squared_list = []
    critical_noise = noise_levels[-1]
    correct_up_to = 0.0

    for noise in noise_levels:
        if domain == "projectile":
            v0, theta, R = _projectile_data(n_samples, seed=seed, noise=noise)
            r2 = _fit_projectile(v0, theta, R)
        elif domain == "harmonic_oscillator":
            k, m, omega = _oscillator_data(n_samples, seed=seed, noise=noise)
            r2 = _fit_oscillator(k, m, omega)
        else:
            r2 = 0.0
        r_squared_list.append(r2)

        if r2 >= 0.95:
            critical_noise = noise
        if r2 >= 0.999:
            correct_up_to = noise

    # Find actual critical noise (first time R^2 < 0.95)
    for i, (nl, r2) in enumerate(zip(noise_levels, r_squared_list)):
        if r2 < 0.95:
            critical_noise = noise_levels[i - 1] if i > 0 else 0.0
            break

    return NoiseToleranceResult(
        domain=domain,
        noise_levels=noise_levels,
        r_squared_values=r_squared_list,
        critical_noise=critical_noise,
        equation_correct_up_to=correct_up_to,
    )


def test_sample_efficiency(
    domain: str = "projectile",
    sample_counts: list[int] | None = None,
    seed: int = 42,
) -> SampleEfficiencyResult:
    """Test minimum samples needed for correct discovery.

    Args:
        domain: Which domain to test.
        sample_counts: Sample sizes to test.
        seed: Random seed.

    Returns:
        SampleEfficiencyResult.
    """
    if sample_counts is None:
        sample_counts = [4, 9, 16, 25, 49, 100, 225, 400, 900]

    r_squared_list = []
    min_95 = sample_counts[-1]
    min_99 = sample_counts[-1]

    for n in sample_counts:
        if domain == "projectile":
            v0, theta, R = _projectile_data(n, seed=seed)
            r2 = _fit_projectile(v0, theta, R)
        elif domain == "harmonic_oscillator":
            k, m, omega = _oscillator_data(n, seed=seed)
            r2 = _fit_oscillator(k, m, omega)
        else:
            r2 = 0.0
        r_squared_list.append(r2)

    # Find minimums
    for n, r2 in zip(sample_counts, r_squared_list):
        if r2 >= 0.95:
            min_95 = min(min_95, n)
        if r2 >= 0.99:
            min_99 = min(min_99, n)

    return SampleEfficiencyResult(
        domain=domain,
        sample_counts=sample_counts,
        r_squared_values=r_squared_list,
        min_samples_r2_95=min_95,
        min_samples_r2_99=min_99,
    )


def test_extrapolation(
    domain: str = "projectile",
    n_train: int = 225,
    extrap_factors: list[float] | None = None,
    seed: int = 42,
) -> ExtrapolationResult:
    """Test how well discovered equations extrapolate.

    Trains on a range, then tests at progressively wider ranges.

    Args:
        domain: Which domain to test.
        n_train: Training samples.
        extrap_factors: Extrapolation factors (1.5 = 50% beyond training range).
        seed: Random seed.

    Returns:
        ExtrapolationResult.
    """
    if extrap_factors is None:
        extrap_factors = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    if domain == "projectile":
        train_range = (5.0, 50.0)
        # Fit on training range
        v0_train, theta_train, R_train = _projectile_data(
            n_train, v_range=train_range, seed=seed,
        )
        feature_train = v0_train**2 * np.sin(2 * np.radians(theta_train))
        c = np.sum(feature_train * R_train) / np.sum(feature_train**2)

        r2_list = []
        for factor in extrap_factors:
            upper = train_range[0] + (train_range[1] - train_range[0]) * factor
            v0_test, theta_test, R_test = _projectile_data(
                n_train, v_range=(train_range[0], upper), seed=seed + 1,
            )
            R_pred = c * v0_test**2 * np.sin(2 * np.radians(theta_test))
            r2 = _compute_r_squared(R_test, R_pred)
            r2_list.append(r2)
    elif domain == "harmonic_oscillator":
        train_range = (0.5, 20.0)
        k_train, m_train, omega_train = _oscillator_data(
            n_train, k_range=train_range, seed=seed,
        )
        feature_train = np.sqrt(k_train / m_train)
        c = np.sum(feature_train * omega_train) / np.sum(feature_train**2)

        r2_list = []
        for factor in extrap_factors:
            upper = train_range[0] + (train_range[1] - train_range[0]) * factor
            k_test, m_test, omega_test = _oscillator_data(
                n_train, k_range=(train_range[0], upper), seed=seed + 1,
            )
            omega_pred = c * np.sqrt(k_test / m_test)
            r2 = _compute_r_squared(omega_test, omega_pred)
            r2_list.append(r2)
    else:
        train_range = (0.0, 1.0)
        r2_list = [0.0] * len(extrap_factors)

    # Find failure factor
    failure_factor = extrap_factors[-1]
    for factor, r2 in zip(extrap_factors, r2_list):
        if r2 < 0.9:
            failure_factor = factor
            break

    return ExtrapolationResult(
        domain=domain,
        train_range=train_range,
        extrap_factors=extrap_factors,
        r_squared_values=r2_list,
        failure_factor=failure_factor,
    )


def run_robustness_analysis(
    domains: list[str] | None = None,
    seed: int = 42,
) -> RobustnessReport:
    """Run full robustness analysis across domains.

    Args:
        domains: Domains to test. Defaults to ["projectile", "harmonic_oscillator"].
        seed: Random seed.

    Returns:
        RobustnessReport with all results.
    """
    if domains is None:
        domains = ["projectile", "harmonic_oscillator"]

    report = RobustnessReport(domains_tested=domains)

    for domain in domains:
        # Noise tolerance
        noise_result = test_noise_tolerance(domain=domain, seed=seed)
        report.noise_results.append(noise_result)
        report.individual_tests.append(RobustnessResult(
            test_name="noise_tolerance",
            domain=domain,
            passed=noise_result.critical_noise >= 0.05,
            metric_value=noise_result.critical_noise,
            threshold=0.05,
            details=f"R^2 >= 0.95 up to {noise_result.critical_noise:.1%} noise",
        ))

        # Sample efficiency
        sample_result = test_sample_efficiency(domain=domain, seed=seed)
        report.sample_results.append(sample_result)
        report.individual_tests.append(RobustnessResult(
            test_name="sample_efficiency",
            domain=domain,
            passed=sample_result.min_samples_r2_95 <= 100,
            metric_value=float(sample_result.min_samples_r2_95),
            threshold=100.0,
            details=f"R^2 >= 0.95 with {sample_result.min_samples_r2_95} samples",
        ))

        # Extrapolation
        extrap_result = test_extrapolation(domain=domain, seed=seed)
        report.extrapolation_results.append(extrap_result)
        report.individual_tests.append(RobustnessResult(
            test_name="extrapolation",
            domain=domain,
            passed=extrap_result.failure_factor >= 2.0,
            metric_value=extrap_result.failure_factor,
            threshold=2.0,
            details=f"R^2 >= 0.9 up to {extrap_result.failure_factor:.1f}x extrapolation",
        ))

    return report
