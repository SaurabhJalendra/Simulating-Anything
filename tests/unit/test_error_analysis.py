"""Tests for the error analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.analysis.error_analysis import (
    BootstrapResult,
    CoefficientResult,
    _compute_r_squared,
    _count_significant_figures,
    bootstrap_r_squared,
    coefficient_uncertainty,
    format_ci,
)


class TestBootstrapRSquared:
    """Tests for bootstrap_r_squared function."""

    def test_perfect_fit(self):
        """R-squared should be 1.0 when predictions exactly match truth."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bootstrap_r_squared(y_true, y_pred, n_bootstrap=500, seed=42)

        assert isinstance(result, BootstrapResult)
        assert result.point_estimate == pytest.approx(1.0)
        assert result.mean == pytest.approx(1.0)
        assert result.ci_lower == pytest.approx(1.0)
        assert result.ci_upper == pytest.approx(1.0)

    def test_noisy_fit(self):
        """Noisy data should give R-squared < 1.0 with a non-degenerate CI."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        y_true = 2.0 * x + 1.0
        noise = rng.normal(0, 1.0, size=100)
        y_pred = y_true + noise

        result = bootstrap_r_squared(y_true, y_pred, n_bootstrap=1000, seed=42)

        assert result.point_estimate < 1.0
        assert result.point_estimate > 0.5
        assert result.ci_lower < result.point_estimate
        assert result.ci_upper > result.point_estimate
        assert result.std > 0.0

    def test_ci_width_increases_with_noise(self):
        """Noisier data should produce wider confidence intervals."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        y_true = 3.0 * x

        # Low noise
        y_pred_low = y_true + rng.normal(0, 0.5, size=100)
        result_low = bootstrap_r_squared(
            y_true, y_pred_low, n_bootstrap=1000, seed=42
        )

        # High noise
        y_pred_high = y_true + rng.normal(0, 5.0, size=100)
        result_high = bootstrap_r_squared(
            y_true, y_pred_high, n_bootstrap=1000, seed=42
        )

        ci_width_low = result_low.ci_upper - result_low.ci_lower
        ci_width_high = result_high.ci_upper - result_high.ci_lower
        assert ci_width_high > ci_width_low, (
            f"High noise CI width {ci_width_high:.6f} should exceed "
            f"low noise CI width {ci_width_low:.6f}"
        )

    def test_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        rng = np.random.default_rng(123)
        y_true = rng.normal(0, 1, 50)
        y_pred = y_true + rng.normal(0, 0.1, 50)

        result1 = bootstrap_r_squared(y_true, y_pred, n_bootstrap=500, seed=99)
        result2 = bootstrap_r_squared(y_true, y_pred, n_bootstrap=500, seed=99)

        assert result1.mean == result2.mean
        assert result1.std == result2.std
        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper

    def test_larger_sample_tighter_ci(self):
        """Larger samples should generally produce tighter CIs."""
        rng = np.random.default_rng(42)

        # Small sample
        x_small = np.linspace(0, 10, 20)
        y_true_small = 2.0 * x_small
        y_pred_small = y_true_small + rng.normal(0, 1.0, size=20)
        result_small = bootstrap_r_squared(
            y_true_small, y_pred_small, n_bootstrap=2000, seed=42
        )

        # Large sample with the same noise scale
        x_large = np.linspace(0, 10, 500)
        y_true_large = 2.0 * x_large
        y_pred_large = y_true_large + rng.normal(0, 1.0, size=500)
        result_large = bootstrap_r_squared(
            y_true_large, y_pred_large, n_bootstrap=2000, seed=42
        )

        ci_width_small = result_small.ci_upper - result_small.ci_lower
        ci_width_large = result_large.ci_upper - result_large.ci_lower
        assert ci_width_large < ci_width_small, (
            f"Large sample CI width {ci_width_large:.6f} should be smaller than "
            f"small sample CI width {ci_width_small:.6f}"
        )

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            bootstrap_r_squared(np.array([1, 2, 3]), np.array([1, 2]))

    def test_result_fields(self):
        """BootstrapResult should have all expected fields populated."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bootstrap_r_squared(y, y, n_bootstrap=100, ci=0.90, seed=0)

        assert result.n_bootstrap == 100
        assert result.n_samples == 5
        assert result.ci_level == 0.90
        assert result.significant_figures >= 1


class TestCoefficientUncertainty:
    """Tests for coefficient_uncertainty function."""

    def test_linear_coefficient(self):
        """Should recover the slope of a simple linear relationship."""
        rng = np.random.default_rng(42)
        x = np.linspace(1, 10, 200).reshape(-1, 1)
        y = 3.0 * x.ravel() + rng.normal(0, 0.1, 200)

        results = coefficient_uncertainty(
            x, y, n_bootstrap=500, seed=42, feature_names=["slope"]
        )

        assert len(results) == 1
        coef = results[0]
        assert isinstance(coef, CoefficientResult)
        assert coef.name == "slope"
        assert coef.point_estimate == pytest.approx(3.0, abs=0.1)
        assert coef.ci_lower < 3.0
        assert coef.ci_upper > 3.0

    def test_multiple_features(self):
        """Should return one CoefficientResult per feature."""
        rng = np.random.default_rng(42)
        X = rng.uniform(1, 10, (100, 3))
        y = 2.0 * X[:, 0] + 0.5 * X[:, 1] - 1.0 * X[:, 2]

        results = coefficient_uncertainty(
            X, y, n_bootstrap=500, seed=42,
            feature_names=["a", "b", "c"],
        )

        assert len(results) == 3
        assert results[0].name == "a"
        assert results[0].point_estimate == pytest.approx(2.0, abs=0.1)
        assert results[1].name == "b"
        assert results[1].point_estimate == pytest.approx(0.5, abs=0.1)
        assert results[2].name == "c"
        assert results[2].point_estimate == pytest.approx(-1.0, abs=0.1)

    def test_mismatched_lengths_raises(self):
        """Mismatched X rows and y length should raise ValueError."""
        with pytest.raises(ValueError, match="samples"):
            coefficient_uncertainty(np.ones((10, 2)), np.ones(5))


class TestProjectileErrorAnalysis:
    """Tests for run_projectile_error_analysis."""

    def test_runs_with_small_grid(self):
        """Should complete quickly with minimal data and no PySR."""
        from simulating_anything.analysis.error_analysis import (
            run_projectile_error_analysis,
        )

        result = run_projectile_error_analysis(
            n_trials=2, n_speeds=3, n_angles=3, n_bootstrap=50, seed=42
        )

        assert result.domain == "projectile"
        assert result.r_squared is not None
        # Projectile simulation matches theory very well
        assert result.r_squared.point_estimate > 0.99
        assert len(result.coefficients) == 1
        # 1/g coefficient should be close to 1/9.81 ~ 0.1019
        coef = result.coefficients[0]
        assert coef.point_estimate == pytest.approx(1.0 / 9.81, rel=0.01)


class TestFormatCI:
    """Tests for format_ci function."""

    def test_basic_format(self):
        result = format_ci(0.9999, 0.9990, 1.0000)
        assert result == "0.9999 (0.9990, 1.0000)"

    def test_custom_decimals(self):
        result = format_ci(0.5, 0.4, 0.6, decimals=2)
        assert result == "0.50 (0.40, 0.60)"

    def test_negative_values(self):
        result = format_ci(-1.5, -2.0, -1.0, decimals=1)
        assert result == "-1.5 (-2.0, -1.0)"


class TestHelpers:
    """Tests for internal helper functions."""

    def test_compute_r_squared_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert _compute_r_squared(y, y) == pytest.approx(1.0)

    def test_compute_r_squared_constant(self):
        """Constant y_true with zero residuals should return 1.0."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        assert _compute_r_squared(y_true, y_pred) == pytest.approx(1.0)

    def test_significant_figures_high_precision(self):
        """Tight CI should give many significant figures."""
        sig = _count_significant_figures(1.0, 0.9999, 1.0001)
        assert sig >= 3

    def test_significant_figures_low_precision(self):
        """Wide CI should give fewer significant figures."""
        sig = _count_significant_figures(1.0, 0.5, 1.5)
        assert sig == 1
