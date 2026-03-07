"""Tests for statistical significance module."""
from __future__ import annotations

import math

import numpy as np
import pytest

from simulating_anything.analysis.significance import (
    EffectSize,
    PermutationTestResult,
    SignificanceReport,
    WilcoxonResult,
    bootstrap_r_squared,
    cohens_d,
    permutation_test_r_squared,
    run_significance_analysis,
    wilcoxon_signed_rank,
    _normal_cdf,
)


class TestNormalCDF:
    """Test normal CDF approximation."""

    def test_zero(self):
        assert abs(_normal_cdf(0) - 0.5) < 1e-10

    def test_large_positive(self):
        assert _normal_cdf(5) > 0.999

    def test_large_negative(self):
        assert _normal_cdf(-5) < 0.001

    def test_symmetry(self):
        assert abs(_normal_cdf(1) + _normal_cdf(-1) - 1.0) < 1e-10

    def test_known_values(self):
        # CDF(1.96) ~ 0.975
        assert abs(_normal_cdf(1.96) - 0.975) < 0.001


class TestPermutationTest:
    """Test permutation test for R^2."""

    def test_perfect_fit_significant(self):
        x = np.linspace(0, 10, 100)
        y_true = 2 * x + 1
        y_pred = 2 * x + 1
        result = permutation_test_r_squared(
            y_true, y_pred, n_permutations=1000,
        )
        assert result.significant
        assert result.p_value < 0.01
        assert result.observed_statistic > 0.99

    def test_random_not_significant(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 50)
        y_pred = rng.normal(0, 1, 50)
        result = permutation_test_r_squared(
            y_true, y_pred, n_permutations=1000,
        )
        assert result.p_value > 0.05

    def test_moderate_correlation(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        y_true = 2 * x + rng.normal(0, 2, 100)
        y_pred = 2 * x
        result = permutation_test_r_squared(
            y_true, y_pred, n_permutations=1000,
        )
        assert result.significant
        assert result.p_value < 0.05

    def test_result_fields(self):
        y_true = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        result = permutation_test_r_squared(
            y_true, y_pred, n_permutations=500,
        )
        assert isinstance(result, PermutationTestResult)
        assert result.n_permutations == 500
        assert 0 <= result.p_value <= 1


class TestBootstrap:
    """Test bootstrap confidence intervals."""

    def test_perfect_fit_narrow_ci(self):
        x = np.linspace(0, 10, 100)
        y_true = 2 * x + 1
        y_pred = 2 * x + 1
        point, ci_low, ci_high = bootstrap_r_squared(
            y_true, y_pred, n_bootstrap=1000,
        )
        assert point > 0.999
        assert ci_low > 0.99
        assert ci_high >= ci_low

    def test_noisy_fit_wider_ci(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y_true = 2 * x + rng.normal(0, 3, 50)
        y_pred = 2 * x
        point, ci_low, ci_high = bootstrap_r_squared(
            y_true, y_pred, n_bootstrap=1000,
        )
        assert ci_high - ci_low > 0.01  # CI should be wider than perfect

    def test_ci_contains_point(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 100)
        y_pred = y_true + rng.normal(0, 0.5, 100)
        point, ci_low, ci_high = bootstrap_r_squared(
            y_true, y_pred, n_bootstrap=2000,
        )
        # Point estimate should usually be within CI
        assert ci_low <= point <= ci_high


class TestWilcoxon:
    """Test Wilcoxon signed-rank test."""

    def test_identical_scores(self):
        scores = np.array([0.9, 0.95, 0.92, 0.88, 0.93])
        result = wilcoxon_signed_rank(scores, scores)
        assert not result.significant
        assert result.ties == 5

    def test_clearly_different(self):
        scores_a = np.array([0.95, 0.97, 0.96, 0.94, 0.98, 0.93, 0.99, 0.92, 0.96, 0.95])
        scores_b = np.array([0.70, 0.72, 0.68, 0.71, 0.73, 0.69, 0.74, 0.67, 0.72, 0.70])
        result = wilcoxon_signed_rank(scores_a, scores_b)
        assert result.significant
        assert result.method_a_wins == 10

    def test_result_fields(self):
        scores_a = np.array([0.9, 0.8, 0.85, 0.92, 0.88])
        scores_b = np.array([0.85, 0.82, 0.80, 0.90, 0.86])
        result = wilcoxon_signed_rank(scores_a, scores_b)
        assert isinstance(result, WilcoxonResult)
        assert result.n_pairs == 5
        assert 0 <= result.p_value <= 1
        assert result.method_a_wins + result.method_b_wins + result.ties == 5

    def test_empty_after_ties(self):
        scores = np.array([1.0, 1.0])
        result = wilcoxon_signed_rank(scores, scores)
        assert result.p_value == 1.0


class TestCohensD:
    """Test Cohen's d effect size."""

    def test_no_difference(self):
        group = np.array([1, 2, 3, 4, 5], dtype=float)
        result = cohens_d(group, group)
        assert abs(result.cohens_d) < 0.01
        assert result.interpretation == "negligible"

    def test_large_effect(self):
        group_a = np.array([10, 11, 12, 13, 14], dtype=float)
        group_b = np.array([1, 2, 3, 4, 5], dtype=float)
        result = cohens_d(group_a, group_b)
        assert result.interpretation == "large"
        assert result.cohens_d > 0.8

    def test_negative_effect(self):
        group_a = np.array([1, 2, 3], dtype=float)
        group_b = np.array([10, 11, 12], dtype=float)
        result = cohens_d(group_a, group_b)
        assert result.cohens_d < 0

    def test_effect_sizes_ordered(self):
        rng = np.random.default_rng(42)
        base = rng.normal(5, 1, 50)
        small_diff = rng.normal(5.5, 1, 50)
        large_diff = rng.normal(15, 1, 50)
        d_small = cohens_d(small_diff, base)
        d_large = cohens_d(large_diff, base)
        assert abs(d_large.cohens_d) > abs(d_small.cohens_d)


class TestSignificanceReport:
    """Test the integrated significance analysis."""

    def test_perfect_regression(self):
        x = np.linspace(1, 10, 100)
        y_true = 3 * x**2
        y_pred = 3 * x**2
        report = run_significance_analysis(
            "test_domain", y_true, y_pred,
            n_permutations=500, n_bootstrap=500,
        )
        assert report.domain == "test_domain"
        assert report.r_squared > 0.999
        assert report.permutation_test is not None
        assert report.permutation_test.significant
        assert report.bootstrap_ci is not None
        ci_low, ci_high = report.bootstrap_ci
        assert ci_low > 0.99

    def test_noisy_regression(self):
        rng = np.random.default_rng(42)
        x = np.linspace(1, 10, 50)
        y_true = 2 * x + rng.normal(0, 3, 50)
        y_pred = 2 * x
        report = run_significance_analysis(
            "noisy", y_true, y_pred,
            n_permutations=500, n_bootstrap=500,
        )
        assert len(report.notes) >= 2

    def test_random_regression(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 30)
        y_pred = rng.normal(0, 1, 30)
        report = run_significance_analysis(
            "random", y_true, y_pred,
            n_permutations=500, n_bootstrap=500,
        )
        assert not report.permutation_test.significant
