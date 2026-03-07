"""Tests for robustness and failure mode analysis."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.analysis.robustness import (
    ExtrapolationResult,
    NoiseToleranceResult,
    RobustnessReport,
    RobustnessResult,
    SampleEfficiencyResult,
    run_robustness_analysis,
    test_extrapolation,
    test_noise_tolerance,
    test_sample_efficiency,
)


class TestNoiseToleranceResult:
    """Test NoiseToleranceResult dataclass."""

    def test_creation(self):
        r = NoiseToleranceResult(
            domain="test", noise_levels=[0.0, 0.1],
            r_squared_values=[1.0, 0.9],
            critical_noise=0.1, equation_correct_up_to=0.0,
        )
        assert r.domain == "test"
        assert r.critical_noise == 0.1


class TestNoiseTolerance:
    """Test noise tolerance analysis."""

    def test_projectile_clean(self):
        result = test_noise_tolerance(domain="projectile", n_samples=225)
        # Clean data should have R^2 = 1.0
        assert result.r_squared_values[0] > 0.999

    def test_projectile_degrades(self):
        result = test_noise_tolerance(domain="projectile", n_samples=225)
        # R^2 should decrease with noise
        assert result.r_squared_values[0] >= result.r_squared_values[-1]

    def test_projectile_critical_noise(self):
        result = test_noise_tolerance(domain="projectile", n_samples=225)
        # Should tolerate at least some noise
        assert result.critical_noise >= 0.01

    def test_oscillator(self):
        result = test_noise_tolerance(domain="harmonic_oscillator", n_samples=100)
        assert result.r_squared_values[0] > 0.999
        assert result.critical_noise >= 0.01

    def test_custom_noise_levels(self):
        result = test_noise_tolerance(
            domain="projectile", noise_levels=[0.0, 0.5, 1.0],
        )
        assert len(result.noise_levels) == 3
        assert len(result.r_squared_values) == 3


class TestSampleEfficiency:
    """Test sample efficiency analysis."""

    def test_projectile(self):
        result = test_sample_efficiency(domain="projectile")
        # Should achieve R^2 >= 0.95 with reasonable samples
        assert result.min_samples_r2_95 <= 225

    def test_oscillator(self):
        result = test_sample_efficiency(domain="harmonic_oscillator")
        assert result.min_samples_r2_95 <= 225

    def test_r_squared_improves_with_samples(self):
        result = test_sample_efficiency(domain="projectile")
        # Generally should improve with more samples
        last = result.r_squared_values[-1]
        first = result.r_squared_values[0]
        assert last >= first or abs(last - first) < 0.05

    def test_custom_sample_counts(self):
        result = test_sample_efficiency(
            domain="projectile", sample_counts=[9, 25, 100],
        )
        assert len(result.sample_counts) == 3


class TestExtrapolation:
    """Test extrapolation analysis."""

    def test_projectile_extrapolation(self):
        result = test_extrapolation(domain="projectile")
        # Correct-form equation should extrapolate perfectly
        assert result.r_squared_values[0] > 0.99  # at 1.0x (training range)

    def test_oscillator_extrapolation(self):
        result = test_extrapolation(domain="harmonic_oscillator")
        assert result.r_squared_values[0] > 0.99

    def test_fields(self):
        result = test_extrapolation(domain="projectile")
        assert isinstance(result.train_range, tuple)
        assert len(result.extrap_factors) == len(result.r_squared_values)
        assert result.failure_factor > 0

    def test_custom_factors(self):
        result = test_extrapolation(
            domain="projectile", extrap_factors=[1.0, 2.0],
        )
        assert len(result.extrap_factors) == 2


class TestRobustnessResult:
    """Test RobustnessResult dataclass."""

    def test_passed(self):
        r = RobustnessResult(
            test_name="test", domain="proj",
            passed=True, metric_value=0.1, threshold=0.05,
        )
        assert r.passed

    def test_failed(self):
        r = RobustnessResult(
            test_name="test", domain="proj",
            passed=False, metric_value=0.01, threshold=0.05,
        )
        assert not r.passed


class TestRobustnessReport:
    """Test RobustnessReport."""

    def test_empty(self):
        r = RobustnessReport(domains_tested=[])
        assert r.n_passed() == 0
        assert r.n_failed() == 0
        assert r.pass_rate() == 0.0

    def test_counts(self):
        r = RobustnessReport(
            domains_tested=["a"],
            individual_tests=[
                RobustnessResult("t1", "a", True, 0.5, 0.1),
                RobustnessResult("t2", "a", False, 0.01, 0.1),
                RobustnessResult("t3", "a", True, 0.2, 0.1),
            ],
        )
        assert r.n_passed() == 2
        assert r.n_failed() == 1
        assert abs(r.pass_rate() - 2 / 3) < 0.01


class TestFullAnalysis:
    """Test integrated robustness analysis."""

    def test_default_domains(self):
        report = run_robustness_analysis()
        assert len(report.domains_tested) == 2
        assert "projectile" in report.domains_tested

    def test_has_all_test_types(self):
        report = run_robustness_analysis(domains=["projectile"])
        test_names = [t.test_name for t in report.individual_tests]
        assert "noise_tolerance" in test_names
        assert "sample_efficiency" in test_names
        assert "extrapolation" in test_names

    def test_results_populated(self):
        report = run_robustness_analysis(domains=["projectile"])
        assert len(report.noise_results) == 1
        assert len(report.sample_results) == 1
        assert len(report.extrapolation_results) == 1

    def test_mostly_passes(self):
        report = run_robustness_analysis()
        # Pipeline should pass most robustness tests
        assert report.pass_rate() >= 0.5
