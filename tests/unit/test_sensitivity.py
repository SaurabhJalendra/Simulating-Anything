"""Tests for sensitivity analysis framework."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.analysis.sensitivity import (
    SensitivityResult,
    sensitivity_data_quantity,
    sensitivity_noise,
    sensitivity_param_range,
)


class TestDataQuantitySensitivity:
    def test_r2_improves_with_more_data(self):
        """More data points should give equal or better R²."""
        result = sensitivity_data_quantity(
            sample_counts=[10, 50, 225],
        )
        assert len(result.r_squared) == 3
        # R² should generally be very high for this clean problem
        assert result.r_squared[-1] > 0.999

    def test_even_small_data_recovers_form(self):
        """Even 10 data points should get R² > 0.99."""
        result = sensitivity_data_quantity(
            sample_counts=[10],
        )
        assert result.r_squared[0] > 0.99

    def test_result_structure(self):
        result = sensitivity_data_quantity(sample_counts=[25])
        assert result.domain == "projectile"
        assert result.variable == "n_samples"
        assert len(result.values) == 1
        assert len(result.r_squared) == 1
        assert len(result.discovered_form) == 1


class TestNoiseSensitivity:
    def test_no_noise_perfect(self):
        """Zero noise should give R² = 1.0."""
        result = sensitivity_noise(
            noise_levels=[0.0],
            n_samples=100,
        )
        assert result.r_squared[0] > 0.9999

    def test_moderate_noise_still_good(self):
        """5% noise should still give R² > 0.99."""
        result = sensitivity_noise(
            noise_levels=[0.05],
            n_samples=225,
        )
        assert result.r_squared[0] > 0.99

    def test_high_noise_degrades(self):
        """50% noise should degrade R² below perfect."""
        result = sensitivity_noise(
            noise_levels=[0.5],
            n_samples=225,
        )
        assert result.r_squared[0] < 0.99


class TestParamRangeSensitivity:
    def test_full_range_perfect(self):
        """Full parameter range should give R² ~= 1.0."""
        result = sensitivity_param_range(
            range_fractions=[1.0],
            n_samples=225,
        )
        assert result.r_squared[0] > 0.999

    def test_narrow_range_still_works(self):
        """Even narrow range should recover correct form."""
        result = sensitivity_param_range(
            range_fractions=[0.1],
            n_samples=100,
        )
        # Correct functional form should fit well even on narrow range
        assert result.r_squared[0] > 0.99

    def test_monotonic_values(self):
        """All R² values should be high for clean data."""
        result = sensitivity_param_range(
            range_fractions=[0.1, 0.5, 1.0],
            n_samples=100,
        )
        for r2 in result.r_squared:
            assert r2 > 0.99
