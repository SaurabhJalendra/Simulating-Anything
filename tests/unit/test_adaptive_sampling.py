"""Test adaptive parameter-sweep sampling (ADR-0001 Change #2)."""

from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.campaign.adaptive_sampling import (
    adaptive_parameter_grid,
    density_near,
    detect_gradient_peaks,
)


class TestDetectGradientPeaks:
    def test_sigmoid_peak_detected(self):
        """A sharp sigmoid observable should produce one detected peak
        near the inflection point."""
        params = np.linspace(0.0, 1.0, 30)
        # Sharp sigmoid at p=0.5
        observables = 1.0 / (1.0 + np.exp(-50.0 * (params - 0.5)))
        peaks = detect_gradient_peaks(params, observables)
        assert len(peaks) >= 1
        assert any(abs(p - 0.5) < 0.1 for p in peaks)

    def test_flat_observable_no_peaks(self):
        """A constant observable should produce no peaks."""
        params = np.linspace(0.0, 1.0, 30)
        observables = np.ones_like(params)
        peaks = detect_gradient_peaks(params, observables)
        assert peaks == []

    def test_linear_observable_no_peaks(self):
        """A linear observable has uniform gradient — no peaks."""
        params = np.linspace(0.0, 1.0, 30)
        observables = 2.0 * params + 1.0
        peaks = detect_gradient_peaks(params, observables)
        assert peaks == []

    def test_two_peaks_detected(self):
        """Two distinct sigmoid steps produce two peaks."""
        params = np.linspace(0.0, 1.0, 50)
        observables = 1.0 / (1.0 + np.exp(-80.0 * (params - 0.3))) + 1.0 / (
            1.0 + np.exp(-80.0 * (params - 0.7))
        )
        peaks = detect_gradient_peaks(params, observables)
        assert len(peaks) >= 2
        assert any(abs(p - 0.3) < 0.1 for p in peaks)
        assert any(abs(p - 0.7) < 0.1 for p in peaks)

    def test_too_few_points_returns_empty(self):
        peaks = detect_gradient_peaks(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert peaks == []

    def test_nan_observables_filtered(self):
        """Failed simulations (NaN) should not crash detection."""
        params = np.linspace(0.0, 1.0, 30)
        observables = 1.0 / (1.0 + np.exp(-50.0 * (params - 0.5)))
        observables[5:8] = np.nan
        peaks = detect_gradient_peaks(params, observables)
        # Should still detect the peak (or return empty if too few points,
        # but here we have plenty)
        assert isinstance(peaks, list)


class TestAdaptiveParameterGrid:
    def test_no_coarse_data_returns_uniform(self):
        grid = adaptive_parameter_grid((0.0, 1.0), n_total=20)
        assert len(grid) == 20
        np.testing.assert_allclose(grid, np.linspace(0.0, 1.0, 20))

    def test_concentrates_near_detected_peak(self):
        """With a sharp sigmoid coarse observation, the refined grid should
        have more points near the inflection than the uniform baseline."""
        params_coarse = np.linspace(0.0, 1.0, 15)
        obs_coarse = 1.0 / (1.0 + np.exp(-50.0 * (params_coarse - 0.5)))

        adaptive = adaptive_parameter_grid(
            (0.0, 1.0),
            n_total=30,
            coarse_observables=obs_coarse,
            coarse_params=params_coarse,
            refinement_fraction=0.6,
            refinement_window=0.1,
        )
        uniform = np.linspace(0.0, 1.0, 30)

        adaptive_density = density_near(adaptive, target=0.5, window=0.1)
        uniform_density = density_near(uniform, target=0.5, window=0.1)
        # Adaptive should have at least 2x the density of uniform near peak
        assert adaptive_density >= 2 * uniform_density, (
            f"Adaptive density near peak ({adaptive_density}) "
            f"not >= 2x uniform ({uniform_density})"
        )

    def test_no_peak_falls_back_to_uniform(self):
        """If coarse data shows no peak, adaptive returns uniform grid."""
        params_coarse = np.linspace(0.0, 1.0, 15)
        obs_coarse = 0.5 * params_coarse  # linear, no peak

        grid = adaptive_parameter_grid(
            (0.0, 1.0),
            n_total=20,
            coarse_observables=obs_coarse,
            coarse_params=params_coarse,
        )
        np.testing.assert_allclose(grid, np.linspace(0.0, 1.0, 20))

    def test_grid_size_bounded(self):
        params_coarse = np.linspace(0.0, 1.0, 15)
        obs_coarse = 1.0 / (1.0 + np.exp(-50.0 * (params_coarse - 0.5)))
        grid = adaptive_parameter_grid(
            (0.0, 1.0),
            n_total=25,
            coarse_observables=obs_coarse,
            coarse_params=params_coarse,
        )
        assert len(grid) <= 25
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_grid_is_sorted_unique(self):
        params_coarse = np.linspace(0.0, 1.0, 15)
        obs_coarse = 1.0 / (1.0 + np.exp(-50.0 * (params_coarse - 0.5)))
        grid = adaptive_parameter_grid(
            (0.0, 1.0),
            n_total=30,
            coarse_observables=obs_coarse,
            coarse_params=params_coarse,
        )
        assert np.all(np.diff(grid) > 0), "Grid should be strictly sorted ascending"
        assert len(grid) == len(np.unique(grid)), "Grid should have unique values"

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            adaptive_parameter_grid((1.0, 0.0), n_total=10)
        with pytest.raises(ValueError):
            adaptive_parameter_grid((0.0, 1.0), n_total=1)

    def test_two_peaks_concentrate_at_both(self):
        """Two-peak coarse data should produce density at both peaks."""
        params_coarse = np.linspace(0.0, 1.0, 30)
        obs_coarse = 1.0 / (1.0 + np.exp(-80.0 * (params_coarse - 0.3))) + 1.0 / (
            1.0 + np.exp(-80.0 * (params_coarse - 0.7))
        )
        grid = adaptive_parameter_grid(
            (0.0, 1.0),
            n_total=40,
            coarse_observables=obs_coarse,
            coarse_params=params_coarse,
        )
        d_at_03 = density_near(grid, target=0.3, window=0.1)
        d_at_07 = density_near(grid, target=0.7, window=0.1)
        d_uniform = density_near(np.linspace(0.0, 1.0, 40), target=0.3, window=0.1)
        assert d_at_03 >= d_uniform
        assert d_at_07 >= d_uniform
