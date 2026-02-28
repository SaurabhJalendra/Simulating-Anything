"""Tests for pipeline ablation study.

Verifies each ablation function returns correct structure and reasonable values.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.analysis.pipeline_ablation import (
    AblationExperiment,
    ablate_analysis_harmonic,
    ablate_data_quantity_lv,
    ablate_sampling_projectile,
    run_pipeline_ablation,
)


class TestAblationExperiment:
    """Test the AblationExperiment dataclass."""

    def test_create(self):
        exp = AblationExperiment(
            domain="test", component="sampling", variant="grid",
            r_squared=0.99, correct_form=True, n_samples=100,
        )
        assert exp.domain == "test"
        assert exp.r_squared == 0.99
        assert exp.correct_form is True

    def test_description_default(self):
        exp = AblationExperiment(
            domain="test", component="test", variant="test",
            r_squared=0.5, correct_form=False, n_samples=10,
        )
        assert exp.description == ""


class TestSamplingAblation:
    """Test sampling strategy ablation for projectile."""

    def test_returns_four_strategies(self):
        results = ablate_sampling_projectile()
        assert len(results) == 4

    def test_all_are_ablation_experiments(self):
        results = ablate_sampling_projectile()
        for r in results:
            assert isinstance(r, AblationExperiment)

    def test_grid_has_high_r2(self):
        results = ablate_sampling_projectile()
        grid = [r for r in results if "grid" in r.variant][0]
        assert grid.r_squared > 0.999, "Grid sampling should yield near-perfect R²"

    def test_grid_finds_correct_form(self):
        results = ablate_sampling_projectile()
        grid = [r for r in results if "grid" in r.variant][0]
        assert grid.correct_form, "Grid sampling should recover 1/g coefficient"

    def test_all_strategies_have_samples(self):
        results = ablate_sampling_projectile()
        for r in results:
            assert r.n_samples > 0
            assert r.domain == "projectile"
            assert r.component == "sampling"

    def test_variant_names(self):
        results = ablate_sampling_projectile()
        variants = {r.variant for r in results}
        assert "grid (15x15)" in variants
        assert "random uniform" in variants
        assert "clustered (narrow)" in variants
        assert "edge-focused" in variants


class TestAnalysisMethodAblation:
    """Test analysis method ablation for harmonic oscillator."""

    def test_returns_four_methods(self):
        results = ablate_analysis_harmonic()
        assert len(results) == 4

    def test_fft_has_high_r2(self):
        results = ablate_analysis_harmonic()
        fft = [r for r in results if "FFT" in r.variant][0]
        assert fft.r_squared > 0.9, "FFT should accurately detect frequency"

    def test_polynomial_has_wrong_form(self):
        results = ablate_analysis_harmonic()
        poly = [r for r in results if "polynomial" in r.variant][0]
        assert poly.correct_form is False, "Polynomial fit has wrong functional form"

    def test_all_methods_labeled(self):
        results = ablate_analysis_harmonic()
        for r in results:
            assert r.domain == "harmonic_oscillator"
            assert r.component == "analysis_method"

    def test_variant_names(self):
        results = ablate_analysis_harmonic()
        variants = {r.variant for r in results}
        assert "FFT peak" in variants
        assert "zero-crossing" in variants
        assert "autocorrelation" in variants
        assert "polynomial (wrong form)" in variants


class TestDataQuantityAblation:
    """Test data quantity ablation for Lotka-Volterra."""

    def test_returns_seven_sizes(self):
        results = ablate_data_quantity_lv()
        assert len(results) == 7

    def test_more_data_improves_convergence(self):
        results = ablate_data_quantity_lv()
        # Long trajectories should converge better than short ones
        short = [r for r in results if r.n_samples <= 501][0]  # 500 steps
        long = [r for r in results if r.n_samples >= 10001][-1]  # 10000+ steps
        assert long.r_squared >= short.r_squared, \
            "Longer trajectories should give better equilibrium estimates"

    def test_long_trajectory_correct_form(self):
        results = ablate_data_quantity_lv()
        long = [r for r in results if r.n_samples >= 10001]
        # At least one long trajectory should converge
        assert any(r.correct_form for r in long), \
            "Long LV trajectories should converge to equilibrium within 10%"

    def test_all_labeled_correctly(self):
        results = ablate_data_quantity_lv()
        for r in results:
            assert r.domain == "lotka_volterra"
            assert r.component == "data_quantity"


class TestFullAblation:
    """Test the full ablation pipeline."""

    def test_run_returns_dict(self, tmp_path):
        results = run_pipeline_ablation(output_dir=tmp_path / "ablation")
        assert isinstance(results, dict)
        assert "sampling_strategy" in results
        assert "analysis_method" in results
        assert "data_quantity" in results

    def test_saves_json(self, tmp_path):
        out = tmp_path / "ablation"
        run_pipeline_ablation(output_dir=out)
        assert (out / "ablation_results.json").exists()

    def test_results_structure(self, tmp_path):
        results = run_pipeline_ablation(output_dir=tmp_path / "ablation")
        for category in results.values():
            assert isinstance(category, list)
            for entry in category:
                assert "variant" in entry
                assert "r_squared" in entry
                assert "correct_form" in entry
                assert "n_samples" in entry
