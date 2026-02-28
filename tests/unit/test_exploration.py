"""Tests for uncertainty-driven exploration."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.exploration.uncertainty_driven import UncertaintyDrivenExplorer
from simulating_anything.types.trajectory import TrajectoryData, TrajectoryMetadata


class TestUncertaintyDrivenExplorer:
    """Tests for the UncertaintyDrivenExplorer."""

    def _make_explorer(self, **kwargs) -> UncertaintyDrivenExplorer:
        defaults = {
            "sweep_ranges": {"alpha": (0.5, 2.0), "beta": (0.1, 0.8)},
            "n_points_per_dim": 5,
        }
        defaults.update(kwargs)
        return UncertaintyDrivenExplorer(**defaults)

    def test_grid_construction(self):
        explorer = self._make_explorer()
        assert explorer._grid_physical.shape == (25, 2)  # 5x5 grid
        assert explorer._grid_physical[:, 0].min() >= 0.5
        assert explorer._grid_physical[:, 0].max() <= 2.0
        assert explorer._grid_physical[:, 1].min() >= 0.1
        assert explorer._grid_physical[:, 1].max() <= 0.8

    def test_propose_parameters_returns_valid(self):
        explorer = self._make_explorer()
        params = explorer.propose_parameters()
        assert "alpha" in params
        assert "beta" in params
        assert 0.5 <= params["alpha"] <= 2.0
        assert 0.1 <= params["beta"] <= 0.8

    def test_propose_parameters_no_repeat(self):
        explorer = self._make_explorer()
        proposals = set()
        for _ in range(5):
            params = explorer.propose_parameters()
            key = (round(params["alpha"], 4), round(params["beta"], 4))
            proposals.add(key)
        assert len(proposals) == 5  # All different

    def test_update_reduces_uncertainty(self):
        explorer = self._make_explorer()
        params = explorer.propose_parameters()
        initial_mean = np.mean(explorer._uncertainties)

        traj = TrajectoryData(
            parameters=params,
            metadata=TrajectoryMetadata(novelty_score=0.5),
        )
        explorer.update(traj)

        new_mean = np.mean(explorer._uncertainties)
        assert new_mean < initial_mean

    def test_exploration_progress(self):
        explorer = self._make_explorer()
        progress = explorer.get_exploration_progress()
        assert progress["total_grid_points"] == 25
        assert progress["visited"] == 0
        assert progress["coverage_fraction"] == 0.0

        explorer.propose_parameters()
        progress = explorer.get_exploration_progress()
        assert progress["visited"] == 1

    def test_set_uncertainties(self):
        explorer = self._make_explorer()
        new_u = np.random.rand(25)
        explorer.set_uncertainties(new_u)
        np.testing.assert_array_equal(explorer._uncertainties, new_u)

    def test_single_dim_grid(self):
        explorer = UncertaintyDrivenExplorer(
            sweep_ranges={"x": (0.0, 1.0)},
            n_points_per_dim=10,
        )
        assert explorer._grid_physical.shape == (10, 1)

    def test_high_dim_grid(self):
        explorer = UncertaintyDrivenExplorer(
            sweep_ranges={"a": (0, 1), "b": (0, 1), "c": (0, 1), "d": (0, 1)},
            n_points_per_dim=5,
        )
        # 4D uses quasi-random, capped at n_points^3 = 125
        assert explorer._grid_physical.shape[1] == 4

    def test_full_exploration_loop(self):
        """Run a complete explore-update loop."""
        explorer = self._make_explorer(n_points_per_dim=3)
        for i in range(9):  # 3x3 = 9 grid points
            params = explorer.propose_parameters()
            traj = TrajectoryData(
                parameters=params,
                metadata=TrajectoryMetadata(novelty_score=0.1 * i),
            )
            explorer.update(traj)

        progress = explorer.get_exploration_progress()
        assert progress["coverage_fraction"] == 1.0
        assert progress["n_trajectories"] == 9


class TestAblationModule:
    """Tests for the ablation studies module."""

    def test_run_ablation_basic(self):
        from simulating_anything.analysis.ablation import run_ablation

        def metric_fn(params):
            return params["a"] * params["b"] + params["c"]

        baseline = {"a": 2.0, "b": 3.0, "c": 1.0}
        results = run_ablation(metric_fn, baseline, metric_name="product")

        assert len(results) == 3
        # All results should have non-zero effect
        for r in results:
            assert r.factor_name in ["a", "b", "c"]
            assert r.original_value == 7.0  # 2*3 + 1
            assert r.effect_size >= 0

    def test_ablation_essential_factors(self):
        from simulating_anything.analysis.ablation import run_ablation

        def metric_fn(params):
            return params["critical"] * 10 + params["minor"] * 0.01

        baseline = {"critical": 1.0, "minor": 1.0}
        results = run_ablation(metric_fn, baseline)

        # Critical should be marked essential (effect > 50%)
        critical_result = next(r for r in results if r.factor_name == "critical")
        assert critical_result.is_essential

    def test_ablation_sorted_by_effect(self):
        from simulating_anything.analysis.ablation import run_ablation

        def metric_fn(params):
            return params["a"] * 100 + params["b"] * 10 + params["c"]

        baseline = {"a": 1.0, "b": 1.0, "c": 1.0}
        results = run_ablation(metric_fn, baseline)

        # Should be sorted by effect size descending
        for i in range(len(results) - 1):
            assert results[i].effect_size >= results[i + 1].effect_size

    def test_ablation_custom_fraction(self):
        from simulating_anything.analysis.ablation import run_ablation

        def metric_fn(params):
            return params["x"]

        baseline = {"x": 10.0}
        results = run_ablation(metric_fn, baseline, ablation_fraction=5.0)

        assert len(results) == 1
        assert results[0].ablated_value == 5.0
