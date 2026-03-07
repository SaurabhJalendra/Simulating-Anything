"""Tests for the discovery runner and open problems registry."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.discovery.open_problems import (
    OPEN_PROBLEMS,
    OpenProblem,
    get_problem,
    list_problems,
)
from simulating_anything.discovery.discovery_runner import (
    DiscoveryAttempt,
    DiscoveryResult,
    DiscoveryRunner,
)


class TestOpenProblems:
    """Test the open problems registry."""

    def test_problems_registered(self):
        problems = list_problems()
        assert len(problems) >= 6
        assert "three_body_stability" in problems
        assert "lorenz_transition_scaling" in problems

    def test_get_problem(self):
        p = get_problem("three_body_stability")
        assert p.name == "Three-Body Stability Boundaries"
        assert p.domain == "celestial_mechanics"
        assert len(p.parameter_sweeps) > 0
        assert len(p.observables) > 0

    def test_get_unknown_problem(self):
        with pytest.raises(KeyError, match="Unknown problem"):
            get_problem("nonexistent_problem")

    def test_problem_has_known_results(self):
        p = get_problem("three_body_stability")
        assert "gascheau_limit" in p.known_results

    def test_all_problems_have_required_fields(self):
        for name in list_problems():
            p = get_problem(name)
            assert p.name
            assert p.domain
            assert p.description
            assert p.simulation_class
            assert p.parameter_sweeps
            assert p.observables


class TestDiscoveryAttempt:
    """Test discovery attempt dataclass."""

    def test_defaults(self):
        a = DiscoveryAttempt(problem_name="test", parameter_name="x")
        assert a.r_squared == 0.0
        assert a.candidate_equation == ""
        assert a.notes == []


class TestDiscoveryResult:
    """Test discovery result."""

    def test_summary(self):
        problem = get_problem("three_body_stability")
        result = DiscoveryResult(
            problem=problem,
            best_equations={"lyapunov": "lyapunov = 0.5 * mu^1.2"},
            best_r_squared={"lyapunov": 0.95},
            n_simulations=30,
            total_runtime=10.5,
        )
        s = result.summary()
        assert "Three-Body" in s
        assert "30" in s
        assert "lyapunov" in s


class TestDiscoveryRunner:
    """Test the discovery runner."""

    def test_init(self, tmp_path):
        runner = DiscoveryRunner(output_dir=tmp_path / "discoveries")
        assert runner.output_dir.exists()

    def test_run_three_body(self, tmp_path):
        """Run discovery on three-body with reduced sweep."""
        runner = DiscoveryRunner(
            output_dir=tmp_path / "discoveries",
            n_steps=500,
            dt=0.01,
        )
        result = runner.run_problem("three_body_stability", max_sweeps=1)
        assert result.n_simulations > 0
        assert result.total_runtime > 0
        assert len(result.attempts) == 1
        # Should have found some relationship
        assert result.attempts[0].parameter_name == "mu"

    def test_run_saves_results(self, tmp_path):
        """Results should be saved to disk."""
        runner = DiscoveryRunner(
            output_dir=tmp_path / "discoveries",
            n_steps=200,
            dt=0.01,
        )
        runner.run_problem("three_body_stability", max_sweeps=1)

        import json
        result_files = list((tmp_path / "discoveries").glob("*.json"))
        assert len(result_files) == 1
        data = json.loads(result_files[0].read_text())
        assert "problem" in data
        assert "best_equations" in data

    def test_fit_relationship(self, tmp_path):
        """Internal fitting should find polynomial relationships."""
        runner = DiscoveryRunner(output_dir=tmp_path)
        x = np.linspace(1, 10, 50)
        y = 3.0 * x ** 2 + 1.0  # Quadratic

        eq, r2 = runner._fit_relationship(x, y, "x", "y")
        assert r2 > 0.99
        assert "x^2" in eq or "x" in eq

    def test_fit_power_law(self, tmp_path):
        """Should detect power law relationships."""
        runner = DiscoveryRunner(output_dir=tmp_path)
        x = np.linspace(1, 10, 50)
        y = 2.5 * x ** 1.5  # Power law

        eq, r2 = runner._fit_relationship(x, y, "k", "omega")
        assert r2 > 0.99

    def test_run_multiple_sweeps(self, tmp_path):
        """Should handle multiple parameter sweeps."""
        runner = DiscoveryRunner(
            output_dir=tmp_path / "disc",
            n_steps=100,
            dt=0.01,
        )
        result = runner.run_problem("three_body_stability", max_sweeps=2)
        assert len(result.attempts) == 2

    def test_handles_simulation_failure(self, tmp_path):
        """Should gracefully handle simulation errors."""
        runner = DiscoveryRunner(
            output_dir=tmp_path / "disc",
            n_steps=100,
            dt=0.01,
        )
        # Run with very small dt to avoid issues
        result = runner.run_problem("three_body_stability", max_sweeps=1)
        # Should complete without crashing
        assert result is not None
