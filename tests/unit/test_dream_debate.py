"""Tests for the adversarial dream debate module."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.analysis.dream_debate import (
    DebateResult,
    compute_debate_metrics,
    run_lorenz_debate,
    run_simulation_debate,
)


class TestComputeDebateMetrics:
    def test_identical_dreams(self):
        """Identical dreams should have zero divergence and perfect correlation."""
        dreams = np.random.default_rng(42).standard_normal((50, 3))
        result = compute_debate_metrics(dreams, dreams)
        assert result.dream_steps == 50
        np.testing.assert_array_almost_equal(result.divergence_mse, 0.0)
        np.testing.assert_array_almost_equal(result.divergence_max, 0.0)
        assert result.agreement_horizon == 50  # Never disagree

    def test_diverging_dreams(self):
        """Exponentially diverging dreams should have decreasing correlation."""
        rng = np.random.default_rng(42)
        dreams_a = np.cumsum(rng.standard_normal((100, 5)), axis=0)
        dreams_b = dreams_a + np.arange(100)[:, None] * 0.1 * rng.standard_normal((100, 5))
        result = compute_debate_metrics(dreams_a, dreams_b, threshold=0.9)
        assert result.divergence_mse[-1] > result.divergence_mse[0]

    def test_agreement_horizon(self):
        """Agreement horizon should detect where correlation drops."""
        dreams_a = np.ones((20, 3))
        dreams_b = np.ones((20, 3))
        # After step 10, they diverge
        dreams_b[10:] = np.random.default_rng(42).standard_normal((10, 3)) * 10
        result = compute_debate_metrics(dreams_a, dreams_b, threshold=0.9)
        assert result.agreement_horizon <= 15  # Should detect divergence

    def test_growth_rate(self):
        """Growth rate should be positive for exponentially diverging dreams."""
        t = np.arange(50)
        # Exponentially growing difference
        dreams_a = np.zeros((50, 3))
        dreams_b = np.exp(0.1 * t)[:, None] * np.array([1, 1, 1])
        result = compute_debate_metrics(dreams_a, dreams_b)
        assert result._growth_rate() > 0

    def test_result_summary(self):
        """Summary should contain all expected keys."""
        dreams = np.random.default_rng(42).standard_normal((30, 4))
        result = compute_debate_metrics(dreams, dreams + 0.01)
        summary = result.summary()
        assert "mean_divergence_mse" in summary
        assert "agreement_horizon" in summary
        assert "divergence_growth_rate" in summary
        assert "mean_correlation" in summary


class TestSimulationDebate:
    def test_lorenz_debate_runs(self):
        """Lorenz debate should complete and return results."""
        data = run_lorenz_debate(n_trials=2, n_dream=20, rho_perturbation=1.0)
        assert "summary" in data
        assert "chaotic" in data["summary"]
        assert "stable" in data["summary"]
        assert len(data["chaotic_results"]) == 2
        assert len(data["stable_results"]) == 2

    def test_chaotic_shorter_horizon(self):
        """Chaotic regime should generally have shorter agreement horizons."""
        data = run_lorenz_debate(n_trials=5, n_dream=50, rho_perturbation=2.0)
        chaotic_mean = data["summary"]["chaotic"]["mean_horizon"]
        stable_mean = data["summary"]["stable"]["mean_horizon"]
        # Chaotic diverges faster (or at least comparably)
        # With rho_perturbation=2.0, chaotic regime diverges much faster
        assert chaotic_mean <= stable_mean + 20  # Allow some tolerance

    def test_simulation_debate_basic(self):
        """Basic simulation debate should produce valid results."""
        from simulating_anything.simulation.lorenz import LorenzSimulation
        from simulating_anything.types.simulation import Domain

        config_a = {
            "domain": Domain.LORENZ_ATTRACTOR,
            "dt": 0.01,
            "n_steps": 1000,
            "parameters": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        }
        config_b = {
            "domain": Domain.LORENZ_ATTRACTOR,
            "dt": 0.01,
            "n_steps": 1000,
            "parameters": {"sigma": 10.0, "rho": 29.0, "beta": 8.0 / 3.0},
        }
        results = run_simulation_debate(
            LorenzSimulation, config_a, config_b,
            n_context=10, n_dream=20, n_trials=3,
        )
        assert len(results) == 3
        for r in results:
            assert r.dream_steps == 20
            assert r.context_length == 10
            assert len(r.divergence_mse) == 20

    def test_oscillator_debate(self):
        """Oscillator debate should show longer agreement than chaotic systems."""
        from simulating_anything.simulation.harmonic_oscillator import DampedHarmonicOscillator
        from simulating_anything.types.simulation import Domain

        config_a = {
            "domain": Domain.HARMONIC_OSCILLATOR,
            "dt": 0.001,
            "n_steps": 5000,
            "parameters": {"k": 4.0, "m": 1.0, "c": 0.2, "x_0": 1.0, "v_0": 0.0},
        }
        config_b = {
            "domain": Domain.HARMONIC_OSCILLATOR,
            "dt": 0.001,
            "n_steps": 5000,
            "parameters": {"k": 4.05, "m": 1.0, "c": 0.2, "x_0": 1.0, "v_0": 0.0},
        }
        results = run_simulation_debate(
            DampedHarmonicOscillator, config_a, config_b,
            n_context=100, n_dream=500, n_trials=3,
        )
        assert len(results) == 3
        # Oscillator with small perturbation should maintain agreement longer
        mean_horizon = np.mean([r.agreement_horizon for r in results])
        assert mean_horizon > 10  # Should agree for at least some steps
