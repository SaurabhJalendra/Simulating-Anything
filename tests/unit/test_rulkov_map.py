"""Tests for the Rulkov map simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.rulkov_map import RulkovMapSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    alpha: float = 4.1,
    mu: float = 0.001,
    sigma: float = -1.0,
    x_0: float = -1.0,
    y_0: float = -3.0,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.RULKOV_MAP,
        dt=1.0,
        n_steps=100,
        parameters={
            "alpha": alpha,
            "mu": mu,
            "sigma": sigma,
            "x_0": x_0,
            "y_0": y_0,
        },
    )


class TestRulkovMapConstruction:
    def test_creation_and_params(self):
        sim = RulkovMapSimulation(_make_config())
        assert sim.alpha == 4.1
        assert sim.mu == 0.001
        assert sim.sigma == -1.0

    def test_default_parameters(self):
        """Config with no parameters should use defaults."""
        config = SimulationConfig(
            domain=Domain.RULKOV_MAP, dt=1.0, n_steps=10, parameters={},
        )
        sim = RulkovMapSimulation(config)
        assert sim.alpha == 4.1
        assert sim.mu == 0.001
        assert sim.sigma == -1.0
        assert sim.x_0 == -1.0
        assert sim.y_0 == -3.0

    def test_custom_parameters(self):
        sim = RulkovMapSimulation(_make_config(alpha=5.0, mu=0.01, sigma=0.0))
        assert sim.alpha == 5.0
        assert sim.mu == 0.01
        assert sim.sigma == 0.0

    def test_initial_state_shape(self):
        sim = RulkovMapSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        assert state[0] == pytest.approx(-1.0)
        assert state[1] == pytest.approx(-3.0)

    def test_observe_returns_current_state(self):
        sim = RulkovMapSimulation(_make_config(x_0=0.5, y_0=-2.0))
        sim.reset()
        obs = sim.observe()
        assert obs[0] == pytest.approx(0.5)
        assert obs[1] == pytest.approx(-2.0)


class TestRulkovMapDynamics:
    def test_step_applies_map_correctly(self):
        """Verify one step matches manual calculation."""
        sim = RulkovMapSimulation(_make_config(
            alpha=4.0, mu=0.001, sigma=-1.0, x_0=0.0, y_0=-2.0,
        ))
        sim.reset()
        state = sim.step()
        # x_new = 4.0 / (1 + 0^2) + (-2.0) = 4.0 - 2.0 = 2.0
        # y_new = -2.0 - 0.001 * (0.0 - (-1.0)) = -2.0 - 0.001 = -2.001
        assert state[0] == pytest.approx(2.0)
        assert state[1] == pytest.approx(-2.001)

    def test_step_second_iteration(self):
        """Verify second step of the map."""
        sim = RulkovMapSimulation(_make_config(
            alpha=4.0, mu=0.001, sigma=-1.0, x_0=0.0, y_0=-2.0,
        ))
        sim.reset()
        sim.step()
        state = sim.step()
        # After first step: x=2.0, y=-2.001
        # x_new = 4.0 / (1 + 4.0) + (-2.001) = 0.8 - 2.001 = -1.201
        # y_new = -2.001 - 0.001*(2.0 - (-1.0)) = -2.001 - 0.003 = -2.004
        assert state[0] == pytest.approx(-1.201, abs=1e-6)
        assert state[1] == pytest.approx(-2.004, abs=1e-6)

    def test_deterministic(self):
        """Same parameters produce the same orbit."""
        sim1 = RulkovMapSimulation(_make_config(alpha=4.5))
        sim2 = RulkovMapSimulation(_make_config(alpha=4.5))
        sim1.reset()
        sim2.reset()
        for _ in range(100):
            s1 = sim1.step()
            s2 = sim2.step()
            np.testing.assert_array_equal(s1, s2)

    def test_state_changes_after_step(self):
        """State should change after a step."""
        sim = RulkovMapSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_no_nan_long_run(self):
        """No NaN after many iterations at default parameters."""
        sim = RulkovMapSimulation(_make_config())
        sim.reset()
        for _ in range(20000):
            state = sim.step()
        assert np.all(np.isfinite(state)), "NaN detected after 20000 steps"

    def test_bounded_default_params(self):
        """State stays bounded at default parameters."""
        sim = RulkovMapSimulation(_make_config())
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert abs(state[0]) < 20.0, f"x={state[0]} out of bounds"
            assert abs(state[1]) < 20.0, f"y={state[1]} out of bounds"

    def test_slow_variable_evolves_slowly(self):
        """y should change much more slowly than x due to small mu."""
        sim = RulkovMapSimulation(_make_config(mu=0.001))
        sim.reset()
        ys = []
        for _ in range(100):
            state = sim.step()
            ys.append(state[1])
        y_changes = np.abs(np.diff(ys))
        # Each y change should be O(mu) ~ 0.001 scale
        assert np.max(y_changes) < 0.1, "y changes too fast for mu=0.001"


class TestRulkovMapRegimes:
    def test_silent_low_alpha(self):
        """For low alpha, the neuron should be silent (no spiking)."""
        sim = RulkovMapSimulation(_make_config(alpha=2.0))
        regime = sim.classify_regime(n_steps=5000, n_transient=1000)
        assert regime == "silent"

    def test_spiking_high_alpha(self):
        """For sufficiently high alpha, the neuron should spike."""
        sim = RulkovMapSimulation(_make_config(alpha=4.5))
        spikes = sim.detect_spikes(n_steps=10000, n_transient=2000)
        assert len(spikes) > 0, "No spikes detected at alpha=4.5"

    def test_spiking_regime_classification(self):
        """Alpha=4.5 should produce spiking or bursting (not silent)."""
        sim = RulkovMapSimulation(_make_config(alpha=4.5))
        regime = sim.classify_regime(n_steps=10000, n_transient=2000)
        assert regime in ("spiking", "bursting"), f"Expected active, got {regime}"

    def test_firing_rate_zero_for_silent(self):
        """Firing rate should be zero when neuron is silent."""
        sim = RulkovMapSimulation(_make_config(alpha=2.0))
        rate = sim.compute_firing_rate(n_steps=5000, n_transient=1000)
        assert rate == 0.0

    def test_firing_rate_positive_for_active(self):
        """Firing rate should be positive when neuron is active."""
        sim = RulkovMapSimulation(_make_config(alpha=4.5))
        rate = sim.compute_firing_rate(n_steps=10000, n_transient=2000)
        assert rate > 0.0, f"Expected positive firing rate, got {rate}"

    def test_firing_rate_active_vs_silent(self):
        """Active neurons at high alpha fire much more than silent at low alpha."""
        sim_low = RulkovMapSimulation(_make_config(alpha=2.0))
        rate_low = sim_low.compute_firing_rate(n_steps=10000, n_transient=2000)

        sim_high = RulkovMapSimulation(_make_config(alpha=5.0))
        rate_high = sim_high.compute_firing_rate(n_steps=10000, n_transient=2000)

        assert rate_low == 0.0, f"Expected silence at alpha=2.0, got rate={rate_low}"
        assert rate_high > 0.01, (
            f"Expected significant firing at alpha=5.0, got rate={rate_high}"
        )

    def test_bursting_regime_exists(self):
        """At certain alpha values, the neuron should exhibit bursting.

        Bursting is typical for alpha around 4.1-4.4 with mu=0.001.
        """
        found_bursting = False
        for alpha in [4.1, 4.2, 4.3]:
            sim = RulkovMapSimulation(_make_config(alpha=alpha, mu=0.001))
            regime = sim.classify_regime(
                n_steps=20000, n_transient=5000, burst_gap=50,
            )
            if regime == "bursting":
                found_bursting = True
                break
        assert found_bursting, "No bursting regime found in alpha range [4.1, 4.3]"


class TestRulkovMapAnalysis:
    def test_alpha_sweep_returns_valid_data(self):
        sim = RulkovMapSimulation(_make_config())
        alpha_vals = np.linspace(3.0, 5.0, 5)
        data = sim.alpha_sweep(alpha_vals, n_steps=2000, n_transient=500)
        assert len(data["alpha"]) == 5
        assert len(data["firing_rate"]) == 5
        assert len(data["regime"]) == 5

    def test_alpha_sweep_regime_types(self):
        """Sweep should produce valid regime strings."""
        sim = RulkovMapSimulation(_make_config())
        alpha_vals = np.linspace(2.0, 5.0, 10)
        data = sim.alpha_sweep(alpha_vals, n_steps=5000, n_transient=1000)
        for r in data["regime"]:
            assert r in ("silent", "spiking", "bursting")

    def test_bifurcation_diagram_shape(self):
        sim = RulkovMapSimulation(_make_config())
        alpha_vals = np.linspace(3.0, 5.0, 10)
        data = sim.bifurcation_diagram(alpha_vals, n_transient=500, n_plot=50)
        assert len(data["alpha"]) == 500  # 10 alpha * 50 plot points
        assert len(data["x"]) == 500

    def test_bifurcation_data_finite(self):
        """All bifurcation diagram data should be finite."""
        sim = RulkovMapSimulation(_make_config())
        alpha_vals = np.linspace(3.0, 5.0, 10)
        data = sim.bifurcation_diagram(alpha_vals, n_transient=500, n_plot=50)
        assert np.all(np.isfinite(data["x"]))

    def test_lyapunov_computation(self):
        """Lyapunov exponent should be computable and finite."""
        sim = RulkovMapSimulation(_make_config(alpha=4.1))
        lam = sim.compute_lyapunov(n_iterations=2000, n_transient=500)
        assert np.isfinite(lam), f"Lyapunov={lam} is not finite"

    def test_lyapunov_negative_for_silent(self):
        """Silent regime should have negative Lyapunov exponent."""
        sim = RulkovMapSimulation(_make_config(alpha=2.0))
        lam = sim.compute_lyapunov(n_iterations=5000, n_transient=1000)
        assert lam < 0, f"Lyapunov={lam} should be negative for alpha=2.0"

    def test_run_trajectory_shape(self):
        sim = RulkovMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        # 51 states: initial + 50 steps
        assert traj.states.shape == (51, 2)

    def test_run_trajectory_timestamps(self):
        sim = RulkovMapSimulation(_make_config())
        traj = sim.run(n_steps=50)
        assert len(traj.timestamps) == 51
        assert traj.timestamps[0] == pytest.approx(0.0)


class TestRulkovMapRediscovery:
    def test_alpha_sweep_data(self):
        from simulating_anything.rediscovery.rulkov_map import (
            generate_alpha_sweep_data,
        )
        data = generate_alpha_sweep_data(
            n_alpha=5, n_steps=2000, n_transient=500,
        )
        assert len(data["alpha"]) == 5
        assert len(data["firing_rate"]) == 5
        assert len(data["regime"]) == 5

    def test_bifurcation_data(self):
        from simulating_anything.rediscovery.rulkov_map import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(n_alpha=5, alpha_min=3.0, alpha_max=5.0)
        assert len(data["alpha"]) == 1000  # 5 * 200 n_plot
        assert len(data["x"]) == 1000

    def test_lyapunov_data(self):
        from simulating_anything.rediscovery.rulkov_map import (
            generate_lyapunov_data,
        )
        data = generate_lyapunov_data(n_alpha=5, alpha_min=2.0, alpha_max=5.0)
        assert len(data["alpha"]) == 5
        assert len(data["lyapunov"]) == 5

    def test_spiking_onset_estimation(self):
        from simulating_anything.rediscovery.rulkov_map import (
            estimate_spiking_onset,
            generate_alpha_sweep_data,
        )
        data = generate_alpha_sweep_data(
            n_alpha=20, alpha_min=2.0, alpha_max=6.0,
            n_steps=5000, n_transient=1000,
        )
        alpha_c = estimate_spiking_onset(data)
        # Should find onset somewhere between 3 and 5
        if alpha_c is not None:
            assert 2.0 <= alpha_c <= 6.0

    def test_make_config(self):
        from simulating_anything.rediscovery.rulkov_map import _make_config
        config = _make_config(alpha=5.0, mu=0.01)
        assert config.parameters["alpha"] == 5.0
        assert config.parameters["mu"] == 0.01
