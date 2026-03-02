"""Tests for the stochastic FitzHugh-Nagumo neuron model."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.fhn_stochastic import FHNStochasticSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    current: float = 0.3,
    sigma: float = 0.3,
    dt: float = 0.01,
    n_steps: int = 5000,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.FHN_STOCHASTIC,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        parameters={
            "I": current, "sigma": sigma,
            "a": 0.7, "b": 0.8, "eps": 0.08,
            "v_0": -1.0, "w_0": -0.5,
        },
    )


class TestFHNStochasticCreation:
    """Tests for simulation object creation and parameter handling."""

    def test_creation(self):
        sim = FHNStochasticSimulation(_make_config())
        assert sim.a == 0.7
        assert sim.b_param == 0.8
        assert sim.eps == 0.08
        assert sim.I == 0.3
        assert sim.sigma == 0.3

    def test_default_parameters(self):
        """Default parameters should be set when not provided."""
        config = SimulationConfig(
            domain=Domain.FHN_STOCHASTIC,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = FHNStochasticSimulation(config)
        assert sim.a == 0.7
        assert sim.b_param == 0.8
        assert sim.eps == 0.08
        assert sim.I == 0.3
        assert sim.sigma == 0.3

    def test_custom_parameters(self):
        config = SimulationConfig(
            domain=Domain.FHN_STOCHASTIC,
            dt=0.01,
            n_steps=100,
            parameters={"a": 1.0, "b": 0.5, "eps": 0.1, "I": 0.4, "sigma": 0.5},
        )
        sim = FHNStochasticSimulation(config)
        assert sim.a == 1.0
        assert sim.b_param == 0.5
        assert sim.eps == 0.1
        assert sim.I == 0.4
        assert sim.sigma == 0.5


class TestFHNStochasticBasics:
    """Tests for reset, step, observe, run."""

    def test_reset_returns_initial_state(self):
        sim = FHNStochasticSimulation(_make_config())
        state = sim.reset(seed=42)
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [-1.0, -0.5])

    def test_reset_resets_step_count(self):
        sim = FHNStochasticSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(10):
            sim.step()
        assert sim._step_count == 10
        sim.reset(seed=42)
        assert sim._step_count == 0

    def test_step_advances_state(self):
        sim = FHNStochasticSimulation(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = FHNStochasticSimulation(_make_config())
        state = sim.reset(seed=42)
        observed = sim.observe()
        np.testing.assert_array_equal(state, observed)

    def test_state_shape_is_2d(self):
        sim = FHNStochasticSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            state = sim.step()
            assert state.shape == (2,), f"State shape {state.shape} != (2,)"

    def test_run_returns_trajectory_data(self):
        sim = FHNStochasticSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states is not None
        assert traj.states.shape == (101, 2)
        assert traj.timestamps is not None
        assert len(traj.timestamps) == 101

    def test_run_trajectory_starts_at_initial_state(self):
        sim = FHNStochasticSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        np.testing.assert_allclose(traj.states[0], [-1.0, -0.5])


class TestFHNStochasticDeterministicDerivatives:
    """Tests for the deterministic part of the dynamics."""

    def test_derivatives_at_origin(self):
        sim = FHNStochasticSimulation(_make_config(current=0.0))
        y = np.array([0.0, 0.0])
        dy = sim._derivatives(y)
        # dv = 0 - 0 - 0 + 0 = 0
        # dw = 0.08 * (0 + 0.7 - 0) = 0.056
        np.testing.assert_allclose(dy, [0.0, 0.056], atol=1e-10)

    def test_derivatives_with_current(self):
        sim = FHNStochasticSimulation(_make_config(current=0.5))
        y = np.array([0.0, 0.0])
        dy = sim._derivatives(y)
        # dv = 0 - 0 - 0 + 0.5 = 0.5
        # dw = 0.08 * (0 + 0.7 - 0) = 0.056
        np.testing.assert_allclose(dy, [0.5, 0.056], atol=1e-10)

    def test_derivatives_cubic_nonlinearity(self):
        """Check the v^3/3 term."""
        sim = FHNStochasticSimulation(_make_config(current=0.0))
        y = np.array([1.0, 0.0])
        dy = sim._derivatives(y)
        # dv = 1 - 1/3 - 0 + 0 = 2/3
        np.testing.assert_allclose(dy[0], 2.0 / 3.0, atol=1e-10)


class TestFHNStochasticNoise:
    """Tests for stochastic behavior."""

    def test_zero_noise_subthreshold_no_spikes(self):
        """With zero noise and subthreshold I, no spiking after transient."""
        config = _make_config(current=0.3, sigma=0.0, n_steps=10000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=10000)
        # Skip initial transient (first 2000 steps) where the trajectory
        # settles from the initial condition to the stable fixed point
        v_trace = traj.states[2000:, 0]
        n_spikes = sim.count_spikes(v_trace)
        assert n_spikes == 0, f"Expected 0 spikes with sigma=0, I=0.3, got {n_spikes}"

    def test_zero_noise_suprathreshold_oscillates(self):
        """With zero noise and suprathreshold I, regular oscillation."""
        config = _make_config(current=0.5, sigma=0.0, dt=0.01, n_steps=20000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=20000)
        v_trace = traj.states[:, 0]
        # Skip transient (first 5000 steps)
        v_after = v_trace[5000:]
        v_range = np.max(v_after) - np.min(v_after)
        assert v_range > 0.5, f"No oscillation at I=0.5, sigma=0: range={v_range}"

    def test_moderate_noise_induces_spikes(self):
        """With moderate noise and subthreshold I, noise-induced spiking."""
        config = _make_config(current=0.3, sigma=0.3, n_steps=50000, seed=42)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=50000)
        v_trace = traj.states[:, 0]
        n_spikes = sim.count_spikes(v_trace)
        assert n_spikes > 0, (
            "Expected noise-induced spikes with sigma=0.3, I=0.3"
        )

    def test_large_noise_produces_spikes(self):
        """Very large noise should produce many spikes."""
        config = _make_config(current=0.3, sigma=1.0, n_steps=20000, seed=42)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=20000)
        v_trace = traj.states[:, 0]
        n_spikes = sim.count_spikes(v_trace)
        assert n_spikes > 0, "Expected spikes with large noise sigma=1.0"

    def test_different_seeds_different_trajectories(self):
        """Different seeds should produce different stochastic trajectories."""
        config1 = _make_config(sigma=0.3, n_steps=1000, seed=42)
        sim1 = FHNStochasticSimulation(config1)
        traj1 = sim1.run(n_steps=1000)

        config2 = _make_config(sigma=0.3, n_steps=1000, seed=99)
        sim2 = FHNStochasticSimulation(config2)
        traj2 = sim2.run(n_steps=1000)

        assert not np.allclose(traj1.states, traj2.states), (
            "Different seeds should produce different trajectories"
        )

    def test_same_seed_same_trajectory(self):
        """Same seed should produce identical trajectories (reproducible)."""
        config = _make_config(sigma=0.3, n_steps=500, seed=42)

        sim1 = FHNStochasticSimulation(config)
        traj1 = sim1.run(n_steps=500)

        sim2 = FHNStochasticSimulation(config)
        traj2 = sim2.run(n_steps=500)

        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_zero_noise_is_deterministic(self):
        """With sigma=0, trajectories should be deterministic regardless of seed."""
        config1 = _make_config(sigma=0.0, n_steps=500, seed=42)
        sim1 = FHNStochasticSimulation(config1)
        traj1 = sim1.run(n_steps=500)

        config2 = _make_config(sigma=0.0, n_steps=500, seed=99)
        sim2 = FHNStochasticSimulation(config2)
        traj2 = sim2.run(n_steps=500)

        np.testing.assert_allclose(traj1.states, traj2.states, atol=1e-12)


class TestFHNStochasticBoundedness:
    """Tests that trajectories remain bounded."""

    def test_trajectory_bounded(self):
        """State should not diverge for reasonable parameters."""
        config = _make_config(current=0.3, sigma=0.3, dt=0.01, n_steps=10000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=10000)
        assert np.all(np.abs(traj.states[:, 0]) < 10), "v diverged"
        assert np.all(np.abs(traj.states[:, 1]) < 10), "w diverged"

    def test_no_nans(self):
        """State should not contain NaN values."""
        config = _make_config(current=0.3, sigma=0.5, dt=0.01, n_steps=5000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=5000)
        assert not np.any(np.isnan(traj.states)), "NaN in trajectory"

    def test_bounded_with_large_noise(self):
        """Even with large noise, state should not blow up."""
        config = _make_config(current=0.3, sigma=2.0, dt=0.01, n_steps=5000, seed=42)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=5000)
        assert np.all(np.isfinite(traj.states)), "Non-finite values in trajectory"
        # The cubic nonlinearity provides restoring force
        assert np.all(np.abs(traj.states[:, 0]) < 50), "v too large"


class TestFHNStochasticSpikeAnalysis:
    """Tests for spike counting and ISI analysis methods."""

    def test_count_spikes_no_spikes(self):
        """No spikes when sigma=0 and I is subthreshold (after transient)."""
        config = _make_config(current=0.3, sigma=0.0, n_steps=5000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=5000)
        # Skip transient from initial condition settling to fixed point
        v_after_transient = traj.states[2000:, 0]
        n_spikes = sim.count_spikes(v_after_transient)
        assert n_spikes == 0

    def test_count_spikes_suprathreshold(self):
        """Suprathreshold I without noise should show multiple spikes."""
        config = _make_config(current=0.5, sigma=0.0, dt=0.01, n_steps=20000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=20000)
        n_spikes = sim.count_spikes(traj.states[:, 0])
        assert n_spikes > 2, f"Expected multiple spikes, got {n_spikes}"

    def test_count_spikes_with_custom_threshold(self):
        """Spike count should change with threshold."""
        config = _make_config(current=0.5, sigma=0.0, dt=0.01, n_steps=20000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=20000)
        v = traj.states[:, 0]

        spikes_low = sim.count_spikes(v, threshold=-0.5)
        spikes_high = sim.count_spikes(v, threshold=0.5)
        # Lower threshold should detect at least as many crossings
        assert spikes_low >= spikes_high

    def test_interspike_intervals_empty_for_no_spikes(self):
        config = _make_config(current=0.3, sigma=0.0, n_steps=5000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=5000)
        # Skip transient
        isis = sim.interspike_intervals(traj.states[2000:, 0])
        assert len(isis) == 0

    def test_interspike_intervals_positive(self):
        """ISI values should all be positive."""
        config = _make_config(current=0.5, sigma=0.0, dt=0.01, n_steps=20000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=20000)
        isis = sim.interspike_intervals(traj.states[:, 0])
        if len(isis) > 0:
            assert np.all(isis > 0), "ISI must be positive"

    def test_mean_isi_zero_for_no_spikes(self):
        config = _make_config(current=0.3, sigma=0.0, n_steps=5000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=5000)
        # Skip transient
        mean_isi = sim.mean_interspike_interval(traj.states[2000:, 0])
        assert mean_isi == 0.0

    def test_mean_isi_positive_for_oscillation(self):
        config = _make_config(current=0.5, sigma=0.0, dt=0.01, n_steps=20000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=20000)
        mean_isi = sim.mean_interspike_interval(traj.states[:, 0])
        assert mean_isi > 0, f"Expected positive mean ISI, got {mean_isi}"

    def test_cv_inf_for_no_spikes(self):
        config = _make_config(current=0.3, sigma=0.0, n_steps=5000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=5000)
        # Skip transient
        cv = sim.coefficient_of_variation(traj.states[2000:, 0])
        assert cv == float("inf")

    def test_cv_low_for_regular_oscillation(self):
        """Deterministic oscillation should have very low CV."""
        config = _make_config(current=0.5, sigma=0.0, dt=0.01, n_steps=40000)
        sim = FHNStochasticSimulation(config)
        traj = sim.run(n_steps=40000)
        cv = sim.coefficient_of_variation(traj.states[:, 0])
        if np.isfinite(cv):
            assert cv < 0.1, f"Expected low CV for regular oscillation, got {cv}"

    def test_cv_higher_with_noise(self):
        """Adding noise should generally increase CV compared to deterministic."""
        # Deterministic
        config_det = _make_config(current=0.5, sigma=0.0, dt=0.01, n_steps=40000)
        sim_det = FHNStochasticSimulation(config_det)
        traj_det = sim_det.run(n_steps=40000)
        cv_det = sim_det.coefficient_of_variation(traj_det.states[:, 0])

        # Noisy
        config_noisy = _make_config(current=0.5, sigma=0.2, dt=0.01, n_steps=40000, seed=42)
        sim_noisy = FHNStochasticSimulation(config_noisy)
        traj_noisy = sim_noisy.run(n_steps=40000)
        cv_noisy = sim_noisy.coefficient_of_variation(traj_noisy.states[:, 0])

        if np.isfinite(cv_det) and np.isfinite(cv_noisy):
            # Noisy CV should be higher (or at least comparable)
            assert cv_noisy >= cv_det * 0.5, (
                f"Noisy CV={cv_noisy} should not be much less than det CV={cv_det}"
            )


class TestFHNStochasticNullclines:
    """Tests for nullcline properties."""

    def test_v_nullcline(self):
        sim = FHNStochasticSimulation(_make_config(current=0.0))
        w_val = sim.nullcline_v(0.0)
        # w = 0 - 0 + 0 = 0
        assert abs(w_val) < 1e-10

    def test_w_nullcline(self):
        sim = FHNStochasticSimulation(_make_config(current=0.0))
        w_val = sim.nullcline_w(0.0)
        # w = (0 + 0.7) / 0.8 = 0.875
        np.testing.assert_allclose(w_val, 0.875, atol=1e-10)


class TestFHNStochasticRediscovery:
    """Tests for rediscovery data generation."""

    def test_noise_sweep_data_generation(self):
        from simulating_anything.rediscovery.fhn_stochastic import (
            generate_noise_sweep_data,
        )
        data = generate_noise_sweep_data(
            n_sigma=5,
            sigma_min=0.1,
            sigma_max=0.5,
            n_steps=2000,
            n_trials=1,
            seed=42,
        )
        assert len(data["sigma"]) == 5
        assert len(data["spike_count"]) == 5
        assert len(data["mean_isi"]) == 5
        assert len(data["cv"]) == 5
        assert np.all(data["spike_count"] >= 0)

    def test_noise_sweep_sigma_range(self):
        from simulating_anything.rediscovery.fhn_stochastic import (
            generate_noise_sweep_data,
        )
        data = generate_noise_sweep_data(
            n_sigma=3,
            sigma_min=0.1,
            sigma_max=0.9,
            n_steps=1000,
            n_trials=1,
            seed=42,
        )
        np.testing.assert_allclose(data["sigma"][0], 0.1)
        np.testing.assert_allclose(data["sigma"][-1], 0.9)

    def test_subthreshold_vs_suprathreshold(self):
        from simulating_anything.rediscovery.fhn_stochastic import (
            generate_subthreshold_vs_suprathreshold,
        )
        results = generate_subthreshold_vs_suprathreshold(
            n_steps=5000, dt=0.01, sigma=0.3, seed=42,
        )
        assert "subthreshold" in results
        assert "suprathreshold" in results
        assert results["subthreshold"]["I"] == 0.3
        assert results["suprathreshold"]["I"] == 0.5
        assert results["subthreshold"]["sigma"] == 0.3
        assert results["suprathreshold"]["sigma"] == 0.3

    def test_rediscovery_runner_returns_dict(self):
        from simulating_anything.rediscovery.fhn_stochastic import (
            run_fhn_stochastic_rediscovery,
        )
        results = run_fhn_stochastic_rediscovery(
            output_dir="output/rediscovery/fhn_stochastic_test",
            n_sigma=3,
            n_trials=1,
        )
        assert isinstance(results, dict)
        assert results["domain"] == "fhn_stochastic"
        assert "targets" in results

    def test_spike_count_default_trace(self):
        """count_spikes with no argument should run a trajectory internally."""
        config = _make_config(current=0.5, sigma=0.0, dt=0.01, n_steps=10000)
        sim = FHNStochasticSimulation(config)
        sim.reset(seed=42)
        # Calling without v_trace runs internally
        n_spikes = sim.count_spikes()
        assert isinstance(n_spikes, int)
        assert n_spikes >= 0
