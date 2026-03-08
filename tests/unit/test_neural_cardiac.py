"""Tests for coupled neural-cardiac oscillator simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.neural_cardiac import (
    NeuralCardiacSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestNeuralCardiacSimulation:

    def _make_sim(self, **kwargs) -> NeuralCardiacSimulation:
        defaults = {
            "coupling_nc": 0.1,
            "coupling_cv": 0.05,
            "I_ext": 0.5,
            "mu_c": 1.0,
            "eps_c": 0.3,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.05, n_steps=200, parameters=defaults
        )
        return NeuralCardiacSimulation(config)

    def test_initial_state_shape(self):
        sim = self._make_sim()
        obs = sim.reset()
        assert obs.shape == (4,)

    def test_step_returns_correct_shape(self):
        sim = self._make_sim()
        sim.reset()
        obs = sim.step()
        assert obs.shape == (4,)

    def test_trajectory_bounded(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        for _ in range(5000):
            obs = sim.step()
            assert np.all(np.isfinite(obs)), f"Non-finite state: {obs}"
            assert np.max(np.abs(obs)) < 10, f"State unbounded: {obs}"

    def test_deterministic_reset(self):
        sim = self._make_sim()
        obs1 = sim.reset(seed=42)
        sim.step()
        sim.step()
        obs2 = sim.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_differ(self):
        sim = self._make_sim()
        obs1 = sim.reset(seed=0)
        obs2 = sim.reset(seed=99)
        assert not np.allclose(obs1, obs2)

    def test_neural_state_accessor(self):
        sim = self._make_sim()
        sim.reset()
        neural = sim.neural_state
        assert neural.shape == (2,)
        np.testing.assert_array_equal(neural, sim.observe()[:2])

    def test_cardiac_state_accessor(self):
        sim = self._make_sim()
        sim.reset()
        cardiac = sim.cardiac_state
        assert cardiac.shape == (2,)
        np.testing.assert_array_equal(cardiac, sim.observe()[2:4])

    def test_zero_coupling_decouples(self):
        sim_coupled = self._make_sim(coupling_nc=0.2, coupling_cv=0.1)
        sim_decoupled = self._make_sim(coupling_nc=0.0, coupling_cv=0.0)
        sim_coupled.reset(seed=0)
        sim_decoupled.reset(seed=0)

        for _ in range(500):
            sim_coupled.step()
            sim_decoupled.step()

        # States should differ when coupling is present
        assert not np.allclose(
            sim_coupled.observe(), sim_decoupled.observe(), atol=0.01
        )

    def test_neural_oscillates(self):
        sim = self._make_sim(I_ext=0.5)
        sim.reset(seed=0)
        v_values = []
        for _ in range(5000):
            obs = sim.step()
            v_values.append(obs[0])
        v = np.array(v_values)
        # Neural v should oscillate with FHN dynamics
        assert v.max() > 0.5
        assert v.min() < -0.5

    def test_cardiac_oscillates(self):
        sim = self._make_sim(mu_c=1.0)
        sim.reset(seed=0)
        x_values = []
        for _ in range(5000):
            obs = sim.step()
            x_values.append(obs[2])
        x = np.array(x_values)
        # Cardiac x should oscillate (VdP dynamics)
        assert x.max() > 0.3
        assert x.min() < -0.3

    def test_coupling_affects_dynamics(self):
        sim_low = self._make_sim(coupling_nc=0.01, coupling_cv=0.005)
        sim_high = self._make_sim(coupling_nc=0.3, coupling_cv=0.15)
        sim_low.reset(seed=0)
        sim_high.reset(seed=0)

        for _ in range(2000):
            sim_low.step()
            sim_high.step()

        obs_low = sim_low.observe()
        obs_high = sim_high.observe()
        # Different coupling should lead to different trajectories
        assert not np.allclose(obs_low, obs_high, atol=0.01)

    def test_phase_ratio_estimation(self):
        sim = self._make_sim()
        ratio = sim.estimate_phase_ratio(n_steps=5000, skip_transient=1000)
        assert np.isfinite(ratio)
        assert ratio > 0

    def test_run_method(self):
        sim = self._make_sim()
        sim.reset()
        traj = sim.run(100)
        assert traj.states.shape[1] == 4
        assert traj.states.shape[0] >= 100

    def test_observe_returns_copy(self):
        sim = self._make_sim()
        sim.reset()
        obs1 = sim.observe()
        obs2 = sim.observe()
        obs1[0] = 999.0
        assert obs2[0] != 999.0


class TestNeuralCardiacRediscovery:

    def test_coupling_sweep_data(self):
        from simulating_anything.rediscovery.neural_cardiac import (
            generate_coupling_sweep_data,
        )
        data = generate_coupling_sweep_data(n_points=5, n_steps=1000, dt=0.1)
        assert len(data["couplings"]) == 5
        assert len(data["correlations"]) == 5
        assert all(-1 <= c <= 1 for c in data["correlations"])

    def test_ode_data_shape(self):
        from simulating_anything.rediscovery.neural_cardiac import (
            generate_ode_data,
        )
        X, dXdt = generate_ode_data(n_steps=200, dt=0.05)
        assert X.shape == (199, 4)
        assert dXdt.shape == (199, 4)
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(dXdt))
