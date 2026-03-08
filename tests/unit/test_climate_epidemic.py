"""Tests for coupled climate-epidemic simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.climate_epidemic import (
    ClimateEpidemicSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestClimateEpidemicSimulation:

    def _make_sim(self, **kwargs) -> ClimateEpidemicSimulation:
        defaults = {
            "coupling_TC": 0.3,
            "beta_0": 0.5,
            "gamma_epi": 0.1,
            "mu_pop": 0.01,
            "mu_c": 0.5,
            "omega": 2.0,
            "T_0": 0.5,
            "S_0": 0.95,
            "I_0": 0.04,
            "R_0_init": 0.01,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CUSTOM, dt=0.05, n_steps=200, parameters=defaults
        )
        return ClimateEpidemicSimulation(config)

    def test_initial_state_shape(self):
        sim = self._make_sim()
        obs = sim.reset()
        assert obs.shape == (6,)

    def test_step_returns_correct_shape(self):
        sim = self._make_sim()
        sim.reset()
        obs = sim.step()
        assert obs.shape == (6,)

    def test_sir_sums_to_one(self):
        sim = self._make_sim()
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
        obs = sim.observe()
        sir_sum = obs[3] + obs[4] + obs[5]
        assert abs(sir_sum - 1.0) < 1e-6

    def test_sir_stays_positive(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        for _ in range(1000):
            obs = sim.step()
            assert np.all(obs[3:6] >= 0), f"Negative SIR values: {obs[3:6]}"

    def test_trajectory_bounded(self):
        sim = self._make_sim()
        sim.reset(seed=0)
        for _ in range(2000):
            obs = sim.step()
            assert np.all(np.isfinite(obs)), f"Non-finite state: {obs}"
            assert np.max(np.abs(obs[:3])) < 100, f"Climate unbounded: {obs[:3]}"

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

    def test_climate_state_accessor(self):
        sim = self._make_sim()
        sim.reset()
        climate = sim.climate_state
        assert climate.shape == (3,)
        np.testing.assert_array_equal(climate, sim.observe()[:3])

    def test_epidemic_state_accessor(self):
        sim = self._make_sim()
        sim.reset()
        epi = sim.epidemic_state
        assert epi.shape == (3,)
        np.testing.assert_array_equal(epi, sim.observe()[3:6])

    def test_effective_r0(self):
        sim = self._make_sim(coupling_TC=0.0)
        sim.reset()
        # Without coupling, R0 = beta_0 / gamma_epi = 0.5 / 0.1 = 5.0
        assert abs(sim.effective_r0() - 5.0) < 0.01

    def test_coupling_modulates_r0(self):
        sim = self._make_sim(coupling_TC=0.5, T_0=1.0)
        sim.reset()
        # With T=1.0 and coupling=0.5: beta_eff = 0.5*(1+0.5*1.0) = 0.75
        # R0_eff = 0.75 / 0.1 = 7.5
        assert sim.effective_r0() > 5.0

    def test_zero_coupling_decouples(self):
        sim_coupled = self._make_sim(coupling_TC=0.5)
        sim_decoupled = self._make_sim(coupling_TC=0.0, coupling_TI=0.0)
        sim_coupled.reset(seed=0)
        sim_decoupled.reset(seed=0)

        for _ in range(500):
            sim_coupled.step()
            sim_decoupled.step()

        # Epidemic states should differ when coupling is present
        assert not np.allclose(
            sim_coupled.observe()[3:6],
            sim_decoupled.observe()[3:6],
            atol=0.001,
        )

    def test_epidemic_outbreak_occurs(self):
        sim = self._make_sim(beta_0=0.5, gamma_epi=0.1, S_0=0.99, I_0=0.01)
        sim.reset(seed=0)
        max_I = 0
        for _ in range(2000):
            obs = sim.step()
            max_I = max(max_I, obs[4])
        # With R0=5, should see substantial epidemic
        assert max_I > 0.1, f"Peak infected too low: {max_I}"

    def test_run_method(self):
        sim = self._make_sim()
        sim.reset()
        traj = sim.run(100)
        assert traj.states.shape[1] == 6
        assert traj.states.shape[0] >= 100


class TestClimateEpidemicRediscovery:

    def test_coupling_sweep_data(self):
        from simulating_anything.rediscovery.climate_epidemic import (
            generate_coupling_sweep_data,
        )
        data = generate_coupling_sweep_data(n_points=5, n_steps=500, dt=0.1)
        assert len(data["couplings"]) == 5
        assert len(data["peak_infected"]) == 5
        assert all(0 <= p <= 1 for p in data["peak_infected"])

    def test_ode_data_shape(self):
        from simulating_anything.rediscovery.climate_epidemic import (
            generate_ode_data,
        )
        X, dXdt = generate_ode_data(n_steps=200, dt=0.05)
        assert X.shape == (199, 6)
        assert dXdt.shape == (199, 6)
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(dXdt))

    def test_synchronization_data(self):
        from simulating_anything.rediscovery.climate_epidemic import (
            generate_synchronization_data,
        )
        data = generate_synchronization_data(
            n_coupling_points=5, n_steps=1000, dt=0.1
        )
        assert len(data["corr_T_I"]) == 5
        assert all(-1 <= c <= 1 for c in data["corr_T_I"])
