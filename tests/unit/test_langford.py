"""Tests for the Langford system simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.langford import LangfordSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestLangfordSimulation:
    """Tests for the Langford system simulation."""

    def _make_sim(self, **kwargs) -> LangfordSimulation:
        defaults = {
            "a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5,
            "e": 0.25, "f": 0.1,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LANGFORD,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return LangfordSimulation(config)

    def test_initial_state(self):
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert np.isclose(state[0], 0.1)
        assert np.isclose(state[1], 0.0)
        assert np.isclose(state[2], 0.0)

    def test_custom_initial_conditions(self):
        sim = self._make_sim(x_0=2.0, y_0=-1.5, z_0=3.0)
        state = sim.reset()
        assert np.isclose(state[0], 2.0)
        assert np.isclose(state[1], -1.5)
        assert np.isclose(state[2], 3.0)

    def test_step_advances(self):
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_matches_state(self):
        sim = self._make_sim()
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (3,)
        np.testing.assert_array_equal(obs, sim._state)

    def test_derivatives_at_origin(self):
        """Test derivatives at the origin [0, 0, 0]."""
        sim = self._make_sim()
        sim.reset()
        # At [0, 0, 0]:
        # dx = (0 - 0.7)*0 - 3.5*0 = 0
        # dy = 3.5*0 + (0 - 0.7)*0 = 0
        # dz = 0.6 + 0.95*0 - 0/3 - 0*(1+0) + 0 = 0.6
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], 0.6)

    def test_derivatives_known_values(self):
        """Test derivatives at a known non-trivial point."""
        sim = self._make_sim(a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.0)
        sim.reset()
        # At [1, 0, 1] with f=0:
        # r2 = 1
        # dx = (1 - 0.7)*1 - 3.5*0 = 0.3
        # dy = 3.5*1 + (1 - 0.7)*0 = 3.5
        # dz = 0.6 + 0.95*1 - 1/3 - 1*(1 + 0.25) = -0.0333...
        derivs = sim._derivatives(np.array([1.0, 0.0, 1.0]))
        expected_dx = 0.3
        expected_dy = 3.5
        expected_dz = 0.6 + 0.95 - 1.0 / 3.0 - 1.0 * (1.0 + 0.25)
        assert np.isclose(derivs[0], expected_dx, atol=1e-10)
        assert np.isclose(derivs[1], expected_dy, atol=1e-10)
        assert np.isclose(derivs[2], expected_dz, atol=1e-10)

    def test_derivatives_with_f_term(self):
        """Test that the f*z*x^3 term is computed correctly."""
        sim = self._make_sim(a=0.0, b=0.0, c=0.0, d=0.0, e=0.0, f=1.0)
        sim.reset()
        # At [2, 0, 3]:
        # dx = (3 - 0)*2 - 0 = 6
        # dy = 0 + (3 - 0)*0 = 0
        # dz = 0 + 0 - 27/3 - 4*(1+0) + 1.0*3*8 = -9 - 4 + 24 = 11
        derivs = sim._derivatives(np.array([2.0, 0.0, 3.0]))
        assert np.isclose(derivs[0], 6.0)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], 11.0)

    def test_trajectory_bounded(self):
        """Langford trajectories should remain bounded."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, f"Trajectory diverged: {state}"

    def test_trajectory_finite_various_a(self):
        """Trajectory should stay finite for a range of a values."""
        for a in [0.5, 0.7, 0.95, 1.2]:
            sim = self._make_sim(a=a)
            sim.reset()
            for _ in range(5000):
                state = sim.step()
            assert np.all(np.isfinite(state)), f"NaN/Inf at a={a}"

    def test_run_trajectory(self):
        """Test full trajectory collection via run()."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

    def test_default_parameters(self):
        """Default parameters should match specification."""
        config = SimulationConfig(
            domain=Domain.LANGFORD,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = LangfordSimulation(config)
        assert sim.a == 0.95
        assert sim.b == 0.7
        assert sim.c == 0.6
        assert sim.d == 3.5
        assert sim.e == 0.25
        assert sim.f == 0.1

    def test_simplified_form_f_zero(self):
        """With f=0, the system should be the simplified Langford form."""
        sim = self._make_sim(f=0.0)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        # r2 = 2, dz = 0.6 + 0.95 - 1/3 - 2*(1 + 0.25) + 0
        expected_dz = 0.6 + 0.95 - 1.0 / 3.0 - 2.0 * (1.0 + 0.25)
        assert np.isclose(derivs[2], expected_dz, atol=1e-10)

    def test_deterministic(self):
        """Two identical simulations should produce the same trajectory."""
        sim1 = self._make_sim()
        sim2 = self._make_sim()
        traj1 = sim1.run(n_steps=200)
        traj2 = sim2.run(n_steps=200)
        np.testing.assert_array_equal(traj1.states, traj2.states)


class TestLangfordLyapunov:
    """Tests for Lyapunov exponent estimation."""

    def test_lyapunov_finite(self):
        """Lyapunov exponent should be finite."""
        config = SimulationConfig(
            domain=Domain.LANGFORD,
            dt=0.01,
            n_steps=5000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5,
                "e": 0.25, "f": 0.1,
            },
        )
        sim = LangfordSimulation(config)
        sim.reset()
        lam = sim.estimate_lyapunov(n_steps=5000, dt=0.01)
        assert np.isfinite(lam)

    def test_lyapunov_bounded(self):
        """Lyapunov exponent should not be unreasonably large."""
        config = SimulationConfig(
            domain=Domain.LANGFORD,
            dt=0.01,
            n_steps=10000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5,
                "e": 0.25, "f": 0.1,
            },
        )
        sim = LangfordSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=10000, dt=0.01)
        assert abs(lam) < 50.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_lyapunov_stable_regime(self):
        """For very large b the system should be stable (negative Lyapunov)."""
        config = SimulationConfig(
            domain=Domain.LANGFORD,
            dt=0.01,
            n_steps=10000,
            parameters={
                "a": 0.5, "b": 5.0, "c": 0.6, "d": 3.5,
                "e": 0.25, "f": 0.0,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        )
        sim = LangfordSimulation(config)
        sim.reset()
        for _ in range(2000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=5000, dt=0.01)
        assert lam < 1.0, f"Expected small Lyapunov for stable, got {lam}"


class TestLangfordRadiusHistory:
    """Tests for radius computation."""

    def test_radius_history_shape(self):
        sim = LangfordSimulation(SimulationConfig(
            domain=Domain.LANGFORD, dt=0.01, n_steps=5000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5,
                "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        ))
        radii = sim.compute_radius_history(n_steps=1000, n_transient=500)
        assert radii.shape == (1000,)
        assert np.all(radii >= 0)
        assert np.all(np.isfinite(radii))

    def test_radius_nonnegative(self):
        """Radius should always be non-negative."""
        sim = LangfordSimulation(SimulationConfig(
            domain=Domain.LANGFORD, dt=0.01, n_steps=5000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5,
                "e": 0.25, "f": 0.1,
                "x_0": 1.0, "y_0": 1.0, "z_0": 0.0,
            },
        ))
        radii = sim.compute_radius_history(n_steps=500, n_transient=200)
        assert np.all(radii >= 0)


class TestLangfordFrequencySpectrum:
    """Tests for frequency spectrum computation."""

    def test_spectrum_shape(self):
        sim = LangfordSimulation(SimulationConfig(
            domain=Domain.LANGFORD, dt=0.01, n_steps=10000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5,
                "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        ))
        freqs, power = sim.compute_frequency_spectrum(
            n_steps=4096, n_transient=2000,
        )
        assert len(freqs) == len(power)
        assert len(freqs) == 4096 // 2 + 1
        assert np.all(power >= 0)
        assert np.all(np.isfinite(power))

    def test_spectrum_has_peaks(self):
        """An oscillating system should have non-trivial spectral peaks."""
        sim = LangfordSimulation(SimulationConfig(
            domain=Domain.LANGFORD, dt=0.01, n_steps=10000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5,
                "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        ))
        freqs, power = sim.compute_frequency_spectrum(
            n_steps=4096, n_transient=3000,
        )
        assert np.max(power[1:]) > 0, "Power spectrum is all zeros"


class TestLangfordTrajectoryStatistics:
    """Tests for trajectory statistics computation."""

    def test_stats_keys(self):
        """Statistics should contain all expected keys."""
        sim = LangfordSimulation(SimulationConfig(
            domain=Domain.LANGFORD, dt=0.01, n_steps=5000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5,
                "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        ))
        stats = sim.compute_trajectory_statistics(
            n_steps=2000, n_transient=500,
        )
        expected_keys = [
            "x_mean", "y_mean", "z_mean",
            "x_std", "y_std", "z_std",
            "r_mean", "r_std", "r_max",
            "z_min", "z_max",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"
            assert np.isfinite(stats[key]), f"Non-finite value for {key}"

    def test_stats_radius_nonnegative(self):
        """Mean radius should be non-negative."""
        sim = LangfordSimulation(SimulationConfig(
            domain=Domain.LANGFORD, dt=0.01, n_steps=5000,
            parameters={
                "a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5,
                "e": 0.25, "f": 0.1,
                "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
            },
        ))
        stats = sim.compute_trajectory_statistics(
            n_steps=2000, n_transient=1000,
        )
        assert stats["r_mean"] >= 0


class TestLangfordRediscovery:
    """Tests for Langford data generation functions."""

    def test_ode_data(self):
        from simulating_anything.rediscovery.langford import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 0.95
        assert data["b"] == 0.7

    def test_ode_data_stays_finite(self):
        from simulating_anything.rediscovery.langford import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_frequency_data(self):
        from simulating_anything.rediscovery.langford import (
            generate_frequency_data,
        )

        data = generate_frequency_data(n_steps=1024, dt=0.01)
        assert "frequencies" in data
        assert "power" in data
        assert "n_peaks" in data
        assert len(data["frequencies"]) == len(data["power"])

    def test_bifurcation_sweep_data(self):
        from simulating_anything.rediscovery.langford import (
            generate_bifurcation_sweep_data,
        )

        data = generate_bifurcation_sweep_data(
            param_name="b", n_values=5, n_steps=2000, dt=0.01,
        )
        assert len(data["param_values"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["max_radius"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_torus_detection_function(self):
        from simulating_anything.rediscovery.langford import detect_torus

        # Create a simple synthetic spectrum with two peaks
        freqs = np.linspace(0, 50, 1001)
        power = np.zeros_like(freqs)
        power[100] = 10.0
        power[171] = 5.0
        result = detect_torus(freqs, power)
        assert "is_torus" in result
