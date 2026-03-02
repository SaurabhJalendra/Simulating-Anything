"""Tests for the Nose-Hoover thermostat simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.nose_hoover import NoseHooverSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestNoseHooverSimulation:
    """Tests for the Nose-Hoover system simulation basics."""

    def _make_sim(self, **kwargs) -> NoseHooverSimulation:
        defaults = {
            "a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return NoseHooverSimulation(config)

    def test_creation(self):
        """Simulation is created with correct parameters."""
        sim = self._make_sim(a=1.5)
        assert sim.a == 1.5

    def test_default_parameters(self):
        """Default a should be 1.0."""
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = NoseHooverSimulation(config)
        assert sim.a == 1.0

    def test_reset(self):
        """Initial state should match specified initial conditions."""
        sim = self._make_sim(x_0=1.0, y_0=2.0, z_0=-0.5)
        state = sim.reset()
        assert state.shape == (3,)
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], 2.0)
        assert np.isclose(state[2], -0.5)

    def test_observe_shape(self):
        """Observe should return a 3-element array."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_step_advances(self):
        """State should change after a step."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_deterministic(self):
        """Same parameters should produce the same trajectory."""
        sim1 = self._make_sim(a=1.0, x_0=0.0, y_0=5.0, z_0=0.0)
        sim2 = self._make_sim(a=1.0, x_0=0.0, y_0=5.0, z_0=0.0)
        sim1.reset()
        sim2.reset()
        for _ in range(100):
            s1 = sim1.step()
            s2 = sim2.step()
        np.testing.assert_array_almost_equal(s1, s2, decimal=12)

    def test_stability(self):
        """No NaN or Inf after many steps."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State not finite: {state}"

    def test_bounded(self):
        """Trajectory stays bounded for chaotic regime."""
        sim = self._make_sim(a=1.0)
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.linalg.norm(state) < 100, f"Trajectory diverged: {state}"


class TestNoseHooverDerivatives:
    """Tests for the ODE right-hand side."""

    def _make_sim(self, **kwargs) -> NoseHooverSimulation:
        defaults = {"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return NoseHooverSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin with a=1: dx=0, dy=0, dz=a=1."""
        sim = self._make_sim(a=1.0)
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], 1.0)

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point."""
        sim = self._make_sim(a=1.0)
        sim.reset()
        # At state [1, 2, 3]:
        # dx = y = 2
        # dy = -x + y*z = -1 + 2*3 = 5
        # dz = a - y^2 = 1 - 4 = -3
        derivs = sim._derivatives(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(derivs[0], 2.0)
        assert np.isclose(derivs[1], 5.0)
        assert np.isclose(derivs[2], -3.0)

    def test_derivatives_with_different_a(self):
        """dz/dt should equal a - y^2."""
        sim = self._make_sim(a=2.5)
        sim.reset()
        # At [0, 1, 0]: dx=1, dy=0+0=0, dz=2.5-1=1.5
        derivs = sim._derivatives(np.array([0.0, 1.0, 0.0]))
        assert np.isclose(derivs[0], 1.0)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], 1.5)


class TestNoseHooverFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, a=1.0) -> NoseHooverSimulation:
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=1000,
            parameters={"a": a},
        )
        return NoseHooverSimulation(config)

    def test_no_fixed_points_a_nonzero(self):
        """For a != 0, no fixed points exist (y=0 conflicts with dz=a)."""
        sim = self._make_sim(a=1.0)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 0

    def test_fixed_point_a_zero(self):
        """For a=0, the origin is a fixed point."""
        sim = self._make_sim(a=0.0)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) >= 1
        # Origin should be among fixed points
        found_origin = False
        for fp in fps:
            if np.linalg.norm(fp) < 1e-10:
                found_origin = True
                break
        assert found_origin


class TestNoseHooverDivergence:
    """Tests for volume preservation and divergence."""

    def _make_sim(self, **kwargs) -> NoseHooverSimulation:
        defaults = {"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=20000,
            parameters=defaults,
        )
        return NoseHooverSimulation(config)

    def test_divergence_equals_z(self):
        """Pointwise divergence should equal z."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([1.0, 2.0, 3.5])
        assert np.isclose(sim.compute_divergence(state), 3.5)

    def test_divergence_at_origin(self):
        """Divergence at origin is zero."""
        sim = self._make_sim()
        sim.reset()
        assert np.isclose(sim.compute_divergence(np.array([0.0, 0.0, 0.0])), 0.0)

    def test_time_averaged_divergence_near_zero(self):
        """On the attractor, time-averaged divergence should be near zero."""
        sim = self._make_sim(a=1.0)
        sim.reset()
        result = sim.check_volume_preservation(
            n_steps=10000, n_transient=3000
        )
        assert abs(result["mean_divergence"]) < 1.0, (
            f"Mean divergence {result['mean_divergence']:.4f} too far from 0"
        )
        assert result["std_divergence"] > 0, "Divergence std should be positive"


class TestNoseHooverLyapunov:
    """Tests for Lyapunov exponent and chaos detection."""

    def test_chaotic_at_a1(self):
        """At a=1.0 (classic), largest Lyapunov exponent should be positive."""
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=30000,
            parameters={"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
        )
        sim = NoseHooverSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.0, f"Lyapunov {lam:.4f} not positive for chaotic a=1.0"

    def test_lyapunov_finite(self):
        """Lyapunov exponent should be finite."""
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=10000,
            parameters={"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
        )
        sim = NoseHooverSimulation(config)
        sim.reset()
        for _ in range(2000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=5000, dt=0.01)
        assert np.isfinite(lam), f"Lyapunov is not finite: {lam}"

    def test_bifurcation_sweep(self):
        """Sweep should produce valid data for all a values."""
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=5000,
            parameters={"a": 1.0},
        )
        sim = NoseHooverSimulation(config)
        sim.reset()
        a_values = np.linspace(0.5, 2.0, 5)
        data = sim.bifurcation_sweep(
            a_values, n_transient=1000, n_measure=5000
        )
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


class TestNoseHooverThermostat:
    """Tests for thermostat-specific properties."""

    def _make_sim(self, **kwargs) -> NoseHooverSimulation:
        defaults = {"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=30000,
            parameters=defaults,
        )
        return NoseHooverSimulation(config)

    def test_hamiltonian_computation(self):
        """H = (x^2 + y^2)/2 should be computed correctly."""
        sim = self._make_sim()
        sim.reset()
        H = sim.compute_hamiltonian(np.array([3.0, 4.0, 0.0]))
        assert np.isclose(H, 12.5)  # (9 + 16)/2

    def test_temperature_equilibration(self):
        """<y^2> should be close to a on the attractor."""
        sim = self._make_sim(a=1.0)
        sim.reset()
        result = sim.check_temperature_equilibration(
            n_steps=15000, n_transient=5000
        )
        # <y^2> should be roughly near a=1.0
        assert result["mean_y_squared"] > 0, "Mean y^2 should be positive"
        assert result["mean_y_squared"] < 5.0, (
            f"Mean y^2 = {result['mean_y_squared']:.2f} too far from a=1.0"
        )

    def test_temperature_scales_with_a(self):
        """Larger a should give larger <y^2>."""
        sim_low = self._make_sim(a=0.5)
        sim_low.reset()
        res_low = sim_low.check_temperature_equilibration(
            n_steps=10000, n_transient=5000
        )

        sim_high = self._make_sim(a=2.0)
        sim_high.reset()
        res_high = sim_high.check_temperature_equilibration(
            n_steps=10000, n_transient=5000
        )

        assert res_high["mean_y_squared"] > res_low["mean_y_squared"], (
            f"<y^2> at a=2.0 ({res_high['mean_y_squared']:.3f}) should exceed "
            f"a=0.5 ({res_low['mean_y_squared']:.3f})"
        )


class TestNoseHooverNumerics:
    """Tests for numerical accuracy and integration."""

    def test_rk4_convergence(self):
        """Smaller dt should give more accurate results."""
        # Run with dt=0.05
        config1 = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.05,
            n_steps=200,
            parameters={"a": 1.0, "x_0": 0.0, "y_0": 1.0, "z_0": 0.0},
        )
        sim1 = NoseHooverSimulation(config1)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state_coarse = sim1.observe().copy()

        # Run with dt=0.01 (5x finer, same total time)
        config2 = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=1000,
            parameters={"a": 1.0, "x_0": 0.0, "y_0": 1.0, "z_0": 0.0},
        )
        sim2 = NoseHooverSimulation(config2)
        sim2.reset()
        for _ in range(1000):
            sim2.step()
        state_fine = sim2.observe().copy()

        # For RK4, states should be close over 10 time units
        error = np.linalg.norm(state_coarse - state_fine)
        assert error < 0.1, (
            f"RK4 convergence error {error:.6f} too large between dt=0.05 and dt=0.01"
        )

    def test_trajectory_statistics(self):
        """Trajectory statistics should be computable and finite."""
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=10000,
            parameters={"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
        )
        sim = NoseHooverSimulation(config)
        stats = sim.compute_trajectory_statistics(
            n_steps=5000, n_transient=2000
        )
        for key, val in stats.items():
            assert np.isfinite(val), f"Non-finite {key}: {val}"
        # Std should be positive in chaotic regime
        assert stats["x_std"] > 0
        assert stats["y_std"] > 0
        assert stats["z_std"] > 0

    def test_run_trajectory(self):
        """run() should return TrajectoryData with correct shape."""
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=100,
            parameters={"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
        )
        sim = NoseHooverSimulation(config)
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

    def test_period_measurement(self):
        """Period measurement should return a finite value."""
        config = SimulationConfig(
            domain=Domain.NOSE_HOOVER,
            dt=0.01,
            n_steps=30000,
            parameters={"a": 1.0, "x_0": 0.0, "y_0": 5.0, "z_0": 0.0},
        )
        sim = NoseHooverSimulation(config)
        sim.reset()
        T = sim.measure_period(n_transient=3000, n_measure=20000)
        # For chaotic system, period is still measurable (average crossing time)
        assert T > 0, f"Period should be positive, got {T}"


class TestNoseHooverRediscovery:
    """Tests for the Nose-Hoover rediscovery data generation."""

    def test_ode_data(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.nose_hoover import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert np.isclose(data["a"], 1.0)
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_vs_a_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.nose_hoover import (
            generate_lyapunov_vs_a_data,
        )

        data = generate_lyapunov_vs_a_data(n_a=5, n_steps=3000, dt=0.01)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_temperature_data(self):
        """Temperature equilibration data should produce valid output."""
        from simulating_anything.rediscovery.nose_hoover import (
            generate_temperature_data,
        )

        data = generate_temperature_data(
            a_values=np.array([0.5, 1.0, 2.0]),
            n_steps=3000,
            n_transient=1000,
            dt=0.01,
        )
        assert len(data["a"]) == 3
        assert len(data["mean_y_squared"]) == 3
        assert np.all(np.isfinite(data["mean_y_squared"]))
        # All <y^2> should be positive
        assert np.all(data["mean_y_squared"] > 0)
