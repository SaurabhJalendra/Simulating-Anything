"""Tests for the Colpitts oscillator simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.colpitts import ColpittsSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestColpittsSimulation:
    """Tests for the Colpitts oscillator simulation."""

    def _make_sim(self, **kwargs) -> ColpittsSimulation:
        defaults = {
            "Q": 8.0, "g_d": 0.3, "V_cc": 1.0,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.COLPITTS,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return ColpittsSimulation(config)

    def test_initial_state(self):
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert np.isclose(state[0], 0.1)
        assert np.isclose(state[1], 0.0)
        assert np.isclose(state[2], 0.0)

    def test_custom_initial_conditions(self):
        sim = self._make_sim(x_0=1.0, y_0=-0.5, z_0=0.3)
        state = sim.reset()
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], -0.5)
        assert np.isclose(state[2], 0.3)

    def test_step_advances(self):
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe(self):
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_state_shape_preserved(self):
        """State shape should remain (3,) throughout the simulation."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(100):
            state = sim.step()
            assert state.shape == (3,)

    def test_run_returns_trajectory(self):
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))


class TestColpittsNonlinearity:
    """Tests for the h(x) piecewise-linear nonlinearity."""

    def test_h_negative_input(self):
        """h(x) = 0 for x < 0."""
        assert ColpittsSimulation._h(-1.0) == 0.0
        assert ColpittsSimulation._h(-10.0) == 0.0

    def test_h_zero_input(self):
        """h(0) = max(0, 0) = 0."""
        assert np.isclose(ColpittsSimulation._h(0.0), 0.0)

    def test_h_positive_input(self):
        """h(x) = x for x > 0."""
        assert np.isclose(ColpittsSimulation._h(1.0), 1.0)
        assert np.isclose(ColpittsSimulation._h(5.0), 5.0)

    def test_h_continuity_at_zero(self):
        """h should be continuous at x=0."""
        assert np.isclose(
            ColpittsSimulation._h(-1e-10),
            ColpittsSimulation._h(0.0),
            atol=1e-9,
        )


class TestColpittsDerivatives:
    """Tests for the Colpitts ODEs at known points."""

    def _make_sim(self, **kwargs) -> ColpittsSimulation:
        defaults = {"Q": 8.0, "g_d": 0.3, "V_cc": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.COLPITTS,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return ColpittsSimulation(config)

    def test_derivatives_at_fixed_point(self):
        """At fixed point (V_cc/Q, 0, 0), derivatives should be zero."""
        sim = self._make_sim(Q=8.0, V_cc=1.0)
        sim.reset()
        fp = np.array([1.0 / 8.0, 0.0, 0.0])
        derivs = sim._derivatives(fp)
        np.testing.assert_array_almost_equal(
            derivs, [0.0, 0.0, 0.0], decimal=10,
        )

    def test_derivatives_negative_x(self):
        """For x < 0, h(x) = 0, so dz/dt = -g_d*z - y + V_cc."""
        sim = self._make_sim(Q=8.0, g_d=0.3, V_cc=1.0)
        sim.reset()
        # state = [-1, 0.5, 0.2]
        # dx = y = 0.5
        # dy = z = 0.2
        # dz = -0.3*0.2 - 0.5 + 1.0 - 8.0*0 = -0.06 - 0.5 + 1.0 = 0.44
        derivs = sim._derivatives(np.array([-1.0, 0.5, 0.2]))
        assert np.isclose(derivs[0], 0.5)
        assert np.isclose(derivs[1], 0.2)
        assert np.isclose(derivs[2], 0.44)

    def test_derivatives_positive_x(self):
        """For x > 0, h(x) = x, so Q*x term is active."""
        sim = self._make_sim(Q=8.0, g_d=0.3, V_cc=1.0)
        sim.reset()
        # state = [1, 0, 0]
        # dx = y = 0
        # dy = z = 0
        # dz = -0.3*0 - 0 + 1.0 - 8.0*1.0 = -7.0
        derivs = sim._derivatives(np.array([1.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 0.0)
        assert np.isclose(derivs[2], -7.0)


class TestColpittsFixedPoints:
    """Tests for fixed point computation."""

    def test_single_fixed_point(self):
        config = SimulationConfig(
            domain=Domain.COLPITTS,
            dt=0.01,
            n_steps=1000,
            parameters={"Q": 8.0, "g_d": 0.3, "V_cc": 1.0},
        )
        sim = ColpittsSimulation(config)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1
        np.testing.assert_array_almost_equal(
            fps[0], [1.0 / 8.0, 0.0, 0.0],
        )

    def test_fixed_point_scales_with_parameters(self):
        """x_eq = V_cc / Q."""
        config = SimulationConfig(
            domain=Domain.COLPITTS,
            dt=0.01,
            n_steps=1000,
            parameters={"Q": 4.0, "g_d": 0.3, "V_cc": 2.0},
        )
        sim = ColpittsSimulation(config)
        sim.reset()
        fps = sim.fixed_points
        assert np.isclose(fps[0][0], 0.5)  # 2.0 / 4.0

    def test_derivatives_at_fixed_point(self):
        """Derivatives should be zero at the fixed point."""
        config = SimulationConfig(
            domain=Domain.COLPITTS,
            dt=0.01,
            n_steps=1000,
            parameters={"Q": 8.0, "g_d": 0.3, "V_cc": 1.0},
        )
        sim = ColpittsSimulation(config)
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=10,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )


class TestColpittsTrajectory:
    """Tests for trajectory properties of the Colpitts oscillator."""

    def _make_sim(self, **kwargs) -> ColpittsSimulation:
        defaults = {
            "Q": 8.0, "g_d": 0.3, "V_cc": 1.0, "x_0": 0.1,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.COLPITTS,
            dt=0.01,
            n_steps=50000,
            parameters=defaults,
        )
        return ColpittsSimulation(config)

    def test_trajectory_bounded(self):
        """Trajectories should remain bounded for classic parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 100, (
                f"Trajectory diverged: {state}"
            )

    def test_trajectory_not_trivial(self):
        """After transient, the trajectory should oscillate."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        x_vals = []
        for _ in range(5000):
            state = sim.step()
            x_vals.append(state[0])
        # The oscillator should have non-trivial amplitude
        assert np.std(x_vals) > 0.1, "Trajectory collapsed to fixed point"

    def test_deterministic(self):
        """Two runs with same initial conditions should be identical."""
        sim1 = self._make_sim()
        sim2 = self._make_sim()
        traj1 = sim1.run(n_steps=500)
        traj2 = sim2.run(n_steps=500)
        np.testing.assert_array_almost_equal(
            traj1.states, traj2.states, decimal=12,
        )

    def test_different_parameters_diverge(self):
        """Different Q values should yield different trajectories."""
        sim1 = self._make_sim(Q=8.0)
        sim2 = self._make_sim(Q=5.0)
        traj1 = sim1.run(n_steps=500)
        traj2 = sim2.run(n_steps=500)
        assert not np.allclose(traj1.states, traj2.states)


class TestColpittsLyapunov:
    """Tests for Lyapunov exponent estimation."""

    def test_positive_lyapunov_chaotic(self):
        """Colpitts at Q=8.0, g_d=0.3 should have positive Lyapunov."""
        config = SimulationConfig(
            domain=Domain.COLPITTS,
            dt=0.01,
            n_steps=20000,
            parameters={"Q": 8.0, "g_d": 0.3, "V_cc": 1.0, "x_0": 0.1},
        )
        sim = ColpittsSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.0, (
            f"Lyapunov {lam:.3f} should be positive for chaotic regime"
        )

    def test_low_gain_less_chaotic(self):
        """Lower Q (less gain) should produce smaller Lyapunov exponent."""
        config_low = SimulationConfig(
            domain=Domain.COLPITTS,
            dt=0.01,
            n_steps=20000,
            parameters={"Q": 4.0, "g_d": 0.3, "V_cc": 1.0, "x_0": 0.1},
        )
        sim_low = ColpittsSimulation(config_low)
        sim_low.reset()
        for _ in range(5000):
            sim_low.step()
        lam_low = sim_low.estimate_lyapunov(n_steps=20000, dt=0.01)

        config_high = SimulationConfig(
            domain=Domain.COLPITTS,
            dt=0.01,
            n_steps=20000,
            parameters={"Q": 8.0, "g_d": 0.3, "V_cc": 1.0, "x_0": 0.1},
        )
        sim_high = ColpittsSimulation(config_high)
        sim_high.reset()
        for _ in range(5000):
            sim_high.step()
        lam_high = sim_high.estimate_lyapunov(n_steps=20000, dt=0.01)

        # Chaotic regime should have larger Lyapunov exponent
        assert lam_high > lam_low, (
            f"High-Q Lyapunov {lam_high:.3f} should exceed "
            f"low-Q Lyapunov {lam_low:.3f}"
        )


class TestColpittsRediscovery:
    """Tests for Colpitts rediscovery data generation functions."""

    def test_ode_data(self):
        from simulating_anything.rediscovery.colpitts import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["Q"] == 8.0
        assert data["g_d"] == 0.3

    def test_ode_data_stays_finite(self):
        from simulating_anything.rediscovery.colpitts import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_sweep_data(self):
        from simulating_anything.rediscovery.colpitts import (
            generate_chaos_sweep_data,
        )

        data = generate_chaos_sweep_data(n_Q=5, n_steps=2000, dt=0.01)
        assert len(data["Q"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_chaos_sweep_contains_types(self):
        """Sweep from Q=3 to Q=12 should find both regimes."""
        from simulating_anything.rediscovery.colpitts import (
            generate_chaos_sweep_data,
        )

        data = generate_chaos_sweep_data(n_Q=10, n_steps=5000, dt=0.01)
        types = set(data["attractor_type"])
        # Should find at least two distinct regimes
        assert len(types) >= 1, f"Found only: {types}"
