"""Tests for Chua's circuit simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.chua import ChuaCircuit
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestChuaSimulation:
    """Tests for Chua's circuit simulation."""

    def _make_sim(self, **kwargs) -> ChuaCircuit:
        defaults = {
            "alpha": 15.6, "beta": 28.0,
            "m0": -1.143, "m1": -0.714,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CHUA,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return ChuaCircuit(config)

    def test_creation(self):
        """Simulation is created with correct parameters."""
        sim = self._make_sim()
        assert sim.alpha == 15.6
        assert sim.beta == 28.0
        assert sim.m0 == -1.143
        assert sim.m1 == -0.714

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=0.5, y_0=-0.3, z_0=0.1)
        state = sim.reset()
        assert np.isclose(state[0], 0.5)
        assert np.isclose(state[1], -0.3)
        assert np.isclose(state[2], 0.1)

    def test_step_advances_state(self):
        """A single step changes the state."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_current_state(self):
        """observe() returns current state with correct shape."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)


class TestChuaDiode:
    """Tests for Chua's diode piecewise-linear function."""

    def _make_sim(self, **kwargs) -> ChuaCircuit:
        defaults = {
            "alpha": 15.6, "beta": 28.0,
            "m0": -1.143, "m1": -0.714,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CHUA,
            dt=0.001,
            n_steps=1000,
            parameters=defaults,
        )
        return ChuaCircuit(config)

    def test_inner_region(self):
        """For |x| < 1, f(x) = m1*x."""
        sim = self._make_sim()
        for x in [0.0, 0.5, -0.5, 0.99, -0.99]:
            assert np.isclose(sim.chua_diode(x), sim.m1 * x), (
                f"f({x}) = {sim.chua_diode(x)}, expected {sim.m1 * x}"
            )

    def test_outer_region_positive(self):
        """For x >= 1, f(x) = m0*x + (m1-m0)."""
        sim = self._make_sim()
        for x in [1.0, 2.0, 5.0, 10.0]:
            expected = sim.m0 * x + (sim.m1 - sim.m0)
            assert np.isclose(sim.chua_diode(x), expected, atol=1e-12), (
                f"f({x}) = {sim.chua_diode(x)}, expected {expected}"
            )

    def test_outer_region_negative(self):
        """For x <= -1, f(x) = m0*x - (m1-m0)."""
        sim = self._make_sim()
        for x in [-1.0, -2.0, -5.0, -10.0]:
            expected = sim.m0 * x - (sim.m1 - sim.m0)
            assert np.isclose(sim.chua_diode(x), expected, atol=1e-12), (
                f"f({x}) = {sim.chua_diode(x)}, expected {expected}"
            )

    def test_continuity_at_breakpoints(self):
        """f(x) should be continuous at x = +/- 1."""
        sim = self._make_sim()
        # Left limit at x=1: m1*1 = m1
        # Right limit at x=1: m0*1 + (m1-m0) = m1
        f_at_1 = sim.chua_diode(1.0)
        f_just_below = sim.chua_diode(1.0 - 1e-10)
        assert np.isclose(f_at_1, f_just_below, atol=1e-6)

        f_at_neg1 = sim.chua_diode(-1.0)
        f_just_above = sim.chua_diode(-1.0 + 1e-10)
        assert np.isclose(f_at_neg1, f_just_above, atol=1e-6)

    def test_odd_symmetry(self):
        """f(-x) = -f(x) for the symmetric diode function."""
        sim = self._make_sim()
        for x in [0.0, 0.5, 1.0, 2.0, 5.0]:
            assert np.isclose(sim.chua_diode(-x), -sim.chua_diode(x), atol=1e-12)


class TestChuaDerivatives:
    """Tests for derivatives computation."""

    def _make_sim(self, **kwargs) -> ChuaCircuit:
        defaults = {
            "alpha": 15.6, "beta": 28.0,
            "m0": -1.143, "m1": -0.714,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CHUA,
            dt=0.001,
            n_steps=1000,
            parameters=defaults,
        )
        return ChuaCircuit(config)

    def test_derivatives_at_origin(self):
        """At origin: f(0)=0, so dx=alpha*(0-0-0)=0, dy=0-0+0=0, dz=-beta*0=0."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific point in the inner region."""
        sim = self._make_sim(alpha=15.6, beta=28.0, m0=-1.143, m1=-0.714)
        sim.reset()
        # At state [0.5, 0.5, 0.5] (inner region, |x|<1):
        # f(0.5) = m1 * 0.5 = -0.714 * 0.5 = -0.357
        # dx = 15.6 * (0.5 - 0.5 - (-0.357)) = 15.6 * 0.357 = 5.5692
        # dy = 0.5 - 0.5 + 0.5 = 0.5
        # dz = -28.0 * 0.5 = -14.0
        derivs = sim._derivatives(np.array([0.5, 0.5, 0.5]))
        assert np.isclose(derivs[0], 15.6 * 0.357, atol=1e-10)
        assert np.isclose(derivs[1], 0.5)
        assert np.isclose(derivs[2], -14.0)


class TestChuaFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> ChuaCircuit:
        defaults = {
            "alpha": 15.6, "beta": 28.0,
            "m0": -1.143, "m1": -0.714,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CHUA,
            dt=0.001,
            n_steps=1000,
            parameters=defaults,
        )
        return ChuaCircuit(config)

    def test_three_fixed_points(self):
        """Standard parameters should give exactly three fixed points."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3

    def test_origin_is_fixed_point(self):
        """The origin should always be a fixed point."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 0.0])

    def test_symmetric_nonzero_fixed_points(self):
        """The two non-origin fixed points should be symmetric about origin."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3
        # FP1 and FP2 should be negatives of each other
        np.testing.assert_array_almost_equal(fps[1], -fps[2])

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be zero at each fixed point."""
        sim = self._make_sim()
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=10,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )

    def test_fixed_point_values(self):
        """Check analytical fixed point locations."""
        m0, m1 = -1.143, -0.714
        sim = self._make_sim(m0=m0, m1=m1)
        sim.reset()
        fps = sim.fixed_points
        # x_pos = (m0 - m1) / (1 + m0)
        x_expected = (m0 - m1) / (1.0 + m0)
        assert np.isclose(fps[1][0], x_expected)
        assert np.isclose(fps[1][1], 0.0)
        assert np.isclose(fps[1][2], -x_expected)


class TestChuaTrajectory:
    """Tests for trajectory behavior."""

    def _make_sim(self, **kwargs) -> ChuaCircuit:
        defaults = {
            "alpha": 15.6, "beta": 28.0,
            "m0": -1.143, "m1": -0.714,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CHUA,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return ChuaCircuit(config)

    def test_trajectory_stays_bounded(self):
        """Chua trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 100, f"Trajectory diverged: {state}"

    def test_trajectory_shape_from_run(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))

    def test_double_scroll_visits_both_lobes(self):
        """In the chaotic regime, trajectory should visit both sides of x=0."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        x_vals = []
        for _ in range(20000):
            state = sim.step()
            x_vals.append(state[0])
        x_arr = np.array(x_vals)
        has_positive = np.any(x_arr > 0.5)
        has_negative = np.any(x_arr < -0.5)
        assert has_positive, "Trajectory never visited positive lobe"
        assert has_negative, "Trajectory never visited negative lobe"

    def test_positive_lyapunov_chaotic(self):
        """At standard chaotic parameters, Lyapunov exponent should be positive."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.001)
        assert lam > 0.01, f"Lyapunov {lam:.4f} not positive for chaotic regime"


class TestChuaRediscovery:
    """Tests for Chua data generation functions."""

    def test_trajectory_data_shape(self):
        from simulating_anything.rediscovery.chua import generate_trajectory_data

        data = generate_trajectory_data(n_steps=100, dt=0.001)
        assert data["states"].shape == (101, 3)
        assert data["alpha"] == 15.6
        assert data["beta"] == 28.0
        assert data["m0"] == -1.143
        assert data["m1"] == -0.714

    def test_trajectory_data_stays_finite(self):
        from simulating_anything.rediscovery.chua import generate_trajectory_data

        data = generate_trajectory_data(n_steps=1000, dt=0.001)
        assert np.all(np.isfinite(data["states"]))

    def test_attractor_data_shape(self):
        from simulating_anything.rediscovery.chua import generate_attractor_data

        data = generate_attractor_data(n_alpha=5, dt=0.001)
        assert len(data["alpha"]) == 5
        assert len(data["lyapunov_exponent"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.chua import generate_trajectory_data

        data = generate_trajectory_data(n_steps=200, dt=0.001)
        states = data["states"]
        # SINDy expects (n_timesteps, n_variables) with float dtype
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data
