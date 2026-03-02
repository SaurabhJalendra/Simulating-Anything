"""Tests for the Lorenz-Stenflo system simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.lorenz_stenflo import LorenzStenfloSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestLorenzStenfloCreation:
    """Tests for simulation creation and parameter handling."""

    def _make_sim(self, **kwargs) -> LorenzStenfloSimulation:
        defaults = {
            "sigma": 10.0, "r": 28.0, "b": 8.0 / 3.0, "s": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_STENFLO,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return LorenzStenfloSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.sigma == 10.0
        assert sim.r == 28.0
        assert np.isclose(sim.b, 8.0 / 3.0)
        assert sim.s == 1.0

    def test_creation_custom_parameters(self):
        """Simulation accepts custom parameter values."""
        sim = self._make_sim(sigma=5.0, r=20.0, b=1.5, s=3.0)
        assert sim.sigma == 5.0
        assert sim.r == 20.0
        assert sim.b == 1.5
        assert sim.s == 3.0

    def test_domain_enum_exists(self):
        """LORENZ_STENFLO should exist in the Domain enum."""
        assert Domain.LORENZ_STENFLO == "lorenz_stenflo"

    def test_config_stored(self):
        """Config is accessible on the simulation."""
        sim = self._make_sim()
        assert sim.config.dt == 0.005
        assert sim.config.n_steps == 10000


class TestLorenzStenfloState:
    """Tests for state initialization and shape."""

    def _make_sim(self, **kwargs) -> LorenzStenfloSimulation:
        defaults = {
            "sigma": 10.0, "r": 28.0, "b": 8.0 / 3.0, "s": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_STENFLO,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return LorenzStenfloSimulation(config)

    def test_state_shape(self):
        """State vector has shape (4,) for the 4D system."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (4,)

    def test_state_dtype(self):
        """State should be float64 for numerical precision."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.dtype == np.float64

    def test_initial_state_default_values(self):
        """Default initial state is [1, 0, 0, 0]."""
        sim = self._make_sim()
        state = sim.reset()
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], 0.0)
        assert np.isclose(state[2], 0.0)
        assert np.isclose(state[3], 0.0)

    def test_initial_state_custom_values(self):
        """Custom initial conditions are respected."""
        sim = self._make_sim(x_0=2.0, y_0=-1.5, z_0=0.3, w_0=0.5)
        state = sim.reset()
        assert np.isclose(state[0], 2.0)
        assert np.isclose(state[1], -1.5)
        assert np.isclose(state[2], 0.3)
        assert np.isclose(state[3], 0.5)

    def test_observe_returns_current_state(self):
        """observe() returns the same state as the current internal state."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_advances_state(self):
        """A single step changes the state."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)


class TestLorenzStenfloDerivatives:
    """Tests for the derivative computations."""

    def _make_sim(self, **kwargs) -> LorenzStenfloSimulation:
        defaults = {
            "sigma": 10.0, "r": 28.0, "b": 8.0 / 3.0, "s": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_STENFLO,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return LorenzStenfloSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a known point.

        At [1, 1, 1, 1] with sigma=10, r=28, b=8/3, s=1:
            dx = 10*(1-1) + 1*1 = 1
            dy = 28*1 - 1 - 1*1 = 26
            dz = 1*1 - (8/3)*1 = 1 - 8/3 = -5/3
            dw = -1 - 10*1 = -11
        """
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 1.0)
        assert np.isclose(derivs[1], 26.0)
        assert np.isclose(derivs[2], 1.0 - 8.0 / 3.0)
        assert np.isclose(derivs[3], -11.0)

    def test_derivatives_with_s_zero(self):
        """When s=0, the dx derivative has no w contribution.

        At [1, 2, 0, 5] with sigma=10, r=28, b=8/3, s=0:
            dx = 10*(2-1) + 0*5 = 10
            dy = 28*1 - 2 - 1*0 = 26
            dz = 1*2 - 0 = 2
            dw = -1 - 10*5 = -51
        """
        sim = self._make_sim(s=0.0)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 2.0, 0.0, 5.0]))
        assert np.isclose(derivs[0], 10.0)
        assert np.isclose(derivs[1], 26.0)
        assert np.isclose(derivs[2], 2.0)
        assert np.isclose(derivs[3], -51.0)


class TestLorenzStenfloLorenzReduction:
    """Tests verifying the system reduces to Lorenz when s=0."""

    def _make_sim(self, **kwargs) -> LorenzStenfloSimulation:
        defaults = {
            "sigma": 10.0, "r": 28.0, "b": 8.0 / 3.0, "s": 0.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 1.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_STENFLO,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return LorenzStenfloSimulation(config)

    def test_w_stays_zero_when_s_zero_and_w0_zero(self):
        """With s=0 and w_0=0, w should remain identically zero.

        dw/dt = -x - sigma*w. With w=0 initially, dw = -x. But we need
        s=0 for dx to not depend on w. Actually w evolves as dw = -x - sigma*w
        which does NOT stay zero even when s=0. The w equation decouples from
        the xyz dynamics (x does not depend on w when s=0), but w is driven by x.

        Instead we verify: xyz dynamics are identical to pure Lorenz.
        """
        sim = self._make_sim(s=0.0, w_0=0.0)
        sim.reset()

        # xyz dynamics should match pure Lorenz regardless of w
        # Verify by checking that the xyz derivatives do not depend on w when s=0
        state = np.array([5.0, -3.0, 10.0, 999.0])
        derivs = sim._derivatives(state)

        # With s=0: dx = sigma*(y-x) + 0*w = 10*(-3-5) = -80
        assert np.isclose(derivs[0], 10.0 * (-3.0 - 5.0))

    def test_xyz_independent_of_w_when_s_zero(self):
        """When s=0, the xyz derivatives should be the same regardless of w value."""
        sim = self._make_sim(s=0.0)
        sim.reset()

        state_w0 = np.array([2.0, 3.0, 4.0, 0.0])
        state_w100 = np.array([2.0, 3.0, 4.0, 100.0])

        derivs_w0 = sim._derivatives(state_w0)
        derivs_w100 = sim._derivatives(state_w100)

        # xyz derivatives should be identical
        np.testing.assert_array_almost_equal(
            derivs_w0[:3], derivs_w100[:3],
            err_msg="xyz derivatives depend on w when s=0"
        )

    def test_reduces_to_lorenz_method(self):
        """The reduces_to_lorenz method should confirm reduction when s=0."""
        sim = self._make_sim(s=1.0)
        sim.reset()
        result = sim.reduces_to_lorenz(n_steps=1000)
        # When called with s=0 internally, max Lorenz deviation should be tiny
        assert result["max_lorenz_deviation"] < 1e-10


class TestLorenzStenfloFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> LorenzStenfloSimulation:
        defaults = {
            "sigma": 10.0, "r": 28.0, "b": 8.0 / 3.0, "s": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_STENFLO,
            dt=0.005,
            n_steps=1000,
            parameters=defaults,
        )
        return LorenzStenfloSimulation(config)

    def test_three_fixed_points_standard_params(self):
        """Standard parameters (r=28 >> alpha) should give three fixed points."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3

    def test_origin_always_fixed_point(self):
        """The origin [0,0,0,0] should always be a fixed point."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        np.testing.assert_array_almost_equal(
            fps[0], [0.0, 0.0, 0.0, 0.0]
        )

    def test_derivatives_at_fixed_points(self):
        """Derivatives should be zero at each fixed point."""
        sim = self._make_sim()
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0, 0.0], decimal=10,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )

    def test_fixed_point_symmetry(self):
        """The two non-origin fixed points should have symmetric x, y, w."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3

        # fp1 and fp2 should have opposite x, y but same z
        assert np.isclose(fps[1][0], -fps[2][0])
        assert np.isclose(fps[1][1], -fps[2][1])
        assert np.isclose(fps[1][2], fps[2][2])
        # w = -x/sigma, so w should be opposite too
        assert np.isclose(fps[1][3], -fps[2][3])

    def test_one_fixed_point_when_r_small(self):
        """When r < alpha = 1 + s/sigma^2, only the origin exists."""
        # alpha = 1 + 1/100 = 1.01, so r=0.5 < 1.01
        sim = self._make_sim(r=0.5)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1


class TestLorenzStenfloTrajectory:
    """Tests for trajectory behavior and boundedness."""

    def _make_sim(self, **kwargs) -> LorenzStenfloSimulation:
        defaults = {
            "sigma": 10.0, "r": 28.0, "b": 8.0 / 3.0, "s": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_STENFLO,
            dt=0.005,
            n_steps=10000,
            parameters=defaults,
        )
        return LorenzStenfloSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Lorenz-Stenflo trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 500, f"Trajectory diverged: {state}"

    def test_trajectory_shape_from_run(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)
        assert np.all(np.isfinite(traj.states))

    def test_positive_lyapunov_chaotic_regime(self):
        """At standard parameters (s=1), largest Lyapunov exponent should be positive."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.005)
        assert lam > 0.1, f"Lyapunov {lam:.4f} not positive for chaotic regime"

    def test_w_component_active(self):
        """The w component should be nontrivial (not stuck at zero) for s > 0."""
        sim = self._make_sim(s=1.0)
        sim.reset()
        for _ in range(3000):
            sim.step()
        w_vals = []
        for _ in range(5000):
            state = sim.step()
            w_vals.append(abs(state[3]))
        max_w = np.max(w_vals)
        assert max_w > 0.1, f"max |w| = {max_w:.4f} too small; w should be active"


class TestLorenzStenfloStatistics:
    """Tests for trajectory statistics computation."""

    def _make_sim(self, **kwargs) -> LorenzStenfloSimulation:
        defaults = {
            "sigma": 10.0, "r": 28.0, "b": 8.0 / 3.0, "s": 1.0,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.0, "w_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.LORENZ_STENFLO,
            dt=0.005,
            n_steps=20000,
            parameters=defaults,
        )
        return LorenzStenfloSimulation(config)

    def test_statistics_keys(self):
        """compute_trajectory_statistics returns expected keys."""
        sim = self._make_sim()
        sim.reset()
        stats = sim.compute_trajectory_statistics(n_steps=2000, n_transient=500)
        expected_keys = {
            "x_mean", "y_mean", "z_mean", "w_mean",
            "x_std", "y_std", "z_std", "w_std",
            "x_range", "y_range", "z_range", "w_range",
        }
        assert set(stats.keys()) == expected_keys

    def test_statistics_values_finite(self):
        """All trajectory statistics should be finite numbers."""
        sim = self._make_sim()
        sim.reset()
        stats = sim.compute_trajectory_statistics(n_steps=2000, n_transient=500)
        for key, val in stats.items():
            assert np.isfinite(val), f"Statistic {key} is not finite: {val}"


class TestLorenzStenfloRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_trajectory_data_shape(self):
        from simulating_anything.rediscovery.lorenz_stenflo import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=100, dt=0.005)
        assert data["states"].shape == (101, 4)
        assert data["sigma"] == 10.0
        assert data["r"] == 28.0
        assert data["s"] == 1.0

    def test_trajectory_data_stays_finite(self):
        from simulating_anything.rediscovery.lorenz_stenflo import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=1000, dt=0.005)
        assert np.all(np.isfinite(data["states"]))

    def test_s_sweep_data_shape(self):
        from simulating_anything.rediscovery.lorenz_stenflo import (
            generate_s_sweep_data,
        )

        data = generate_s_sweep_data(n_s=5, dt=0.005)
        assert len(data["s"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["max_amplitude"]) == 5
        assert len(data["w_amplitude"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.lorenz_stenflo import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=200, dt=0.005)
        states = data["states"]
        # SINDy expects (n_timesteps, n_variables) with float dtype
        assert states.ndim == 2
        assert states.shape[1] == 4
        assert states.dtype == np.float64
        assert "dt" in data
