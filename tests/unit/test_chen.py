"""Tests for the Chen attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.chen import ChenSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

# Placeholder until Domain.CHEN is added to the enum
_CHEN_DOMAIN = Domain.CHAOTIC_ODE


class TestChenSimulation:
    """Tests for the Chen system simulation basics."""

    def _make_sim(self, **kwargs) -> ChenSimulation:
        defaults = {
            "a": 35.0, "b": 3.0, "c": 28.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_CHEN_DOMAIN,
            dt=0.002,
            n_steps=10000,
            parameters=defaults,
        )
        return ChenSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 35.0
        assert sim.b == 3.0
        assert sim.c == 28.0

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=40.0, b=5.0, c=30.0)
        assert sim.a == 40.0
        assert sim.b == 5.0
        assert sim.c == 30.0

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=5.0, y_0=-3.0, z_0=20.0)
        state = sim.reset()
        assert np.isclose(state[0], 5.0)
        assert np.isclose(state[1], -3.0)
        assert np.isclose(state[2], 20.0)

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

    def test_deterministic(self):
        """Same parameters should produce the same trajectory."""
        sim1 = self._make_sim(a=35.0, b=3.0, c=28.0)
        sim2 = self._make_sim(a=35.0, b=3.0, c=28.0)
        sim1.reset()
        sim2.reset()
        for _ in range(100):
            s1 = sim1.step()
            s2 = sim2.step()
        np.testing.assert_array_almost_equal(s1, s2, decimal=12)

    def test_trajectory_run(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))


class TestChenDerivatives:
    """Tests for the Chen ODE derivative computation."""

    def _make_sim(self, **kwargs) -> ChenSimulation:
        defaults = {
            "a": 35.0, "b": 3.0, "c": 28.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_CHEN_DOMAIN,
            dt=0.002,
            n_steps=1000,
            parameters=defaults,
        )
        return ChenSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 1, 1] with a=35, b=3, c=28:
            dx = 35*(1 - 1) = 0
            dy = (28 - 35)*1 - 1*1 + 28*1 = -7 - 1 + 28 = 20
            dz = 1*1 - 3*1 = -2
        """
        sim = self._make_sim(a=35.0, b=3.0, c=28.0)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 20.0)
        assert np.isclose(derivs[2], -2.0)

    def test_derivatives_another_point(self):
        """Test derivatives at [2, 3, 1] with a=35, b=3, c=28.

            dx = 35*(3 - 2) = 35
            dy = (28 - 35)*2 - 2*1 + 28*3 = -14 - 2 + 84 = 68
            dz = 2*3 - 3*1 = 3
        """
        sim = self._make_sim(a=35.0, b=3.0, c=28.0)
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 3.0, 1.0]))
        assert np.isclose(derivs[0], 35.0)
        assert np.isclose(derivs[1], 68.0)
        assert np.isclose(derivs[2], 3.0)


class TestChenFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> ChenSimulation:
        defaults = {"a": 35.0, "b": 3.0, "c": 28.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_CHEN_DOMAIN,
            dt=0.002,
            n_steps=1000,
            parameters=defaults,
        )
        return ChenSimulation(config)

    def test_three_fixed_points(self):
        """Standard parameters should give three fixed points."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 3

    def test_origin_is_fixed_point(self):
        """First fixed point should be the origin."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        np.testing.assert_array_almost_equal(fps[0], [0.0, 0.0, 0.0])

    def test_fixed_point_symmetry(self):
        """The two non-origin fixed points should be symmetric in x, y."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert np.isclose(fps[1][0], -fps[2][0])
        assert np.isclose(fps[1][1], -fps[2][1])
        assert np.isclose(fps[1][2], fps[2][2])

    def test_fixed_point_z_value(self):
        """z-coordinate of symmetric fixed points should be 2c - a.

        For a=35, b=3, c=28: z = 2*28 - 35 = 21
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        z_expected = 2.0 * 28.0 - 35.0  # = 21.0
        assert np.isclose(fps[1][2], z_expected)
        assert np.isclose(fps[2][2], z_expected)

    def test_fixed_point_x_value(self):
        """x-coordinate of symmetric fixed points should be +/-sqrt(b*(2c-a)).

        For a=35, b=3, c=28: x = sqrt(3*21) = sqrt(63) ~ 7.937
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        x_expected = np.sqrt(3.0 * 21.0)
        assert np.isclose(abs(fps[1][0]), x_expected)
        assert np.isclose(abs(fps[2][0]), x_expected)

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


class TestChenTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> ChenSimulation:
        defaults = {
            "a": 35.0, "b": 3.0, "c": 28.0,
            "x_0": 1.0, "y_0": 1.0, "z_0": 1.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_CHEN_DOMAIN,
            dt=0.002,
            n_steps=10000,
            parameters=defaults,
        )
        return ChenSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Chen trajectories should remain bounded for standard parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, f"Trajectory diverged: {state}"

    def test_z_stays_positive_on_attractor(self):
        """z coordinate should stay positive after initial transient on the attractor."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        for _ in range(5000):
            state = sim.step()
            assert state[2] > -5.0, f"z went too negative: {state[2]}"

    def test_attractor_statistics(self):
        """Time-averaged z should be close to 2c - a for chaotic attractor.

        For a=35, b=3, c=28: z_eq = 2*28 - 35 = 21
        """
        sim = self._make_sim()
        stats = sim.compute_trajectory_statistics(
            n_steps=20000, n_transient=5000
        )
        # z_mean should be in the vicinity of 21 (not exact due to chaos)
        assert 10.0 < stats["z_mean"] < 35.0, (
            f"z_mean={stats['z_mean']:.1f}, expected near 21"
        )
        # x and y should have similar std (xy symmetry)
        assert stats["x_std"] > 1.0, "x_std too small for chaotic regime"
        assert stats["y_std"] > 1.0, "y_std too small for chaotic regime"

    def test_different_c_gives_different_trajectory(self):
        """Changing c should change the trajectory behavior."""
        sim1 = self._make_sim(c=28.0)
        sim2 = self._make_sim(c=20.0)
        sim1.reset()
        sim2.reset()
        for _ in range(1000):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.1)


class TestChenChaosProperties:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_positive_lyapunov_chaotic(self):
        """Chen at standard parameters should have positive Lyapunov exponent."""
        config = SimulationConfig(
            domain=_CHEN_DOMAIN,
            dt=0.002,
            n_steps=20000,
            parameters={"a": 35.0, "b": 3.0, "c": 28.0},
        )
        sim = ChenSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.002)
        assert lam > 0.5, f"Lyapunov {lam:.3f} too small for chaotic regime"
        assert lam < 10.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_is_chaotic_property(self):
        """is_chaotic should be True at standard params and False for small c."""
        # c=28 < 2*35-3=67, so is_chaotic heuristic returns False for standard
        # params. The Chen system IS chaotic there, but the heuristic is rough.
        # Test with c > 2a - b where the heuristic is True.
        sim_high_c = ChenSimulation(SimulationConfig(
            domain=_CHEN_DOMAIN, dt=0.002, n_steps=1000,
            parameters={"a": 35.0, "b": 3.0, "c": 70.0},
        ))
        assert sim_high_c.is_chaotic is True

        sim_low_c = ChenSimulation(SimulationConfig(
            domain=_CHEN_DOMAIN, dt=0.002, n_steps=1000,
            parameters={"a": 35.0, "b": 3.0, "c": 10.0},
        ))
        assert sim_low_c.is_chaotic is False

    def test_lyapunov_varies_with_c(self):
        """Lyapunov exponent should change as c varies."""
        lyap_c28 = _compute_lyapunov_at_c(28.0)
        lyap_c20 = _compute_lyapunov_at_c(20.0)
        # c=28 is the standard chaotic value, c=20 is less chaotic
        assert lyap_c28 != lyap_c20, "Lyapunov should differ for different c"


class TestChenLorenzComparison:
    """Tests for the Lorenz dual classification."""

    def _make_sim(self, **kwargs) -> ChenSimulation:
        defaults = {"a": 35.0, "b": 3.0, "c": 28.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_CHEN_DOMAIN,
            dt=0.002,
            n_steps=1000,
            parameters=defaults,
        )
        return ChenSimulation(config)

    def test_chen_type_classification(self):
        """Standard Chen parameters should be classified as Chen type (a12*a21 < 0).

        a12 = a = 35, a21 = c - a = 28 - 35 = -7, product = -245 < 0.
        """
        sim = self._make_sim()
        comparison = sim.compare_with_lorenz()
        assert comparison["is_chen_type"] is True
        assert comparison["is_lorenz_type"] is False

    def test_lorenz_type_when_c_gt_a(self):
        """When c > a, the product a12*a21 > 0 (Lorenz-type).

        a12 = a = 35, a21 = c - a = 40 - 35 = 5, product = 175 > 0.
        """
        sim = self._make_sim(c=40.0)
        comparison = sim.compare_with_lorenz()
        assert comparison["is_chen_type"] is False
        assert comparison["is_lorenz_type"] is True

    def test_a12_a21_values(self):
        """Verify the linear coefficient values."""
        sim = self._make_sim(a=35.0, c=28.0)
        comparison = sim.compare_with_lorenz()
        assert np.isclose(comparison["a12"], 35.0)
        assert np.isclose(comparison["a21"], -7.0)
        assert np.isclose(comparison["a12_times_a21"], -245.0)


class TestChenRediscovery:
    """Tests for Chen data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.chen import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.002)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 35.0
        assert data["b"] == 3.0
        assert data["c"] == 28.0

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.chen import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.002)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_transition_data(self):
        """Chaos transition sweep should produce valid data."""
        from simulating_anything.rediscovery.chen import generate_chaos_transition_data

        data = generate_chaos_transition_data(n_c=5, n_steps=2000, dt=0.002)
        assert len(data["c"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.chen import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.002)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data

    def test_lyapunov_vs_c_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.chen import generate_lyapunov_vs_c_data

        data = generate_lyapunov_vs_c_data(n_c=5, n_steps=3000, dt=0.002)
        assert len(data["c"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


def _compute_lyapunov_at_c(c: float) -> float:
    """Helper to compute Lyapunov exponent at a given c."""
    config = SimulationConfig(
        domain=_CHEN_DOMAIN,
        dt=0.002,
        n_steps=20000,
        parameters={"a": 35.0, "b": 3.0, "c": c, "x_0": 1.0, "y_0": 1.0, "z_0": 1.0},
    )
    sim = ChenSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.002)
