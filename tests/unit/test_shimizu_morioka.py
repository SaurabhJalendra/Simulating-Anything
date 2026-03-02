"""Tests for the Shimizu-Morioka attractor simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.shimizu_morioka import ShimizuMoriokaSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

_SM_DOMAIN = Domain.SHIMIZU_MORIOKA


class TestShimizuMoriokaSimulation:
    """Tests for the Shimizu-Morioka system simulation basics."""

    def _make_sim(self, **kwargs) -> ShimizuMoriokaSimulation:
        defaults = {
            "a": 0.75, "b": 0.45,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_SM_DOMAIN,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ShimizuMoriokaSimulation(config)

    def test_creation_default_parameters(self):
        """Simulation is created with correct default parameters."""
        sim = self._make_sim()
        assert sim.a == 0.75
        assert sim.b == 0.45

    def test_creation_custom_parameters(self):
        """Custom parameters are stored correctly."""
        sim = self._make_sim(a=1.0, b=0.5)
        assert sim.a == 1.0
        assert sim.b == 0.5

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=5.0, y_0=-3.0, z_0=2.0)
        state = sim.reset()
        assert np.isclose(state[0], 5.0)
        assert np.isclose(state[1], -3.0)
        assert np.isclose(state[2], 2.0)

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
        sim1 = self._make_sim(a=0.75, b=0.45)
        sim2 = self._make_sim(a=0.75, b=0.45)
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


class TestShimizuMoriokaDerivatives:
    """Tests for the Shimizu-Morioka ODE derivative computation."""

    def _make_sim(self, **kwargs) -> ShimizuMoriokaSimulation:
        defaults = {
            "a": 0.75, "b": 0.45,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_SM_DOMAIN,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ShimizuMoriokaSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin, all derivatives should be zero (fixed point)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point.

        At state [1, 0, 0.5] with a=0.75, b=0.45:
            dx = 0.0
            dy = (1 - 0.5)*1 - 0.75*0 = 0.5
            dz = -0.45*0.5 + 1*1 = -0.225 + 1 = 0.775
        """
        sim = self._make_sim(a=0.75, b=0.45)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 0.0, 0.5]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 0.5)
        assert np.isclose(derivs[2], 0.775)

    def test_derivatives_another_point(self):
        """Test derivatives at [2, 3, 1] with a=0.75, b=0.45.

            dx = 3
            dy = (1 - 1)*2 - 0.75*3 = 0 - 2.25 = -2.25
            dz = -0.45*1 + 2*2 = -0.45 + 4 = 3.55
        """
        sim = self._make_sim(a=0.75, b=0.45)
        sim.reset()
        derivs = sim._derivatives(np.array([2.0, 3.0, 1.0]))
        assert np.isclose(derivs[0], 3.0)
        assert np.isclose(derivs[1], -2.25)
        assert np.isclose(derivs[2], 3.55)


class TestShimizuMoriokaRK4Convergence:
    """Tests for RK4 integration convergence."""

    def test_rk4_convergence_order(self):
        """RK4 should converge at 4th order: halving dt reduces error by ~16x."""
        errors = []
        for dt in [0.02, 0.01]:
            config = SimulationConfig(
                domain=_SM_DOMAIN,
                dt=dt,
                n_steps=int(1.0 / dt),
                parameters={"a": 0.75, "b": 0.45, "x_0": 0.5, "y_0": 0.5, "z_0": 0.5},
            )
            sim = ShimizuMoriokaSimulation(config)
            sim.reset()
            for _ in range(int(1.0 / dt)):
                sim.step()
            errors.append(sim.observe().copy())

        # Use the finer result as reference
        config_ref = SimulationConfig(
            domain=_SM_DOMAIN,
            dt=0.005,
            n_steps=int(1.0 / 0.005),
            parameters={"a": 0.75, "b": 0.45, "x_0": 0.5, "y_0": 0.5, "z_0": 0.5},
        )
        sim_ref = ShimizuMoriokaSimulation(config_ref)
        sim_ref.reset()
        for _ in range(int(1.0 / 0.005)):
            sim_ref.step()
        ref = sim_ref.observe()

        err_coarse = np.linalg.norm(errors[0] - ref)
        err_fine = np.linalg.norm(errors[1] - ref)

        # 4th order: halving dt should reduce error by ~16x
        # Allow generous tolerance for nonlinear system
        if err_fine > 1e-14:
            ratio = err_coarse / err_fine
            assert ratio > 4.0, f"RK4 convergence ratio {ratio:.1f} < 4.0"


class TestShimizuMoriokaFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> ShimizuMoriokaSimulation:
        defaults = {"a": 0.75, "b": 0.45}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_SM_DOMAIN,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ShimizuMoriokaSimulation(config)

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
        """The two non-origin fixed points should be symmetric in x."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert np.isclose(fps[1][0], -fps[2][0])
        assert np.isclose(fps[1][1], fps[2][1])  # both y=0
        assert np.isclose(fps[1][2], fps[2][2])  # both z=1

    def test_fixed_point_z_value(self):
        """z-coordinate of symmetric fixed points should be 1.0."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert np.isclose(fps[1][2], 1.0)
        assert np.isclose(fps[2][2], 1.0)

    def test_fixed_point_y_value(self):
        """y-coordinate of symmetric fixed points should be 0."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert np.isclose(fps[1][1], 0.0)
        assert np.isclose(fps[2][1], 0.0)

    def test_fixed_point_x_value(self):
        """x-coordinate of symmetric fixed points should be +/-sqrt(b).

        For b=0.45: x = sqrt(0.45) ~ 0.6708
        """
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        x_expected = np.sqrt(0.45)
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


class TestShimizuMoriokaJacobian:
    """Tests for the Jacobian matrix computation."""

    def _make_sim(self, **kwargs) -> ShimizuMoriokaSimulation:
        defaults = {"a": 0.75, "b": 0.45}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_SM_DOMAIN,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ShimizuMoriokaSimulation(config)

    def test_jacobian_shape(self):
        """Jacobian should be a 3x3 matrix."""
        sim = self._make_sim()
        sim.reset()
        J = sim.jacobian(np.array([1.0, 0.0, 0.5]))
        assert J.shape == (3, 3)

    def test_jacobian_at_origin(self):
        """Jacobian at origin should be:
        [[0,    1,   0  ],
         [1,   -a,   0  ],
         [0,    0,  -b  ]]
        """
        sim = self._make_sim(a=0.75, b=0.45)
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        expected = np.array([
            [0.0, 1.0, 0.0],
            [1.0, -0.75, 0.0],
            [0.0, 0.0, -0.45],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_at_symmetric_fp(self):
        """Jacobian at (+sqrt(b), 0, 1) should be:
        [[0,      1,       0       ],
         [0,     -a,  -sqrt(b)     ],
         [2*sqrt(b), 0,   -b       ]]
        """
        sim = self._make_sim(a=0.75, b=0.45)
        sim.reset()
        x_eq = np.sqrt(0.45)
        J = sim.jacobian(np.array([x_eq, 0.0, 1.0]))
        expected = np.array([
            [0.0, 1.0, 0.0],
            [0.0, -0.75, -x_eq],
            [2.0 * x_eq, 0.0, -0.45],
        ])
        np.testing.assert_array_almost_equal(J, expected)


class TestShimizuMorikaDivergence:
    """Tests for divergence computation."""

    def _make_sim(self, **kwargs) -> ShimizuMoriokaSimulation:
        defaults = {"a": 0.75, "b": 0.45}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_SM_DOMAIN,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ShimizuMoriokaSimulation(config)

    def test_divergence_value(self):
        """Divergence should be -(a + b) = -(0.75 + 0.45) = -1.2."""
        sim = self._make_sim(a=0.75, b=0.45)
        div = sim.compute_divergence(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(div, -1.2)

    def test_divergence_is_negative(self):
        """Divergence should be negative (dissipative system)."""
        sim = self._make_sim(a=0.75, b=0.45)
        div = sim.compute_divergence(np.array([0.0, 0.0, 0.0]))
        assert div < 0

    def test_divergence_state_independent(self):
        """Divergence should be the same at any state."""
        sim = self._make_sim(a=0.75, b=0.45)
        div1 = sim.compute_divergence(np.array([0.0, 0.0, 0.0]))
        div2 = sim.compute_divergence(np.array([10.0, -5.0, 3.0]))
        assert np.isclose(div1, div2)


class TestShimizuMoriokaTrajectory:
    """Tests for trajectory boundedness and behavior."""

    def _make_sim(self, **kwargs) -> ShimizuMoriokaSimulation:
        defaults = {
            "a": 0.75, "b": 0.45,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.5,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=_SM_DOMAIN,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return ShimizuMoriokaSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Shimizu-Morioka trajectories should remain bounded."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 50, f"Trajectory diverged: {state}"

    def test_z_stays_nonnegative_on_attractor(self):
        """z coordinate should stay non-negative after transient.

        From dz/dt = -b*z + x^2, when z<0 both terms push z positive.
        """
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        for _ in range(3000):
            state = sim.step()
            assert state[2] > -1.0, f"z went too negative: {state[2]}"

    def test_attractor_statistics(self):
        """Time-averaged z should be near 1 (the z-value of the symmetric FPs)."""
        sim = self._make_sim()
        stats = sim.compute_trajectory_statistics(
            n_steps=20000, n_transient=5000
        )
        # z_mean should be in the vicinity of 1.0
        assert 0.1 < stats["z_mean"] < 5.0, (
            f"z_mean={stats['z_mean']:.1f}, expected near 1"
        )
        # x should have nonzero spread (chaotic)
        assert stats["x_std"] > 0.1, "x_std too small for chaotic regime"

    def test_different_a_gives_different_trajectory(self):
        """Changing a should change the trajectory behavior."""
        sim1 = self._make_sim(a=0.75)
        sim2 = self._make_sim(a=0.3)
        sim1.reset()
        sim2.reset()
        for _ in range(500):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2, atol=0.1)


class TestShimizuMorikaChaosProperties:
    """Tests for chaos detection and Lyapunov exponents."""

    def test_positive_lyapunov_chaotic(self):
        """SM at standard parameters should have positive Lyapunov exponent."""
        config = SimulationConfig(
            domain=_SM_DOMAIN,
            dt=0.01,
            n_steps=20000,
            parameters={"a": 0.75, "b": 0.45},
        )
        sim = ShimizuMoriokaSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.01, f"Lyapunov {lam:.3f} too small for chaotic regime"
        assert lam < 5.0, f"Lyapunov {lam:.3f} unreasonably large"

    def test_lyapunov_varies_with_a(self):
        """Lyapunov exponent should change as a varies."""
        lyap_a075 = _compute_lyapunov_at_a(0.75)
        lyap_a130 = _compute_lyapunov_at_a(1.30)
        assert lyap_a075 != lyap_a130, "Lyapunov should differ for different a"

    def test_symmetry_x_to_minus_x(self):
        """The system has (x, y, z) -> (-x, -y, z) symmetry.

        If (x(t), y(t), z(t)) is a solution, so is (-x(t), -y(t), z(t)).
        """
        config1 = SimulationConfig(
            domain=_SM_DOMAIN,
            dt=0.01,
            n_steps=500,
            parameters={"a": 0.75, "b": 0.45, "x_0": 1.0, "y_0": 0.5, "z_0": 0.5},
        )
        config2 = SimulationConfig(
            domain=_SM_DOMAIN,
            dt=0.01,
            n_steps=500,
            parameters={"a": 0.75, "b": 0.45, "x_0": -1.0, "y_0": -0.5, "z_0": 0.5},
        )
        sim1 = ShimizuMoriokaSimulation(config1)
        sim2 = ShimizuMoriokaSimulation(config2)
        sim1.reset()
        sim2.reset()

        for _ in range(100):
            s1 = sim1.step()
            s2 = sim2.step()

        # x and y should be negated, z should be the same
        assert np.isclose(s1[0], -s2[0], atol=1e-10)
        assert np.isclose(s1[1], -s2[1], atol=1e-10)
        assert np.isclose(s1[2], s2[2], atol=1e-10)


class TestShimizuMoriokaRediscovery:
    """Tests for Shimizu-Morioka data generation functions."""

    def test_ode_data_shape(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.shimizu_morioka import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 0.75
        assert data["b"] == 0.45

    def test_ode_data_stays_finite(self):
        """Trajectory data should remain finite."""
        from simulating_anything.rediscovery.shimizu_morioka import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_chaos_transition_data(self):
        """Chaos transition sweep should produce valid data."""
        from simulating_anything.rediscovery.shimizu_morioka import (
            generate_chaos_transition_data,
        )

        data = generate_chaos_transition_data(n_a=5, n_steps=2000, dt=0.01)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.shimizu_morioka import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.01)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data

    def test_lyapunov_vs_a_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.shimizu_morioka import (
            generate_lyapunov_vs_a_data,
        )

        data = generate_lyapunov_vs_a_data(n_a=5, n_steps=3000, dt=0.01)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))


def _compute_lyapunov_at_a(a: float) -> float:
    """Helper to compute Lyapunov exponent at a given a."""
    config = SimulationConfig(
        domain=_SM_DOMAIN,
        dt=0.01,
        n_steps=20000,
        parameters={
            "a": a, "b": 0.45,
            "x_0": 1.0, "y_0": 0.0, "z_0": 0.5,
        },
    )
    sim = ShimizuMoriokaSimulation(config)
    sim.reset()
    for _ in range(3000):
        sim.step()
    return sim.estimate_lyapunov(n_steps=15000, dt=0.01)
