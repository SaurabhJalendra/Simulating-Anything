"""Tests for the WINDMI solar wind-magnetosphere-ionosphere model."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.windmi import WindmiSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestWindmiSimulation:
    """Tests for the WINDMI simulation basics."""

    def _make_sim(self, **kwargs) -> WindmiSimulation:
        defaults = {
            "a": 0.7, "b": 2.5,
            "x_0": 0.1, "y_0": 0.0, "z_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WINDMI,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return WindmiSimulation(config)

    def test_creation(self):
        """Simulation is created with correct parameters."""
        sim = self._make_sim()
        assert sim.a == 0.7
        assert sim.b == 2.5

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=1.0, y_0=-0.5, z_0=0.3)
        state = sim.reset()
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], -0.5)
        assert np.isclose(state[2], 0.3)

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

    def test_state_shape_preserved(self):
        """State shape should remain (3,) throughout the simulation."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(100):
            state = sim.step()
            assert state.shape == (3,)

    def test_run_returns_trajectory(self):
        """run() should return TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))


class TestWindmiDerivatives:
    """Tests for the WINDMI ODEs at known points."""

    def _make_sim(self, **kwargs) -> WindmiSimulation:
        defaults = {"a": 0.7, "b": 2.5}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WINDMI,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return WindmiSimulation(config)

    def test_derivatives_at_origin(self):
        """Derivatives at origin: dx=0, dy=0, dz=-0.7*0-0+2.5-exp(0)=1.5."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], 0.0)
        assert np.isclose(derivs[1], 0.0)
        # dz = -0.7*0 - 0 + 2.5 - exp(0) = 2.5 - 1.0 = 1.5
        assert np.isclose(derivs[2], 1.5)

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point [1, 0.5, 0.2]."""
        sim = self._make_sim(a=0.7, b=2.5)
        sim.reset()
        # At state [1, 0.5, 0.2]:
        # dx = y = 0.5
        # dy = z = 0.2
        # dz = -0.7*0.2 - 0.5 + 2.5 - exp(1) = -0.14 - 0.5 + 2.5 - 2.71828
        expected_dz = -0.7 * 0.2 - 0.5 + 2.5 - np.exp(1.0)
        derivs = sim._derivatives(np.array([1.0, 0.5, 0.2]))
        assert np.isclose(derivs[0], 0.5)
        assert np.isclose(derivs[1], 0.2)
        assert np.isclose(derivs[2], expected_dz)

    def test_derivatives_at_fixed_point(self):
        """At fixed point (ln(b), 0, 0), derivatives should be zero."""
        sim = self._make_sim(a=0.7, b=2.5)
        sim.reset()
        fp = np.array([np.log(2.5), 0.0, 0.0])
        derivs = sim._derivatives(fp)
        np.testing.assert_array_almost_equal(
            derivs, [0.0, 0.0, 0.0], decimal=10,
        )

    def test_jerk_form_consistency(self):
        """x''' (jerk) should equal dz/dt at any state."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([0.5, -0.3, 0.8])
        jerk = sim.compute_jerk(state)
        dz = sim._derivatives(state)[2]
        assert np.isclose(jerk, dz)


class TestWindmiFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> WindmiSimulation:
        defaults = {"a": 0.7, "b": 2.5}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WINDMI,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return WindmiSimulation(config)

    def test_single_fixed_point(self):
        """WINDMI has exactly one fixed point for b > 0."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1

    def test_fixed_point_value(self):
        """Fixed point should be at (ln(b), 0, 0)."""
        sim = self._make_sim(b=2.5)
        sim.reset()
        fps = sim.fixed_points
        expected = np.array([np.log(2.5), 0.0, 0.0])
        np.testing.assert_array_almost_equal(fps[0], expected, decimal=10)

    def test_fixed_point_scales_with_b(self):
        """x* = ln(b) should scale logarithmically with b."""
        for b_val in [1.0, 2.0, 5.0, 10.0]:
            sim = self._make_sim(b=b_val)
            sim.reset()
            fps = sim.fixed_points
            assert np.isclose(fps[0][0], np.log(b_val))
            assert np.isclose(fps[0][1], 0.0)
            assert np.isclose(fps[0][2], 0.0)

    def test_no_fixed_point_negative_b(self):
        """For b <= 0, no real fixed point exists (ln undefined)."""
        sim = self._make_sim(b=-1.0)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 0

    def test_derivatives_at_fixed_point_zero(self):
        """Derivatives should be zero at the fixed point."""
        sim = self._make_sim(b=3.0)
        sim.reset()
        for fp in sim.fixed_points:
            derivs = sim._derivatives(fp)
            np.testing.assert_array_almost_equal(
                derivs, [0.0, 0.0, 0.0], decimal=10,
                err_msg=f"Non-zero derivatives at fixed point {fp}",
            )


class TestWindmiJacobian:
    """Tests for the Jacobian and eigenvalue computation."""

    def test_jacobian_shape(self):
        """Jacobian should be 3x3."""
        config = SimulationConfig(
            domain=Domain.WINDMI, dt=0.01, n_steps=1000,
            parameters={"a": 0.7, "b": 2.5},
        )
        sim = WindmiSimulation(config)
        sim.reset()
        J = sim.jacobian_at_fixed_point
        assert J.shape == (3, 3)

    def test_jacobian_structure(self):
        """Jacobian at fixed point should have the expected structure."""
        config = SimulationConfig(
            domain=Domain.WINDMI, dt=0.01, n_steps=1000,
            parameters={"a": 0.7, "b": 2.5},
        )
        sim = WindmiSimulation(config)
        sim.reset()
        J = sim.jacobian_at_fixed_point
        # Row 0: [0, 1, 0]
        assert np.isclose(J[0, 0], 0.0)
        assert np.isclose(J[0, 1], 1.0)
        assert np.isclose(J[0, 2], 0.0)
        # Row 1: [0, 0, 1]
        assert np.isclose(J[1, 0], 0.0)
        assert np.isclose(J[1, 1], 0.0)
        assert np.isclose(J[1, 2], 1.0)
        # Row 2: [-b, -1, -a]
        assert np.isclose(J[2, 0], -2.5)
        assert np.isclose(J[2, 1], -1.0)
        assert np.isclose(J[2, 2], -0.7)

    def test_eigenvalues_count(self):
        """Should return 3 eigenvalues for the 3x3 Jacobian."""
        config = SimulationConfig(
            domain=Domain.WINDMI, dt=0.01, n_steps=1000,
            parameters={"a": 0.7, "b": 2.5},
        )
        sim = WindmiSimulation(config)
        sim.reset()
        eigs = sim.eigenvalues_at_fixed_point()
        assert len(eigs) == 3

    def test_characteristic_polynomial(self):
        """Eigenvalues should satisfy lambda^3 + a*lambda^2 + lambda + b = 0."""
        config = SimulationConfig(
            domain=Domain.WINDMI, dt=0.01, n_steps=1000,
            parameters={"a": 0.7, "b": 2.5},
        )
        sim = WindmiSimulation(config)
        sim.reset()
        eigs = sim.eigenvalues_at_fixed_point()
        for lam in eigs:
            residual = lam**3 + 0.7 * lam**2 + lam + 2.5
            assert np.abs(residual) < 1e-10, (
                f"Eigenvalue {lam} does not satisfy char. poly: residual={residual}"
            )


class TestWindmiTrajectory:
    """Tests for trajectory behavior."""

    def _make_sim(self, **kwargs) -> WindmiSimulation:
        defaults = {"a": 0.7, "b": 2.5, "x_0": 0.1}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WINDMI,
            dt=0.01,
            n_steps=50000,
            parameters=defaults,
        )
        return WindmiSimulation(config)

    def test_trajectory_stays_bounded(self):
        """WINDMI trajectories should remain bounded for classic parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 200, (
                f"Trajectory diverged: {state}"
            )

    def test_trajectory_not_trivial(self):
        """After transient, the trajectory should oscillate (not converge)."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(5000):
            sim.step()
        x_vals = []
        for _ in range(5000):
            state = sim.step()
            x_vals.append(state[0])
        assert np.std(x_vals) > 0.01, "Trajectory collapsed to fixed point"

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
        """Different b values should yield different trajectories."""
        sim1 = self._make_sim(b=2.5)
        sim2 = self._make_sim(b=1.5)
        traj1 = sim1.run(n_steps=500)
        traj2 = sim2.run(n_steps=500)
        assert not np.allclose(traj1.states, traj2.states)


class TestWindmiLyapunov:
    """Tests for Lyapunov exponent estimation."""

    def test_positive_lyapunov_chaotic(self):
        """WINDMI at a=0.7, b=2.5 should have positive Lyapunov exponent."""
        config = SimulationConfig(
            domain=Domain.WINDMI,
            dt=0.01,
            n_steps=20000,
            parameters={"a": 0.7, "b": 2.5, "x_0": 0.1},
        )
        sim = WindmiSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.01)
        assert lam > 0.0, (
            f"Lyapunov {lam:.3f} should be positive for chaotic regime"
        )

    def test_small_b_less_chaotic(self):
        """Smaller b (less driving) should produce smaller Lyapunov exponent."""
        config_low = SimulationConfig(
            domain=Domain.WINDMI, dt=0.01, n_steps=20000,
            parameters={"a": 0.7, "b": 0.5, "x_0": 0.1},
        )
        sim_low = WindmiSimulation(config_low)
        sim_low.reset()
        for _ in range(5000):
            sim_low.step()
        lam_low = sim_low.estimate_lyapunov(n_steps=20000, dt=0.01)

        config_high = SimulationConfig(
            domain=Domain.WINDMI, dt=0.01, n_steps=20000,
            parameters={"a": 0.7, "b": 2.5, "x_0": 0.1},
        )
        sim_high = WindmiSimulation(config_high)
        sim_high.reset()
        for _ in range(5000):
            sim_high.step()
        lam_high = sim_high.estimate_lyapunov(n_steps=20000, dt=0.01)

        # Chaotic regime (high b) should have larger Lyapunov
        assert lam_high > lam_low, (
            f"High-b Lyapunov {lam_high:.3f} should exceed "
            f"low-b Lyapunov {lam_low:.3f}"
        )


class TestWindmiBifurcation:
    """Tests for bifurcation sweep."""

    def test_bifurcation_sweep_shape(self):
        """Bifurcation sweep should return correct array shapes."""
        config = SimulationConfig(
            domain=Domain.WINDMI, dt=0.01, n_steps=1000,
            parameters={"a": 0.7, "b": 2.5, "x_0": 0.1},
        )
        sim = WindmiSimulation(config)
        sim.reset()
        b_vals = np.linspace(1.0, 3.0, 5)
        result = sim.bifurcation_sweep(
            b_vals, n_transient=1000, n_measure=2000
        )
        assert len(result["b"]) == 5
        assert len(result["lyapunov_exponent"]) == 5
        assert len(result["attractor_type"]) == 5

    def test_bifurcation_sweep_attractor_types(self):
        """Sweep should produce valid attractor type labels."""
        config = SimulationConfig(
            domain=Domain.WINDMI, dt=0.01, n_steps=1000,
            parameters={"a": 0.7, "b": 2.5, "x_0": 0.1},
        )
        sim = WindmiSimulation(config)
        sim.reset()
        b_vals = np.linspace(0.5, 4.0, 8)
        result = sim.bifurcation_sweep(
            b_vals, n_transient=1000, n_measure=2000
        )
        valid_types = {"chaotic", "stable", "marginal"}
        for atype in result["attractor_type"]:
            assert atype in valid_types, f"Unknown attractor type: {atype}"


class TestWindmiRediscovery:
    """Tests for WINDMI rediscovery data generation functions."""

    def test_ode_data(self):
        from simulating_anything.rediscovery.windmi import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 0.7
        assert data["b"] == 2.5

    def test_ode_data_stays_finite(self):
        from simulating_anything.rediscovery.windmi import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_substorm_sweep_data(self):
        from simulating_anything.rediscovery.windmi import (
            generate_substorm_sweep_data,
        )

        data = generate_substorm_sweep_data(n_b=5, n_steps=2000, dt=0.01)
        assert len(data["b"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_substorm_sweep_contains_types(self):
        """Sweep from b=0.5 to b=5.0 should find at least one regime."""
        from simulating_anything.rediscovery.windmi import (
            generate_substorm_sweep_data,
        )

        data = generate_substorm_sweep_data(n_b=10, n_steps=5000, dt=0.01)
        types = set(data["attractor_type"])
        assert len(types) >= 1, f"Found only: {types}"

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.windmi import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=200, dt=0.01)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data
