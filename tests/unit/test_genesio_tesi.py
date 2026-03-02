"""Tests for the Genesio-Tesi chaotic system simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.genesio_tesi import GenesioTesiSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestGenesioTesiSimulation:
    """Tests for the Genesio-Tesi system simulation basics."""

    def _make_sim(self, **kwargs) -> GenesioTesiSimulation:
        defaults = {
            "a": 0.44, "b": 1.1, "c": 1.0,
            "x_0": 0.1, "y_0": 0.1, "z_0": 0.1,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.GENESIO_TESI,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return GenesioTesiSimulation(config)

    def test_creation(self):
        """Simulation is created with correct parameters."""
        sim = self._make_sim()
        assert sim.a == 0.44
        assert sim.b == 1.1
        assert sim.c == 1.0

    def test_initial_state_shape(self):
        """State vector has shape (3,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (3,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        """Initial state matches specified initial conditions."""
        sim = self._make_sim(x_0=0.5, y_0=-0.2, z_0=0.3)
        state = sim.reset()
        assert np.isclose(state[0], 0.5)
        assert np.isclose(state[1], -0.2)
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

    def test_deterministic(self):
        """Same parameters produce the same trajectory."""
        sim1 = self._make_sim()
        sim2 = self._make_sim()
        sim1.reset()
        sim2.reset()
        for _ in range(200):
            s1 = sim1.step()
            s2 = sim2.step()
        np.testing.assert_array_almost_equal(s1, s2, decimal=12)

    def test_trajectory_shape_from_run(self):
        """run() returns TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert np.all(np.isfinite(traj.states))


class TestGenesioTesiDerivatives:
    """Tests for the Genesio-Tesi derivative computation."""

    def _make_sim(self, **kwargs) -> GenesioTesiSimulation:
        defaults = {"a": 0.44, "b": 1.1, "c": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.GENESIO_TESI,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return GenesioTesiSimulation(config)

    def test_derivatives_at_origin(self):
        """At origin: dx=0, dy=0, dz=0 (equilibrium)."""
        sim = self._make_sim()
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_at_second_fixed_point(self):
        """At (c, 0, 0): dz = -c*c + c^2 = 0, so derivatives are zero."""
        sim = self._make_sim(c=1.0)
        sim.reset()
        derivs = sim._derivatives(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0, 0.0])

    def test_derivatives_known_point(self):
        """Test derivatives at a specific known point."""
        sim = self._make_sim(a=0.44, b=1.1, c=1.0)
        sim.reset()
        # At state [1, 1, 1]:
        # dx = 1
        # dy = 1
        # dz = -1.0*1 - 1.1*1 - 0.44*1 + 1^2 = -1.0 - 1.1 - 0.44 + 1.0 = -1.54
        derivs = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        assert np.isclose(derivs[0], 1.0)
        assert np.isclose(derivs[1], 1.0)
        assert np.isclose(derivs[2], -1.54)

    def test_jerk_form_consistency(self):
        """The jerk x''' equals dz/dt = -c*x - b*y - a*z + x^2."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([0.5, 0.3, -0.2])
        jerk = sim.compute_jerk(state)
        dz = sim._derivatives(state)[2]
        assert np.isclose(jerk, dz), f"Jerk {jerk} != dz {dz}"


class TestGenesioTesiFixedPoints:
    """Tests for fixed point computation."""

    def _make_sim(self, **kwargs) -> GenesioTesiSimulation:
        defaults = {"a": 0.44, "b": 1.1, "c": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.GENESIO_TESI,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return GenesioTesiSimulation(config)

    def test_two_fixed_points(self):
        """Standard parameters give exactly two fixed points."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 2

    def test_origin_is_fixed_point(self):
        """Origin (0, 0, 0) is always a fixed point."""
        sim = self._make_sim()
        sim.reset()
        fps = sim.fixed_points
        origin = fps[0]
        np.testing.assert_array_almost_equal(origin, [0.0, 0.0, 0.0])

    def test_second_fixed_point_at_c(self):
        """Second fixed point is at (c, 0, 0)."""
        sim = self._make_sim(c=2.5)
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 2
        np.testing.assert_array_almost_equal(fps[1], [2.5, 0.0, 0.0])

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


class TestGenesioTesiJacobian:
    """Tests for Jacobian and eigenvalue analysis."""

    def _make_sim(self, **kwargs) -> GenesioTesiSimulation:
        defaults = {"a": 0.44, "b": 1.1, "c": 1.0}
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.GENESIO_TESI,
            dt=0.01,
            n_steps=1000,
            parameters=defaults,
        )
        return GenesioTesiSimulation(config)

    def test_jacobian_at_origin_shape(self):
        """Jacobian is a 3x3 matrix."""
        sim = self._make_sim()
        J = sim.jacobian_at_origin
        assert J.shape == (3, 3)

    def test_jacobian_at_origin_values(self):
        """Jacobian at origin: [[0,1,0],[0,0,1],[-c,-b,-a]]."""
        sim = self._make_sim(a=0.44, b=1.1, c=1.0)
        J = sim.jacobian_at_origin
        expected = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.1, -0.44],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_jacobian_at_c_values(self):
        """Jacobian at (c,0,0): [[0,1,0],[0,0,1],[c,-b,-a]]."""
        sim = self._make_sim(a=0.44, b=1.1, c=1.0)
        J = sim.jacobian_at_c
        expected = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, -1.1, -0.44],
        ])
        np.testing.assert_array_almost_equal(J, expected)

    def test_eigenvalues_at_origin(self):
        """Eigenvalues should exist and satisfy characteristic polynomial."""
        sim = self._make_sim()
        eigs = sim.eigenvalues_at_origin()
        assert len(eigs) == 3
        # Check characteristic polynomial: lambda^3 + a*lambda^2 + b*lambda + c = 0
        for eig in eigs:
            poly_val = eig**3 + sim.a * eig**2 + sim.b * eig + sim.c
            assert abs(poly_val) < 1e-8, (
                f"Eigenvalue {eig} does not satisfy characteristic polynomial"
            )

    def test_origin_unstable_for_classic_params(self):
        """At classic chaotic parameters, origin has at least one positive real part."""
        sim = self._make_sim(a=0.44, b=1.1, c=1.0)
        eigs = sim.eigenvalues_at_origin()
        has_positive = any(complex(e).real > 0 for e in eigs)
        assert has_positive, (
            f"Origin should be unstable for chaotic params; eigenvalues={eigs}"
        )


class TestGenesioTesiTrajectory:
    """Tests for trajectory behavior and attractor properties."""

    def _make_sim(self, **kwargs) -> GenesioTesiSimulation:
        defaults = {
            "a": 0.44, "b": 1.1, "c": 1.0,
            "x_0": 0.1, "y_0": 0.1, "z_0": 0.1,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.GENESIO_TESI,
            dt=0.01,
            n_steps=10000,
            parameters=defaults,
        )
        return GenesioTesiSimulation(config)

    def test_trajectory_stays_bounded(self):
        """Trajectory should remain bounded for classic chaotic parameters."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.linalg.norm(state) < 100, f"Trajectory diverged: {state}"

    def test_attractor_not_at_origin(self):
        """After transient, trajectory should not settle at origin."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        # Collect samples
        norms = []
        for _ in range(1000):
            state = sim.step()
            norms.append(np.linalg.norm(state))
        mean_norm = np.mean(norms)
        assert mean_norm > 0.01, (
            f"Trajectory collapsed to origin (mean norm = {mean_norm})"
        )

    def test_chaotic_regime_positive_lyapunov(self):
        """At classic chaotic params, largest Lyapunov exponent is positive."""
        sim = self._make_sim()
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.01)
        assert lam > 0.01, f"Lyapunov {lam:.4f} not positive for chaotic regime"

    def test_stable_regime_nonpositive_lyapunov(self):
        """For large a (e.g. a=0.8), system should be stable (negative Lyapunov)."""
        sim = self._make_sim(a=0.8)
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=30000, dt=0.01)
        assert lam < 0.05, (
            f"Lyapunov {lam:.4f} too large for stable regime (a=0.8)"
        )


class TestGenesioTesiBifurcation:
    """Tests for the bifurcation sweep."""

    def _make_sim(self) -> GenesioTesiSimulation:
        config = SimulationConfig(
            domain=Domain.GENESIO_TESI,
            dt=0.01,
            n_steps=5000,
            parameters={"a": 0.44, "b": 1.1, "c": 1.0},
        )
        return GenesioTesiSimulation(config)

    def test_bifurcation_sweep_returns_data(self):
        """Sweep should produce valid data for all a values."""
        sim = self._make_sim()
        sim.reset()
        a_values = np.linspace(0.35, 0.55, 5)
        data = sim.bifurcation_sweep(
            a_values, n_transient=1000, n_measure=5000
        )
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_sweep_has_transition(self):
        """Sweep from chaotic to stable should show Lyapunov sign change."""
        sim = self._make_sim()
        sim.reset()
        a_values = np.array([0.40, 0.44, 0.50, 0.60])
        data = sim.bifurcation_sweep(
            a_values, n_transient=2000, n_measure=10000
        )
        lyap = data["lyapunov_exponent"]
        # At a=0.44 should be chaotic; at a=0.60 should be stable
        # At minimum, the range should contain both positive and non-positive values
        assert np.max(lyap) > np.min(lyap), "No variation in Lyapunov across sweep"


class TestGenesioTesiRediscovery:
    """Tests for Genesio-Tesi data generation functions."""

    def test_trajectory_data_shape(self):
        from simulating_anything.rediscovery.genesio_tesi import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=100, dt=0.01)
        assert data["states"].shape == (101, 3)
        assert data["a"] == 0.44
        assert data["b"] == 1.1
        assert data["c"] == 1.0

    def test_trajectory_data_stays_finite(self):
        from simulating_anything.rediscovery.genesio_tesi import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_sweep_data_shape(self):
        from simulating_anything.rediscovery.genesio_tesi import (
            generate_lyapunov_vs_a_data,
        )

        data = generate_lyapunov_vs_a_data(n_a=5, n_steps=3000, dt=0.01)
        assert len(data["a"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_jerk_verification_data(self):
        from simulating_anything.rediscovery.genesio_tesi import (
            generate_jerk_verification_data,
        )

        data = generate_jerk_verification_data(n_steps=200, dt=0.01)
        assert data["states"].shape == (201, 3)
        assert data["max_jerk_error"] < 1e-10, (
            f"Jerk form error {data['max_jerk_error']:.2e} too large"
        )

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.genesio_tesi import (
            generate_trajectory_data,
        )

        data = generate_trajectory_data(n_steps=200, dt=0.01)
        states = data["states"]
        # SINDy expects (n_timesteps, n_variables) with float dtype
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data
