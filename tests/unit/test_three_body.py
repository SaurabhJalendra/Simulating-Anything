"""Tests for the restricted three-body (CR3BP) gravitational simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.three_body import ThreeBodySimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestThreeBodySimulation:
    """Comprehensive tests for the CR3BP simulation."""

    def setup_method(self):
        """Create a default simulation config for each test."""
        self.config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.001,
            n_steps=10000,
            parameters={
                "mu": 0.01,
                "x0": 0.5,
                "y0": 0.5,
                "vx0": 0.0,
                "vy0": 0.0,
            },
        )

    def _make_sim(self, **kwargs) -> ThreeBodySimulation:
        defaults = {
            "mu": 0.01,
            "x0": 0.5,
            "y0": 0.5,
            "vx0": 0.0,
            "vy0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return ThreeBodySimulation(config)

    # ---- Instantiation tests ----

    def test_creation_default(self):
        """Simulation should be created with default parameters."""
        sim = ThreeBodySimulation(self.config)
        assert sim.mu == 0.01
        assert sim.x0 == 0.5
        assert sim.y0 == 0.5
        assert sim.vx0 == 0.0
        assert sim.vy0 == 0.0
        assert sim.dt == 0.001

    def test_creation_empty_params(self):
        """Empty parameters dict should use built-in defaults."""
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = ThreeBodySimulation(config)
        assert sim.mu == 0.01
        assert sim.x0 == 0.5
        assert sim.y0 == 0.5
        assert sim.vx0 == 0.0
        assert sim.vy0 == 0.0

    def test_creation_custom_params(self):
        """Custom parameters should override defaults."""
        sim = self._make_sim(mu=0.1, x0=1.0, y0=-0.3, vx0=0.1, vy0=-0.2)
        assert sim.mu == 0.1
        assert sim.x0 == 1.0
        assert sim.y0 == -0.3
        assert sim.vx0 == 0.1
        assert sim.vy0 == -0.2

    # ---- Reset tests ----

    def test_reset_shape(self):
        """Reset should return a state array of shape (4,)."""
        sim = self._make_sim()
        state = sim.reset()
        assert isinstance(state, np.ndarray)
        assert state.shape == (4,)

    def test_reset_values(self):
        """Reset should return state matching initial conditions."""
        sim = self._make_sim(x0=1.2, y0=-0.7, vx0=0.3, vy0=-0.1)
        state = sim.reset()
        np.testing.assert_allclose(state, [1.2, -0.7, 0.3, -0.1])

    def test_reset_returns_copy(self):
        """Reset should return a copy, not a reference to internal state."""
        sim = self._make_sim()
        state = sim.reset()
        state[0] = 999.0
        obs = sim.observe()
        assert obs[0] != 999.0

    # ---- Step tests ----

    def test_step_shape(self):
        """Step should return the same shape as reset."""
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert state1.shape == state0.shape

    def test_step_no_nan(self):
        """A single step should not produce NaN or Inf."""
        sim = self._make_sim()
        sim.reset()
        state = sim.step()
        assert np.all(np.isfinite(state))

    def test_step_advances_state(self):
        """A single step should change the state."""
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_step_returns_copy(self):
        """Step should return a copy, not a reference to internal state."""
        sim = self._make_sim()
        sim.reset()
        state1 = sim.step()
        state1[0] = 999.0
        obs = sim.observe()
        assert obs[0] != 999.0

    # ---- Observe tests ----

    def test_observe_shape(self):
        """Observe should return shape (4,)."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_observe_matches_reset(self):
        """Observe immediately after reset should return the initial state."""
        sim = self._make_sim()
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_observe_matches_step(self):
        """Observe after a step should match the step return value."""
        sim = self._make_sim()
        sim.reset()
        state = sim.step()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    # ---- Multiple steps / trajectory tests ----

    def test_multiple_steps_no_crash(self):
        """Running 1000 steps should not crash or produce NaN."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(1000):
            state = sim.step()
        assert np.all(np.isfinite(state))

    def test_trajectory_bounded(self):
        """Trajectory should remain bounded over moderate integration."""
        sim = self._make_sim(x0=0.8, y0=0.3, vx0=-0.1, vy0=0.2)
        sim.reset()
        for _ in range(500):
            state = sim.step()
            assert np.all(np.abs(state) < 100), f"State diverged: {state}"

    def test_run_trajectory(self):
        """run() should produce a valid TrajectoryData with correct shape."""
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.01,
            n_steps=200,
            parameters={"mu": 0.01, "x0": 0.5, "y0": 0.5, "vx0": 0.1, "vy0": -0.1},
        )
        sim = ThreeBodySimulation(config)
        traj = sim.run(n_steps=200)
        # 201 rows: 1 initial + 200 steps
        assert traj.states.shape == (201, 4)
        assert len(traj.timestamps) == 201
        assert np.all(np.isfinite(traj.states))

    # ---- Determinism tests ----

    def test_determinism_same_seed(self):
        """Same initial conditions should produce identical trajectories."""
        sim1 = self._make_sim()
        sim1.reset(seed=42)
        states1 = [sim1.step().copy() for _ in range(100)]

        sim2 = self._make_sim()
        sim2.reset(seed=42)
        states2 = [sim2.step().copy() for _ in range(100)]

        for s1, s2 in zip(states1, states2):
            np.testing.assert_array_equal(s1, s2)

    def test_determinism_reset(self):
        """Resetting and re-running should reproduce the same trajectory."""
        sim = self._make_sim()
        sim.reset()
        states1 = [sim.step().copy() for _ in range(50)]

        sim.reset()
        states2 = [sim.step().copy() for _ in range(50)]

        for s1, s2 in zip(states1, states2):
            np.testing.assert_array_equal(s1, s2)

    # ---- Jacobi constant conservation ----

    def test_jacobi_constant_type(self):
        """jacobi_constant should return a float."""
        sim = self._make_sim()
        sim.reset()
        cj = sim.jacobi_constant()
        assert isinstance(cj, (float, np.floating))

    def test_jacobi_constant_explicit_state(self):
        """jacobi_constant should accept an explicit state array."""
        sim = self._make_sim()
        sim.reset()
        state = np.array([0.5, 0.5, 0.0, 0.0])
        cj = sim.jacobi_constant(state)
        assert np.isfinite(cj)

    def test_jacobi_constant_conservation_short(self):
        """Jacobi constant should be approximately conserved over a short run.

        The CR3BP conserves the Jacobi constant exactly in continuous time.
        With RK4 and small dt, drift should be very small.
        """
        sim = self._make_sim(mu=0.01, x0=0.8, y0=0.3, vx0=-0.1, vy0=0.2)
        sim.reset()
        cj_initial = sim.jacobi_constant()

        for _ in range(500):
            sim.step()
        cj_final = sim.jacobi_constant()

        assert abs(cj_final - cj_initial) < 1e-6, (
            f"Jacobi constant drift: {abs(cj_final - cj_initial):.2e}"
        )

    def test_jacobi_constant_conservation_long(self):
        """Jacobi constant should be well-conserved over a longer integration."""
        sim = self._make_sim(mu=0.01, x0=0.6, y0=0.1, vx0=0.0, vy0=0.5)
        sim.reset()
        cj_initial = sim.jacobi_constant()

        max_drift = 0.0
        for _ in range(2000):
            sim.step()
            cj = sim.jacobi_constant()
            drift = abs(cj - cj_initial)
            max_drift = max(max_drift, drift)

        assert max_drift < 1e-4, f"Max Jacobi constant drift: {max_drift:.2e}"

    # ---- Lagrange point tests ----

    def test_lagrange_points_returns_dict(self):
        """lagrange_points should return a dict with keys L1 through L5."""
        sim = self._make_sim()
        lps = sim.lagrange_points()
        assert isinstance(lps, dict)
        for key in ["L1", "L2", "L3", "L4", "L5"]:
            assert key in lps
            assert isinstance(lps[key], np.ndarray)
            assert lps[key].shape == (2,)

    def test_l4_exact(self):
        """L4 should be at (0.5 - mu, sqrt(3)/2) exactly."""
        mu = 0.01
        sim = self._make_sim(mu=mu)
        lps = sim.lagrange_points()
        expected_l4 = np.array([0.5 - mu, np.sqrt(3) / 2])
        np.testing.assert_allclose(lps["L4"], expected_l4, atol=1e-14)

    def test_l5_exact(self):
        """L5 should be at (0.5 - mu, -sqrt(3)/2) exactly."""
        mu = 0.01
        sim = self._make_sim(mu=mu)
        lps = sim.lagrange_points()
        expected_l5 = np.array([0.5 - mu, -np.sqrt(3) / 2])
        np.testing.assert_allclose(lps["L5"], expected_l5, atol=1e-14)

    def test_l4_l5_symmetric(self):
        """L4 and L5 should be mirror images across the x-axis."""
        sim = self._make_sim(mu=0.05)
        lps = sim.lagrange_points()
        assert np.isclose(lps["L4"][0], lps["L5"][0])
        assert np.isclose(lps["L4"][1], -lps["L5"][1])

    def test_collinear_points_on_x_axis(self):
        """L1, L2, L3 should all have y=0 (collinear on x-axis)."""
        sim = self._make_sim(mu=0.01)
        lps = sim.lagrange_points()
        for key in ["L1", "L2", "L3"]:
            assert np.isclose(lps[key][1], 0.0, atol=1e-10), (
                f"{key} y-coordinate: {lps[key][1]}"
            )

    def test_collinear_ordering(self):
        """Collinear Lagrange points should satisfy L3 < L1 < L2 for small mu.

        L3 is beyond the larger mass (near x ~ -1),
        L1 is between the two masses,
        L2 is beyond the smaller mass.
        """
        sim = self._make_sim(mu=0.01)
        lps = sim.lagrange_points()
        assert lps["L3"][0] < lps["L1"][0] < lps["L2"][0], (
            f"L3={lps['L3'][0]:.4f}, L1={lps['L1'][0]:.4f}, L2={lps['L2'][0]:.4f}"
        )

    def test_l1_between_primaries(self):
        """L1 should lie between the two primary bodies."""
        mu = 0.01
        sim = self._make_sim(mu=mu)
        lps = sim.lagrange_points()
        l1_x = lps["L1"][0]
        # m1 is at x = -mu, m2 is at x = 1-mu
        assert -mu < l1_x < (1 - mu)

    def test_lagrange_points_different_mu(self):
        """Lagrange points should shift with changing mass ratio."""
        sim1 = self._make_sim(mu=0.01)
        sim2 = self._make_sim(mu=0.1)
        lps1 = sim1.lagrange_points()
        lps2 = sim2.lagrange_points()
        # L4 x-coordinate = 0.5 - mu, so different mu gives different L4
        assert not np.isclose(lps1["L4"][0], lps2["L4"][0])

    # ---- Lyapunov exponent tests ----

    def test_lyapunov_returns_float(self):
        """compute_lyapunov should return a finite float."""
        sim = self._make_sim(x0=0.5, y0=0.5, vx0=0.3, vy0=-0.3)
        lyap = sim.compute_lyapunov(n_steps=500)
        assert isinstance(lyap, (float, np.floating))
        assert np.isfinite(lyap)

    def test_lyapunov_custom_dt(self):
        """compute_lyapunov should accept a custom dt parameter."""
        sim = self._make_sim()
        lyap = sim.compute_lyapunov(n_steps=200, dt=0.005)
        assert np.isfinite(lyap)

    # ---- Derivatives / equations of motion tests ----

    def test_derivatives_shape(self):
        """_derivatives should return shape (4,)."""
        sim = self._make_sim()
        state = np.array([0.5, 0.5, 0.0, 0.0])
        derivs = sim._derivatives(state)
        assert derivs.shape == (4,)

    def test_derivatives_velocity_components(self):
        """First two components of the derivative should equal vx and vy.

        The equations of motion are dx/dt = vx, dy/dt = vy.
        """
        sim = self._make_sim()
        vx, vy = 0.3, -0.7
        state = np.array([0.5, 0.5, vx, vy])
        derivs = sim._derivatives(state)
        assert np.isclose(derivs[0], vx)
        assert np.isclose(derivs[1], vy)

    def test_derivatives_at_l4_equilibrium(self):
        """At L4 with zero velocity, all derivatives should vanish.

        L4 is an equilibrium of the effective potential. With zero velocity,
        the Coriolis terms (which depend on v) are also zero.
        """
        mu = 0.01
        sim = self._make_sim(mu=mu)
        l4_x = 0.5 - mu
        l4_y = np.sqrt(3) / 2
        state = np.array([l4_x, l4_y, 0.0, 0.0])
        derivs = sim._derivatives(state)
        np.testing.assert_allclose(derivs, [0.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_derivatives_singularity_protection(self):
        """Derivatives should not produce NaN even near a primary body.

        The code clamps r1, r2 to a minimum of 1e-10 to avoid division by zero.
        """
        sim = self._make_sim(mu=0.01)
        mu = sim.mu
        # Place test mass exactly on m1 at (-mu, 0)
        state = np.array([-mu, 0.0, 0.0, 0.0])
        derivs = sim._derivatives(state)
        assert np.all(np.isfinite(derivs)), f"NaN at primary: {derivs}"

    # ---- RK4 integrator test ----

    def test_rk4_step_finite(self):
        """A single RK4 step should produce finite output."""
        sim = self._make_sim()
        state = np.array([0.5, 0.5, 0.0, 0.0])
        new_state = sim._rk4_step(state, 0.001)
        assert np.all(np.isfinite(new_state))
        assert new_state.shape == (4,)

    # ---- Physical consistency tests ----

    def test_symmetric_mu_half(self):
        """For equal masses (mu=0.5), L4 and L5 should have x=0.

        L4 = (0.5 - 0.5, sqrt(3)/2) = (0, sqrt(3)/2).
        """
        sim = self._make_sim(mu=0.5)
        lps = sim.lagrange_points()
        assert np.isclose(lps["L4"][0], 0.0)
        assert np.isclose(lps["L5"][0], 0.0)

    def test_jacobi_at_l4(self):
        """The Jacobi constant at L4 should have a known approximate value.

        At L4 with zero velocity: C_J = 2 * Omega(L4).
        For the standard CR3BP normalization, C_J(L4) is near 3.
        """
        mu = 0.01
        sim = self._make_sim(mu=mu)
        l4_state = np.array([0.5 - mu, np.sqrt(3) / 2, 0.0, 0.0])
        cj = sim.jacobi_constant(l4_state)
        assert np.isfinite(cj)
        assert 2.5 < cj < 3.5, f"C_J at L4 = {cj:.4f}, expected near 3"

    def test_step_count_increments(self):
        """Internal step counter should increment with each step."""
        sim = self._make_sim()
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2

    def test_different_initial_conditions_differ(self):
        """Different initial conditions should produce different trajectories."""
        sim1 = self._make_sim(x0=0.5, y0=0.5)
        sim2 = self._make_sim(x0=0.8, y0=0.2)
        sim1.reset()
        sim2.reset()
        for _ in range(10):
            s1 = sim1.step()
            s2 = sim2.step()
        assert not np.allclose(s1, s2)

    def test_small_dt_better_conservation(self):
        """Smaller dt should give better Jacobi constant conservation.

        This validates that the RK4 integrator converges as expected (4th order).
        """
        drifts = []
        for dt_val in [0.01, 0.001]:
            config = SimulationConfig(
                domain=Domain.CUSTOM,
                dt=dt_val,
                n_steps=10000,
                parameters={
                    "mu": 0.01,
                    "x0": 0.8,
                    "y0": 0.3,
                    "vx0": -0.1,
                    "vy0": 0.2,
                },
            )
            sim = ThreeBodySimulation(config)
            sim.reset()
            cj0 = sim.jacobi_constant()
            # Integrate for the same total time (1 unit)
            n = int(1.0 / dt_val)
            for _ in range(n):
                sim.step()
            cj_end = sim.jacobi_constant()
            drifts.append(abs(cj_end - cj0))

        assert drifts[1] < drifts[0], (
            f"Smaller dt drift ({drifts[1]:.2e}) should be less than "
            f"larger dt drift ({drifts[0]:.2e})"
        )

    def test_orbit_near_l4_bounded(self):
        """Small displacement from L4 should remain bounded for small mu.

        For mu < mu_Routh ~ 0.0385 (Gascheau limit), L4 is linearly stable
        and orbits starting nearby should stay within a finite envelope.
        """
        mu = 0.01
        config = SimulationConfig(
            domain=Domain.CUSTOM,
            dt=0.001,
            n_steps=20000,
            parameters={
                "mu": mu,
                "x0": 0.5 - mu + 0.01,
                "y0": np.sqrt(3) / 2 + 0.01,
                "vx0": 0.0,
                "vy0": 0.0,
            },
        )
        sim = ThreeBodySimulation(config)
        sim.reset()
        max_displacement = 0.0
        l4_x = 0.5 - mu
        l4_y = np.sqrt(3) / 2
        for _ in range(5000):
            state = sim.step()
            disp = np.sqrt((state[0] - l4_x) ** 2 + (state[1] - l4_y) ** 2)
            max_displacement = max(max_displacement, disp)

        assert max_displacement < 1.0, (
            f"Max displacement from L4 = {max_displacement:.4f}"
        )

    def test_different_mass_ratios(self):
        """Simulation should work for various mass ratios without NaN."""
        for mu in [0.001, 0.01, 0.1, 0.3, 0.5]:
            sim = self._make_sim(mu=mu)
            sim.reset()
            for _ in range(100):
                state = sim.step()
            assert np.all(np.isfinite(state)), f"NaN for mu={mu}"

    def test_jacobi_constant_zero_velocity(self):
        """With zero velocity, Jacobi constant equals twice the pseudo-potential."""
        sim = self._make_sim(mu=0.01)
        x, y = 0.7, 0.4
        state_zero_v = np.array([x, y, 0.0, 0.0])
        cj = sim.jacobi_constant(state_zero_v)

        # Manually compute 2*Omega
        mu = 0.01
        r1 = np.sqrt((x + mu) ** 2 + y ** 2)
        r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
        omega = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / r1 + mu / r2
        expected_cj = 2 * omega

        np.testing.assert_allclose(cj, expected_cj, rtol=1e-12)

    def test_jacobi_decreases_with_velocity(self):
        """Adding velocity should decrease the Jacobi constant.

        C_J = -(vx^2 + vy^2) + 2*Omega, so more kinetic energy means lower C_J.
        """
        sim = self._make_sim(mu=0.01)
        x, y = 0.6, 0.3
        cj_still = sim.jacobi_constant(np.array([x, y, 0.0, 0.0]))
        cj_moving = sim.jacobi_constant(np.array([x, y, 0.5, 0.5]))
        assert cj_moving < cj_still
