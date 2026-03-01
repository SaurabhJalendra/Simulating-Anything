"""Tests for the swinging Atwood machine simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.swinging_atwood import SwingingAtwoodSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestSwingingAtwoodSimulation:
    """Tests for the swinging Atwood machine simulation."""

    def _make_sim(self, **kwargs) -> SwingingAtwoodSimulation:
        defaults = {
            "M": 3.0, "m": 1.0, "g": 9.81,
            "r_min": 0.1, "r_0": 1.0, "theta_0": 0.5,
            "r_dot_0": 0.0, "theta_dot_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.SWINGING_ATWOOD,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return SwingingAtwoodSimulation(config)

    def test_creation(self):
        """SwingingAtwoodSimulation should be created with default parameters."""
        sim = self._make_sim()
        assert sim.M == 3.0
        assert sim.m == 1.0
        assert sim.g == 9.81
        assert sim.r_min == 0.1

    def test_custom_parameters(self):
        """Simulation should accept custom parameters."""
        sim = self._make_sim(M=5.0, m=2.0, g=10.0)
        assert sim.M == 5.0
        assert sim.m == 2.0
        assert sim.g == 10.0

    def test_reset_state(self):
        """Reset should initialize to configured initial conditions."""
        sim = self._make_sim(
            r_0=1.5, theta_0=0.3, r_dot_0=0.1, theta_dot_0=-0.2,
        )
        state = sim.reset()
        assert np.isclose(state[0], 1.5)    # r
        assert np.isclose(state[1], 0.3)    # theta
        assert np.isclose(state[2], 0.1)    # r_dot
        assert np.isclose(state[3], -0.2)   # theta_dot

    def test_observe_shape(self):
        """Observe should return 4-element state vector."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_advances(self):
        """A single step should change the state."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_deterministic(self):
        """Same initial conditions should produce same trajectory."""
        sim1 = self._make_sim()
        sim1.reset()
        states1 = []
        for _ in range(100):
            states1.append(sim1.step().copy())

        sim2 = self._make_sim()
        sim2.reset()
        states2 = []
        for _ in range(100):
            states2.append(sim2.step().copy())

        for s1, s2 in zip(states1, states2):
            np.testing.assert_array_equal(s1, s2)

    def test_mass_ratio(self):
        """mass_ratio() should return M/m."""
        sim = self._make_sim(M=6.0, m=2.0)
        assert np.isclose(sim.mass_ratio(), 3.0)

    def test_mass_ratio_default(self):
        """Default mass ratio should be 3.0 (M=3, m=1)."""
        sim = self._make_sim()
        assert np.isclose(sim.mass_ratio(), 3.0)

    def test_energy_conservation(self):
        """Energy should be conserved to high precision when no wall bounce.

        Use M < m so the swinging mass side is heavier and r increases
        (no collision with pulley), giving clean Hamiltonian dynamics.
        """
        sim = self._make_sim(
            M=1.0, m=2.0, r_0=1.5, theta_0=0.3,
            r_dot_0=0.0, theta_dot_0=0.0,
        )
        sim.reset()
        E0 = sim.total_energy()

        for _ in range(10000):
            sim.step()

        E_final = sim.total_energy()
        rel_drift = abs(E_final - E0) / max(abs(E0), 1e-10)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_energy_conservation_large_theta(self):
        """Energy conservation should hold for larger initial angle."""
        sim = self._make_sim(
            M=2.0, m=1.0, r_0=1.5, theta_0=1.0,
            r_dot_0=0.1, theta_dot_0=0.2,
        )
        sim.reset()
        E0 = sim.total_energy()

        for _ in range(10000):
            sim.step()

        E_final = sim.total_energy()
        rel_drift = abs(E_final - E0) / max(abs(E0), 1e-10)
        assert rel_drift < 1e-5, f"Energy drift {rel_drift:.2e} too large"

    def test_r_positive(self):
        """r should stay above r_min at all times."""
        sim = self._make_sim(
            M=1.0, m=3.0, r_0=0.5, theta_0=0.1,
            r_dot_0=-0.5, theta_dot_0=0.0,
        )
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= sim.r_min, (
                f"r = {state[0]} < r_min = {sim.r_min}"
            )

    def test_rk4_stability(self):
        """No NaN or Inf after many steps."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_theta_evolves(self):
        """Angle should change over time for nonzero initial angle."""
        sim = self._make_sim(theta_0=0.5)
        sim.reset()
        theta_values = [sim.observe()[1]]
        for _ in range(1000):
            state = sim.step()
            theta_values.append(state[1])
        theta_values = np.array(theta_values)
        # Theta should not be constant
        assert np.std(theta_values) > 1e-6, "Theta did not evolve"

    def test_vertical_equilibrium(self):
        """theta=0, r_dot=0 is equilibrium when M > m and suitable r_dot.

        With theta=0 and theta_dot=0, the system reduces to
        r'' = (m*g - M*g)/(M+m). For M > m, r'' < 0 so the swinging
        mass moves up (r decreases). This is NOT a static equilibrium
        unless M = m. So we just verify the system evolves smoothly.
        """
        sim = self._make_sim(
            M=3.0, m=1.0, r_0=1.0, theta_0=0.0,
            r_dot_0=0.0, theta_dot_0=0.0,
        )
        sim.reset()

        # With theta=0, theta_dot=0: r'' = (m*g*cos(0) - M*g)/(M+m)
        # = (1*9.81 - 3*9.81)/4 = -4.905 m/s^2
        # After 1 step of dt=0.001: r_dot ~ -0.004905
        state = sim.step()
        assert np.all(np.isfinite(state))
        # r should decrease since M > m
        assert state[0] < 1.0, "r should decrease when M > m and theta = 0"

    def test_bounded_motion(self):
        """r should stay in a reasonable range for standard ICs."""
        sim = self._make_sim(
            M=3.0, m=1.0, r_0=1.0, theta_0=0.5,
        )
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            # r should stay bounded (not diverge to infinity)
            assert state[0] < 100.0, f"r diverged: {state[0]}"
            assert state[0] >= sim.r_min, f"r below minimum: {state[0]}"

    def test_equations_of_motion(self):
        """Verify the RHS at a known state."""
        sim = self._make_sim(M=3.0, m=1.0, g=9.81)
        sim.reset()

        # At state [r=1.0, theta=0, r_dot=0, theta_dot=0]:
        # r'' = (1*1.0*0 + 1*9.81*cos(0) - 3*9.81) / (3+1)
        #     = (0 + 9.81 - 29.43) / 4 = -19.62/4 = -4.905
        # theta'' = (-2*0*0 - 9.81*sin(0)) / 1.0 = 0
        derivs = sim._derivatives(np.array([1.0, 0.0, 0.0, 0.0]))
        assert np.isclose(derivs[0], 0.0)   # dr/dt = r_dot = 0
        assert np.isclose(derivs[1], 0.0)   # dtheta/dt = theta_dot = 0
        assert np.isclose(derivs[2], -4.905, rtol=1e-6)  # r_ddot
        assert np.isclose(derivs[3], 0.0)   # theta_ddot

    def test_equations_of_motion_nonzero_theta(self):
        """Verify RHS at a state with nonzero theta."""
        sim = self._make_sim(M=3.0, m=1.0, g=10.0)
        sim.reset()

        # At state [r=2.0, theta=pi/2, r_dot=0.1, theta_dot=0.5]:
        # r'' = (1*2.0*0.25 + 1*10*cos(pi/2) - 3*10) / 4
        #     = (0.5 + 0 - 30) / 4 = -29.5/4 = -7.375
        # theta'' = (-2*0.1*0.5 - 10*sin(pi/2)) / 2.0
        #         = (-0.1 - 10) / 2.0 = -10.1/2.0 = -5.05
        derivs = sim._derivatives(
            np.array([2.0, np.pi / 2, 0.1, 0.5])
        )
        assert np.isclose(derivs[0], 0.1)     # dr/dt = r_dot
        assert np.isclose(derivs[1], 0.5)     # dtheta/dt = theta_dot
        assert np.isclose(derivs[2], -7.375, rtol=1e-6)
        assert np.isclose(derivs[3], -5.05, rtol=1e-6)

    def test_long_trajectory(self):
        """Simulation should handle 50000 steps without failure."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(50000):
            state = sim.step()
        assert np.all(np.isfinite(state))

    def test_integrable_case(self):
        """mu = 1 should have special (integrable) dynamics.

        For mu = 1, the system has additional constants of motion.
        We verify the trajectory stays finite and smooth.
        """
        sim = self._make_sim(M=1.0, m=1.0, r_0=1.0, theta_0=0.5)
        sim.reset()
        E0 = sim.total_energy()

        for _ in range(20000):
            state = sim.step()
            assert np.all(np.isfinite(state))

        E_final = sim.total_energy()
        rel_drift = abs(E_final - E0) / max(abs(E0), 1e-10)
        assert rel_drift < 1e-5, f"Energy drift at mu=1: {rel_drift:.2e}"

    def test_angular_momentum(self):
        """Angular momentum should be computable and vary over time."""
        sim = self._make_sim(theta_0=0.5, theta_dot_0=1.0)
        sim.reset()
        L0 = sim.angular_momentum()

        for _ in range(1000):
            sim.step()
        L1 = sim.angular_momentum()

        # Angular momentum is NOT conserved (gravity provides torque)
        # so it should change
        assert abs(L0) > 0, "Initial angular momentum should be nonzero"
        assert L0 != L1, "Angular momentum should change over time"

    def test_cartesian_position(self):
        """Cartesian position should satisfy x^2 + y^2 = r^2."""
        sim = self._make_sim(r_0=2.0, theta_0=0.7)
        sim.reset()
        x, y = sim.cartesian_position()
        r = sim.observe()[0]
        assert np.isclose(x**2 + y**2, r**2, rtol=1e-10)

    def test_cartesian_at_vertical(self):
        """At theta=0, mass should be directly below pulley."""
        sim = self._make_sim(r_0=2.0, theta_0=0.0)
        sim.reset()
        x, y = sim.cartesian_position()
        assert np.isclose(x, 0.0, atol=1e-10)
        assert np.isclose(y, -2.0, atol=1e-10)

    def test_multiple_mass_ratios(self):
        """Different mu values should produce different dynamics."""
        trajectories = {}
        for mu in [0.5, 1.0, 2.0, 3.0]:
            sim = self._make_sim(M=mu, m=1.0)
            sim.reset()
            for _ in range(500):
                sim.step()
            trajectories[mu] = sim.observe().copy()

        # States after 500 steps should differ for different mu
        assert not np.allclose(trajectories[0.5], trajectories[2.0])
        assert not np.allclose(trajectories[1.0], trajectories[3.0])

    def test_run_trajectory(self):
        """run() should produce a TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)
        assert len(traj.timestamps) == 101

    def test_trajectory_finite(self):
        """All trajectory states should be finite."""
        sim = self._make_sim()
        traj = sim.run(n_steps=5000)
        assert np.all(np.isfinite(traj.states))


class TestSwingingAtwoodRediscovery:
    """Tests for swinging Atwood data generation."""

    def test_energy_conservation_data(self):
        from simulating_anything.rediscovery.swinging_atwood import (
            generate_energy_conservation_data,
        )
        data = generate_energy_conservation_data(
            n_trajectories=3, n_steps=1000, dt=0.001,
        )
        assert "final_drift" in data
        assert "max_drift" in data
        assert len(data["final_drift"]) == 3
        # Some trajectories may hit the elastic wall, so allow moderate drift.
        # The median drift should still be small for the non-bouncing cases.
        assert np.median(data["max_drift"]) < 0.1, (
            f"Median energy drift too large: {np.median(data['max_drift']):.2e}"
        )

    def test_mass_ratio_sweep_data(self):
        from simulating_anything.rediscovery.swinging_atwood import (
            generate_mass_ratio_sweep_data,
        )
        data = generate_mass_ratio_sweep_data(
            n_mu=3, n_steps=5000, dt=0.001,
        )
        assert "mu" in data
        assert "lyapunov_exponent" in data
        assert len(data["mu"]) == 3
        assert len(data["lyapunov_exponent"]) == 3
        # All Lyapunov exponents should be finite
        assert np.all(np.isfinite(data["lyapunov_exponent"]))
