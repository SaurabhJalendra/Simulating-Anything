"""Tests for the cart-pole (inverted pendulum on cart) simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.cart_pole import CartPole
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestCartPole:
    """Tests for the cart-pole simulation."""

    def _make_sim(self, **kwargs) -> CartPole:
        defaults = {
            "M": 1.0, "m": 0.1, "L": 0.5, "g": 9.81,
            "mu_c": 0.0, "mu_p": 0.0, "F": 0.0,
            "x_0": 0.0, "x_dot_0": 0.0,
            "theta_0": 0.1, "theta_dot_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.CART_POLE,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return CartPole(config)

    def test_creation(self):
        """CartPole should be created with default parameters."""
        sim = self._make_sim()
        assert sim.M == 1.0
        assert sim.m == 0.1
        assert sim.L == 0.5
        assert sim.g == 9.81
        assert sim.mu_c == 0.0
        assert sim.mu_p == 0.0
        assert sim.F == 0.0

    def test_custom_parameters(self):
        """CartPole should accept custom parameters."""
        sim = self._make_sim(M=2.0, m=0.5, L=1.0, g=10.0)
        assert sim.M == 2.0
        assert sim.m == 0.5
        assert sim.L == 1.0
        assert sim.g == 10.0

    def test_initial_state_shape(self):
        """State should be shape (4,): [x, x_dot, theta, theta_dot]."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (4,)

    def test_initial_state_values(self):
        """Initial state should match configured values."""
        sim = self._make_sim(x_0=1.0, x_dot_0=0.5, theta_0=0.2, theta_dot_0=-0.1)
        state = sim.reset()
        assert np.isclose(state[0], 1.0)   # x
        assert np.isclose(state[1], 0.5)   # x_dot
        assert np.isclose(state[2], 0.2)   # theta
        assert np.isclose(state[3], -0.1)  # theta_dot

    def test_observe_shape(self):
        """Observe should return shape (4,)."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_advances_state(self):
        """A single step should change the state."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_energy_conservation_no_friction(self):
        """Energy should be conserved when friction = 0 and F = 0."""
        sim = self._make_sim(
            M=1.0, m=0.1, L=0.5, mu_c=0.0, mu_p=0.0, F=0.0,
            theta_0=0.3, theta_dot_0=0.0,
        )
        sim.reset()
        E0 = sim.total_energy
        for _ in range(10000):
            sim.step()
        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / max(abs(E0), 1e-10)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_energy_conservation_large_amplitude(self):
        """Energy conservation should hold for larger oscillations too."""
        sim = self._make_sim(
            M=2.0, m=0.5, L=1.0, mu_c=0.0, mu_p=0.0, F=0.0,
            theta_0=1.0, theta_dot_0=0.5, x_dot_0=0.3,
        )
        sim.reset()
        E0 = sim.total_energy
        for _ in range(10000):
            sim.step()
        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / max(abs(E0), 1e-10)
        assert rel_drift < 1e-5, f"Energy drift {rel_drift:.2e} too large"

    def test_energy_decreases_with_friction(self):
        """Energy should decrease when friction is present."""
        sim = self._make_sim(
            M=1.0, m=0.1, L=0.5,
            mu_c=0.5, mu_p=0.1, F=0.0,
            theta_0=0.5, theta_dot_0=0.0,
        )
        sim.reset()
        E0 = sim.total_energy
        for _ in range(5000):
            sim.step()
        E_final = sim.total_energy
        # With friction, energy should decrease (or at least not increase)
        assert E_final < E0 + 1e-10, "Energy should not increase with friction"

    def test_small_angle_frequency_property(self):
        """small_angle_frequency should match sqrt(g*(M+m)/(M*L))."""
        sim = self._make_sim(M=1.0, m=0.1, L=0.5, g=9.81)
        expected = np.sqrt(9.81 * (1.0 + 0.1) / (1.0 * 0.5))
        assert np.isclose(sim.small_angle_frequency, expected, rtol=1e-10)

    def test_small_angle_frequency_matches_simulation(self):
        """Measured oscillation frequency should match theory for small angles."""
        M, m, L, g = 1.0, 0.1, 0.5, 9.81
        dt = 0.0005
        n_steps = 40000
        sim = self._make_sim(
            M=M, m=m, L=L, g=g,
            theta_0=0.02, theta_dot_0=0.0,  # Very small angle
        )
        config = SimulationConfig(
            domain=Domain.CART_POLE, dt=dt, n_steps=n_steps,
            parameters={
                "M": M, "m": m, "L": L, "g": g,
                "mu_c": 0.0, "mu_p": 0.0, "F": 0.0,
                "x_0": 0.0, "x_dot_0": 0.0,
                "theta_0": np.pi + 0.02, "theta_dot_0": 0.0,
            },
        )
        sim = CartPole(config)
        sim.reset()

        # Collect theta
        thetas = [sim.observe()[2]]
        for _ in range(n_steps):
            state = sim.step()
            thetas.append(state[2])

        thetas = np.array(thetas)
        # Track deviation from the stable equilibrium at theta=pi
        delta = thetas - np.pi

        # Find period from zero crossings of the deviation
        crossings = []
        for j in range(1, len(delta)):
            if delta[j - 1] < 0 and delta[j] >= 0:
                frac = -delta[j - 1] / (delta[j] - delta[j - 1])
                crossings.append((j - 1 + frac) * dt)

        assert len(crossings) >= 3, f"Not enough zero crossings: {len(crossings)}"
        periods = np.diff(crossings)
        T_measured = float(np.median(periods))
        omega_measured = 2 * np.pi / T_measured
        omega_theory = np.sqrt(g * (M + m) / (M * L))

        rel_error = abs(omega_measured - omega_theory) / omega_theory
        assert rel_error < 0.01, (
            f"Frequency mismatch: measured={omega_measured:.4f}, "
            f"theory={omega_theory:.4f}, error={rel_error:.4%}"
        )

    def test_pendulum_oscillates_near_hanging(self):
        """For small displacement from hanging (theta=pi), pendulum should oscillate."""
        sim = self._make_sim(theta_0=np.pi + 0.1, theta_dot_0=0.0)
        sim.reset()

        # Run for a while and check theta stays near pi
        max_deviation = 0.0
        for _ in range(5000):
            state = sim.step()
            max_deviation = max(max_deviation, abs(state[2] - np.pi))

        # For small initial displacement from stable equilibrium, deviation stays small
        assert max_deviation < 1.0, f"Deviation from pi grew too large: {max_deviation:.4f}"

    def test_upright_pendulum_falls(self):
        """Starting near upright (theta=0), the unstable equilibrium, pendulum should fall."""
        sim = self._make_sim(theta_0=0.1, theta_dot_0=0.0)
        sim.reset()

        for _ in range(2000):
            sim.step()

        final_theta = sim.observe()[2]
        # The angle should grow significantly from near 0 (unstable point)
        assert abs(final_theta) > 0.5, (
            f"Upright pendulum should fall: final theta = {final_theta:.4f}"
        )

    def test_cart_position_changes(self):
        """Cart position should change due to pendulum dynamics."""
        sim = self._make_sim(
            theta_0=0.3, theta_dot_0=0.0, x_0=0.0,
        )
        sim.reset()

        for _ in range(1000):
            sim.step()

        # Cart should have moved from 0 due to reaction forces
        x_final = sim.observe()[0]
        assert abs(x_final) > 1e-6, "Cart should move due to pendulum dynamics"

    def test_pendulum_position(self):
        """pendulum_position should return correct (x_pend, y_pend)."""
        sim = self._make_sim(L=1.0, x_0=0.0, theta_0=0.0)
        sim.reset()

        x_pend, y_pend = sim.pendulum_position()
        # At theta=0, pendulum is directly above cart
        assert np.isclose(x_pend, 0.0, atol=1e-10)
        assert np.isclose(y_pend, 1.0, atol=1e-10)

    def test_pendulum_position_tilted(self):
        """pendulum_position at theta=pi/2 should be horizontal."""
        sim = self._make_sim(L=1.0, x_0=0.0, theta_0=np.pi / 2)
        sim.reset()

        x_pend, y_pend = sim.pendulum_position()
        # At theta=pi/2, pendulum tip is at (1, 0) relative to cart
        assert np.isclose(x_pend, 1.0, atol=1e-10)
        assert np.isclose(y_pend, 0.0, atol=1e-10)

    def test_run_trajectory(self):
        """run() should produce a TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)  # n_steps+1 states
        assert len(traj.timestamps) == 101

    def test_kinetic_potential_sum(self):
        """Kinetic + potential should equal total energy."""
        sim = self._make_sim(theta_0=0.5, theta_dot_0=1.0, x_dot_0=0.3)
        sim.reset()

        KE = sim.kinetic_energy
        PE = sim.potential_energy
        TE = sim.total_energy
        assert np.isclose(KE + PE, TE, rtol=1e-10), (
            f"KE({KE}) + PE({PE}) != TE({TE})"
        )


class TestCartPoleRediscovery:
    """Tests for cart-pole rediscovery data generation."""

    def test_linearized_frequency_data(self):
        """generate_linearized_frequency_data should produce valid frequency data."""
        from simulating_anything.rediscovery.cart_pole import (
            generate_linearized_frequency_data,
        )
        data = generate_linearized_frequency_data(
            n_samples=5, n_steps=20000, dt=0.0005
        )
        assert "M" in data
        assert "m" in data
        assert "L" in data
        assert "omega_measured" in data
        assert "omega_theory" in data
        assert len(data["M"]) > 0

        # Measured should be close to theory
        rel_err = (
            np.abs(data["omega_measured"] - data["omega_theory"])
            / data["omega_theory"]
        )
        assert np.mean(rel_err) < 0.02, f"Mean relative error too large: {np.mean(rel_err):.4%}"

    def test_energy_conservation_data(self):
        """generate_energy_conservation_data should verify energy conservation."""
        from simulating_anything.rediscovery.cart_pole import (
            generate_energy_conservation_data,
        )
        data = generate_energy_conservation_data(
            n_trajectories=3, n_steps=1000, dt=0.001
        )
        assert "final_drift" in data
        assert "max_drift" in data
        assert len(data["final_drift"]) == 3

        # Energy drift should be small for frictionless system
        assert np.max(data["max_drift"]) < 1e-4, (
            f"Energy drift too large: {np.max(data['max_drift']):.2e}"
        )
