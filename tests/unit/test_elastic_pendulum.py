"""Tests for the elastic pendulum simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.elastic_pendulum import ElasticPendulum
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestElasticPendulum:
    """Tests for the elastic pendulum simulation."""

    def _make_sim(self, **kwargs) -> ElasticPendulum:
        defaults = {
            "k": 10.0, "m": 1.0, "L0": 1.0, "g": 9.81,
            "r_0": 1.981, "r_dot_0": 0.0,
            "theta_0": 0.0, "theta_dot_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.ELASTIC_PENDULUM,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return ElasticPendulum(config)

    def test_creation(self):
        """ElasticPendulum should be created with default parameters."""
        sim = self._make_sim()
        assert sim.m == 1.0
        assert sim.k == 10.0
        assert sim.L0 == 1.0
        assert sim.g == 9.81

    def test_custom_parameters(self):
        """ElasticPendulum should accept custom parameters."""
        sim = self._make_sim(m=2.0, k=20.0, L0=1.5, g=10.0)
        assert sim.m == 2.0
        assert sim.k == 20.0
        assert sim.L0 == 1.5
        assert sim.g == 10.0

    def test_initial_state_shape(self):
        """State should be shape (4,): [r, r_dot, theta, theta_dot]."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (4,)

    def test_initial_state_values(self):
        """Initial state should match configured values."""
        sim = self._make_sim(
            r_0=2.0, r_dot_0=0.5, theta_0=0.3, theta_dot_0=-0.1
        )
        state = sim.reset()
        assert np.isclose(state[0], 2.0)    # r
        assert np.isclose(state[1], 0.5)    # r_dot
        assert np.isclose(state[2], 0.3)    # theta
        assert np.isclose(state[3], -0.1)   # theta_dot

    def test_observe_shape(self):
        """Observe should return shape (4,)."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_advances_state(self):
        """A single step should change the state for non-equilibrium initial conditions."""
        sim = self._make_sim(theta_0=0.1)
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_energy_conservation_no_damping(self):
        """Energy should be conserved for the Hamiltonian system."""
        sim = self._make_sim(
            k=10.0, m=1.0, L0=1.0,
            r_0=2.0, r_dot_0=0.1,
            theta_0=0.2, theta_dot_0=0.1,
        )
        sim.reset()
        E0 = sim.total_energy
        for _ in range(10000):
            sim.step()
        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / max(abs(E0), 1e-10)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_energy_conservation_large_amplitude(self):
        """Energy conservation should hold for larger oscillations."""
        sim = self._make_sim(
            k=5.0, m=2.0, L0=1.5,
            r_0=3.5, r_dot_0=0.5,
            theta_0=0.5, theta_dot_0=0.3,
        )
        sim.reset()
        E0 = sim.total_energy
        for _ in range(10000):
            sim.step()
        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / max(abs(E0), 1e-10)
        assert rel_drift < 1e-5, f"Energy drift {rel_drift:.2e} too large"

    def test_radial_frequency_property(self):
        """radial_frequency should match sqrt(k/m)."""
        sim = self._make_sim(k=16.0, m=4.0)
        expected = np.sqrt(16.0 / 4.0)  # = 2.0
        assert np.isclose(sim.radial_frequency, expected, rtol=1e-10)

    def test_angular_frequency_property(self):
        """angular_frequency should match sqrt(g/L0)."""
        sim = self._make_sim(g=9.81, L0=1.0)
        expected = np.sqrt(9.81 / 1.0)
        assert np.isclose(sim.angular_frequency, expected, rtol=1e-10)

    def test_equilibrium_length_property(self):
        """equilibrium_length should be L0 + mg/k."""
        sim = self._make_sim(k=10.0, m=1.0, L0=1.0, g=9.81)
        expected = 1.0 + 1.0 * 9.81 / 10.0  # = 1.981
        assert np.isclose(sim.equilibrium_length, expected, rtol=1e-10)

    def test_small_angle_radial_frequency(self):
        """Measured radial oscillation frequency should match sqrt(k/m) for small perturbation."""
        k, m, L0, g = 10.0, 1.0, 1.0, 9.81
        r_eq = L0 + m * g / k
        dt = 0.001
        n_steps = 30000

        # Small radial perturbation, theta = 0 (pure radial mode)
        config = SimulationConfig(
            domain=Domain.ELASTIC_PENDULUM,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "k": k, "m": m, "L0": L0, "g": g,
                "r_0": r_eq + 0.01, "r_dot_0": 0.0,
                "theta_0": 0.0, "theta_dot_0": 0.0,
            },
        )
        sim = ElasticPendulum(config)
        sim.reset()

        # Collect r displacement from equilibrium
        prev_dr = sim.observe()[0] - r_eq
        crossings = []
        for step in range(n_steps):
            state = sim.step()
            dr = state[0] - r_eq
            if prev_dr < 0 and dr >= 0:
                frac = -prev_dr / (dr - prev_dr)
                crossings.append((step + frac) * dt)
            prev_dr = dr

        assert len(crossings) >= 3, f"Not enough zero crossings: {len(crossings)}"
        periods = np.diff(crossings)
        T_measured = np.median(periods)
        omega_measured = 2 * np.pi / T_measured
        omega_theory = np.sqrt(k / m)

        rel_error = abs(omega_measured - omega_theory) / omega_theory
        assert rel_error < 0.01, (
            f"Radial frequency mismatch: measured={omega_measured:.4f}, "
            f"theory={omega_theory:.4f}, error={rel_error:.4%}"
        )

    def test_pendulum_position_at_vertical(self):
        """At theta=0, mass should be directly below the pivot."""
        sim = self._make_sim(r_0=2.0, theta_0=0.0)
        sim.reset()
        x, y = sim.pendulum_position()
        assert np.isclose(x, 0.0, atol=1e-10)
        assert np.isclose(y, 2.0, atol=1e-10)

    def test_pendulum_position_at_right(self):
        """At theta=pi/2, mass should be directly to the right."""
        sim = self._make_sim(r_0=2.0, theta_0=np.pi / 2)
        sim.reset()
        x, y = sim.pendulum_position()
        assert np.isclose(x, 2.0, atol=1e-10)
        assert np.isclose(y, 0.0, atol=1e-10)

    def test_pendulum_position_geometry(self):
        """Position should satisfy x^2 + y^2 = r^2."""
        sim = self._make_sim(r_0=2.5, theta_0=0.7)
        sim.reset()
        x, y = sim.pendulum_position()
        r = sim.observe()[0]
        assert np.isclose(x**2 + y**2, r**2, rtol=1e-10)

    def test_kinetic_potential_sum(self):
        """Kinetic + potential should equal total energy."""
        sim = self._make_sim(
            r_0=2.0, r_dot_0=0.3, theta_0=0.2, theta_dot_0=0.5
        )
        sim.reset()

        KE = sim.kinetic_energy
        PE = sim.potential_energy
        TE = sim.total_energy
        assert np.isclose(KE + PE, TE, rtol=1e-10), (
            f"KE({KE}) + PE({PE}) != TE({TE})"
        )

    def test_run_trajectory(self):
        """run() should produce a TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)  # n_steps+1 states
        assert len(traj.timestamps) == 101

    def test_trajectory_finite(self):
        """All trajectory states should be finite."""
        sim = self._make_sim(
            r_0=2.0, r_dot_0=0.1, theta_0=0.2, theta_dot_0=0.1
        )
        traj = sim.run(n_steps=5000)
        assert np.all(np.isfinite(traj.states)), "Trajectory contains non-finite values"

    def test_static_equilibrium(self):
        """Starting at equilibrium with no velocity should stay at equilibrium."""
        k, m, L0, g = 10.0, 1.0, 1.0, 9.81
        r_eq = L0 + m * g / k
        sim = self._make_sim(
            k=k, m=m, L0=L0, g=g,
            r_0=r_eq, r_dot_0=0.0,
            theta_0=0.0, theta_dot_0=0.0,
        )
        sim.reset()

        for _ in range(1000):
            sim.step()

        state = sim.observe()
        # Should remain very close to equilibrium
        assert np.isclose(state[0], r_eq, atol=1e-8), (
            f"r deviated from equilibrium: {state[0]} vs {r_eq}"
        )
        assert np.isclose(state[2], 0.0, atol=1e-8), (
            f"theta deviated from zero: {state[2]}"
        )


class TestElasticPendulumRediscovery:
    """Tests for elastic pendulum data generation."""

    def test_frequency_data(self):
        from simulating_anything.rediscovery.elastic_pendulum import (
            generate_frequency_data,
        )
        data = generate_frequency_data(n_samples=5, n_steps=20000, dt=0.001)
        assert "k" in data
        assert "m" in data
        assert "omega_measured" in data
        assert "omega_theory" in data
        assert len(data["k"]) > 0
        # Measured should be close to theory
        rel_err = (
            np.abs(data["omega_measured"] - data["omega_theory"])
            / data["omega_theory"]
        )
        assert np.mean(rel_err) < 0.02, f"Mean relative error too large: {np.mean(rel_err):.4%}"

    def test_energy_conservation_data(self):
        from simulating_anything.rediscovery.elastic_pendulum import (
            generate_energy_conservation_data,
        )
        data = generate_energy_conservation_data(
            n_trajectories=3, n_steps=1000, dt=0.001
        )
        assert "final_drift" in data
        assert "max_drift" in data
        assert len(data["final_drift"]) == 3

        # Energy drift should be small
        assert np.max(data["max_drift"]) < 1e-4, (
            f"Energy drift too large: {np.max(data['max_drift']):.2e}"
        )
