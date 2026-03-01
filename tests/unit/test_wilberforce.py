"""Tests for the Wilberforce pendulum simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.wilberforce import Wilberforce
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestWilberforce:
    """Tests for the Wilberforce pendulum simulation."""

    def _make_sim(self, **kwargs) -> Wilberforce:
        defaults = {
            "m": 0.5, "k": 5.0, "I": 1e-4, "kappa": 1e-3, "eps": 1e-3,
            "z_0": 0.1, "z_dot_0": 0.0, "theta_0": 0.0, "theta_dot_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.WILBERFORCE,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return Wilberforce(config)

    def test_creation(self):
        """Simulation should be created with correct parameters."""
        sim = self._make_sim(m=0.8, k=6.0, I=2e-4, kappa=2e-3, eps=5e-4)
        assert sim.m == 0.8
        assert sim.k == 6.0
        assert sim.I == 2e-4
        assert sim.kappa == 2e-3
        assert sim.eps == 5e-4

    def test_initial_state(self):
        """Reset should produce the correct initial state [z, z_dot, theta, theta_dot]."""
        sim = self._make_sim(z_0=0.05, theta_0=0.3)
        state = sim.reset()
        assert state.shape == (4,)
        assert np.isclose(state[0], 0.05)   # z_0
        assert np.isclose(state[1], 0.0)    # z_dot_0
        assert np.isclose(state[2], 0.3)    # theta_0
        assert np.isclose(state[3], 0.0)    # theta_dot_0

    def test_state_shape(self):
        """State should always have shape (4,)."""
        sim = self._make_sim()
        sim.reset()
        assert sim.observe().shape == (4,)
        sim.step()
        assert sim.observe().shape == (4,)

    def test_observe_shape(self):
        """Observe should return state of shape (4,)."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_step_advances_state(self):
        """A step should change the state."""
        sim = self._make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_energy_conservation(self):
        """Total energy should be conserved with RK4 (no damping)."""
        sim = self._make_sim(m=0.5, k=5.0, I=1e-4, kappa=1e-3, eps=1e-3)
        sim.reset()
        E0 = sim.total_energy
        for _ in range(10000):
            sim.step()
        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-7, f"Energy drift {rel_drift:.2e} too large"

    def test_energy_conservation_strong_coupling(self):
        """Energy conservation should hold even with strong coupling."""
        sim = self._make_sim(m=0.5, k=5.0, I=1e-4, kappa=1e-3, eps=0.1)
        sim.reset()
        E0 = sim.total_energy
        for _ in range(10000):
            sim.step()
        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_omega_z_property(self):
        """omega_z should match sqrt(k/m)."""
        sim = self._make_sim(k=9.0, m=1.0)
        assert np.isclose(sim.omega_z, 3.0)

    def test_omega_theta_property(self):
        """omega_theta should match sqrt(kappa/I)."""
        sim = self._make_sim(kappa=4e-3, I=1e-4)
        # sqrt(4e-3 / 1e-4) = sqrt(40) ~ 6.3246
        assert np.isclose(sim.omega_theta, np.sqrt(40.0))

    def test_beat_frequency_property(self):
        """Beat frequency should be |omega_z - omega_theta|."""
        sim = self._make_sim(k=4.0, m=1.0, kappa=0.09, I=0.01)
        # omega_z = 2, omega_theta = 3
        expected = abs(2.0 - 3.0)
        assert np.isclose(sim.beat_frequency, expected)

    def test_energy_transfer_period(self):
        """Energy transfer period should be 2*pi / beat_frequency."""
        sim = self._make_sim(k=4.0, m=1.0, kappa=0.09, I=0.01)
        bf = sim.beat_frequency
        expected = 2.0 * np.pi / bf
        assert np.isclose(sim.energy_transfer_period, expected)

    def test_energy_transfer_period_inf_when_degenerate(self):
        """When omega_z == omega_theta and eps=0, beat freq is zero."""
        # Set kappa/I = k/m so omega_z = omega_theta
        sim = self._make_sim(k=10.0, m=1.0, kappa=10.0, I=1.0, eps=0.0)
        assert sim.beat_frequency < 1e-10
        assert sim.energy_transfer_period == float("inf")

    def test_zero_coupling_independent_oscillators(self):
        """With eps=0, the z and theta modes should be independent."""
        sim = self._make_sim(
            eps=0.0,
            z_0=0.1, z_dot_0=0.0, theta_0=0.0, theta_dot_0=0.0,
        )
        sim.reset()

        # Theta should stay at zero if only z is excited and eps=0
        for _ in range(5000):
            state = sim.step()
        assert abs(state[2]) < 1e-12, "theta should remain zero with no coupling"
        assert abs(state[3]) < 1e-12, "theta_dot should remain zero with no coupling"

    def test_energy_transfer_between_modes(self):
        """With coupling, energy should transfer between translational and rotational."""
        # Set omega_z ~ omega_theta for resonant transfer
        m, k = 0.5, 5.0
        omega_z = np.sqrt(k / m)
        I_val = 1e-4
        kappa = omega_z**2 * I_val  # Match frequencies
        eps = 5e-3  # Strong enough coupling to see transfer

        sim = self._make_sim(
            m=m, k=k, I=I_val, kappa=kappa, eps=eps,
            z_0=0.1, z_dot_0=0.0, theta_0=0.0, theta_dot_0=0.0,
        )
        sim.reset()

        # Initially all energy in translational mode
        E_trans_init = sim.translational_energy
        E_rot_init = sim.rotational_energy
        assert E_trans_init > 0.01
        assert E_rot_init < 1e-15

        # Run for many oscillation cycles to allow energy transfer
        # Transfer time ~ 4*pi*sqrt(m*I)/eps
        T_transfer = 4.0 * np.pi * np.sqrt(m * I_val) / eps
        n_transfer_steps = int(T_transfer / sim.config.dt)
        # Run for about half a transfer period
        for _ in range(n_transfer_steps // 2):
            sim.step()

        E_rot_later = sim.rotational_energy
        # Rotational mode should have gained significant energy
        assert E_rot_later > 0.1 * E_trans_init, (
            f"Energy transfer failed: E_rot={E_rot_later:.6f}, "
            f"E_trans_init={E_trans_init:.6f}"
        )

    def test_total_energy_equals_sum(self):
        """Total energy should approximately equal translational + rotational + coupling."""
        sim = self._make_sim(z_0=0.1, theta_0=0.2)
        sim.reset()
        for _ in range(1000):
            sim.step()

        E_total = sim.total_energy
        E_trans = sim.translational_energy
        E_rot = sim.rotational_energy
        # Coupling energy: 0.5 * eps * z * theta
        z, _, theta, _ = sim.observe()
        E_coupling = 0.5 * sim.eps * z * theta
        assert np.isclose(E_total, E_trans + E_rot + E_coupling, rtol=1e-10)

    def test_reproducibility(self):
        """Two simulations with same parameters should produce identical trajectories."""
        sim1 = self._make_sim()
        sim2 = self._make_sim()

        traj1 = sim1.run(n_steps=100)
        traj2 = sim2.run(n_steps=100)

        assert np.allclose(traj1.states, traj2.states, atol=1e-12)

    def test_run_trajectory(self):
        """run() should produce a TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)
        assert len(traj.timestamps) == 101


class TestWilberforceRediscovery:
    """Tests for Wilberforce pendulum data generation."""

    def test_normal_mode_data(self):
        from simulating_anything.rediscovery.wilberforce import (
            generate_normal_mode_data,
        )
        data = generate_normal_mode_data(n_samples=5, n_steps=20000, dt=0.001)
        assert "k" in data
        assert "m" in data
        assert "omega_z_measured" in data
        assert "omega_theta_measured" in data
        assert len(data["k"]) > 0
        # Measured frequencies should be close to theory
        z_err = (
            np.abs(data["omega_z_measured"] - data["omega_z_theory"])
            / data["omega_z_theory"]
        )
        assert np.mean(z_err) < 0.02  # 2% tolerance

    def test_ode_data(self):
        from simulating_anything.rediscovery.wilberforce import generate_ode_data
        data = generate_ode_data(n_steps=100, dt=0.001)
        assert data["states"].shape == (101, 4)
        assert data["m"] == 0.5
        assert data["k"] == 5.0
        assert data["eps"] == 1e-3
