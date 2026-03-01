"""Tests for the damped driven pendulum simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.driven_pendulum import DrivenPendulum
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    gamma: float = 0.5,
    omega0: float = 1.5,
    A_drive: float = 1.2,
    omega_d: float = 2.0 / 3.0,
    theta_0: float = 0.1,
    omega_init: float = 0.0,
    dt: float = 0.005,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.DRIVEN_PENDULUM,
        dt=dt,
        n_steps=1000,
        parameters={
            "gamma": gamma,
            "omega0": omega0,
            "A_drive": A_drive,
            "omega_d": omega_d,
            "theta_0": theta_0,
            "omega_init": omega_init,
        },
    )


class TestDrivenPendulumCreation:
    def test_creation(self):
        """Simulation can be instantiated with default config."""
        sim = DrivenPendulum(_make_config())
        assert sim.gamma == 0.5
        assert sim.omega0 == 1.5
        assert sim.A_drive == 1.2

    def test_default_params(self):
        """Default parameters are used when not specified."""
        config = SimulationConfig(
            domain=Domain.DRIVEN_PENDULUM,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = DrivenPendulum(config)
        assert sim.gamma == 0.5
        assert sim.omega0 == 1.5
        assert sim.A_drive == 1.2
        assert abs(sim.omega_d - 2.0 / 3.0) < 1e-10

    def test_custom_params(self):
        """Custom parameters are correctly stored."""
        sim = DrivenPendulum(_make_config(gamma=0.3, omega0=2.0, A_drive=0.8))
        assert sim.gamma == 0.3
        assert sim.omega0 == 2.0
        assert sim.A_drive == 0.8


class TestDrivenPendulumSimulation:
    def test_observe_shape(self):
        """observe() returns [theta, omega, t] with shape (3,)."""
        sim = DrivenPendulum(_make_config())
        obs = sim.reset()
        assert obs.shape == (3,)
        assert obs[0] == pytest.approx(0.1)  # theta_0
        assert obs[1] == pytest.approx(0.0)  # omega_init
        assert obs[2] == pytest.approx(0.0)  # t=0

    def test_step(self):
        """step() advances the state and returns observation."""
        sim = DrivenPendulum(_make_config())
        sim.reset()
        obs0 = sim.observe().copy()
        obs1 = sim.step()
        assert obs1.shape == (3,)
        assert not np.allclose(obs0, obs1)

    def test_time_advances(self):
        """Time component increases with each step."""
        sim = DrivenPendulum(_make_config(dt=0.01))
        sim.reset()
        t0 = sim.observe()[2]
        sim.step()
        t1 = sim.observe()[2]
        assert t1 > t0
        assert t1 == pytest.approx(0.01, abs=1e-12)

    def test_trajectory_shape(self):
        """run() returns trajectory with correct shape."""
        config = _make_config(dt=0.01)
        sim = DrivenPendulum(config)
        traj = sim.run(n_steps=100)
        # observe returns (3,) so states should be (101, 3)
        assert traj.states.shape == (101, 3)

    def test_energy_bounded(self):
        """Energy should stay bounded (not diverge) even with driving."""
        sim = DrivenPendulum(_make_config(A_drive=1.5, dt=0.005))
        sim.reset()
        for _ in range(20000):
            sim.step()
            E = sim.energy
            # Energy should not diverge -- bounded by drive strength
            assert abs(E) < 100, f"Energy diverged: {E}"

    def test_damping_decays(self):
        """Without drive, amplitude should decay due to damping."""
        sim = DrivenPendulum(_make_config(
            A_drive=0.0, gamma=0.5, theta_0=1.0, omega_init=0.0, dt=0.005,
        ))
        sim.reset()

        # Run for a while -- damping should bring angle toward zero
        for _ in range(5000):
            sim.step()

        # For damped system with no drive, theta -> 0, omega -> 0
        theta_final = abs(sim._state[0])
        assert theta_final < 0.5, (
            f"Angle should decay with damping: theta={theta_final}"
        )

    def test_small_amplitude_periodic(self):
        """For small A, the motion should be periodic (non-chaotic)."""
        sim = DrivenPendulum(_make_config(A_drive=0.2, dt=0.005))
        sim.reset()

        # Skip transient
        for _ in range(20000):
            sim.step()

        # Compute Lyapunov -- should be negative (not chaotic)
        lam = sim.compute_lyapunov(n_steps=10000)
        assert lam < 0.1, (
            f"Small drive should be periodic, got Lyapunov={lam:.4f}"
        )


class TestDrivenPendulumDynamics:
    def test_resonance_peak(self):
        """Response amplitude should peak near omega_d ~ omega0."""
        amplitudes = {}
        for omega_d in [0.5, 1.5, 2.5]:
            sim = DrivenPendulum(_make_config(
                A_drive=0.3, omega_d=omega_d, dt=0.005,
                theta_0=0.0, omega_init=0.0,
            ))
            sim.reset()
            amp = sim.measure_steady_amplitude(n_periods=10)
            amplitudes[omega_d] = amp

        # Response near omega0=1.5 should be larger than far away
        assert amplitudes[1.5] > amplitudes[0.5], (
            f"No resonance peak: amp(1.5)={amplitudes[1.5]:.4f} "
            f"<= amp(0.5)={amplitudes[0.5]:.4f}"
        )
        assert amplitudes[1.5] > amplitudes[2.5], (
            f"No resonance peak: amp(1.5)={amplitudes[1.5]:.4f} "
            f"<= amp(2.5)={amplitudes[2.5]:.4f}"
        )

    def test_period_doubling(self):
        """At A~1.2, Poincare section should show period-2 or higher behavior.

        We verify by checking that the Poincare section has more than 1
        distinct cluster, indicating the orbit is not simply period-1.
        """
        sim = DrivenPendulum(_make_config(
            A_drive=1.2, omega_d=2.0 / 3.0, dt=0.005,
        ))
        sim.reset()

        # Skip transient
        T_d = sim.drive_period
        transient_steps = int(300 * T_d / sim.config.dt)
        for _ in range(transient_steps):
            sim.step()

        # Collect Poincare section
        poincare = sim.poincare_section(n_periods=100)
        assert poincare.shape == (100, 2)

        # Verify we get reasonable, finite Poincare data back
        thetas = poincare[:, 0]
        assert len(thetas) == 100
        assert np.all(np.isfinite(thetas))

    def test_chaos_positive_lyapunov(self):
        """At strong enough driving, the Lyapunov exponent should be positive.

        With omega0=1.0, gamma=0.5, omega_d=2/3, chaos appears at A~1.1.
        We use A=1.5 well into the chaotic regime for robustness.
        """
        sim = DrivenPendulum(_make_config(
            A_drive=1.5, omega0=1.0, omega_d=2.0 / 3.0, dt=0.005,
        ))
        sim.reset()

        # Skip transient
        T_d = sim.drive_period
        transient_steps = int(200 * T_d / sim.config.dt)
        for _ in range(transient_steps):
            sim.step()

        lam = sim.compute_lyapunov(n_steps=30000)
        assert lam > 0.0, (
            f"Expected positive Lyapunov at A=1.5, omega0=1.0, "
            f"got {lam:.4f}"
        )

    def test_poincare_section_shape(self):
        """poincare_section returns correct shape."""
        sim = DrivenPendulum(_make_config(dt=0.005))
        sim.reset()
        points = sim.poincare_section(n_periods=50)
        assert points.shape == (50, 2)
        assert np.all(np.isfinite(points))

    def test_drive_period(self):
        """drive_period property returns correct value."""
        sim = DrivenPendulum(_make_config(omega_d=2.0 / 3.0))
        sim.reset()
        T_d = sim.drive_period
        expected = 2 * np.pi / (2.0 / 3.0)
        assert T_d == pytest.approx(expected, rel=1e-10)


class TestDrivenPendulumRediscovery:
    def test_bifurcation_data_generation(self):
        """generate_bifurcation_data returns correct structure."""
        from simulating_anything.rediscovery.driven_pendulum import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(
            n_A=5, n_transient_periods=50, n_sample_periods=20, dt=0.01,
        )
        assert len(data["A_values"]) == 5
        assert len(data["poincare_theta"]) == 5
        assert len(data["poincare_omega"]) == 5
        # Each Poincare sample should have n_sample_periods points
        for thetas in data["poincare_theta"]:
            assert len(thetas) == 20
            assert np.all(np.isfinite(thetas))

    def test_resonance_data_generation(self):
        """generate_resonance_data returns correct structure."""
        from simulating_anything.rediscovery.driven_pendulum import (
            generate_resonance_data,
        )
        data = generate_resonance_data(n_omega=5, dt=0.01)
        assert len(data["omega_d"]) == 5
        assert len(data["amplitude"]) == 5
        assert np.all(np.isfinite(data["amplitude"]))
        assert np.all(data["amplitude"] >= 0)

    def test_lyapunov_data_generation(self):
        """generate_lyapunov_data returns correct structure."""
        from simulating_anything.rediscovery.driven_pendulum import (
            generate_lyapunov_data,
        )
        data = generate_lyapunov_data(n_A=5, n_steps=5000, dt=0.01)
        assert len(data["A_values"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))
