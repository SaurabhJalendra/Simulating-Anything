"""Tests for the Stuart-Landau oscillator simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.stuart_landau import StuartLandauSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    mu: float = 1.0,
    omega: float = 1.0,
    beta: float = 0.0,
    dt: float = 0.005,
    x_0: float = 0.1,
    y_0: float = 0.0,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.STUART_LANDAU,
        dt=dt,
        n_steps=1000,
        parameters={
            "mu": mu, "omega": omega, "beta": beta,
            "x_0": x_0, "y_0": y_0,
        },
    )


class TestStuartLandauConstruction:
    def test_default_parameters(self):
        sim = StuartLandauSimulation(_make_config())
        assert sim.mu == 1.0
        assert sim.omega == 1.0
        assert sim.beta == 0.0

    def test_custom_parameters(self):
        sim = StuartLandauSimulation(_make_config(mu=2.5, omega=3.0, beta=-0.5))
        assert sim.mu == 2.5
        assert sim.omega == 3.0
        assert sim.beta == -0.5

    def test_initial_state_shape(self):
        sim = StuartLandauSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        assert state.dtype == np.float64

    def test_initial_state_values(self):
        sim = StuartLandauSimulation(_make_config(x_0=0.5, y_0=0.3))
        state = sim.reset()
        np.testing.assert_allclose(state, [0.5, 0.3])


class TestStuartLandauProperties:
    def test_theoretical_amplitude_positive_mu(self):
        sim = StuartLandauSimulation(_make_config(mu=4.0))
        assert sim.theoretical_amplitude == pytest.approx(2.0)

    def test_theoretical_amplitude_negative_mu(self):
        sim = StuartLandauSimulation(_make_config(mu=-1.0))
        assert sim.theoretical_amplitude == 0.0

    def test_theoretical_amplitude_zero_mu(self):
        sim = StuartLandauSimulation(_make_config(mu=0.0))
        assert sim.theoretical_amplitude == 0.0

    def test_theoretical_frequency_no_coupling(self):
        sim = StuartLandauSimulation(_make_config(omega=2.0, beta=0.0))
        assert sim.theoretical_frequency == pytest.approx(2.0)

    def test_theoretical_frequency_with_coupling(self):
        sim = StuartLandauSimulation(_make_config(mu=1.0, omega=2.0, beta=0.5))
        # omega_eff = omega - beta*mu = 2.0 - 0.5*1.0 = 1.5
        assert sim.theoretical_frequency == pytest.approx(1.5)

    def test_is_oscillatory_positive_mu(self):
        sim = StuartLandauSimulation(_make_config(mu=1.0))
        assert sim.is_oscillatory is True

    def test_is_oscillatory_negative_mu(self):
        sim = StuartLandauSimulation(_make_config(mu=-0.5))
        assert sim.is_oscillatory is False

    def test_is_oscillatory_zero_mu(self):
        sim = StuartLandauSimulation(_make_config(mu=0.0))
        assert sim.is_oscillatory is False


class TestStuartLandauDynamics:
    def test_step_advances_state(self):
        sim = StuartLandauSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_state(self):
        sim = StuartLandauSimulation(_make_config())
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_trajectory_bounded(self):
        """Trajectory should remain bounded for mu > 0."""
        sim = StuartLandauSimulation(_make_config(mu=2.0))
        sim.reset()
        for _ in range(5000):
            sim.step()
            x, y = sim.observe()
            assert abs(x) < 10, f"x diverged: {x}"
            assert abs(y) < 10, f"y diverged: {y}"

    def test_derivatives_at_origin(self):
        """At origin with mu > 0, growth should be outward."""
        sim = StuartLandauSimulation(_make_config(mu=1.0, omega=1.0, beta=0.0))
        y = np.array([0.1, 0.0])
        dy = sim._derivatives(y)
        # dx/dt = mu*x - omega*y - r^2*(x - beta*y)
        # = 1.0*0.1 - 1.0*0.0 - 0.01*(0.1 - 0.0) = 0.1 - 0.001 = 0.099
        # dy/dt = omega*x + mu*y - r^2*(beta*x + y)
        # = 1.0*0.1 + 1.0*0.0 - 0.01*(0.0 + 0.0) = 0.1
        np.testing.assert_allclose(dy, [0.099, 0.1], atol=1e-10)

    def test_derivatives_at_limit_cycle(self):
        """On the limit cycle r = sqrt(mu), radial derivative should be ~0."""
        sim = StuartLandauSimulation(_make_config(mu=1.0, omega=1.0, beta=0.0))
        # Point on limit cycle: r = 1.0, theta = 0 => x=1, y=0
        y = np.array([1.0, 0.0])
        dy = sim._derivatives(y)
        # dx/dt = mu*x - omega*y - r^2*(x) = 1.0 - 0 - 1.0 = 0
        # dy/dt = omega*x + mu*y - r^2*(y) = 1.0 + 0 - 0 = 1.0
        np.testing.assert_allclose(dy, [0.0, 1.0], atol=1e-10)

    def test_run_returns_trajectory_data(self):
        sim = StuartLandauSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101


class TestStuartLandauLimitCycle:
    def test_amplitude_converges_to_sqrt_mu(self):
        """For mu > 0, the amplitude should converge to sqrt(mu)."""
        for mu in [0.5, 1.0, 2.0, 4.0]:
            sim = StuartLandauSimulation(_make_config(mu=mu, dt=0.005, x_0=0.1))
            sim.reset()

            # Let transient decay
            for _ in range(20000):
                sim.step()

            # Measure amplitude over several cycles
            r_vals = []
            for _ in range(5000):
                sim.step()
                r_vals.append(sim.compute_amplitude())

            mean_r = np.mean(r_vals)
            expected = np.sqrt(mu)
            assert abs(mean_r - expected) < 0.05 * expected, (
                f"mu={mu}: mean_r={mean_r:.4f}, expected={expected:.4f}"
            )

    def test_amplitude_different_initial_conditions(self):
        """Same mu, different IC should converge to same limit cycle."""
        sim1 = StuartLandauSimulation(_make_config(mu=1.0, x_0=0.01))
        sim1.reset()
        sim2 = StuartLandauSimulation(_make_config(mu=1.0, x_0=3.0))
        sim2.reset()

        for _ in range(20000):
            sim1.step()
            sim2.step()

        r1 = sim1.compute_amplitude()
        r2 = sim2.compute_amplitude()
        np.testing.assert_allclose(r1, r2, atol=0.05)

    def test_limit_cycle_radius_monotonic_in_mu(self):
        """Higher mu should yield larger limit cycle radius."""
        radii = []
        for mu in [0.5, 1.0, 2.0, 3.0]:
            sim = StuartLandauSimulation(_make_config(mu=mu, dt=0.005))
            sim.reset()
            amp = sim.measure_amplitude(n_periods=3)
            radii.append(amp)

        for i in range(len(radii) - 1):
            assert radii[i + 1] > radii[i], (
                f"Radius not monotonic: {radii}"
            )


class TestStuartLandauStableFixedPoint:
    def test_negative_mu_decays_to_origin(self):
        """For mu < 0, the origin is stable; amplitude should decay."""
        sim = StuartLandauSimulation(_make_config(mu=-0.5, x_0=1.0))
        sim.reset()

        for _ in range(20000):
            sim.step()

        r_final = sim.compute_amplitude()
        assert r_final < 0.01, f"Did not decay: r={r_final}"

    def test_strongly_negative_mu_fast_decay(self):
        """More negative mu should decay faster."""
        sim = StuartLandauSimulation(_make_config(mu=-2.0, x_0=1.0))
        sim.reset()

        for _ in range(5000):
            sim.step()

        r_final = sim.compute_amplitude()
        assert r_final < 0.001, f"Decay too slow: r={r_final}"


class TestStuartLandauHopfBifurcation:
    def test_hopf_transition(self):
        """Across mu = 0, behavior should change from decay to oscillation."""
        # Below bifurcation
        sim_below = StuartLandauSimulation(_make_config(mu=-0.3, x_0=0.5))
        sim_below.reset()
        for _ in range(15000):
            sim_below.step()
        r_below = sim_below.compute_amplitude()

        # Above bifurcation
        sim_above = StuartLandauSimulation(_make_config(mu=0.3, x_0=0.1))
        sim_above.reset()
        for _ in range(15000):
            sim_above.step()
        r_above = sim_above.compute_amplitude()

        assert r_below < 0.05, f"Below Hopf should decay: r={r_below}"
        assert r_above > 0.3, f"Above Hopf should oscillate: r={r_above}"


class TestStuartLandauFrequency:
    def test_frequency_near_omega_beta_zero(self):
        """With beta=0, frequency should be close to omega."""
        sim = StuartLandauSimulation(_make_config(mu=1.0, omega=2.0, beta=0.0))
        sim.reset()
        freq = sim.compute_frequency(n_periods=5)
        # Allow 5% tolerance
        assert abs(freq - 2.0) < 0.15, f"freq={freq}, expected ~2.0"

    def test_frequency_shifts_with_beta(self):
        """With nonzero beta, frequency should shift: omega - beta*mu."""
        sim = StuartLandauSimulation(
            _make_config(mu=1.0, omega=1.0, beta=0.5)
        )
        sim.reset()
        freq = sim.compute_frequency(n_periods=5)
        expected = 1.0 - 0.5 * 1.0  # 0.5
        assert abs(freq - expected) < 0.15, f"freq={freq}, expected ~{expected}"

    def test_frequency_zero_for_negative_mu(self):
        """Below bifurcation, compute_frequency returns 0."""
        sim = StuartLandauSimulation(_make_config(mu=-1.0))
        sim.reset()
        freq = sim.compute_frequency()
        assert freq == 0.0


class TestStuartLandauPhase:
    def test_phase_initial(self):
        sim = StuartLandauSimulation(_make_config(x_0=1.0, y_0=0.0))
        sim.reset()
        phase = sim.compute_phase()
        assert phase == pytest.approx(0.0, abs=1e-10)

    def test_phase_quarter_turn(self):
        sim = StuartLandauSimulation(_make_config(x_0=0.0, y_0=1.0))
        sim.reset()
        phase = sim.compute_phase()
        assert phase == pytest.approx(np.pi / 2, abs=1e-10)


class TestStuartLandauMuSweep:
    def test_sweep_returns_arrays(self):
        sim = StuartLandauSimulation(_make_config())
        result = sim.mu_sweep(mu_values=np.array([-0.5, 0.0, 0.5, 1.0]))
        assert "mu" in result
        assert "amplitude" in result
        assert "frequency" in result
        assert len(result["mu"]) == 4
        assert len(result["amplitude"]) == 4
        assert len(result["frequency"]) == 4

    def test_sweep_amplitude_increases_with_mu(self):
        sim = StuartLandauSimulation(_make_config(dt=0.005))
        result = sim.mu_sweep(
            mu_values=np.array([0.5, 1.0, 2.0, 3.0]), dt=0.005,
        )
        amps = result["amplitude"]
        for i in range(len(amps) - 1):
            assert amps[i + 1] > amps[i], f"Amplitude not increasing: {amps}"


class TestStuartLandauRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.stuart_landau import generate_ode_data

        data = generate_ode_data(mu=1.0, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["mu"] == 1.0
        assert data["omega"] == 1.0

    def test_ode_data_contains_oscillation(self):
        from simulating_anything.rediscovery.stuart_landau import generate_ode_data

        data = generate_ode_data(mu=1.0, n_steps=5000, dt=0.005)
        x = data["x"]
        # x should oscillate: check that it has both positive and negative values
        assert np.any(x > 0.1) and np.any(x < -0.1)

    def test_amplitude_data_generation(self):
        from simulating_anything.rediscovery.stuart_landau import (
            generate_amplitude_data,
        )

        data = generate_amplitude_data(n_mu=5, dt=0.005)
        assert len(data["mu"]) == 5
        assert len(data["amplitude"]) == 5
        assert len(data["frequency"]) == 5
        assert len(data["amplitude_theory"]) == 5

    def test_amplitude_data_matches_theory(self):
        from simulating_anything.rediscovery.stuart_landau import (
            generate_amplitude_data,
        )

        data = generate_amplitude_data(n_mu=5, omega=1.0, beta=0.0, dt=0.005)
        valid = data["amplitude"] > 0.01
        if np.sum(valid) > 0:
            errors = np.abs(
                data["amplitude"][valid] - data["amplitude_theory"][valid]
            )
            mean_error = np.mean(errors)
            assert mean_error < 0.2, f"Mean error vs theory: {mean_error}"

    def test_hopf_data_generation(self):
        from simulating_anything.rediscovery.stuart_landau import generate_hopf_data

        data = generate_hopf_data(n_mu=10, dt=0.005)
        assert len(data["mu"]) == 10
        assert len(data["amplitude"]) == 10

    def test_hopf_data_transition(self):
        from simulating_anything.rediscovery.stuart_landau import generate_hopf_data

        data = generate_hopf_data(n_mu=20, dt=0.005)
        # Negative mu should have small amplitude
        neg_mask = data["mu"] < -0.3
        if np.any(neg_mask):
            assert np.mean(data["amplitude"][neg_mask]) < 0.1

        # Positive mu > 0.5 should have significant amplitude
        pos_mask = data["mu"] > 0.5
        if np.any(pos_mask):
            assert np.mean(data["amplitude"][pos_mask]) > 0.3
