"""Tests for the KdV (Korteweg-de Vries) equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.kdv import KdVSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    c: float = 2.0, x0: float = 10.0, N: int = 256, L: float = 40.0,
    dt: float = 1e-4, n_steps: int = 100,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.KDV,
        dt=dt,
        n_steps=n_steps,
        parameters={"c": c, "N": float(N), "L": L, "x0": x0},
    )


class TestKdVCreation:
    """Basic creation and configuration tests."""

    def test_creation(self):
        sim = KdVSimulation(_make_config())
        assert sim.c == 2.0
        assert sim.N == 256
        assert sim.L == 40.0

    def test_default_parameters(self):
        config = SimulationConfig(domain=Domain.KDV, dt=1e-4, n_steps=100)
        sim = KdVSimulation(config)
        assert sim.c == 2.0
        assert sim.x0 == 10.0
        assert sim.N == 256
        assert sim.L == 40.0

    def test_custom_parameters(self):
        sim = KdVSimulation(_make_config(c=4.0, x0=5.0, N=128, L=20.0))
        assert sim.c == 4.0
        assert sim.x0 == 5.0
        assert sim.N == 128
        assert sim.L == 20.0


class TestKdVStateShape:
    """Tests for state shape and basic operations."""

    def test_initial_state_shape(self):
        sim = KdVSimulation(_make_config(N=256))
        state = sim.reset()
        assert state.shape == (256,)

    def test_observe_shape(self):
        sim = KdVSimulation(_make_config(N=128))
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (128,)

    def test_step_returns_correct_shape(self):
        sim = KdVSimulation(_make_config(N=256))
        sim.reset()
        state = sim.step()
        assert state.shape == (256,)

    def test_state_dtype(self):
        sim = KdVSimulation(_make_config())
        state = sim.reset()
        assert state.dtype == np.float64


class TestSolitonProfile:
    """Tests for the soliton initial condition."""

    def test_soliton_peak_amplitude(self):
        """Peak amplitude should be c/2."""
        c = 2.0
        sim = KdVSimulation(_make_config(c=c, N=512, L=40.0))
        sim.init_type = "soliton"
        sim.reset()
        measured = sim.measure_peak_amplitude()
        expected = c / 2.0
        np.testing.assert_allclose(measured, expected, rtol=1e-6)

    def test_soliton_peak_at_x0(self):
        """Soliton peak should be at position x0."""
        x0 = 15.0
        sim = KdVSimulation(_make_config(c=2.0, x0=x0, N=512, L=40.0))
        sim.init_type = "soliton"
        sim.reset()
        peak_pos = sim.measure_peak_position()
        np.testing.assert_allclose(peak_pos, x0, atol=sim.dx)

    def test_soliton_shape_matches_theory(self):
        """Soliton profile should match (c/2)*sech^2(sqrt(c)/2*(x-x0))."""
        c = 2.0
        x0 = 20.0
        N = 512
        L = 40.0
        sim = KdVSimulation(_make_config(c=c, x0=x0, N=N, L=L))
        sim.init_type = "soliton"
        sim.reset()

        x = sim.x
        arg = np.sqrt(c) / 2.0 * (x - x0)
        theory = (c / 2.0) / np.cosh(arg) ** 2

        np.testing.assert_allclose(sim.observe(), theory, atol=1e-10)

    def test_soliton_positive(self):
        """Soliton should be non-negative everywhere."""
        sim = KdVSimulation(_make_config(c=3.0))
        sim.init_type = "soliton"
        state = sim.reset()
        assert np.all(state >= 0), "Soliton should be non-negative"

    def test_amplitude_varies_with_c(self):
        """Faster solitons should be taller: A = c/2.

        Soliton centered at L/2 for symmetric grid placement.
        Uses N=1024 to resolve narrow high-c solitons.
        """
        for c in [0.5, 1.0, 2.0, 4.0, 8.0]:
            sim = KdVSimulation(_make_config(c=c, x0=30.0, N=1024, L=60.0))
            sim.init_type = "soliton"
            sim.reset()
            amp = sim.measure_peak_amplitude()
            np.testing.assert_allclose(amp, c / 2.0, rtol=2e-3,
                                       err_msg=f"Failed for c={c}")


class TestSolitonTranslation:
    """Tests for soliton propagation and speed."""

    def test_soliton_translates(self):
        """Soliton peak should move to the right over time."""
        sim = KdVSimulation(_make_config(c=2.0, x0=10.0, N=512, L=60.0, dt=1e-4))
        sim.init_type = "soliton"
        sim.reset()
        x0 = sim.measure_peak_position()

        for _ in range(500):
            sim.step()
        x1 = sim.measure_peak_position()

        displacement = x1 - x0
        if displacement < -sim.L / 2:
            displacement += sim.L
        assert displacement > 0, "Soliton should move to the right"

    def test_shape_preserved_during_translation(self):
        """Soliton should maintain its shape during propagation."""
        c = 2.0
        sim = KdVSimulation(_make_config(c=c, x0=10.0, N=512, L=60.0, dt=1e-4))
        sim.init_type = "soliton"
        sim.reset()
        amp_0 = sim.measure_peak_amplitude()

        for _ in range(500):
            sim.step()
        amp_1 = sim.measure_peak_amplitude()

        np.testing.assert_allclose(amp_0, amp_1, rtol=0.01,
                                   err_msg="Soliton amplitude changed during propagation")

    def test_soliton_speed_matches_c(self):
        """Measured soliton speed should approximately match c."""
        c = 2.0
        # Use enough steps so the soliton travels many grid spacings
        # for accurate speed measurement via parabolic interpolation.
        sim = KdVSimulation(_make_config(c=c, x0=10.0, N=512, L=80.0, dt=1e-4))
        sim.init_type = "soliton"
        sim.reset()

        measured_speed = sim.measure_soliton_speed(n_steps=5000)
        np.testing.assert_allclose(measured_speed, c, rtol=0.05,
                                   err_msg=f"Speed {measured_speed} != c={c}")

    def test_faster_soliton_travels_farther(self):
        """Soliton with higher c should travel farther in same time."""
        n_steps = 300
        dt = 1e-4

        displacements = []
        for c in [1.0, 4.0]:
            sim = KdVSimulation(_make_config(c=c, x0=10.0, N=512, L=80.0, dt=dt))
            sim.init_type = "soliton"
            sim.reset()
            x0 = sim.measure_peak_position()
            for _ in range(n_steps):
                sim.step()
            x1 = sim.measure_peak_position()
            d = x1 - x0
            if d < -sim.L / 2:
                d += sim.L
            displacements.append(d)

        assert displacements[1] > displacements[0], (
            "Faster soliton should travel farther"
        )


class TestSpeedAmplitudeRelation:
    """Tests for the c = 2*A relation."""

    def test_speed_amplitude_sweep(self):
        """Sweep over c values and verify A = c/2."""
        sim = KdVSimulation(_make_config(N=512, L=60.0))
        c_values = [0.5, 1.0, 2.0, 4.0, 6.0]
        data = sim.speed_amplitude_sweep(c_values)

        np.testing.assert_allclose(
            data["amplitudes"], data["theoretical_amplitudes"], rtol=1e-4,
        )

    def test_analytical_amplitude(self):
        assert KdVSimulation.analytical_soliton_amplitude(2.0) == pytest.approx(1.0)
        assert KdVSimulation.analytical_soliton_amplitude(4.0) == pytest.approx(2.0)
        assert KdVSimulation.analytical_soliton_amplitude(0.5) == pytest.approx(0.25)

    def test_analytical_width_decreases_with_c(self):
        """Faster solitons should be narrower."""
        w1 = KdVSimulation.analytical_soliton_width(1.0)
        w2 = KdVSimulation.analytical_soliton_width(4.0)
        assert w2 < w1, "Faster soliton should be narrower"


class TestConservation:
    """Tests for conserved quantities."""

    def test_mass_conservation(self):
        """Mass integral(u) should be conserved."""
        sim = KdVSimulation(_make_config(c=2.0, x0=10.0, N=256, L=40.0, dt=1e-4))
        sim.init_type = "soliton"
        sim.reset()

        M0 = sim.compute_mass()
        for _ in range(500):
            sim.step()
        M1 = sim.compute_mass()

        drift = abs(M1 - M0) / abs(M0)
        assert drift < 1e-4, f"Mass drift {drift:.2e} exceeds threshold"

    def test_momentum_conservation(self):
        """Momentum integral(u^2) should be conserved."""
        sim = KdVSimulation(_make_config(c=2.0, x0=10.0, N=256, L=40.0, dt=1e-4))
        sim.init_type = "soliton"
        sim.reset()

        P0 = sim.compute_momentum()
        for _ in range(500):
            sim.step()
        P1 = sim.compute_momentum()

        drift = abs(P1 - P0) / abs(P0)
        assert drift < 1e-3, f"Momentum drift {drift:.2e} exceeds threshold"

    def test_energy_conservation(self):
        """Energy integral(u^3 - u_x^2/2) should be conserved."""
        sim = KdVSimulation(_make_config(c=2.0, x0=10.0, N=256, L=40.0, dt=1e-4))
        sim.init_type = "soliton"
        sim.reset()

        E0 = sim.compute_energy()
        for _ in range(500):
            sim.step()
        E1 = sim.compute_energy()

        drift = abs(E1 - E0) / abs(E0)
        assert drift < 1e-3, f"Energy drift {drift:.2e} exceeds threshold"

    def test_mass_positive_for_soliton(self):
        """Mass should be positive for a positive soliton."""
        sim = KdVSimulation(_make_config(c=2.0))
        sim.init_type = "soliton"
        sim.reset()
        assert sim.compute_mass() > 0

    def test_momentum_positive_for_soliton(self):
        """Momentum (integral u^2) is always non-negative."""
        sim = KdVSimulation(_make_config(c=2.0))
        sim.init_type = "soliton"
        sim.reset()
        assert sim.compute_momentum() > 0


class TestTwoSolitonCollision:
    """Tests for elastic soliton collision."""

    def test_two_soliton_init(self):
        """Two-soliton initial condition should have two peaks."""
        sim = KdVSimulation(_make_config(c=2.0, N=512, L=60.0))
        sim.init_type = "two_soliton"
        state = sim.reset()

        # Find peaks: look for local maxima above threshold
        threshold = 0.1
        above = state > threshold
        # Count connected regions above threshold
        transitions = np.diff(above.astype(int))
        n_peaks = np.sum(transitions == 1)
        assert n_peaks >= 2, f"Expected 2 peaks, found {n_peaks}"

    def test_set_two_solitons(self):
        """Custom two-soliton setup should work."""
        sim = KdVSimulation(_make_config(N=512, L=60.0))
        sim.reset()
        sim.set_two_solitons(4.0, 10.0, 1.0, 30.0)
        state = sim.observe()
        assert np.max(state) > 1.5  # Fast soliton has amplitude 2.0

    def test_mass_conserved_during_collision(self):
        """Mass should be conserved through a two-soliton collision."""
        sim = KdVSimulation(_make_config(N=512, L=80.0, dt=1e-4))
        sim.set_two_solitons(4.0, 15.0, 1.0, 40.0)

        M0 = sim.compute_mass()
        for _ in range(2000):
            sim.step()
        M1 = sim.compute_mass()

        drift = abs(M1 - M0) / abs(M0)
        assert drift < 1e-3, f"Mass drift during collision: {drift:.2e}"

    def test_momentum_conserved_during_collision(self):
        """Momentum should be conserved through a two-soliton collision."""
        sim = KdVSimulation(_make_config(N=512, L=80.0, dt=1e-4))
        sim.set_two_solitons(4.0, 15.0, 1.0, 40.0)

        P0 = sim.compute_momentum()
        for _ in range(2000):
            sim.step()
        P1 = sim.compute_momentum()

        drift = abs(P1 - P0) / abs(P0)
        assert drift < 5e-3, f"Momentum drift during collision: {drift:.2e}"


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_deterministic_runs(self):
        """Two runs with same config should give identical results."""
        config = _make_config(c=2.0, N=128, dt=1e-4, n_steps=200)

        sim1 = KdVSimulation(config)
        sim1.init_type = "soliton"
        sim1.reset()
        for _ in range(200):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = KdVSimulation(config)
        sim2.init_type = "soliton"
        sim2.reset()
        for _ in range(200):
            sim2.step()
        s2 = sim2.observe()

        np.testing.assert_array_equal(s1, s2)

    def test_reset_returns_same_state(self):
        """Reset should always return the same initial condition."""
        sim = KdVSimulation(_make_config())
        s0 = sim.reset().copy()
        for _ in range(50):
            sim.step()
        s_reset = sim.reset()
        np.testing.assert_array_equal(s0, s_reset)


class TestStability:
    """Numerical stability tests."""

    def test_no_nan_inf(self):
        """Solution should stay finite over many steps."""
        sim = KdVSimulation(_make_config(c=2.0, N=256, L=40.0, dt=1e-4))
        sim.init_type = "soliton"
        sim.reset()

        for _ in range(2000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "Solution became NaN/Inf"

    def test_zero_init_stable(self):
        """Zero initial condition should remain zero."""
        sim = KdVSimulation(_make_config(N=128, L=20.0, dt=1e-4))
        sim.init_type = "zero"
        sim.reset()

        for _ in range(500):
            sim.step()

        state = sim.observe()
        assert np.max(np.abs(state)) < 1e-12, (
            f"Zero state drifted: max |u| = {np.max(np.abs(state))}"
        )

    def test_gaussian_init_does_not_blow_up(self):
        """Gaussian initial condition should not blow up."""
        sim = KdVSimulation(_make_config(N=256, L=40.0, dt=1e-4))
        sim.init_type = "gaussian"
        sim.reset()

        max_initial = np.max(np.abs(sim.observe()))
        for _ in range(500):
            sim.step()

        max_final = np.max(np.abs(sim.observe()))
        # Should not grow by more than 10x
        assert max_final < 10 * max_initial, (
            f"Gaussian blew up: {max_initial:.4f} -> {max_final:.4f}"
        )


class TestParameterSweep:
    """Tests for parameter sweeps."""

    def test_different_c_values(self):
        """Simulation should work for various soliton speeds."""
        for c in [0.5, 1.0, 2.0, 4.0]:
            sim = KdVSimulation(_make_config(c=c, N=256, L=60.0, dt=1e-4))
            sim.init_type = "soliton"
            sim.reset()
            for _ in range(100):
                state = sim.step()
            assert np.all(np.isfinite(state)), f"Failed for c={c}"

    def test_different_grid_sizes(self):
        """Simulation should work for different N."""
        for N in [64, 128, 256]:
            sim = KdVSimulation(_make_config(N=N, dt=1e-4))
            sim.init_type = "soliton"
            state = sim.reset()
            assert state.shape == (N,)
            for _ in range(50):
                state = sim.step()
            assert np.all(np.isfinite(state)), f"Failed for N={N}"

    def test_different_domain_lengths(self):
        """Simulation should work for various L.

        The domain must be large enough that the soliton tails decay to
        near zero before wrapping around the periodic boundary.
        """
        for L in [30.0, 40.0, 80.0]:
            sim = KdVSimulation(_make_config(L=L, N=256, dt=1e-4))
            sim.init_type = "soliton"
            sim.reset()
            for _ in range(100):
                state = sim.step()
            assert np.all(np.isfinite(state)), f"Failed for L={L}"


class TestTrajectoryData:
    """Tests for the run() method and TrajectoryData output."""

    def test_run_returns_trajectory(self):
        sim = KdVSimulation(_make_config(N=128, dt=1e-4, n_steps=50))
        sim.init_type = "soliton"
        traj = sim.run(50)

        assert traj.states.shape == (51, 128)  # 50 steps + initial
        assert len(traj.timestamps) == 51
        assert traj.timestamps[0] == 0.0

    def test_trajectory_timestamps_correct(self):
        dt = 1e-4
        n_steps = 20
        sim = KdVSimulation(_make_config(N=64, dt=dt, n_steps=n_steps))
        sim.init_type = "soliton"
        traj = sim.run(n_steps)

        expected_times = np.arange(n_steps + 1) * dt
        np.testing.assert_allclose(traj.timestamps, expected_times, rtol=1e-10)

    def test_trajectory_parameters(self):
        sim = KdVSimulation(_make_config(c=3.0, N=64, dt=1e-4, n_steps=10))
        sim.init_type = "soliton"
        traj = sim.run(10)
        assert traj.parameters["c"] == 3.0


class TestRediscoveryData:
    """Tests for the rediscovery data generation functions."""

    def test_speed_amplitude_data(self):
        from simulating_anything.rediscovery.kdv import generate_speed_amplitude_data
        data = generate_speed_amplitude_data(n_speeds=5, N=128, L=40.0)
        assert len(data["c_values"]) == 5
        assert len(data["amplitudes"]) == 5
        assert len(data["theoretical_amplitudes"]) == 5
        # Check A = c/2 relation
        np.testing.assert_allclose(
            data["amplitudes"], data["theoretical_amplitudes"], rtol=1e-3,
        )

    def test_conservation_data(self):
        from simulating_anything.rediscovery.kdv import generate_conservation_data
        data = generate_conservation_data(n_steps=200, dt=1e-4, N=128, L=40.0)
        assert len(data["masses"]) >= 2
        assert len(data["momenta"]) >= 2
        assert len(data["energies"]) >= 2
        # Mass should be roughly conserved
        M = data["masses"]
        drift = abs(M[-1] - M[0]) / abs(M[0])
        assert drift < 0.01

    def test_collision_data(self):
        from simulating_anything.rediscovery.kdv import generate_collision_data
        data = generate_collision_data(n_steps=500, dt=1e-4, N=256, L=60.0)
        assert data["mass_drift"] < 0.01
        assert data["momentum_drift"] < 0.01
        assert data["c_fast"] > data["c_slow"]
