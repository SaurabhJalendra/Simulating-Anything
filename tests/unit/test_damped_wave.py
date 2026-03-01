"""Tests for the 1D damped wave equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.damped_wave import DampedWave1D
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    c: float = 1.0,
    gamma: float = 0.1,
    N: int = 64,
    L: float = 2 * np.pi,
    dt: float = 0.001,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.DAMPED_WAVE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "c": c,
            "gamma": gamma,
            "N": float(N),
            "L": L,
        },
    )


def _make_sim(**kwargs) -> DampedWave1D:
    return DampedWave1D(_make_config(**kwargs))


class TestDampedWaveCreation:
    def test_creation(self):
        sim = _make_sim()
        assert sim.c == 1.0
        assert sim.gamma == 0.1
        assert sim.N == 64
        assert sim.L == pytest.approx(2 * np.pi)

    def test_custom_params(self):
        sim = _make_sim(c=2.0, gamma=0.5, N=128, L=10.0)
        assert sim.c == 2.0
        assert sim.gamma == 0.5
        assert sim.N == 128
        assert sim.L == 10.0

    def test_default_params(self):
        """Default parameters are used when not specified."""
        config = SimulationConfig(
            domain=Domain.DAMPED_WAVE,
            dt=0.001,
            n_steps=100,
            parameters={},
        )
        sim = DampedWave1D(config)
        assert sim.c == 1.0
        assert sim.gamma == 0.1
        assert sim.N == 64

    def test_observe_shape(self):
        """Observe returns [u_1,...,u_N, v_1,...,v_N] with shape (2*N,)."""
        N = 64
        sim = _make_sim(N=N)
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2 * N,)

    def test_step_advances_state(self):
        sim = _make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_trajectory_shape(self):
        """Run returns trajectory with correct state dimension."""
        N = 32
        n_steps = 50
        sim = _make_sim(N=N, n_steps=n_steps)
        traj = sim.run(n_steps=n_steps)
        assert traj.states.shape == (n_steps + 1, 2 * N)


class TestDampedWaveEnergy:
    def test_energy_conserved_undamped(self):
        """Energy should be conserved when gamma=0 (< 1e-6 drift over 10000 steps)."""
        sim = _make_sim(c=1.0, gamma=0.0, N=64, dt=0.001)
        sim.reset()
        E0 = sim.total_energy
        assert E0 > 0  # Sanity: should have nonzero energy

        for _ in range(10000):
            sim.step()

        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_energy_decreases_with_damping(self):
        """Energy should decrease monotonically with damping gamma > 0."""
        sim = _make_sim(c=1.0, gamma=0.5, N=64, dt=0.001)
        sim.reset()
        E_prev = sim.total_energy

        # Check energy decreases at each sample point
        decreasing = True
        for step in range(500):
            sim.step()
            if (step + 1) % 50 == 0:
                E = sim.total_energy
                if E > E_prev + 1e-10:  # small tolerance for numerical noise
                    decreasing = False
                E_prev = E

        assert decreasing, "Energy should decrease monotonically with damping"

    def test_energy_higher_damping_faster_decay(self):
        """Higher damping should cause faster energy decay."""
        E_final = []
        for gam in [0.1, 1.0]:
            sim = _make_sim(c=1.0, gamma=gam, N=64, dt=0.001)
            sim.reset()
            for _ in range(2000):
                sim.step()
            E_final.append(sim.total_energy)

        # Higher damping -> less energy remaining
        assert E_final[1] < E_final[0]

    def test_kinetic_potential_energy_nonneg(self):
        """Kinetic and potential energies should be non-negative."""
        sim = _make_sim()
        sim.reset()
        for _ in range(100):
            sim.step()
        assert sim.kinetic_energy >= 0
        assert sim.potential_energy >= 0


class TestDampedWavePhysics:
    def test_wave_speed_from_dispersion(self):
        """Wave speed c can be measured from mode-1 frequency: c = omega/k.

        For an undamped wave (gamma=0), mode 1 on [0, 2*pi] has k=1,
        so omega = c. We measure omega from zero crossings of the mode.
        """
        c_val = 2.0
        N = 64
        L = 2 * np.pi
        dt = 0.001
        n_steps = 10000

        sim = _make_sim(c=c_val, gamma=0.0, N=N, L=L, dt=dt)
        sim.init_type = "sine"
        sim.reset()

        # Track displacement at x = L/4
        track_idx = N // 4
        positions = [sim.observe()[track_idx]]
        for _ in range(n_steps):
            sim.step()
            positions.append(sim.observe()[track_idx])

        positions = np.array(positions)

        # Find positive-going zero crossings
        crossings = []
        for j in range(1, len(positions)):
            if positions[j - 1] < 0 and positions[j] >= 0:
                frac = -positions[j - 1] / (positions[j] - positions[j - 1])
                crossings.append((j - 1 + frac) * dt)

        assert len(crossings) >= 3, f"Only {len(crossings)} crossings found"
        periods = np.diff(crossings)
        T_measured = float(np.median(periods))
        omega_measured = 2 * np.pi / T_measured

        # k=1 for mode 1 on [0, 2*pi], so c_measured = omega/k = omega
        c_measured = omega_measured
        rel_err = abs(c_measured - c_val) / c_val
        assert rel_err < 0.02, (
            f"Wave speed error {rel_err:.4%}: "
            f"measured={c_measured:.4f}, expected={c_val:.4f}"
        )

    def test_mode_frequency_matches_theory(self):
        """For a sine initial condition (mode 1), the oscillation frequency
        should match omega_1 = sqrt(c^2 * k^2 - gamma^2/4).
        """
        c_val = 2.0
        gamma_val = 0.1
        N = 64
        L = 2 * np.pi
        dt = 0.001
        n_steps = 10000

        sim = _make_sim(c=c_val, gamma=gamma_val, N=N, L=L, dt=dt)
        sim.init_type = "sine"
        sim.reset()

        # Track displacement at x = L/4
        track_idx = N // 4
        positions = [sim.observe()[track_idx]]
        for _ in range(n_steps):
            sim.step()
            positions.append(sim.observe()[track_idx])

        positions = np.array(positions)

        # Find positive-going zero crossings
        crossings = []
        for j in range(1, len(positions)):
            if positions[j - 1] < 0 and positions[j] >= 0:
                frac = -positions[j - 1] / (positions[j] - positions[j - 1])
                crossings.append((j - 1 + frac) * dt)

        assert len(crossings) >= 3, f"Only {len(crossings)} crossings found"
        periods = np.diff(crossings)
        T_measured = float(np.median(periods))
        omega_measured = 2 * np.pi / T_measured

        k1 = 2 * np.pi / L  # = 1.0
        omega_theory = np.sqrt(c_val ** 2 * k1 ** 2 - gamma_val ** 2 / 4)

        rel_err = abs(omega_measured - omega_theory) / omega_theory
        assert rel_err < 0.02, (
            f"Mode frequency error {rel_err:.4%}: "
            f"measured={omega_measured:.4f}, theory={omega_theory:.4f}"
        )

    def test_damping_rate(self):
        """Mode amplitude envelope should decay at rate gamma/2.

        The displacement Fourier mode oscillates as cos(omega*t)*exp(-gamma/2*t),
        so we track the peak envelope of |u_hat[mode]| and fit the decay rate.
        """
        gamma_val = 0.2
        c_val = 1.0
        N = 64
        L = 2 * np.pi
        dt = 0.001
        n_steps = 20000
        mode = 1

        sim = _make_sim(c=c_val, gamma=gamma_val, N=N, L=L, dt=dt)
        sim.init_type = "sine"
        sim.reset()

        # Collect mode amplitude samples every 10 steps
        amps = []
        times_arr = []
        for s in range(n_steps):
            sim.step()
            if (s + 1) % 10 == 0:
                u_hat = np.fft.fft(sim.observe()[:N])
                a = np.abs(u_hat[mode])
                amps.append(a)
                times_arr.append((s + 1) * dt)

        amps = np.array(amps)
        times_arr = np.array(times_arr)

        # Find local maxima of the amplitude (peak envelope)
        peak_indices = []
        for i in range(1, len(amps) - 1):
            if amps[i] > amps[i - 1] and amps[i] > amps[i + 1]:
                peak_indices.append(i)

        assert len(peak_indices) >= 3, (
            f"Only {len(peak_indices)} amplitude peaks found"
        )

        peak_amps = amps[peak_indices]
        peak_times = times_arr[peak_indices]

        # Fit log(peak_amp) = log(a0) - (gamma/2)*t via linear regression
        log_amps = np.log(peak_amps)
        coeffs = np.polyfit(peak_times, log_amps, 1)
        measured_rate = -coeffs[0]
        theory_rate = gamma_val / 2.0

        rel_err = abs(measured_rate - theory_rate) / theory_rate
        assert rel_err < 0.02, (
            f"Damping rate error {rel_err:.4%}: "
            f"measured={measured_rate:.4f}, theory={theory_rate:.4f}"
        )

    def test_theoretical_frequency_method(self):
        """The theoretical_frequency method should return correct values."""
        sim = _make_sim(c=2.0, gamma=0.0, N=64, L=2 * np.pi)
        sim.reset()

        # For mode 1, k = 1.0, gamma=0: omega = c*k = 2.0
        omega = sim.theoretical_frequency(1)
        assert omega == pytest.approx(2.0, rel=1e-10)

    def test_theoretical_frequency_damped(self):
        """Damped frequency should be less than undamped."""
        c_val = 2.0
        gam = 1.0
        sim = _make_sim(c=c_val, gamma=gam, N=64, L=2 * np.pi)
        sim.reset()

        omega_damped = sim.theoretical_frequency(1)
        omega_undamped = c_val * (2 * np.pi / sim.L)

        assert omega_damped < omega_undamped
        expected = np.sqrt(c_val ** 2 - gam ** 2 / 4)
        assert omega_damped == pytest.approx(expected, rel=1e-10)

    def test_overdamped_frequency_zero(self):
        """If gamma > 2*c*k, the mode is overdamped and frequency should be 0."""
        # c=1, k=1 (mode 1, L=2*pi), so 2*c*k = 2. gamma=3 > 2.
        sim = _make_sim(c=1.0, gamma=3.0, N=64, L=2 * np.pi)
        sim.reset()
        omega = sim.theoretical_frequency(1)
        assert omega == 0.0

    def test_theoretical_decay_rate(self):
        sim = _make_sim(gamma=0.6)
        sim.reset()
        assert sim.theoretical_decay_rate() == pytest.approx(0.3)

    def test_dominant_wavenumber(self):
        """For a sine init (mode 1), dominant wavenumber should be 1."""
        sim = _make_sim(N=64)
        sim.init_type = "sine"
        sim.reset()
        assert sim.dominant_wavenumber == 1

    def test_mode_amplitudes_shape(self):
        """mode_amplitudes should return N complex coefficients."""
        N = 64
        sim = _make_sim(N=N)
        sim.reset()
        amps = sim.mode_amplitudes()
        assert amps.shape == (N,)
        assert np.iscomplexobj(amps)


class TestDampedWaveRediscovery:
    def test_decay_rate_data_generation(self):
        from simulating_anything.rediscovery.damped_wave import (
            generate_decay_rate_data,
        )
        data = generate_decay_rate_data(
            n_gamma=5, n_steps=2000, dt=0.001, N=32,
        )
        assert len(data["gamma_values"]) == 5
        assert len(data["decay_measured"]) == 5
        valid = np.isfinite(data["decay_measured"])
        assert np.sum(valid) >= 3

    def test_dispersion_data_generation(self):
        from simulating_anything.rediscovery.damped_wave import (
            generate_dispersion_data,
        )
        # Use enough steps so that even slow waves (c=0.5, period~12.6s)
        # have at least a few zero crossings. Total time = 20000*0.001 = 20s.
        data = generate_dispersion_data(
            n_c=5, n_steps=20000, dt=0.001, N=32, gamma=0.0,
        )
        assert len(data["c_values"]) == 5
        assert len(data["omega_measured"]) == 5
        valid = np.isfinite(data["omega_measured"])
        assert np.sum(valid) >= 3

    def test_wave_speed_data_generation(self):
        from simulating_anything.rediscovery.damped_wave import (
            generate_wave_speed_data,
        )
        data = generate_wave_speed_data(
            n_c=5, n_steps=1000, dt=0.001, N=64,
        )
        assert len(data["c_values"]) == 5
        assert len(data["speed_measured"]) == 5
