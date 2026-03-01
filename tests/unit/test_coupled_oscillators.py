"""Tests for the coupled harmonic oscillators simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.coupled_oscillators import CoupledOscillators
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestCoupledOscillators:
    """Tests for the coupled oscillators simulation."""

    def _make_sim(self, **kwargs) -> CoupledOscillators:
        defaults = {
            "k": 4.0, "m": 1.0, "kc": 0.5,
            "x1_0": 1.0, "v1_0": 0.0, "x2_0": 0.0, "v2_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.COUPLED_OSCILLATORS,
            dt=0.001,
            n_steps=10000,
            parameters=defaults,
        )
        return CoupledOscillators(config)

    def test_creation(self):
        """Simulation should be created with correct parameters."""
        sim = self._make_sim(k=3.0, m=2.0, kc=0.7)
        assert sim.k == 3.0
        assert sim.m == 2.0
        assert sim.kc == 0.7

    def test_initial_state(self):
        """Reset should produce the correct initial state [x1, v1, x2, v2]."""
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (4,)
        assert np.isclose(state[0], 1.0)  # x1_0
        assert np.isclose(state[1], 0.0)  # v1_0
        assert np.isclose(state[2], 0.0)  # x2_0
        assert np.isclose(state[3], 0.0)  # v2_0

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
        """Total energy should be conserved (no damping)."""
        sim = self._make_sim(k=4.0, m=1.0, kc=0.5)
        sim.reset()
        E0 = sim.total_energy
        for _ in range(10000):
            sim.step()
        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-8, f"Energy drift {rel_drift:.2e} too large"

    def test_frequency_properties(self):
        """Normal mode frequency properties should match theory."""
        sim = self._make_sim(k=4.0, m=1.0, kc=0.5)
        assert np.isclose(sim.omega_symmetric, 2.0)  # sqrt(4/1)
        assert np.isclose(
            sim.omega_antisymmetric,
            np.sqrt(5.0),  # sqrt((4 + 2*0.5)/1)
        )
        expected_beat = np.sqrt(5.0) - 2.0
        assert np.isclose(sim.beat_frequency, expected_beat)

    def test_symmetric_mode_frequency(self):
        """Symmetric mode (x1=x2) should oscillate at omega_s = sqrt(k/m)."""
        k, m, kc = 4.0, 1.0, 1.0
        sim = self._make_sim(k=k, m=m, kc=kc, x1_0=1.0, x2_0=1.0)
        sim.reset()

        positions = [sim.observe()[0]]
        for _ in range(30000):
            positions.append(sim.step()[0])
        positions = np.array(positions)

        # Measure frequency via zero crossings
        crossings = []
        for j in range(1, len(positions)):
            if positions[j - 1] < 0 and positions[j] >= 0:
                frac = -positions[j - 1] / (positions[j] - positions[j - 1])
                crossings.append((j - 1 + frac) * 0.001)

        assert len(crossings) >= 3
        periods = np.diff(crossings)
        T_measured = np.median(periods)
        omega_measured = 2 * np.pi / T_measured
        omega_theory = np.sqrt(k / m)
        assert abs(omega_measured - omega_theory) / omega_theory < 0.005

    def test_antisymmetric_mode_frequency(self):
        """Antisymmetric mode (x1=-x2) should oscillate at omega_a."""
        k, m, kc = 4.0, 1.0, 1.0
        sim = self._make_sim(k=k, m=m, kc=kc, x1_0=1.0, x2_0=-1.0)
        sim.reset()

        positions = [sim.observe()[0]]
        for _ in range(30000):
            positions.append(sim.step()[0])
        positions = np.array(positions)

        crossings = []
        for j in range(1, len(positions)):
            if positions[j - 1] < 0 and positions[j] >= 0:
                frac = -positions[j - 1] / (positions[j] - positions[j - 1])
                crossings.append((j - 1 + frac) * 0.001)

        assert len(crossings) >= 3
        periods = np.diff(crossings)
        T_measured = np.median(periods)
        omega_measured = 2 * np.pi / T_measured
        omega_theory = np.sqrt((k + 2 * kc) / m)
        assert abs(omega_measured - omega_theory) / omega_theory < 0.005

    def test_zero_coupling_independent(self):
        """With kc=0, oscillators should be independent."""
        sim = self._make_sim(
            k=4.0, m=1.0, kc=0.0,
            x1_0=1.0, v1_0=0.0, x2_0=0.0, v2_0=0.0,
        )
        sim.reset()

        # Oscillator 2 should stay at rest if kc=0
        for _ in range(5000):
            state = sim.step()
        assert abs(state[2]) < 1e-12, "x2 should remain zero with no coupling"
        assert abs(state[3]) < 1e-12, "v2 should remain zero with no coupling"

    def test_energy_transfer(self):
        """Energy should transfer between oscillators (beats)."""
        sim = self._make_sim(
            k=4.0, m=1.0, kc=0.5,
            x1_0=1.0, v1_0=0.0, x2_0=0.0, v2_0=0.0,
        )
        sim.reset()

        # Initially all energy in oscillator 1
        E1_init = sim.energy_oscillator_1()
        E2_init = sim.energy_oscillator_2()
        assert E1_init > 0.1
        assert E2_init < 1e-12

        # Run for ~half a beat period: energy should move to osc 2
        T_beat = sim.beat_period
        half_beat_steps = int(T_beat / (2.0 * sim.config.dt))
        for _ in range(half_beat_steps):
            sim.step()

        _ = sim.energy_oscillator_1()  # Not asserted, but verifies no crash
        E2_half = sim.energy_oscillator_2()
        # Oscillator 2 should have gained significant energy
        assert E2_half > 0.3 * E1_init, (
            f"Energy transfer failed: E2={E2_half:.4f}, E1_init={E1_init:.4f}"
        )

    def test_beat_frequency_correct(self):
        """Measured beat frequency should match theory."""
        k, m, kc = 4.0, 1.0, 0.5
        sim = self._make_sim(k=k, m=m, kc=kc)
        sim.reset()

        # Collect x1 amplitude envelope
        n_steps = 80000
        positions = np.zeros(n_steps + 1)
        positions[0] = sim.observe()[0]
        for i in range(n_steps):
            positions[i + 1] = sim.step()[0]

        # Find peaks of |x1|
        abs_pos = np.abs(positions)
        dt = 0.001
        window = 30  # ~half oscillation period
        envelope_times = []
        envelope_vals = []
        for j in range(window, len(abs_pos) - window):
            local_max = np.max(abs_pos[j - window:j + window + 1])
            if abs_pos[j] == local_max and abs_pos[j] > 0.3:
                envelope_times.append(j * dt)
                envelope_vals.append(abs_pos[j])

        # Find peaks of envelope (beat maxima)
        beat_times = []
        for j in range(1, len(envelope_vals) - 1):
            if (envelope_vals[j] > envelope_vals[j - 1]
                    and envelope_vals[j] > envelope_vals[j + 1]):
                beat_times.append(envelope_times[j])

        assert len(beat_times) >= 2, "Not enough beat cycles detected"
        T_beat_measured = np.median(np.diff(beat_times))
        omega_beat_measured = 2 * np.pi / T_beat_measured
        omega_beat_theory = np.sqrt((k + 2 * kc) / m) - np.sqrt(k / m)

        rel_error = abs(omega_beat_measured - omega_beat_theory) / omega_beat_theory
        assert rel_error < 0.05, (
            f"Beat frequency error {rel_error:.2%}: "
            f"measured={omega_beat_measured:.4f}, "
            f"theory={omega_beat_theory:.4f}"
        )

    def test_set_symmetric_mode(self):
        """set_symmetric_mode should set x1=x2, v1=v2=0."""
        sim = self._make_sim()
        sim.reset()
        sim.set_symmetric_mode(amplitude=2.0)
        state = sim.observe()
        assert np.isclose(state[0], 2.0)
        assert np.isclose(state[1], 0.0)
        assert np.isclose(state[2], 2.0)
        assert np.isclose(state[3], 0.0)

    def test_set_antisymmetric_mode(self):
        """set_antisymmetric_mode should set x1=-x2, v1=v2=0."""
        sim = self._make_sim()
        sim.reset()
        sim.set_antisymmetric_mode(amplitude=1.5)
        state = sim.observe()
        assert np.isclose(state[0], 1.5)
        assert np.isclose(state[1], 0.0)
        assert np.isclose(state[2], -1.5)
        assert np.isclose(state[3], 0.0)

    def test_run_trajectory(self):
        """run() should produce a TrajectoryData with correct shape."""
        sim = self._make_sim()
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)
        assert len(traj.timestamps) == 101

    def test_beat_period_inf_no_coupling(self):
        """With kc=0, beat period should be infinite."""
        sim = self._make_sim(kc=0.0)
        assert sim.beat_period == float("inf")


class TestCoupledOscillatorsRediscovery:
    """Tests for coupled oscillators data generation."""

    def test_normal_mode_data(self):
        from simulating_anything.rediscovery.coupled_oscillators import (
            generate_normal_mode_data,
        )
        data = generate_normal_mode_data(n_samples=3, n_steps=20000, dt=0.001)
        assert "k" in data
        assert "m" in data
        assert "kc" in data
        assert "omega_s_measured" in data
        assert "omega_a_measured" in data
        assert len(data["k"]) > 0
        # Symmetric mode should match theory
        s_err = (
            np.abs(data["omega_s_measured"] - data["omega_s_theory"])
            / data["omega_s_theory"]
        )
        assert np.mean(s_err) < 0.02

    def test_ode_data(self):
        from simulating_anything.rediscovery.coupled_oscillators import (
            generate_ode_data,
        )
        data = generate_ode_data(n_steps=100, dt=0.001)
        assert data["states"].shape == (101, 4)
        assert data["k"] == 4.0
        assert data["kc"] == 0.5
