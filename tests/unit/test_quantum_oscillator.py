"""Tests for the quantum harmonic oscillator simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.quantum_oscillator import QuantumHarmonicOscillator
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestQuantumHarmonicOscillator:
    """Tests for the quantum oscillator simulation."""

    def _make_sim(self, **kwargs) -> QuantumHarmonicOscillator:
        defaults = {
            "m": 1.0, "omega": 1.0, "hbar": 1.0,
            "N": 128.0, "x_max": 10.0, "x_0": 2.0, "p_0": 0.0,
        }
        defaults.update(kwargs)
        config = SimulationConfig(
            domain=Domain.QUANTUM_OSCILLATOR,
            dt=0.01,
            n_steps=500,
            parameters=defaults,
        )
        return QuantumHarmonicOscillator(config)

    def test_creation(self):
        """Simulation can be created with default parameters."""
        sim = self._make_sim()
        assert sim.m == 1.0
        assert sim.omega == 1.0
        assert sim.hbar == 1.0
        assert sim.N == 128

    def test_default_params(self):
        """Default parameters are set correctly."""
        config = SimulationConfig(
            domain=Domain.QUANTUM_OSCILLATOR,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = QuantumHarmonicOscillator(config)
        assert sim.m == 1.0
        assert sim.omega == 1.0
        assert sim.hbar == 1.0
        assert sim.N == 128
        assert sim.x_max == 10.0
        assert sim.x_0 == 2.0

    def test_custom_params(self):
        """Custom parameters are picked up correctly."""
        sim = self._make_sim(m=2.0, omega=3.0, hbar=0.5, N=64.0, x_max=8.0)
        assert sim.m == 2.0
        assert sim.omega == 3.0
        assert sim.hbar == 0.5
        assert sim.N == 64
        assert sim.x_max == 8.0

    def test_step(self):
        """Single step produces a changed state."""
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert state1.shape == (128,)
        # State should change (wavepacket moves)
        assert not np.allclose(state0, state1)

    def test_trajectory(self):
        """Full trajectory run produces correct shape."""
        sim = self._make_sim(N=64.0)
        traj = sim.run(n_steps=50)
        # n_steps + 1 states (including initial)
        assert traj.states.shape == (51, 64)
        # All probability densities should be non-negative
        assert np.all(traj.states >= -1e-15)

    def test_norm_conservation(self):
        """Wavefunction norm should be conserved to high precision."""
        sim = self._make_sim(N=256.0, x_max=12.0)
        sim.reset()
        norm_0 = sim.norm

        for _ in range(500):
            sim.step()

        norm_f = sim.norm
        assert abs(norm_f - norm_0) < 1e-10, (
            f"Norm drift: {abs(norm_f - norm_0):.2e}"
        )

    def test_energy_conservation(self):
        """For an eigenstate, energy expectation should be constant."""
        sim = self._make_sim(N=256.0, x_max=12.0)
        sim.reset()

        # Prepare ground state (eigenstate)
        E_0 = sim.measure_energy_from_eigenstate(0)

        # Evolve and check energy stays constant
        for _ in range(200):
            sim.step()

        E_final = sim.energy
        assert abs(E_final - E_0) / abs(E_0) < 1e-6, (
            f"Energy drift: {abs(E_final - E_0):.2e}, E_0={E_0:.6f}"
        )

    def test_ground_state_energy(self):
        """Ground state energy should be E_0 = 0.5 * hbar * omega."""
        sim = self._make_sim(N=256.0, x_max=15.0, omega=1.0)
        sim.reset()

        E_measured = sim.measure_energy_from_eigenstate(0)
        E_theory = 0.5 * sim.hbar * sim.omega

        assert abs(E_measured - E_theory) / E_theory < 0.01, (
            f"E_measured={E_measured:.6f}, E_theory={E_theory:.6f}"
        )

    def test_coherent_state_oscillation(self):
        """Coherent state <x> should oscillate as x_0*cos(omega*t)."""
        x_0 = 2.0
        omega = 1.0
        sim = self._make_sim(
            N=256.0, x_max=15.0, x_0=x_0, omega=omega, p_0=0.0,
        )
        sim.reset()

        dt = 0.01
        # Evolve for a quarter period: t = pi/(2*omega)
        quarter_T = int(np.pi / (2 * omega) / dt)
        for _ in range(quarter_T):
            sim.step()

        t = quarter_T * dt
        x_expect = sim.position_expectation
        x_theory = x_0 * np.cos(omega * t)

        # At t = pi/2, cos(t) ~ 0, so <x> should be near 0
        assert abs(x_expect - x_theory) < 0.15, (
            f"<x>={x_expect:.4f}, theory={x_theory:.4f}"
        )

    def test_probability_positive(self):
        """Probability density |psi|^2 should be non-negative everywhere."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(100):
            sim.step()
            obs = sim.observe()
            assert np.all(obs >= -1e-15), f"Negative probability: min={np.min(obs)}"

    def test_wavefunction_localization(self):
        """Ground state probability should peak at x=0."""
        sim = self._make_sim(N=256.0, x_max=15.0, x_0=0.0)
        sim.reset()

        # Prepare ground state
        sim.measure_energy_from_eigenstate(0)
        prob = np.abs(sim._psi) ** 2

        # Peak should be at center (x=0)
        peak_idx = np.argmax(prob)
        x_peak = sim.x[peak_idx]
        assert abs(x_peak) < 0.5, f"Peak at x={x_peak}, expected near 0"

    def test_observe_returns_probability_density(self):
        """observe() should return |psi|^2, not the complex wavefunction."""
        sim = self._make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.dtype in [np.float64, np.float32]
        assert obs.shape == (128,)
        assert np.all(np.isreal(obs))

    def test_eigenstate_energy_spectrum(self):
        """Multiple eigenstates should have E_n = (n+0.5)*hbar*omega."""
        sim = self._make_sim(N=256.0, x_max=15.0)
        sim.reset()

        for n in range(5):
            E_n = sim.measure_energy_from_eigenstate(n)
            E_theory = sim.hbar * sim.omega * (n + 0.5)
            rel_err = abs(E_n - E_theory) / E_theory
            assert rel_err < 0.02, (
                f"n={n}: E_measured={E_n:.6f}, E_theory={E_theory:.6f}, "
                f"rel_err={rel_err:.4f}"
            )


class TestQuantumOscillatorRediscovery:
    """Tests for quantum oscillator data generation."""

    def test_energy_spectrum_data(self):
        from simulating_anything.rediscovery.quantum_oscillator import (
            generate_energy_spectrum_data,
        )
        data = generate_energy_spectrum_data(n_states=5)
        assert "n" in data
        assert "E_measured" in data
        assert "E_theory" in data
        assert len(data["n"]) == 5
        # Ground state should be close to 0.5
        assert abs(data["E_measured"][0] - 0.5) < 0.05

    def test_omega_sweep_data(self):
        from simulating_anything.rediscovery.quantum_oscillator import (
            generate_omega_sweep_data,
        )
        data = generate_omega_sweep_data(n_points=5)
        assert "omega" in data
        assert "E0_measured" in data
        assert "E0_theory" in data
        assert len(data["omega"]) == 5
        # E_0 should scale linearly with omega
        rel_err = np.abs(
            data["E0_measured"] - data["E0_theory"]
        ) / data["E0_theory"]
        assert np.mean(rel_err) < 0.05
