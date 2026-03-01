"""Tests for the Toda lattice simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.toda_lattice import TodaLattice
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_sim(
    N: int = 8,
    a: float = 1.0,
    mode: int = 1,
    amplitude: float = 0.1,
    dt: float = 0.001,
    n_steps: int = 1000,
) -> TodaLattice:
    config = SimulationConfig(
        domain=Domain.TODA_LATTICE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "N": float(N),
            "a": a,
            "mode": float(mode),
            "amplitude": amplitude,
        },
    )
    return TodaLattice(config)


class TestTodaLatticeCreation:
    def test_creation(self):
        sim = _make_sim()
        assert sim.N == 8
        assert sim.a == 1.0

    def test_default_params(self):
        """Default parameters are used when not specified."""
        config = SimulationConfig(
            domain=Domain.TODA_LATTICE,
            dt=0.001,
            n_steps=100,
            parameters={},
        )
        sim = TodaLattice(config)
        assert sim.N == 8
        assert sim.a == 1.0

    def test_custom_params(self):
        sim = _make_sim(N=16, a=2.0)
        assert sim.N == 16
        assert sim.a == 2.0

    def test_observe_shape(self):
        """Observe returns [x_0,...,x_{N-1}, p_0,...,p_{N-1}] with shape (2*N,)."""
        N = 12
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
        N = 8
        n_steps = 50
        sim = _make_sim(N=N, n_steps=n_steps)
        traj = sim.run(n_steps=n_steps)
        assert traj.states.shape == (n_steps + 1, 2 * N)


class TestTodaLatticePhysics:
    def test_energy_conservation(self):
        """Total energy should be conserved with RK4 over 1000 steps."""
        sim = _make_sim(a=1.0, amplitude=0.3, dt=0.001)
        sim.reset()
        E0 = sim.total_energy
        assert E0 > 0  # Sanity: should have some energy

        for _ in range(1000):
            sim.step()

        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_energy_conservation_long(self):
        """Energy drift stays small over 10000 steps."""
        sim = _make_sim(a=1.0, mode=2, amplitude=0.3, dt=0.001)
        sim.reset()
        E0 = sim.total_energy

        for _ in range(10000):
            sim.step()

        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-5, f"Energy drift {rel_drift:.2e} too large"

    def test_momentum_conservation(self):
        """Total momentum should be conserved under periodic boundary conditions."""
        sim = _make_sim(a=1.0, mode=1, amplitude=0.5, dt=0.001)
        sim.reset()
        p0 = sim.total_momentum

        for _ in range(5000):
            sim.step()

        p_final = sim.total_momentum
        # Momentum should be exactly zero for mode excitation (sine wave sums to ~0)
        assert abs(p_final - p0) < 1e-10, (
            f"Momentum drift: {abs(p_final - p0):.2e}"
        )

    def test_ke_pe_sum_to_total(self):
        """Kinetic + potential energy should equal total energy."""
        sim = _make_sim(a=2.0, amplitude=0.5, dt=0.001)
        sim.reset()

        for _ in range(100):
            sim.step()

        ke = sim.kinetic_energy
        pe = sim.potential_energy
        total = sim.total_energy
        assert np.isclose(ke + pe, total, rtol=1e-12), (
            f"KE={ke:.6f} + PE={pe:.6f} != total={total:.6f}"
        )

    def test_zero_displacement_equilibrium(self):
        """All particles at rest with zero displacement should stay at rest."""
        config = SimulationConfig(
            domain=Domain.TODA_LATTICE,
            dt=0.001,
            n_steps=100,
            parameters={
                "N": 8.0, "a": 1.0, "mode": 0.0, "amplitude": 0.0,
            },
        )
        sim = TodaLattice(config)
        sim.reset()
        # Force exact zero state
        sim._state = np.zeros(2 * sim.N)

        for _ in range(100):
            sim.step()

        assert np.allclose(sim.observe(), 0.0, atol=1e-15)

    def test_small_amplitude_harmonic_limit(self):
        """At small amplitudes, Toda frequency should match harmonic prediction.

        The harmonic normal mode frequency for mode 1 with periodic BC is:
            omega_1 = 2 * sqrt(a) * sin(pi / N)
        """
        N = 8
        a = 1.0
        dt = 0.001
        n_steps = 40000
        sim = _make_sim(
            N=N, a=a, mode=1, amplitude=0.001, dt=dt, n_steps=n_steps,
        )
        sim.reset()

        # Track displacement of particle N//4 (mode 1 antinode, not node)
        particle_idx = N // 4
        positions = [sim.observe()[particle_idx]]
        for _ in range(n_steps):
            state = sim.step()
            positions.append(state[particle_idx])

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

        omega_theory = 2.0 * np.sqrt(a) * np.sin(np.pi / N)
        rel_err = abs(omega_measured - omega_theory) / omega_theory
        assert rel_err < 0.02, (
            f"Harmonic limit error {rel_err:.4%}: "
            f"measured={omega_measured:.4f}, theory={omega_theory:.4f}"
        )

    def test_trajectory_bounded(self):
        """Trajectory should remain bounded (no blow-up)."""
        sim = _make_sim(a=1.0, mode=1, amplitude=0.5, dt=0.001, n_steps=5000)
        traj = sim.run(n_steps=5000)
        assert np.all(np.isfinite(traj.states))
        assert np.max(np.abs(traj.states)) < 100.0

    def test_positions_property(self):
        """Positions property returns first N elements of state."""
        sim = _make_sim(N=8)
        sim.reset()
        pos = sim.positions
        assert pos.shape == (8,)
        assert np.allclose(pos, sim._state[:8])

    def test_momenta_property(self):
        """Momenta property returns last N elements of state."""
        sim = _make_sim(N=8)
        sim.reset()
        mom = sim.momenta
        assert mom.shape == (8,)
        assert np.allclose(mom, sim._state[8:])

    def test_harmonic_frequencies(self):
        """Harmonic frequencies should follow 2*sqrt(a)*|sin(pi*n/N)|."""
        N = 8
        a = 4.0
        sim = _make_sim(N=N, a=a)
        freqs = sim.harmonic_frequencies()
        assert freqs.shape == (N,)
        # Mode 0 should have zero frequency (translation mode)
        assert np.isclose(freqs[0], 0.0, atol=1e-14)
        # Check mode 1
        expected_1 = 2.0 * np.sqrt(a) * np.sin(np.pi * 1 / N)
        assert np.isclose(freqs[1], expected_1, rtol=1e-10)

    def test_random_init(self):
        """Mode=0 should produce random initial displacements."""
        sim = _make_sim(N=16, mode=0, amplitude=0.1)
        sim.reset()
        x = sim.positions
        # Should have varied displacements, not all zero
        assert np.std(x) > 0.01


class TestTodaLatticeRediscovery:
    def test_energy_conservation_data_shapes(self):
        from simulating_anything.rediscovery.toda_lattice import (
            generate_energy_conservation_data,
        )
        data = generate_energy_conservation_data(
            n_trajectories=3, N=8, a=1.0, dt=0.001, n_steps=1000,
        )
        assert "E_initial" in data
        assert "E_final" in data
        assert "relative_drift" in data
        assert len(data["relative_drift"]) == 3
        assert np.all(data["relative_drift"] < 1e-5)

    def test_harmonic_limit_data_shapes(self):
        from simulating_anything.rediscovery.toda_lattice import (
            generate_harmonic_limit_data,
        )
        data = generate_harmonic_limit_data(
            N=8, a=1.0, n_amplitudes=5, dt=0.001, n_steps=10000,
        )
        assert "amplitudes" in data
        assert "omega_measured" in data
        assert "omega_harmonic" in data
        assert len(data["amplitudes"]) == 5
        assert isinstance(data["omega_harmonic"], float)

    def test_soliton_data_shapes(self):
        from simulating_anything.rediscovery.toda_lattice import (
            generate_soliton_data,
        )
        data = generate_soliton_data(
            N=32, a=1.0, n_amplitudes=5, dt=0.001, n_steps=1000,
        )
        assert "amplitudes" in data
        assert "speeds" in data
        assert "sound_speed" in data
        assert len(data["amplitudes"]) == 5
        assert data["sound_speed"] == 1.0  # sqrt(1.0) = 1.0
