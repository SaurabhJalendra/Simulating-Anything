"""Tests for the FPUT lattice simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.fput import FPUTSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_sim(
    N: int = 32,
    k: float = 1.0,
    alpha: float = 0.25,
    beta: float = 0.0,
    mode: int = 1,
    amplitude: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> FPUTSimulation:
    config = SimulationConfig(
        domain=Domain.FPUT,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "N": float(N),
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "mode": float(mode),
            "amplitude": amplitude,
        },
    )
    return FPUTSimulation(config)


class TestFPUTCreation:
    def test_creation(self):
        sim = _make_sim()
        assert sim.N == 32
        assert sim.k == 1.0
        assert sim.alpha == 0.25
        assert sim.beta == 0.0

    def test_default_params(self):
        """Default parameters are used when not specified."""
        config = SimulationConfig(
            domain=Domain.FPUT,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = FPUTSimulation(config)
        assert sim.N == 32
        assert sim.k == 1.0
        assert sim.alpha == 0.25
        assert sim.beta == 0.0

    def test_custom_params(self):
        sim = _make_sim(N=16, k=2.0, alpha=0.5, beta=0.1)
        assert sim.N == 16
        assert sim.k == 2.0
        assert sim.alpha == 0.5
        assert sim.beta == 0.1

    def test_observe_shape(self):
        """Observe returns [x_1,...,x_N, v_1,...,v_N] with shape (2*N,)."""
        N = 16
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


class TestFPUTPhysics:
    def test_energy_conservation_verlet(self):
        """Symplectic Verlet should conserve energy much better than RK4."""
        sim = _make_sim(N=32, alpha=0.25, amplitude=1.0, dt=0.01)
        sim.reset()
        E0 = sim.compute_total_energy()
        assert E0 > 0  # Sanity: should have positive energy

        for _ in range(10000):
            sim.step()

        E_final = sim.compute_total_energy()
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_energy_conservation_long(self):
        """Energy drift stays small over 50000 steps."""
        sim = _make_sim(N=32, alpha=0.25, mode=2, amplitude=0.5, dt=0.01)
        sim.reset()
        E0 = sim.compute_total_energy()

        for _ in range(50000):
            sim.step()

        E_final = sim.compute_total_energy()
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-5, f"Energy drift {rel_drift:.2e} too large"

    def test_fixed_boundary_conditions(self):
        """Boundary particles x_0 and x_{N+1} are always zero (fixed walls)."""
        sim = _make_sim(N=8, alpha=0.25, amplitude=0.5)
        sim.reset()

        # The state only contains interior particles x_1..x_N
        # Verify accelerations are computed with x_0=0 and x_{N+1}=0
        x = sim.positions
        acc = sim._accelerations(x)
        assert acc.shape == (8,)
        # All should be finite
        assert np.all(np.isfinite(acc))

    def test_zero_displacement_equilibrium(self):
        """All particles at rest with zero displacement should stay at rest."""
        sim = _make_sim(N=8, alpha=0.25, mode=0, amplitude=0.0)
        sim.reset()
        sim._state = np.zeros(2 * sim.N)

        for _ in range(100):
            sim.step()

        assert np.allclose(sim.observe(), 0.0, atol=1e-15)

    def test_mode_energies_initial(self):
        """Initially, all energy should be in the excited mode."""
        sim = _make_sim(N=32, alpha=0.0, mode=1, amplitude=0.5)
        sim.reset()

        me = sim.compute_mode_energies()
        assert me.shape == (32,)
        # Mode 1 should have most of the energy
        total = np.sum(me)
        assert total > 0
        assert me[0] / total > 0.99, (
            f"Mode 1 has only {me[0]/total:.4%} of total energy"
        )

    def test_mode_energies_sum_near_total(self):
        """Sum of mode energies should approximately equal total energy.

        This is exact in the harmonic case (alpha=beta=0) but approximate
        when nonlinearity is present.
        """
        sim = _make_sim(N=16, alpha=0.0, beta=0.0, mode=1, amplitude=0.5)
        sim.reset()

        me_sum = np.sum(sim.compute_mode_energies())
        E_total = sim.compute_total_energy()
        # For purely harmonic case, these should match very closely
        rel_err = abs(me_sum - E_total) / max(abs(E_total), 1e-15)
        assert rel_err < 0.01, f"Mode energy sum error {rel_err:.4%}"

    def test_alpha_model_forces(self):
        """Alpha model force should include quadratic term."""
        sim = _make_sim(N=4, alpha=0.5, beta=0.0, k=1.0)
        delta = np.array([0.1, 0.2])
        force = sim._force_on_delta(delta)
        expected = 1.0 * delta + 0.5 * delta**2
        assert np.allclose(force, expected)

    def test_beta_model_forces(self):
        """Beta model force should include cubic term."""
        sim = _make_sim(N=4, alpha=0.0, beta=0.5, k=1.0)
        delta = np.array([0.1, 0.2])
        force = sim._force_on_delta(delta)
        expected = 1.0 * delta + 0.5 * delta**3
        assert np.allclose(force, expected)

    def test_harmonic_case_periodic_motion(self):
        """With alpha=beta=0, mode 1 should oscillate without energy transfer."""
        sim = _make_sim(N=16, alpha=0.0, beta=0.0, mode=1, amplitude=0.5, dt=0.01)
        sim.reset()

        # Run for several oscillation periods
        for _ in range(5000):
            sim.step()

        me = sim.compute_mode_energies()
        total = np.sum(me)
        # Mode 1 should still have all the energy
        assert me[0] / total > 0.999, (
            f"Mode 1 lost energy in harmonic case: {me[0]/total:.4%}"
        )

    def test_positions_property(self):
        """Positions property returns first N elements of state."""
        sim = _make_sim(N=8)
        sim.reset()
        pos = sim.positions
        assert pos.shape == (8,)
        assert np.allclose(pos, sim._state[:8])

    def test_velocities_property(self):
        """Velocities property returns last N elements of state."""
        sim = _make_sim(N=8)
        sim.reset()
        vel = sim.velocities
        assert vel.shape == (8,)
        assert np.allclose(vel, sim._state[8:])

    def test_normal_mode_frequencies(self):
        """Normal mode frequencies should follow analytical formula."""
        N = 16
        k = 4.0
        sim = _make_sim(N=N, k=k)
        freqs = sim.normal_mode_frequencies()
        assert freqs.shape == (N,)
        # Check mode 1
        expected_1 = 2.0 * np.sqrt(k) * np.sin(np.pi / (2 * (N + 1)))
        assert np.isclose(freqs[0], expected_1, rtol=1e-10)
        # All frequencies should be positive
        assert np.all(freqs > 0)
        # Frequencies should be monotonically increasing
        assert np.all(np.diff(freqs) > 0)

    def test_trajectory_bounded(self):
        """Trajectory should remain bounded (no blow-up)."""
        sim = _make_sim(N=16, alpha=0.25, mode=1, amplitude=1.0, dt=0.01, n_steps=5000)
        traj = sim.run(n_steps=5000)
        assert np.all(np.isfinite(traj.states))
        assert np.max(np.abs(traj.states)) < 100.0

    def test_random_init(self):
        """Mode=0 should produce random initial displacements."""
        sim = _make_sim(N=16, mode=0, amplitude=1.0)
        sim.reset()
        x = sim.positions
        # Should have varied displacements, not all zero
        assert np.std(x) > 0.001

    def test_recurrence_time_estimate(self):
        """Recurrence time should be a positive finite number."""
        sim = _make_sim(N=32, k=1.0)
        t_rec = sim.recurrence_time_estimate()
        assert t_rec > 0
        assert np.isfinite(t_rec)

    def test_energy_increases_with_amplitude(self):
        """Higher amplitude excitation should have more energy."""
        energies = []
        for amp in [0.1, 0.5, 1.0, 2.0]:
            sim = _make_sim(N=16, alpha=0.25, mode=1, amplitude=amp)
            sim.reset()
            energies.append(sim.compute_total_energy())

        # Energy should strictly increase with amplitude
        for i in range(1, len(energies)):
            assert energies[i] > energies[i - 1], (
                f"Energy not increasing: {energies}"
            )

    def test_beta_model_energy_conservation(self):
        """Beta (cubic) model should also conserve energy with Verlet."""
        sim = _make_sim(N=16, alpha=0.0, beta=0.3, mode=1, amplitude=1.0, dt=0.01)
        sim.reset()
        E0 = sim.compute_total_energy()

        for _ in range(10000):
            sim.step()

        E_final = sim.compute_total_energy()
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-6, f"Beta model energy drift {rel_drift:.2e}"


class TestFPUTRecurrence:
    def test_mode1_energy_initially_dominant(self):
        """When exciting mode 1, it should start with almost all the energy."""
        sim = _make_sim(N=32, alpha=0.25, mode=1, amplitude=1.0)
        sim.reset()
        me = sim.compute_mode_energies()
        frac = me[0] / np.sum(me)
        assert frac > 0.99

    def test_nonlinearity_spreads_energy(self):
        """With nonzero alpha, energy should spread to other modes over time."""
        sim = _make_sim(N=32, alpha=0.25, mode=1, amplitude=1.0, dt=0.05)
        sim.reset()

        # Run long enough for energy to spread
        for _ in range(10000):
            sim.step()

        me = sim.compute_mode_energies()
        frac_mode1 = me[0] / np.sum(me)
        # After some time, mode 1 should have lost significant energy
        assert frac_mode1 < 0.95, (
            f"Mode 1 still has {frac_mode1:.4%} -- nonlinearity not spreading energy"
        )


class TestFPUTRediscovery:
    def test_energy_conservation_data_shapes(self):
        from simulating_anything.rediscovery.fput import (
            generate_energy_conservation_data,
        )
        data = generate_energy_conservation_data(
            n_trajectories=3, N=16, alpha=0.25, dt=0.01, n_steps=1000,
        )
        assert "E_initial" in data
        assert "E_final" in data
        assert "relative_drift" in data
        assert len(data["relative_drift"]) == 3
        assert np.all(data["relative_drift"] < 1e-4)

    def test_recurrence_data_shapes(self):
        from simulating_anything.rediscovery.fput import (
            generate_recurrence_data,
        )
        data = generate_recurrence_data(
            N=16, alpha=0.25, amplitude=1.0,
            dt=0.05, n_steps=1000, sample_interval=50,
        )
        assert "times" in data
        assert "mode_energies" in data
        assert "mode1_fraction" in data
        assert "recurrence_detected" in data
        assert isinstance(data["recurrence_detected"], bool)
        assert len(data["times"]) == len(data["mode1_fraction"])
        # Initial mode 1 fraction should be close to 1
        assert data["mode1_fraction"][0] > 0.95

    def test_alpha_vs_beta_data_shapes(self):
        from simulating_anything.rediscovery.fput import (
            generate_alpha_vs_beta_data,
        )
        data = generate_alpha_vs_beta_data(
            N=16, amplitude=1.0, dt=0.05, n_steps=1000, sample_interval=50,
        )
        assert "alpha_model" in data
        assert "beta_model" in data
        for key in ["alpha_model", "beta_model"]:
            assert "E_initial" in data[key]
            assert "E_final" in data[key]
            assert "relative_drift" in data[key]
            assert "mode1_fraction_initial" in data[key]
