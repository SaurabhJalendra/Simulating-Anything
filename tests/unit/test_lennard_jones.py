"""Tests for the Lennard-Jones molecular dynamics simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.lennard_jones import LennardJonesSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N: int = 16,
    L: float = 8.0,
    T: float = 1.0,
    dt: float = 0.002,
    n_steps: int = 200,
    epsilon: float = 1.0,
    sigma: float = 1.0,
    r_cut: float = 2.5,
    m: float = 1.0,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.LENNARD_JONES,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "N": float(N),
            "L": L,
            "T": T,
            "epsilon": epsilon,
            "sigma": sigma,
            "r_cut": r_cut,
            "m": m,
        },
    )


class TestLennardJones:
    """Tests for the Lennard-Jones molecular dynamics simulation."""

    # -- Instantiation and basic properties --

    def test_creation(self):
        """LennardJonesSimulation should be created with correct parameters."""
        sim = LennardJonesSimulation(_make_config())
        assert sim.N == 16
        assert sim.L == 8.0
        assert sim.T_target == 1.0
        assert sim.eps == 1.0
        assert sim.sigma == 1.0
        assert sim.m == 1.0

    def test_default_params(self):
        """Empty parameters should yield defaults."""
        config = SimulationConfig(
            domain=Domain.LENNARD_JONES,
            dt=0.002,
            n_steps=100,
            parameters={},
        )
        sim = LennardJonesSimulation(config)
        assert sim.N == 64
        assert sim.L == 10.0
        assert sim.T_target == 1.0
        assert sim.eps == 1.0
        assert sim.sigma == 1.0
        assert sim.m == 1.0

    def test_custom_params(self):
        """Custom N, T, L should be stored correctly."""
        sim = LennardJonesSimulation(_make_config(N=25, T=2.5, L=12.0))
        assert sim.N == 25
        assert sim.T_target == 2.5
        assert sim.L == 12.0

    # -- Reset shape and state --

    def test_reset_shape(self):
        """Reset should return state of shape (4*N,)."""
        N = 16
        sim = LennardJonesSimulation(_make_config(N=N))
        state = sim.reset(seed=42)
        assert state.shape == (4 * N,)

    def test_reset_no_nan(self):
        """Reset state should contain no NaN values."""
        sim = LennardJonesSimulation(_make_config())
        state = sim.reset(seed=42)
        assert not np.any(np.isnan(state))

    def test_reset_positions_in_box(self):
        """All positions after reset should be within [0, L)."""
        L = 8.0
        N = 16
        sim = LennardJonesSimulation(_make_config(N=N, L=L))
        state = sim.reset(seed=42)
        positions = state[:2 * N].reshape(N, 2)
        assert np.all(positions >= 0.0)
        assert np.all(positions < L)

    def test_reset_zero_com_velocity(self):
        """Center of mass velocity should be zero after reset."""
        N = 16
        sim = LennardJonesSimulation(_make_config(N=N))
        state = sim.reset(seed=42)
        velocities = state[2 * N:].reshape(N, 2)
        com_vel = np.mean(velocities, axis=0)
        assert np.allclose(com_vel, 0.0, atol=1e-12)

    # -- Step --

    def test_step_no_nan(self):
        """Step should not produce NaN values."""
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(10):
            state = sim.step()
        assert not np.any(np.isnan(state))

    def test_step_advances_state(self):
        """A single step should change the state."""
        sim = LennardJonesSimulation(_make_config())
        s0 = sim.reset(seed=42).copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_observe_matches_step(self):
        """Observe should return the same state as the last step."""
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        s1 = sim.step()
        obs = sim.observe()
        assert np.allclose(s1, obs)

    # -- Energy --

    def test_kinetic_energy_positive(self):
        """Kinetic energy should be positive after reset."""
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        ke = sim.kinetic_energy()
        assert ke > 0.0

    def test_total_energy_finite(self):
        """Total energy should be finite."""
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        e = sim.total_energy()
        assert np.isfinite(e)

    def test_energy_conservation_short_run(self):
        """Total energy should be approximately conserved over a short run.

        Velocity Verlet is symplectic so energy drift should be small
        for moderate step counts with small dt.
        """
        sim = LennardJonesSimulation(_make_config(N=16, dt=0.001, L=8.0))
        sim.reset(seed=42)
        e0 = sim.total_energy()

        for _ in range(50):
            sim.step()

        e1 = sim.total_energy()
        # Allow some drift but should be within a few percent for small dt
        rel_drift = abs(e1 - e0) / (abs(e0) + 1e-10)
        assert rel_drift < 0.05, f"Energy drift {rel_drift:.4f} exceeds 5%"

    def test_potential_energy_type(self):
        """Potential energy should return a float."""
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        pe = sim.potential_energy()
        assert isinstance(pe, float)

    # -- Temperature --

    def test_temperature_positive(self):
        """Instantaneous temperature should be positive."""
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        T = sim.temperature()
        assert T > 0.0

    def test_temperature_order_of_magnitude(self):
        """Temperature should be in the right ballpark of T_target.

        After initialization with Maxwell-Boltzmann velocities at T_target,
        the instantaneous temperature should be roughly near T_target.
        """
        T_target = 2.0
        sim = LennardJonesSimulation(_make_config(N=16, T=T_target))
        sim.reset(seed=42)
        T_inst = sim.temperature()
        # With only 16 particles, fluctuations are large; check within factor of 3
        assert 0.3 * T_target < T_inst < 3.0 * T_target

    # -- Pressure --

    def test_pressure_finite(self):
        """Pressure should be finite."""
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        P = sim.pressure()
        assert np.isfinite(P)

    def test_pressure_returns_float(self):
        """Pressure should return a float."""
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        P = sim.pressure()
        assert isinstance(P, float)

    # -- Radial distribution function --

    def test_rdf_shape(self):
        """Radial distribution function should return arrays of correct shape."""
        n_bins = 30
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        bins, gr = sim.radial_distribution(n_bins=n_bins)
        assert bins.shape == (n_bins,)
        assert gr.shape == (n_bins,)

    def test_rdf_bins_positive(self):
        """RDF bins should all be positive distances."""
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        bins, _ = sim.radial_distribution(n_bins=25)
        assert np.all(bins > 0.0)

    def test_rdf_nonnegative(self):
        """g(r) values should be non-negative."""
        sim = LennardJonesSimulation(_make_config())
        sim.reset(seed=42)
        _, gr = sim.radial_distribution(n_bins=25)
        assert np.all(gr >= 0.0)

    # -- Mean square displacement --

    def test_msd_shape(self):
        """MSD should have the same length as the position history."""
        N = 16
        n_frames = 20
        sim = LennardJonesSimulation(_make_config(N=N))
        sim.reset(seed=42)

        # Collect position history
        pos_history = [sim._pos.copy()]
        for _ in range(n_frames - 1):
            sim.step()
            pos_history.append(sim._pos.copy())
        pos_arr = np.array(pos_history)

        msd = sim.mean_square_displacement(pos_arr)
        assert msd.shape == (n_frames,)

    def test_msd_starts_at_zero(self):
        """MSD at t=0 should be zero (reference is the initial position)."""
        N = 16
        sim = LennardJonesSimulation(_make_config(N=N))
        sim.reset(seed=42)
        pos_history = [sim._pos.copy()]
        for _ in range(5):
            sim.step()
            pos_history.append(sim._pos.copy())
        pos_arr = np.array(pos_history)

        msd = sim.mean_square_displacement(pos_arr)
        assert np.isclose(msd[0], 0.0)

    def test_msd_nonnegative(self):
        """MSD should be non-negative at all times."""
        N = 16
        sim = LennardJonesSimulation(_make_config(N=N))
        sim.reset(seed=42)
        pos_history = [sim._pos.copy()]
        for _ in range(10):
            sim.step()
            pos_history.append(sim._pos.copy())
        pos_arr = np.array(pos_history)

        msd = sim.mean_square_displacement(pos_arr)
        assert np.all(msd >= 0.0)

    # -- Determinism --

    def test_determinism(self):
        """Two runs with the same seed should produce identical states."""
        sim = LennardJonesSimulation(_make_config())

        s1 = sim.reset(seed=123)
        for _ in range(10):
            s1 = sim.step()

        s2 = sim.reset(seed=123)
        for _ in range(10):
            s2 = sim.step()

        assert np.allclose(s1, s2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different initial states."""
        sim = LennardJonesSimulation(_make_config())
        s1 = sim.reset(seed=1)
        s2 = sim.reset(seed=2)
        assert not np.allclose(s1, s2)

    # -- Trajectory --

    def test_trajectory_shape(self):
        """run() should return trajectory with correct shape."""
        N = 16
        n_steps = 50
        sim = LennardJonesSimulation(_make_config(N=N, n_steps=n_steps))
        traj = sim.run(n_steps=n_steps)
        assert traj.states.shape == (n_steps + 1, 4 * N)

    def test_trajectory_no_nan(self):
        """Trajectory should not contain any NaN values."""
        N = 16
        n_steps = 30
        sim = LennardJonesSimulation(_make_config(N=N, n_steps=n_steps))
        traj = sim.run(n_steps=n_steps)
        assert not np.any(np.isnan(traj.states))

    # -- Cutoff shift --

    def test_cutoff_shift_computed(self):
        """The potential cutoff shift should be precomputed."""
        sim = LennardJonesSimulation(_make_config())
        # v_shift = 4*eps * ((sigma/r_cut)^12 - (sigma/r_cut)^6)
        sr6 = (sim.sigma / sim.r_cut) ** 6
        expected = 4.0 * sim.eps * (sr6 ** 2 - sr6)
        assert np.isclose(sim._v_shift, expected)

    # -- Positions stay in box --

    def test_positions_in_box_after_steps(self):
        """After stepping, positions should remain within [0, L) due to PBC."""
        N = 16
        L = 8.0
        sim = LennardJonesSimulation(_make_config(N=N, L=L))
        sim.reset(seed=42)
        for _ in range(20):
            sim.step()

        positions = sim._pos
        assert np.all(positions >= 0.0)
        assert np.all(positions < L)
