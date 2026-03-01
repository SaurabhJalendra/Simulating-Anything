"""Tests for the Vicsek model (flocking / active matter) simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.vicsek import VicsekSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N: int = 100,
    L: float = 10.0,
    v0: float = 0.5,
    R: float = 1.0,
    eta: float = 0.3,
    dt: float = 1.0,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.VICSEK,
        dt=dt,
        n_steps=1000,
        seed=seed,
        parameters={
            "N": float(N),
            "L": L,
            "v0": v0,
            "R": R,
            "eta": eta,
        },
    )


class TestVicsekSimulation:
    def test_initial_positions_in_box(self):
        """All initial positions should be in [0, L)."""
        sim = VicsekSimulation(_make_config(N=200))
        sim.reset(seed=42)
        x, y = sim.get_positions()
        assert np.all(x >= 0) and np.all(x < 10.0)
        assert np.all(y >= 0) and np.all(y < 10.0)

    def test_initial_headings_range(self):
        """All initial headings should be in [-pi, pi)."""
        sim = VicsekSimulation(_make_config(N=200))
        sim.reset(seed=42)
        theta = sim.get_headings()
        assert np.all(theta >= -np.pi)
        assert np.all(theta < np.pi)

    def test_periodic_boundary_conditions(self):
        """Particles that cross the boundary should wrap around."""
        # Place a particle near the right edge moving right
        config = _make_config(N=1, L=5.0, v0=3.0, eta=0.0, dt=1.0)
        sim = VicsekSimulation(config)
        sim.reset(seed=42)

        # Manually place particle near right boundary heading right
        sim._pos[0] = [4.8, 2.5]
        sim._theta[0] = 0.0  # heading right

        sim.step()
        x, y = sim.get_positions()
        # After moving 3.0 from x=4.8 in a box of L=5, should wrap
        assert 0 <= x[0] < 5.0, f"x={x[0]} not in [0, 5)"

    def test_speed_constant(self):
        """All particles should move at speed v0 per step."""
        v0 = 0.5
        dt = 1.0
        config = _make_config(N=50, v0=v0, eta=0.0, dt=dt, L=100.0)
        sim = VicsekSimulation(config)
        sim.reset(seed=42)

        # Get positions before and after, but handle PBC
        pos_before = sim._pos.copy()
        sim.step()
        pos_after = sim._pos.copy()

        # Displacement with minimum image convention
        delta = pos_after - pos_before
        L = sim.L
        delta -= L * np.round(delta / L)
        dist = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
        expected = v0 * dt
        np.testing.assert_allclose(dist, expected, atol=1e-10)

    def test_low_noise_alignment(self):
        """With very low noise, particles should align (high order parameter)."""
        config = _make_config(N=50, L=5.0, eta=0.01, R=2.0)
        sim = VicsekSimulation(config)
        sim.reset(seed=42)
        for _ in range(300):
            sim.step()
        phi = sim.order_parameter()
        assert phi > 0.5, f"phi={phi} too low for eta=0.01"

    def test_high_noise_disorder(self):
        """With high noise, particles should be disordered (low order parameter)."""
        config = _make_config(N=100, L=10.0, eta=0.95, R=1.0)
        sim = VicsekSimulation(config)
        sim.reset(seed=42)
        for _ in range(200):
            sim.step()
        phi = sim.order_parameter()
        assert phi < 0.5, f"phi={phi} too high for eta=0.95"

    def test_order_parameter_range(self):
        """Order parameter should be in [0, 1]."""
        sim = VicsekSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(50):
            sim.step()
        phi = sim.order_parameter()
        assert 0.0 <= phi <= 1.0 + 1e-10

    def test_neighbor_finding_includes_self(self):
        """Each particle should be in its own neighborhood."""
        sim = VicsekSimulation(_make_config(N=10))
        sim.reset(seed=42)
        for i in range(sim.N):
            mask = sim._find_neighbors(i)
            assert mask[i], f"Particle {i} not in own neighborhood"

    def test_neighbor_finding_periodic(self):
        """Neighbors across periodic boundary should be detected."""
        config = _make_config(N=2, L=10.0, R=2.0)
        sim = VicsekSimulation(config)
        sim.reset(seed=42)

        # Place two particles near opposite edges
        sim._pos[0] = [0.5, 5.0]
        sim._pos[1] = [9.5, 5.0]  # Distance across boundary = 1.0

        mask = sim._find_neighbors(0)
        assert mask[1], "Particle across periodic boundary not found as neighbor"

    def test_heading_average_identical(self):
        """Mean of identical headings should return the same heading."""
        config = _make_config(N=5, eta=0.0, R=100.0, L=5.0)
        sim = VicsekSimulation(config)
        sim.reset(seed=42)

        # Set all headings to same value
        target = 1.5
        sim._theta[:] = target
        new_theta = sim._compute_new_headings()
        np.testing.assert_allclose(new_theta, target, atol=1e-10)

    def test_deterministic_same_seed(self):
        """Same seed should give identical trajectories."""
        config = _make_config(N=20)
        sim1 = VicsekSimulation(config)
        sim1.reset(seed=42)
        for _ in range(50):
            sim1.step()
        state1 = sim1.observe()

        sim2 = VicsekSimulation(config)
        sim2.reset(seed=42)
        for _ in range(50):
            sim2.step()
        state2 = sim2.observe()

        np.testing.assert_array_equal(state1, state2)

    def test_observe_shape(self):
        """Observe should return array of shape (3*N,)."""
        N = 50
        sim = VicsekSimulation(_make_config(N=N))
        state = sim.reset(seed=42)
        assert state.shape == (3 * N,)
        sim.step()
        assert sim.observe().shape == (3 * N,)

    def test_reset_returns_state(self):
        """Reset should return the initial state with correct shape."""
        N = 30
        sim = VicsekSimulation(_make_config(N=N))
        state = sim.reset(seed=42)
        assert isinstance(state, np.ndarray)
        assert state.shape == (3 * N,)

    def test_step_advances_state(self):
        """Step should change the state."""
        sim = VicsekSimulation(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_particle_count_preserved(self):
        """Number of particles should remain constant."""
        N = 50
        sim = VicsekSimulation(_make_config(N=N))
        sim.reset(seed=42)
        assert sim.N == N
        for _ in range(100):
            sim.step()
        assert sim.N == N
        assert sim.observe().shape == (3 * N,)

    def test_density_computation(self):
        """Density should be N / L^2."""
        N, L = 100, 10.0
        sim = VicsekSimulation(_make_config(N=N, L=L))
        sim.reset(seed=42)
        expected = N / L ** 2
        np.testing.assert_allclose(sim.compute_density(), expected)

    def test_noise_sweep(self):
        """Order parameter should decrease with increasing noise."""
        config = _make_config(N=50, L=7.0, R=1.5)
        sim = VicsekSimulation(config)
        sim.reset(seed=42)
        sweep = sim.order_parameter_sweep(
            eta_values=[0.05, 0.5, 0.95],
            n_steps=200,
            n_avg=50,
            seed=42,
        )
        assert len(sweep["phi_mean"]) == 3
        # phi at low noise should be >= phi at high noise (monotonic trend)
        assert sweep["phi_mean"][0] >= sweep["phi_mean"][-1] - 0.15, (
            f"No decreasing trend: {sweep['phi_mean']}"
        )

    def test_conservation_particle_count(self):
        """Running many steps should not create or destroy particles."""
        N = 30
        sim = VicsekSimulation(_make_config(N=N))
        sim.reset(seed=42)
        for _ in range(200):
            sim.step()
        state = sim.observe()
        assert state.shape == (3 * N,)
        x, y = sim.get_positions()
        assert len(x) == N
        assert len(y) == N

    def test_positions_in_box_after_steps(self):
        """After many steps, all positions should remain in [0, L)."""
        L = 10.0
        sim = VicsekSimulation(_make_config(N=100, L=L))
        sim.reset(seed=42)
        for _ in range(200):
            sim.step()
        x, y = sim.get_positions()
        assert np.all(x >= 0) and np.all(x < L)
        assert np.all(y >= 0) and np.all(y < L)

    def test_heading_wrap(self):
        """Headings should stay in [-pi, pi] after updates."""
        sim = VicsekSimulation(_make_config(N=50, eta=0.9))
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        theta = sim.get_headings()
        assert np.all(theta >= -np.pi - 1e-10)
        assert np.all(theta <= np.pi + 1e-10)

    def test_run_trajectory(self):
        """run() should produce a TrajectoryData with correct shapes."""
        N = 20
        n_steps = 50
        config = SimulationConfig(
            domain=Domain.VICSEK,
            dt=1.0,
            n_steps=n_steps,
            parameters={"N": float(N), "L": 10.0, "v0": 0.5, "R": 1.0, "eta": 0.3},
        )
        sim = VicsekSimulation(config)
        traj = sim.run(n_steps=n_steps)
        assert traj.states.shape == (n_steps + 1, 3 * N)

    def test_get_positions_returns_copies(self):
        """get_positions should return copies, not views."""
        sim = VicsekSimulation(_make_config(N=10))
        sim.reset(seed=42)
        x1, y1 = sim.get_positions()
        x1[0] = -999.0
        x2, _ = sim.get_positions()
        assert x2[0] != -999.0

    def test_get_headings_returns_copy(self):
        """get_headings should return a copy, not a view."""
        sim = VicsekSimulation(_make_config(N=10))
        sim.reset(seed=42)
        h1 = sim.get_headings()
        h1[0] = -999.0
        h2 = sim.get_headings()
        assert h2[0] != -999.0


class TestVicsekRediscovery:
    def test_noise_sweep_data(self):
        """Noise sweep should produce valid phi values."""
        from simulating_anything.rediscovery.vicsek import generate_noise_sweep_data
        data = generate_noise_sweep_data(
            n_eta=5, N=30, L=7.0, n_steps=100, n_avg=30, n_trials=1,
        )
        assert len(data["eta"]) == 5
        assert len(data["phi_mean"]) == 5
        assert np.all(data["phi_mean"] >= 0)
        assert np.all(data["phi_mean"] <= 1.0 + 0.01)

    def test_transition_trend(self):
        """phi should generally decrease as eta increases."""
        from simulating_anything.rediscovery.vicsek import generate_noise_sweep_data
        data = generate_noise_sweep_data(
            n_eta=5, N=50, L=7.0, R=1.5, n_steps=200, n_avg=50, n_trials=2,
        )
        # First phi (low noise) should be >= last phi (high noise)
        assert data["phi_mean"][0] >= data["phi_mean"][-1] - 0.15

    def test_density_sweep_data(self):
        """Density sweep should produce valid data."""
        from simulating_anything.rediscovery.vicsek import generate_density_sweep_data
        data = generate_density_sweep_data(
            N_values=[20, 50], L=7.0, eta=0.2,
            n_steps=100, n_avg=30, n_trials=1,
        )
        assert len(data["N"]) == 2
        assert len(data["phi"]) == 2
        assert np.all(data["phi"] >= 0)
        assert np.all(data["phi"] <= 1.0 + 0.01)
