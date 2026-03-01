"""Tests for the Coupled Map Lattice simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.coupled_map_lattice import (
    CoupledMapLatticeSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N: int = 100, r: float = 3.9, eps: float = 0.3, dt: float = 1.0,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.COUPLED_MAP_LATTICE,
        dt=dt,
        n_steps=100,
        parameters={"N": float(N), "r": r, "eps": eps},
    )


class TestCoupledMapLatticeSimulation:
    def test_initial_state_shape(self):
        sim = CoupledMapLatticeSimulation(_make_config(N=50))
        state = sim.reset(seed=42)
        assert state.shape == (50,)

    def test_initial_state_in_unit_interval(self):
        sim = CoupledMapLatticeSimulation(_make_config(N=200))
        state = sim.reset(seed=42)
        assert np.all(state > 0)
        assert np.all(state < 1)

    def test_step_advances_state(self):
        sim = CoupledMapLatticeSimulation(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_state_stays_finite(self):
        """State should remain finite under iteration."""
        sim = CoupledMapLatticeSimulation(_make_config(N=50, r=3.9, eps=0.3))
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
        assert np.all(np.isfinite(sim.observe()))

    def test_state_bounded_for_r_leq_4(self):
        """For r <= 4, the logistic map stays in [0,1] and coupling is convex."""
        sim = CoupledMapLatticeSimulation(_make_config(N=50, r=3.9, eps=0.3))
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
            x = sim.observe()
            assert np.all(x >= -0.01), f"min={np.min(x)}"
            assert np.all(x <= 1.01), f"max={np.max(x)}"

    def test_zero_coupling_independent(self):
        """With eps=0, each site evolves independently via the logistic map."""
        sim = CoupledMapLatticeSimulation(_make_config(N=10, r=3.9, eps=0.0))
        state0 = sim.reset(seed=42)
        sim.step()
        state1 = sim.observe()

        # Verify each site: x_i(1) = r * x_i(0) * (1 - x_i(0))
        expected = 3.9 * state0 * (1.0 - state0)
        np.testing.assert_allclose(state1, expected, atol=1e-12)

    def test_full_coupling_synchronization(self):
        """With eps=1 and r in the stable fixed-point regime (r<3),
        all sites should converge to the same value."""
        sim = CoupledMapLatticeSimulation(_make_config(N=20, r=2.5, eps=1.0))
        sim.reset(seed=42)
        for _ in range(5000):
            sim.step()
        # Spatial variance should be very small for stable r
        assert sim.compute_spatial_variance() < 1e-10

    def test_periodic_boundary(self):
        """Verify periodic boundary conditions by checking the step update."""
        sim = CoupledMapLatticeSimulation(_make_config(N=5, r=3.0, eps=0.4))
        state0 = sim.reset(seed=42)
        sim.step()
        state1 = sim.observe()

        # Manual computation for site 0 with periodic BC
        f = 3.0 * state0 * (1.0 - state0)
        x0_expected = (1.0 - 0.4) * f[0] + 0.2 * (f[4] + f[1])
        np.testing.assert_allclose(state1[0], x0_expected, atol=1e-12)

        # Site N-1 with periodic BC
        xN_expected = (1.0 - 0.4) * f[4] + 0.2 * (f[3] + f[0])
        np.testing.assert_allclose(state1[4], xN_expected, atol=1e-12)

    def test_different_lattice_sizes(self):
        """Should work for various N values."""
        for N in [10, 50, 200]:
            sim = CoupledMapLatticeSimulation(_make_config(N=N))
            state = sim.reset(seed=42)
            assert state.shape == (N,)
            sim.step()
            assert sim.observe().shape == (N,)

    def test_run_produces_trajectory(self):
        """The run() method should produce a TrajectoryData object."""
        sim = CoupledMapLatticeSimulation(_make_config(N=20))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 20)  # n_steps + 1


class TestCoupledMapLatticeMeasurements:
    def test_spatial_variance_positive(self):
        sim = CoupledMapLatticeSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        v = sim.compute_spatial_variance()
        assert v >= 0

    def test_mean_field_in_range(self):
        sim = CoupledMapLatticeSimulation(_make_config(r=3.9))
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
        mf = sim.compute_mean_field()
        assert 0 < mf < 1

    def test_sync_order_parameter_range(self):
        sim = CoupledMapLatticeSimulation(_make_config(eps=0.3))
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
        s = sim.synchronization_order_parameter()
        # Should be between -inf and 1, but practically in [0, 1] for eps > 0
        assert s <= 1.0 + 1e-6

    def test_sync_order_increases_with_coupling(self):
        """Synchronization should increase with coupling strength."""
        orders = []
        for eps in [0.0, 0.3, 0.6]:
            sim = CoupledMapLatticeSimulation(_make_config(N=50, eps=eps))
            sim.reset(seed=42)
            for _ in range(2000):
                sim.step()
            orders.append(sim.synchronization_order_parameter())
        # General trend: higher eps -> higher sync order
        assert orders[-1] > orders[0], (
            f"Sync order should increase: {orders}"
        )

    def test_coupling_sweep_shapes(self):
        sim = CoupledMapLatticeSimulation(_make_config(N=20))
        eps_values = np.linspace(0.0, 0.5, 5)
        data = sim.coupling_sweep(
            eps_values, n_transient=200, n_measure=100, seed=42,
        )
        assert len(data["eps"]) == 5
        assert len(data["variance_mean"]) == 5
        assert len(data["sync_order"]) == 5

    def test_coupling_sweep_variance_decreases(self):
        """Spatial variance should generally decrease with coupling."""
        sim = CoupledMapLatticeSimulation(_make_config(N=50))
        eps_values = np.array([0.0, 0.2, 0.5])
        data = sim.coupling_sweep(
            eps_values, n_transient=1000, n_measure=500, seed=42,
        )
        # Variance at eps=0.5 should be less than at eps=0
        assert data["variance_mean"][-1] < data["variance_mean"][0], (
            f"Variance should decrease: {data['variance_mean']}"
        )

    def test_lyapunov_positive_for_chaos(self):
        """At r=3.9, eps=0 (uncoupled chaos), Lyapunov should be positive."""
        sim = CoupledMapLatticeSimulation(_make_config(N=20, r=3.9, eps=0.0))
        lam = sim.compute_lyapunov(n_steps=2000, n_transient=500, seed=42)
        assert lam > 0, f"Lyapunov={lam} should be positive for chaotic regime"

    def test_lyapunov_finite(self):
        """Lyapunov exponent should be finite."""
        sim = CoupledMapLatticeSimulation(_make_config(N=20, eps=0.3))
        lam = sim.compute_lyapunov(n_steps=1000, n_transient=200, seed=42)
        assert np.isfinite(lam)

    def test_space_time_diagram_shape(self):
        sim = CoupledMapLatticeSimulation(_make_config(N=30))
        diagram = sim.space_time_diagram(n_steps=100, n_transient=50, seed=42)
        assert diagram.shape == (100, 30)

    def test_space_time_diagram_bounded(self):
        sim = CoupledMapLatticeSimulation(_make_config(N=30, r=3.9, eps=0.3))
        diagram = sim.space_time_diagram(n_steps=100, n_transient=200, seed=42)
        assert np.all(np.isfinite(diagram))
        assert np.all(diagram >= -0.01)
        assert np.all(diagram <= 1.01)


class TestCoupledMapLatticeRediscovery:
    def test_coupling_sweep_data(self):
        from simulating_anything.rediscovery.coupled_map_lattice import (
            generate_coupling_sweep_data,
        )
        data = generate_coupling_sweep_data(
            n_eps=5, N=20, n_transient=200, n_measure=100, seed=42,
        )
        assert len(data["eps"]) == 5
        assert len(data["variance_mean"]) == 5
        assert np.all(data["variance_mean"] >= 0)

    def test_lyapunov_sweep_data(self):
        from simulating_anything.rediscovery.coupled_map_lattice import (
            generate_lyapunov_sweep_data,
        )
        data = generate_lyapunov_sweep_data(
            n_eps=5, N=10, n_steps=500, n_transient=200, seed=42,
        )
        assert len(data["eps"]) == 5
        assert len(data["lyapunov"]) == 5
        assert np.all(np.isfinite(data["lyapunov"]))

    def test_space_time_data(self):
        from simulating_anything.rediscovery.coupled_map_lattice import (
            generate_space_time_data,
        )
        data = generate_space_time_data(
            eps_values=[0.0, 0.3], N=20, n_steps=50, n_transient=100, seed=42,
        )
        assert "eps_0.00" in data
        assert "eps_0.30" in data
        assert data["eps_0.00"].shape == (50, 20)
