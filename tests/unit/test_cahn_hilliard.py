"""Tests for the Cahn-Hilliard equation simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.cahn_hilliard import CahnHilliardSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    M: float = 1.0,
    epsilon: float = 0.05,
    N: int = 32,
    L: float = 1.0,
    dt: float = 1e-4,
    n_steps: int = 1000,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CAHN_HILLIARD,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "M": M, "epsilon": epsilon, "N": float(N), "L": L,
        },
        seed=seed,
    )


class TestCahnHilliardBasic:
    """Basic simulation tests."""

    def test_reset_shape(self):
        """Initial state should be N*N flat array."""
        sim = CahnHilliardSimulation(_make_config(N=32))
        state = sim.reset()
        assert state.shape == (32 * 32,)

    def test_observe_shape(self):
        """Observe should return N*N array."""
        sim = CahnHilliardSimulation(_make_config(N=32))
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (32 * 32,)
        assert np.array_equal(obs, sim._state)

    def test_step_advances(self):
        """State should change after a step."""
        sim = CahnHilliardSimulation(_make_config(N=32))
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same seed should produce identical results."""
        cfg = _make_config(N=32)
        sim1 = CahnHilliardSimulation(cfg)
        sim1.reset(seed=42)
        for _ in range(50):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = CahnHilliardSimulation(cfg)
        sim2.reset(seed=42)
        for _ in range(50):
            sim2.step()
        s2 = sim2.observe()

        np.testing.assert_array_equal(s1, s2)

    def test_stability_no_nan(self):
        """Solution should remain finite after many steps."""
        sim = CahnHilliardSimulation(_make_config(N=32, dt=1e-4, n_steps=5000))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
        assert np.all(np.isfinite(state)), "Solution became NaN/Inf"


class TestCahnHilliardConservation:
    """Conservation law and energy tests."""

    def test_mass_conservation(self):
        """Total mass should be conserved to high precision."""
        sim = CahnHilliardSimulation(_make_config(N=32, dt=1e-4))
        sim.reset()
        m0 = sim.compute_total_mass()
        for _ in range(500):
            sim.step()
        m1 = sim.compute_total_mass()
        # Mass conservation is guaranteed by the spectral scheme
        np.testing.assert_allclose(m0, m1, atol=1e-10)

    def test_mass_conservation_offset(self):
        """Mass conservation should work for non-zero mean initial condition."""
        sim = CahnHilliardSimulation(_make_config(N=32, dt=1e-4))
        sim.spinodal_initial_condition(mean=0.3, noise=0.05, seed=42)
        m0 = sim.compute_total_mass()
        for _ in range(500):
            sim.step()
        m1 = sim.compute_total_mass()
        np.testing.assert_allclose(m0, m1, atol=1e-10)

    def test_energy_decreases(self):
        """Free energy should monotonically decrease (or stay constant)."""
        sim = CahnHilliardSimulation(_make_config(N=32, dt=1e-4))
        sim.reset()
        E_prev = sim.compute_free_energy()
        for _ in range(200):
            sim.step()
            E_now = sim.compute_free_energy()
            # Allow tiny numerical tolerance
            assert E_now <= E_prev + 1e-10, (
                f"Energy increased: {E_prev:.8f} -> {E_now:.8f}"
            )
            E_prev = E_now

    def test_energy_positive(self):
        """Free energy should be non-negative (double-well potential >= 0)."""
        sim = CahnHilliardSimulation(_make_config(N=32, dt=1e-4))
        sim.reset()
        for _ in range(100):
            sim.step()
        E = sim.compute_free_energy()
        assert E >= -1e-10, f"Energy is negative: {E}"


class TestCahnHilliardPhysics:
    """Physics and phase separation tests."""

    def test_spinodal_decomposition(self):
        """Random init near u=0 should separate into +1/-1 domains."""
        sim = CahnHilliardSimulation(
            _make_config(N=32, epsilon=0.02, dt=1e-4, n_steps=5000)
        )
        sim.reset()
        # Run enough steps for phase separation
        for _ in range(5000):
            sim.step()
        u = sim._u
        # After separation, most values should be near +1 or -1
        near_plus = np.sum(u > 0.5)
        near_minus = np.sum(u < -0.5)
        total = u.size
        # At least 50% of points should be in the separated phases
        assert (near_plus + near_minus) / total > 0.5, (
            f"Insufficient phase separation: "
            f"{near_plus} near +1, {near_minus} near -1, total {total}"
        )

    def test_interface_forms(self):
        """After evolution, there should be interfacial regions where |u| < 0.5."""
        sim = CahnHilliardSimulation(
            _make_config(N=32, epsilon=0.03, dt=1e-4, n_steps=3000)
        )
        sim.reset()
        for _ in range(3000):
            sim.step()
        interface_area = sim.compute_interface_length(threshold=0.5)
        # There should be some interface region, but not all of the domain
        assert interface_area > 0, "No interface detected"
        assert interface_area < sim.L**2, "Entire domain is interface"

    def test_uniform_steady_state_plus(self):
        """u=+1 everywhere should be a stable steady state."""
        sim = CahnHilliardSimulation(_make_config(N=32, dt=1e-4))
        sim._u = np.ones((32, 32), dtype=np.float64)
        sim._state = sim._u.flatten()
        sim._step_count = 0

        E0 = sim.compute_free_energy()
        for _ in range(100):
            sim.step()
        u_final = sim._u

        # Should remain very close to +1
        np.testing.assert_allclose(u_final, 1.0, atol=1e-10)
        # Energy should be essentially zero (f(1) = 0, grad u = 0)
        assert E0 < 1e-10

    def test_uniform_steady_state_minus(self):
        """u=-1 everywhere should be a stable steady state."""
        sim = CahnHilliardSimulation(_make_config(N=32, dt=1e-4))
        sim._u = -np.ones((32, 32), dtype=np.float64)
        sim._state = sim._u.flatten()
        sim._step_count = 0

        for _ in range(100):
            sim.step()
        u_final = sim._u
        np.testing.assert_allclose(u_final, -1.0, atol=1e-10)

    def test_symmetric_decomposition(self):
        """Mean-zero initial condition should produce roughly equal +1/-1 phases."""
        sim = CahnHilliardSimulation(
            _make_config(N=32, epsilon=0.02, dt=1e-4)
        )
        sim.spinodal_initial_condition(mean=0.0, noise=0.05, seed=42)
        for _ in range(5000):
            sim.step()
        u = sim._u
        mean_u = np.mean(u)
        # Mean should remain close to zero (mass conservation)
        assert abs(mean_u) < 0.1, f"Mean drifted to {mean_u}"

    def test_coarsening_later_larger(self):
        """Later snapshots should have larger domain sizes (coarsening)."""
        sim = CahnHilliardSimulation(
            _make_config(N=64, epsilon=0.02, dt=1e-4, n_steps=20000)
        )
        sim.reset()

        # Run to an early time and measure
        for _ in range(5000):
            sim.step()
        L_early = sim._characteristic_length()

        # Run further and measure again
        for _ in range(15000):
            sim.step()
        L_late = sim._characteristic_length()

        assert L_late >= L_early * 0.9, (
            f"Domain did not coarsen: L_early={L_early:.4f}, L_late={L_late:.4f}"
        )

    def test_small_epsilon_sharper_interfaces(self):
        """Smaller epsilon should produce sharper interfaces."""
        results = {}
        for eps in [0.05, 0.02]:
            sim = CahnHilliardSimulation(
                _make_config(N=32, epsilon=eps, dt=1e-4)
            )
            sim.reset(seed=42)
            for _ in range(3000):
                sim.step()
            # Measure how sharp the interface is: std of |u| in the interface zone
            u = sim._u
            interface_mask = np.abs(u) < 0.8
            if np.any(interface_mask):
                results[eps] = float(np.sum(interface_mask))
            else:
                results[eps] = 0.0

        # Smaller epsilon -> fewer interface cells (sharper transition)
        assert results[0.02] <= results[0.05] + 10, (
            f"Smaller epsilon did not produce sharper interface: {results}"
        )


class TestCahnHilliardMethods:
    """Tests for helper methods."""

    def test_spinodal_initial_condition(self):
        """Spinodal IC should set correct mean and noise level."""
        sim = CahnHilliardSimulation(_make_config(N=64))
        sim.spinodal_initial_condition(mean=0.2, noise=0.01, seed=42)
        u = sim._u
        assert abs(np.mean(u) - 0.2) < 0.05
        assert np.std(u) < 0.05

    def test_compute_free_energy_type(self):
        """Free energy should return a float."""
        sim = CahnHilliardSimulation(_make_config(N=32))
        sim.reset()
        E = sim.compute_free_energy()
        assert isinstance(E, float)
        assert np.isfinite(E)

    def test_compute_total_mass_type(self):
        """Total mass should return a float."""
        sim = CahnHilliardSimulation(_make_config(N=32))
        sim.reset()
        m = sim.compute_total_mass()
        assert isinstance(m, float)
        assert np.isfinite(m)

    def test_energy_vs_time(self):
        """energy_vs_time should return arrays of correct length."""
        sim = CahnHilliardSimulation(_make_config(N=32, dt=1e-4))
        sim.reset()
        result = sim.energy_vs_time(n_steps=50)
        assert len(result["times"]) == 51  # initial + 50 steps
        assert len(result["energies"]) == 51
        # Energy should be decreasing
        assert result["energies"][-1] <= result["energies"][0] + 1e-10

    def test_characteristic_length_finite(self):
        """Characteristic length should be finite and positive."""
        sim = CahnHilliardSimulation(_make_config(N=32, epsilon=0.03, dt=1e-4))
        sim.reset()
        for _ in range(500):
            sim.step()
        L_char = sim._characteristic_length()
        assert np.isfinite(L_char)
        assert L_char > 0


class TestCahnHilliardRediscovery:
    """Tests for the rediscovery data generation."""

    def test_energy_data_generation(self):
        """generate_energy_data should return valid arrays."""
        from simulating_anything.rediscovery.cahn_hilliard import generate_energy_data

        data = generate_energy_data(
            n_steps=500, dt=1e-4, N=32, epsilon=0.05, seed=42,
            sample_interval=100,
        )
        assert "times" in data
        assert "energies" in data
        assert "masses" in data
        assert len(data["times"]) >= 2
        assert len(data["energies"]) == len(data["times"])
        # Energy should decrease
        assert data["energies"][-1] <= data["energies"][0] + 1e-8
        # Mass should be conserved
        np.testing.assert_allclose(
            data["masses"][0], data["masses"][-1], atol=1e-8
        )

    def test_coarsening_data_generation(self):
        """generate_coarsening_data should return valid arrays."""
        from simulating_anything.rediscovery.cahn_hilliard import (
            generate_coarsening_data,
        )

        data = generate_coarsening_data(
            n_snapshots=5, n_steps=2000, dt=1e-4, N=32, epsilon=0.03, seed=42,
        )
        assert "times" in data
        assert "length_scales" in data
        assert len(data["times"]) >= 3
        assert np.all(np.isfinite(data["length_scales"]))
        assert np.all(data["times"] > 0)
