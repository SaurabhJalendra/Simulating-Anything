"""Tests for the Sine-Gordon equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sine_gordon import SineGordonSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    c: float = 1.0, N: int = 128, L: float = 40.0, dt: float = 0.01,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.SINE_GORDON,
        dt=dt,
        n_steps=100,
        parameters={"c": c, "N": float(N), "L": L},
    )


class TestSineGordonSimulation:
    """Core simulation tests."""

    def test_initial_state_shape(self):
        sim = SineGordonSimulation(_make_config(N=128))
        state = sim.reset()
        assert state.shape == (256,), "State should be [u, u_t] = 2*N"

    def test_observe_shape(self):
        sim = SineGordonSimulation(_make_config(N=64))
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (128,)

    def test_step_advances(self):
        sim = SineGordonSimulation(_make_config())
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1), "State should change after step"

    def test_deterministic(self):
        """Two runs with same config should give identical results."""
        config = _make_config()
        sim1 = SineGordonSimulation(config)
        sim1.reset()
        for _ in range(50):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = SineGordonSimulation(config)
        sim2.reset()
        for _ in range(50):
            sim2.step()
        s2 = sim2.observe()

        np.testing.assert_array_equal(s1, s2)

    def test_reset_returns_to_initial(self):
        sim = SineGordonSimulation(_make_config())
        s0 = sim.reset().copy()
        for _ in range(20):
            sim.step()
        s_reset = sim.reset()
        np.testing.assert_array_equal(s0, s_reset)


class TestKinkProfile:
    """Tests for kink soliton physics."""

    def test_kink_profile(self):
        """Verify kink shape matches 4*arctan(exp((x-x0)))."""
        config = _make_config(N=512, L=40.0)
        sim = SineGordonSimulation(config)
        sim.init_type = "kink"
        sim.reset()

        u = sim._state[:sim.N]
        x0 = sim.L / 2
        theory = 4.0 * np.arctan(np.exp(sim.x - x0))
        np.testing.assert_allclose(u, theory, atol=1e-10)

    def test_antikink_profile(self):
        """Antikink should be negative of kink."""
        config = _make_config(N=512, L=40.0)
        sim = SineGordonSimulation(config)

        sim.init_type = "kink"
        sim.reset()
        u_kink = sim._state[:sim.N].copy()

        sim.init_type = "antikink"
        sim.reset()
        u_antikink = sim._state[:sim.N]

        np.testing.assert_allclose(u_antikink, -u_kink, atol=1e-10)

    def test_kink_approaches_2pi(self):
        """Kink should go from ~0 to ~2*pi across the domain."""
        config = _make_config(N=512, L=80.0)
        sim = SineGordonSimulation(config)
        sim.init_type = "kink"
        sim.reset()
        u = sim._state[:sim.N]
        assert u[0] < 0.1, "Left boundary should be near 0"
        assert u[-1] > 2 * np.pi - 0.1, "Right boundary should be near 2*pi"


class TestEnergyConservation:
    """Energy conservation tests for symplectic integrator."""

    def test_energy_conservation(self):
        """Energy drift should be very small over 1000 steps."""
        config = _make_config(N=128, L=40.0, dt=0.01)
        sim = SineGordonSimulation(config)
        sim.init_type = "kink"
        sim.reset()

        E0 = sim.compute_energy()
        for _ in range(1000):
            sim.step()
        E1 = sim.compute_energy()

        drift = abs(E1 - E0) / E0
        assert drift < 1e-4, f"Energy drift {drift:.2e} exceeds threshold"

    def test_energy_positive(self):
        """Energy should always be non-negative."""
        config = _make_config(N=128, L=40.0, dt=0.01)
        sim = SineGordonSimulation(config)
        sim.init_type = "kink"
        sim.reset()

        for _ in range(200):
            sim.step()
            E = sim.compute_energy()
            assert E >= 0, f"Energy became negative: {E}"

    def test_kink_rest_energy(self):
        """Rest kink energy should be close to 8*c."""
        c = 1.0
        config = _make_config(c=c, N=512, L=80.0)
        sim = SineGordonSimulation(config)
        sim.init_type = "kink"
        sim.reset()

        E = sim.compute_energy()
        E_theory = 8.0 * c
        rel_err = abs(E - E_theory) / E_theory
        assert rel_err < 0.05, (
            f"Rest energy {E:.4f} vs theory {E_theory:.4f}, error {rel_err:.2%}"
        )


class TestTopologicalCharge:
    """Tests for topological charge computation."""

    def test_kink_charge(self):
        """Kink should have topological charge Q ~ 1."""
        config = _make_config(N=512, L=80.0)
        sim = SineGordonSimulation(config)
        sim.init_type = "kink"
        sim.reset()
        Q = sim.compute_topological_charge()
        assert abs(Q - 1.0) < 0.1, f"Kink charge {Q} should be ~1"

    def test_antikink_charge(self):
        """Antikink should have topological charge Q ~ -1."""
        config = _make_config(N=512, L=80.0)
        sim = SineGordonSimulation(config)
        sim.init_type = "antikink"
        sim.reset()
        Q = sim.compute_topological_charge()
        assert abs(Q + 1.0) < 0.1, f"Antikink charge {Q} should be ~-1"

    def test_vacuum_charge(self):
        """Vacuum state should have charge Q = 0."""
        config = _make_config(N=128)
        sim = SineGordonSimulation(config)
        sim.init_type = "vacuum"
        sim.reset()
        Q = sim.compute_topological_charge()
        assert abs(Q) < 0.01, f"Vacuum charge {Q} should be ~0"


class TestBreather:
    """Tests for breather solutions."""

    def test_breather_stays_localized(self):
        """Breather should remain localized (not disperse) over many steps."""
        config = _make_config(N=256, L=40.0, dt=0.005)
        sim = SineGordonSimulation(config)
        sim.init_type = "breather"
        sim.reset()

        u0 = sim._state[:sim.N].copy()
        peak0 = np.max(np.abs(u0))

        for _ in range(2000):
            sim.step()

        u_final = sim._state[:sim.N]
        peak_final = np.max(np.abs(u_final))
        # Breather amplitude should not drop drastically
        assert peak_final > 0.3 * peak0, (
            f"Breather dispersed: initial peak {peak0:.4f}, "
            f"final peak {peak_final:.4f}"
        )


class TestLorentzContraction:
    """Tests for relativistic Lorentz contraction of kink width."""

    def test_lorentz_contraction(self):
        """Width should decrease with velocity as sqrt(1 - v^2/c^2)."""
        config = _make_config(N=512, L=80.0)
        sim = SineGordonSimulation(config)
        sim.reset()

        velocities = [0.0, 0.3, 0.6, 0.8]
        data = sim.measure_lorentz_contraction(velocities)

        # Width at v=0 should be largest
        assert data["measured_widths"][0] > data["measured_widths"][-1], (
            "Width should decrease with velocity"
        )

        # Check agreement with theory
        valid = data["theoretical_widths"] > 1e-6
        rel_err = np.abs(
            data["measured_widths"][valid] - data["theoretical_widths"][valid]
        ) / data["theoretical_widths"][valid]
        assert np.mean(rel_err) < 0.05, (
            f"Lorentz contraction error {np.mean(rel_err):.2%} exceeds 5%"
        )


class TestStability:
    """Numerical stability tests."""

    def test_no_nan_inf(self):
        """Solution should stay finite over 5000 steps."""
        config = _make_config(N=128, L=40.0, dt=0.01)
        sim = SineGordonSimulation(config)
        sim.init_type = "kink"
        sim.reset()

        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "Solution became NaN/Inf"

    def test_vacuum_stable(self):
        """Vacuum state (u=0) should remain stable."""
        config = _make_config(N=128, L=20.0, dt=0.01)
        sim = SineGordonSimulation(config)
        sim.init_type = "vacuum"
        sim.reset()

        for _ in range(500):
            sim.step()

        state = sim.observe()
        assert np.max(np.abs(state)) < 1e-12, (
            f"Vacuum drifted: max |u| = {np.max(np.abs(state))}"
        )

    def test_periodic_boundary(self):
        """Finite differences use periodic wrapping (roll)."""
        config = _make_config(N=64, L=20.0, dt=0.005)
        sim = SineGordonSimulation(config)
        sim.init_type = "vacuum"
        # Set a single-point perturbation near the boundary
        sim.reset()
        u = sim._state[:sim.N].copy()
        u[0] = 0.1
        ut = sim._state[sim.N:]
        sim._state = np.concatenate([u, ut])

        # The perturbation should couple to the last grid point via periodicity
        for _ in range(10):
            sim.step()
        state = sim.observe()
        u_final = state[:sim.N]
        # Last point should have picked up some perturbation
        assert abs(u_final[-1]) > 1e-10, "Periodic BC not working"


class TestRediscovery:
    """Tests for the rediscovery data generation."""

    def test_lorentz_data_generation(self):
        from simulating_anything.rediscovery.sine_gordon import (
            generate_lorentz_contraction_data,
        )
        data = generate_lorentz_contraction_data(n_velocities=5, N=128, L=40.0)
        assert len(data["velocities"]) == 5
        assert len(data["measured_widths"]) == 5
        assert len(data["theoretical_widths"]) == 5
        assert data["rest_width"] > 0

    def test_energy_conservation_data(self):
        from simulating_anything.rediscovery.sine_gordon import (
            generate_energy_conservation_data,
        )
        data = generate_energy_conservation_data(
            n_steps=100, dt=0.01, N=64, L=20.0,
        )
        assert len(data["energies"]) >= 2
        assert len(data["charges"]) >= 2
        # Energy should be roughly conserved
        E = data["energies"]
        drift = abs(E[-1] - E[0]) / E[0]
        assert drift < 0.01

    def test_velocity_sweep_data(self):
        from simulating_anything.rediscovery.sine_gordon import (
            generate_velocity_sweep_data,
        )
        data = generate_velocity_sweep_data(n_velocities=5, N=128, L=40.0)
        assert len(data["velocities"]) == 5
        assert len(data["measured_energies"]) == 5
        assert len(data["theoretical_energies"]) == 5


class TestAnalyticalFormulas:
    """Tests for analytical helper methods."""

    def test_analytical_kink_energy_at_rest(self):
        E = SineGordonSimulation.analytical_kink_energy(c=1.0, v=0.0)
        assert E == pytest.approx(8.0, rel=1e-10)

    def test_analytical_kink_energy_increases_with_v(self):
        E_rest = SineGordonSimulation.analytical_kink_energy(c=1.0, v=0.0)
        E_fast = SineGordonSimulation.analytical_kink_energy(c=1.0, v=0.5)
        assert E_fast > E_rest, "Energy should increase with velocity"
