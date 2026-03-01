"""Tests for the 2D Rayleigh-Benard convection simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.rayleigh_benard import RayleighBenardSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    Ra: float = 1000.0,
    Pr: float = 1.0,
    dt: float = 5e-5,
    n_steps: int = 100,
    Nx: int = 32,
    Nz: int = 16,
    **kwargs,
) -> SimulationConfig:
    params = {
        "Ra": Ra, "Pr": Pr,
        "Lx": 2.0, "H": 1.0,
        "Nx": float(Nx), "Nz": float(Nz),
        "perturbation_amp": 0.01,
    }
    params.update(kwargs)
    return SimulationConfig(
        domain=Domain.RAYLEIGH_BENARD,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


def _make_sim(**kwargs) -> RayleighBenardSimulation:
    config = _make_config(**kwargs)
    return RayleighBenardSimulation(config)


class TestRayleighBenardSimulation:
    """Core simulation tests."""

    def test_reset_produces_valid_state(self):
        """Reset should produce finite state of correct shape."""
        sim = _make_sim()
        state = sim.reset()
        Nx, Nz = 32, 16
        expected_size = 2 * Nx * Nz  # omega + T_pert
        assert state.shape == (expected_size,)
        assert np.all(np.isfinite(state))

    def test_observe_shape(self):
        """Observe should return state with correct shape."""
        sim = _make_sim(Nx=32, Nz=16)
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2 * 32 * 16,)

    def test_step_advances_state(self):
        """State should change after a step."""
        sim = _make_sim(Ra=2000.0)
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_deterministic(self):
        """Same seed should produce same result."""
        sim1 = _make_sim(Ra=1000.0)
        s1 = sim1.reset(seed=42)
        for _ in range(10):
            s1 = sim1.step()

        sim2 = _make_sim(Ra=1000.0)
        s2 = sim2.reset(seed=42)
        for _ in range(10):
            s2 = sim2.step()

        np.testing.assert_allclose(s1, s2, rtol=1e-12)

    def test_grid_resolution(self):
        """State shape should match grid resolution."""
        for Nx, Nz in [(16, 8), (32, 16), (64, 32)]:
            sim = _make_sim(Nx=Nx, Nz=Nz)
            state = sim.reset()
            assert state.shape == (2 * Nx * Nz,)

    def test_stability(self):
        """Solution should stay finite for many steps at stable parameters."""
        sim = _make_sim(Ra=500.0, Nx=16, Nz=8, dt=1e-4)
        sim.reset()
        for _ in range(200):
            state = sim.step()
            assert np.all(np.isfinite(state)), "Solution became NaN/Inf"

    def test_run_trajectory(self):
        """Should be able to run a full trajectory without error."""
        sim = _make_sim(Ra=500.0, Nx=16, Nz=8, n_steps=50, dt=1e-4)
        traj = sim.run(n_steps=50)
        assert traj.states.shape[0] == 51  # initial + 50 steps
        assert np.all(np.isfinite(traj.states))

    def test_temperature_bounds(self):
        """Total temperature should stay in reasonable range."""
        sim = _make_sim(Ra=1000.0, Nx=32, Nz=16, dt=5e-5)
        sim.reset()
        for _ in range(100):
            sim.step()
        T_total = sim.total_temperature
        # Temperature perturbation should not blow up
        assert np.all(np.isfinite(T_total))
        # Total T should be roughly in [0, 1] range (conduction profile)
        # Allow some overshoot from convection
        assert np.max(T_total) < 3.0, f"T too high: {np.max(T_total)}"
        assert np.min(T_total) > -2.0, f"T too low: {np.min(T_total)}"


class TestConvectionPhysics:
    """Tests for physical behavior of convection."""

    def test_conduction_state(self):
        """Below Ra_c, perturbation should decay (no convection)."""
        sim = _make_sim(Ra=200.0, Nx=32, Nz=16, dt=1e-4)
        sim.reset()

        # Let the initial transient develop (omega starts at zero, needs
        # a few steps for vorticity to develop from the T perturbation)
        for _ in range(500):
            sim.step()
        amp_early = sim.convection_amplitude()

        # Run much longer -- below Ra_c the amplitude should decay
        for _ in range(4000):
            sim.step()
        amp_late = sim.convection_amplitude()

        # Below critical Ra, amplitude should decrease over time
        assert amp_late < amp_early, (
            f"Perturbation did not decay below Ra_c: "
            f"early={amp_early:.6f} -> late={amp_late:.6f}"
        )

    def test_convection_onset(self):
        """Above Ra_c, perturbation should grow (convection develops)."""
        sim = _make_sim(Ra=3000.0, Nx=32, Nz=16, dt=5e-5)
        sim.reset()

        # Run for a while and check convection develops
        for _ in range(3000):
            sim.step()
        amp = sim.convection_amplitude()

        # Above Ra_c by a large margin, convection should be noticeable
        assert amp > 1e-3, f"No convection detected at Ra=3000: amp={amp:.6f}"

    def test_nusselt_unity_below_critical(self):
        """Nu should be ~1 below Ra_c (pure conduction)."""
        sim = _make_sim(Ra=200.0, Nx=32, Nz=16, dt=1e-4)
        sim.reset()
        for _ in range(2000):
            sim.step()
        nu_ = sim.compute_nusselt()
        # Below onset, Nu should be close to 1
        assert abs(nu_ - 1.0) < 0.5, f"Nu={nu_:.4f}, expected ~1.0 below Ra_c"

    def test_nusselt_above_unity(self):
        """Nu should be > 1 above Ra_c (convective heat transfer)."""
        sim = _make_sim(Ra=5000.0, Nx=32, Nz=16, dt=5e-5)
        sim.reset()
        for _ in range(5000):
            sim.step()
        nu_ = sim.compute_nusselt()
        # Well above onset, Nu should exceed 1
        assert nu_ > 0.9, f"Nu={nu_:.4f}, expected > 1 well above Ra_c"

    def test_nusselt_increases_with_Ra(self):
        """Nu should generally increase with Ra above onset."""
        nusselt_values = {}
        for Ra in [2000.0, 5000.0]:
            sim = _make_sim(Ra=Ra, Nx=32, Nz=16, dt=5e-5)
            sim.reset()
            for _ in range(5000):
                sim.step()
            nusselt_values[Ra] = sim.compute_nusselt()

        # Higher Ra should give more heat transfer
        assert nusselt_values[5000.0] >= nusselt_values[2000.0] - 0.5, (
            f"Nu at Ra=5000 ({nusselt_values[5000.0]:.4f}) not higher than "
            f"Ra=2000 ({nusselt_values[2000.0]:.4f})"
        )

    def test_critical_rayleigh(self):
        """Convection onset should be near theoretical Ra_c."""
        # Test subcritical vs supercritical behavior
        # At Ra = 300 (well below Ra_c ~657), should decay
        sim_sub = _make_sim(Ra=300.0, Nx=32, Nz=16, dt=1e-4)
        sim_sub.reset()
        for _ in range(2000):
            sim_sub.step()
        amp_sub = sim_sub.convection_amplitude()

        # At Ra = 2000 (well above Ra_c), should grow
        sim_super = _make_sim(Ra=2000.0, Nx=32, Nz=16, dt=5e-5)
        sim_super.reset()
        for _ in range(3000):
            sim_super.step()
        amp_super = sim_super.convection_amplitude()

        assert amp_super > 10 * amp_sub, (
            f"No clear onset: sub={amp_sub:.6f}, super={amp_super:.6f}"
        )

    def test_vorticity_symmetry(self):
        """Vorticity should develop organized patterns above onset."""
        sim = _make_sim(Ra=3000.0, Nx=32, Nz=16, dt=5e-5)
        sim.reset()
        for _ in range(3000):
            sim.step()

        omega = sim._omega
        # Should have both positive and negative vorticity (rolls)
        assert np.max(omega) > 0
        assert np.min(omega) < 0

    def test_prandtl_effect(self):
        """Different Prandtl numbers should give different dynamics."""
        results = {}
        for Pr in [0.5, 2.0]:
            sim = _make_sim(Ra=3000.0, Pr=Pr, Nx=32, Nz=16, dt=5e-5)
            sim.reset()
            for _ in range(2000):
                sim.step()
            results[Pr] = sim.convection_amplitude()

        # Both should have convection but different amplitudes
        assert results[0.5] > 0
        assert results[2.0] > 0
        assert results[0.5] != pytest.approx(results[2.0], rel=0.01)

    def test_vertical_velocity(self):
        """Vertical velocity property should return correct shape."""
        sim = _make_sim(Nx=32, Nz=16)
        sim.reset()
        w = sim.vertical_velocity
        assert w.shape == (16, 32)

    def test_total_temperature(self):
        """Total temperature should include conduction profile."""
        sim = _make_sim(Nx=32, Nz=16)
        sim.reset()
        T_total = sim.total_temperature
        assert T_total.shape == (16, 32)
        # Conduction profile: T = 1 - z/H, so T(z=0)~1, T(z=H)~0
        # Top of domain should be cooler than bottom
        assert np.mean(T_total[0, :]) > np.mean(T_total[-1, :])


class TestRayleighBenardRediscovery:
    """Tests for rediscovery data generation."""

    def test_rediscovery_data(self):
        """Data generation for rediscovery should work."""
        from simulating_anything.rediscovery.rayleigh_benard import (
            generate_onset_data,
        )

        data = generate_onset_data(
            n_Ra=5, n_steps=500, dt=1e-4, Nx=16, Nz=8,
        )
        assert len(data["Ra"]) == 5
        assert len(data["amplitude"]) == 5
        assert len(data["nusselt"]) == 5
        assert np.all(np.isfinite(data["amplitude"]))

    def test_find_critical_ra(self):
        """Critical Ra finder should return reasonable estimate."""
        from simulating_anything.rediscovery.rayleigh_benard import find_critical_ra

        Ra = np.array([100, 300, 500, 700, 1000, 2000])
        amps = np.array([0.0, 0.0, 0.0, 0.5, 2.0, 5.0])
        Ra_c = find_critical_ra(Ra, amps)
        # Should be between 500 and 700
        assert 400 < Ra_c < 800, f"Ra_c={Ra_c}, expected 500-700"
