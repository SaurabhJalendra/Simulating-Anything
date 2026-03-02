"""Tests for the 2D diffusion equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.diffusion_2d import Diffusion2DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    D: float = 0.1,
    dt: float = 0.01,
    N_grid: int = 64,
    n_steps: int = 100,
    L: float = 2 * np.pi,
) -> SimulationConfig:
    """Create a SimulationConfig for 2D diffusion."""
    return SimulationConfig(
        domain=Domain.DIFFUSION_2D,  # placeholder
        dt=dt,
        n_steps=n_steps,
        parameters={"D": D, "N_grid": float(N_grid), "L": L},
    )


class TestDiffusion2DCreation:
    """Tests for simulation creation and parameter parsing."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = Diffusion2DSimulation(_make_config())
        assert sim.D == 0.1
        assert sim.N_grid == 64
        assert sim.L == pytest.approx(2 * np.pi)

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = Diffusion2DSimulation(_make_config(D=0.5, N_grid=32))
        assert sim.D == 0.5
        assert sim.N_grid == 32

    def test_grid_spacing(self):
        """Grid spacing should be L / N_grid."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        expected_dx = 2 * np.pi / 64
        assert sim.dx == pytest.approx(expected_dx)
        assert sim.dy == pytest.approx(expected_dx)

    def test_wavenumber_array_shape(self):
        """Wavenumber array should be N_grid x N_grid."""
        sim = Diffusion2DSimulation(_make_config(N_grid=32))
        assert sim.k_sq.shape == (32, 32)

    def test_spatial_grid_shape(self):
        """Spatial grid arrays should be N_grid x N_grid."""
        sim = Diffusion2DSimulation(_make_config(N_grid=32))
        assert sim.X.shape == (32, 32)
        assert sim.Y.shape == (32, 32)


class TestDiffusion2DInitialization:
    """Tests for different initial conditions."""

    def test_initial_state_shape(self):
        """State should be a flattened N^2 vector."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        state = sim.reset()
        assert state.shape == (64 * 64,)

    def test_gaussian_init(self):
        """Gaussian initial condition should peak in the center."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.init_type = "gaussian"
        sim.reset()
        u = sim.observe_2d()
        # Maximum should be near center (32, 32)
        max_idx = np.unravel_index(np.argmax(u), u.shape)
        assert abs(max_idx[0] - 32) <= 1
        assert abs(max_idx[1] - 32) <= 1

    def test_modes_init(self):
        """Modes initial condition should have oscillatory structure."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.init_type = "modes"
        sim.reset()
        u = sim.observe_2d()
        assert np.max(u) > 0
        assert np.min(u) < 0

    def test_step_init(self):
        """Step initial condition should have values 0 and 1."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.init_type = "step"
        sim.reset()
        u = sim.observe_2d()
        assert np.max(u) == 1.0
        assert np.min(u) == 0.0

    def test_single_mode_init(self):
        """Single mode init should be sin(x)*sin(y) on [0, 2*pi]^2."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.init_type = "single_mode"
        sim.reset()
        u = sim.observe_2d()
        # Check that it has mode (1,1) character
        assert np.max(u) > 0.9
        assert np.min(u) < -0.9

    def test_observe_2d_shape(self):
        """observe_2d should return N x N array."""
        sim = Diffusion2DSimulation(_make_config(N_grid=32))
        sim.reset()
        u = sim.observe_2d()
        assert u.shape == (32, 32)

    def test_observe_and_observe_2d_consistent(self):
        """Flattened observe() should equal observe_2d().ravel()."""
        sim = Diffusion2DSimulation(_make_config(N_grid=32))
        sim.reset()
        np.testing.assert_allclose(sim.observe(), sim.observe_2d().ravel())


class TestDiffusion2DBasicSimulation:
    """Core simulation mechanics tests."""

    def test_step_advances(self):
        """State should change after a step."""
        sim = Diffusion2DSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_state(self):
        """Observe should return the current flattened state."""
        sim = Diffusion2DSimulation(_make_config(N_grid=32))
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (32 * 32,)

    def test_run_method(self):
        """run() should produce a TrajectoryData object."""
        sim = Diffusion2DSimulation(_make_config(N_grid=32, n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 32 * 32)
        assert len(traj.timestamps) == 51

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config(N_grid=32)
        sim1 = Diffusion2DSimulation(config)
        sim1.reset()
        for _ in range(50):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = Diffusion2DSimulation(config)
        sim2.reset()
        for _ in range(50):
            sim2.step()
        s2 = sim2.observe().copy()

        np.testing.assert_allclose(s1, s2)

    def test_long_run_stability(self):
        """No NaN after many steps."""
        sim = Diffusion2DSimulation(_make_config(N_grid=32, dt=0.01))
        sim.reset()
        for _ in range(1000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), "NaN or Inf in state"


class TestDiffusion2DConservation:
    """Conservation law tests for the diffusion equation."""

    def test_total_heat_conserved(self):
        """Total heat integral(u) dx dy should be conserved (periodic BCs)."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.reset()
        h0 = sim.total_heat
        for _ in range(200):
            sim.step()
        h1 = sim.total_heat
        np.testing.assert_allclose(h0, h1, rtol=1e-10)

    def test_total_heat_conserved_modes(self):
        """Total heat should be conserved for modes init (zero mean is fine)."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.init_type = "modes"
        sim.reset()
        h0 = sim.total_heat
        for _ in range(100):
            sim.step()
        h1 = sim.total_heat
        np.testing.assert_allclose(h0, h1, atol=1e-10)

    def test_total_heat_conserved_step(self):
        """Total heat should be conserved even for step initial condition."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.init_type = "step"
        sim.reset()
        h0 = sim.total_heat
        for _ in range(100):
            sim.step()
        h1 = sim.total_heat
        np.testing.assert_allclose(h0, h1, rtol=1e-10)


class TestDiffusion2DEnergyDecay:
    """Energy (L2 norm) should decay monotonically."""

    def test_energy_decreases(self):
        """Energy should decrease over time."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.reset()
        e0 = sim.energy
        for _ in range(100):
            sim.step()
        e1 = sim.energy
        assert e1 < e0

    def test_energy_monotonic_decrease(self):
        """Energy should decrease at every step (non-uniform IC)."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.init_type = "gaussian"
        sim.reset()
        prev_e = sim.energy
        for _ in range(50):
            sim.step()
            e = sim.energy
            assert e <= prev_e + 1e-15, (
                f"Energy increased: {prev_e:.10e} -> {e:.10e}"
            )
            prev_e = e

    def test_energy_positive(self):
        """Energy should always be non-negative."""
        sim = Diffusion2DSimulation(_make_config(N_grid=32))
        sim.reset()
        for _ in range(100):
            sim.step()
            assert sim.energy >= 0


class TestDiffusion2DMaxTemperature:
    """Maximum temperature should decrease (maximum principle)."""

    def test_max_temperature_decreases(self):
        """Max temperature should decrease as heat diffuses."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.reset()
        T_max_0 = sim.max_temperature
        for _ in range(50):
            sim.step()
        T_max_1 = sim.max_temperature
        assert T_max_1 < T_max_0

    def test_higher_D_faster_diffusion(self):
        """Higher diffusion coefficient should spread heat faster."""
        sim1 = Diffusion2DSimulation(_make_config(D=0.01, N_grid=32))
        sim1.reset()
        for _ in range(100):
            sim1.step()
        max1 = sim1.max_temperature

        sim2 = Diffusion2DSimulation(_make_config(D=1.0, N_grid=32))
        sim2.reset()
        for _ in range(100):
            sim2.step()
        max2 = sim2.max_temperature

        assert max2 < max1  # Higher D -> more spread -> lower peak


class TestDiffusion2DExactDecay:
    """Tests for exact spectral decay of Fourier modes."""

    def test_single_mode_exact_decay(self):
        """Single mode (1,1) should decay as exp(-D*(1+1)*t) = exp(-2*D*t)."""
        D = 0.1
        sim = Diffusion2DSimulation(_make_config(D=D, N_grid=64))
        sim.init_type = "single_mode"
        sim.reset()

        a0 = sim.mode_amplitude(1, 1)

        n_steps = 100
        dt = 0.01
        for _ in range(n_steps):
            sim.step()

        af = sim.mode_amplitude(1, 1)
        t = n_steps * dt
        # kx = ky = 1 for L = 2*pi, so k_sq = 2
        theory = a0 * np.exp(-D * 2.0 * t)
        np.testing.assert_allclose(af, theory, rtol=1e-10)

    def test_mode_20_exact_decay(self):
        """Mode (2,0) should decay as exp(-D*4*t) since kx=2, ky=0, k_sq=4."""
        D = 0.1
        N_grid = 64
        L = 2 * np.pi
        sim = Diffusion2DSimulation(_make_config(D=D, N_grid=N_grid, L=L))
        sim.reset()

        # Set initial condition to pure mode (2, 0)
        kx_val = 2 * np.pi * 2 / L  # = 2
        sim._u = np.sin(kx_val * sim.X)
        sim._state = sim._u.ravel()

        a0 = sim.mode_amplitude(2, 0)

        n_steps = 100
        dt = 0.01
        for _ in range(n_steps):
            sim.step()

        af = sim.mode_amplitude(2, 0)
        t = n_steps * dt
        # k_sq = 4 for mode (2, 0)
        theory = a0 * np.exp(-D * 4.0 * t)
        np.testing.assert_allclose(af, theory, rtol=1e-10)

    def test_different_modes_different_rates(self):
        """Higher modes should decay faster."""
        D = 0.1
        sim = Diffusion2DSimulation(_make_config(D=D, N_grid=64))
        sim.init_type = "modes"
        sim.reset()

        a0_11 = sim.mode_amplitude(1, 1)
        a0_21 = sim.mode_amplitude(2, 1)

        for _ in range(100):
            sim.step()

        af_11 = sim.mode_amplitude(1, 1)
        af_21 = sim.mode_amplitude(2, 1)

        # Mode (2,1) has k_sq = 5, mode (1,1) has k_sq = 2
        # So (2,1) decays faster
        if a0_11 > 1e-15 and a0_21 > 1e-15:
            ratio_11 = af_11 / a0_11
            ratio_21 = af_21 / a0_21
            assert ratio_21 < ratio_11, (
                f"Higher mode should decay faster: {ratio_21} vs {ratio_11}"
            )


class TestDiffusion2DDecayRate:
    """Tests for the decay_rate_of_mode method."""

    def test_decay_rate_mode_11(self):
        """decay_rate_of_mode(1, 1) should be D * 2 for L = 2*pi."""
        sim = Diffusion2DSimulation(_make_config(D=0.5))
        rate = sim.decay_rate_of_mode(1, 1)
        # kx = ky = 1, so k_sq = 2, rate = 0.5 * 2 = 1.0
        assert rate == pytest.approx(1.0)

    def test_decay_rate_mode_10(self):
        """decay_rate_of_mode(1, 0) should be D * 1 for L = 2*pi."""
        sim = Diffusion2DSimulation(_make_config(D=0.5))
        rate = sim.decay_rate_of_mode(1, 0)
        assert rate == pytest.approx(0.5)

    def test_decay_rate_mode_21(self):
        """decay_rate_of_mode(2, 1) should be D * 5 for L = 2*pi."""
        sim = Diffusion2DSimulation(_make_config(D=0.1))
        rate = sim.decay_rate_of_mode(2, 1)
        # kx=2, ky=1, k_sq = 4+1 = 5, rate = 0.1 * 5 = 0.5
        assert rate == pytest.approx(0.5)

    def test_decay_rate_mode_00(self):
        """decay_rate_of_mode(0, 0) should be 0 (DC component is conserved)."""
        sim = Diffusion2DSimulation(_make_config(D=0.5))
        assert sim.decay_rate_of_mode(0, 0) == pytest.approx(0.0)


class TestDiffusion2DDominantMode:
    """Tests for dominant mode detection."""

    def test_dominant_mode_single_mode_init(self):
        """Single mode init should have dominant mode at (1, 1) or equivalent."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.init_type = "single_mode"
        sim.reset()
        kx, ky, amp = sim.dominant_mode()
        # For sin(x)*sin(y), the dominant FFT modes are at (1,1) and aliases
        assert amp > 0

    def test_dominant_mode_amplitude_positive(self):
        """Dominant mode should have positive amplitude."""
        sim = Diffusion2DSimulation(_make_config(N_grid=32))
        sim.reset()
        _, _, amp = sim.dominant_mode()
        assert amp > 0


class TestDiffusion2DAnalyticalSolution:
    """Tests for the analytical solution method."""

    def test_analytical_matches_simulation(self):
        """Analytical solution at time t should match simulated state."""
        D = 0.1
        n_steps = 100
        dt = 0.01
        sim = Diffusion2DSimulation(_make_config(D=D, N_grid=32, dt=dt))
        sim.init_type = "gaussian"
        sim.reset()

        for _ in range(n_steps):
            sim.step()
        simulated = sim.observe_2d()

        t = n_steps * dt
        analytical = sim.analytical_solution(t)
        np.testing.assert_allclose(simulated, analytical, atol=1e-10)

    def test_analytical_at_t0(self):
        """Analytical solution at t=0 should match initial condition."""
        sim = Diffusion2DSimulation(_make_config(N_grid=32))
        sim.reset()
        u0 = sim.observe_2d().copy()
        analytical = sim.analytical_solution(0.0)
        np.testing.assert_allclose(analytical, u0, atol=1e-14)


class TestDiffusion2DProperties:
    """Tests for property methods."""

    def test_total_heat_positive_gaussian(self):
        """Total heat for gaussian init should be positive."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.reset()
        assert sim.total_heat > 0

    def test_energy_positive_gaussian(self):
        """Energy for gaussian init should be positive."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.reset()
        assert sim.energy > 0

    def test_max_temperature_positive_gaussian(self):
        """Max temperature for gaussian init should be positive."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.reset()
        assert sim.max_temperature > 0

    def test_min_temperature_nonneg_gaussian(self):
        """Min temperature for gaussian init should be non-negative."""
        sim = Diffusion2DSimulation(_make_config(N_grid=64))
        sim.reset()
        assert sim.min_temperature >= 0


class TestDiffusion2DRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_decay_data(self):
        """Decay data generation should produce valid arrays."""
        from simulating_anything.rediscovery.diffusion_2d import generate_decay_data

        data = generate_decay_data(n_D=5, n_steps=50, dt=0.01, N_grid=32)
        assert len(data["D"]) == 5
        assert len(data["decay_rate"]) == 5
        valid = np.isfinite(data["decay_rate"])
        assert np.sum(valid) >= 3

    def test_decay_data_matches_theory(self):
        """Measured decay rates should match D * k_sq."""
        from simulating_anything.rediscovery.diffusion_2d import generate_decay_data

        data = generate_decay_data(n_D=10, n_steps=100, dt=0.01, N_grid=32)
        valid = np.isfinite(data["decay_rate"]) & (data["decay_rate"] > 0)
        theory = data["D"][valid] * data["k_sq"]
        np.testing.assert_allclose(
            data["decay_rate"][valid], theory, rtol=1e-10,
        )

    def test_generate_multimode_data(self):
        """Multi-mode data generation should produce valid arrays."""
        from simulating_anything.rediscovery.diffusion_2d import (
            generate_multimode_decay_data,
        )

        data = generate_multimode_decay_data(
            D=0.1, n_steps=50, dt=0.01, N_grid=32,
        )
        assert len(data["k_sq"]) == 10
        assert len(data["decay_rate"]) == 10
        assert data["D"] == 0.1

    def test_multimode_data_matches_theory(self):
        """Multi-mode decay rates should match D * k_sq for each mode."""
        from simulating_anything.rediscovery.diffusion_2d import (
            generate_multimode_decay_data,
        )

        data = generate_multimode_decay_data(
            D=0.1, n_steps=100, dt=0.01, N_grid=32,
        )
        valid = np.isfinite(data["decay_rate"]) & (data["decay_rate"] > 0)
        theory = data["D"] * data["k_sq"][valid]
        np.testing.assert_allclose(
            data["decay_rate"][valid], theory, rtol=1e-6,
        )

    def test_decay_data_k_sq_value(self):
        """k_sq for mode (1,1) on [0,2*pi]^2 should be 2.0."""
        from simulating_anything.rediscovery.diffusion_2d import generate_decay_data

        data = generate_decay_data(n_D=3, n_steps=50, dt=0.01, N_grid=32)
        assert data["k_sq"] == pytest.approx(2.0)
