"""Tests for the Gierer-Meinhardt reaction-diffusion simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.gierer_meinhardt import GiererMeinhardtSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    rho_a: float = 0.1,
    rho_h: float = 0.1,
    mu_a: float = 0.02,
    mu_h: float = 0.03,
    D_a: float = 0.005,
    D_h: float = 0.2,
    rho_0: float = 0.001,
    N_grid: int = 64,
    L: float = 50.0,
    dt: float = 0.1,
    n_steps: int = 100,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.GIERER_MEINHARDT,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "rho_a": rho_a,
            "rho_h": rho_h,
            "mu_a": mu_a,
            "mu_h": mu_h,
            "D_a": D_a,
            "D_h": D_h,
            "rho_0": rho_0,
            "N_grid": float(N_grid),
            "L": L,
        },
        seed=seed,
    )


class TestGiererMeinhardtCreation:
    def test_create_simulation(self):
        sim = GiererMeinhardtSimulation(_make_config())
        assert sim.N == 64
        assert sim.rho_a == 0.1
        assert sim.rho_h == 0.1
        assert sim.mu_a == 0.02
        assert sim.mu_h == 0.03
        assert sim.D_a == 0.005
        assert sim.D_h == 0.2
        assert sim.rho_0 == 0.001

    def test_default_parameters(self):
        """Defaults match the specification."""
        config = SimulationConfig(
            domain=Domain.GIERER_MEINHARDT,
            dt=0.1,
            n_steps=10,
            parameters={},
        )
        sim = GiererMeinhardtSimulation(config)
        assert sim.rho_a == 0.1
        assert sim.rho_h == 0.1
        assert sim.mu_a == 0.02
        assert sim.mu_h == 0.03
        assert sim.D_a == 0.005
        assert sim.D_h == 0.2
        assert sim.rho_0 == 0.001
        assert sim.N == 128
        assert sim.L == 50.0

    def test_custom_parameters(self):
        sim = GiererMeinhardtSimulation(_make_config(
            rho_a=0.2, mu_a=0.05, D_h=0.5,
        ))
        assert sim.rho_a == 0.2
        assert sim.mu_a == 0.05
        assert sim.D_h == 0.5

    def test_grid_spacing(self):
        sim = GiererMeinhardtSimulation(_make_config(N_grid=64, L=32.0))
        assert sim.dx == pytest.approx(0.5)


class TestGiererMeinhardtSteadyState:
    def test_homogeneous_steady_state(self):
        """Verify a0 = (rho_a*mu_h/rho_h + rho_0)/mu_a."""
        sim = GiererMeinhardtSimulation(_make_config())
        a0, h0 = sim.homogeneous_steady_state()
        # a0 = (0.1*0.03/0.1 + 0.001) / 0.02 = (0.03 + 0.001)/0.02 = 1.55
        expected_a0 = (0.1 * 0.03 / 0.1 + 0.001) / 0.02
        assert a0 == pytest.approx(expected_a0)
        # h0 = rho_h * a0^2 / mu_h = 0.1 * 1.55^2 / 0.03
        expected_h0 = 0.1 * expected_a0 ** 2 / 0.03
        assert h0 == pytest.approx(expected_h0)

    def test_steady_state_different_params(self):
        sim = GiererMeinhardtSimulation(_make_config(
            rho_a=0.2, rho_h=0.05, mu_a=0.04, mu_h=0.01, rho_0=0.0,
        ))
        a0, h0 = sim.homogeneous_steady_state()
        expected_a0 = (0.2 * 0.01 / 0.05 + 0.0) / 0.04
        expected_h0 = 0.05 * expected_a0 ** 2 / 0.01
        assert a0 == pytest.approx(expected_a0)
        assert h0 == pytest.approx(expected_h0)

    def test_steady_state_satisfies_equations(self):
        """The steady state should satisfy both reaction equations exactly."""
        sim = GiererMeinhardtSimulation(_make_config())
        a0, h0 = sim.homogeneous_steady_state()
        # da/dt = rho_a * a0^2/h0 - mu_a*a0 + rho_0 should be 0
        da = sim.rho_a * a0 ** 2 / h0 - sim.mu_a * a0 + sim.rho_0
        # dh/dt = rho_h * a0^2 - mu_h*h0 should be 0
        dh = sim.rho_h * a0 ** 2 - sim.mu_h * h0
        assert da == pytest.approx(0.0, abs=1e-12)
        assert dh == pytest.approx(0.0, abs=1e-12)

    def test_steady_state_with_zero_basal(self):
        """With rho_0=0, a0 = rho_a*mu_h/(rho_h*mu_a)."""
        sim = GiererMeinhardtSimulation(_make_config(rho_0=0.0))
        a0, h0 = sim.homogeneous_steady_state()
        expected_a0 = sim.rho_a * sim.mu_h / (sim.rho_h * sim.mu_a)
        assert a0 == pytest.approx(expected_a0)


class TestGiererMeinhardtState:
    def test_initial_state_shape(self):
        sim = GiererMeinhardtSimulation(_make_config(N_grid=64))
        state = sim.reset()
        assert state.shape == (128,)  # 2 * 64

    def test_initial_state_shape_different_N(self):
        sim = GiererMeinhardtSimulation(_make_config(N_grid=32))
        state = sim.reset()
        assert state.shape == (64,)  # 2 * 32

    def test_step_returns_correct_shape(self):
        sim = GiererMeinhardtSimulation(_make_config(N_grid=64))
        sim.reset()
        state = sim.step()
        assert state.shape == (128,)

    def test_observe_matches_state(self):
        sim = GiererMeinhardtSimulation(_make_config(N_grid=64))
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (128,)
        # First N values are a, last N are h
        assert np.all(obs[:64] > 0)
        assert np.all(obs[64:] > 0)

    def test_initial_state_near_steady_state(self):
        """Initial a and h fields should be near the homogeneous steady state."""
        sim = GiererMeinhardtSimulation(_make_config(N_grid=32))
        sim.reset()
        a0, h0 = sim.homogeneous_steady_state()
        np.testing.assert_allclose(np.mean(sim.a_field), a0, rtol=0.05)
        np.testing.assert_allclose(np.mean(sim.h_field), h0, rtol=0.05)

    def test_step_advances_state(self):
        sim = GiererMeinhardtSimulation(_make_config(N_grid=32))
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_concentrations_positive(self):
        """Concentrations should remain strictly positive."""
        sim = GiererMeinhardtSimulation(_make_config(n_steps=500))
        sim.reset()
        for _ in range(500):
            sim.step()
        assert np.all(sim._a > 0)
        assert np.all(sim._h > 0)


class TestGiererMeinhardtProperties:
    def test_a_field_shape(self):
        sim = GiererMeinhardtSimulation(_make_config(N_grid=64))
        sim.reset()
        assert sim.a_field.shape == (64,)

    def test_h_field_shape(self):
        sim = GiererMeinhardtSimulation(_make_config(N_grid=64))
        sim.reset()
        assert sim.h_field.shape == (64,)

    def test_a_field_returns_copy(self):
        sim = GiererMeinhardtSimulation(_make_config())
        sim.reset()
        field = sim.a_field
        field[:] = 999.0
        assert np.all(sim._a != 999.0)

    def test_h_field_returns_copy(self):
        sim = GiererMeinhardtSimulation(_make_config())
        sim.reset()
        field = sim.h_field
        field[:] = 999.0
        assert np.all(sim._h != 999.0)

    def test_mean_a_positive(self):
        sim = GiererMeinhardtSimulation(_make_config())
        sim.reset()
        assert sim.mean_a > 0

    def test_mean_h_positive(self):
        sim = GiererMeinhardtSimulation(_make_config())
        sim.reset()
        assert sim.mean_h > 0

    def test_total_a_property(self):
        sim = GiererMeinhardtSimulation(_make_config())
        sim.reset()
        expected = float(np.sum(sim._a) * sim.dx)
        assert sim.total_a == pytest.approx(expected)

    def test_total_h_property(self):
        sim = GiererMeinhardtSimulation(_make_config())
        sim.reset()
        expected = float(np.sum(sim._h) * sim.dx)
        assert sim.total_h == pytest.approx(expected)

    def test_spatial_heterogeneity_a(self):
        sim = GiererMeinhardtSimulation(_make_config())
        sim.reset()
        het = sim.spatial_heterogeneity_a
        assert het >= 0.0

    def test_spatial_heterogeneity_h(self):
        sim = GiererMeinhardtSimulation(_make_config())
        sim.reset()
        het = sim.spatial_heterogeneity_h
        assert het >= 0.0


class TestGiererMeinhardtTuringAnalysis:
    def test_turing_analysis_returns_dict(self):
        sim = GiererMeinhardtSimulation(_make_config())
        analysis = sim.turing_analysis()
        assert "turing_unstable" in analysis
        assert "homogeneous_stable" in analysis
        assert "a0" in analysis
        assert "h0" in analysis

    def test_turing_unstable_with_large_diffusion_ratio(self):
        """With D_h/D_a = 40, the system should be Turing unstable."""
        sim = GiererMeinhardtSimulation(_make_config(D_a=0.005, D_h=0.2))
        analysis = sim.turing_analysis()
        assert analysis["homogeneous_stable"] is True
        assert analysis["turing_unstable"] is True

    def test_turing_stable_with_equal_diffusion(self):
        """With D_h = D_a, there should be no Turing instability."""
        sim = GiererMeinhardtSimulation(_make_config(D_a=0.1, D_h=0.1))
        analysis = sim.turing_analysis()
        assert analysis["turing_unstable"] is False

    def test_turing_wavelength_reported(self):
        """When Turing unstable, the analysis should report a wavelength."""
        sim = GiererMeinhardtSimulation(_make_config(D_a=0.005, D_h=0.2))
        analysis = sim.turing_analysis()
        if analysis["turing_unstable"]:
            assert "wavelength_most_unstable" in analysis
            assert analysis["wavelength_most_unstable"] > 0


class TestGiererMeinhardtDiffusion:
    def test_no_diffusion_converges_to_fixed_point(self):
        """Without diffusion, the system should converge to the steady state."""
        sim = GiererMeinhardtSimulation(_make_config(
            D_a=0.0, D_h=0.0, N_grid=4, dt=0.1, n_steps=100,
        ))
        sim.reset()
        for _ in range(20000):
            sim.step()
        a0, h0 = sim.homogeneous_steady_state()
        np.testing.assert_allclose(np.mean(sim.a_field), a0, rtol=0.05)
        np.testing.assert_allclose(np.mean(sim.h_field), h0, rtol=0.05)

    def test_no_diffusion_reduces_to_ode(self):
        """With D_a=D_h=0, each grid point evolves as an independent GM ODE."""
        config = _make_config(
            D_a=0.0, D_h=0.0, dt=0.05, n_steps=50, N_grid=8,
        )
        sim = GiererMeinhardtSimulation(config)
        sim.reset(seed=42)

        # Store initial values at point 0
        a_init = sim._a[0]
        h_init = sim._h[0]

        # Run PDE simulation
        for _ in range(50):
            sim.step()

        a_pde = sim._a[0]
        h_pde = sim._h[0]

        # Run pure ODE (Euler) for same initial conditions
        dt = 0.05
        a, h = a_init, h_init
        for _ in range(50):
            h_safe = max(h, 1e-15)
            da = sim.rho_a * a ** 2 / h_safe - sim.mu_a * a + sim.rho_0
            dh = sim.rho_h * a ** 2 - sim.mu_h * h
            a = max(a + dt * da, 1e-10)
            h = max(h + dt * dh, 1e-10)

        np.testing.assert_allclose(a_pde, a, rtol=0.02)
        np.testing.assert_allclose(h_pde, h, rtol=0.02)

    def test_diffusion_smooths_gradients(self):
        """With high diffusion and no reaction, gradients should decrease."""
        config = _make_config(
            rho_a=0.0, rho_h=0.0, mu_a=0.0, mu_h=0.0, rho_0=0.0,
            D_a=0.5, D_h=0.5,
            dt=0.01, n_steps=100, N_grid=64,
        )
        sim = GiererMeinhardtSimulation(config)
        sim.reset(seed=42)

        # Set a sharp step function in the activator
        sim._a[:32] = 5.0
        sim._a[32:] = 1.0
        sim._state = np.concatenate([sim._a, sim._h])

        grad_initial = np.max(np.abs(np.diff(sim._a)))

        for _ in range(100):
            sim.step()

        grad_final = np.max(np.abs(np.diff(sim._a)))
        assert grad_final < grad_initial


class TestGiererMeinhardtStability:
    def test_trajectory_bounded(self):
        """Concentrations should not blow up."""
        sim = GiererMeinhardtSimulation(_make_config(n_steps=1000))
        sim.reset()
        for _ in range(1000):
            sim.step()
            assert np.all(np.isfinite(sim._a)), "a contains non-finite values"
            assert np.all(np.isfinite(sim._h)), "h contains non-finite values"
            assert np.max(sim._a) < 1e6, f"a diverged: max={np.max(sim._a)}"
            assert np.max(sim._h) < 1e6, f"h diverged: max={np.max(sim._h)}"

    def test_mean_a_stays_near_steady_state(self):
        """Mean activator should stay near a0 for moderate runs."""
        sim = GiererMeinhardtSimulation(_make_config(n_steps=1000))
        sim.reset()
        a0, _ = sim.homogeneous_steady_state()
        for _ in range(1000):
            sim.step()
        # Mean a should be within factor of 5 of steady state
        assert sim.mean_a > a0 / 5
        assert sim.mean_a < a0 * 5


class TestGiererMeinhardtWavelength:
    def test_dominant_wavelength_with_sinusoidal(self):
        """Dominant wavelength should match an injected sinusoidal pattern."""
        sim = GiererMeinhardtSimulation(_make_config(N_grid=128, L=50.0))
        sim.reset()

        k_target = 5  # 5 full wavelengths in domain
        a0, _ = sim.homogeneous_steady_state()
        sim._a = a0 + 0.5 * np.sin(2 * np.pi * k_target * sim.x / sim.L)
        sim._state = np.concatenate([sim._a, sim._h])

        wl = sim.dominant_wavelength()
        expected_wl = sim.L / k_target
        assert wl == pytest.approx(expected_wl, rel=0.01)

    def test_pattern_energy_zero_initially(self):
        """Pattern energy should be near zero at initialization (near uniform)."""
        sim = GiererMeinhardtSimulation(_make_config(N_grid=64))
        sim.reset()
        energy = sim.compute_pattern_energy()
        a0, _ = sim.homogeneous_steady_state()
        # Energy is variance of a, which should be small relative to a0^2
        assert energy < 0.01 * a0 ** 2

    def test_compute_pattern_energy_type(self):
        sim = GiererMeinhardtSimulation(_make_config())
        sim.reset()
        energy = sim.compute_pattern_energy()
        assert isinstance(energy, float)
        assert energy >= 0.0


class TestGiererMeinhardtTrajectory:
    def test_run_trajectory(self):
        """The run() method should produce a valid TrajectoryData."""
        sim = GiererMeinhardtSimulation(_make_config(N_grid=32, n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 64)  # 50 steps + initial, 2*32=64
        assert np.all(np.isfinite(traj.states))

    def test_run_trajectory_positive(self):
        """All trajectory states should be positive."""
        sim = GiererMeinhardtSimulation(_make_config(N_grid=16, n_steps=100))
        traj = sim.run(n_steps=100)
        assert np.all(traj.states > 0)

    def test_reproducibility(self):
        """Same seed should produce identical trajectories."""
        config = _make_config(N_grid=32, n_steps=50, seed=123)
        sim1 = GiererMeinhardtSimulation(config)
        traj1 = sim1.run(n_steps=50)

        sim2 = GiererMeinhardtSimulation(config)
        traj2 = sim2.run(n_steps=50)

        np.testing.assert_array_equal(traj1.states, traj2.states)


class TestGiererMeinhardtRediscovery:
    def test_steady_state_data_generation(self):
        from simulating_anything.rediscovery.gierer_meinhardt import (
            generate_steady_state_data,
        )
        data = generate_steady_state_data(n_samples=3)
        assert len(data["rho_a"]) == 9  # 3 x 3 grid
        assert len(data["a0_theory"]) == 9
        assert len(data["h0_theory"]) == 9
        assert len(data["a0_sim"]) == 9
        assert len(data["h0_sim"]) == 9

    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.gierer_meinhardt import (
            generate_ode_data,
        )
        data = generate_ode_data(n_steps=500, dt=0.05)
        assert data["states"].shape == (500, 2)
        assert len(data["time"]) == 500
        assert data["rho_a"] == 0.1

    def test_wavelength_data_generation(self):
        from simulating_anything.rediscovery.gierer_meinhardt import (
            generate_wavelength_data,
        )
        data = generate_wavelength_data(
            n_Dh=3, n_steps=100, N_grid=32, L=20.0,
        )
        assert len(data["D_h"]) == 3
        assert len(data["wavelength"]) == 3
        assert len(data["energy"]) == 3

    def test_pattern_data_generation(self):
        from simulating_anything.rediscovery.gierer_meinhardt import (
            generate_pattern_data,
        )
        data = generate_pattern_data(
            n_Dh=2, n_Da=2, n_steps=100, N_grid=16, L=10.0,
        )
        assert len(data["D_a"]) == 4  # 2 x 2
        assert len(data["D_h"]) == 4
        assert len(data["heterogeneity_a"]) == 4
        assert len(data["patterned"]) == 4
        assert np.all(data["heterogeneity_a"] >= 0)
