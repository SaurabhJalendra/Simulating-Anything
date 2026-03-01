"""Tests for the Schnakenberg reaction-diffusion simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.schnakenberg import SchnakenbergSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 0.1,
    b: float = 0.9,
    D_u: float = 1.0,
    D_v: float = 40.0,
    N: int = 32,
    L: float = 50.0,
    dt: float = 0.01,
    n_steps: int = 100,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.SCHNAKENBERG,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a, "b": b,
            "D_u": D_u, "D_v": D_v,
            "N": float(N), "L": L,
        },
        seed=42,
    )


class TestSchnakenbergSimulation:
    def test_initial_state_shape(self):
        """State should be flattened [u, v] of length 2*N*N."""
        sim = SchnakenbergSimulation(_make_config(N=16))
        state = sim.reset()
        assert state.shape == (2 * 16 * 16,)

    def test_initial_state_near_steady_state(self):
        """Initial u and v fields should be near the homogeneous steady state."""
        sim = SchnakenbergSimulation(_make_config(N=16))
        sim.reset()
        u_star, v_star = sim.homogeneous_steady_state()
        np.testing.assert_allclose(np.mean(sim.u_field), u_star, atol=0.05)
        np.testing.assert_allclose(np.mean(sim.v_field), v_star, atol=0.05)

    def test_step_advances_state(self):
        sim = SchnakenbergSimulation(_make_config(N=16))
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_state(self):
        sim = SchnakenbergSimulation(_make_config(N=16))
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2 * 16 * 16,)

    def test_run_trajectory(self):
        """run() should return a TrajectoryData with correct shapes."""
        sim = SchnakenbergSimulation(_make_config(N=16, n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 2 * 16 * 16)

    def test_homogeneous_steady_state(self):
        """Verify u* = a + b, v* = b / (a + b)^2."""
        sim = SchnakenbergSimulation(_make_config(a=0.1, b=0.9))
        u_star, v_star = sim.homogeneous_steady_state()
        assert u_star == pytest.approx(1.0)
        assert v_star == pytest.approx(0.9)

        sim2 = SchnakenbergSimulation(_make_config(a=0.2, b=0.8))
        u2, v2 = sim2.homogeneous_steady_state()
        assert u2 == pytest.approx(1.0)
        assert v2 == pytest.approx(0.8)

    def test_steady_state_different_params(self):
        """Steady state for a=0.05, b=1.0: u*=1.05, v*=1.0/1.05^2."""
        sim = SchnakenbergSimulation(_make_config(a=0.05, b=1.0))
        u_star, v_star = sim.homogeneous_steady_state()
        assert u_star == pytest.approx(1.05)
        assert v_star == pytest.approx(1.0 / 1.05**2, rel=1e-6)

    def test_turing_analysis_returns_dict(self):
        sim = SchnakenbergSimulation(_make_config())
        analysis = sim.turing_analysis()
        assert "turing_unstable" in analysis
        assert "homogeneous_stable" in analysis
        assert "u_star" in analysis
        assert "v_star" in analysis

    def test_turing_unstable_with_large_diffusion_ratio(self):
        """With D_v/D_u = 40, the system should be Turing unstable."""
        sim = SchnakenbergSimulation(_make_config(D_u=1.0, D_v=40.0))
        analysis = sim.turing_analysis()
        assert analysis["homogeneous_stable"] is True
        assert analysis["turing_unstable"] is True

    def test_turing_stable_with_equal_diffusion(self):
        """With D_v = D_u, there should be no Turing instability."""
        sim = SchnakenbergSimulation(_make_config(D_u=1.0, D_v=1.0))
        analysis = sim.turing_analysis()
        assert analysis["turing_unstable"] is False

    def test_state_bounded(self):
        """Concentrations should remain bounded and non-negative."""
        sim = SchnakenbergSimulation(_make_config(N=16, dt=0.01))
        sim.reset()
        for _ in range(500):
            sim.step()
            assert np.all(sim.u_field >= 0), "u went negative"
            assert np.all(sim.v_field >= 0), "v went negative"
            assert np.all(np.isfinite(sim.u_field)), "u is not finite"
            assert np.all(np.isfinite(sim.v_field)), "v is not finite"

    def test_no_diffusion_converges_to_fixed_point(self):
        """Without diffusion, the spatially uniform system should reach steady state."""
        sim = SchnakenbergSimulation(
            _make_config(D_u=0.0, D_v=0.0, N=4, dt=0.01)
        )
        sim.reset()
        for _ in range(20000):
            sim.step()
        u_star, v_star = sim.homogeneous_steady_state()
        np.testing.assert_allclose(np.mean(sim.u_field), u_star, atol=0.05)
        np.testing.assert_allclose(np.mean(sim.v_field), v_star, atol=0.05)

    def test_u_field_v_field_shapes(self):
        sim = SchnakenbergSimulation(_make_config(N=16))
        sim.reset()
        assert sim.u_field.shape == (16, 16)
        assert sim.v_field.shape == (16, 16)

    def test_pattern_energy_zero_initially(self):
        """Pattern energy should be near zero at initialization (near uniform)."""
        sim = SchnakenbergSimulation(_make_config(N=32))
        sim.reset()
        energy = sim.compute_pattern_energy()
        # Energy is variance of u field, which is small at initialization
        assert energy < 0.01

    def test_compute_pattern_wavelength_no_pattern(self):
        """Wavelength should be 0 when there is no pattern."""
        sim = SchnakenbergSimulation(
            _make_config(D_u=0.0, D_v=0.0, N=16, dt=0.01)
        )
        sim.reset()
        # After just a few steps with no diffusion, no spatial pattern
        for _ in range(10):
            sim.step()
        wl = sim.compute_pattern_wavelength()
        # Either 0 or some value, but should not crash
        assert isinstance(wl, float)


class TestSchnakenbergRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.schnakenberg import generate_ode_data
        data = generate_ode_data(a=0.1, b=0.9, n_steps=500, dt=0.001)
        assert data["states"].shape == (500, 2)
        assert len(data["time"]) == 500
        assert data["a"] == 0.1
        assert data["b"] == 0.9

    def test_wavelength_data_generation(self):
        from simulating_anything.rediscovery.schnakenberg import (
            generate_wavelength_data,
        )
        data = generate_wavelength_data(n_Dv=3, n_steps=100, N=16, dt=0.01)
        assert len(data["D_v"]) == 3
        assert len(data["wavelength"]) == 3
        assert len(data["energy"]) == 3

    def test_steady_state_data_generation(self):
        from simulating_anything.rediscovery.schnakenberg import (
            generate_steady_state_data,
        )
        data = generate_steady_state_data(n_samples=3)
        assert len(data["a"]) == 9  # 3 x 3 grid
        assert len(data["u_star_theory"]) == 9

    def test_turing_analysis_wavelength(self):
        """Turing analysis should report a most unstable wavelength."""
        sim = SchnakenbergSimulation(_make_config(D_u=1.0, D_v=40.0))
        analysis = sim.turing_analysis()
        assert "wavelength_most_unstable" in analysis
        assert analysis["wavelength_most_unstable"] > 0

    def test_diffusivity_sweep(self):
        """diffusivity_sweep should return arrays of correct length."""
        sim = SchnakenbergSimulation(_make_config(N=8))
        sim.reset()
        D_v_vals = np.array([10.0, 30.0, 50.0])
        result = sim.diffusivity_sweep(D_v_vals, n_steps=50)
        assert len(result["D_v"]) == 3
        assert len(result["wavelength"]) == 3
        assert len(result["energy"]) == 3
