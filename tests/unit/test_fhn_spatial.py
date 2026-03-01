"""Tests for the 1D spatial FitzHugh-Nagumo PDE simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.fhn_spatial import FHNSpatial
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 0.7,
    b: float = 0.8,
    eps: float = 0.08,
    D_v: float = 1.0,
    N: int = 128,
    L: float = 50.0,
    dt: float = 0.05,
    n_steps: int = 100,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.FHN_SPATIAL,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a": a,
            "b": b,
            "eps": eps,
            "D_v": D_v,
            "N": float(N),
            "L": L,
        },
    )


class TestFHNSpatialCreation:
    def test_create_simulation(self):
        sim = FHNSpatial(_make_config())
        assert sim.N == 128
        assert sim.a == 0.7
        assert sim.b_param == 0.8
        assert sim.eps == 0.08
        assert sim.D_v == 1.0
        assert sim.L == 50.0

    def test_custom_parameters(self):
        sim = FHNSpatial(_make_config(a=0.5, eps=0.1, D_v=2.0))
        assert sim.a == 0.5
        assert sim.eps == 0.1
        assert sim.D_v == 2.0

    def test_grid_setup(self):
        sim = FHNSpatial(_make_config(N=64, L=20.0))
        assert sim.N == 64
        assert sim.dx == pytest.approx(20.0 / 64)
        assert len(sim.x) == 64
        assert len(sim.k) == 64

    def test_cfl_violation_raises(self):
        """Extremely large dt should raise ValueError."""
        with pytest.raises(ValueError, match="CFL"):
            FHNSpatial(_make_config(dt=100.0, D_v=5.0, N=32, L=10.0))


class TestFHNSpatialState:
    def test_initial_state_shape(self):
        sim = FHNSpatial(_make_config(N=64))
        state = sim.reset()
        assert state.shape == (128,)  # 2 * N = 2 * 64

    def test_step_returns_correct_shape(self):
        sim = FHNSpatial(_make_config(N=64))
        sim.reset()
        state = sim.step()
        assert state.shape == (128,)

    def test_observe_matches_state(self):
        sim = FHNSpatial(_make_config(N=64))
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (128,)

    def test_initial_pulse_present(self):
        """Reset should produce a localized pulse in v."""
        sim = FHNSpatial(_make_config(N=128))
        sim.reset()
        v = sim.v_field
        # The pulse should create a region significantly above the rest state
        v_rest = sim._find_rest_v()
        assert np.max(v) > v_rest + 0.5


class TestFHNSpatialDynamics:
    def test_step_advances_state(self):
        sim = FHNSpatial(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_state_bounded(self):
        """State should not blow up over moderate simulation."""
        sim = FHNSpatial(_make_config(n_steps=500, dt=0.05))
        sim.reset()
        for _ in range(500):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state))
        # FHN v should stay bounded roughly in [-3, 3]
        v = state[:sim.N]
        assert np.all(np.abs(v) < 10.0)

    def test_pulse_propagation(self):
        """The v peak should move in space as the pulse propagates."""
        sim = FHNSpatial(_make_config(N=256, L=100.0, dt=0.05, n_steps=1000))
        sim.reset()
        peak_pos_0 = np.argmax(sim.v_field) * sim.dx

        for _ in range(500):
            sim.step()
        peak_pos_1 = np.argmax(sim.v_field) * sim.dx

        # Peak should have moved (handling periodic wrap)
        delta = abs(peak_pos_1 - peak_pos_0)
        if delta > sim.L / 2:
            delta = sim.L - delta
        assert delta > 1.0  # Should have moved at least 1 unit

    def test_no_diffusion_local_dynamics(self):
        """With D_v=0, each grid point should evolve as local FHN."""
        config = _make_config(D_v=0.0, N=16, L=10.0, dt=0.02, n_steps=50)
        sim = FHNSpatial(config)
        sim.reset(seed=42)

        # Store initial v and w at point 5
        v0 = sim._v_field[5]
        w0 = sim._w_field[5]

        # Run PDE simulation
        for _ in range(50):
            sim.step()
        v_pde = sim._v_field[5]
        w_pde = sim._w_field[5]

        # Run local FHN ODE (RK4) for same initial conditions
        dt = 0.02
        v, w = v0, w0
        a, b, eps = sim.a, sim.b_param, sim.eps
        for _ in range(50):
            def f_v(vi, wi):
                return vi - vi ** 3 / 3 - wi

            def f_w(vi, wi):
                return eps * (vi + a - b * wi)

            kv1 = f_v(v, w)
            kw1 = f_w(v, w)
            kv2 = f_v(v + 0.5 * dt * kv1, w + 0.5 * dt * kw1)
            kw2 = f_w(v + 0.5 * dt * kv1, w + 0.5 * dt * kw1)
            kv3 = f_v(v + 0.5 * dt * kv2, w + 0.5 * dt * kw2)
            kw3 = f_w(v + 0.5 * dt * kv2, w + 0.5 * dt * kw2)
            kv4 = f_v(v + dt * kv3, w + dt * kw3)
            kw4 = f_w(v + dt * kv3, w + dt * kw3)
            v = v + (dt / 6) * (kv1 + 2 * kv2 + 2 * kv3 + kv4)
            w = w + (dt / 6) * (kw1 + 2 * kw2 + 2 * kw3 + kw4)

        np.testing.assert_allclose(v_pde, v, rtol=0.01)
        np.testing.assert_allclose(w_pde, w, rtol=0.01)

    def test_trajectory_reproducibility(self):
        """Same seed should produce identical trajectories."""
        sim1 = FHNSpatial(_make_config(N=64, dt=0.05, n_steps=100))
        sim1.reset(seed=42)
        for _ in range(100):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = FHNSpatial(_make_config(N=64, dt=0.05, n_steps=100))
        sim2.reset(seed=42)
        for _ in range(100):
            sim2.step()
        s2 = sim2.observe().copy()

        np.testing.assert_array_equal(s1, s2)


class TestFHNSpatialProperties:
    def test_v_field_property(self):
        sim = FHNSpatial(_make_config(N=64))
        sim.reset()
        v = sim.v_field
        assert v.shape == (64,)

    def test_w_field_property(self):
        sim = FHNSpatial(_make_config(N=64))
        sim.reset()
        w = sim.w_field
        assert w.shape == (64,)

    def test_mean_v(self):
        sim = FHNSpatial(_make_config(N=64))
        sim.reset()
        expected = float(np.mean(sim._v_field))
        assert sim.mean_v == pytest.approx(expected)

    def test_max_v(self):
        sim = FHNSpatial(_make_config(N=64))
        sim.reset()
        expected = float(np.max(sim._v_field))
        assert sim.max_v == pytest.approx(expected)

    def test_pulse_count_initial(self):
        """Initially there should be one pulse from the IC."""
        sim = FHNSpatial(_make_config(N=128))
        sim.reset()
        assert sim.pulse_count >= 1

    def test_run_trajectory(self):
        """The run() method should produce a valid TrajectoryData."""
        sim = FHNSpatial(_make_config(N=32, n_steps=20, dt=0.05))
        traj = sim.run(n_steps=20)
        assert traj.states.shape == (21, 64)  # 20 steps + initial, 2*32
        assert np.all(np.isfinite(traj.states))


class TestFHNSpatialRediscovery:
    def test_wave_speed_data_generation(self):
        from simulating_anything.rediscovery.fhn_spatial import generate_wave_speed_data

        data = generate_wave_speed_data(
            n_D=3, n_steps=200, dt=0.05, N=64, L=50.0,
        )
        assert len(data["D_v"]) >= 1
        assert len(data["wave_speed"]) >= 1
        assert np.all(data["wave_speed"] >= 0)

    def test_pulse_data_generation(self):
        from simulating_anything.rediscovery.fhn_spatial import generate_pulse_data

        data = generate_pulse_data(
            D_v=1.0, n_steps=100, dt=0.05, N=64, L=50.0,
        )
        assert "v" in data
        assert "w" in data
        assert "pulse_fwhm" in data
        assert data["pulse_fwhm"] >= 0
        assert len(data["v"]) == 64
