"""Tests for the 1D advection equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.advection_1d import Advection1DSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    c: float = 1.0,
    N: int = 256,
    L: float = 10.0,
    sigma: float = 0.5,
    dt: float | None = None,
    n_steps: int = 500,
) -> SimulationConfig:
    dx = L / N
    if dt is None:
        dt = 0.8 * dx / abs(c) if abs(c) > 0 else 0.01
    return SimulationConfig(
        domain=Domain.ADVECTION_1D,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "c": c,
            "N": float(N),
            "L": L,
            "sigma": sigma,
        },
    )


def _make_sim(**kwargs) -> Advection1DSimulation:
    return Advection1DSimulation(_make_config(**kwargs))


# ---------------------------------------------------------------------------
# Creation and initialization
# ---------------------------------------------------------------------------

class TestAdvectionCreation:
    def test_creation_defaults(self):
        sim = _make_sim()
        assert sim.c == 1.0
        assert sim.N == 256
        assert sim.L == 10.0
        assert sim.sigma == 0.5

    def test_custom_params(self):
        sim = _make_sim(c=2.5, N=128, L=20.0, sigma=1.0)
        assert sim.c == 2.5
        assert sim.N == 128
        assert sim.L == 20.0
        assert sim.sigma == 1.0

    def test_default_params_empty(self):
        """Default parameters are used when not specified."""
        config = SimulationConfig(
            domain=Domain.ADVECTION_1D,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = Advection1DSimulation(config)
        assert sim.c == 1.0
        assert sim.N == 256
        assert sim.L == 10.0

    def test_grid_spacing(self):
        sim = _make_sim(N=100, L=10.0)
        assert sim.dx == pytest.approx(0.1)

    def test_grid_points(self):
        sim = _make_sim(N=64, L=8.0)
        assert len(sim.x) == 64
        assert sim.x[0] == 0.0
        assert sim.x[-1] < 8.0  # endpoint=False


class TestAdvectionInitialState:
    def test_state_shape(self):
        sim = _make_sim(N=256)
        state = sim.reset()
        assert state.shape == (256,)

    def test_state_shape_custom(self):
        sim = _make_sim(N=128)
        state = sim.reset()
        assert state.shape == (128,)

    def test_gaussian_pulse_peak(self):
        """Gaussian pulse should peak at L/4."""
        sim = _make_sim(L=10.0, N=256)
        state = sim.reset()
        peak_idx = np.argmax(state)
        peak_x = sim.x[peak_idx]
        assert peak_x == pytest.approx(10.0 / 4, abs=sim.dx)

    def test_initial_values_positive(self):
        sim = _make_sim()
        state = sim.reset()
        assert np.all(state >= 0)

    def test_peak_value_near_one(self):
        """Gaussian peak value should be close to 1."""
        sim = _make_sim()
        sim.reset()
        assert sim.peak_value == pytest.approx(1.0, abs=0.01)

    def test_dtype_float64(self):
        sim = _make_sim()
        state = sim.reset()
        assert state.dtype == np.float64


# ---------------------------------------------------------------------------
# Stepping and time evolution
# ---------------------------------------------------------------------------

class TestAdvectionStepping:
    def test_step_changes_state(self):
        sim = _make_sim()
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = _make_sim()
        state = sim.reset()
        observed = sim.observe()
        np.testing.assert_array_equal(state, observed)

    def test_multiple_steps_no_nan(self):
        sim = _make_sim(n_steps=500)
        sim.reset()
        for _ in range(500):
            state = sim.step()
        assert not np.any(np.isnan(state))

    def test_multiple_steps_no_inf(self):
        sim = _make_sim(n_steps=500)
        sim.reset()
        for _ in range(500):
            state = sim.step()
        assert not np.any(np.isinf(state))

    def test_step_count_increments(self):
        sim = _make_sim()
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2


# ---------------------------------------------------------------------------
# Physics: pulse translation
# ---------------------------------------------------------------------------

class TestAdvectionTranslation:
    def test_pulse_translates_right(self):
        """With c > 0, pulse should move to the right."""
        sim = _make_sim(c=1.0, N=256, L=10.0)
        sim.reset()
        pos0 = sim.peak_position

        for _ in range(50):
            sim.step()

        pos1 = sim.peak_position
        # Account for periodic wrapping
        delta = (pos1 - pos0) % sim.L
        assert delta > 0

    def test_pulse_translates_left(self):
        """With c < 0, pulse should move to the left."""
        sim = _make_sim(c=-1.0, N=256, L=10.0)
        sim.reset()
        pos0 = sim.peak_position

        for _ in range(50):
            sim.step()

        pos1 = sim.peak_position
        # Peak should have moved left
        delta = (pos1 - pos0) % sim.L
        # When moving left, delta wraps to near L
        assert delta > sim.L / 2

    def test_speed_matches_c(self):
        """Measured pulse speed should approximately match c."""
        c = 2.0
        L = 10.0
        sim = _make_sim(c=c, N=256, L=L)
        sim.reset()
        pos0 = sim.peak_position

        n_steps = 100
        for _ in range(n_steps):
            sim.step()

        t = n_steps * sim.config.dt
        pos1 = sim.peak_position

        delta_x = (pos1 - pos0) % L
        if delta_x > L / 2:
            delta_x -= L

        measured_speed = delta_x / t
        assert measured_speed == pytest.approx(c, rel=0.1)

    def test_negative_speed_matches_c(self):
        """Measured pulse speed should match negative c."""
        c = -1.5
        L = 10.0
        sim = _make_sim(c=c, N=256, L=L)
        sim.reset()
        pos0 = sim.peak_position

        n_steps = 100
        for _ in range(n_steps):
            sim.step()

        t = n_steps * sim.config.dt
        pos1 = sim.peak_position

        delta_x = pos1 - pos0
        if delta_x > L / 2:
            delta_x -= L
        if delta_x < -L / 2:
            delta_x += L

        measured_speed = delta_x / t
        assert measured_speed == pytest.approx(c, rel=0.1)

    def test_faster_c_moves_further(self):
        """Higher wave speed should move the pulse further in the same time."""
        n_steps = 50

        sim1 = _make_sim(c=1.0, N=256, L=10.0)
        sim1.reset()
        pos1_0 = sim1.peak_position
        for _ in range(n_steps):
            sim1.step()
        pos1_1 = sim1.peak_position
        delta1 = (pos1_1 - pos1_0) % sim1.L

        sim2 = _make_sim(c=3.0, N=256, L=10.0)
        sim2.reset()
        pos2_0 = sim2.peak_position
        for _ in range(n_steps):
            sim2.step()
        pos2_1 = sim2.peak_position
        delta2 = (pos2_1 - pos2_0) % sim2.L

        # c=3 should move roughly 3x further than c=1
        # But dt is different (CFL-based), so compare actual distances
        t1 = n_steps * sim1.config.dt
        t2 = n_steps * sim2.config.dt
        speed1 = delta1 / t1
        speed2 = delta2 / t2
        assert speed2 > speed1


# ---------------------------------------------------------------------------
# Physics: periodic boundary conditions
# ---------------------------------------------------------------------------

class TestAdvectionPeriodicBC:
    def test_pulse_wraps_around(self):
        """Pulse should wrap around the periodic domain."""
        c = 2.0
        L = 10.0
        sim = _make_sim(c=c, N=256, L=L)
        sim.reset()

        # Run long enough for the pulse to travel more than L
        dt = sim.config.dt
        n_for_full_period = int(L / (c * dt)) + 100
        for _ in range(n_for_full_period):
            sim.step()

        # Pulse should still be well-defined (no NaN, finite values)
        state = sim.observe()
        assert not np.any(np.isnan(state))
        assert np.max(state) > 0.1  # Pulse still has amplitude

    def test_periodicity_preserved(self):
        """After traveling exactly L, pulse should return near initial position."""
        c = 1.0
        L = 10.0
        N = 256
        sim = _make_sim(c=c, N=N, L=L)
        sim.reset()

        dt = sim.config.dt
        # Steps for one full period
        n_full = int(round(L / (c * dt)))
        for _ in range(n_full):
            sim.step()

        # Peak should be near the initial position (L/4)
        pos = sim.peak_position
        initial_pos = L / 4.0
        # Allow some numerical diffusion to shift the peak slightly
        dist = abs(pos - initial_pos)
        dist = min(dist, L - dist)  # periodic distance
        assert dist < L * 0.1  # Within 10% of domain size


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

class TestAdvectionConservation:
    def test_mass_conserved(self):
        """Total mass (integral of u) should be conserved."""
        sim = _make_sim(N=256, L=10.0)
        sim.reset()
        m0 = sim.total_mass

        for _ in range(200):
            sim.step()

        m1 = sim.total_mass
        np.testing.assert_allclose(m0, m1, rtol=1e-10)

    def test_mass_conserved_negative_c(self):
        """Mass conservation should hold for negative wave speed too."""
        sim = _make_sim(c=-2.0, N=256, L=10.0)
        sim.reset()
        m0 = sim.total_mass

        for _ in range(200):
            sim.step()

        m1 = sim.total_mass
        np.testing.assert_allclose(m0, m1, rtol=1e-10)

    def test_mass_conserved_high_cfl(self):
        """Mass conservation at CFL near 1."""
        N = 128
        L = 10.0
        c = 1.0
        dx = L / N
        dt = 0.95 * dx / c  # CFL = 0.95
        sim = _make_sim(c=c, N=N, L=L, dt=dt)
        sim.reset()
        m0 = sim.total_mass

        for _ in range(100):
            sim.step()

        m1 = sim.total_mass
        np.testing.assert_allclose(m0, m1, rtol=1e-10)


# ---------------------------------------------------------------------------
# CFL stability
# ---------------------------------------------------------------------------

class TestAdvectionCFL:
    def test_cfl_number_computation(self):
        N = 100
        L = 10.0
        c = 2.0
        dx = L / N
        dt = 0.5 * dx / c  # CFL = 0.5
        sim = _make_sim(c=c, N=N, L=L, dt=dt)
        assert sim.cfl_number == pytest.approx(0.5, rel=1e-10)

    def test_cfl_auto_adjustment(self):
        """If dt is too large, constructor should adjust it for stability."""
        N = 100
        L = 10.0
        c = 1.0
        dx = L / N
        dt_bad = 2.0 * dx / c  # CFL = 2.0, unstable
        sim = _make_sim(c=c, N=N, L=L, dt=dt_bad)
        # dt should have been reduced
        assert sim.cfl_number <= 1.0

    def test_stable_below_cfl_1(self):
        """Simulation should be stable with CFL < 1."""
        sim = _make_sim(c=1.0, N=64, L=10.0)
        sim.reset()
        for _ in range(500):
            sim.step()
        state = sim.observe()
        assert not np.any(np.isnan(state))
        assert np.max(np.abs(state)) < 10.0

    def test_cfl_zero_speed(self):
        """c=0 should give CFL=0 and static solution."""
        sim = _make_sim(c=0.0, N=64, L=10.0, dt=0.01)
        sim.reset()
        s0 = sim.observe().copy()
        for _ in range(100):
            sim.step()
        s1 = sim.observe()
        np.testing.assert_array_equal(s0, s1)


# ---------------------------------------------------------------------------
# Exact solution comparison
# ---------------------------------------------------------------------------

class TestAdvectionExactSolution:
    def test_exact_solution_at_t0(self):
        """Exact solution at t=0 should match initial condition."""
        sim = _make_sim(N=256, L=10.0)
        sim.reset()
        exact = sim.exact_solution(0.0)
        np.testing.assert_allclose(sim.observe(), exact, atol=1e-14)

    def test_l2_error_at_t0(self):
        """L2 error at t=0 should be zero."""
        sim = _make_sim(N=256, L=10.0)
        sim.reset()
        err = sim.l2_error(0.0)
        assert err == pytest.approx(0.0, abs=1e-13)

    def test_l2_error_grows_slowly(self):
        """L2 error should grow slowly (numerical diffusion)."""
        sim = _make_sim(c=1.0, N=256, L=10.0)
        sim.reset()
        dt = sim.config.dt

        for _ in range(100):
            sim.step()

        t = 100 * dt
        err = sim.l2_error(t)
        # Error should be small but nonzero (upwind diffusion)
        assert err > 0
        assert err < 1.0  # Should not be huge

    def test_finer_grid_less_error(self):
        """Finer grid should produce smaller error."""
        n_steps = 100

        sim_coarse = _make_sim(c=1.0, N=64, L=10.0)
        sim_coarse.reset()
        for _ in range(n_steps):
            sim_coarse.step()
        t1 = n_steps * sim_coarse.config.dt
        err_coarse = sim_coarse.l2_error(t1)

        sim_fine = _make_sim(c=1.0, N=256, L=10.0)
        sim_fine.reset()
        for _ in range(n_steps):
            sim_fine.step()
        t2 = n_steps * sim_fine.config.dt
        err_fine = sim_fine.l2_error(t2)

        # Fine grid should have smaller error, but dt also changes with CFL
        # so we compare at the same physical time by adjusting steps
        # Just check that fine grid error is smaller
        assert err_fine < err_coarse


# ---------------------------------------------------------------------------
# Reset determinism
# ---------------------------------------------------------------------------

class TestAdvectionDeterminism:
    def test_reset_deterministic(self):
        """Two resets should produce identical states."""
        sim = _make_sim()
        s1 = sim.reset()
        s2 = sim.reset()
        np.testing.assert_array_equal(s1, s2)

    def test_trajectories_deterministic(self):
        """Two runs should produce identical trajectories."""
        sim = _make_sim()

        sim.reset()
        states1 = []
        for _ in range(10):
            states1.append(sim.step().copy())

        sim.reset()
        states2 = []
        for _ in range(10):
            states2.append(sim.step().copy())

        for s1, s2 in zip(states1, states2):
            np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# TrajectoryData
# ---------------------------------------------------------------------------

class TestAdvectionTrajectory:
    def test_run_returns_trajectory(self):
        sim = _make_sim(n_steps=50)
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 256)  # n_steps + 1

    def test_trajectory_timestamps(self):
        sim = _make_sim(n_steps=50)
        traj = sim.run(n_steps=50)
        assert len(traj.timestamps) == 51
        assert traj.timestamps[0] == 0.0
        assert traj.timestamps[-1] > 0.0

    def test_trajectory_parameters(self):
        sim = _make_sim(c=2.0, N=128)
        traj = sim.run(n_steps=10)
        assert traj.parameters["c"] == 2.0
        assert traj.parameters["N"] == 128.0

    def test_trajectory_no_nan(self):
        sim = _make_sim(n_steps=100)
        traj = sim.run(n_steps=100)
        assert not np.any(np.isnan(traj.states))


# ---------------------------------------------------------------------------
# Rediscovery data generation
# ---------------------------------------------------------------------------

class TestAdvectionRediscovery:
    def test_speed_data_generation(self):
        from simulating_anything.rediscovery.advection_1d import generate_speed_data
        data = generate_speed_data(n_c=5, n_steps=50, N=64, L=10.0)
        assert len(data["c_values"]) == 5
        assert len(data["speed_measured"]) == 5
        valid = np.isfinite(data["speed_measured"])
        assert np.sum(valid) >= 3

    def test_speed_data_correlation(self):
        """Measured speeds should correlate well with theoretical."""
        from simulating_anything.rediscovery.advection_1d import generate_speed_data
        data = generate_speed_data(n_c=10, n_steps=100, N=128, L=10.0)
        valid = np.isfinite(data["speed_measured"])
        if np.sum(valid) >= 3:
            corr = np.corrcoef(
                data["speed_measured"][valid],
                data["speed_theory"][valid],
            )[0, 1]
            assert corr > 0.9

    def test_diffusion_data_generation(self):
        from simulating_anything.rediscovery.advection_1d import generate_diffusion_data
        data = generate_diffusion_data(
            n_steps_values=[0, 25, 50],
            N=64,
            L=10.0,
        )
        assert len(data["times"]) == 3
        assert len(data["sigma_measured"]) == 3
        # Sigma should increase over time (numerical diffusion)
        assert data["sigma_measured"][-1] >= data["sigma_measured"][0]

    def test_exact_comparison_data(self):
        from simulating_anything.rediscovery.advection_1d import (
            generate_exact_comparison_data,
        )
        data = generate_exact_comparison_data(c=1.0, N=64, L=10.0, n_steps=50)
        assert len(data["times"]) > 2
        assert len(data["l2_errors"]) > 2
        # Error at t=0 should be ~0
        assert data["l2_errors"][0] == pytest.approx(0.0, abs=1e-12)

    def test_cfl_stability_data(self):
        from simulating_anything.rediscovery.advection_1d import (
            generate_cfl_stability_data,
        )
        data = generate_cfl_stability_data(
            n_cfl=10,
            c=1.0,
            N=32,
            L=10.0,
            n_steps=50,
        )
        assert len(data["cfl_values"]) == 10
        assert len(data["stable"]) == 10
        # At least some low CFL values should be stable
        assert data["stable"][0]
        # High CFL (> 1) should be unstable
        high_cfl_mask = data["cfl_values"] > 1.1
        if np.any(high_cfl_mask):
            assert not np.all(data["stable"][high_cfl_mask])
