"""Tests for the Goodwin oscillator model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.goodwin import GoodwinSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a: float = 1.0,
    K: float = 1.0,
    n: float = 9.0,
    b: float = 0.1,
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.1,
    delta: float = 0.1,
    dt: float = 0.1,
    n_steps: int = 10000,
    **kwargs: float,
) -> SimulationConfig:
    params: dict[str, float] = {
        "a": a, "K": K, "n": n, "b": b,
        "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
    }
    params.update(kwargs)
    return SimulationConfig(
        domain=Domain.GOODWIN,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


# ──────────────────────── Basic Simulation Tests ────────────────────────


class TestGoodwinSimulation:
    def test_initial_state_shape(self):
        sim = GoodwinSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_positive(self):
        sim = GoodwinSimulation(_make_config())
        state = sim.reset()
        assert np.all(state > 0), f"Initial state has non-positive values: {state}"

    def test_step_advances_state(self):
        sim = GoodwinSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = GoodwinSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_run_returns_trajectory(self):
        sim = GoodwinSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_trajectory_timestamps_correct(self):
        dt = 0.1
        sim = GoodwinSimulation(_make_config(dt=dt, n_steps=50))
        traj = sim.run(n_steps=50)
        expected = np.arange(51) * dt
        np.testing.assert_allclose(traj.timestamps, expected, atol=1e-10)

    def test_state_dtype(self):
        sim = GoodwinSimulation(_make_config())
        state = sim.reset()
        assert state.dtype == np.float64


# ──────────────────────── Hill Function Tests ────────────────────────


class TestHillFunction:
    def test_hill_function_at_zero(self):
        """h(0) = a / K^n."""
        sim = GoodwinSimulation(_make_config(a=1.0, K=1.0, n=9.0))
        sim.reset()
        h = sim.hill_function(0.0)
        assert h == pytest.approx(1.0 / 1.0**9)

    def test_hill_function_at_K(self):
        """h(K) = a / (K^n + K^n) = a / (2 * K^n)."""
        sim = GoodwinSimulation(_make_config(a=1.0, K=1.0, n=9.0))
        sim.reset()
        h = sim.hill_function(1.0)
        assert h == pytest.approx(0.5)

    def test_hill_function_decreasing(self):
        """Hill function should decrease as z increases (negative feedback)."""
        sim = GoodwinSimulation(_make_config(a=1.0, K=1.0, n=9.0))
        sim.reset()
        z_vals = np.linspace(0.01, 3.0, 20)
        h_vals = [sim.hill_function(z) for z in z_vals]
        for i in range(len(h_vals) - 1):
            assert h_vals[i] > h_vals[i + 1], "Hill function not monotonically decreasing"

    def test_hill_function_positive(self):
        """Hill function should always be positive."""
        sim = GoodwinSimulation(_make_config())
        sim.reset()
        for z in [0.0, 0.5, 1.0, 2.0, 10.0]:
            assert sim.hill_function(z) > 0

    def test_hill_function_array(self):
        """Hill function should work with numpy arrays."""
        sim = GoodwinSimulation(_make_config(a=1.0, K=1.0, n=9.0))
        sim.reset()
        z_arr = np.array([0.0, 0.5, 1.0, 2.0])
        h = sim.hill_function(z_arr)
        assert h.shape == (4,)
        assert np.all(h > 0)

    def test_hill_function_steep_for_large_n(self):
        """Large n should give a steep transition around K."""
        sim = GoodwinSimulation(_make_config(n=50.0, K=1.0))
        sim.reset()
        # Below K: Hill function ~ a/K^n ~ 1
        h_below = sim.hill_function(0.5)
        # Above K: Hill function ~ 0
        h_above = sim.hill_function(1.5)
        assert h_below > 0.9
        assert h_above < 0.01


# ──────────────────────── Equilibrium Tests ────────────────────────


class TestGoodwinEquilibrium:
    def test_equilibrium_exists(self):
        sim = GoodwinSimulation(_make_config())
        eq = sim.equilibrium()
        assert len(eq) == 3
        assert all(v > 0 for v in eq)

    def test_derivatives_at_equilibrium(self):
        """At the equilibrium, derivatives should be zero."""
        sim = GoodwinSimulation(_make_config())
        sim.reset()
        eq = np.array(sim.equilibrium())
        dy = sim._derivatives(eq)
        np.testing.assert_allclose(dy, [0.0, 0.0, 0.0], atol=1e-10)

    def test_equilibrium_different_params(self):
        """Equilibrium should change with parameters."""
        sim1 = GoodwinSimulation(_make_config(a=1.0))
        sim2 = GoodwinSimulation(_make_config(a=2.0))
        eq1 = sim1.equilibrium()
        eq2 = sim2.equilibrium()
        # Higher production rate a should give higher equilibrium concentrations
        assert eq2[0] > eq1[0]

    def test_equilibrium_chain_relations(self):
        """y* = (alpha/beta)*x*, z* = (gamma/delta)*y*."""
        a, K, n, b = 1.0, 1.0, 9.0, 0.1
        alpha, beta, gamma, delta = 0.1, 0.1, 0.1, 0.1
        sim = GoodwinSimulation(_make_config(
            a=a, K=K, n=n, b=b,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        ))
        x_star, y_star, z_star = sim.equilibrium()
        assert y_star == pytest.approx((alpha / beta) * x_star, rel=1e-8)
        assert z_star == pytest.approx((gamma / delta) * y_star, rel=1e-8)

    def test_equilibrium_satisfies_hill_balance(self):
        """At equilibrium: a / (K^n + z*^n) = b * x*."""
        sim = GoodwinSimulation(_make_config())
        x_star, _, z_star = sim.equilibrium()
        lhs = sim.hill_function(z_star)
        rhs = sim.b * x_star
        assert lhs == pytest.approx(rhs, rel=1e-8)


# ──────────────────────── Oscillation Tests ────────────────────────


class TestGoodwinOscillation:
    def test_oscillation_for_n_9(self):
        """Default n=9 should produce oscillations (n > 8)."""
        sim = GoodwinSimulation(_make_config(n=9.0, dt=0.1))
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        # Measure variation
        x_vals = []
        for _ in range(5000):
            sim.step()
            x_vals.append(sim.observe()[0])
        amplitude = max(x_vals) - min(x_vals)
        assert amplitude > 0.01, f"No oscillation detected, amplitude={amplitude}"

    def test_no_oscillation_for_n_2(self):
        """n=2 should converge to stable fixed point (n < 8)."""
        sim = GoodwinSimulation(_make_config(n=2.0, dt=0.1))
        eq = sim.equilibrium()
        sim.reset()
        # Run long enough for convergence
        for _ in range(20000):
            sim.step()
        state = sim.observe()
        # Should be near equilibrium
        np.testing.assert_allclose(state, np.array(eq), atol=0.05)

    def test_no_oscillation_for_n_5(self):
        """n=5 should also converge (well below threshold)."""
        sim = GoodwinSimulation(_make_config(n=5.0, dt=0.1))
        eq = sim.equilibrium()
        sim.reset()
        for _ in range(20000):
            sim.step()
        state = sim.observe()
        np.testing.assert_allclose(state, np.array(eq), atol=0.1)

    def test_is_oscillatory_flag(self):
        sim_below = GoodwinSimulation(_make_config(n=5.0))
        assert not sim_below.is_oscillatory

        sim_above = GoodwinSimulation(_make_config(n=9.0))
        assert sim_above.is_oscillatory

    def test_is_oscillatory_boundary(self):
        sim_at = GoodwinSimulation(_make_config(n=8.0))
        assert not sim_at.is_oscillatory  # n must be strictly > 8

        sim_just_above = GoodwinSimulation(_make_config(n=8.01))
        assert sim_just_above.is_oscillatory


# ──────────────────────── State Non-negativity Tests ────────────────────────


class TestGoodwinNonNegativity:
    def test_state_non_negative_oscillatory(self):
        """All concentrations should remain non-negative during oscillation."""
        sim = GoodwinSimulation(_make_config(n=9.0, dt=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= -1e-10), f"Negative state: {state}"

    def test_state_non_negative_stable(self):
        """All concentrations should remain non-negative at stable fixed point."""
        sim = GoodwinSimulation(_make_config(n=2.0, dt=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= -1e-10), f"Negative state: {state}"

    def test_trajectory_bounded(self):
        """Trajectory should remain bounded for default parameters."""
        sim = GoodwinSimulation(_make_config(n=9.0, dt=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.abs(state) < 100), f"State diverged: {state}"


# ──────────────────────── Period Tests ────────────────────────


class TestGoodwinPeriod:
    def test_compute_period_oscillatory(self):
        """Should return finite period when n > 8."""
        sim = GoodwinSimulation(_make_config(n=9.0, dt=0.1))
        sim.reset()
        period = sim.compute_period(n_transient=3000, n_measure=5000)
        assert np.isfinite(period)
        assert period > 0

    def test_compute_period_stable(self):
        """Should return inf when n < 8 (no oscillation)."""
        sim = GoodwinSimulation(_make_config(n=2.0, dt=0.1))
        sim.reset()
        period = sim.compute_period()
        assert period == float("inf")

    def test_period_positive(self):
        """Period should be strictly positive."""
        sim = GoodwinSimulation(_make_config(n=10.0, dt=0.1))
        sim.reset()
        period = sim.compute_period(n_transient=3000, n_measure=5000)
        assert period > 1.0, f"Period too short: {period}"

    def test_measure_amplitude_oscillatory(self):
        """Should return positive amplitude when oscillating."""
        sim = GoodwinSimulation(_make_config(n=9.0, dt=0.1))
        sim.reset()
        amp = sim.measure_amplitude(transient_time=300.0)
        assert amp > 0.01, f"Amplitude too small: {amp}"


# ──────────────────────── Hill Sweep Tests ────────────────────────


class TestGoodwinHillSweep:
    def test_hill_sweep_returns_correct_keys(self):
        sim = GoodwinSimulation(_make_config())
        sim.reset()
        result = sim.hill_sweep(n_range=(5.0, 10.0), n_points=5, n_steps=3000)
        assert "n_values" in result
        assert "amplitudes" in result
        assert "periods" in result

    def test_hill_sweep_correct_length(self):
        sim = GoodwinSimulation(_make_config())
        sim.reset()
        result = sim.hill_sweep(n_range=(5.0, 10.0), n_points=7, n_steps=3000)
        assert len(result["n_values"]) == 7
        assert len(result["amplitudes"]) == 7
        assert len(result["periods"]) == 7

    def test_hill_sweep_amplitudes_increase_with_n(self):
        """Amplitudes should generally increase as n goes above threshold."""
        sim = GoodwinSimulation(_make_config())
        sim.reset()
        result = sim.hill_sweep(n_range=(6.0, 12.0), n_points=8, n_steps=5000)
        # The last few points (n > 8) should have larger amplitudes than first few
        low_n_amps = result["amplitudes"][:3]  # n ~ 6-8
        high_n_amps = result["amplitudes"][-3:]  # n ~ 10-12
        assert np.mean(high_n_amps) > np.mean(low_n_amps)


# ──────────────────────── Reproducibility Tests ────────────────────────


class TestGoodwinReproducibility:
    def test_deterministic_trajectory(self):
        """Two runs with same config should produce identical trajectories."""
        config = _make_config(n=9.0, dt=0.1)
        sim1 = GoodwinSimulation(config)
        traj1 = sim1.run(n_steps=200)
        sim2 = GoodwinSimulation(config)
        traj2 = sim2.run(n_steps=200)
        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_different_params_different_trajectory(self):
        """Different parameters should give different trajectories."""
        traj1 = GoodwinSimulation(_make_config(n=9.0)).run(n_steps=200)
        traj2 = GoodwinSimulation(_make_config(n=10.0)).run(n_steps=200)
        assert not np.allclose(traj1.states, traj2.states)


# ──────────────────────── Rediscovery Data Generation Tests ────────────────────────


class TestGoodwinRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.goodwin import generate_ode_data

        data = generate_ode_data(n_steps=500, dt=0.1)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert data["n"] == 9.0

    def test_hill_sweep_data_generation(self):
        from simulating_anything.rediscovery.goodwin import generate_hill_sweep_data

        data = generate_hill_sweep_data(n_range=(5.0, 10.0), n_points=5, dt=0.1)
        assert len(data["n_values"]) == 5
        assert len(data["amplitudes"]) == 5

    def test_period_data_generation(self):
        from simulating_anything.rediscovery.goodwin import generate_period_data

        data = generate_period_data(n_range=(9.0, 11.0), n_points=3, dt=0.1)
        assert len(data["n_values"]) == 3
        assert len(data["periods"]) == 3
