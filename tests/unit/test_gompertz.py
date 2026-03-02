"""Tests for the Gompertz growth model simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.gompertz import GompertzSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 0.5,
    K: float = 100.0,
    N_0: float = 1.0,
    dt: float = 0.1,
    n_steps: int = 500,
) -> SimulationConfig:
    """Create a SimulationConfig for the Gompertz model."""
    return SimulationConfig(
        domain=Domain.GOMPERTZ,
        dt=dt,
        n_steps=n_steps,
        parameters={"r": r, "K": K, "N_0": N_0},
    )


# ---------------------------------------------------------------------------
# Basic simulation tests
# ---------------------------------------------------------------------------

class TestGompertzBasic:
    """Tests for simulation creation and basic mechanics."""

    def test_creation_and_params(self):
        """Simulation should be created with specified parameters."""
        sim = GompertzSimulation(_make_config(r=0.3, K=50.0, N_0=2.0))
        assert sim.r == 0.3
        assert sim.K == 50.0
        assert sim.N_0 == 2.0

    def test_default_parameters(self):
        """Default parameters should match docstring specification."""
        config = SimulationConfig(
            domain=Domain.GOMPERTZ, dt=0.1, n_steps=10, parameters={},
        )
        sim = GompertzSimulation(config)
        assert sim.r == 0.5
        assert sim.K == 100.0
        assert sim.N_0 == 1.0

    def test_initial_state_shape(self):
        """State vector should be 1D with a single element."""
        sim = GompertzSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)

    def test_initial_state_value(self):
        """Initial state should equal N_0."""
        sim = GompertzSimulation(_make_config(N_0=5.0))
        state = sim.reset()
        assert state[0] == pytest.approx(5.0)

    def test_step_advances_state(self):
        """State should change after a step."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0))
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        """Observe should return the state set by reset."""
        sim = GompertzSimulation(_make_config(N_0=10.0))
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_run_returns_trajectory(self):
        """run() should produce a TrajectoryData object of correct shape."""
        sim = GompertzSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 1)
        assert len(traj.timestamps) == 101

    def test_dt_applied_correctly(self):
        """Trajectory timestamps should reflect dt."""
        sim = GompertzSimulation(_make_config(dt=0.2, n_steps=5))
        traj = sim.run(n_steps=5)
        np.testing.assert_allclose(traj.timestamps, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = GompertzSimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = GompertzSimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)


# ---------------------------------------------------------------------------
# Growth dynamics tests
# ---------------------------------------------------------------------------

class TestGompertzDynamics:
    """Tests for growth dynamics and analytical properties."""

    def test_growth_positive(self):
        """Population should increase when N_0 < K."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0))
        sim.reset()
        for _ in range(10):
            sim.step()
        assert sim.observe()[0] > 1.0

    def test_approaches_carrying_capacity(self):
        """Population should approach K for long runs."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0, dt=0.1))
        sim.reset()
        for _ in range(5000):
            sim.step()
        assert sim.observe()[0] == pytest.approx(100.0, abs=0.1)

    def test_starts_above_K_decreases(self):
        """If N_0 > K, population should decrease toward K.

        Note: Gompertz dN/dt = r*N*ln(K/N); when N > K, ln(K/N) < 0 so dN/dt < 0.
        """
        sim = GompertzSimulation(_make_config(r=0.5, K=50.0, N_0=80.0, dt=0.01))
        sim.reset()
        for _ in range(10000):
            sim.step()
        assert sim.observe()[0] == pytest.approx(50.0, abs=0.1)

    def test_inflection_point_formula(self):
        """Inflection point should be K / e."""
        sim = GompertzSimulation(_make_config(K=100.0))
        assert sim.inflection_point == pytest.approx(100.0 / np.e)

    def test_max_growth_rate_formula(self):
        """Maximum growth rate should be r * K / e."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0))
        assert sim.max_growth_rate == pytest.approx(0.5 * 100.0 / np.e)

    def test_analytical_solution_at_t0(self):
        """Analytical solution at t=0 should equal N_0."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=5.0))
        assert sim.analytical_solution(0.0) == pytest.approx(5.0)

    def test_analytical_solution_at_large_t(self):
        """Analytical solution should approach K as t -> infinity."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0))
        assert sim.analytical_solution(1000.0) == pytest.approx(100.0, abs=0.01)

    def test_analytical_matches_numerical(self):
        """RK4 numerical solution should match analytical solution closely."""
        config = _make_config(r=0.5, K=100.0, N_0=1.0, dt=0.01, n_steps=2000)
        sim = GompertzSimulation(config)
        sim.reset()

        times = np.arange(2001) * 0.01
        analytical = sim.analytical_solution(times)

        states = [sim.observe().copy()]
        for _ in range(2000):
            sim.step()
            states.append(sim.observe().copy())
        numerical = np.array(states)[:, 0]

        np.testing.assert_allclose(numerical, analytical, rtol=1e-4)

    def test_inflection_time_computed(self):
        """Inflection time should be positive and finite for N_0 < K/e."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0))
        t_inf = sim.inflection_time()
        assert 0 < t_inf < 100.0

    def test_inflection_time_zero_when_above(self):
        """Inflection time should be 0 when N_0 >= K/e."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=50.0))
        t_inf = sim.inflection_time()
        assert t_inf == pytest.approx(0.0)

    def test_inflection_time_matches_simulation(self):
        """The measured inflection time from simulation should match the formula."""
        r, K, N_0 = 0.5, 100.0, 1.0
        sim = GompertzSimulation(_make_config(r=r, K=K, N_0=N_0, dt=0.01))
        t_inf_theory = sim.inflection_time()

        # Simulate and find the time of maximum growth rate
        sim.reset()
        max_rate = 0.0
        t_max_rate = 0.0
        prev_N = N_0
        for step in range(5000):
            sim.step()
            curr_N = float(sim.observe()[0])
            rate = (curr_N - prev_N) / 0.01
            if rate > max_rate:
                max_rate = rate
                t_max_rate = (step + 1) * 0.01
            prev_N = curr_N

        assert t_max_rate == pytest.approx(t_inf_theory, abs=0.1)

    def test_growth_rate_at_inflection(self):
        """Growth rate at inflection point should equal r * K / e."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0))
        N_inf = sim.inflection_point
        rate = sim.growth_rate_at(N_inf)
        assert rate == pytest.approx(sim.max_growth_rate, rel=1e-10)

    def test_growth_rate_at_K_is_zero(self):
        """Growth rate at carrying capacity should be zero."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0))
        assert sim.growth_rate_at(100.0) == pytest.approx(0.0)

    def test_growth_rate_at_zero_is_zero(self):
        """Growth rate at N=0 should return 0 (boundary)."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0))
        assert sim.growth_rate_at(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Parameter variation tests
# ---------------------------------------------------------------------------

class TestGompertzParameters:
    """Tests for different parameter values."""

    def test_higher_r_faster_growth(self):
        """Higher r should reach K faster."""
        sim_slow = GompertzSimulation(_make_config(r=0.3, K=100.0, N_0=1.0))
        sim_fast = GompertzSimulation(_make_config(r=1.0, K=100.0, N_0=1.0))
        sim_slow.reset()
        sim_fast.reset()

        for _ in range(200):
            sim_slow.step()
            sim_fast.step()

        assert sim_fast.observe()[0] > sim_slow.observe()[0]

    def test_different_K_values(self):
        """Final population should match respective K values."""
        for K in [50.0, 100.0, 200.0]:
            sim = GompertzSimulation(_make_config(r=0.5, K=K, N_0=1.0, dt=0.1))
            sim.reset()
            for _ in range(5000):
                sim.step()
            assert sim.observe()[0] == pytest.approx(K, abs=0.5)

    def test_different_N0_same_final(self):
        """Different initial conditions should converge to the same K."""
        final_values = []
        for N_0 in [0.1, 1.0, 10.0, 50.0]:
            sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=N_0, dt=0.1))
            sim.reset()
            for _ in range(5000):
                sim.step()
            final_values.append(sim.observe()[0])

        for val in final_values:
            assert val == pytest.approx(100.0, abs=0.5)

    def test_small_r_slow_convergence(self):
        """Very small r should not reach K in a short simulation."""
        sim = GompertzSimulation(_make_config(r=0.01, K=100.0, N_0=1.0, dt=0.1))
        sim.reset()
        for _ in range(100):
            sim.step()
        # Should still be far from K after only 10 time units with r=0.01
        assert sim.observe()[0] < 50.0

    def test_inflection_scales_with_K(self):
        """Inflection point should be proportional to K."""
        inf_50 = GompertzSimulation(_make_config(K=50.0)).inflection_point
        inf_100 = GompertzSimulation(_make_config(K=100.0)).inflection_point
        assert inf_100 == pytest.approx(2 * inf_50)

    def test_max_growth_rate_scales_with_r(self):
        """Max growth rate should be proportional to r."""
        rate_1 = GompertzSimulation(_make_config(r=0.5, K=100.0)).max_growth_rate
        rate_2 = GompertzSimulation(_make_config(r=1.0, K=100.0)).max_growth_rate
        assert rate_2 == pytest.approx(2 * rate_1)

    def test_max_growth_rate_scales_with_K(self):
        """Max growth rate should be proportional to K."""
        rate_1 = GompertzSimulation(_make_config(r=0.5, K=50.0)).max_growth_rate
        rate_2 = GompertzSimulation(_make_config(r=0.5, K=100.0)).max_growth_rate
        assert rate_2 == pytest.approx(2 * rate_1)


# ---------------------------------------------------------------------------
# Conservation / bounds tests
# ---------------------------------------------------------------------------

class TestGompertzConservation:
    """Tests for population bounds and positivity."""

    def test_population_always_positive(self):
        """Population should remain positive at all times."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=0.01, dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] > 0, f"Population went non-positive: {state[0]}"

    def test_population_bounded_by_K(self):
        """Population should not exceed carrying capacity."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0, dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] <= 100.0 + 1e-10, (
                f"Population exceeded K: {state[0]} > 100.0"
            )

    def test_population_bounded_starting_above(self):
        """When starting above K, population should still be bounded."""
        sim = GompertzSimulation(_make_config(r=0.5, K=50.0, N_0=80.0, dt=0.01))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] > 0, f"Population went non-positive: {state[0]}"

    def test_no_nan_long_run(self):
        """No NaN or Inf after many steps."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0, dt=0.1))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"

    def test_monotonic_growth_below_K(self):
        """Population should monotonically increase when N_0 < K."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0, dt=0.1))
        sim.reset()
        prev = sim.observe()[0]
        for _ in range(1000):
            state = sim.step()
            assert state[0] >= prev - 1e-10, (
                f"Population decreased: {prev} -> {state[0]}"
            )
            prev = state[0]

    def test_monotonic_decrease_above_K(self):
        """Population should monotonically decrease when N_0 > K."""
        sim = GompertzSimulation(_make_config(r=0.5, K=50.0, N_0=80.0, dt=0.01))
        sim.reset()
        prev = sim.observe()[0]
        for _ in range(5000):
            state = sim.step()
            assert state[0] <= prev + 1e-10, (
                f"Population increased above K: {prev} -> {state[0]}"
            )
            prev = state[0]


# ---------------------------------------------------------------------------
# Analytical formula tests
# ---------------------------------------------------------------------------

class TestGompertzAnalytical:
    """Tests for analytical solution and derived formulas."""

    def test_analytical_solution_array(self):
        """Analytical solution should accept array input."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0))
        times = np.array([0.0, 1.0, 5.0, 10.0, 100.0])
        result = sim.analytical_solution(times)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))
        # Should be monotonically increasing
        assert np.all(np.diff(result) >= 0)

    def test_analytical_solution_monotonic(self):
        """Analytical solution should be monotonically increasing (N_0 < K)."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0))
        times = np.linspace(0, 50, 100)
        result = sim.analytical_solution(times)
        assert np.all(np.diff(result) >= 0)

    def test_half_saturation_time(self):
        """Half-saturation time should be computable for N_0 < K/2."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=1.0))
        t_half = sim.half_saturation_time()
        assert 0 < t_half < 100.0
        # Verify: N(t_half) should be approximately K/2
        N_at_half = sim.analytical_solution(t_half)
        assert N_at_half == pytest.approx(50.0, abs=0.5)

    def test_half_saturation_zero_when_above(self):
        """Half-saturation time should be 0 when N_0 >= K/2."""
        sim = GompertzSimulation(_make_config(r=0.5, K=100.0, N_0=60.0))
        assert sim.half_saturation_time() == pytest.approx(0.0)

    def test_r_sweep(self):
        """r_sweep should return arrays of correct length."""
        sim = GompertzSimulation(_make_config())
        r_values = np.linspace(0.1, 1.0, 5)
        result = sim.r_sweep(r_values=r_values, n_steps=200)
        assert len(result["r"]) == 5
        assert len(result["final_N"]) == 5
        assert len(result["inflection_time"]) == 5


# ---------------------------------------------------------------------------
# Rediscovery data generation tests
# ---------------------------------------------------------------------------

class TestGompertzRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_growth_data(self):
        """Growth data should have correct structure and sizes."""
        from simulating_anything.rediscovery.gompertz import generate_growth_data
        data = generate_growth_data(n_samples=25, dt=0.1, n_steps=200)
        n = len(data["r"])
        assert n > 0
        assert len(data["K"]) == n
        assert len(data["inflection_N"]) == n
        assert len(data["max_dNdt"]) == n
        assert np.all(np.isfinite(data["inflection_N"]))
        assert np.all(data["max_dNdt"] > 0)

    def test_generate_ode_data(self):
        """ODE data should produce valid trajectory arrays."""
        from simulating_anything.rediscovery.gompertz import generate_ode_data
        data = generate_ode_data(r=0.5, K=100.0, N_0=1.0, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 1)
        assert len(data["time"]) == 501
        assert data["r"] == 0.5
        assert data["K"] == 100.0
        assert data["N_0"] == 1.0

    def test_generate_inflection_data(self):
        """Inflection data should have measured values close to K/e."""
        from simulating_anything.rediscovery.gompertz import generate_inflection_data
        data = generate_inflection_data(n_K=10, r=0.5, N_0=1.0, dt=0.05, n_steps=2000)
        assert len(data["K"]) == 10
        assert len(data["inflection_N_theory"]) == 10
        assert len(data["inflection_N_measured"]) == 10
        # Theory values should be K/e
        np.testing.assert_allclose(
            data["inflection_N_theory"],
            data["K"] / np.e,
            rtol=1e-10,
        )

    def test_inflection_data_correlation(self):
        """Measured inflection points should correlate with K/e theory."""
        from simulating_anything.rediscovery.gompertz import generate_inflection_data
        data = generate_inflection_data(n_K=20, r=0.5, N_0=1.0, dt=0.05, n_steps=2000)
        correlation = np.corrcoef(
            data["inflection_N_theory"],
            data["inflection_N_measured"],
        )[0, 1]
        assert correlation > 0.95

    def test_growth_data_sweep_ranges(self):
        """Growth data should cover specified parameter ranges."""
        from simulating_anything.rediscovery.gompertz import generate_growth_data
        data = generate_growth_data(n_samples=100, dt=0.1, n_steps=200)
        assert data["r"].min() < 0.2
        assert data["r"].max() > 1.5
        assert data["K"].min() < 20.0
        assert data["K"].max() > 150.0

    def test_ode_data_trajectory_grows(self):
        """ODE trajectory should show growth from N_0 toward K."""
        from simulating_anything.rediscovery.gompertz import generate_ode_data
        data = generate_ode_data(r=0.5, K=100.0, N_0=1.0, n_steps=1000, dt=0.01)
        N = data["N"]
        assert N[-1] > N[0], "Population should grow over time"
        assert N[-1] > 10.0, "Population should be well above N_0 after 10 time units"
