"""Tests for the continuous logistic (Verhulst) ODE simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.logistic_ode import LogisticODESimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 1.0,
    K: float = 100.0,
    N_0: float = 1.0,
    dt: float = 0.1,
    n_steps: int = 500,
) -> SimulationConfig:
    """Create a SimulationConfig for the logistic ODE model."""
    return SimulationConfig(
        domain=Domain.LOGISTIC_ODE,
        dt=dt,
        n_steps=n_steps,
        parameters={"r": r, "K": K, "N_0": N_0},
    )


# ---------------------------------------------------------------------------
# Basic simulation tests
# ---------------------------------------------------------------------------

class TestLogisticODEBasic:
    """Tests for simulation creation and basic mechanics."""

    def test_creation_and_params(self):
        """Simulation should be created with specified parameters."""
        sim = LogisticODESimulation(_make_config(r=2.0, K=50.0, N_0=5.0))
        assert sim.r == 2.0
        assert sim.K == 50.0
        assert sim.N_0 == 5.0

    def test_default_parameters(self):
        """Default parameters should match specification."""
        config = SimulationConfig(
            domain=Domain.LOGISTIC_ODE, dt=0.1, n_steps=10, parameters={},
        )
        sim = LogisticODESimulation(config)
        assert sim.r == 1.0
        assert sim.K == 100.0
        assert sim.N_0 == 1.0

    def test_initial_state_shape(self):
        """State vector should be 1D with a single element."""
        sim = LogisticODESimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)

    def test_initial_state_dtype(self):
        """State should be float64."""
        sim = LogisticODESimulation(_make_config())
        state = sim.reset()
        assert state.dtype == np.float64

    def test_initial_state_value(self):
        """Initial state should equal N_0."""
        sim = LogisticODESimulation(_make_config(N_0=5.0))
        state = sim.reset()
        assert state[0] == pytest.approx(5.0)

    def test_step_advances_state(self):
        """State should change after a step."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0))
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        """Observe should return the state set by reset."""
        sim = LogisticODESimulation(_make_config(N_0=10.0))
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_run_returns_trajectory(self):
        """run() should produce a TrajectoryData object of correct shape."""
        sim = LogisticODESimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 1)
        assert len(traj.timestamps) == 101

    def test_dt_applied_correctly(self):
        """Trajectory timestamps should reflect dt."""
        sim = LogisticODESimulation(_make_config(dt=0.2, n_steps=5))
        traj = sim.run(n_steps=5)
        np.testing.assert_allclose(traj.timestamps, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = LogisticODESimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = LogisticODESimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)


# ---------------------------------------------------------------------------
# Growth dynamics tests
# ---------------------------------------------------------------------------

class TestLogisticODEDynamics:
    """Tests for growth dynamics and analytical properties."""

    def test_growth_positive(self):
        """Population should increase when N_0 < K."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0))
        sim.reset()
        for _ in range(10):
            sim.step()
        assert sim.observe()[0] > 1.0

    def test_approaches_carrying_capacity(self):
        """Population should approach K for long runs."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0, dt=0.1))
        sim.reset()
        for _ in range(5000):
            sim.step()
        assert sim.observe()[0] == pytest.approx(100.0, abs=0.1)

    def test_starts_above_K_decreases(self):
        """If N_0 > K, population should decrease toward K."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=50.0, N_0=80.0, dt=0.01))
        sim.reset()
        for _ in range(10000):
            sim.step()
        assert sim.observe()[0] == pytest.approx(50.0, abs=0.1)

    def test_inflection_point_formula(self):
        """Inflection point should be K / 2."""
        sim = LogisticODESimulation(_make_config(K=100.0))
        assert sim.inflection_point == pytest.approx(50.0)

    def test_max_growth_rate_formula(self):
        """Maximum growth rate should be r * K / 4."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        assert sim.max_growth_rate == pytest.approx(25.0)

    def test_analytical_solution_at_t0(self):
        """Analytical solution at t=0 should equal N_0."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=5.0))
        assert sim.analytical_solution(0.0) == pytest.approx(5.0)

    def test_analytical_solution_at_large_t(self):
        """Analytical solution should approach K as t -> infinity."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0))
        assert sim.analytical_solution(1000.0) == pytest.approx(100.0, abs=0.01)

    def test_analytical_matches_numerical(self):
        """RK4 numerical solution should match analytical solution closely."""
        config = _make_config(r=1.0, K=100.0, N_0=1.0, dt=0.01, n_steps=2000)
        sim = LogisticODESimulation(config)
        sim.reset()

        times = np.arange(2001) * 0.01
        analytical = sim.analytical_solution(times)

        states = [sim.observe().copy()]
        for _ in range(2000):
            sim.step()
            states.append(sim.observe().copy())
        numerical = np.array(states)[:, 0]

        np.testing.assert_allclose(numerical, analytical, rtol=1e-4)

    def test_derivatives_at_zero(self):
        """dN/dt at N=0 should be zero."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        dy = sim._derivatives(np.array([0.0]))
        np.testing.assert_allclose(dy, [0.0], atol=1e-15)

    def test_derivatives_at_K(self):
        """dN/dt at N=K should be zero."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        dy = sim._derivatives(np.array([100.0]))
        np.testing.assert_allclose(dy, [0.0], atol=1e-15)

    def test_derivatives_at_half_K(self):
        """dN/dt at N=K/2 should be r*K/4 (maximum)."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        dy = sim._derivatives(np.array([50.0]))
        np.testing.assert_allclose(dy, [25.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Inflection time tests
# ---------------------------------------------------------------------------

class TestLogisticODEInflectionTime:
    """Tests for inflection time computation."""

    def test_inflection_time_positive(self):
        """Inflection time should be positive for N_0 < K/2."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0))
        t_inf = sim.inflection_time()
        assert 0 < t_inf < 100.0

    def test_inflection_time_zero_when_above(self):
        """Inflection time should be 0 when N_0 >= K/2."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=60.0))
        t_inf = sim.inflection_time()
        assert t_inf == pytest.approx(0.0)

    def test_inflection_time_formula(self):
        """Inflection time should be ln(K/N_0 - 1) / r."""
        r, K, N_0 = 1.0, 100.0, 1.0
        sim = LogisticODESimulation(_make_config(r=r, K=K, N_0=N_0))
        expected = np.log(K / N_0 - 1.0) / r
        assert sim.inflection_time() == pytest.approx(expected)

    def test_inflection_time_matches_simulation(self):
        """Measured inflection time from simulation should match the formula."""
        r, K, N_0 = 1.0, 100.0, 1.0
        sim = LogisticODESimulation(_make_config(r=r, K=K, N_0=N_0, dt=0.01))
        t_inf_theory = sim.inflection_time()

        # Simulate and find the time of maximum growth rate
        sim.reset()
        max_rate = 0.0
        t_max_rate = 0.0
        prev_N = N_0
        for step_i in range(5000):
            sim.step()
            curr_N = float(sim.observe()[0])
            rate = (curr_N - prev_N) / 0.01
            if rate > max_rate:
                max_rate = rate
                t_max_rate = (step_i + 1) * 0.01
            prev_N = curr_N

        assert t_max_rate == pytest.approx(t_inf_theory, abs=0.1)

    def test_inflection_time_decreases_with_r(self):
        """Higher r should yield earlier inflection time."""
        t1 = LogisticODESimulation(
            _make_config(r=0.5, K=100.0, N_0=1.0)
        ).inflection_time()
        t2 = LogisticODESimulation(
            _make_config(r=2.0, K=100.0, N_0=1.0)
        ).inflection_time()
        assert t2 < t1


# ---------------------------------------------------------------------------
# Parameter variation tests
# ---------------------------------------------------------------------------

class TestLogisticODEParameters:
    """Tests for different parameter values."""

    def test_higher_r_faster_growth(self):
        """Higher r should reach K faster."""
        sim_slow = LogisticODESimulation(_make_config(r=0.3, K=100.0, N_0=1.0))
        sim_fast = LogisticODESimulation(_make_config(r=2.0, K=100.0, N_0=1.0))
        sim_slow.reset()
        sim_fast.reset()

        for _ in range(100):
            sim_slow.step()
            sim_fast.step()

        assert sim_fast.observe()[0] > sim_slow.observe()[0]

    def test_different_K_values(self):
        """Final population should match respective K values."""
        for K in [50.0, 100.0, 200.0]:
            sim = LogisticODESimulation(_make_config(r=1.0, K=K, N_0=1.0, dt=0.1))
            sim.reset()
            for _ in range(5000):
                sim.step()
            assert sim.observe()[0] == pytest.approx(K, abs=0.5)

    def test_different_N0_same_final(self):
        """Different initial conditions should converge to the same K."""
        final_values = []
        for N_0 in [0.1, 1.0, 10.0, 50.0]:
            sim = LogisticODESimulation(
                _make_config(r=1.0, K=100.0, N_0=N_0, dt=0.1)
            )
            sim.reset()
            for _ in range(5000):
                sim.step()
            final_values.append(sim.observe()[0])

        for val in final_values:
            assert val == pytest.approx(100.0, abs=0.5)

    def test_inflection_scales_with_K(self):
        """Inflection point should be proportional to K."""
        inf_50 = LogisticODESimulation(_make_config(K=50.0)).inflection_point
        inf_100 = LogisticODESimulation(_make_config(K=100.0)).inflection_point
        assert inf_100 == pytest.approx(2 * inf_50)

    def test_max_growth_rate_scales_with_r(self):
        """Max growth rate should be proportional to r."""
        rate_1 = LogisticODESimulation(_make_config(r=0.5, K=100.0)).max_growth_rate
        rate_2 = LogisticODESimulation(_make_config(r=1.0, K=100.0)).max_growth_rate
        assert rate_2 == pytest.approx(2 * rate_1)

    def test_max_growth_rate_scales_with_K(self):
        """Max growth rate should be proportional to K."""
        rate_1 = LogisticODESimulation(_make_config(r=1.0, K=50.0)).max_growth_rate
        rate_2 = LogisticODESimulation(_make_config(r=1.0, K=100.0)).max_growth_rate
        assert rate_2 == pytest.approx(2 * rate_1)


# ---------------------------------------------------------------------------
# Fixed points tests
# ---------------------------------------------------------------------------

class TestLogisticODEFixedPoints:
    """Tests for fixed point analysis."""

    def test_two_fixed_points(self):
        """Should have exactly two fixed points."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        fps = sim.find_fixed_points()
        assert len(fps) == 2

    def test_fixed_point_values(self):
        """Fixed points should be at N=0 and N=K."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        fps = sim.find_fixed_points()
        N_values = sorted([fp["N"] for fp in fps])
        np.testing.assert_allclose(N_values, [0.0, 100.0])

    def test_origin_unstable(self):
        """N=0 should be unstable for r > 0."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        fps = sim.find_fixed_points()
        origin = [fp for fp in fps if fp["N"] == 0.0][0]
        assert origin["stability"] == "unstable"
        assert origin["eigenvalue"] == pytest.approx(1.0)

    def test_K_stable(self):
        """N=K should be stable for r > 0."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        fps = sim.find_fixed_points()
        at_K = [fp for fp in fps if fp["N"] == 100.0][0]
        assert at_K["stability"] == "stable"
        assert at_K["eigenvalue"] == pytest.approx(-1.0)

    def test_derivatives_zero_at_fixed_points(self):
        """dN/dt should be zero at each fixed point."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        for fp in sim.find_fixed_points():
            dy = sim._derivatives(np.array([fp["N"]]))
            np.testing.assert_allclose(dy, [0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Conservation / bounds tests
# ---------------------------------------------------------------------------

class TestLogisticODEConservation:
    """Tests for population bounds and positivity."""

    def test_population_always_positive(self):
        """Population should remain positive at all times."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=0.01, dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"Population went negative: {state[0]}"

    def test_population_bounded_by_K(self):
        """Population should not exceed carrying capacity (from below)."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0, dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] <= 100.0 + 1e-6, (
                f"Population exceeded K: {state[0]} > 100.0"
            )

    def test_no_nan_long_run(self):
        """No NaN or Inf after many steps."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0, dt=0.1))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"

    def test_monotonic_growth_below_K(self):
        """Population should monotonically increase when N_0 < K."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0, dt=0.1))
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
        sim = LogisticODESimulation(_make_config(r=1.0, K=50.0, N_0=80.0, dt=0.01))
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

class TestLogisticODEAnalytical:
    """Tests for analytical solution and derived formulas."""

    def test_analytical_solution_array(self):
        """Analytical solution should accept array input."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0))
        times = np.array([0.0, 1.0, 5.0, 10.0, 100.0])
        result = sim.analytical_solution(times)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))
        # Should be monotonically increasing
        assert np.all(np.diff(result) >= 0)

    def test_analytical_solution_monotonic(self):
        """Analytical solution should be monotonically increasing (N_0 < K)."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0))
        times = np.linspace(0, 50, 100)
        result = sim.analytical_solution(times)
        assert np.all(np.diff(result) >= 0)

    def test_analytical_solution_sigmoid_shape(self):
        """Analytical solution should have sigmoid shape: accelerating then decelerating."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0, N_0=1.0))
        times = np.linspace(0, 20, 1000)
        N = sim.analytical_solution(times)
        dN = np.diff(N)
        # Growth rate should first increase then decrease
        max_idx = np.argmax(dN)
        assert 0 < max_idx < len(dN) - 1, "No inflection found in sigmoid"

    def test_growth_rate_at_inflection(self):
        """Growth rate at inflection point should equal r * K / 4."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        N_inf = sim.inflection_point
        rate = sim.growth_rate_at(N_inf)
        assert rate == pytest.approx(sim.max_growth_rate, rel=1e-10)

    def test_growth_rate_at_K_is_zero(self):
        """Growth rate at carrying capacity should be zero."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        assert sim.growth_rate_at(100.0) == pytest.approx(0.0)

    def test_growth_rate_at_zero_is_zero(self):
        """Growth rate at N=0 should return 0."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        assert sim.growth_rate_at(0.0) == pytest.approx(0.0)

    def test_r_sweep(self):
        """r_sweep should return arrays of correct length."""
        sim = LogisticODESimulation(_make_config())
        r_values = np.linspace(0.1, 2.0, 5)
        result = sim.r_sweep(r_values=r_values, n_steps=200)
        assert len(result["r"]) == 5
        assert len(result["final_N"]) == 5
        assert len(result["inflection_time"]) == 5


# ---------------------------------------------------------------------------
# Comparison with Gompertz (distinguish logistic from Gompertz)
# ---------------------------------------------------------------------------

class TestLogisticVsGompertz:
    """Tests that distinguish the logistic model from the Gompertz model."""

    def test_inflection_is_K_over_2_not_K_over_e(self):
        """Logistic inflection is K/2, not K/e (Gompertz)."""
        sim = LogisticODESimulation(_make_config(K=100.0))
        assert sim.inflection_point == pytest.approx(50.0)
        # Gompertz would give K/e ~ 36.79
        assert sim.inflection_point != pytest.approx(100.0 / np.e, abs=1.0)

    def test_max_growth_rate_is_rK_over_4_not_rK_over_e(self):
        """Logistic max rate is rK/4, not rK/e (Gompertz)."""
        sim = LogisticODESimulation(_make_config(r=1.0, K=100.0))
        assert sim.max_growth_rate == pytest.approx(25.0)
        # Gompertz would give rK/e ~ 36.79
        assert sim.max_growth_rate != pytest.approx(100.0 / np.e, abs=1.0)

    def test_symmetric_sigmoid(self):
        """Logistic growth curve should be symmetric about the inflection point.

        The logistic curve N(t) is symmetric about (t_inf, K/2):
        N(t_inf + dt) + N(t_inf - dt) = K for all dt.
        """
        sim = LogisticODESimulation(
            _make_config(r=1.0, K=100.0, N_0=1.0)
        )
        t_inf = sim.inflection_time()
        offsets = np.array([0.5, 1.0, 2.0, 3.0])

        for dt_val in offsets:
            N_plus = sim.analytical_solution(t_inf + dt_val)
            N_minus = sim.analytical_solution(t_inf - dt_val)
            assert N_plus + N_minus == pytest.approx(100.0, abs=0.01), (
                f"Not symmetric at offset {dt_val}: "
                f"N+ = {N_plus:.4f}, N- = {N_minus:.4f}, sum = {N_plus + N_minus:.4f}"
            )


# ---------------------------------------------------------------------------
# Rediscovery data generation tests
# ---------------------------------------------------------------------------

class TestLogisticODERediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_growth_data(self):
        """Growth data should have correct structure and sizes."""
        from simulating_anything.rediscovery.logistic_ode import generate_growth_data
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
        from simulating_anything.rediscovery.logistic_ode import generate_ode_data
        data = generate_ode_data(r=1.0, K=100.0, N_0=1.0, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 1)
        assert len(data["time"]) == 501
        assert data["r"] == 1.0
        assert data["K"] == 100.0
        assert data["N_0"] == 1.0

    def test_generate_inflection_data(self):
        """Inflection data should have measured values close to K/2."""
        from simulating_anything.rediscovery.logistic_ode import (
            generate_inflection_data,
        )
        data = generate_inflection_data(
            n_K=10, r=1.0, N_0=1.0, dt=0.05, n_steps=2000,
        )
        assert len(data["K"]) == 10
        assert len(data["inflection_N_theory"]) == 10
        assert len(data["inflection_N_measured"]) == 10
        # Theory values should be K/2
        np.testing.assert_allclose(
            data["inflection_N_theory"],
            data["K"] / 2.0,
            rtol=1e-10,
        )

    def test_inflection_data_correlation(self):
        """Measured inflection points should correlate with K/2 theory."""
        from simulating_anything.rediscovery.logistic_ode import (
            generate_inflection_data,
        )
        data = generate_inflection_data(
            n_K=20, r=1.0, N_0=1.0, dt=0.05, n_steps=2000,
        )
        correlation = np.corrcoef(
            data["inflection_N_theory"],
            data["inflection_N_measured"],
        )[0, 1]
        assert correlation > 0.95

    def test_growth_data_sweep_ranges(self):
        """Growth data should cover specified parameter ranges."""
        from simulating_anything.rediscovery.logistic_ode import generate_growth_data
        data = generate_growth_data(n_samples=100, dt=0.1, n_steps=200)
        assert data["r"].min() < 0.2
        assert data["r"].max() > 2.5
        assert data["K"].min() < 20.0
        assert data["K"].max() > 150.0

    def test_ode_data_trajectory_grows(self):
        """ODE trajectory should show growth from N_0 toward K."""
        from simulating_anything.rediscovery.logistic_ode import generate_ode_data
        data = generate_ode_data(r=1.0, K=100.0, N_0=1.0, n_steps=1000, dt=0.01)
        N = data["N"]
        assert N[-1] > N[0], "Population should grow over time"
        assert N[-1] > 10.0, "Population should be well above N_0"
