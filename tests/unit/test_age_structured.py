"""Tests for the age-structured predator-prey simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.age_structured import AgeStructuredSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    b_N: float = 2.0,
    mu_N: float = 0.5,
    m_j_N: float = 0.1,
    d_N: float = 0.1,
    a: float = 0.02,
    h: float = 0.01,
    e: float = 0.5,
    mu_P: float = 0.3,
    m_j_P: float = 0.15,
    d_P: float = 0.2,
    N_j_0: float = 30.0,
    N_a_0: float = 50.0,
    P_j_0: float = 5.0,
    P_a_0: float = 10.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.AGE_STRUCTURED,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "b_N": b_N, "mu_N": mu_N, "m_j_N": m_j_N, "d_N": d_N,
            "a": a, "h": h, "e": e,
            "mu_P": mu_P, "m_j_P": m_j_P, "d_P": d_P,
            "N_j_0": N_j_0, "N_a_0": N_a_0, "P_j_0": P_j_0, "P_a_0": P_a_0,
        },
    )


class TestAgeStructuredCreation:
    """Tests for simulation creation and parameter handling."""

    def test_creation_default_params(self):
        sim = AgeStructuredSimulation(_make_config())
        assert sim.b_N == 2.0
        assert sim.mu_N == 0.5
        assert sim.m_j_N == 0.1
        assert sim.d_N == 0.1
        assert sim.a == 0.02
        assert sim.h == 0.01
        assert sim.e == 0.5
        assert sim.mu_P == 0.3
        assert sim.m_j_P == 0.15
        assert sim.d_P == 0.2

    def test_creation_custom_params(self):
        sim = AgeStructuredSimulation(_make_config(b_N=3.0, mu_P=0.6))
        assert sim.b_N == 3.0
        assert sim.mu_P == 0.6

    def test_initial_state_shape(self):
        sim = AgeStructuredSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (4,)

    def test_initial_state_values(self):
        sim = AgeStructuredSimulation(
            _make_config(N_j_0=20.0, N_a_0=40.0, P_j_0=3.0, P_a_0=8.0)
        )
        state = sim.reset()
        np.testing.assert_allclose(state, [20.0, 40.0, 3.0, 8.0])

    def test_observe_shape(self):
        sim = AgeStructuredSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_observe_matches_reset(self):
        sim = AgeStructuredSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_allclose(obs, state)

    def test_domain_enum(self):
        config = _make_config()
        assert config.domain == Domain.AGE_STRUCTURED
        assert config.domain.value == "age_structured"


class TestAgeStructuredDynamics:
    """Tests for simulation dynamics and numerical integration."""

    def test_step_advances_state(self):
        sim = AgeStructuredSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_populations_non_negative(self):
        """All populations should remain non-negative."""
        sim = AgeStructuredSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_non_negative_small_initial(self):
        """Non-negativity even with small initial populations."""
        sim = AgeStructuredSimulation(
            _make_config(N_j_0=0.1, N_a_0=0.1, P_j_0=0.1, P_a_0=0.1, dt=0.005)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_finite(self):
        """Populations should remain finite (no NaN/Inf)."""
        sim = AgeStructuredSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_stability_long_run(self):
        """No NaN or Inf after 20000 steps."""
        sim = AgeStructuredSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_total_population_property(self):
        sim = AgeStructuredSimulation(
            _make_config(N_j_0=30.0, N_a_0=50.0, P_j_0=5.0, P_a_0=10.0)
        )
        sim.reset()
        assert sim.total_population == pytest.approx(95.0)

    def test_total_prey_property(self):
        sim = AgeStructuredSimulation(_make_config(N_j_0=30.0, N_a_0=50.0))
        sim.reset()
        assert sim.total_prey == pytest.approx(80.0)

    def test_total_predator_property(self):
        sim = AgeStructuredSimulation(_make_config(P_j_0=5.0, P_a_0=10.0))
        sim.reset()
        assert sim.total_predator == pytest.approx(15.0)


class TestAgeStructuredAgeRatios:
    """Tests for age ratio computations and properties."""

    def test_prey_age_ratio_initial(self):
        sim = AgeStructuredSimulation(_make_config(N_j_0=30.0, N_a_0=50.0))
        sim.reset()
        assert sim.prey_age_ratio == pytest.approx(30.0 / 50.0)

    def test_predator_age_ratio_initial(self):
        sim = AgeStructuredSimulation(_make_config(P_j_0=5.0, P_a_0=10.0))
        sim.reset()
        assert sim.predator_age_ratio == pytest.approx(5.0 / 10.0)

    def test_prey_equilibrium_ratio_formula(self):
        """N_j/N_a = b_N / (mu_N + m_j_N) at predator-free equilibrium."""
        sim = AgeStructuredSimulation(_make_config(b_N=2.0, mu_N=0.5, m_j_N=0.1))
        expected = 2.0 / (0.5 + 0.1)
        assert sim.prey_equilibrium_ratio() == pytest.approx(expected)

    def test_predator_equilibrium_ratio_formula(self):
        """P_j/P_a = d_P / mu_P at equilibrium."""
        sim = AgeStructuredSimulation(_make_config(d_P=0.2, mu_P=0.3))
        expected = 0.2 / 0.3
        assert sim.predator_equilibrium_ratio() == pytest.approx(expected)

    def test_prey_ratio_zero_adult(self):
        """Prey age ratio should be 0 when adult prey is zero."""
        sim = AgeStructuredSimulation(_make_config(N_j_0=10.0, N_a_0=0.0))
        sim.reset()
        assert sim.prey_age_ratio == 0.0

    def test_predator_ratio_zero_adult(self):
        """Predator age ratio should be 0 when adult predator is zero."""
        sim = AgeStructuredSimulation(_make_config(P_j_0=5.0, P_a_0=0.0))
        sim.reset()
        assert sim.predator_age_ratio == 0.0


class TestAgeStructuredEquilibrium:
    """Tests for equilibrium and stability properties."""

    def test_is_coexisting_true(self):
        sim = AgeStructuredSimulation(_make_config())
        sim.reset()
        assert sim.is_coexisting

    def test_is_coexisting_false_when_extinct(self):
        sim = AgeStructuredSimulation(_make_config(P_j_0=0.0, P_a_0=0.0))
        sim.reset()
        assert not sim.is_coexisting

    def test_prey_net_growth_positive(self):
        """Default parameters should give positive prey growth rate."""
        sim = AgeStructuredSimulation(_make_config())
        rate = sim.prey_net_growth_rate()
        # b_N*mu_N = 2.0*0.5 = 1.0 > (mu_N+m_j_N)*d_N = 0.6*0.1 = 0.06
        # Prey should grow in absence of predators
        assert rate > 0, f"Prey growth rate should be positive: {rate}"

    def test_prey_net_growth_negative_when_low_birth(self):
        """Very low birth rate should give negative growth."""
        sim = AgeStructuredSimulation(_make_config(b_N=0.01, d_N=0.5))
        rate = sim.prey_net_growth_rate()
        # b_N*mu_N = 0.01*0.5 = 0.005 < (mu_N+m_j_N)*d_N = 0.6*0.5 = 0.3
        # det = 0.6*0.5 - 0.01*0.5 = 0.295 > 0 and tr < 0 => both eigenvalues negative
        assert rate < 0, f"Prey should decline with low birth: {rate}"

    def test_predator_invasion_rate_positive(self):
        """Predator should be able to invade at large prey density."""
        sim = AgeStructuredSimulation(_make_config())
        # At N_a = 100 (large), functional response ~ a/h = 2.0
        rate = sim.predator_invasion_rate(100.0)
        # e*f*mu_P should exceed (mu_P+m_j_P)*d_P for invasion
        assert rate > 0, f"Predator should invade at high prey: {rate}"

    def test_predator_invasion_rate_negative_no_prey(self):
        """Predator cannot invade when there is no prey."""
        sim = AgeStructuredSimulation(_make_config())
        rate = sim.predator_invasion_rate(0.0)
        # With N_a=0, f(0)=0, so predator matrix has eigenvalues -d_P and -(mu_P+m_j_P)
        # dominant eigenvalue = max(-0.2, -0.45) = -0.2
        assert rate < 0, f"Predator should not invade with no prey: {rate}"
        np.testing.assert_allclose(rate, -0.2, atol=1e-10)

    def test_prey_age_ratio_converges_no_predators(self):
        """Without predators, prey age ratio converges to asymptotic eigenvector ratio."""
        sim = AgeStructuredSimulation(
            _make_config(a=0.0, N_j_0=10.0, N_a_0=10.0, dt=0.005)
        )
        sim.reset()

        # Run long enough for transient to decay, but not so long prey blows up
        # With a=0 prey grows exponentially, so use moderate steps
        for _ in range(2000):
            sim.step()

        measured_ratio = sim.prey_age_ratio
        expected_ratio = sim.prey_asymptotic_ratio()
        # The ratio converges on the eigenmode timescale
        np.testing.assert_allclose(measured_ratio, expected_ratio, rtol=0.1)


class TestAgeStructuredFunctionalResponse:
    """Tests for Holling Type II functional response."""

    def test_functional_response_zero(self):
        sim = AgeStructuredSimulation(_make_config())
        assert sim.functional_response(0.0) == 0.0

    def test_functional_response_increasing(self):
        sim = AgeStructuredSimulation(_make_config())
        fr1 = sim.functional_response(1.0)
        fr10 = sim.functional_response(10.0)
        fr100 = sim.functional_response(100.0)
        assert 0 < fr1 < fr10 < fr100

    def test_functional_response_saturation(self):
        """At very large N_a, f(N_a) -> a/h."""
        sim = AgeStructuredSimulation(_make_config())
        fr_large = sim.functional_response(1e8)
        expected = sim.a / sim.h
        np.testing.assert_allclose(fr_large, expected, rtol=0.001)

    def test_functional_response_formula(self):
        """Verify f(N_a) = a * N_a / (1 + h * N_a)."""
        sim = AgeStructuredSimulation(_make_config(a=0.05, h=0.02))
        N_a = 25.0
        expected = 0.05 * 25.0 / (1.0 + 0.02 * 25.0)
        assert sim.functional_response(N_a) == pytest.approx(expected)


class TestAgeStructuredSpecialCases:
    """Tests for limiting cases and special dynamics."""

    def test_no_predators_prey_grows(self):
        """With zero predator initial conditions, prey should grow."""
        sim = AgeStructuredSimulation(
            _make_config(P_j_0=0.0, P_a_0=0.0, N_j_0=10.0, N_a_0=10.0, dt=0.005)
        )
        sim.reset()
        initial_total = sim.total_prey
        for _ in range(500):
            sim.step()
        # Prey should grow (positive net growth rate)
        assert sim.total_prey > initial_total

    def test_no_predators_predator_stays_zero(self):
        """With zero predator ICs and no prey-to-predator coupling, P stays zero."""
        sim = AgeStructuredSimulation(
            _make_config(P_j_0=0.0, P_a_0=0.0, dt=0.005)
        )
        sim.reset()
        for _ in range(1000):
            sim.step()
            state = sim.observe()
            assert state[2] == pytest.approx(0.0, abs=1e-12)
            assert state[3] == pytest.approx(0.0, abs=1e-12)

    def test_no_prey_predator_decays(self):
        """With zero prey, predators should decay to zero."""
        sim = AgeStructuredSimulation(
            _make_config(N_j_0=0.0, N_a_0=0.0, P_j_0=5.0, P_a_0=10.0, dt=0.005)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
        state = sim.observe()
        # Prey stays at zero
        assert state[0] == pytest.approx(0.0, abs=1e-10)
        assert state[1] == pytest.approx(0.0, abs=1e-10)
        # Predator decays
        assert state[2] < 5.0
        assert state[3] < 10.0

    def test_zero_maturation_juveniles_accumulate(self):
        """With mu_N = 0, juvenile prey cannot become adults."""
        sim = AgeStructuredSimulation(
            _make_config(mu_N=0.0, a=0.0, N_j_0=0.0, N_a_0=10.0, dt=0.005)
        )
        sim.reset()
        for _ in range(500):
            sim.step()
        state = sim.observe()
        # Adults produce juveniles via b_N, but juveniles cannot mature
        # Juvenile prey should accumulate (births - m_j_N * N_j)
        assert state[0] > 0, "Juveniles should accumulate from births"

    def test_high_maturation_fast_turnover(self):
        """Very high maturation rate means juveniles quickly become adults."""
        sim = AgeStructuredSimulation(
            _make_config(mu_N=10.0, a=0.0, N_j_0=50.0, N_a_0=10.0, dt=0.001)
        )
        sim.reset()
        for _ in range(2000):
            sim.step()
        state = sim.observe()
        # With fast maturation, juvenile/adult ratio should be small
        ratio = state[0] / state[1] if state[1] > 1e-10 else float("inf")
        expected = sim.prey_equilibrium_ratio()
        # The theoretical ratio = b_N / (mu_N + m_j_N) = 2.0 / 10.1 ~ 0.198
        assert ratio < 1.0, f"High maturation should give low juv/adult ratio: {ratio}"
        np.testing.assert_allclose(ratio, expected, rtol=0.3)


class TestAgeStructuredTrajectory:
    """Tests for trajectory collection and reproducibility."""

    def test_run_trajectory_shape(self):
        sim = AgeStructuredSimulation(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 4)
        assert len(traj.timestamps) == 201

    def test_run_trajectory_timestamps(self):
        sim = AgeStructuredSimulation(_make_config(dt=0.01, n_steps=100))
        traj = sim.run(n_steps=100)
        expected_times = np.arange(101) * 0.01
        np.testing.assert_allclose(traj.timestamps, expected_times)

    def test_run_trajectory_non_negative(self):
        sim = AgeStructuredSimulation(_make_config(n_steps=1000))
        traj = sim.run(n_steps=1000)
        assert np.all(traj.states >= 0)
        assert np.all(np.isfinite(traj.states))

    def test_reproducibility_with_same_config(self):
        cfg = _make_config(n_steps=100)
        sim1 = AgeStructuredSimulation(cfg)
        traj1 = sim1.run(100)
        sim2 = AgeStructuredSimulation(cfg)
        traj2 = sim2.run(100)
        np.testing.assert_allclose(traj1.states, traj2.states)

    def test_deterministic_step_by_step(self):
        """Two separate simulations with same config produce identical results."""
        config = _make_config()
        sim1 = AgeStructuredSimulation(config)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe()

        sim2 = AgeStructuredSimulation(config)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe()

        np.testing.assert_allclose(state1, state2, atol=1e-12)

    def test_trajectory_data_type(self):
        from simulating_anything.types.trajectory import TrajectoryData
        sim = AgeStructuredSimulation(_make_config(n_steps=50))
        traj = sim.run(n_steps=50)
        assert isinstance(traj, TrajectoryData)
        assert traj.states.dtype == np.float64

    def test_trajectory_parameters(self):
        sim = AgeStructuredSimulation(_make_config(b_N=3.0, n_steps=10))
        traj = sim.run(n_steps=10)
        assert traj.parameters["b_N"] == 3.0


class TestAgeStructuredMaturationSweep:
    """Tests for maturation rate sweep functionality."""

    def test_maturation_sweep_shape(self):
        sim = AgeStructuredSimulation(_make_config())
        sim.reset()
        mu_N_vals = np.array([0.2, 0.5, 1.0])
        result = sim.maturation_sweep(mu_N_vals, n_steps=3000, n_transient=1000)
        assert len(result["mu_N"]) == 3
        assert len(result["N_j_avg"]) == 3
        assert len(result["prey_ratio"]) == 3
        assert len(result["predator_ratio"]) == 3
        assert len(result["amplitude_N"]) == 3

    def test_maturation_sweep_positive_averages(self):
        sim = AgeStructuredSimulation(_make_config())
        sim.reset()
        mu_N_vals = np.array([0.3, 0.5, 0.8])
        result = sim.maturation_sweep(mu_N_vals, n_steps=5000, n_transient=2000)
        assert np.all(result["N_j_avg"] >= 0)
        assert np.all(result["N_a_avg"] >= 0)
        assert np.all(np.isfinite(result["prey_ratio"]))

    def test_maturation_sweep_ratio_trend(self):
        """Higher maturation rate should decrease juvenile/adult prey ratio."""
        sim = AgeStructuredSimulation(_make_config(a=0.0))  # no predation
        sim.reset()
        mu_N_vals = np.array([0.2, 0.5, 1.0, 2.0])
        result = sim.maturation_sweep(mu_N_vals, n_steps=5000, n_transient=3000)
        ratios = result["prey_ratio"]
        # Higher mu_N -> lower N_j/N_a ratio (juveniles mature faster)
        # Theory: ratio = b_N / (mu_N + m_j_N), monotonically decreasing in mu_N
        for i in range(len(ratios) - 1):
            assert ratios[i] > ratios[i + 1] * 0.8, (
                f"Ratio should decrease: {ratios[i]:.3f} vs {ratios[i+1]:.3f}"
            )


class TestAgeStructuredPeriod:
    """Tests for oscillation period detection."""

    def test_fft_period_constant_signal(self):
        signal = np.ones(1000)
        T = AgeStructuredSimulation._fft_period(signal, 0.01)
        assert T == 0.0

    def test_fft_period_sine_wave(self):
        dt = 0.01
        freq = 0.25
        t = np.arange(10000) * dt
        signal = np.sin(2 * np.pi * freq * t)
        T = AgeStructuredSimulation._fft_period(signal, dt)
        np.testing.assert_allclose(T, 1.0 / freq, rtol=0.05)

    def test_compute_period_returns_finite(self):
        sim = AgeStructuredSimulation(_make_config(dt=0.01))
        T = sim.compute_period(n_steps=10000)
        assert np.isfinite(T), f"Period should be finite: {T}"
        assert T >= 0, f"Period should be non-negative: {T}"


class TestAgeStructuredRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.age_structured import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 4)
        assert len(data["time"]) == 501
        assert data["b_N"] == 2.0
        assert data["mu_N"] == 0.5
        assert len(data["N_j"]) == 501
        assert len(data["N_a"]) == 501
        assert len(data["P_j"]) == 501
        assert len(data["P_a"]) == 501

    def test_trajectory_data_non_negative(self):
        from simulating_anything.rediscovery.age_structured import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=1000, dt=0.01)
        assert np.all(data["states"] >= 0)
        assert np.all(np.isfinite(data["states"]))

    def test_maturation_sweep_data_generation(self):
        from simulating_anything.rediscovery.age_structured import (
            generate_maturation_sweep_data,
        )
        data = generate_maturation_sweep_data(
            n_mu_N=5, mu_N_min=0.2, mu_N_max=1.0,
            n_steps=3000, n_transient=1000,
        )
        assert len(data["mu_N"]) == 5
        assert len(data["prey_ratio"]) == 5
        assert len(data["predator_ratio"]) == 5
        assert len(data["theoretical_prey_ratio"]) == 5
        assert np.all(np.isfinite(data["prey_ratio"]))

    def test_equilibrium_data_generation(self):
        from simulating_anything.rediscovery.age_structured import (
            generate_equilibrium_data,
        )
        data = generate_equilibrium_data(
            n_samples=3, n_steps=3000, n_transient=1000, dt=0.01,
        )
        assert len(data["mu_N"]) == 3
        assert len(data["prey_ratio"]) == 3
        assert len(data["predator_ratio"]) == 3
        assert len(data["N_j_avg"]) == 3
        assert len(data["N_a_avg"]) == 3
        assert np.all(np.isfinite(data["prey_ratio"]))
