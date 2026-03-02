"""Tests for the Ivlev predator-prey simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.ivlev_predator_prey import IvlevPredatorPrey
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 1.0,
    K: float = 10.0,
    a: float = 1.5,
    b: float = 0.3,
    e: float = 0.5,
    d: float = 0.3,
    N_0: float = 5.0,
    P_0: float = 2.0,
    dt: float = 0.01,
    n_steps: int = 500,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.IVLEV_PREDATOR_PREY,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "b": b, "e": e, "d": d,
            "N_0": N_0, "P_0": P_0,
        },
    )


class TestIvlevCreation:
    """Tests for simulation creation and initialization."""

    def test_creation_default_params(self):
        sim = IvlevPredatorPrey(_make_config())
        assert sim.r == 1.0
        assert sim.K == 10.0
        assert sim.a == 1.5
        assert sim.b == 0.3
        assert sim.e == 0.5
        assert sim.d == 0.3

    def test_creation_custom_params(self):
        sim = IvlevPredatorPrey(_make_config(r=2.0, K=20.0, a=2.0))
        assert sim.r == 2.0
        assert sim.K == 20.0
        assert sim.a == 2.0

    def test_initial_state_shape(self):
        sim = IvlevPredatorPrey(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        sim = IvlevPredatorPrey(_make_config(N_0=3.0, P_0=1.5))
        state = sim.reset()
        np.testing.assert_allclose(state, [3.0, 1.5])

    def test_observe_shape(self):
        sim = IvlevPredatorPrey(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)


class TestIvlevProperties:
    """Tests for population property accessors."""

    def test_prey_population(self):
        sim = IvlevPredatorPrey(_make_config(N_0=5.0, P_0=2.0))
        sim.reset()
        assert sim.prey_population == pytest.approx(5.0)

    def test_predator_population(self):
        sim = IvlevPredatorPrey(_make_config(N_0=5.0, P_0=2.0))
        sim.reset()
        assert sim.predator_population == pytest.approx(2.0)

    def test_total_population(self):
        sim = IvlevPredatorPrey(_make_config(N_0=5.0, P_0=2.0))
        sim.reset()
        assert sim.total_population == pytest.approx(7.0)

    def test_properties_before_reset(self):
        sim = IvlevPredatorPrey(_make_config())
        assert sim.prey_population == 0.0
        assert sim.predator_population == 0.0
        assert sim.total_population == 0.0


class TestIvlevFunctionalResponse:
    """Tests for the Ivlev functional response f(N) = a*(1-exp(-b*N))."""

    def test_response_at_zero(self):
        """f(0) = 0: no predation when prey absent."""
        sim = IvlevPredatorPrey(_make_config())
        assert sim.functional_response(0.0) == pytest.approx(0.0, abs=1e-15)

    def test_response_saturation(self):
        """f(N) -> a as N -> infinity."""
        sim = IvlevPredatorPrey(_make_config(a=1.5, b=0.3))
        # At large N, exp(-b*N) ~ 0
        f_large = sim.functional_response(100.0)
        assert f_large == pytest.approx(1.5, abs=1e-10)

    def test_response_initial_slope(self):
        """f'(0) = a*b: initial capture efficiency."""
        sim = IvlevPredatorPrey(_make_config(a=1.5, b=0.3))
        dN = 1e-6
        f_prime = (sim.functional_response(dN) - sim.functional_response(0.0)) / dN
        expected = 1.5 * 0.3  # a*b
        assert f_prime == pytest.approx(expected, rel=1e-4)

    def test_response_monotonically_increasing(self):
        """f(N) should be monotonically increasing for N > 0."""
        sim = IvlevPredatorPrey(_make_config())
        N_values = np.linspace(0.01, 50.0, 100)
        f_values = sim.functional_response(N_values)
        diffs = np.diff(f_values)
        assert np.all(diffs > 0), "Functional response should be monotonically increasing"

    def test_response_concave(self):
        """f(N) should be concave (f'' < 0) for all N > 0."""
        sim = IvlevPredatorPrey(_make_config())
        N_values = np.linspace(0.1, 50.0, 100)
        f_values = sim.functional_response(N_values)
        # Second differences should be negative (concave)
        second_diffs = np.diff(f_values, n=2)
        assert np.all(second_diffs < 0), "Functional response should be concave"

    def test_response_array_input(self):
        """Should accept array input."""
        sim = IvlevPredatorPrey(_make_config(a=1.5, b=0.3))
        N_arr = np.array([0.0, 1.0, 5.0, 10.0])
        result = sim.functional_response(N_arr)
        assert result.shape == (4,)
        expected = 1.5 * (1.0 - np.exp(-0.3 * N_arr))
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_response_differs_from_holling_ii(self):
        """Ivlev and Holling II should give different values at same N."""
        sim = IvlevPredatorPrey(_make_config(a=1.5, b=0.3))
        N_test = 5.0
        ivlev = sim.functional_response(N_test)
        # Holling II: a*N/(1+a*h*N) with a=1.5, h=1/b for comparison
        holling = 1.5 * N_test / (1.0 + 1.5 * (1.0 / 0.3) * N_test)
        assert ivlev != pytest.approx(holling, abs=0.01), (
            "Ivlev should differ from Holling Type II"
        )


class TestIvlevEquilibrium:
    """Tests for coexistence equilibrium computation."""

    def test_coexistence_equilibrium_exists(self):
        """With default params, coexistence equilibrium should exist."""
        sim = IvlevPredatorPrey(_make_config())
        N_star, P_star = sim.coexistence_equilibrium()
        assert N_star > 0
        assert P_star > 0

    def test_equilibrium_N_star_formula(self):
        """N* = -ln(1 - d/(e*a)) / b."""
        sim = IvlevPredatorPrey(_make_config(a=1.5, b=0.3, e=0.5, d=0.3))
        N_star, _ = sim.coexistence_equilibrium()
        ratio = 0.3 / (0.5 * 1.5)  # d/(e*a) = 0.4
        expected_N = -np.log(1.0 - ratio) / 0.3
        assert N_star == pytest.approx(expected_N, rel=1e-10)

    def test_derivatives_zero_at_equilibrium(self):
        """At coexistence equilibrium, derivatives should be near zero."""
        sim = IvlevPredatorPrey(_make_config())
        N_star, P_star = sim.coexistence_equilibrium()
        state = np.array([N_star, P_star])
        dy = sim._derivatives(state)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-10)

    def test_no_coexistence_when_predator_unsustainable(self):
        """When d >= e*a, no coexistence equilibrium exists."""
        # d/(e*a) = 2.0/(0.5*1.5) = 2.667 >= 1
        sim = IvlevPredatorPrey(_make_config(d=2.0))
        with pytest.raises(ValueError, match="cannot sustain"):
            sim.coexistence_equilibrium()

    def test_no_coexistence_when_N_star_exceeds_K(self):
        """When N* >= K, no coexistence equilibrium exists."""
        # N* = -ln(1 - d/(e*a)) / b. With d=0.1, e=0.5, a=1.5, b=0.3:
        # N* = -ln(1-0.133)/0.3 = 0.476. Need K < 0.476.
        sim = IvlevPredatorPrey(_make_config(K=0.4, d=0.1))
        with pytest.raises(ValueError):
            sim.coexistence_equilibrium()

    def test_equilibrium_N_star_less_than_K(self):
        """N* should be less than K for valid equilibrium."""
        sim = IvlevPredatorPrey(_make_config())
        N_star, _ = sim.coexistence_equilibrium()
        assert N_star < sim.K


class TestIvlevDynamics:
    """Tests for simulation dynamics."""

    def test_step_advances_state(self):
        sim = IvlevPredatorPrey(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_populations_non_negative(self):
        """All populations should remain non-negative."""
        sim = IvlevPredatorPrey(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_bounded(self):
        """Populations should not blow up to infinity."""
        sim = IvlevPredatorPrey(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            assert np.all(state < 200), f"Population blow-up: {state}"

    def test_prey_bounded_by_K(self):
        """Prey population should remain bounded near or below K."""
        sim = IvlevPredatorPrey(_make_config(K=10.0, dt=0.01))
        sim.reset()
        max_N = 0.0
        for _ in range(5000):
            sim.step()
            max_N = max(max_N, sim.observe()[0])
        # Prey may transiently exceed K during oscillations but should be bounded
        assert max_N < 3.0 * sim.K, (
            f"Prey exceeded 3*K: max_N={max_N:.2f}, K={sim.K}"
        )

    def test_predator_extinction_without_prey(self):
        """Without prey, predators should decline toward zero."""
        sim = IvlevPredatorPrey(_make_config(N_0=0.0, P_0=5.0, d=0.3))
        sim.reset()
        # P decays as P_0*exp(-d*t). Need t large enough: exp(-0.3*t) < 0.002
        # => t > ln(500)/0.3 ~ 20.7, so 3000 steps at dt=0.01 = 30 time units
        for _ in range(3000):
            sim.step()
        assert sim.predator_population < 0.01

    def test_prey_grows_without_predator(self):
        """Without predators, prey should grow to carrying capacity."""
        sim = IvlevPredatorPrey(_make_config(N_0=1.0, P_0=0.0, K=10.0))
        sim.reset()
        for _ in range(5000):
            sim.step()
        assert sim.prey_population == pytest.approx(10.0, rel=0.01)

    def test_long_run_stability(self):
        """No NaN or Inf after many steps."""
        sim = IvlevPredatorPrey(_make_config(dt=0.01))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"


class TestIvlevTrajectory:
    """Tests for run() trajectory output."""

    def test_run_trajectory_shape(self):
        sim = IvlevPredatorPrey(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 2)  # n_steps + 1 including initial
        assert len(traj.timestamps) == 201

    def test_reproducibility_with_same_config(self):
        cfg = _make_config(n_steps=100)
        sim1 = IvlevPredatorPrey(cfg)
        traj1 = sim1.run(100)
        sim2 = IvlevPredatorPrey(cfg)
        traj2 = sim2.run(100)
        np.testing.assert_allclose(traj1.states, traj2.states)

    def test_trajectory_non_negative(self):
        """Entire trajectory should have non-negative populations."""
        sim = IvlevPredatorPrey(_make_config(n_steps=2000))
        traj = sim.run(2000)
        assert np.all(traj.states >= 0), "Negative values in trajectory"


class TestIvlevDerivatives:
    """Tests for the ODE right-hand side."""

    def test_prey_growth_alone(self):
        """With P=0, dN/dt = r*N*(1-N/K) (logistic growth)."""
        sim = IvlevPredatorPrey(_make_config(r=1.0, K=10.0))
        state = np.array([5.0, 0.0])
        dy = sim._derivatives(state)
        expected_dN = 1.0 * 5.0 * (1.0 - 5.0 / 10.0)
        assert dy[0] == pytest.approx(expected_dN, rel=1e-10)
        assert dy[1] == pytest.approx(0.0, abs=1e-15)  # No predator growth

    def test_predator_decline_at_zero_prey(self):
        """With N=0, dP/dt = -d*P (exponential decline)."""
        sim = IvlevPredatorPrey(_make_config(d=0.3))
        state = np.array([0.0, 5.0])
        dy = sim._derivatives(state)
        assert dy[0] == pytest.approx(0.0, abs=1e-15)
        assert dy[1] == pytest.approx(-0.3 * 5.0, rel=1e-10)

    def test_prey_at_carrying_capacity(self):
        """At N=K with P=0, dN/dt = 0 (equilibrium)."""
        sim = IvlevPredatorPrey(_make_config(K=10.0))
        state = np.array([10.0, 0.0])
        dy = sim._derivatives(state)
        assert dy[0] == pytest.approx(0.0, abs=1e-10)

    def test_functional_response_in_derivatives(self):
        """Verify Ivlev functional response is used in predation term."""
        sim = IvlevPredatorPrey(_make_config(a=1.5, b=0.3, e=0.5, d=0.3))
        N, P = 5.0, 2.0
        state = np.array([N, P])
        dy = sim._derivatives(state)

        f_N = 1.5 * (1.0 - np.exp(-0.3 * 5.0))
        expected_dN = 1.0 * N * (1.0 - N / 10.0) - f_N * P
        expected_dP = 0.5 * f_N * P - 0.3 * P

        assert dy[0] == pytest.approx(expected_dN, rel=1e-10)
        assert dy[1] == pytest.approx(expected_dP, rel=1e-10)


class TestIvlevRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.ivlev_predator_prey import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["r"] == 1.0
        assert data["K"] == 10.0
        assert data["a"] == 1.5
        assert data["b"] == 0.3

    def test_functional_response_data_generation(self):
        from simulating_anything.rediscovery.ivlev_predator_prey import (
            generate_functional_response_data,
        )
        data = generate_functional_response_data(n_points=50, N_max=20.0)
        assert len(data["N"]) == 50
        assert len(data["f_N"]) == 50
        assert data["a"] == 1.5
        assert data["b"] == 0.3
        # f(N) should be positive and bounded by a
        assert np.all(data["f_N"] > 0)
        assert np.all(data["f_N"] <= 1.5 + 1e-10)

    def test_stability_sweep_data_generation(self):
        from simulating_anything.rediscovery.ivlev_predator_prey import (
            generate_stability_sweep_data,
        )
        data = generate_stability_sweep_data(
            n_K_values=5, n_steps=500, dt=0.01,
        )
        assert len(data["K"]) == 5
        assert len(data["N_avg"]) == 5
        assert len(data["N_amplitude"]) == 5
        assert len(data["P_amplitude"]) == 5

    def test_trajectory_data_non_negative(self):
        from simulating_anything.rediscovery.ivlev_predator_prey import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=1000, dt=0.01)
        assert np.all(data["states"] >= 0), "Trajectory has negative values"

    def test_functional_response_monotonicity(self):
        from simulating_anything.rediscovery.ivlev_predator_prey import (
            generate_functional_response_data,
        )
        data = generate_functional_response_data(n_points=100, N_max=30.0)
        diffs = np.diff(data["f_N"])
        assert np.all(diffs > 0), "Functional response should be monotonically increasing"

    def test_stability_sweep_averages_positive(self):
        from simulating_anything.rediscovery.ivlev_predator_prey import (
            generate_stability_sweep_data,
        )
        data = generate_stability_sweep_data(
            n_K_values=5, K_min=5.0, K_max=20.0,
            n_steps=2000, dt=0.01,
        )
        assert np.all(data["N_avg"] >= 0)
        assert np.all(data["P_avg"] >= 0)
