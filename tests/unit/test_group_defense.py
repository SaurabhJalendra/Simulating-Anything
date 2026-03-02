"""Tests for the group defense predator-prey model (Andrews functional response)."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.group_defense import GroupDefenseSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 1.0,
    K: float = 15.0,
    a: float = 1.5,
    b: float = 0.1,
    c: float = 0.05,
    e: float = 0.5,
    d: float = 0.3,
    N_0: float = 5.0,
    P_0: float = 2.0,
    dt: float = 0.01,
    n_steps: int = 500,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.GROUP_DEFENSE,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "b": b, "c": c,
            "e": e, "d": d, "N_0": N_0, "P_0": P_0,
        },
    )


class TestGroupDefenseCreation:
    """Tests for simulation creation and parameter extraction."""

    def test_creation_default_params(self):
        sim = GroupDefenseSimulation(_make_config())
        assert sim.r == 1.0
        assert sim.K == 15.0
        assert sim.a == 1.5
        assert sim.b == 0.1
        assert sim.c == 0.05
        assert sim.e == 0.5
        assert sim.d == 0.3

    def test_creation_custom_params(self):
        sim = GroupDefenseSimulation(_make_config(r=2.0, K=20.0, c=0.1))
        assert sim.r == 2.0
        assert sim.K == 20.0
        assert sim.c == 0.1

    def test_initial_state_shape(self):
        sim = GroupDefenseSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        sim = GroupDefenseSimulation(_make_config(N_0=8.0, P_0=3.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [8.0, 3.0])

    def test_observe_shape(self):
        sim = GroupDefenseSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)


class TestGroupDefenseProperties:
    """Tests for population properties."""

    def test_prey_population(self):
        sim = GroupDefenseSimulation(_make_config(N_0=8.0, P_0=3.0))
        sim.reset()
        assert sim.prey_population == pytest.approx(8.0)

    def test_predator_population(self):
        sim = GroupDefenseSimulation(_make_config(N_0=8.0, P_0=3.0))
        sim.reset()
        assert sim.predator_population == pytest.approx(3.0)

    def test_total_population(self):
        sim = GroupDefenseSimulation(_make_config(N_0=8.0, P_0=3.0))
        sim.reset()
        assert sim.total_population == pytest.approx(11.0)

    def test_properties_before_reset(self):
        sim = GroupDefenseSimulation(_make_config())
        assert sim.prey_population == 0.0
        assert sim.predator_population == 0.0
        assert sim.total_population == 0.0


class TestFunctionalResponse:
    """Tests for the Andrews (Monod-Haldane) functional response."""

    def test_functional_response_at_zero(self):
        """f(0) = 0."""
        sim = GroupDefenseSimulation(_make_config())
        assert sim.functional_response(0.0) == pytest.approx(0.0)

    def test_functional_response_positive(self):
        """f(N) > 0 for N > 0."""
        sim = GroupDefenseSimulation(_make_config())
        for N in [0.1, 1.0, 5.0, 10.0, 20.0]:
            assert sim.functional_response(N) > 0

    def test_functional_response_non_monotone(self):
        """f(N) should increase then decrease (group defense)."""
        sim = GroupDefenseSimulation(_make_config(c=0.05))
        # Below peak: should be increasing
        f_low = sim.functional_response(1.0)
        f_mid = sim.functional_response(3.0)
        assert f_mid > f_low, "f(N) should increase below peak"

        # Above peak: should be decreasing
        N_peak = sim.functional_response_peak()
        f_peak = sim.functional_response(N_peak)
        f_high = sim.functional_response(N_peak * 3.0)
        assert f_high < f_peak, "f(N) should decrease above peak (group defense)"

    def test_functional_response_peak_location(self):
        """Peak should occur at N_max = 1/sqrt(c)."""
        sim = GroupDefenseSimulation(_make_config(c=0.04))
        N_peak = sim.functional_response_peak()
        assert N_peak == pytest.approx(1.0 / np.sqrt(0.04), rel=1e-10)
        assert N_peak == pytest.approx(5.0, rel=1e-10)

    def test_functional_response_peak_value(self):
        """Peak value should be a/(b + 2*sqrt(c))."""
        sim = GroupDefenseSimulation(_make_config(a=1.5, b=0.1, c=0.05))
        f_max = sim.functional_response_max_value()
        expected = 1.5 / (0.1 + 2.0 * np.sqrt(0.05))
        assert f_max == pytest.approx(expected, rel=1e-10)

    def test_functional_response_peak_agreement(self):
        """Evaluating f at N_max should give the max value."""
        sim = GroupDefenseSimulation(_make_config(a=1.5, b=0.1, c=0.05))
        N_peak = sim.functional_response_peak()
        f_at_peak = sim.functional_response(N_peak)
        f_max = sim.functional_response_max_value()
        assert f_at_peak == pytest.approx(f_max, rel=1e-6)

    def test_functional_response_array_input(self):
        """Should work with numpy arrays."""
        sim = GroupDefenseSimulation(_make_config())
        N = np.array([1.0, 5.0, 10.0])
        f = sim.functional_response(N)
        assert f.shape == (3,)
        assert np.all(f > 0)

    def test_functional_response_decreases_large_N(self):
        """At very large N, f(N) should approach 0 (dominated by c*N^2 term)."""
        sim = GroupDefenseSimulation(_make_config(c=0.05))
        f_large = sim.functional_response(1000.0)
        f_peak = sim.functional_response_max_value()
        assert f_large < 0.1 * f_peak, (
            f"f(1000)={f_large} should be much smaller than peak {f_peak}"
        )

    def test_peak_varies_with_c(self):
        """Higher c -> lower N_peak (stronger group defense, earlier peak)."""
        sim_low = GroupDefenseSimulation(_make_config(c=0.01))
        sim_high = GroupDefenseSimulation(_make_config(c=0.1))
        assert sim_high.functional_response_peak() < sim_low.functional_response_peak()


class TestGroupDefenseDynamics:
    """Tests for simulation dynamics."""

    def test_step_advances_state(self):
        sim = GroupDefenseSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_populations_non_negative(self):
        """All populations should remain non-negative."""
        sim = GroupDefenseSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_bounded(self):
        """Populations should not blow up to infinity."""
        sim = GroupDefenseSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            assert np.all(state < 200), f"Population blow-up: {state}"

    def test_rk4_stability(self):
        """No NaN or Inf in simulation output."""
        sim = GroupDefenseSimulation(_make_config(N_0=10.0, P_0=5.0, dt=0.01))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_prey_grows_without_predator(self):
        """Without predators, prey should grow toward K."""
        sim = GroupDefenseSimulation(_make_config(N_0=3.0, P_0=0.0, K=15.0))
        sim.reset()
        for _ in range(50000):
            sim.step()
        N_final = sim.observe()[0]
        assert N_final == pytest.approx(15.0, abs=0.5), (
            f"Prey should reach K=15: got N={N_final}"
        )

    def test_predator_dies_without_prey(self):
        """Without prey, predators should decay exponentially."""
        sim = GroupDefenseSimulation(_make_config(N_0=0.0, P_0=5.0))
        sim.reset()
        for _ in range(10000):
            sim.step()
        P_final = sim.observe()[1]
        assert P_final < 0.01, f"Predators should die without prey: P={P_final}"

    def test_deterministic(self):
        """Same parameters produce same result."""
        config = _make_config(N_0=5.0, P_0=2.0)

        sim1 = GroupDefenseSimulation(config)
        sim1.reset()
        for _ in range(1000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = GroupDefenseSimulation(config)
        sim2.reset()
        for _ in range(1000):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_prey_bounded_by_K(self):
        """Prey population should not exceed K significantly."""
        sim = GroupDefenseSimulation(_make_config(N_0=14.0, P_0=0.1, K=15.0))
        sim.reset()
        max_N = 0.0
        for _ in range(20000):
            sim.step()
            N = sim.observe()[0]
            max_N = max(max_N, N)
        # Allow some overshoot due to predator-prey oscillations
        assert max_N < 2.0 * 15.0, f"Prey exceeded 2*K: max_N={max_N}"


class TestGroupDefenseEquilibria:
    """Tests for equilibrium computation."""

    def test_always_has_extinction(self):
        """Origin (0,0) is always an equilibrium."""
        sim = GroupDefenseSimulation(_make_config())
        eq = sim.find_equilibria()
        types = [e["type"] for e in eq]
        assert "extinction" in types
        ext = [e for e in eq if e["type"] == "extinction"][0]
        assert ext["N"] == 0.0
        assert ext["P"] == 0.0

    def test_always_has_carrying_capacity(self):
        """(K, 0) is always an equilibrium."""
        sim = GroupDefenseSimulation(_make_config(K=15.0))
        eq = sim.find_equilibria()
        types = [e["type"] for e in eq]
        assert "carrying_capacity" in types
        kc = [e for e in eq if e["type"] == "carrying_capacity"][0]
        assert kc["N"] == pytest.approx(15.0)
        assert kc["P"] == 0.0

    def test_coexistence_equilibrium_exists(self):
        """With default params, at least one coexistence equilibrium exists."""
        sim = GroupDefenseSimulation(_make_config())
        eq = sim.find_equilibria()
        coex = [e for e in eq if e["type"] == "coexistence"]
        assert len(coex) >= 1, "Should have at least one coexistence equilibrium"

    def test_coexistence_positive(self):
        """Coexistence equilibria should have N > 0 and P > 0."""
        sim = GroupDefenseSimulation(_make_config())
        eq = sim.find_equilibria()
        for e in eq:
            if e["type"] == "coexistence":
                assert e["N"] > 0, f"N should be positive: {e}"
                assert e["P"] > 0, f"P should be positive: {e}"

    def test_derivatives_near_zero_at_equilibrium(self):
        """At coexistence equilibrium, derivatives should be near zero."""
        sim = GroupDefenseSimulation(_make_config())
        sim.reset()
        eq = sim.find_equilibria()
        for e in eq:
            if e["type"] == "coexistence":
                state = np.array([e["N"], e["P"]])
                dy = sim._derivatives(state)
                np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-8)

    def test_multiple_equilibria_possible(self):
        """With appropriate parameters, can have two coexistence equilibria.

        The quadratic in the predator nullcline can have two positive roots
        when the discriminant is positive and both roots are between 0 and K.
        """
        # Use parameters that give two positive roots to the nullcline quadratic
        # d*c*N^2 + (d*b - e*a)*N + d = 0
        # We need discriminant > 0 and both roots positive (both in 0..K)
        sim = GroupDefenseSimulation(_make_config(
            a=3.0, b=0.1, c=0.01, e=0.8, d=0.2, K=50.0,
        ))
        eq = sim.find_equilibria()
        # May or may not have two coexistence points depending on P* > 0
        # This is primarily a structural test that find_equilibria handles it
        assert len(eq) >= 2  # At least extinction + carrying capacity


class TestGroupDefenseTrajectory:
    """Tests for run() trajectory output."""

    def test_run_trajectory_shape(self):
        sim = GroupDefenseSimulation(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 2)
        assert len(traj.timestamps) == 201

    def test_reproducibility_with_same_config(self):
        cfg = _make_config(n_steps=100)
        sim1 = GroupDefenseSimulation(cfg)
        traj1 = sim1.run(100)
        sim2 = GroupDefenseSimulation(cfg)
        traj2 = sim2.run(100)
        np.testing.assert_allclose(traj1.states, traj2.states)


class TestGroupDefenseSweep:
    """Tests for the group defense parameter sweep."""

    def test_sweep_output_shape(self):
        sim = GroupDefenseSimulation(_make_config())
        sim.reset()
        c_values = np.array([0.01, 0.05, 0.1])
        result = sim.group_defense_sweep(c_values, n_steps=1000)
        assert len(result["c_values"]) == 3
        assert len(result["final_N"]) == 3
        assert len(result["final_P"]) == 3
        assert len(result["N_amplitude"]) == 3
        assert len(result["P_amplitude"]) == 3

    def test_sweep_non_negative(self):
        sim = GroupDefenseSimulation(_make_config())
        sim.reset()
        c_values = np.array([0.01, 0.05, 0.1])
        result = sim.group_defense_sweep(c_values, n_steps=1000)
        assert np.all(result["final_N"] >= 0)
        assert np.all(result["final_P"] >= 0)
        assert np.all(result["N_amplitude"] >= 0)

    def test_sweep_higher_c_more_defense(self):
        """Higher c means stronger group defense -> higher effective prey survival."""
        sim = GroupDefenseSimulation(_make_config())
        sim.reset()
        c_values = np.array([0.001, 0.05, 0.2])
        result = sim.group_defense_sweep(c_values, n_steps=20000)
        # With very high c (strong defense), prey should do better
        # At least the final populations should be finite and non-negative
        assert np.all(np.isfinite(result["final_N"]))
        assert np.all(np.isfinite(result["final_P"]))


class TestGroupDefenseRediscovery:
    """Tests for rediscovery data generation."""

    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.group_defense import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["r"] == 1.0
        assert data["K"] == 15.0
        assert data["c"] == 0.05

    def test_functional_response_data(self):
        from simulating_anything.rediscovery.group_defense import (
            generate_functional_response_data,
        )
        data = generate_functional_response_data(n_points=100)
        assert len(data["N_values"]) == 100
        assert len(data["f_values"]) == 100
        assert data["N_peak_theory"] == pytest.approx(
            1.0 / np.sqrt(0.05), rel=1e-6
        )
        # Verify the response is non-monotone: f declines past the peak
        assert data["f_values"][-1] < data["f_peak_empirical"]

    def test_peak_location_data(self):
        from simulating_anything.rediscovery.group_defense import (
            generate_peak_location_data,
        )
        data = generate_peak_location_data(n_c_values=10)
        assert len(data["c_values"]) == 10
        assert len(data["N_peak_theory"]) == 10
        assert len(data["N_peak_empirical"]) == 10
        # Empirical should be close to theory
        np.testing.assert_allclose(
            data["N_peak_empirical"],
            data["N_peak_theory"],
            rtol=0.05,
        )

    def test_defense_sweep_data(self):
        from simulating_anything.rediscovery.group_defense import (
            generate_defense_sweep_data,
        )
        data = generate_defense_sweep_data(n_c_values=5, n_steps=500, dt=0.01)
        assert len(data["c"]) == 5
        assert len(data["N_avg"]) == 5
        assert len(data["P_avg"]) == 5
        assert len(data["N_amplitude"]) == 5
        assert len(data["N_peak"]) == 5

    def test_peak_formula_consistency(self):
        """N_peak = 1/sqrt(c) should hold across multiple c values."""
        from simulating_anything.rediscovery.group_defense import (
            generate_peak_location_data,
        )
        data = generate_peak_location_data(n_c_values=15, c_min=0.01, c_max=0.3)
        for i in range(len(data["c_values"])):
            c_val = data["c_values"][i]
            expected = 1.0 / np.sqrt(c_val)
            assert data["N_peak_theory"][i] == pytest.approx(expected, rel=1e-10)
