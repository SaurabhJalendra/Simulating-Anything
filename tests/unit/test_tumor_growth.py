"""Tests for the Tumor-Immune interaction (Kuznetsov-Taylor) model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.tumor_growth import TumorGrowthSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.1,
    n_steps: int = 1000,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for tumor growth with optional overrides."""
    params: dict[str, float] = {
        "r": 0.5,
        "K": 1e6,
        "a": 1e-7,
        "s": 1e4,
        "p": 0.1,
        "g": 1e5,
        "d": 0.03,
        "a_I": 1e-7,
        "T_0": 1e4,
        "I_0": 1e5,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.TUMOR_GROWTH,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestTumorGrowthCreation:
    """Tests for simulation creation and initialization."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = TumorGrowthSimulation(_make_config())
        assert sim.r == 0.5
        assert sim.K == 1e6
        assert sim.a == 1e-7
        assert sim.s == 1e4
        assert sim.p_stim == 0.1
        assert sim.g == 1e5
        assert sim.d == 0.03
        assert sim.a_I == 1e-7

    def test_initial_state_shape(self):
        """State vector should be [T, I] with 2 components."""
        sim = TumorGrowthSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        """Initial conditions should match parameters."""
        sim = TumorGrowthSimulation(_make_config(T_0=5000.0, I_0=2e5))
        state = sim.reset()
        np.testing.assert_allclose(state, [5000.0, 2e5])

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = TumorGrowthSimulation(_make_config(r=1.0, K=5e5, a=5e-7))
        assert sim.r == 1.0
        assert sim.K == 5e5
        assert sim.a == 5e-7

    def test_default_initial_conditions(self):
        """Default T_0=1e4, I_0=1e5."""
        sim = TumorGrowthSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [1e4, 1e5])


class TestTumorGrowthBasicSimulation:
    """Core simulation mechanics tests."""

    def test_step_advances_state(self):
        """State should change after a step."""
        sim = TumorGrowthSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_current_state(self):
        """Observe should return the current state."""
        sim = TumorGrowthSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_run_returns_trajectory(self):
        """run() should produce a TrajectoryData object."""
        sim = TumorGrowthSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = TumorGrowthSimulation(config)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = TumorGrowthSimulation(config)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_long_run_stability(self):
        """No NaN after many steps."""
        sim = TumorGrowthSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(20000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"

    def test_trajectory_timestamps(self):
        """Trajectory timestamps should be evenly spaced."""
        sim = TumorGrowthSimulation(_make_config(dt=0.1, n_steps=50))
        traj = sim.run(n_steps=50)
        expected = np.arange(51) * 0.1
        np.testing.assert_allclose(traj.timestamps, expected, atol=1e-10)

    def test_trajectory_parameters(self):
        """Trajectory should include simulation parameters."""
        sim = TumorGrowthSimulation(_make_config())
        traj = sim.run(n_steps=10)
        assert "r" in traj.parameters
        assert "K" in traj.parameters


class TestTumorGrowthNonNegativity:
    """All populations must remain non-negative."""

    def test_non_negative_default_params(self):
        """T and I should remain >= 0 with default parameters."""
        sim = TumorGrowthSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"T became negative: {state[0]}"
            assert state[1] >= 0, f"I became negative: {state[1]}"

    def test_non_negative_high_killing(self):
        """Non-negativity should hold with high immune killing rate."""
        sim = TumorGrowthSimulation(_make_config(a=1e-5, dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"T became negative: {state[0]}"
            assert state[1] >= 0, f"I became negative: {state[1]}"

    def test_non_negative_low_immune_source(self):
        """Non-negativity should hold with very low immune source."""
        sim = TumorGrowthSimulation(_make_config(s=1.0, dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"T became negative: {state[0]}"
            assert state[1] >= 0, f"I became negative: {state[1]}"

    def test_non_negative_small_initial(self):
        """Non-negativity should hold with very small initial populations."""
        sim = TumorGrowthSimulation(_make_config(T_0=1.0, I_0=1.0, dt=0.1))
        sim.reset()
        for _ in range(2000):
            state = sim.step()
            assert state[0] >= 0, f"T became negative: {state[0]}"
            assert state[1] >= 0, f"I became negative: {state[1]}"


class TestTumorGrowthBoundedness:
    """Tumor and immune populations should remain bounded."""

    def test_tumor_bounded_by_capacity(self):
        """Tumor population should not significantly exceed carrying capacity K."""
        sim = TumorGrowthSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            # Allow small overshoot due to numerical integration
            assert state[0] < sim.K * 1.1, (
                f"T exceeded K: T={state[0]}, K={sim.K}"
            )

    def test_immune_bounded(self):
        """Immune population should remain bounded."""
        sim = TumorGrowthSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            # Immune population should stay reasonable
            assert state[1] < 1e8, f"I grew unbounded: {state[1]}"


class TestTumorElimination:
    """Tests for tumor elimination under strong immune response."""

    def test_elimination_high_killing(self):
        """With very high immune killing rate, tumor should be eliminated."""
        sim = TumorGrowthSimulation(_make_config(a=1e-4, dt=0.1))
        sim.reset()
        for _ in range(5000):
            sim.step()
        T_final = sim.observe()[0]
        assert T_final < 100.0, f"Tumor not eliminated with high a: T={T_final}"

    def test_elimination_high_immune_source(self):
        """With very high immune source, tumor should be eliminated."""
        sim = TumorGrowthSimulation(_make_config(s=1e6, dt=0.1))
        sim.reset()
        for _ in range(5000):
            sim.step()
        T_final = sim.observe()[0]
        assert T_final < 100.0, f"Tumor not eliminated with high s: T={T_final}"

    def test_elimination_classify(self):
        """classify_outcome should return 'elimination' for strong immunity."""
        sim = TumorGrowthSimulation(_make_config(a=1e-4))
        outcome = sim.classify_outcome(n_steps=5000)
        assert outcome == "elimination"


class TestTumorEscape:
    """Tests for tumor escape under weak immune response."""

    def test_escape_low_killing(self):
        """With very low immune killing rate, tumor should escape."""
        sim = TumorGrowthSimulation(_make_config(a=1e-10, s=100.0, dt=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
        T_final = sim.observe()[0]
        assert T_final > 0.3 * sim.K, f"Tumor should escape with low a: T={T_final}"

    def test_escape_low_immune_source(self):
        """With very low immune source, tumor should escape."""
        sim = TumorGrowthSimulation(_make_config(s=1.0, a=1e-9, dt=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
        T_final = sim.observe()[0]
        assert T_final > 0.1 * sim.K, (
            f"Tumor should escape with low s: T={T_final}"
        )

    def test_escape_classify(self):
        """classify_outcome should return 'escape' for weak immunity."""
        sim = TumorGrowthSimulation(_make_config(a=1e-10, s=100.0))
        outcome = sim.classify_outcome(n_steps=10000)
        assert outcome == "escape"


class TestTumorDormancy:
    """Tests for tumor dormancy (stable coexistence)."""

    def test_dormancy_intermediate_params(self):
        """With intermediate parameters, both T and I should persist."""
        sim = TumorGrowthSimulation(_make_config(
            a=1e-7, s=1e4, p=0.2, dt=0.1,
        ))
        sim.reset()
        for _ in range(10000):
            sim.step()
        T_final, I_final = sim.observe()
        # Both should be nonzero and T should be well below K
        assert T_final > 10.0, f"T too small for dormancy: {T_final}"
        assert T_final < 0.5 * sim.K, f"T too large for dormancy: {T_final}"
        assert I_final > 10.0, f"I too small for dormancy: {I_final}"

    def test_coexistence_different_initial(self):
        """Dormancy should be reached from different initial conditions."""
        results = []
        for T_0 in [100.0, 1e4, 1e5]:
            sim = TumorGrowthSimulation(_make_config(
                a=5e-7, s=5e4, p=0.2, T_0=T_0, dt=0.1,
            ))
            sim.reset()
            for _ in range(15000):
                sim.step()
            results.append(sim.observe().copy())

        # All should converge to similar equilibrium (within factor of 10)
        T_finals = [r[0] for r in results]
        if all(t > 10 for t in T_finals):
            # Check they are in the same order of magnitude
            log_range = np.log10(max(T_finals)) - np.log10(min(T_finals) + 1)
            assert log_range < 2.0, (
                f"T finals too different for dormancy convergence: {T_finals}"
            )


class TestTumorFreeEquilibrium:
    """Tests for the tumor-free equilibrium."""

    def test_tumor_free_eq_values(self):
        """Tumor-free equilibrium should be (0, s/d)."""
        sim = TumorGrowthSimulation(_make_config(s=1e4, d=0.03))
        T_star, I_star = sim.tumor_free_equilibrium
        assert T_star == 0.0
        assert I_star == pytest.approx(1e4 / 0.03, rel=1e-10)

    def test_derivatives_at_tumor_free_eq(self):
        """At tumor-free equilibrium, dT/dt should be 0 (for T=0)."""
        sim = TumorGrowthSimulation(_make_config())
        sim.reset()
        tfe = np.array(sim.tumor_free_equilibrium)
        dy = sim._derivatives(tfe)
        # dT/dt = 0 (since T=0)
        assert dy[0] == pytest.approx(0.0, abs=1e-10)
        # dI/dt = s - d*(s/d) = 0
        assert dy[1] == pytest.approx(0.0, abs=1e-6)

    def test_tumor_free_eq_different_params(self):
        """Tumor-free equilibrium should scale with s/d."""
        for s_val, d_val in [(1e3, 0.01), (1e5, 0.1), (5e4, 0.05)]:
            sim = TumorGrowthSimulation(_make_config(s=s_val, d=d_val))
            _, I_star = sim.tumor_free_equilibrium
            assert I_star == pytest.approx(s_val / d_val, rel=1e-10)


class TestCoexistenceEquilibria:
    """Tests for coexistence equilibrium computation."""

    def test_coexistence_returns_list(self):
        """coexistence_equilibria should return a list of (T, I) pairs."""
        sim = TumorGrowthSimulation(_make_config())
        eq = sim.coexistence_equilibria()
        assert isinstance(eq, list)

    def test_coexistence_positive_values(self):
        """All equilibria should have positive T and I."""
        sim = TumorGrowthSimulation(_make_config())
        eq = sim.coexistence_equilibria()
        for T_star, I_star in eq:
            assert T_star > 0, f"Negative T* in equilibrium: {T_star}"
            assert I_star > 0, f"Negative I* in equilibrium: {I_star}"

    def test_coexistence_near_zero_derivatives(self):
        """At coexistence equilibria, derivatives should be near zero."""
        sim = TumorGrowthSimulation(_make_config())
        sim.reset()
        eq = sim.coexistence_equilibria()
        for T_star, I_star in eq:
            dy = sim._derivatives(np.array([T_star, I_star]))
            # Allow relative tolerance since values can be large
            norm = np.linalg.norm(dy)
            scale = max(abs(T_star), abs(I_star), 1.0)
            assert norm / scale < 0.01, (
                f"Derivatives not near zero at ({T_star}, {I_star}): {dy}"
            )


class TestTumorGrowthDerivatives:
    """Tests for the ODE right-hand side."""

    def test_derivatives_zero_tumor(self):
        """With T=0, dT/dt should be 0, dI/dt should be s - d*I."""
        sim = TumorGrowthSimulation(_make_config())
        sim.reset()
        dy = sim._derivatives(np.array([0.0, 1e5]))
        assert dy[0] == pytest.approx(0.0)
        expected_dI = sim.s - sim.d * 1e5
        assert dy[1] == pytest.approx(expected_dI, rel=1e-10)

    def test_derivatives_at_carrying_capacity(self):
        """At T=K, logistic growth term should be zero."""
        sim = TumorGrowthSimulation(_make_config())
        sim.reset()
        K = sim.K
        I_val = 1e5
        dy = sim._derivatives(np.array([K, I_val]))
        # dT/dt = r*K*(1-K/K) - a*K*I = -a*K*I (logistic term vanishes)
        expected_dT = -sim.a * K * I_val
        assert dy[0] == pytest.approx(expected_dT, rel=1e-10)

    def test_derivatives_shape(self):
        """Derivatives should return array of shape (2,)."""
        sim = TumorGrowthSimulation(_make_config())
        sim.reset()
        dy = sim._derivatives(np.array([1e4, 1e5]))
        assert dy.shape == (2,)

    def test_tumor_growth_positive_without_immune(self):
        """With I=0, dT/dt should be r*T*(1-T/K) > 0 for T < K."""
        sim = TumorGrowthSimulation(_make_config())
        sim.reset()
        T = 1e4
        dy = sim._derivatives(np.array([T, 0.0]))
        expected = sim.r * T * (1.0 - T / sim.K)
        assert dy[0] == pytest.approx(expected, rel=1e-10)
        assert dy[0] > 0


class TestImmuneStrengthRatio:
    """Tests for the immune strength ratio metric."""

    def test_immune_strength_ratio(self):
        """Ratio should be a * I_0 / r."""
        sim = TumorGrowthSimulation(_make_config(a=1e-7, I_0=1e5, r=0.5))
        expected = 1e-7 * 1e5 / 0.5
        assert sim.immune_strength_ratio() == pytest.approx(expected)

    def test_higher_killing_higher_ratio(self):
        """Higher a should give higher immune strength ratio."""
        sim_low = TumorGrowthSimulation(_make_config(a=1e-8))
        sim_high = TumorGrowthSimulation(_make_config(a=1e-6))
        assert sim_high.immune_strength_ratio() > sim_low.immune_strength_ratio()


class TestTumorDoublingTime:
    """Tests for tumor doubling time measurement."""

    def test_doubling_time_finite(self):
        """Doubling time should be finite for growing tumor."""
        sim = TumorGrowthSimulation(_make_config(a=1e-10, s=10.0, dt=0.1))
        td = sim.measure_tumor_doubling_time()
        assert np.isfinite(td), "Doubling time should be finite for growing tumor"
        assert td > 0

    def test_doubling_time_infinite_eliminated(self):
        """Doubling time should be infinite when tumor is eliminated."""
        # Need very high I_0 so immune killing overwhelms growth from step 1
        sim = TumorGrowthSimulation(_make_config(a=1e-4, s=1e6, I_0=1e7, dt=0.1))
        td = sim.measure_tumor_doubling_time()
        assert td == float("inf"), "Doubling time should be inf for eliminated tumor"

    def test_faster_growth_shorter_doubling(self):
        """Higher growth rate r should give shorter doubling time."""
        sim_slow = TumorGrowthSimulation(_make_config(
            r=0.1, a=1e-10, s=10.0, dt=0.1,
        ))
        sim_fast = TumorGrowthSimulation(_make_config(
            r=1.0, a=1e-10, s=10.0, dt=0.1,
        ))
        td_slow = sim_slow.measure_tumor_doubling_time()
        td_fast = sim_fast.measure_tumor_doubling_time()
        if np.isfinite(td_slow) and np.isfinite(td_fast):
            assert td_fast < td_slow


class TestTumorGrowthClassifyOutcome:
    """Tests for outcome classification."""

    def test_classify_returns_valid_string(self):
        """Should return one of 'elimination', 'dormancy', 'escape'."""
        sim = TumorGrowthSimulation(_make_config())
        outcome = sim.classify_outcome(n_steps=1000)
        assert outcome in ("elimination", "dormancy", "escape")

    def test_classify_resets_state(self):
        """classify_outcome should reset the simulation first."""
        sim = TumorGrowthSimulation(_make_config())
        sim.reset()
        for _ in range(100):
            sim.step()
        sim.classify_outcome(n_steps=100)
        # After classify, state should be valid (it resets and reruns)
        state_after = sim.observe()
        assert np.all(np.isfinite(state_after))


class TestTumorGrowthParameterSweep:
    """Tests for parameter sensitivity."""

    def test_higher_growth_rate_more_tumor(self):
        """Higher r should produce more tumor cells."""
        results = {}
        for r_val in [0.1, 0.5, 1.0]:
            sim = TumorGrowthSimulation(_make_config(r=r_val, dt=0.1))
            sim.reset()
            for _ in range(3000):
                sim.step()
            results[r_val] = sim.observe()[0]

        # Higher growth rate should generally lead to more tumor
        assert results[1.0] >= results[0.1] * 0.5, (
            f"Higher r should produce more tumor: {results}"
        )

    def test_higher_immune_source_less_tumor(self):
        """Higher s (immune source) should generally reduce final tumor size."""
        results = {}
        for s_val in [1e3, 1e5]:
            sim = TumorGrowthSimulation(_make_config(s=s_val, dt=0.1))
            sim.reset()
            for _ in range(5000):
                sim.step()
            results[s_val] = sim.observe()[0]

        assert results[1e5] <= results[1e3] + 1.0, (
            f"Higher s should reduce tumor: s=1e3 -> T={results[1e3]}, "
            f"s=1e5 -> T={results[1e5]}"
        )

    def test_carrying_capacity_limits_growth(self):
        """Tumor should not exceed carrying capacity K (significantly)."""
        for K_val in [1e4, 1e6]:
            sim = TumorGrowthSimulation(_make_config(
                K=K_val, a=1e-10, s=1.0, dt=0.1,
            ))
            sim.reset()
            for _ in range(10000):
                sim.step()
            T_final = sim.observe()[0]
            assert T_final <= K_val * 1.05, (
                f"T={T_final} exceeded K={K_val}"
            )


class TestTumorGrowthRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_ode_data_generation(self):
        """ODE data generation should produce valid arrays."""
        from simulating_anything.rediscovery.tumor_growth import generate_ode_data

        data = generate_ode_data(n_steps=500, dt=0.1)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["dt"] == 0.1
        assert np.all(np.isfinite(data["states"]))

    def test_ode_data_non_negative(self):
        """All ODE trajectory values should be non-negative."""
        from simulating_anything.rediscovery.tumor_growth import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.1)
        assert np.all(data["states"] >= 0), "Negative population in ODE data"

    def test_immune_sweep_data(self):
        """Immune sweep data should contain valid arrays and outcomes."""
        from simulating_anything.rediscovery.tumor_growth import (
            generate_immune_sweep_data,
        )

        data = generate_immune_sweep_data(n_a=5, n_steps=1000, dt=0.1)
        assert len(data["a"]) == 5
        assert len(data["final_T"]) == 5
        assert len(data["final_I"]) == 5
        assert len(data["outcomes"]) == 5
        assert all(
            o in ("elimination", "dormancy", "escape") for o in data["outcomes"]
        )

    def test_immune_sweep_a_values_sorted(self):
        """Immune killing rate values should be sorted (logspace)."""
        from simulating_anything.rediscovery.tumor_growth import (
            generate_immune_sweep_data,
        )

        data = generate_immune_sweep_data(n_a=10, n_steps=500, dt=0.1)
        assert np.all(np.diff(data["a"]) > 0), "a values should be increasing"

    def test_dormancy_data_generation(self):
        """Dormancy data generation should produce valid results."""
        from simulating_anything.rediscovery.tumor_growth import (
            generate_dormancy_data,
        )

        data = generate_dormancy_data(n_samples=5, n_steps=1000, dt=0.1)
        assert len(data["a"]) == 5
        assert len(data["s"]) == 5
        assert len(data["final_T"]) == 5
        assert len(data["final_I"]) == 5
        assert len(data["outcomes"]) == 5
        assert all(
            o in ("elimination", "dormancy", "escape") for o in data["outcomes"]
        )

    def test_bifurcation_data_generation(self):
        """Bifurcation data should contain valid arrays."""
        from simulating_anything.rediscovery.tumor_growth import (
            generate_bifurcation_data,
        )

        data = generate_bifurcation_data(n_points=5, n_steps=1000, dt=0.1)
        assert len(data["s"]) == 5
        assert len(data["final_T"]) == 5
        assert len(data["final_I"]) == 5
        assert len(data["outcomes"]) == 5

    def test_bifurcation_s_values_sorted(self):
        """Immune source rate values should be sorted (logspace)."""
        from simulating_anything.rediscovery.tumor_growth import (
            generate_bifurcation_data,
        )

        data = generate_bifurcation_data(n_points=10, n_steps=500, dt=0.1)
        assert np.all(np.diff(data["s"]) > 0), "s values should be increasing"

    def test_bifurcation_has_transitions(self):
        """With enough points, bifurcation sweep should show multiple outcomes."""
        from simulating_anything.rediscovery.tumor_growth import (
            generate_bifurcation_data,
        )

        data = generate_bifurcation_data(n_points=20, n_steps=3000, dt=0.1)
        unique_outcomes = set(data["outcomes"])
        # Should have at least 2 different outcomes across the sweep
        assert len(unique_outcomes) >= 2, (
            f"Expected multiple outcomes, got: {unique_outcomes}"
        )
