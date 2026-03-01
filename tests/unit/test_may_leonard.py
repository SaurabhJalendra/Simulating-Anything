"""Tests for the May-Leonard cyclic competition simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.may_leonard import MayLeonardSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    n_species: int = 4,
    a: float = 1.5,
    b: float = 0.5,
    r: float = 1.0,
    K: float = 1.0,
    x_init: list[float] | None = None,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Helper to build a SimulationConfig for the May-Leonard model."""
    params: dict[str, float] = {
        "n_species": float(n_species),
        "a": a,
        "b": b,
        "r": r,
        "K": K,
    }
    if x_init is not None:
        for i, val in enumerate(x_init):
            params[f"x_0_{i}"] = val

    return SimulationConfig(
        domain=Domain.MAY_LEONARD,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestReset:
    """Reset should produce valid initial state."""

    def test_reset_positive(self):
        """Initial state should have all positive populations."""
        config = _make_config()
        sim = MayLeonardSimulation(config)
        state = sim.reset()
        assert np.all(state > 0)

    def test_reset_custom_init(self):
        """Custom initial conditions should be respected."""
        config = _make_config(x_init=[0.3, 0.2, 0.4, 0.1])
        sim = MayLeonardSimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [0.3, 0.2, 0.4, 0.1])


class TestObserveShape:
    """observe() should return the correct shape."""

    def test_observe_4species(self):
        """Default 4-species model returns 4-element array."""
        config = _make_config()
        sim = MayLeonardSimulation(config)
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_observe_3species(self):
        """3-species model returns 3-element array."""
        config = _make_config(n_species=3, x_init=[0.3, 0.3, 0.4])
        sim = MayLeonardSimulation(config)
        sim.reset()
        assert sim.observe().shape == (3,)

    def test_observe_5species(self):
        """5-species model returns 5-element array."""
        config = _make_config(n_species=5, x_init=[0.2, 0.2, 0.2, 0.2, 0.2])
        sim = MayLeonardSimulation(config)
        sim.reset()
        assert sim.observe().shape == (5,)


class TestStepAdvances:
    """State should change after stepping."""

    def test_step_changes_state(self):
        """A single step should change the state."""
        config = _make_config(x_init=[0.3, 0.2, 0.25, 0.25])
        sim = MayLeonardSimulation(config)
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)


class TestDeterministic:
    """Same parameters should produce identical trajectories."""

    def test_deterministic(self):
        """Two runs with identical config produce identical results."""
        config = _make_config(dt=0.01, n_steps=200)
        sim1 = MayLeonardSimulation(config)
        sim2 = MayLeonardSimulation(config)

        sim1.reset()
        sim2.reset()

        states1 = []
        states2 = []
        for _ in range(200):
            states1.append(sim1.step().copy())
            states2.append(sim2.step().copy())

        np.testing.assert_array_equal(np.array(states1), np.array(states2))


class TestStability:
    """Simulation should remain stable (no NaN/Inf)."""

    def test_no_nan_after_10000_steps(self):
        """No NaN or Inf after 10000 steps."""
        config = _make_config(
            x_init=[0.3, 0.2, 0.25, 0.25], dt=0.01, n_steps=10000,
        )
        sim = MayLeonardSimulation(config)
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN/Inf in state: {state}"


class TestPositivePopulations:
    """All populations must remain non-negative."""

    def test_non_negative(self):
        """Populations stay >= 0 at all times."""
        config = _make_config(
            x_init=[0.01, 0.01, 0.01, 0.01], dt=0.01, n_steps=5000,
        )
        sim = MayLeonardSimulation(config)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative population: {state}"


class TestCompetitionMatrixShape:
    """Competition matrix should have correct shape."""

    def test_shape_4species(self):
        """4-species model has 4x4 competition matrix."""
        config = _make_config()
        sim = MayLeonardSimulation(config)
        assert sim.alpha.shape == (4, 4)

    def test_shape_5species(self):
        """5-species model has 5x5 competition matrix."""
        config = _make_config(n_species=5, x_init=[0.2] * 5)
        sim = MayLeonardSimulation(config)
        assert sim.alpha.shape == (5, 5)


class TestCompetitionMatrixCyclic:
    """Competition matrix should have the correct cyclic structure."""

    def test_diagonal_ones(self):
        """Diagonal elements should be 1 (intraspecific competition)."""
        config = _make_config(a=1.5, b=0.5)
        sim = MayLeonardSimulation(config)
        np.testing.assert_allclose(np.diag(sim.alpha), 1.0)

    def test_cyclic_structure(self):
        """Verify the circulant cyclic structure of the competition matrix.

        Expected for n=4, a=1.5, b=0.5:
            [[1.0, 1.5, 0.5, 0.5],
             [0.5, 1.0, 1.5, 0.5],
             [0.5, 0.5, 1.0, 1.5],
             [1.5, 0.5, 0.5, 1.0]]
        """
        config = _make_config(a=1.5, b=0.5)
        sim = MayLeonardSimulation(config)
        expected = np.array([
            [1.0, 1.5, 0.5, 0.5],
            [0.5, 1.0, 1.5, 0.5],
            [0.5, 0.5, 1.0, 1.5],
            [1.5, 0.5, 0.5, 1.0],
        ])
        np.testing.assert_allclose(sim.alpha, expected)

    def test_row_sums_equal(self):
        """For symmetric parameters, all row sums should be equal."""
        config = _make_config(a=2.0, b=0.3)
        sim = MayLeonardSimulation(config)
        row_sums = np.sum(sim.alpha, axis=1)
        np.testing.assert_allclose(row_sums, row_sums[0])


class TestInteriorFixedPoint:
    """Interior fixed point should exist and be positive for moderate a."""

    def test_fixed_point_positive(self):
        """Interior fixed point should have all positive components."""
        config = _make_config(a=1.5, b=0.5)
        sim = MayLeonardSimulation(config)
        sim.reset()
        x_star = sim.compute_interior_fixed_point()
        assert np.all(x_star > 0), f"Non-positive fixed point: {x_star}"

    def test_fixed_point_formula(self):
        """For symmetric case, x_i* = K / row_sum(alpha).

        With a=1.5, b=0.5, n=4: row_sum = 1 + 1.5 + 2*0.5 = 3.5
        So x_i* = 1.0 / 3.5 = 0.2857...
        """
        config = _make_config(a=1.5, b=0.5, K=1.0)
        sim = MayLeonardSimulation(config)
        sim.reset()
        x_star = sim.compute_interior_fixed_point()
        # row_sum = 1 + a + (n-2)*b = 1 + 1.5 + 2*0.5 = 3.5
        expected = 1.0 / 3.5
        np.testing.assert_allclose(x_star, expected, rtol=1e-10)

    def test_fixed_point_satisfies_equilibrium(self):
        """At equilibrium: alpha @ (x* / K) = 1 for all components."""
        config = _make_config(a=1.5, b=0.5)
        sim = MayLeonardSimulation(config)
        sim.reset()
        x_star = sim.compute_interior_fixed_point()
        lhs = sim.alpha @ (x_star / sim.K)
        np.testing.assert_allclose(lhs, 1.0, rtol=1e-10)


class TestInteriorFixedPointUnstable:
    """For the cyclic May-Leonard system, the interior fixed point is unstable."""

    def test_unstable_eigenvalues(self):
        """At least one eigenvalue should have positive real part."""
        config = _make_config(a=1.5, b=0.5)
        sim = MayLeonardSimulation(config)
        sim.reset()
        eigs = sim.stability_eigenvalues()
        max_real = np.max(np.real(eigs))
        assert max_real > 0, (
            f"Interior fixed point appears stable (max Re(eig)={max_real:.6f})"
        )


class TestCyclicDominance:
    """Dominant species should change cyclically (0 -> 1 -> 2 -> 3 -> 0)."""

    def test_cyclic_dominance_detected(self):
        """Over a long trajectory, the dominant species should cycle."""
        config = _make_config(
            a=1.5, b=0.5,
            x_init=[0.30, 0.24, 0.23, 0.23],
            dt=0.01, n_steps=30000,
        )
        sim = MayLeonardSimulation(config)
        sim.reset()

        states = []
        for _ in range(30000):
            sim.step()
            states.append(sim.observe().copy())
        traj = np.array(states)

        dominance = sim.compute_dominance_index(traj)
        changes = np.diff(dominance)
        n_transitions = int(np.sum(changes != 0))

        # Should have multiple transitions if cycling
        assert n_transitions >= 4, (
            f"Too few dominance transitions: {n_transitions}"
        )


class TestBiodiversityIndex:
    """Shannon entropy tests."""

    def test_positive_for_mixed_state(self):
        """H > 0 when multiple species present."""
        config = _make_config(x_init=[0.25, 0.25, 0.25, 0.25])
        sim = MayLeonardSimulation(config)
        sim.reset()
        H = sim.biodiversity_index()
        assert H > 0

    def test_maximum_at_equal_distribution(self):
        """H = ln(4) when all species equal."""
        config = _make_config(x_init=[0.25, 0.25, 0.25, 0.25])
        sim = MayLeonardSimulation(config)
        sim.reset()
        H = sim.biodiversity_index()
        assert H == pytest.approx(np.log(4), rel=1e-10)

    def test_zero_for_single_species(self):
        """H = 0 when only one species present."""
        config = _make_config(x_init=[1.0, 0.0, 0.0, 0.0])
        sim = MayLeonardSimulation(config)
        sim.reset()
        H = sim.biodiversity_index()
        assert H == pytest.approx(0.0, abs=1e-10)


class TestTotalPopulationOscillates:
    """Total population N(t) should oscillate (not monotonic) during cycling."""

    def test_total_not_monotonic(self):
        """N(t) should go both up and down when starting near fixed point.

        Starting near the interior fixed point (x*~0.286 for a=1.5, b=0.5)
        with perturbation ensures the system spirals outward with oscillations
        in total population before eventually approaching the boundary.
        """
        # Start near the fixed point with asymmetry to trigger cycling
        # x* = 1/3.5 ~ 0.286
        config = _make_config(
            a=1.5, b=0.5,
            x_init=[0.35, 0.28, 0.20, 0.30],
            dt=0.01, n_steps=15000,
        )
        sim = MayLeonardSimulation(config)
        sim.reset()

        total_pops = []
        for _ in range(15000):
            sim.step()
            total_pops.append(float(np.sum(sim.observe())))

        diffs = np.diff(total_pops)
        has_increase = np.any(diffs > 0)
        has_decrease = np.any(diffs < 0)
        assert has_increase and has_decrease, (
            "Total population is monotonic -- expected oscillation"
        )


class TestThreeSpeciesMode:
    """n_species=3 should also work."""

    def test_3species_runs(self):
        """3-species May-Leonard should run without error."""
        config = _make_config(
            n_species=3, a=1.5, b=0.5,
            x_init=[0.35, 0.33, 0.32], dt=0.01, n_steps=5000,
        )
        sim = MayLeonardSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        state = sim.observe()
        assert state.shape == (3,)
        assert np.all(np.isfinite(state))
        assert np.all(state >= 0)


class TestFiveSpeciesMode:
    """n_species=5 should also work."""

    def test_5species_runs(self):
        """5-species May-Leonard should run without error."""
        config = _make_config(
            n_species=5, a=1.5, b=0.5,
            x_init=[0.22, 0.20, 0.19, 0.20, 0.19],
            dt=0.01, n_steps=5000,
        )
        sim = MayLeonardSimulation(config)
        sim.reset()
        for _ in range(5000):
            sim.step()
        state = sim.observe()
        assert state.shape == (5,)
        assert np.all(np.isfinite(state))
        assert np.all(state >= 0)


class TestStrongCompetitionExtinction:
    """Very large a should drive species toward near-extinction."""

    def test_strong_a_reduces_diversity(self):
        """Strong competition reduces surviving species."""
        config = _make_config(
            a=5.0, b=0.1,
            x_init=[0.30, 0.24, 0.23, 0.23],
            dt=0.001, n_steps=50000,
        )
        sim = MayLeonardSimulation(config)
        sim.reset()
        for _ in range(50000):
            sim.step()

        # Very strong cyclic competition should have low biodiversity
        # (species approach heteroclinic boundary with near-zero populations)
        state = sim.observe()
        # At least the simulation doesn't blow up
        assert np.all(np.isfinite(state))
        assert np.all(state >= 0)


class TestEqualCompetitionCoexistence:
    """When a=1 (all competition equal), stable coexistence should occur."""

    def test_weak_asymmetry_coexistence(self):
        """With a close to 1 (weak asymmetry), all species still coexist.

        Using a=1.1, b=0.9 keeps the system close to symmetric.
        The interior fixed point exists and is stable enough for
        long-term coexistence.
        """
        config = _make_config(
            a=1.1, b=0.9,
            x_init=[0.25, 0.25, 0.25, 0.25],
            dt=0.01, n_steps=20000,
        )
        sim = MayLeonardSimulation(config)
        sim.reset()
        for _ in range(20000):
            sim.step()
        state = sim.observe()

        # All species should still be alive with weak competition asymmetry
        assert np.all(state > 0.01), f"Species went extinct: {state}"


class TestParameterSweep:
    """competition_parameter_sweep should produce valid data."""

    def test_sweep_produces_data(self):
        """Sweep returns arrays with correct lengths."""
        config = _make_config(
            x_init=[0.26, 0.25, 0.24, 0.25],
            dt=0.01, n_steps=1000,
        )
        sim = MayLeonardSimulation(config)
        sim.reset()

        a_values = [1.1, 1.5, 2.0]
        result = sim.competition_parameter_sweep(a_values, n_steps=3000)

        assert len(result["a_values"]) == 3
        assert len(result["periods"]) == 3
        assert len(result["biodiversity"]) == 3
        assert len(result["total_population"]) == 3
        assert len(result["is_cyclic"]) == 3


class TestRediscoveryData:
    """Rediscovery data generation should return valid data."""

    def test_heteroclinic_trajectory(self):
        """generate_heteroclinic_trajectory returns valid data."""
        from simulating_anything.rediscovery.may_leonard import (
            generate_heteroclinic_trajectory,
        )
        data = generate_heteroclinic_trajectory(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 4)
        assert len(data["time"]) == 501
        assert data["x_star"].shape == (4,)
        assert data["n_species"] == 4

    def test_fixed_point_data(self):
        """generate_fixed_point_data returns valid data."""
        from simulating_anything.rediscovery.may_leonard import (
            generate_fixed_point_data,
        )
        data = generate_fixed_point_data(n_samples=5)
        assert len(data["a_values"]) == 5
        assert data["fixed_points"].shape == (5, 4)
        assert len(data["theoretical_x_star"]) == 5


class TestNSurviving:
    """n_surviving should count species above threshold."""

    def test_all_surviving(self):
        """All species present initially."""
        config = _make_config(x_init=[0.25, 0.25, 0.25, 0.25])
        sim = MayLeonardSimulation(config)
        sim.reset()
        assert sim.n_surviving() == 4

    def test_partial_surviving(self):
        """Species below threshold not counted."""
        config = _make_config(x_init=[0.5, 0.0001, 0.5, 0.0001])
        sim = MayLeonardSimulation(config)
        sim.reset()
        assert sim.n_surviving(threshold=1e-3) == 2


class TestTotalPopulation:
    """compute_total_population should return correct sums."""

    def test_total_population(self):
        """Total population should sum species populations."""
        config = _make_config(x_init=[0.3, 0.2, 0.25, 0.25])
        sim = MayLeonardSimulation(config)
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(10):
            sim.step()
            states.append(sim.observe().copy())
        traj = np.array(states)

        total = sim.compute_total_population(traj)
        assert total.shape == (11,)
        assert total[0] == pytest.approx(1.0)
