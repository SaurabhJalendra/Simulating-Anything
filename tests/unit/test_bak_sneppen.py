"""Tests for the Bak-Sneppen self-organized criticality simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.bak_sneppen import BakSneppen
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N: int = 50,
    dt: float = 1.0,
    n_steps: int = 1000,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.BAK_SNEPPEN,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        parameters={"N": float(N)},
    )


class TestBakSneppenCreation:
    def test_creation(self):
        sim = BakSneppen(_make_config())
        assert sim.N == 50

    def test_default_params(self):
        config = SimulationConfig(
            domain=Domain.BAK_SNEPPEN,
            dt=1.0,
            n_steps=100,
            parameters={},
        )
        sim = BakSneppen(config)
        assert sim.N == 50

    def test_custom_N(self):
        sim = BakSneppen(_make_config(N=100))
        assert sim.N == 100


class TestBakSneppenState:
    def test_initial_state_shape(self):
        sim = BakSneppen(_make_config(N=50))
        state = sim.reset(seed=42)
        assert state.shape == (50,)

    def test_initial_state_in_unit_interval(self):
        sim = BakSneppen(_make_config(N=100))
        state = sim.reset(seed=42)
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0)

    def test_observe_matches_state(self):
        sim = BakSneppen(_make_config())
        state = sim.reset(seed=42)
        observed = sim.observe()
        np.testing.assert_array_equal(state, observed)


class TestBakSneppenStep:
    def test_step_changes_state(self):
        sim = BakSneppen(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.array_equal(s0, s1)

    def test_step_changes_at_least_three_values(self):
        """Each step replaces the minimum and its 2 neighbors."""
        sim = BakSneppen(_make_config(N=50))
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        n_changed = np.sum(s0 != s1)
        # At least 3 values should change (min + 2 neighbors)
        # Could be fewer if neighbors overlap for very small N, but N=50 is fine
        assert n_changed >= 3, f"Only {n_changed} values changed, expected >= 3"

    def test_step_changes_exactly_three_for_large_N(self):
        """For large enough N with distinct indices, exactly 3 values change."""
        sim = BakSneppen(_make_config(N=100))
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        n_changed = np.sum(s0 != s1)
        assert n_changed == 3, f"{n_changed} values changed, expected 3"

    def test_fitness_stays_in_unit_interval(self):
        """After many steps, all fitnesses remain in [0, 1]."""
        sim = BakSneppen(_make_config(N=50))
        sim.reset(seed=42)
        for _ in range(1000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0.0), f"Min fitness {np.min(state)} < 0"
            assert np.all(state <= 1.0), f"Max fitness {np.max(state)} > 1"

    def test_all_finite_after_many_steps(self):
        sim = BakSneppen(_make_config(N=50))
        sim.reset(seed=42)
        for _ in range(5000):
            sim.step()
        assert np.all(np.isfinite(sim.observe()))


class TestBakSneppenSOC:
    def test_mean_fitness_above_half(self):
        """After many steps, mean fitness should converge above 0.5."""
        sim = BakSneppen(_make_config(N=100))
        sim.reset(seed=42)
        for _ in range(10000):
            sim.step()
        mean_f = sim.mean_fitness
        assert mean_f > 0.5, f"Mean fitness {mean_f} not above 0.5"

    def test_threshold_estimation(self):
        """Measured SOC threshold should be in reasonable range [0.5, 0.8]."""
        sim = BakSneppen(_make_config(N=100))
        threshold = sim.measure_soc_threshold(
            n_transient=5000,
            n_measure=5000,
            seed=42,
        )
        assert 0.5 < threshold < 0.8, (
            f"Threshold {threshold} outside expected range [0.5, 0.8]"
        )

    def test_threshold_near_two_thirds(self):
        """For large enough system, threshold should be near 2/3."""
        sim = BakSneppen(_make_config(N=200))
        threshold = sim.measure_soc_threshold(
            n_transient=10000,
            n_measure=10000,
            seed=42,
        )
        # Allow 10% relative error from 2/3 ~ 0.667
        np.testing.assert_allclose(threshold, 2.0 / 3.0, atol=0.07)


class TestBakSneppenProperties:
    def test_min_fitness_property(self):
        sim = BakSneppen(_make_config())
        sim.reset(seed=42)
        assert sim.min_fitness == pytest.approx(np.min(sim.observe()))

    def test_mean_fitness_property(self):
        sim = BakSneppen(_make_config())
        sim.reset(seed=42)
        assert sim.mean_fitness == pytest.approx(np.mean(sim.observe()))

    def test_different_N_values(self):
        """Should work for different system sizes."""
        for N in [10, 50, 100, 200]:
            sim = BakSneppen(_make_config(N=N))
            state = sim.reset(seed=42)
            assert state.shape == (N,)
            sim.step()
            assert sim.observe().shape == (N,)


class TestBakSneppenReproducibility:
    def test_reproducible_with_seed(self):
        """Same seed produces same trajectory."""
        sim1 = BakSneppen(_make_config(seed=42))
        sim1.reset(seed=42)
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = BakSneppen(_make_config(seed=42))
        sim2.reset(seed=42)
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_array_equal(state1, state2)

    def test_different_seeds_differ(self):
        """Different seeds produce different trajectories."""
        sim1 = BakSneppen(_make_config(seed=42))
        sim1.reset(seed=42)
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = BakSneppen(_make_config(seed=99))
        sim2.reset(seed=99)
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        assert not np.array_equal(state1, state2)


class TestBakSneppenRediscovery:
    def test_threshold_data_generation(self):
        from simulating_anything.rediscovery.bak_sneppen import generate_threshold_data
        data = generate_threshold_data(
            N_values=[20, 50],
            n_transient=1000,
            n_measure=1000,
            n_trials=2,
        )
        assert len(data["N"]) == 2
        assert len(data["threshold_mean"]) == 2
        assert len(data["threshold_std"]) == 2
        assert np.all(data["threshold_mean"] > 0)
        assert np.all(data["threshold_mean"] < 1)

    def test_avalanche_data_generation(self):
        from simulating_anything.rediscovery.bak_sneppen import generate_avalanche_data
        data = generate_avalanche_data(
            N=50,
            n_transient=1000,
            n_avalanches=50,
            n_trials=1,
        )
        assert data["n_avalanches"] > 0
        assert data["mean_size"] > 0

    def test_gap_evolution_data(self):
        from simulating_anything.rediscovery.bak_sneppen import generate_gap_evolution_data
        data = generate_gap_evolution_data(N=50, n_steps=500)
        assert len(data["steps"]) == 500
        assert len(data["min_fitness"]) == 500
        assert len(data["mean_fitness"]) == 500
        assert np.all(data["min_fitness"] >= 0)
        assert np.all(data["min_fitness"] <= 1)

    def test_trajectory_run(self):
        """Test that run() produces a valid TrajectoryData."""
        sim = BakSneppen(_make_config(N=30, n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 30)  # 101 = initial + 100 steps
        assert np.all(traj.states >= 0.0)
        assert np.all(traj.states <= 1.0)
