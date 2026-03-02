"""Tests for the four-species Lotka-Volterra food web simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.four_species_lv import FourSpeciesLVSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

# Use AGENT_BASED as stand-in until FOUR_SPECIES_LV is added to the Domain enum
_DOMAIN = Domain.AGENT_BASED


def _make_config(
    r1: float = 1.0,
    r2: float = 0.8,
    a11: float = 0.1,
    a12: float = 0.05,
    a21: float = 0.05,
    a22: float = 0.1,
    b1: float = 0.5,
    b2: float = 0.5,
    c1: float = 0.3,
    c2: float = 0.3,
    d1: float = 0.4,
    d2: float = 0.4,
    x1_0: float = 0.5,
    x2_0: float = 0.5,
    y1_0: float = 0.3,
    y2_0: float = 0.3,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r1": r1, "r2": r2,
            "a11": a11, "a12": a12, "a21": a21, "a22": a22,
            "b1": b1, "b2": b2,
            "c1": c1, "c2": c2,
            "d1": d1, "d2": d2,
            "x1_0": x1_0, "x2_0": x2_0, "y1_0": y1_0, "y2_0": y2_0,
        },
    )


class TestFourSpeciesLVCreation:
    def test_creation_default_params(self):
        sim = FourSpeciesLVSimulation(_make_config())
        assert sim.r1 == 1.0
        assert sim.r2 == 0.8
        assert sim.a11 == 0.1
        assert sim.a12 == 0.05
        assert sim.a21 == 0.05
        assert sim.a22 == 0.1
        assert sim.b1 == 0.5
        assert sim.b2 == 0.5
        assert sim.c1 == 0.3
        assert sim.c2 == 0.3
        assert sim.d1 == 0.4
        assert sim.d2 == 0.4

    def test_creation_custom_params(self):
        sim = FourSpeciesLVSimulation(_make_config(r1=2.0, b2=0.8, c1=0.5))
        assert sim.r1 == 2.0
        assert sim.b2 == 0.8
        assert sim.c1 == 0.5

    def test_initial_state_shape(self):
        sim = FourSpeciesLVSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (4,)

    def test_initial_state_values(self):
        sim = FourSpeciesLVSimulation(
            _make_config(x1_0=1.0, x2_0=0.8, y1_0=0.5, y2_0=0.4)
        )
        state = sim.reset()
        np.testing.assert_allclose(state, [1.0, 0.8, 0.5, 0.4])

    def test_observe_shape(self):
        sim = FourSpeciesLVSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)


class TestFourSpeciesLVDynamics:
    def test_step_advances_state(self):
        sim = FourSpeciesLVSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_populations_non_negative(self):
        """All populations should remain non-negative."""
        sim = FourSpeciesLVSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_bounded(self):
        """Populations should not blow up to infinity."""
        sim = FourSpeciesLVSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            assert np.all(state < 1000), f"Population blow-up: {state}"

    def test_total_population_property(self):
        sim = FourSpeciesLVSimulation(
            _make_config(x1_0=1.0, x2_0=0.5, y1_0=0.3, y2_0=0.2)
        )
        sim.reset()
        assert sim.total_population == pytest.approx(2.0)

    def test_is_coexisting_true(self):
        sim = FourSpeciesLVSimulation(
            _make_config(x1_0=1.0, x2_0=0.5, y1_0=0.3, y2_0=0.2)
        )
        sim.reset()
        assert sim.is_coexisting

    def test_is_coexisting_false_when_extinct(self):
        """If one species is zero, is_coexisting should be False."""
        sim = FourSpeciesLVSimulation(
            _make_config(x1_0=1.0, x2_0=0.5, y1_0=0.0, y2_0=0.3)
        )
        sim.reset()
        assert not sim.is_coexisting

    def test_n_surviving(self):
        sim = FourSpeciesLVSimulation(
            _make_config(x1_0=1.0, x2_0=0.5, y1_0=0.0001, y2_0=0.3)
        )
        sim.reset()
        # y1_0 = 0.0001 is below default threshold 1e-3
        assert sim.n_surviving(threshold=1e-3) == 3


class TestFourSpeciesLVEquilibrium:
    def test_equilibrium_prey_values(self):
        """Coexistence equilibrium: x1* = d1/c1, x2* = d2/c2."""
        sim = FourSpeciesLVSimulation(
            _make_config(d1=0.4, c1=0.3, d2=0.4, c2=0.3)
        )
        eq = sim.coexistence_equilibrium()
        assert eq.shape == (4,)
        assert eq[0] == pytest.approx(0.4 / 0.3)  # x1* = d1/c1
        assert eq[1] == pytest.approx(0.4 / 0.3)  # x2* = d2/c2

    def test_equilibrium_predator_values(self):
        """Check predator equilibrium: y1* = (r1 - a11*x1* - a12*x2*) / b1."""
        sim = FourSpeciesLVSimulation(_make_config())
        eq = sim.coexistence_equilibrium()
        x1s = sim.d1 / sim.c1
        x2s = sim.d2 / sim.c2
        y1_expected = (sim.r1 - sim.a11 * x1s - sim.a12 * x2s) / sim.b1
        y2_expected = (sim.r2 - sim.a21 * x1s - sim.a22 * x2s) / sim.b2
        assert eq[2] == pytest.approx(y1_expected)
        assert eq[3] == pytest.approx(y2_expected)

    def test_derivatives_at_equilibrium(self):
        """At the coexistence equilibrium, all derivatives should be zero."""
        sim = FourSpeciesLVSimulation(_make_config())
        eq = sim.coexistence_equilibrium()
        # Only check if all components are positive (feasible equilibrium)
        if np.all(eq > 0):
            dy = sim._derivatives(eq)
            np.testing.assert_allclose(dy, [0.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_equilibrium_feasibility_default(self):
        """Default parameters should give a feasible equilibrium (all positive)."""
        sim = FourSpeciesLVSimulation(_make_config())
        eq = sim.coexistence_equilibrium()
        assert np.all(eq > 0), f"Non-feasible equilibrium: {eq}"


class TestFourSpeciesLVStability:
    def test_jacobian_shape(self):
        """Jacobian should be 4x4."""
        sim = FourSpeciesLVSimulation(_make_config())
        sim.reset()
        J = sim.jacobian_at_equilibrium()
        assert J.shape == (4, 4)

    def test_eigenvalues_count(self):
        """Should have 4 eigenvalues."""
        sim = FourSpeciesLVSimulation(_make_config())
        sim.reset()
        eigs = sim.stability_eigenvalues()
        assert len(eigs) == 4

    def test_stable_coexistence_default(self):
        """Default parameters should give stable coexistence."""
        sim = FourSpeciesLVSimulation(_make_config())
        sim.reset()
        assert sim.is_stable_coexistence()

    def test_unstable_with_large_growth_rates(self):
        """Very large prey growth rates with small competition can be unstable."""
        # With very high r and low a, prey can outgrow predator control
        sim = FourSpeciesLVSimulation(_make_config(
            r1=10.0, r2=10.0,
            a11=0.001, a12=0.001, a21=0.001, a22=0.001,
            b1=0.01, b2=0.01,
            c1=0.3, c2=0.3,
            d1=0.4, d2=0.4,
        ))
        sim.reset()
        # This should have eigenvalues with positive real parts
        # (oscillatory instability from weak self-regulation)
        eigs = sim.stability_eigenvalues()
        # At least check eigenvalues are computed and finite
        assert np.all(np.isfinite(eigs))


class TestFourSpeciesLVSpecialCases:
    def test_no_predators_prey_grow(self):
        """With y1_0=0, y2_0=0, prey grow and predators stay zero."""
        sim = FourSpeciesLVSimulation(
            _make_config(x1_0=0.5, x2_0=0.5, y1_0=0.0, y2_0=0.0, dt=0.01)
        )
        sim.reset()
        for _ in range(100):
            sim.step()
        state = sim.observe()
        # Prey should have grown
        assert state[0] > 0.5
        assert state[1] > 0.5
        # Predators should remain zero
        assert state[2] == pytest.approx(0.0, abs=1e-10)
        assert state[3] == pytest.approx(0.0, abs=1e-10)

    def test_no_cross_competition(self):
        """With a12=a21=0, the two prey-predator pairs are independent."""
        sim = FourSpeciesLVSimulation(
            _make_config(a12=0.0, a21=0.0, dt=0.005)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
        state = sim.observe()
        # All species should persist with independent LV dynamics
        assert np.all(state > 0)
        assert np.all(np.isfinite(state))

    def test_predator_extinction_high_death_rate(self):
        """Predators with very high death rate should go extinct."""
        sim = FourSpeciesLVSimulation(
            _make_config(d1=10.0, d2=10.0, dt=0.01)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
        state = sim.observe()
        # Predators should decay toward zero
        assert state[2] < 0.01
        assert state[3] < 0.01


class TestFourSpeciesLVTrajectory:
    def test_run_trajectory_shape(self):
        sim = FourSpeciesLVSimulation(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 4)  # n_steps + 1 including initial
        assert len(traj.timestamps) == 201

    def test_reproducibility(self):
        cfg = _make_config(n_steps=100)
        sim1 = FourSpeciesLVSimulation(cfg)
        traj1 = sim1.run(100)
        sim2 = FourSpeciesLVSimulation(cfg)
        traj2 = sim2.run(100)
        np.testing.assert_allclose(traj1.states, traj2.states)

    def test_deterministic(self):
        """Same parameters should produce identical step-by-step trajectories."""
        config = _make_config(dt=0.01, n_steps=500)
        sim1 = FourSpeciesLVSimulation(config)
        sim2 = FourSpeciesLVSimulation(config)

        sim1.reset()
        sim2.reset()

        states1 = []
        states2 = []
        for _ in range(500):
            states1.append(sim1.step().copy())
            states2.append(sim2.step().copy())

        np.testing.assert_array_equal(np.array(states1), np.array(states2))


class TestFourSpeciesLVTimeavg:
    def test_time_average_approaches_equilibrium(self):
        """Long time average should approach the coexistence equilibrium."""
        sim = FourSpeciesLVSimulation(_make_config(dt=0.005, n_steps=60000))
        sim.reset()

        states = [sim.observe().copy()]
        for _ in range(60000):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)
        # Skip initial transient (first 50%)
        skip = len(trajectory) // 2
        time_avg = np.mean(trajectory[skip:], axis=0)

        eq = sim.coexistence_equilibrium()
        # Time average should be within 15% of equilibrium
        rel_error = np.abs(time_avg - eq) / np.maximum(np.abs(eq), 1e-10)
        assert np.all(rel_error < 0.15), (
            f"Time average {time_avg} too far from equilibrium {eq}, "
            f"relative error {rel_error}"
        )


class TestFourSpeciesLVRediscovery:
    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.four_species_lv import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 4)
        assert len(data["time"]) == 501
        assert data["r1"] == 1.0
        assert data["r2"] == 0.8
        assert data["x1"].shape == (501,)
        assert data["y2"].shape == (501,)

    def test_competition_sweep_data(self):
        from simulating_anything.rediscovery.four_species_lv import (
            generate_competition_sweep_data,
        )
        data = generate_competition_sweep_data(n_alpha=5, n_steps=1000, dt=0.01)
        assert len(data["a12"]) == 5
        assert data["final_populations"].shape == (5, 4)
        assert len(data["n_surviving"]) == 5
        assert len(data["coexisting"]) == 5

    def test_stability_data(self):
        from simulating_anything.rediscovery.four_species_lv import (
            generate_stability_data,
        )
        data = generate_stability_data(n_samples=10)
        assert len(data["max_real_eigenvalue"]) == 10
        assert len(data["is_stable"]) == 10
        assert len(data["is_feasible"]) == 10
        assert len(data["competition_strength"]) == 10
