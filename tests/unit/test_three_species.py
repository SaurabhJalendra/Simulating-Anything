"""Tests for the three-species food chain simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.three_species import ThreeSpecies
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    a1: float = 1.0,
    b1: float = 0.5,
    a2: float = 0.5,
    b2: float = 0.2,
    a3: float = 0.3,
    x0: float = 1.0,
    y0: float = 0.5,
    z0: float = 0.5,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.THREE_SPECIES,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "a1": a1, "b1": b1, "a2": a2, "b2": b2, "a3": a3,
            "x0": x0, "y0": y0, "z0": z0,
        },
    )


class TestThreeSpeciesCreation:
    def test_creation_default_params(self):
        sim = ThreeSpecies(_make_config())
        assert sim.a1 == 1.0
        assert sim.b1 == 0.5
        assert sim.a2 == 0.5
        assert sim.b2 == 0.2
        assert sim.a3 == 0.3

    def test_creation_custom_params(self):
        sim = ThreeSpecies(_make_config(a1=2.0, b2=0.4))
        assert sim.a1 == 2.0
        assert sim.b2 == 0.4

    def test_initial_state_shape(self):
        sim = ThreeSpecies(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_values(self):
        sim = ThreeSpecies(_make_config(x0=2.0, y0=1.0, z0=0.5))
        state = sim.reset()
        np.testing.assert_allclose(state, [2.0, 1.0, 0.5])

    def test_observe_shape(self):
        sim = ThreeSpecies(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)


class TestThreeSpeciesDynamics:
    def test_step_advances_state(self):
        sim = ThreeSpecies(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_populations_non_negative(self):
        """All populations should remain non-negative."""
        sim = ThreeSpecies(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_bounded(self):
        """Populations should not blow up to infinity."""
        sim = ThreeSpecies(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            assert np.all(state < 1000), f"Population blow-up: {state}"

    def test_total_population_property(self):
        sim = ThreeSpecies(_make_config(x0=1.0, y0=0.5, z0=0.5))
        sim.reset()
        assert sim.total_population == pytest.approx(2.0)

    def test_is_coexisting_true(self):
        sim = ThreeSpecies(_make_config(x0=1.0, y0=0.5, z0=0.5))
        sim.reset()
        assert sim.is_coexisting

    def test_is_coexisting_false_when_extinct(self):
        """If one species is zero, is_coexisting should be False."""
        sim = ThreeSpecies(_make_config(x0=1.0, y0=0.5, z0=0.0))
        sim.reset()
        assert not sim.is_coexisting


class TestThreeSpeciesEquilibrium:
    def test_equilibrium_point_values(self):
        """Predator-free equilibrium: x* = a2/b1, y* = a1/b1."""
        sim = ThreeSpecies(_make_config(a1=1.0, b1=0.5, a2=0.5))
        eq = sim.equilibrium_point()
        assert eq.shape == (3,)
        assert eq[0] == pytest.approx(1.0)   # a2/b1 = 0.5/0.5
        assert eq[1] == pytest.approx(2.0)   # a1/b1 = 1.0/0.5
        assert eq[2] == pytest.approx(0.0)   # predator-free

    def test_derivatives_at_equilibrium_without_predator(self):
        """At the predator-free equilibrium, dx/dt and dy/dt should be zero."""
        sim = ThreeSpecies(_make_config(a1=1.0, b1=0.5, a2=0.5))
        eq = sim.equilibrium_point()
        dy = sim._derivatives(eq)
        # dx/dt = a1*x - b1*x*y = 1.0*1.0 - 0.5*1.0*2.0 = 0
        # dy/dt = -a2*y + b1*x*y - b2*y*z = -0.5*2.0 + 0.5*1.0*2.0 - 0 = 0
        # dz/dt = -a3*z + b2*y*z = 0 (z=0)
        np.testing.assert_allclose(dy, [0.0, 0.0, 0.0], atol=1e-10)

    def test_predator_invasion_rate(self):
        """Test predator invasion rate computation."""
        sim = ThreeSpecies(_make_config(a1=1.0, b1=0.5, b2=0.2, a3=0.3))
        # b2 * (a1/b1) - a3 = 0.2 * 2.0 - 0.3 = 0.1
        assert sim.predator_invasion_rate() == pytest.approx(0.1)


class TestThreeSpeciesSpecialCases:
    def test_no_predators_reduces_to_2species_lv(self):
        """With z0=0 and a3 large, z stays zero: standard 2-species LV."""
        sim = ThreeSpecies(_make_config(
            a1=1.1, b1=0.4, a2=0.4, b2=0.1, a3=10.0,
            x0=2.0, y0=1.0, z0=0.0, dt=0.01,
        ))
        sim.reset()
        for _ in range(2000):
            sim.step()
            z = sim.observe()[2]
            assert z == pytest.approx(0.0, abs=1e-10), f"z should stay zero: {z}"

    def test_no_herbivores_dynamics(self):
        """With y0=0, grass grows exponentially and predator decays."""
        sim = ThreeSpecies(_make_config(
            a1=1.0, b1=0.5, a2=0.5, b2=0.2, a3=0.3,
            x0=1.0, y0=0.0, z0=1.0, dt=0.01,
        ))
        sim.reset()
        for _ in range(100):
            sim.step()
        state = sim.observe()
        # Grass should grow (exponential with rate a1)
        assert state[0] > 1.0
        # Herbivore should stay zero
        assert state[1] == pytest.approx(0.0, abs=1e-10)
        # Predator should decay (exponential with rate -a3)
        assert state[2] < 1.0


class TestThreeSpeciesTrajectory:
    def test_run_trajectory_shape(self):
        sim = ThreeSpecies(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 3)  # n_steps + 1 including initial
        assert len(traj.timestamps) == 201

    def test_reproducibility_with_same_seed(self):
        cfg = _make_config(n_steps=100)
        sim1 = ThreeSpecies(cfg)
        traj1 = sim1.run(100)
        sim2 = ThreeSpecies(cfg)
        traj2 = sim2.run(100)
        np.testing.assert_allclose(traj1.states, traj2.states)


class TestThreeSpeciesRediscovery:
    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.three_species import generate_trajectory_data
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert data["a1"] == 1.0
        assert data["b1"] == 0.5

    def test_equilibrium_data_generation(self):
        from simulating_anything.rediscovery.three_species import generate_equilibrium_data
        data = generate_equilibrium_data(n_samples=5, n_steps=500, dt=0.01)
        assert len(data["a1"]) == 5
        assert len(data["x_avg"]) == 5
        assert len(data["y_avg"]) == 5
        assert len(data["z_avg"]) == 5
