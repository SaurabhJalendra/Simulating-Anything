"""Tests for the Predator-Two-Prey simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.predator_two_prey import (
    PredatorTwoPreySimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r1: float = 1.0,
    r2: float = 0.8,
    K1: float = 10.0,
    K2: float = 8.0,
    a1: float = 0.5,
    a2: float = 0.4,
    b1: float = 0.2,
    b2: float = 0.15,
    d: float = 0.6,
    x1_0: float = 5.0,
    x2_0: float = 4.0,
    y_0: float = 2.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    """Helper to build a SimulationConfig for the predator-two-prey model."""
    return SimulationConfig(
        domain=Domain.PREDATOR_TWO_PREY,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r1": r1, "r2": r2, "K1": K1, "K2": K2,
            "a1": a1, "a2": a2, "b1": b1, "b2": b2, "d": d,
            "x1_0": x1_0, "x2_0": x2_0, "y_0": y_0,
        },
    )


class TestResetState:
    """Tests for initialization and reset."""

    def test_reset_returns_initial_conditions(self):
        """Reset should return the initial state [x1_0, x2_0, y_0]."""
        config = _make_config(x1_0=5.0, x2_0=4.0, y_0=2.0)
        sim = PredatorTwoPreySimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [5.0, 4.0, 2.0])

    def test_reset_custom_initial(self):
        """Custom initial conditions should be respected."""
        config = _make_config(x1_0=1.0, x2_0=2.0, y_0=3.0)
        sim = PredatorTwoPreySimulation(config)
        state = sim.reset()
        np.testing.assert_allclose(state, [1.0, 2.0, 3.0])


class TestObserveShape:
    """Tests for observe() output shape."""

    def test_observe_shape(self):
        """observe() should return 3-element array [x1, x2, y]."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_observe_after_steps(self):
        """observe() shape should be stable after stepping."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        for _ in range(100):
            sim.step()
        assert sim.observe().shape == (3,)


class TestStepAdvances:
    """Tests that step() changes state."""

    def test_step_changes_state(self):
        """State should differ after a step."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)


class TestNonNegative:
    """Populations must remain non-negative."""

    def test_non_negative_populations(self):
        """All populations must stay >= 0."""
        config = _make_config(
            x1_0=0.1, x2_0=0.1, y_0=5.0,
            dt=0.01, n_steps=10000,
        )
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_near_zero_initial(self):
        """Very small initial populations should not go negative."""
        config = _make_config(
            x1_0=0.001, x2_0=0.001, y_0=0.001,
            dt=0.01, n_steps=5000,
        )
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0)


class TestBounded:
    """Populations should stay bounded (no divergence)."""

    def test_total_bounded(self):
        """Total population should not diverge."""
        config = _make_config(dt=0.01, n_steps=20000)
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        for _ in range(20000):
            sim.step()
            total = np.sum(sim.observe())
            assert total < 1e6, f"Population diverged: total={total}"


class TestRK4Stability:
    """Numerical stability tests."""

    def test_no_nan_inf(self):
        """No NaN or Inf after many steps."""
        config = _make_config(dt=0.01, n_steps=50000)
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN/Inf in state: {state}"


class TestDeterministic:
    """Deterministic simulation tests."""

    def test_deterministic(self):
        """Same parameters should produce identical trajectories."""
        config = _make_config(dt=0.01, n_steps=500)
        sim1 = PredatorTwoPreySimulation(config)
        sim2 = PredatorTwoPreySimulation(config)

        sim1.reset()
        sim2.reset()

        states1, states2 = [], []
        for _ in range(500):
            states1.append(sim1.step().copy())
            states2.append(sim2.step().copy())

        np.testing.assert_array_equal(np.array(states1), np.array(states2))


class TestLogisticGrowth:
    """Test that prey follows logistic growth without predator."""

    def test_prey1_logistic_no_predator(self):
        """Prey 1 alone should converge to K1."""
        config = _make_config(
            x1_0=1.0, x2_0=0.0, y_0=0.0,
            dt=0.01, n_steps=20000,
        )
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        for _ in range(20000):
            sim.step()
        x1 = sim.observe()[0]
        assert x1 == pytest.approx(10.0, rel=0.01)

    def test_prey2_logistic_no_predator(self):
        """Prey 2 alone should converge to K2."""
        config = _make_config(
            x1_0=0.0, x2_0=1.0, y_0=0.0,
            dt=0.01, n_steps=20000,
        )
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        for _ in range(20000):
            sim.step()
        x2 = sim.observe()[1]
        assert x2 == pytest.approx(8.0, rel=0.01)


class TestPredatorExtinction:
    """Tests for predator extinction when no prey available."""

    def test_predator_dies_without_prey(self):
        """Predator with no prey should decay to zero."""
        config = _make_config(
            x1_0=0.0, x2_0=0.0, y_0=5.0,
            dt=0.01, n_steps=10000,
        )
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        for _ in range(10000):
            sim.step()
        y = sim.observe()[2]
        assert y < 1e-10


class TestJacobian:
    """Tests for the Jacobian matrix computation."""

    def test_jacobian_shape(self):
        """Jacobian should be 3x3."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        J = sim.jacobian()
        assert J.shape == (3, 3)

    def test_jacobian_at_origin(self):
        """Jacobian at (0,0,0) should be diagonal [r1, r2, -d]."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        J = sim.jacobian(np.array([0.0, 0.0, 0.0]))
        # At origin: J = diag(r1, r2, -d)
        assert J[0, 0] == pytest.approx(1.0)   # r1
        assert J[1, 1] == pytest.approx(0.8)   # r2
        assert J[2, 2] == pytest.approx(-0.6)  # -d
        assert J[0, 1] == pytest.approx(0.0)
        assert J[1, 0] == pytest.approx(0.0)

    def test_jacobian_custom_state(self):
        """Jacobian evaluated at a custom state should be correct."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        # At (5, 4, 1): check specific entries
        state = np.array([5.0, 4.0, 1.0])
        J = sim.jacobian(state)
        # J[0,0] = r1*(1 - 2*x1/K1) - a1*y = 1.0*(1-1.0) - 0.5*1 = -0.5
        assert J[0, 0] == pytest.approx(-0.5)
        # J[0,2] = -a1*x1 = -0.5*5 = -2.5
        assert J[0, 2] == pytest.approx(-2.5)
        # J[2,0] = b1*y = 0.2*1 = 0.2
        assert J[2, 0] == pytest.approx(0.2)
        # J[2,1] = b2*y = 0.15*1 = 0.15
        assert J[2, 1] == pytest.approx(0.15)


class TestFixedPoints:
    """Tests for fixed point computation."""

    def test_origin_is_fixed_point(self):
        """(0, 0, 0) should always be a fixed point."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        fps = sim.fixed_points()
        origins = [fp for fp in fps if np.allclose(fp, [0, 0, 0])]
        assert len(origins) == 1

    def test_prey_only_fixed_points(self):
        """(K1, 0, 0), (0, K2, 0), and (K1, K2, 0) should be fixed points."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        fps = sim.fixed_points()

        has_k1 = any(np.allclose(fp, [10.0, 0.0, 0.0]) for fp in fps)
        has_k2 = any(np.allclose(fp, [0.0, 8.0, 0.0]) for fp in fps)
        has_both = any(np.allclose(fp, [10.0, 8.0, 0.0]) for fp in fps)
        assert has_k1
        assert has_k2
        assert has_both

    def test_fixed_points_are_stationary(self):
        """Derivatives at fixed points should be approximately zero."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        fps = sim.fixed_points()
        for fp in fps:
            deriv = sim._derivatives(fp)
            np.testing.assert_allclose(
                deriv, 0.0, atol=1e-10,
                err_msg=f"Non-zero derivative at fixed point {fp}",
            )

    def test_at_least_four_fixed_points(self):
        """There should be at least 4 fixed points (trivial + boundary)."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        fps = sim.fixed_points()
        assert len(fps) >= 4


class TestDivergence:
    """Tests for vector field divergence."""

    def test_divergence_at_origin(self):
        """Divergence at origin = r1 + r2 - d."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        div = sim.compute_divergence(np.array([0.0, 0.0, 0.0]))
        expected = 1.0 + 0.8 - 0.6  # r1 + r2 - d = 1.2
        assert div == pytest.approx(expected)

    def test_divergence_scalar(self):
        """Divergence should return a scalar."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        div = sim.compute_divergence()
        assert isinstance(div, float)


class TestNSurviving:
    """Tests for species survival counting."""

    def test_all_surviving_initially(self):
        """All 3 species should be present initially with default IC."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        assert sim.n_surviving() == 3

    def test_two_surviving(self):
        """Species below threshold should not be counted."""
        config = _make_config(x1_0=5.0, x2_0=0.0001, y_0=2.0)
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        assert sim.n_surviving(threshold=1e-3) == 2


class TestTotalPopulation:
    """Tests for total population property."""

    def test_total_population(self):
        """Total population should equal sum of all species."""
        config = _make_config(x1_0=5.0, x2_0=4.0, y_0=2.0)
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        assert sim.total_population == pytest.approx(11.0)

    def test_total_population_none_state(self):
        """Total population should be 0.0 before reset."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        assert sim.total_population == 0.0


class TestIsCoexisting:
    """Tests for coexistence detection."""

    def test_coexisting_initially(self):
        """With default IC all species present => coexisting."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        assert sim.is_coexisting

    def test_not_coexisting_when_species_zero(self):
        """Not coexisting when one species is zero."""
        config = _make_config(x2_0=0.0)
        sim = PredatorTwoPreySimulation(config)
        sim.reset()
        assert not sim.is_coexisting


class TestApparentCompetition:
    """Tests for apparent competition index."""

    def test_apparent_competition_index(self):
        """ACI = (b1*a2 + b2*a1) / d."""
        config = _make_config()
        sim = PredatorTwoPreySimulation(config)
        # (0.2*0.4 + 0.15*0.5) / 0.6 = (0.08 + 0.075) / 0.6 = 0.2583...
        expected = (0.2 * 0.4 + 0.15 * 0.5) / 0.6
        assert sim.apparent_competition_index() == pytest.approx(expected)


class TestRunMethod:
    """Tests for the inherited run() method."""

    def test_run_returns_trajectory_data(self):
        """run() should return a TrajectoryData with correct shape."""
        config = _make_config(dt=0.01, n_steps=100)
        sim = PredatorTwoPreySimulation(config)
        traj = sim.run()
        assert traj.states.shape == (101, 3)

    def test_run_all_finite(self):
        """All trajectory states should be finite."""
        config = _make_config(dt=0.01, n_steps=1000)
        sim = PredatorTwoPreySimulation(config)
        traj = sim.run()
        assert np.all(np.isfinite(traj.states))


class TestRediscoveryData:
    """Tests for rediscovery data generation functions."""

    def test_trajectory_data_shape(self):
        """generate_trajectory_data should return valid arrays."""
        from simulating_anything.rediscovery.predator_two_prey import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert len(data["x1"]) == 501
        assert len(data["x2"]) == 501
        assert len(data["y"]) == 501

    def test_coexistence_sweep_shape(self):
        """generate_coexistence_sweep_data should produce valid results."""
        from simulating_anything.rediscovery.predator_two_prey import (
            generate_coexistence_sweep_data,
        )
        data = generate_coexistence_sweep_data(
            n_samples=5, n_steps=1000, dt=0.01,
        )
        assert len(data["a1"]) == 5
        assert len(data["n_surviving"]) == 5
        assert len(data["x1_final"]) == 5

    def test_apparent_competition_data(self):
        """generate_apparent_competition_data should produce valid results."""
        from simulating_anything.rediscovery.predator_two_prey import (
            generate_apparent_competition_data,
        )
        data = generate_apparent_competition_data(
            n_points=5, n_steps=1000, dt=0.01,
        )
        assert len(data["b2"]) == 5
        assert len(data["x1_final"]) == 5
        assert len(data["x2_final"]) == 5
        assert len(data["y_final"]) == 5
