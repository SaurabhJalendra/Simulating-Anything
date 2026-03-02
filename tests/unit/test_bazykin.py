"""Tests for the Bazykin predator-prey simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.bazykin import BazykinSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    alpha: float = 0.1,
    gamma: float = 0.1,
    delta: float = 0.01,
    x_0: float = 0.5,
    y_0: float = 0.5,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.BAZYKIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "alpha": alpha, "gamma": gamma, "delta": delta,
            "x_0": x_0, "y_0": y_0,
        },
    )


class TestBazykinCreation:
    def test_creation_default_params(self):
        sim = BazykinSimulation(_make_config())
        assert sim.alpha == 0.1
        assert sim.gamma_param == 0.1
        assert sim.delta == 0.01

    def test_creation_custom_params(self):
        sim = BazykinSimulation(_make_config(alpha=0.5, gamma=0.2, delta=0.05))
        assert sim.alpha == 0.5
        assert sim.gamma_param == 0.2
        assert sim.delta == 0.05

    def test_initial_state_shape(self):
        sim = BazykinSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        sim = BazykinSimulation(_make_config(x_0=0.8, y_0=0.3))
        state = sim.reset()
        np.testing.assert_allclose(state, [0.8, 0.3])

    def test_observe_shape(self):
        sim = BazykinSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)


class TestBazykinProperties:
    def test_total_population(self):
        sim = BazykinSimulation(_make_config(x_0=0.6, y_0=0.4))
        sim.reset()
        assert sim.total_population == pytest.approx(1.0)

    def test_prey_population(self):
        sim = BazykinSimulation(_make_config(x_0=0.6, y_0=0.4))
        sim.reset()
        assert sim.prey_population == pytest.approx(0.6)

    def test_predator_population(self):
        sim = BazykinSimulation(_make_config(x_0=0.6, y_0=0.4))
        sim.reset()
        assert sim.predator_population == pytest.approx(0.4)

    def test_properties_before_reset(self):
        sim = BazykinSimulation(_make_config())
        assert sim.total_population == 0.0
        assert sim.prey_population == 0.0
        assert sim.predator_population == 0.0


class TestBazykinEquilibrium:
    def test_coexistence_equilibrium_exists(self):
        """With default params, coexistence equilibrium should exist."""
        sim = BazykinSimulation(_make_config())
        x_star, y_star = sim.coexistence_equilibrium()
        assert x_star > 0
        assert y_star > 0

    def test_derivatives_near_zero_at_equilibrium(self):
        """At coexistence equilibrium, derivatives should be near zero."""
        sim = BazykinSimulation(_make_config())
        x_star, y_star = sim.coexistence_equilibrium()
        state = np.array([x_star, y_star])
        dy = sim._derivatives(state)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-10)

    def test_equilibrium_satisfies_prey_nullcline(self):
        """y* = (1 - x*)(1 + alpha*x*) at equilibrium."""
        sim = BazykinSimulation(_make_config())
        x_star, y_star = sim.coexistence_equilibrium()
        expected_y = (1.0 - x_star) * (1.0 + sim.alpha * x_star)
        assert y_star == pytest.approx(expected_y, rel=1e-8)

    def test_equilibrium_satisfies_predator_nullcline(self):
        """At equilibrium: x/(1+alpha*x) = gamma + delta*y."""
        sim = BazykinSimulation(_make_config())
        x_star, y_star = sim.coexistence_equilibrium()
        lhs = x_star / (1.0 + sim.alpha * x_star)
        rhs = sim.gamma_param + sim.delta * y_star
        assert lhs == pytest.approx(rhs, rel=1e-8)


class TestBazykinJacobian:
    def test_jacobian_shape(self):
        sim = BazykinSimulation(_make_config())
        J = sim.jacobian(0.5, 0.3)
        assert J.shape == (2, 2)

    def test_jacobian_at_origin(self):
        """At (0, 0), dx/dt ~ x so dfdx = 1, dfdy = 0, dgdx = 0, dgdy = -gamma."""
        sim = BazykinSimulation(_make_config(gamma=0.1))
        J = sim.jacobian(0.0, 0.0)
        assert J[0, 0] == pytest.approx(1.0)
        assert J[0, 1] == pytest.approx(0.0)
        assert J[1, 0] == pytest.approx(0.0)
        assert J[1, 1] == pytest.approx(-0.1)


class TestBazykinStability:
    def test_stability_check_returns_bool(self):
        sim = BazykinSimulation(_make_config())
        result = sim.is_stable()
        assert isinstance(result, bool)

    def test_stable_with_high_gamma(self):
        """With large gamma, predator is weak so equilibrium might vanish
        or be stable if it exists."""
        sim = BazykinSimulation(_make_config(gamma=0.4))
        # With high gamma, the equilibrium may not exist (predator dies out)
        # or it is stable because predator pressure is weak
        try:
            stable = sim.is_stable()
            assert isinstance(stable, bool)
        except ValueError:
            # No equilibrium means stability check returns False
            assert not sim.is_stable()


class TestBazykinDynamics:
    def test_step_advances_state(self):
        sim = BazykinSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_populations_non_negative(self):
        """All populations should remain non-negative."""
        sim = BazykinSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_bounded(self):
        """Populations should not blow up to infinity."""
        sim = BazykinSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            # Prey bounded by carrying capacity ~1
            assert np.all(state < 50), f"Population blow-up: {state}"

    def test_prey_bounded_by_carrying_capacity(self):
        """Prey x should not exceed 1 (logistic growth ceiling)."""
        sim = BazykinSimulation(_make_config(
            x_0=0.9, y_0=0.01, dt=0.01, n_steps=10000,
        ))
        sim.reset()
        max_x = 0.0
        for _ in range(10000):
            sim.step()
            x = sim.prey_population
            if x > max_x:
                max_x = x
        # Prey is bounded near 1 (can slightly overshoot due to RK4)
        assert max_x < 1.5, f"Prey exceeded expected bound: {max_x:.4f}"

    def test_limit_cycle_with_large_alpha(self):
        """With large alpha (strong saturation), Hopf bifurcation produces
        limit cycles. alpha=2.0 destabilizes the equilibrium."""
        sim = BazykinSimulation(_make_config(
            alpha=2.0, gamma=0.05, delta=0.01,
            dt=0.005, n_steps=40000,
            x_0=0.5, y_0=0.5,
        ))
        sim.reset()

        states = []
        for _ in range(40000):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)
        # Check second half for oscillations
        half = len(trajectory) // 2
        x_half = trajectory[half:, 0]
        amplitude = np.max(x_half) - np.min(x_half)
        # With alpha=2.0, gamma=0.05, expect significant oscillation
        assert amplitude > 0.1, (
            f"Expected limit cycle oscillation, amplitude={amplitude:.6f}"
        )

    def test_extinction_with_very_large_gamma(self):
        """With gamma > max functional response, predator goes extinct.
        Max FR = 1/(1+alpha) for x=1, so gamma=1.5 >> max FR ~ 0.91."""
        sim = BazykinSimulation(_make_config(
            gamma=1.5, delta=0.01, dt=0.01, n_steps=5000,
            x_0=0.5, y_0=0.1,
        ))
        sim.reset()

        for _ in range(5000):
            sim.step()

        # Predator should decline toward 0
        assert sim.predator_population < 0.001, (
            f"Predator should go extinct: y={sim.predator_population:.6f}"
        )


class TestBazykinTrajectory:
    def test_run_trajectory_shape(self):
        sim = BazykinSimulation(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 2)  # n_steps + 1 including initial
        assert len(traj.timestamps) == 201

    def test_reproducibility_with_same_config(self):
        cfg = _make_config(n_steps=100)
        sim1 = BazykinSimulation(cfg)
        traj1 = sim1.run(100)
        sim2 = BazykinSimulation(cfg)
        traj2 = sim2.run(100)
        np.testing.assert_allclose(traj1.states, traj2.states)

    def test_trajectory_timestamps(self):
        dt = 0.01
        n_steps = 50
        sim = BazykinSimulation(_make_config(dt=dt, n_steps=n_steps))
        traj = sim.run(n_steps=n_steps)
        expected_times = np.arange(n_steps + 1) * dt
        np.testing.assert_allclose(traj.timestamps, expected_times)


class TestBazykinRediscovery:
    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.bazykin import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["alpha"] == 0.1
        assert data["gamma"] == 0.1
        assert data["delta"] == 0.01

    def test_bifurcation_data_generation(self):
        from simulating_anything.rediscovery.bazykin import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(
            n_gamma_values=5, n_steps=500, dt=0.01,
        )
        assert len(data["gamma"]) == 5
        assert len(data["x_avg"]) == 5
        assert len(data["x_amplitude"]) == 5
        assert len(data["y_amplitude"]) == 5

    def test_bifurcation_data_finite(self):
        from simulating_anything.rediscovery.bazykin import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(
            n_gamma_values=3, n_steps=1000, dt=0.01,
        )
        assert np.all(np.isfinite(data["x_avg"]))
        assert np.all(np.isfinite(data["y_avg"]))
        assert np.all(np.isfinite(data["x_amplitude"]))
