"""Tests for the Rosenzweig-MacArthur predator-prey simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.rosenzweig_macarthur import RosenzweigMacArthur
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 1.0,
    K: float = 10.0,
    a: float = 0.5,
    h: float = 0.5,
    e: float = 0.5,
    d: float = 0.1,
    x_0: float = 1.0,
    y_0: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.ROSENZWEIG_MACARTHUR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "h": h, "e": e, "d": d,
            "x_0": x_0, "y_0": y_0,
        },
    )


class TestRosenzweigMacArthurCreation:
    def test_creation_default_params(self):
        sim = RosenzweigMacArthur(_make_config())
        assert sim.r == 1.0
        assert sim.K == 10.0
        assert sim.a == 0.5
        assert sim.h == 0.5
        assert sim.e == 0.5
        assert sim.d == 0.1

    def test_creation_custom_params(self):
        sim = RosenzweigMacArthur(_make_config(r=2.0, K=20.0, e=0.8))
        assert sim.r == 2.0
        assert sim.K == 20.0
        assert sim.e == 0.8

    def test_initial_state_shape(self):
        sim = RosenzweigMacArthur(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        sim = RosenzweigMacArthur(_make_config(x_0=3.0, y_0=2.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [3.0, 2.0])

    def test_observe_shape(self):
        sim = RosenzweigMacArthur(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)


class TestRosenzweigMacArthurProperties:
    def test_total_population(self):
        sim = RosenzweigMacArthur(_make_config(x_0=3.0, y_0=2.0))
        sim.reset()
        assert sim.total_population == pytest.approx(5.0)

    def test_prey_population(self):
        sim = RosenzweigMacArthur(_make_config(x_0=3.0, y_0=2.0))
        sim.reset()
        assert sim.prey_population == pytest.approx(3.0)

    def test_predator_population(self):
        sim = RosenzweigMacArthur(_make_config(x_0=3.0, y_0=2.0))
        sim.reset()
        assert sim.predator_population == pytest.approx(2.0)

    def test_properties_before_reset(self):
        sim = RosenzweigMacArthur(_make_config())
        assert sim.total_population == 0.0
        assert sim.prey_population == 0.0
        assert sim.predator_population == 0.0


class TestRosenzweigMacArthurEquilibrium:
    def test_coexistence_equilibrium_exists(self):
        """With default params, coexistence equilibrium should exist."""
        sim = RosenzweigMacArthur(_make_config())
        x_star, y_star = sim.coexistence_equilibrium()
        assert x_star > 0
        assert y_star > 0

    def test_coexistence_equilibrium_values(self):
        """Verify equilibrium formula: x* = d/(e*a - a*h*d)."""
        sim = RosenzweigMacArthur(_make_config(
            r=1.0, K=10.0, a=0.5, h=0.5, e=0.5, d=0.1,
        ))
        x_star, y_star = sim.coexistence_equilibrium()
        # x* = 0.1 / (0.25 - 0.025) = 0.1 / 0.225 = 0.4444...
        expected_x = 0.1 / (0.5 * 0.5 - 0.5 * 0.5 * 0.1)
        assert x_star == pytest.approx(expected_x, rel=1e-10)

    def test_derivatives_near_zero_at_equilibrium(self):
        """At coexistence equilibrium, derivatives should be near zero."""
        sim = RosenzweigMacArthur(_make_config())
        x_star, y_star = sim.coexistence_equilibrium()
        state = np.array([x_star, y_star])
        dy = sim._derivatives(state)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-10)

    def test_no_coexistence_when_predator_unsustainable(self):
        """When e*a <= a*h*d, no coexistence equilibrium exists."""
        # e*a = 0.01*0.5 = 0.005, a*h*d = 0.5*0.5*0.1 = 0.025
        sim = RosenzweigMacArthur(_make_config(e=0.01))
        with pytest.raises(ValueError, match="cannot sustain"):
            sim.coexistence_equilibrium()

    def test_critical_K_value(self):
        """K_c = 2*d / (e*a - a*h*d) with defaults."""
        sim = RosenzweigMacArthur(_make_config())
        K_c = sim.critical_K()
        # K_c = 2*0.1 / (0.25 - 0.025) = 0.2/0.225 = 0.8889
        expected = 2.0 * 0.1 / (0.5 * 0.5 - 0.5 * 0.5 * 0.1)
        assert K_c == pytest.approx(expected, rel=1e-10)


class TestRosenzweigMacArthurStability:
    def test_stable_for_small_K(self):
        """With K just above K_c threshold but x* > K/2, equilibrium is stable."""
        # K_c ~ 0.889, use K=0.9 so x*=0.444 < K/2=0.45 -- actually unstable
        # We need K small enough that x* > K/2
        # x* = 0.444 always (independent of K). Stable when K < 2*x* = 0.889
        # So use a scenario where x* is larger.
        # With d=0.5, e=1.0, a=1.0, h=0.1: x* = 0.5/(1.0-0.1*0.5) = 0.5/0.95 = 0.526
        # K_c = 2*0.526 = 1.053. Use K=0.6 (x*=0.526 > 0.3=K/2)
        sim = RosenzweigMacArthur(_make_config(
            r=1.0, K=0.6, a=1.0, h=0.1, e=1.0, d=0.5,
        ))
        assert sim.is_stable()

    def test_unstable_for_large_K(self):
        """With default params and K=10, equilibrium should be unstable (limit cycle)."""
        sim = RosenzweigMacArthur(_make_config(K=10.0))
        # x* = 0.444, K/2 = 5.0; x* < K/2 so unstable
        assert not sim.is_stable()


class TestRosenzweigMacArthurDynamics:
    def test_step_advances_state(self):
        sim = RosenzweigMacArthur(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_populations_non_negative(self):
        """All populations should remain non-negative."""
        sim = RosenzweigMacArthur(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_bounded(self):
        """Populations should not blow up to infinity."""
        sim = RosenzweigMacArthur(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            # Prey bounded by K, predator should also remain bounded
            assert np.all(state < 100), f"Population blow-up: {state}"

    def test_limit_cycle_for_large_K(self):
        """With K >> K_c, should see oscillatory behavior (limit cycle)."""
        sim = RosenzweigMacArthur(_make_config(
            K=10.0, dt=0.01, n_steps=20000, x_0=1.0, y_0=1.0,
        ))
        sim.reset()

        states = []
        for _ in range(20000):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)
        # Check second half for oscillations
        half = len(trajectory) // 2
        x_half = trajectory[half:, 0]
        amplitude = np.max(x_half) - np.min(x_half)
        # With K=10 >> K_c~0.889, should have significant oscillation
        assert amplitude > 0.5, f"Expected limit cycle, amplitude={amplitude:.4f}"


class TestRosenzweigMacArthurTrajectory:
    def test_run_trajectory_shape(self):
        sim = RosenzweigMacArthur(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 2)  # n_steps + 1 including initial
        assert len(traj.timestamps) == 201

    def test_reproducibility_with_same_config(self):
        cfg = _make_config(n_steps=100)
        sim1 = RosenzweigMacArthur(cfg)
        traj1 = sim1.run(100)
        sim2 = RosenzweigMacArthur(cfg)
        traj2 = sim2.run(100)
        np.testing.assert_allclose(traj1.states, traj2.states)


class TestRosenzweigMacArthurRediscovery:
    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.rosenzweig_macarthur import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["r"] == 1.0
        assert data["K"] == 10.0
        assert data["a"] == 0.5

    def test_bifurcation_data_generation(self):
        from simulating_anything.rediscovery.rosenzweig_macarthur import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(n_K_values=5, n_steps=500, dt=0.01)
        assert len(data["K"]) == 5
        assert len(data["x_avg"]) == 5
        assert len(data["x_amplitude"]) == 5
        assert len(data["y_amplitude"]) == 5
