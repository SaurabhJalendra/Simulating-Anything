"""Tests for the bouncing ball simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.bouncing_ball import BouncingBallSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    e: float = 0.5,
    A: float = 0.1,
    omega: float = 2 * np.pi,
    g: float = 9.81,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.BOUNCING_BALL,
        dt=1.0,
        n_steps=500,
        parameters={"e": e, "A": A, "omega": omega, "g": g},
    )


class TestBouncingBallSimulation:
    def test_reset(self):
        """Reset returns a valid 2-element state."""
        sim = BouncingBallSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        assert state[0] == 0.0  # initial phase
        assert state[1] > 0  # positive initial velocity

    def test_observe_shape(self):
        """Observe returns 2-element state [phase, velocity]."""
        sim = BouncingBallSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_step_advances(self):
        """State changes after a step."""
        sim = BouncingBallSimulation(_make_config())
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_deterministic(self):
        """Same IC produces same trajectory."""
        config = _make_config()
        sim1 = BouncingBallSimulation(config)
        sim2 = BouncingBallSimulation(config)
        sim1.reset()
        sim2.reset()
        for _ in range(50):
            s1 = sim1.step()
            s2 = sim2.step()
            np.testing.assert_array_equal(s1, s2)

    def test_phase_mod1(self):
        """Phase stays in [0, 1) after many steps."""
        sim = BouncingBallSimulation(_make_config(A=0.3))
        sim.reset()
        for _ in range(200):
            state = sim.step()
            assert 0 <= state[0] < 1.0, f"phase={state[0]} outside [0, 1)"

    def test_restitution(self):
        """With e=0 (perfectly inelastic), velocity comes only from table."""
        config = _make_config(e=0.0, A=0.1)
        sim = BouncingBallSimulation(config)
        sim.reset()
        # With e=0 the restitution term e*v = 0, so v_new = A*omega*cos(...)
        # The velocity should be bounded by A*omega
        A_omega = 0.1 * 2 * np.pi
        for _ in range(100):
            state = sim.step()
            # Velocity should be at most (1+e)*A*omega = A*omega for e=0
            assert abs(state[1]) <= A_omega + 0.01

    def test_restitution_effect(self):
        """With e=1 (elastic), no energy is lost in the collision."""
        config_elastic = _make_config(e=1.0, A=0.1)
        config_inelastic = _make_config(e=0.3, A=0.1)
        sim_e = BouncingBallSimulation(config_elastic)
        sim_i = BouncingBallSimulation(config_inelastic)
        sim_e.reset()
        sim_i.reset()

        # Run both for some steps and compare average velocities
        v_elastic = []
        v_inelastic = []
        for _ in range(100):
            se = sim_e.step()
            si = sim_i.step()
            v_elastic.append(abs(se[1]))
            v_inelastic.append(abs(si[1]))

        # Elastic should have higher average velocity
        assert np.mean(v_elastic) > np.mean(v_inelastic)

    def test_velocity_bounded(self):
        """Velocity stays bounded for moderate amplitude."""
        sim = BouncingBallSimulation(_make_config(A=0.1))
        sim.reset()
        for _ in range(500):
            state = sim.step()
            # Velocity should not diverge to infinity
            assert abs(state[1]) < 100, f"velocity={state[1]} too large"

    def test_energy_loss(self):
        """With e < 1, average velocity should decrease or stay bounded.

        For e < 1 without table driving (A=0), velocity decays to zero.
        """
        # Use A=0 to test pure energy loss
        config = _make_config(e=0.5, A=0.0)
        sim = BouncingBallSimulation(config)
        # Manually set initial velocity
        sim.reset()
        sim._state = np.array([0.0, 5.0], dtype=np.float64)

        # With A=0, v_{n+1} = e * v_n, so velocity should decay
        velocities = [5.0]
        for _ in range(10):
            state = sim.step()
            velocities.append(state[1])

        # Each velocity should be e times the previous
        for i in range(1, len(velocities)):
            expected = 0.5 * velocities[i - 1]
            assert abs(velocities[i] - expected) < 0.01

    def test_period1(self):
        """At low amplitude, orbit should converge to period 1."""
        config = _make_config(A=0.005, e=0.5)
        sim = BouncingBallSimulation(config)
        sim.reset()
        period = sim.period_detection(n_steps=500, n_transient=300)
        assert period == 1, f"Expected period 1, got {period}"

    def test_sticking_detection(self):
        """Sticking detected when velocity is very low."""
        sim = BouncingBallSimulation(_make_config())
        sim.reset()
        sim._state = np.array([0.0, 0.001], dtype=np.float64)
        assert sim.is_sticking(velocity_threshold=0.01)

    def test_sticking_not_detected(self):
        """Sticking not detected when velocity is above threshold."""
        sim = BouncingBallSimulation(_make_config())
        sim.reset()
        sim._state = np.array([0.0, 1.0], dtype=np.float64)
        assert not sim.is_sticking(velocity_threshold=0.01)

    def test_bifurcation_diagram(self):
        """Bifurcation diagram produces valid data."""
        sim = BouncingBallSimulation(_make_config())
        A_vals = np.linspace(0.01, 0.3, 5)
        data = sim.bifurcation_diagram(A_vals, n_skip=50, n_record=10)
        assert len(data["A"]) == 50  # 5 A * 10 record
        assert len(data["velocity"]) == 50
        assert np.all(np.isfinite(data["A"]))
        assert np.all(np.isfinite(data["velocity"]))

    def test_period_detection(self):
        """Period detection returns a positive integer for stable orbit."""
        config = _make_config(A=0.01, e=0.5)
        sim = BouncingBallSimulation(config)
        sim.reset()
        period = sim.period_detection(n_steps=500, n_transient=300)
        assert period >= 1

    def test_lyapunov(self):
        """Lyapunov exponent is computable and finite."""
        config = _make_config(A=0.3, e=0.5)
        sim = BouncingBallSimulation(config)
        sim.reset()
        lam = sim.compute_lyapunov(n_steps=1000, n_transient=200)
        assert np.isfinite(lam)

    def test_amplitude_effect(self):
        """Higher amplitude creates more complex dynamics (higher Lyapunov)."""
        lyapunovs = []
        for A_val in [0.01, 0.2, 0.5]:
            config = _make_config(A=A_val, e=0.5)
            sim = BouncingBallSimulation(config)
            sim.reset()
            lam = sim.compute_lyapunov(n_steps=2000, n_transient=300)
            lyapunovs.append(lam)

        # Lyapunov should generally increase with amplitude
        # At minimum, highest A should have higher Lyapunov than lowest A
        assert lyapunovs[2] > lyapunovs[0], (
            f"Lyapunov at A=0.5 ({lyapunovs[2]:.4f}) should exceed "
            f"Lyapunov at A=0.01 ({lyapunovs[0]:.4f})"
        )

    def test_run_trajectory(self):
        """run() produces a valid TrajectoryData."""
        config = _make_config(A=0.1, e=0.5)
        config = config.model_copy(update={"n_steps": 50})
        sim = BouncingBallSimulation(config)
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 2)  # 50 steps + initial
        assert np.all(np.isfinite(traj.states))


class TestBouncingBallRediscovery:
    def test_bifurcation_data(self):
        """Bifurcation data generation works."""
        from simulating_anything.rediscovery.bouncing_ball import (
            generate_bifurcation_data,
        )

        data = generate_bifurcation_data(n_A=5, A_min=0.01, A_max=0.3)
        assert len(data["A_values"]) == 5
        assert len(data["periods"]) == 5
        assert len(data["bif_A"]) > 0
        assert len(data["bif_v"]) > 0

    def test_lyapunov_data(self):
        """Lyapunov data generation works."""
        from simulating_anything.rediscovery.bouncing_ball import (
            generate_lyapunov_data,
        )

        data = generate_lyapunov_data(n_A=5, A_min=0.01, A_max=0.3)
        assert len(data["A"]) == 5
        assert len(data["lyapunov"]) == 5
        assert np.all(np.isfinite(data["lyapunov"]))

    def test_period_doubling_identification(self):
        """Period-doubling identification returns valid structure."""
        from simulating_anything.rediscovery.bouncing_ball import (
            identify_period_doubling,
        )

        # Synthetic data: periods increasing with A
        periods = np.array([1, 1, 1, 2, 2, 4, 4, -1, -1, -1])
        A_values = np.linspace(0.01, 0.5, 10)
        result = identify_period_doubling(periods, A_values)
        assert "bifurcation_points" in result
        assert len(result["bifurcation_points"]) >= 2
