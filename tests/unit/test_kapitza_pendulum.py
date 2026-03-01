"""Tests for the Kapitza pendulum simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.kapitza_pendulum import KapitzaPendulumSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    L: float = 1.0,
    g: float = 9.81,
    a: float = 0.1,
    omega: float = 50.0,
    gamma: float = 0.1,
    theta_0: float = 0.1,
    theta_dot_0: float = 0.0,
    dt: float = 0.0001,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.KAPITZA_PENDULUM,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "L": L,
            "g": g,
            "a": a,
            "omega": omega,
            "gamma": gamma,
            "theta_0": theta_0,
            "theta_dot_0": theta_dot_0,
        },
    )


class TestKapitzaPendulumSimulation:
    def test_initial_state(self):
        """Reset should return [theta_0, theta_dot_0, t=0]."""
        sim = KapitzaPendulumSimulation(_make_config(theta_0=0.2))
        state = sim.reset()
        assert state.shape == (3,)
        np.testing.assert_allclose(state, [0.2, 0.0, 0.0])

    def test_step_advances(self):
        """After one step, state should differ from initial."""
        sim = KapitzaPendulumSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_time_advances(self):
        """Time component should increase with each step."""
        sim = KapitzaPendulumSimulation(_make_config(dt=0.0001))
        sim.reset()
        sim.step()
        state = sim.observe()
        assert state[2] == pytest.approx(0.0001, rel=1e-6)

    def test_deterministic(self):
        """Two runs with same config produce identical trajectories."""
        config = _make_config()
        sim1 = KapitzaPendulumSimulation(config)
        sim2 = KapitzaPendulumSimulation(config)

        sim1.reset()
        sim2.reset()
        for _ in range(100):
            sim1.step()
            sim2.step()

        np.testing.assert_allclose(sim1.observe(), sim2.observe())

    def test_normal_position_stable(self):
        """theta=0 should be a stable equilibrium -- pendulum returns to bottom."""
        sim = KapitzaPendulumSimulation(
            _make_config(theta_0=0.1, a=0.05, omega=50.0, gamma=0.1)
        )
        sim.reset()
        for _ in range(50000):
            sim.step()
        # With damping, theta should decrease toward 0
        theta = sim._state[0]
        assert abs(theta) < 0.5, f"theta={theta} not near 0 (normal position)"

    def test_inverted_unstable_weak_forcing(self):
        """With weak forcing (a*omega small), inverted position should be unstable."""
        # a*omega = 0.5, criterion = 0.5^2/(2*9.81*1) = 0.0127 << 1
        sim = KapitzaPendulumSimulation(
            _make_config(theta_0=np.pi - 0.05, a=0.01, omega=50.0, gamma=0.1)
        )
        sim.reset()
        for _ in range(100000):
            sim.step()
        # Should have fallen away from pi
        theta = sim._state[0]
        deviation = abs(theta - np.pi)
        deviation = min(deviation, 2 * np.pi - deviation)
        assert deviation > 0.5, (
            f"Pendulum stayed near pi (dev={deviation:.3f}) "
            f"despite weak forcing -- should be unstable"
        )

    def test_inverted_stable_strong_forcing(self):
        """With strong forcing (a*omega >> sqrt(2*g*L)), inverted position is stable."""
        # a*omega = 0.5*80 = 40, criterion = 40^2/(2*9.81*1) = 81.5 >> 1
        sim = KapitzaPendulumSimulation(
            _make_config(theta_0=np.pi - 0.05, a=0.5, omega=80.0, gamma=0.1)
        )
        sim.reset()
        for _ in range(200000):
            sim.step()
        # Average over fast oscillation to get slow angle
        thetas = []
        for _ in range(5000):
            sim.step()
            thetas.append(sim._state[0])
        mean_theta = np.mean(thetas)
        deviation = abs(mean_theta - np.pi)
        deviation = min(deviation, 2 * np.pi - deviation)
        assert deviation < 0.5, (
            f"Inverted pendulum drifted (dev={deviation:.3f}) "
            f"despite strong forcing -- Kapitza effect should stabilize"
        )

    def test_stability_criterion_value(self):
        """stability_criterion() should return a^2*omega^2/(2*g*L)."""
        sim = KapitzaPendulumSimulation(
            _make_config(a=0.1, omega=50.0, L=1.0, g=9.81)
        )
        expected = (0.1**2 * 50.0**2) / (2 * 9.81 * 1.0)
        assert sim.stability_criterion() == pytest.approx(expected, rel=1e-10)

    def test_stability_criterion_above_one(self):
        """When criterion > 1, inverted position should be theoretically stable."""
        # a=0.5, omega=80: criterion = 0.25*6400/(2*9.81) = 81.5
        sim = KapitzaPendulumSimulation(_make_config(a=0.5, omega=80.0))
        assert sim.stability_criterion() > 1.0

    def test_stability_criterion_below_one(self):
        """When criterion < 1, inverted position should be unstable."""
        # a=0.01, omega=50: criterion = 0.0001*2500/(2*9.81) = 0.0127
        sim = KapitzaPendulumSimulation(_make_config(a=0.01, omega=50.0))
        assert sim.stability_criterion() < 1.0

    def test_effective_potential_shape(self):
        """V_eff should have correct shape and physical behavior."""
        sim = KapitzaPendulumSimulation(_make_config(a=0.5, omega=80.0))
        theta = np.linspace(-np.pi, np.pi, 100)
        V = sim.effective_potential(theta)
        assert V.shape == (100,)
        # V_eff(0) should be a local minimum (always)
        V_0 = sim.effective_potential(np.array([0.0]))[0]
        V_near = sim.effective_potential(np.array([0.1]))[0]
        assert V_0 < V_near, "V_eff should have minimum at theta=0"

    def test_effective_potential_pi_minimum_strong(self):
        """When criterion > 1, V_eff should have local minimum at theta=pi."""
        # a*omega = 40: second derivative at pi = g*L - (a*omega)^2/(2*L)
        # = 9.81 - 800 < 0, so it is a minimum
        sim = KapitzaPendulumSimulation(_make_config(a=0.5, omega=80.0))
        theta_near_pi = np.array([np.pi - 0.1, np.pi, np.pi + 0.1])
        V = sim.effective_potential(theta_near_pi)
        # V(pi) should be less than V(pi +/- 0.1)
        assert V[1] < V[0], "V_eff(pi) should be local minimum for strong forcing"
        assert V[1] < V[2], "V_eff(pi) should be local minimum for strong forcing"

    def test_effective_potential_pi_no_minimum_weak(self):
        """When criterion < 1, V_eff should NOT have minimum at theta=pi."""
        # a*omega = 0.5: second derivative at pi = 9.81 - 0.125 > 0 => maximum
        sim = KapitzaPendulumSimulation(_make_config(a=0.01, omega=50.0))
        theta_near_pi = np.array([np.pi - 0.1, np.pi, np.pi + 0.1])
        V = sim.effective_potential(theta_near_pi)
        # V(pi) should be greater than V(pi +/- 0.1) => maximum, not minimum
        assert V[1] > V[0], "V_eff(pi) should be a maximum for weak forcing"
        assert V[1] > V[2], "V_eff(pi) should be a maximum for weak forcing"

    def test_energy_dissipation(self):
        """With damping, mechanical energy should decrease over time."""
        sim = KapitzaPendulumSimulation(
            _make_config(theta_0=1.0, gamma=0.5, a=0.0, omega=50.0)
        )
        sim.reset()
        E0 = sim.mechanical_energy
        for _ in range(50000):
            sim.step()
        E1 = sim.mechanical_energy
        assert E1 < E0, f"Energy should decrease with damping: E0={E0}, E1={E1}"

    def test_trajectory_bounded(self):
        """State should remain bounded during simulation."""
        sim = KapitzaPendulumSimulation(_make_config(theta_0=0.5, a=0.1, omega=50.0))
        sim.reset()
        for _ in range(10000):
            sim.step()
            theta, theta_dot = sim._state
            assert abs(theta) < 100, f"theta diverged: {theta}"
            assert abs(theta_dot) < 1000, f"theta_dot diverged: {theta_dot}"

    def test_run_trajectory(self):
        """run() should produce valid TrajectoryData."""
        sim = KapitzaPendulumSimulation(_make_config(n_steps=100))
        traj = sim.run(100)
        assert traj.states.shape == (101, 3)  # 101 = initial + 100 steps
        assert len(traj.timestamps) == 101

    def test_derivatives_at_origin(self):
        """At theta=0, theta_dot=0, t=0: gravity term = 0, forcing = 0."""
        sim = KapitzaPendulumSimulation(_make_config())
        y = np.array([0.0, 0.0])
        dy = sim._derivatives(y, 0.0)
        # dtheta/dt = theta_dot = 0
        # dtheta_dot/dt = -(g/L)*sin(0) - gamma*0 + (a*omega^2/L)*cos(0)*sin(0) = 0
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-10)

    def test_derivatives_gravity_term(self):
        """At theta=pi/2, gravity should pull pendulum down."""
        sim = KapitzaPendulumSimulation(
            _make_config(a=0.0, gamma=0.0)  # No forcing, no damping
        )
        y = np.array([np.pi / 2, 0.0])
        dy = sim._derivatives(y, 0.0)
        # dtheta/dt = 0
        assert dy[0] == pytest.approx(0.0)
        # dtheta_dot/dt = -(g/L)*sin(pi/2) = -9.81
        assert dy[1] == pytest.approx(-9.81, rel=1e-6)

    def test_check_inverted_stability_strong(self):
        """check_inverted_stability should report stable for strong forcing."""
        sim = KapitzaPendulumSimulation(
            _make_config(a=0.5, omega=80.0, gamma=0.1)
        )
        sim.reset()
        result = sim.check_inverted_stability(n_steps=200000)
        assert result["is_stable"], (
            f"Should be stable (param={result['stability_parameter']:.2f}), "
            f"but final_deviation={result['final_deviation']:.3f}"
        )
        assert result["stability_parameter"] > 1.0

    def test_check_inverted_stability_weak(self):
        """check_inverted_stability should report unstable for weak forcing."""
        sim = KapitzaPendulumSimulation(
            _make_config(a=0.01, omega=50.0, gamma=0.1)
        )
        sim.reset()
        result = sim.check_inverted_stability(n_steps=100000)
        assert not result["is_stable"], (
            f"Should be unstable (param={result['stability_parameter']:.4f}), "
            f"but final_deviation={result['final_deviation']:.3f}"
        )
        assert result["stability_parameter"] < 1.0


class TestKapitzaRediscovery:
    def test_stability_sweep_data_generation(self):
        """generate_stability_sweep_data should return valid arrays."""
        from simulating_anything.rediscovery.kapitza_pendulum import (
            generate_stability_sweep_data,
        )
        data = generate_stability_sweep_data(
            n_samples=5, a_omega_min=1.0, a_omega_max=10.0, n_steps=10000
        )
        assert len(data["a_omega"]) == 5
        assert len(data["is_stable"]) == 5
        assert len(data["final_deviation"]) == 5
        assert len(data["stability_param"]) == 5
        assert data["critical_a_omega_theory"] > 0

    def test_effective_potential_data(self):
        """generate_effective_potential_data should produce valid V_eff curves."""
        from simulating_anything.rediscovery.kapitza_pendulum import (
            generate_effective_potential_data,
        )
        data = generate_effective_potential_data(n_theta=50, n_a_omega=3)
        assert data["theta"].shape == (50,)
        assert data["V_eff"].shape == (3, 50)
        assert len(data["has_pi_minimum"]) == 3
        assert data["critical_a_omega"] > 0

    def test_criterion_data(self):
        """generate_criterion_data should produce valid parameter samples."""
        from simulating_anything.rediscovery.kapitza_pendulum import (
            generate_criterion_data,
        )
        data = generate_criterion_data(n_samples=10)
        assert len(data["g"]) == 10
        assert len(data["L"]) == 10
        assert len(data["a_omega_sq"]) == 10
        assert len(data["stability_param"]) == 10
        # stability_param = a_omega_sq / (2*g*L)
        expected = data["a_omega_sq"] / (2 * data["g"] * data["L"])
        np.testing.assert_allclose(data["stability_param"], expected, rtol=1e-10)

    def test_bifurcation_data(self):
        """generate_bifurcation_data should return valid sweep data."""
        from simulating_anything.rediscovery.kapitza_pendulum import (
            generate_bifurcation_data,
        )
        data = generate_bifurcation_data(
            n_a_omega=5, a_omega_min=2.0, a_omega_max=8.0, n_steps=10000
        )
        assert len(data["a_omega"]) == 5
        assert len(data["mean_theta"]) == 5
        assert len(data["deviation_from_pi"]) == 5
        assert data["critical_a_omega_theory"] == pytest.approx(
            np.sqrt(2 * 9.81 * 1.0), rel=1e-6
        )
