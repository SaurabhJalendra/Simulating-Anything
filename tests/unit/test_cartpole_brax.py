"""Tests for the Brax-style cart-pole simulation."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.cartpole_brax import CartPoleBraxSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    m_c: float = 1.0,
    m_p: float = 0.1,
    L: float = 0.5,
    g: float = 9.81,
    force_mag: float = 10.0,
    mu_c: float = 0.0,
    mu_p: float = 0.0,
    dt: float = 0.001,
    n_steps: int = 5000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CARTPOLE_BRAX,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "m_c": m_c,
            "m_p": m_p,
            "L": L,
            "g": g,
            "force_mag": force_mag,
            "mu_c": mu_c,
            "mu_p": mu_p,
        },
    )


class TestCartPoleBrax:
    """Tests for the CartPoleBraxSimulation."""

    # -- Instantiation --

    def test_creation(self):
        """CartPoleBraxSimulation should be created with correct parameters."""
        sim = CartPoleBraxSimulation(_make_config())
        assert sim.m_c == 1.0
        assert sim.m_p == 0.1
        assert sim.L == 0.5
        assert sim.g == 9.81
        assert sim.force_mag == 10.0
        assert sim.mu_c == 0.0
        assert sim.mu_p == 0.0

    def test_default_params(self):
        """Empty parameters should yield defaults."""
        config = SimulationConfig(
            domain=Domain.CARTPOLE_BRAX,
            dt=0.001,
            n_steps=100,
            parameters={},
        )
        sim = CartPoleBraxSimulation(config)
        assert sim.m_c == 1.0
        assert sim.m_p == 0.1
        assert sim.L == 0.5
        assert sim.g == 9.81
        assert sim.force_mag == 10.0

    def test_custom_params(self):
        """Custom parameters should be stored correctly."""
        sim = CartPoleBraxSimulation(_make_config(m_c=2.0, m_p=0.5, L=1.0, g=10.0))
        assert sim.m_c == 2.0
        assert sim.m_p == 0.5
        assert sim.L == 1.0
        assert sim.g == 10.0

    # -- Reset --

    def test_reset_shape(self):
        """Reset should return state of shape (4,)."""
        sim = CartPoleBraxSimulation(_make_config())
        state = sim.reset(seed=42)
        assert state.shape == (4,)

    def test_reset_no_nan(self):
        """Reset state should contain no NaN values."""
        sim = CartPoleBraxSimulation(_make_config())
        state = sim.reset(seed=42)
        assert not np.any(np.isnan(state))

    def test_reset_near_upright(self):
        """Initial state should be near upright (theta ~ 0)."""
        sim = CartPoleBraxSimulation(_make_config())
        state = sim.reset(seed=42)
        assert abs(state[2]) < 0.1  # theta near zero

    def test_reset_clears_action(self):
        """Reset should clear any previously set action."""
        sim = CartPoleBraxSimulation(_make_config())
        sim.reset(seed=42)
        sim.set_action(0.8)
        sim.reset(seed=42)
        assert sim._action == 0.0

    # -- Step --

    def test_step_no_nan(self):
        """Step should not produce NaN values."""
        sim = CartPoleBraxSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(50):
            state = sim.step()
        assert not np.any(np.isnan(state))

    def test_step_advances_state(self):
        """A single step should change the state."""
        sim = CartPoleBraxSimulation(_make_config())
        s0 = sim.reset(seed=42).copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_observe_matches_step(self):
        """Observe should return the same state as the last step."""
        sim = CartPoleBraxSimulation(_make_config())
        sim.reset(seed=42)
        s1 = sim.step()
        obs = sim.observe()
        assert np.allclose(s1, obs)

    # -- set_action --

    def test_set_action(self):
        """set_action should store the action value."""
        sim = CartPoleBraxSimulation(_make_config())
        sim.reset(seed=42)
        sim.set_action(0.5)
        assert sim._action == 0.5

    def test_set_action_clips(self):
        """set_action should clip values to [-1, 1]."""
        sim = CartPoleBraxSimulation(_make_config())
        sim.reset(seed=42)
        sim.set_action(5.0)
        assert sim._action == 1.0
        sim.set_action(-3.0)
        assert sim._action == -1.0

    def test_action_affects_dynamics(self):
        """Applying a force should produce different dynamics than zero force."""
        sim = CartPoleBraxSimulation(_make_config())

        # Run with zero action
        sim.reset(seed=42)
        sim.set_action(0.0)
        for _ in range(50):
            s_no_force = sim.step()

        # Run with positive action
        sim.reset(seed=42)
        sim.set_action(1.0)
        for _ in range(50):
            s_force = sim.step()

        assert not np.allclose(s_no_force, s_force)

    # -- Zero action: pole falls --

    def test_pole_falls_zero_action(self):
        """With zero action and initial perturbation, theta should grow."""
        sim = CartPoleBraxSimulation(_make_config(dt=0.005))
        state0 = sim.reset(seed=42)
        initial_theta = abs(state0[2])

        # Run for a while with no action
        for _ in range(200):
            sim.step()

        final_theta = abs(sim._state[2])
        # Pole should have fallen further from upright
        assert final_theta > initial_theta

    # -- Energy --

    def test_total_energy_finite(self):
        """Total energy should be finite."""
        sim = CartPoleBraxSimulation(_make_config())
        sim.reset(seed=42)
        e = sim.total_energy()
        assert np.isfinite(e)

    def test_total_energy_type(self):
        """Total energy should return a float."""
        sim = CartPoleBraxSimulation(_make_config())
        sim.reset(seed=42)
        e = sim.total_energy()
        assert isinstance(e, (float, np.floating))

    def test_energy_conservation_no_friction_no_force(self):
        """Energy should be conserved with no friction and no applied force.

        The cart-pole is a conservative system when there is no friction
        or external force. RK4 with small dt should conserve energy well.
        """
        sim = CartPoleBraxSimulation(_make_config(
            mu_c=0.0, mu_p=0.0, dt=0.0005,
        ))
        sim.reset(seed=42)
        sim.set_action(0.0)
        e0 = sim.total_energy()

        for _ in range(200):
            sim.step()

        e1 = sim.total_energy()
        rel_drift = abs(e1 - e0) / (abs(e0) + 1e-10)
        assert rel_drift < 0.01, f"Energy drift {rel_drift:.6f} exceeds 1%"

    def test_energy_not_conserved_with_friction(self):
        """Energy should decrease when friction is present."""
        sim = CartPoleBraxSimulation(_make_config(
            mu_c=0.5, mu_p=0.1, dt=0.001,
        ))
        # Start with a significant perturbation for visible friction effect
        sim.reset(seed=42)
        # Override state with larger initial velocity for measurable dissipation
        sim._state = np.array([0.0, 1.0, 0.3, 0.5])
        sim.set_action(0.0)
        e0 = sim.total_energy()

        for _ in range(500):
            sim.step()

        e1 = sim.total_energy()
        # Friction should dissipate energy
        assert e1 < e0

    # -- is_balanced --

    def test_is_balanced_true_near_upright(self):
        """is_balanced should return True when theta is small."""
        sim = CartPoleBraxSimulation(_make_config())
        sim.reset(seed=42)
        # After reset, theta is near 0
        assert sim.is_balanced(angle_threshold=0.2)

    def test_is_balanced_false_when_fallen(self):
        """is_balanced should return False when pole has fallen."""
        sim = CartPoleBraxSimulation(_make_config(dt=0.005))
        sim.reset(seed=42)

        # Let the pole fall
        for _ in range(500):
            sim.step()

        # After enough time, pole should have fallen past threshold
        assert not sim.is_balanced(angle_threshold=0.2)

    def test_is_balanced_custom_threshold(self):
        """is_balanced should respect custom angle_threshold."""
        sim = CartPoleBraxSimulation(_make_config())
        sim.reset(seed=42)
        # Force a known theta
        sim._state[2] = 0.15
        assert sim.is_balanced(angle_threshold=0.2)
        assert not sim.is_balanced(angle_threshold=0.1)

    # -- linearized_frequency --

    def test_linearized_frequency(self):
        """linearized_frequency should return sqrt(g/L)."""
        g = 9.81
        L = 0.5
        sim = CartPoleBraxSimulation(_make_config(g=g, L=L))
        expected = np.sqrt(g / L)
        assert np.isclose(sim.linearized_frequency(), expected)

    def test_linearized_frequency_custom_params(self):
        """linearized_frequency should scale with g and L."""
        g = 10.0
        L = 2.0
        sim = CartPoleBraxSimulation(_make_config(g=g, L=L))
        expected = np.sqrt(g / L)
        assert np.isclose(sim.linearized_frequency(), expected)

    # -- Determinism --

    def test_determinism(self):
        """Two runs with the same seed should produce identical states."""
        sim = CartPoleBraxSimulation(_make_config())

        sim.reset(seed=123)
        for _ in range(20):
            s1 = sim.step()

        sim.reset(seed=123)
        for _ in range(20):
            s2 = sim.step()

        assert np.allclose(s1, s2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different initial states."""
        sim = CartPoleBraxSimulation(_make_config())
        s1 = sim.reset(seed=1)
        s2 = sim.reset(seed=2)
        assert not np.allclose(s1, s2)

    # -- RK4 integration accuracy --

    def test_rk4_convergence(self):
        """RK4 solutions at different dt should converge as dt decreases.

        Run the same scenario with two step sizes and compare the final
        state to a reference run at very small dt. The finer run should
        be closer to the reference.
        """
        def run_to_time(dt_val, total_time):
            n_steps = int(total_time / dt_val)
            sim = CartPoleBraxSimulation(_make_config(dt=dt_val, mu_c=0.0, mu_p=0.0))
            sim.reset(seed=42)
            # Start from moderate angle for nontrivial dynamics
            sim._state = np.array([0.0, 0.0, 0.3, 0.0])
            sim.set_action(0.0)
            for _ in range(n_steps):
                sim.step()
            return sim._state.copy()

        total_time = 0.1  # Short time to avoid blow-up near unstable eq
        s_ref = run_to_time(0.0001, total_time)    # Reference (very fine)
        s_coarse = run_to_time(0.01, total_time)    # Coarse
        s_fine = run_to_time(0.001, total_time)      # Fine

        err_coarse = np.linalg.norm(s_coarse - s_ref)
        err_fine = np.linalg.norm(s_fine - s_ref)

        # Fine dt should be much closer to reference than coarse dt
        assert err_fine < err_coarse

    # -- Trajectory --

    def test_trajectory_shape(self):
        """run() should return trajectory with correct shape."""
        n_steps = 50
        sim = CartPoleBraxSimulation(_make_config(n_steps=n_steps))
        traj = sim.run(n_steps=n_steps)
        assert traj.states.shape == (n_steps + 1, 4)

    def test_trajectory_no_nan(self):
        """Trajectory should not contain NaN for short runs."""
        n_steps = 100
        sim = CartPoleBraxSimulation(_make_config(n_steps=n_steps))
        traj = sim.run(n_steps=n_steps)
        assert not np.any(np.isnan(traj.states))

    # -- Derivatives structure --

    def test_derivatives_at_equilibrium(self):
        """At upright equilibrium with zero velocities, derivatives should be near zero."""
        sim = CartPoleBraxSimulation(_make_config())
        sim.reset(seed=42)
        state_eq = np.array([0.0, 0.0, 0.0, 0.0])
        sim.set_action(0.0)
        derivs = sim._derivatives(state_eq)
        # At equilibrium: x_dot=0, theta_dot=0, accelerations~0
        assert np.allclose(derivs, 0.0, atol=1e-10)
