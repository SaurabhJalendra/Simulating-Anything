"""Tests for the Mackey-Glass delay differential equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.mackey_glass import MackeyGlassSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    beta: float = 0.2,
    gamma: float = 0.1,
    tau: float = 17.0,
    n: float = 10.0,
    x_0: float = 0.9,
    dt: float = 0.1,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.MACKEY_GLASS,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "beta": beta,
            "gamma": gamma,
            "tau": tau,
            "n": n,
            "x_0": x_0,
        },
    )


class TestMackeyGlassSimulation:
    """Core simulation tests."""

    def test_initial_state(self):
        sim = MackeyGlassSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (1,)
        assert state[0] == pytest.approx(0.9)

    def test_observe_shape(self):
        """observe() returns a 1-element array."""
        sim = MackeyGlassSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (1,)
        assert obs[0] == pytest.approx(0.9)

    def test_step_advances(self):
        """State changes after a step."""
        sim = MackeyGlassSimulation(_make_config())
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = MackeyGlassSimulation(config)
        sim1.reset()
        traj1 = [sim1.step()[0] for _ in range(100)]

        sim2 = MackeyGlassSimulation(config)
        sim2.reset()
        traj2 = [sim2.step()[0] for _ in range(100)]

        np.testing.assert_array_equal(traj1, traj2)

    def test_positive(self):
        """x stays positive for standard parameters."""
        sim = MackeyGlassSimulation(_make_config())
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert state[0] > 0, f"x went non-positive: {state[0]}"

    def test_bounded(self):
        """x stays in a reasonable range (0 to ~2) for default parameters."""
        sim = MackeyGlassSimulation(_make_config())
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert state[0] < 3.0, f"x too large: {state[0]}"
            assert np.isfinite(state[0]), "x became NaN/Inf"

    def test_stability_no_nan(self):
        """No NaN after many steps with default parameters."""
        sim = MackeyGlassSimulation(_make_config(n_steps=100000))
        sim.reset()
        for _ in range(50000):
            state = sim.step()
        assert np.all(np.isfinite(state))

    def test_long_trajectory(self):
        """Can run for 100000 steps without issues."""
        sim = MackeyGlassSimulation(_make_config(n_steps=100000))
        sim.reset()
        for _ in range(100000):
            state = sim.step()
        assert np.all(np.isfinite(state))
        assert state[0] > 0


class TestMackeyGlassHistoryBuffer:
    """Tests for the delay history buffer."""

    def test_history_buffer_size(self):
        """Buffer has correct length = int(tau / dt) + 1."""
        config = _make_config(tau=17.0, dt=0.1)
        sim = MackeyGlassSimulation(config)
        sim.reset()
        expected_size = int(17.0 / 0.1) + 1  # 171
        assert len(sim._buffer) == expected_size

    def test_reset_fills_history(self):
        """After reset, all buffer entries equal x_0."""
        sim = MackeyGlassSimulation(_make_config(x_0=0.9))
        sim.reset()
        np.testing.assert_array_equal(sim._buffer, 0.9)

    def test_get_history(self):
        """get_history() returns array of correct size."""
        sim = MackeyGlassSimulation(_make_config())
        sim.reset()
        history = sim.get_history()
        assert len(history) == sim._buf_size

    def test_buffer_different_tau(self):
        """Different tau gives different buffer sizes."""
        sim1 = MackeyGlassSimulation(_make_config(tau=5.0, dt=0.1))
        sim1.reset()
        sim2 = MackeyGlassSimulation(_make_config(tau=20.0, dt=0.1))
        sim2.reset()
        assert len(sim1._buffer) < len(sim2._buffer)


class TestMackeyGlassEquilibrium:
    """Tests for equilibrium convergence."""

    def test_equilibrium_formula(self):
        """x* = (beta/gamma - 1)^(1/n) for default parameters."""
        sim = MackeyGlassSimulation(_make_config())
        sim.reset()
        x_star = sim.equilibrium
        # beta=0.2, gamma=0.1 -> ratio=2, x* = (2-1)^(1/10) = 1.0
        assert x_star == pytest.approx(1.0)

    def test_equilibrium_different_params(self):
        """Equilibrium changes with beta."""
        sim = MackeyGlassSimulation(_make_config(beta=0.3, gamma=0.1, n=10.0))
        sim.reset()
        x_star = sim.equilibrium
        # ratio = 3, x* = (3-1)^(1/10) = 2^0.1
        assert x_star == pytest.approx(2.0 ** 0.1, abs=1e-6)

    def test_equilibrium_zero_when_beta_le_gamma(self):
        """No positive equilibrium when beta <= gamma."""
        sim = MackeyGlassSimulation(_make_config(beta=0.05, gamma=0.1))
        sim.reset()
        assert sim.equilibrium == 0.0

    def test_convergence_small_tau(self):
        """For tau=2, system converges to equilibrium x* = 1.0."""
        config = _make_config(tau=2.0, n_steps=80000, dt=0.1)
        sim = MackeyGlassSimulation(config)
        sim.reset()

        for _ in range(80000):
            sim.step()

        # Should be close to equilibrium
        x_final = sim.observe()[0]
        assert x_final == pytest.approx(1.0, abs=0.05)


class TestMackeyGlassDynamics:
    """Tests for dynamical behavior across tau regimes."""

    def test_oscillation_medium_tau(self):
        """For tau=10, system oscillates (non-trivial amplitude)."""
        config = _make_config(tau=10.0, n_steps=50000, dt=0.1)
        sim = MackeyGlassSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(20000):
            sim.step()

        x_vals = []
        for _ in range(10000):
            state = sim.step()
            x_vals.append(state[0])

        x_arr = np.array(x_vals)
        amplitude = np.max(x_arr) - np.min(x_arr)
        # Should have visible oscillations
        assert amplitude > 0.01, f"No oscillation detected: amplitude={amplitude}"

    def test_chaos_large_tau(self):
        """For tau=17, system is chaotic (high amplitude variation)."""
        config = _make_config(tau=17.0, n_steps=50000, dt=0.1)
        sim = MackeyGlassSimulation(config)
        sim.reset()

        # Skip transient
        for _ in range(20000):
            sim.step()

        x_vals = []
        for _ in range(20000):
            state = sim.step()
            x_vals.append(state[0])

        x_arr = np.array(x_vals)
        amplitude = np.max(x_arr) - np.min(x_arr)
        # Chaotic regime should have larger amplitude than oscillatory
        assert amplitude > 0.05, f"Chaotic amplitude too small: {amplitude}"

    def test_delay_effect(self):
        """Changing tau changes the dynamics qualitatively."""
        x_finals = []
        for tau_val in [2.0, 10.0, 20.0]:
            config = _make_config(tau=tau_val, n_steps=30000, dt=0.1)
            sim = MackeyGlassSimulation(config)
            sim.reset()
            for _ in range(30000):
                sim.step()
            x_finals.append(sim.observe()[0])

        # Different tau should give different final x values
        assert not (
            np.isclose(x_finals[0], x_finals[1], atol=0.01)
            and np.isclose(x_finals[1], x_finals[2], atol=0.01)
        ), "All tau values gave same dynamics"

    def test_gamma_decay(self):
        """When beta=0, x decays exponentially due to gamma term."""
        config = _make_config(beta=0.0, gamma=0.1, tau=5.0, x_0=1.0, dt=0.1)
        sim = MackeyGlassSimulation(config)
        sim.reset()

        # After many steps, x should be close to 0
        for _ in range(500):
            sim.step()

        x = sim.observe()[0]
        # Exact: x(t) = x_0 * exp(-gamma * t) = exp(-0.1 * 50) = exp(-5) ~ 0.0067
        assert x < 0.1, f"Decay too slow: x={x}"
        assert x >= 0, "x went negative"

    def test_production_function(self):
        """beta*x/(1+x^n) has a hump shape: rises then falls."""
        sim = MackeyGlassSimulation(_make_config())
        sim.reset()

        # Evaluate at several x values
        x_test = [0.0, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0]
        y_test = [sim._production(x) for x in x_test]

        # Should start at 0
        assert y_test[0] == pytest.approx(0.0)

        # Should have a maximum somewhere around x ~ 1 (for n=10)
        assert max(y_test) > 0

        # Production should decrease for large x
        assert y_test[-1] < y_test[3], "Production should decrease for x > x_max"


class TestMackeyGlassTauSweep:
    """Tests for the tau_sweep method."""

    def test_tau_sweep_returns_correct_shape(self):
        """tau_sweep returns arrays matching input length."""
        sim = MackeyGlassSimulation(_make_config())
        sim.reset()

        tau_vals = np.array([2.0, 10.0, 20.0])
        result = sim.tau_sweep(tau_vals, n_steps=5000, n_transient=2000)

        assert len(result["tau"]) == 3
        assert len(result["amplitude"]) == 3
        assert len(result["lyapunov"]) == 3

    def test_tau_sweep_different_amplitudes(self):
        """Different tau values should give different amplitudes."""
        sim = MackeyGlassSimulation(_make_config())
        sim.reset()

        tau_vals = np.array([2.0, 20.0])
        result = sim.tau_sweep(tau_vals, n_steps=10000, n_transient=5000)

        # tau=2 should have much smaller amplitude than tau=20
        assert result["amplitude"][0] < result["amplitude"][1]


class TestMackeyGlassRediscovery:
    """Tests for the rediscovery data generation functions."""

    def test_equilibrium_data(self):
        from simulating_anything.rediscovery.mackey_glass import generate_equilibrium_data

        data = generate_equilibrium_data(n_beta=5, n_steps=10000, dt=0.1)
        assert len(data["beta"]) == 5
        assert len(data["measured_equilibrium"]) == 5
        assert len(data["theory_equilibrium"]) == 5

    def test_tau_sweep_data(self):
        from simulating_anything.rediscovery.mackey_glass import generate_tau_sweep_data

        data = generate_tau_sweep_data(n_tau=5, n_steps=5000, dt=0.1)
        assert len(data["tau"]) == 5
        assert len(data["amplitude"]) == 5
        assert len(data["lyapunov"]) == 5
