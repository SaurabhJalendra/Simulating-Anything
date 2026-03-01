"""Tests for the Duffing oscillator simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.duffing import DuffingOscillator
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    alpha: float = 1.0,
    beta: float = 1.0,
    delta: float = 0.2,
    gamma_f: float = 0.3,
    omega: float = 1.0,
    x_0: float = 0.5,
    v_0: float = 0.0,
    dt: float = 0.005,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.DUFFING,
        dt=dt,
        n_steps=1000,
        parameters={
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "gamma_f": gamma_f,
            "omega": omega,
            "x_0": x_0,
            "v_0": v_0,
        },
    )


class TestDuffingSimulation:
    def test_duffing_creation(self):
        """DuffingOscillator can be instantiated with default config."""
        sim = DuffingOscillator(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [0.5, 0.0])

    def test_duffing_step(self):
        """A single step should change the state."""
        sim = DuffingOscillator(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_duffing_trajectory(self):
        """Run 500 steps and verify trajectory shape."""
        sim = DuffingOscillator(_make_config())
        traj = sim.run(n_steps=500)
        # 500 steps + 1 initial = 501
        assert traj.states.shape == (501, 2)
        assert np.all(np.isfinite(traj.states))

    def test_duffing_reset_deterministic(self):
        """Two resets with same config produce identical trajectories."""
        config = _make_config()
        sim1 = DuffingOscillator(config)
        traj1 = sim1.run(n_steps=200)

        sim2 = DuffingOscillator(config)
        traj2 = sim2.run(n_steps=200)

        np.testing.assert_allclose(traj1.states, traj2.states, atol=1e-12)

    def test_duffing_energy_bounded(self):
        """Energy should remain bounded during simulation."""
        sim = DuffingOscillator(_make_config(gamma_f=0.3))
        sim.reset()
        energies = [sim.total_energy]
        for _ in range(5000):
            sim.step()
            energies.append(sim.total_energy)
        energies = np.array(energies)
        assert np.all(np.isfinite(energies))
        assert np.all(energies < 100), f"Energy diverged: max={energies.max()}"

    def test_duffing_forcing(self):
        """With gamma_f=0 and delta=0, energy should be conserved."""
        sim = DuffingOscillator(
            _make_config(gamma_f=0.0, delta=0.0, x_0=0.5, v_0=0.0, dt=0.001)
        )
        sim.reset()
        E0 = sim.total_energy
        for _ in range(5000):
            sim.step()
        E_final = sim.total_energy
        # RK4 should conserve energy to high precision over moderate time
        assert abs(E_final - E0) / max(E0, 1e-15) < 1e-4, (
            f"Energy not conserved: E0={E0}, E_final={E_final}"
        )

    def test_duffing_damping(self):
        """With delta > 0 and gamma_f=0, energy should decrease."""
        sim = DuffingOscillator(
            _make_config(delta=0.5, gamma_f=0.0, x_0=1.0, v_0=0.0)
        )
        sim.reset()
        E0 = sim.total_energy
        for _ in range(2000):
            sim.step()
        E_final = sim.total_energy
        assert E_final < E0, (
            f"Energy should decrease with damping: E0={E0}, E_final={E_final}"
        )

    def test_duffing_chaos(self):
        """High gamma_f should produce complex (chaotic-looking) trajectories.

        We verify this by checking that the trajectory visits a wider range
        of x values compared to the low-forcing case.
        """
        # Low forcing
        sim_low = DuffingOscillator(
            _make_config(gamma_f=0.1, x_0=0.5, dt=0.005)
        )
        sim_low.reset()
        x_low = []
        for _ in range(10000):
            sim_low.step()
            x_low.append(sim_low.observe()[0])
        range_low = max(x_low) - min(x_low)

        # High forcing
        sim_high = DuffingOscillator(
            _make_config(gamma_f=0.5, x_0=0.5, dt=0.005)
        )
        sim_high.reset()
        x_high = []
        for _ in range(10000):
            sim_high.step()
            x_high.append(sim_high.observe()[0])
        range_high = max(x_high) - min(x_high)

        assert range_high > range_low, (
            f"High forcing should produce wider range: "
            f"range_low={range_low:.4f}, range_high={range_high:.4f}"
        )

    def test_duffing_period(self):
        """Unforced, undamped Duffing should have approximately T = 2*pi/sqrt(alpha)
        for small amplitudes."""
        sim = DuffingOscillator(
            _make_config(
                alpha=1.0, beta=1.0, delta=0.0, gamma_f=0.0,
                x_0=0.1, v_0=0.0, dt=0.005,
            )
        )
        sim.reset()
        T = sim.measure_period(n_periods=5)
        T_theory = 2 * np.pi / np.sqrt(1.0)  # 2*pi for alpha=1
        # For small amplitude x_0=0.1, beta*x^2 << alpha, so period ~ 2*pi
        assert abs(T - T_theory) < 0.5, (
            f"Period {T:.4f} not near {T_theory:.4f} for small-amplitude unforced"
        )

    def test_duffing_default_params(self):
        """Default parameters should match the specification."""
        config = SimulationConfig(
            domain=Domain.DUFFING, dt=0.01, n_steps=100, parameters={}
        )
        sim = DuffingOscillator(config)
        assert sim.alpha == 1.0
        assert sim.beta == 1.0
        assert sim.delta == 0.2
        assert sim.gamma_f == 0.3
        assert sim.omega == 1.0
        assert sim.x_0 == 0.5
        assert sim.v_0 == 0.0

    def test_duffing_custom_params(self):
        """Custom parameters should override defaults."""
        sim = DuffingOscillator(
            _make_config(alpha=2.0, beta=0.5, delta=0.1, gamma_f=0.8, omega=2.0)
        )
        assert sim.alpha == 2.0
        assert sim.beta == 0.5
        assert sim.delta == 0.1
        assert sim.gamma_f == 0.8
        assert sim.omega == 2.0

    def test_duffing_observe_equals_state(self):
        """observe() should return the current state."""
        sim = DuffingOscillator(_make_config())
        sim.reset()
        for _ in range(10):
            state = sim.step()
            observed = sim.observe()
            np.testing.assert_array_equal(state, observed)


class TestDuffingRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.duffing import generate_ode_data

        data = generate_ode_data(
            alpha=1.0, beta=1.0, delta=0.2, gamma_f=0.0,
            n_steps=500, dt=0.01,
        )
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["alpha"] == 1.0
        assert data["beta"] == 1.0

    def test_chaos_sweep(self):
        from simulating_anything.rediscovery.duffing import generate_chaos_sweep

        data = generate_chaos_sweep(n_gamma=5, dt=0.01)
        assert len(data["gamma_f"]) == 5
        assert len(data["poincare_spread"]) == 5
        # All spreads should be non-negative
        assert np.all(data["poincare_spread"] >= 0)

    def test_amplitude_frequency_data(self):
        from simulating_anything.rediscovery.duffing import (
            generate_amplitude_frequency_data,
        )

        data = generate_amplitude_frequency_data(n_omega=5, dt=0.01)
        assert len(data["omega"]) == 5
        assert len(data["amplitude"]) == 5
        # All amplitudes should be positive
        assert np.all(data["amplitude"] > 0)
