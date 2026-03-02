"""Tests for the Ueda oscillator simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.ueda import UedaSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


_DOMAIN = Domain.UEDA


def _make_config(
    delta: float = 0.05,
    B: float = 7.5,
    x_0: float = 2.5,
    y_0: float = 0.0,
    dt: float = 0.005,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=_DOMAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "delta": delta,
            "B": B,
            "x_0": x_0,
            "y_0": y_0,
        },
    )


class TestUedaConstruction:
    """Tests for UedaSimulation construction and defaults."""

    def test_creation_default_params(self):
        """UedaSimulation can be created with empty parameters dict."""
        config = SimulationConfig(
            domain=_DOMAIN, dt=0.01, n_steps=100, parameters={}
        )
        sim = UedaSimulation(config)
        assert sim.delta == 0.05
        assert sim.B == 7.5
        assert sim.x_0 == 2.5
        assert sim.y_0 == 0.0

    def test_creation_custom_params(self):
        """Custom parameters should override defaults."""
        sim = UedaSimulation(_make_config(delta=0.1, B=3.0, x_0=1.0, y_0=0.5))
        assert sim.delta == 0.1
        assert sim.B == 3.0
        assert sim.x_0 == 1.0
        assert sim.y_0 == 0.5

    def test_forcing_period(self):
        """Forcing period should be 2*pi (omega_drive = 1)."""
        sim = UedaSimulation(_make_config())
        assert np.isclose(sim.forcing_period, 2.0 * np.pi)


class TestUedaReset:
    """Tests for reset and initial state."""

    def test_reset_shape(self):
        """Reset should return a 3-element array [x, y, t_phase]."""
        sim = UedaSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_reset_values(self):
        """Initial state should match specified initial conditions."""
        sim = UedaSimulation(_make_config(x_0=1.5, y_0=-0.3))
        state = sim.reset()
        assert np.isclose(state[0], 1.5)
        assert np.isclose(state[1], -0.3)
        assert np.isclose(state[2], 0.0)  # t_phase = 0 at start

    def test_reset_deterministic(self):
        """Two resets produce the same initial state."""
        sim = UedaSimulation(_make_config())
        s1 = sim.reset()
        s2 = sim.reset()
        np.testing.assert_array_equal(s1, s2)


class TestUedaStep:
    """Tests for RK4 stepping."""

    def test_step_advances(self):
        """State should change after a step."""
        sim = UedaSimulation(_make_config())
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_step_advances_time(self):
        """t_phase should increase after stepping."""
        sim = UedaSimulation(_make_config(dt=0.01))
        sim.reset()
        state1 = sim.step()
        # t_phase should be dt modulo 2*pi
        assert np.isclose(state1[2], 0.01)

    def test_step_count_increments(self):
        """Step count should increment after each step."""
        sim = UedaSimulation(_make_config())
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2

    def test_observe_equals_step(self):
        """observe() should match the last step() return value."""
        sim = UedaSimulation(_make_config())
        sim.reset()
        for _ in range(10):
            state = sim.step()
            observed = sim.observe()
            np.testing.assert_array_equal(state, observed)


class TestUedaDerivatives:
    """Tests for the ODE right-hand side."""

    def test_derivatives_at_origin_no_forcing(self):
        """At origin with B=0, derivatives should be zero."""
        sim = UedaSimulation(_make_config(B=0.0))
        sim.reset()
        derivs = sim._derivatives(np.array([0.0, 0.0]), 0.0)
        np.testing.assert_array_almost_equal(derivs, [0.0, 0.0])

    def test_derivatives_cubic_term(self):
        """The cubic restoring force -x^3 should dominate for large x."""
        sim = UedaSimulation(_make_config(B=0.0, delta=0.0))
        sim.reset()
        # At (2.0, 0.0): dx/dt=0, dy/dt=-8.0
        derivs = sim._derivatives(np.array([2.0, 0.0]), 0.0)
        np.testing.assert_almost_equal(derivs[0], 0.0)
        np.testing.assert_almost_equal(derivs[1], -8.0)

    def test_derivatives_damping(self):
        """Damping term -delta*y should appear in dy/dt."""
        sim = UedaSimulation(_make_config(delta=0.1, B=0.0))
        sim.reset()
        # At (0, 5.0): dx/dt=5, dy/dt=-0.1*5=-0.5
        derivs = sim._derivatives(np.array([0.0, 5.0]), 0.0)
        np.testing.assert_almost_equal(derivs[0], 5.0)
        np.testing.assert_almost_equal(derivs[1], -0.5)

    def test_derivatives_forcing(self):
        """Forcing term B*cos(t) at t=0 should equal B."""
        sim = UedaSimulation(_make_config(B=3.0, delta=0.0))
        sim.reset()
        # At (0, 0) with t=0: dx/dt=0, dy/dt=B*cos(0)=3.0
        derivs = sim._derivatives(np.array([0.0, 0.0]), 0.0)
        np.testing.assert_almost_equal(derivs[0], 0.0)
        np.testing.assert_almost_equal(derivs[1], 3.0)

    def test_derivatives_full(self):
        """Full derivative at a known point."""
        sim = UedaSimulation(_make_config(delta=0.1, B=2.0))
        sim.reset()
        t = np.pi / 3  # cos(pi/3) = 0.5
        # At (1.0, 1.0): dx/dt=1, dy/dt=-0.1*1 - 1 + 2*0.5 = -0.1
        derivs = sim._derivatives(np.array([1.0, 1.0]), t)
        np.testing.assert_almost_equal(derivs[0], 1.0)
        np.testing.assert_almost_equal(derivs[1], -0.1)


class TestUedaEnergy:
    """Tests for energy-related quantities."""

    def test_energy_conservative_case(self):
        """With delta=0 and B=0, mechanical energy should be conserved."""
        sim = UedaSimulation(
            _make_config(delta=0.0, B=0.0, x_0=1.0, y_0=0.5, dt=0.001)
        )
        sim.reset()
        E0 = sim.mechanical_energy
        for _ in range(5000):
            sim.step()
        E_final = sim.mechanical_energy
        # RK4 should conserve energy well
        rel_error = abs(E_final - E0) / max(abs(E0), 1e-15)
        assert rel_error < 1e-6, (
            f"Energy not conserved: E0={E0}, E_final={E_final}, "
            f"rel_error={rel_error:.2e}"
        )

    def test_energy_damping_decreases(self):
        """With delta > 0 and B=0, energy should decrease."""
        sim = UedaSimulation(
            _make_config(delta=0.2, B=0.0, x_0=2.0, y_0=0.0)
        )
        sim.reset()
        E0 = sim.mechanical_energy
        for _ in range(2000):
            sim.step()
        E_final = sim.mechanical_energy
        assert E_final < E0, (
            f"Energy should decrease with damping: E0={E0}, E_final={E_final}"
        )

    def test_energy_bounded_forced(self):
        """In the forced-damped case, energy should remain bounded."""
        sim = UedaSimulation(_make_config(delta=0.05, B=7.5))
        sim.reset()
        for _ in range(20000):
            sim.step()
            E = sim.mechanical_energy
            assert np.isfinite(E), f"Energy is not finite: {E}"
            assert E < 1e6, f"Energy diverged: {E}"


class TestUedaPoincare:
    """Tests for Poincare section computation."""

    def test_poincare_shape(self):
        """Poincare section should return correct shape."""
        sim = UedaSimulation(_make_config(dt=0.01))
        sim.reset()
        points = sim.compute_poincare_section(n_transient=50, n_points=20)
        assert points.shape == (20, 2)

    def test_poincare_finite(self):
        """All Poincare section points should be finite."""
        sim = UedaSimulation(_make_config(dt=0.01))
        sim.reset()
        points = sim.compute_poincare_section(n_transient=50, n_points=50)
        assert np.all(np.isfinite(points))

    def test_poincare_periodic_orbit(self):
        """For small B, Poincare section should cluster (period-1 orbit)."""
        sim = UedaSimulation(_make_config(B=0.5, delta=0.1, dt=0.005))
        sim.reset()
        points = sim.compute_poincare_section(n_transient=200, n_points=100)
        # Standard deviation should be small for periodic orbit
        x_std = np.std(points[:, 0])
        assert x_std < 1.0, (
            f"Poincare section too spread for small B: x_std={x_std:.4f}"
        )

    def test_poincare_chaotic_spread(self):
        """For B=7.5, Poincare section should be spread out (strange attractor)."""
        sim = UedaSimulation(_make_config(B=7.5, delta=0.05, dt=0.005))
        sim.reset()
        points = sim.compute_poincare_section(n_transient=300, n_points=200)
        x_range = np.ptp(points[:, 0])
        assert x_range > 0.5, (
            f"Poincare section not spread enough for chaos: range={x_range:.4f}"
        )


class TestUedaChaos:
    """Tests for chaotic behavior and Lyapunov exponent."""

    def test_lyapunov_positive_chaotic(self):
        """At B=7.5, delta=0.05, Lyapunov exponent should be positive."""
        sim = UedaSimulation(
            _make_config(B=7.5, delta=0.05, dt=0.005, n_steps=50000)
        )
        sim.reset()
        # Skip transient
        for _ in range(5000):
            sim.step()
        lam = sim.compute_lyapunov_exponent(n_steps=30000, dt=0.005)
        assert lam > 0.0, (
            f"Lyapunov {lam:.4f} not positive at classic chaotic parameters"
        )

    def test_lyapunov_negative_periodic(self):
        """At low B, Lyapunov exponent should be negative (periodic)."""
        sim = UedaSimulation(
            _make_config(B=0.5, delta=0.1, dt=0.005, n_steps=30000)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
        lam = sim.compute_lyapunov_exponent(n_steps=20000, dt=0.005)
        assert lam < 0.5, (
            f"Lyapunov {lam:.4f} unexpectedly large for low B"
        )

    def test_chaos_at_higher_B_more_spread(self):
        """Higher B should generally produce larger Poincare spread than low B."""
        sim_low = UedaSimulation(
            _make_config(B=1.0, delta=0.05, dt=0.005)
        )
        sim_low.reset()
        pts_low = sim_low.compute_poincare_section(
            n_transient=200, n_points=100
        )
        spread_low = np.std(pts_low[:, 0])

        sim_high = UedaSimulation(
            _make_config(B=7.5, delta=0.05, dt=0.005)
        )
        sim_high.reset()
        pts_high = sim_high.compute_poincare_section(
            n_transient=200, n_points=100
        )
        spread_high = np.std(pts_high[:, 0])

        assert spread_high > spread_low, (
            f"High B spread ({spread_high:.4f}) should exceed "
            f"low B spread ({spread_low:.4f})"
        )


class TestUedaDeterminism:
    """Tests for deterministic reproducibility."""

    def test_trajectory_deterministic(self):
        """Two simulations with same config should produce identical trajectories."""
        config = _make_config(dt=0.005)
        sim1 = UedaSimulation(config)
        traj1 = sim1.run(n_steps=200)

        sim2 = UedaSimulation(config)
        traj2 = sim2.run(n_steps=200)

        np.testing.assert_allclose(traj1.states, traj2.states, atol=1e-12)

    def test_repeated_runs_identical(self):
        """Reset and run again should give the same result."""
        sim = UedaSimulation(_make_config())
        sim.reset()
        states1 = []
        for _ in range(100):
            states1.append(sim.step().copy())

        sim.reset()
        states2 = []
        for _ in range(100):
            states2.append(sim.step().copy())

        for s1, s2 in zip(states1, states2):
            np.testing.assert_array_equal(s1, s2)


class TestUedaNumerics:
    """Tests for numerical accuracy and stability."""

    def test_no_nan_long_trajectory(self):
        """No NaN or Inf after many steps."""
        sim = UedaSimulation(_make_config(B=7.5, dt=0.005))
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State not finite: {state}"

    def test_bounded_trajectory(self):
        """Trajectory should stay bounded in chaotic regime."""
        sim = UedaSimulation(_make_config(B=7.5, delta=0.05, dt=0.005))
        sim.reset()
        for _ in range(20000):
            state = sim.step()
            assert abs(state[0]) < 50, f"x diverged: {state[0]}"
            assert abs(state[1]) < 50, f"y diverged: {state[1]}"

    def test_rk4_convergence(self):
        """Finer dt should give more accurate results."""
        # Coarse: dt=0.01, 100 steps => T=1.0
        sim_coarse = UedaSimulation(
            _make_config(B=3.0, delta=0.1, x_0=1.0, y_0=0.0, dt=0.01)
        )
        sim_coarse.reset()
        for _ in range(100):
            sim_coarse.step()
        state_coarse = sim_coarse._state.copy()

        # Fine: dt=0.002, 500 steps => T=1.0
        sim_fine = UedaSimulation(
            _make_config(B=3.0, delta=0.1, x_0=1.0, y_0=0.0, dt=0.002)
        )
        sim_fine.reset()
        for _ in range(500):
            sim_fine.step()
        state_fine = sim_fine._state.copy()

        error = np.linalg.norm(state_coarse - state_fine)
        assert error < 1e-3, (
            f"RK4 convergence error {error:.6f} too large"
        )

    def test_trajectory_collection(self):
        """run() should produce correct trajectory shape."""
        sim = UedaSimulation(_make_config(dt=0.01, n_steps=500))
        traj = sim.run(n_steps=500)
        # 500 steps + 1 initial = 501 rows; observe returns [x, y, t_phase]
        assert traj.states.shape == (501, 3)
        assert np.all(np.isfinite(traj.states))


class TestUedaParameterSensitivity:
    """Tests for parameter sensitivity and sweep."""

    def test_higher_damping_decays_faster(self):
        """Higher delta should cause faster energy decay (unforced)."""
        sim_low = UedaSimulation(
            _make_config(delta=0.01, B=0.0, x_0=2.0, y_0=0.0, dt=0.005)
        )
        sim_low.reset()
        for _ in range(5000):
            sim_low.step()
        E_low = sim_low.mechanical_energy

        sim_high = UedaSimulation(
            _make_config(delta=0.5, B=0.0, x_0=2.0, y_0=0.0, dt=0.005)
        )
        sim_high.reset()
        for _ in range(5000):
            sim_high.step()
        E_high = sim_high.mechanical_energy

        assert E_high < E_low, (
            f"Higher damping should give lower energy: "
            f"E_low={E_low:.6f}, E_high={E_high:.6f}"
        )

    def test_lyapunov_sweep_produces_data(self):
        """Lyapunov sweep should return valid arrays."""
        sim = UedaSimulation(_make_config(dt=0.01))
        sim.reset()
        B_values = np.array([1.0, 5.0, 7.5])
        data = sim.lyapunov_sweep(
            B_values, n_transient=500, n_measure=3000
        )
        assert len(data["B"]) == 3
        assert len(data["lyapunov_exponent"]) == 3
        assert np.all(np.isfinite(data["lyapunov_exponent"]))
        assert len(data["attractor_type"]) == 3

    def test_trajectory_statistics(self):
        """Trajectory statistics should be computable and finite."""
        sim = UedaSimulation(
            _make_config(B=7.5, delta=0.05, dt=0.005, n_steps=10000)
        )
        stats = sim.compute_trajectory_statistics(
            n_steps=5000, n_transient=2000
        )
        for key, val in stats.items():
            assert np.isfinite(val), f"Non-finite {key}: {val}"
        # Chaotic trajectory should have positive spread
        assert stats["x_std"] > 0
        assert stats["y_std"] > 0

    def test_steady_amplitude(self):
        """Steady-state amplitude should be positive and finite."""
        sim = UedaSimulation(_make_config(B=3.0, delta=0.1, dt=0.005))
        sim.reset()
        amp = sim.measure_steady_amplitude(n_periods=10)
        assert np.isfinite(amp)
        assert amp > 0


class TestUedaRediscovery:
    """Tests for the Ueda rediscovery data generation functions."""

    def test_ode_data_generation(self):
        """ODE data generation should produce correct shapes."""
        from simulating_anything.rediscovery.ueda import generate_ode_data

        data = generate_ode_data(
            delta=0.05, B=0.0, n_steps=500, dt=0.01
        )
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["delta"] == 0.05
        assert data["B"] == 0.0
        assert np.all(np.isfinite(data["states"]))

    def test_lyapunov_vs_B_data(self):
        """Lyapunov sweep should produce valid data."""
        from simulating_anything.rediscovery.ueda import (
            generate_lyapunov_vs_B_data,
        )

        data = generate_lyapunov_vs_B_data(n_B=5, n_steps=3000, dt=0.01)
        assert len(data["B"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert np.all(np.isfinite(data["lyapunov_exponent"]))

    def test_poincare_data(self):
        """Poincare data generation should return correct structure."""
        from simulating_anything.rediscovery.ueda import generate_poincare_data

        data = generate_poincare_data(
            B=7.5, delta=0.05, n_transient=50, n_points=30, dt=0.01
        )
        assert len(data["x"]) == 30
        assert len(data["y"]) == 30
        assert data["B"] == 7.5
        assert np.all(np.isfinite(data["x"]))

    def test_chaos_sweep(self):
        """Chaos sweep should produce valid data."""
        from simulating_anything.rediscovery.ueda import generate_chaos_sweep

        data = generate_chaos_sweep(n_B=5, dt=0.01)
        assert len(data["B"]) == 5
        assert len(data["poincare_spread"]) == 5
        assert np.all(data["poincare_spread"] >= 0)
