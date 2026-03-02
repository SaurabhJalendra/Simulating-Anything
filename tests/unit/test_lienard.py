"""Tests for the Lienard oscillator simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.lienard import LienardSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    mu: float = 1.0,
    alpha: float = 0.1,
    x_0: float = 2.0,
    v_0: float = 0.0,
    dt: float = 0.005,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.LIENARD,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "mu": mu,
            "alpha": alpha,
            "x_0": x_0,
            "v_0": v_0,
        },
    )


# ---------------------------------------------------------------------------
# TestLienardSimulation -- basic API and state management
# ---------------------------------------------------------------------------
class TestLienardSimulation:
    def test_creation(self):
        """LienardSimulation can be instantiated with default config."""
        sim = LienardSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [2.0, 0.0])

    def test_default_params(self):
        """Default parameters should match specification."""
        config = SimulationConfig(
            domain=Domain.LIENARD, dt=0.01, n_steps=100, parameters={}
        )
        sim = LienardSimulation(config)
        assert sim.mu == 1.0
        assert sim.alpha == 0.1
        assert sim.x_0 == 2.0
        assert sim.v_0 == 0.0

    def test_custom_params(self):
        """Custom parameters should override defaults."""
        sim = LienardSimulation(_make_config(mu=3.0, alpha=0.5, x_0=1.0, v_0=0.5))
        assert sim.mu == 3.0
        assert sim.alpha == 0.5
        assert sim.x_0 == 1.0
        assert sim.v_0 == 0.5

    def test_reset_sets_state(self):
        """reset() should set state to [x_0, v_0]."""
        sim = LienardSimulation(_make_config(x_0=1.5, v_0=-0.3))
        state = sim.reset()
        np.testing.assert_allclose(state, [1.5, -0.3])

    def test_step_advances(self):
        """A single step should change the state."""
        sim = LienardSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_state(self):
        """observe() should return the current state."""
        sim = LienardSimulation(_make_config())
        sim.reset()
        for _ in range(10):
            state = sim.step()
            observed = sim.observe()
            np.testing.assert_array_equal(state, observed)

    def test_state_shape(self):
        """State should always be a 2-element vector."""
        sim = LienardSimulation(_make_config())
        sim.reset()
        for _ in range(50):
            state = sim.step()
            assert state.shape == (2,)
            assert state.dtype == np.float64

    def test_run_trajectory(self):
        """run() should return TrajectoryData with correct shape."""
        sim = LienardSimulation(_make_config(n_steps=500))
        traj = sim.run(n_steps=500)
        # 500 steps + 1 initial = 501
        assert traj.states.shape == (501, 2)
        assert np.all(np.isfinite(traj.states))

    def test_run_default_n_steps(self):
        """run() without arg uses config.n_steps."""
        sim = LienardSimulation(_make_config(n_steps=200))
        traj = sim.run()
        assert traj.states.shape == (201, 2)

    def test_reset_deterministic(self):
        """Two runs with same config produce identical trajectories."""
        config = _make_config()
        sim1 = LienardSimulation(config)
        traj1 = sim1.run(n_steps=200)

        sim2 = LienardSimulation(config)
        traj2 = sim2.run(n_steps=200)

        np.testing.assert_allclose(traj1.states, traj2.states, atol=1e-12)


# ---------------------------------------------------------------------------
# TestRK4Integration -- numerical integration accuracy
# ---------------------------------------------------------------------------
class TestRK4Integration:
    def test_derivatives_correct(self):
        """Check derivatives at a known point."""
        sim = LienardSimulation(_make_config(mu=2.0, alpha=0.5))
        sim.reset()
        y = np.array([1.0, 0.5])
        dy = sim._derivatives(y)
        # dx/dt = v = 0.5
        # dv/dt = -mu*(x^2-1)*v - x - alpha*x^3
        #       = -2*(1-1)*0.5 - 1 - 0.5*1 = 0 - 1 - 0.5 = -1.5
        np.testing.assert_allclose(dy, [0.5, -1.5])

    def test_derivatives_at_origin(self):
        """At the origin with v=0, derivative should be zero."""
        sim = LienardSimulation(_make_config(mu=1.0, alpha=0.1))
        sim.reset()
        y = np.array([0.0, 0.0])
        dy = sim._derivatives(y)
        np.testing.assert_allclose(dy, [0.0, 0.0])

    def test_energy_conserved_mu_zero(self):
        """For mu=0, energy should be conserved (Duffing conservative limit)."""
        sim = LienardSimulation(
            _make_config(mu=0.0, alpha=0.1, x_0=2.0, v_0=0.0, dt=0.001)
        )
        sim.reset()
        E0 = sim.compute_energy()
        for _ in range(5000):
            sim.step()
        E_final = sim.compute_energy()
        rel_err = abs(E_final - E0) / max(abs(E0), 1e-15)
        assert rel_err < 1e-6, (
            f"Energy not conserved for mu=0: E0={E0}, E_final={E_final}, "
            f"rel_err={rel_err:.2e}"
        )

    def test_trajectory_smooth(self):
        """Trajectory should be smooth (no jumps between consecutive steps)."""
        sim = LienardSimulation(_make_config(dt=0.005))
        sim.reset()
        prev = sim.observe().copy()
        for _ in range(500):
            curr = sim.step()
            diff = np.linalg.norm(curr - prev)
            assert diff < 1.0, f"Jump too large: {diff}"
            prev = curr.copy()

    def test_small_dt_convergence(self):
        """Smaller dt should give more accurate result (RK4 convergence)."""
        # Run with dt=0.01
        sim1 = LienardSimulation(_make_config(dt=0.01, n_steps=100))
        traj1 = sim1.run(n_steps=100)
        final1 = traj1.states[-1]

        # Run with dt=0.001 (10x more steps to reach same time)
        sim2 = LienardSimulation(_make_config(dt=0.001, n_steps=1000))
        traj2 = sim2.run(n_steps=1000)
        final2 = traj2.states[-1]

        # Run with dt=0.0001 (reference)
        sim3 = LienardSimulation(_make_config(dt=0.0001, n_steps=10000))
        traj3 = sim3.run(n_steps=10000)
        final3 = traj3.states[-1]

        err1 = np.linalg.norm(final1 - final3)
        err2 = np.linalg.norm(final2 - final3)
        # dt=0.001 should be much closer to reference than dt=0.01
        assert err2 < err1, (
            f"Smaller dt should be more accurate: err(0.01)={err1:.2e}, "
            f"err(0.001)={err2:.2e}"
        )

    def test_rk4_stability(self):
        """No NaN or Inf should appear during integration."""
        sim = LienardSimulation(_make_config(mu=3.0, alpha=0.5))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"

    def test_trajectory_bounded(self):
        """Trajectory should remain bounded."""
        sim = LienardSimulation(_make_config(mu=2.0, alpha=0.1))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.abs(state) < 50), f"State diverged: {state}"


# ---------------------------------------------------------------------------
# TestLimitCycle -- limit cycle existence and properties
# ---------------------------------------------------------------------------
class TestLimitCycle:
    def test_limit_cycle_exists(self):
        """For mu > 0, trajectory should settle into oscillation."""
        sim = LienardSimulation(_make_config(mu=1.0, alpha=0.1, x_0=0.01))
        sim.reset()
        # Run through transient
        for _ in range(20000):
            sim.step()
        # Measure amplitude over several cycles
        x_max = 0.0
        for _ in range(5000):
            sim.step()
            x_max = max(x_max, abs(sim.observe()[0]))
        # Should have grown from tiny IC to limit cycle
        assert x_max > 0.5, f"No limit cycle detected: x_max={x_max}"

    def test_amplitude_stabilizes(self):
        """Limit cycle amplitude should stabilize after transient."""
        sim = LienardSimulation(_make_config(mu=1.0, alpha=0.1, x_0=2.0, dt=0.005))
        sim.reset()
        # Skip transient
        for _ in range(20000):
            sim.step()

        # Measure amplitude in two consecutive windows
        def measure_window(n: int) -> float:
            xmax = 0.0
            for _ in range(n):
                sim.step()
                xmax = max(xmax, abs(sim.observe()[0]))
            return xmax

        A1 = measure_window(5000)
        A2 = measure_window(5000)
        # Amplitudes should be close (stable limit cycle)
        assert abs(A1 - A2) / max(A1, 1e-10) < 0.05, (
            f"Amplitude not stable: A1={A1:.4f}, A2={A2:.4f}"
        )

    def test_vdp_limit_alpha_zero(self):
        """With alpha=0, reduces to Van der Pol; amplitude should be near 2."""
        sim = LienardSimulation(
            _make_config(mu=1.0, alpha=0.0, x_0=0.1, v_0=0.0, dt=0.005)
        )
        sim.reset()
        A = sim.measure_amplitude(n_periods=5)
        assert 1.5 < A < 2.5, (
            f"VdP limit (alpha=0): amplitude {A:.4f} not near 2"
        )

    def test_different_ic_same_cycle(self):
        """Different initial conditions should converge to same limit cycle."""
        config1 = _make_config(mu=1.0, alpha=0.1, x_0=0.1, v_0=0.0, dt=0.005)
        config2 = _make_config(mu=1.0, alpha=0.1, x_0=3.0, v_0=0.0, dt=0.005)

        sim1 = LienardSimulation(config1)
        sim1.reset()
        A1 = sim1.measure_amplitude(n_periods=5)

        sim2 = LienardSimulation(config2)
        sim2.reset()
        A2 = sim2.measure_amplitude(n_periods=5)

        # Should converge to same amplitude
        assert abs(A1 - A2) < 0.3, (
            f"Different ICs should give same cycle: A1={A1:.4f}, A2={A2:.4f}"
        )

    def test_oscillation_detected(self):
        """Period measurement should detect oscillation for mu > 0."""
        sim = LienardSimulation(_make_config(mu=1.0, alpha=0.1, dt=0.005))
        sim.reset()
        T = sim.measure_period(n_periods=3)
        assert np.isfinite(T), "Period should be finite for mu > 0"
        assert T > 0, "Period should be positive"


# ---------------------------------------------------------------------------
# TestParameters -- parameter effects on dynamics
# ---------------------------------------------------------------------------
class TestParameters:
    def test_larger_mu_longer_period(self):
        """Larger mu should give longer period (relaxation oscillations)."""
        sim1 = LienardSimulation(_make_config(mu=0.5, alpha=0.1, dt=0.005))
        sim1.reset()
        T1 = sim1.measure_period(n_periods=3)

        sim2 = LienardSimulation(_make_config(mu=3.0, alpha=0.1, dt=0.003))
        sim2.reset()
        T2 = sim2.measure_period(n_periods=3)

        assert T2 > T1, f"Period should increase with mu: T1={T1:.4f}, T2={T2:.4f}"

    def test_alpha_affects_period(self):
        """Nonzero alpha should change the period compared to alpha=0."""
        sim1 = LienardSimulation(_make_config(mu=1.0, alpha=0.0, dt=0.005))
        sim1.reset()
        T1 = sim1.measure_period(n_periods=3)

        sim2 = LienardSimulation(_make_config(mu=1.0, alpha=0.5, dt=0.005))
        sim2.reset()
        T2 = sim2.measure_period(n_periods=3)

        # With hardening spring, period should differ
        assert T1 != pytest.approx(T2, rel=0.01), (
            f"alpha should affect period: T(0)={T1:.4f}, T(0.5)={T2:.4f}"
        )

    def test_alpha_affects_amplitude(self):
        """Larger alpha (hardening spring) should reduce amplitude."""
        sim1 = LienardSimulation(
            _make_config(mu=1.0, alpha=0.0, x_0=0.1, dt=0.005)
        )
        sim1.reset()
        A1 = sim1.measure_amplitude(n_periods=5)

        sim2 = LienardSimulation(
            _make_config(mu=1.0, alpha=0.8, x_0=0.1, dt=0.005)
        )
        sim2.reset()
        A2 = sim2.measure_amplitude(n_periods=5)

        # Hardening spring should reduce amplitude
        assert A2 < A1, (
            f"Hardening spring should reduce amplitude: "
            f"A(0)={A1:.4f}, A(0.8)={A2:.4f}"
        )

    def test_mu_zero_no_limit_cycle(self):
        """For mu=0, damping vanishes; trajectory should be periodic but not VdP."""
        sim = LienardSimulation(
            _make_config(mu=0.0, alpha=0.1, x_0=2.0, v_0=0.0, dt=0.005)
        )
        sim.reset()
        # Energy should be conserved (no damping)
        E0 = sim.compute_energy()
        for _ in range(5000):
            sim.step()
        E_final = sim.compute_energy()
        assert abs(E_final - E0) / max(abs(E0), 1e-15) < 1e-4, (
            f"Energy should be conserved for mu=0: drift={abs(E_final - E0):.2e}"
        )

    def test_small_mu_near_harmonic(self):
        """For small mu and alpha, period should be near 2*pi."""
        sim = LienardSimulation(
            _make_config(mu=0.1, alpha=0.01, x_0=0.5, dt=0.005)
        )
        sim.reset()
        T = sim.measure_period(n_periods=3)
        assert abs(T - 2 * np.pi) < 1.0, (
            f"Period {T:.4f} not near 2*pi for small mu, alpha"
        )


# ---------------------------------------------------------------------------
# TestEnergyMethods -- compute_energy and energy behavior
# ---------------------------------------------------------------------------
class TestEnergyMethods:
    def test_compute_energy_at_rest(self):
        """Energy at (x=0, v=0) should be 0."""
        sim = LienardSimulation(_make_config(alpha=0.1))
        sim.reset()
        E = sim.compute_energy(np.array([0.0, 0.0]))
        assert E == pytest.approx(0.0)

    def test_compute_energy_kinetic_only(self):
        """Energy at (0, v) should be v^2/2."""
        sim = LienardSimulation(_make_config(alpha=0.1))
        sim.reset()
        E = sim.compute_energy(np.array([0.0, 3.0]))
        assert E == pytest.approx(4.5)

    def test_compute_energy_potential_only(self):
        """Energy at (x, 0) should be x^2/2 + alpha*x^4/4."""
        sim = LienardSimulation(_make_config(alpha=0.5))
        sim.reset()
        x = 2.0
        expected = 0.5 * x**2 + 0.25 * 0.5 * x**4  # 2 + 2 = 4
        E = sim.compute_energy(np.array([x, 0.0]))
        assert E == pytest.approx(expected)

    def test_compute_energy_full(self):
        """Energy at (x, v) should be v^2/2 + x^2/2 + alpha*x^4/4."""
        sim = LienardSimulation(_make_config(alpha=0.2))
        sim.reset()
        x, v = 1.5, 1.0
        expected = 0.5 * v**2 + 0.5 * x**2 + 0.25 * 0.2 * x**4
        E = sim.compute_energy(np.array([x, v]))
        assert E == pytest.approx(expected)

    def test_compute_energy_uses_current_state(self):
        """compute_energy() without arg should use current state."""
        sim = LienardSimulation(_make_config(alpha=0.1))
        state = sim.reset()
        E_explicit = sim.compute_energy(state)
        E_implicit = sim.compute_energy()
        assert E_explicit == pytest.approx(E_implicit)

    def test_energy_dissipates_mu_positive(self):
        """For mu > 0, energy should not be conserved (dissipation + injection)."""
        sim = LienardSimulation(
            _make_config(mu=1.0, alpha=0.1, x_0=3.0, v_0=0.0, dt=0.005)
        )
        sim.reset()
        E0 = sim.compute_energy()
        energies = [E0]
        for _ in range(5000):
            sim.step()
            energies.append(sim.compute_energy())
        # Energy should change (not constant) for mu > 0
        energies = np.array(energies)
        assert not np.allclose(energies, E0, atol=0.01), (
            "Energy should vary for mu > 0"
        )

    def test_energy_bounded_on_limit_cycle(self):
        """After transient, energy on limit cycle should be bounded."""
        sim = LienardSimulation(
            _make_config(mu=2.0, alpha=0.1, x_0=2.0, dt=0.005)
        )
        sim.reset()
        # Skip transient
        for _ in range(20000):
            sim.step()
        # Measure energy over limit cycle
        energies = []
        for _ in range(5000):
            sim.step()
            energies.append(sim.compute_energy())
        energies = np.array(energies)
        assert np.all(np.isfinite(energies))
        assert np.all(energies > 0), "Energy should be positive"
        assert np.all(energies < 100), f"Energy diverged: max={energies.max()}"


# ---------------------------------------------------------------------------
# TestAmplitudeAndPeriodMethods -- trajectory analysis methods
# ---------------------------------------------------------------------------
class TestAmplitudeAndPeriodMethods:
    def test_amplitude_from_trajectory(self):
        """amplitude_from_trajectory should return max |x|."""
        sim = LienardSimulation(_make_config())
        states = np.array([[1.0, 0.0], [-2.0, 0.5], [0.5, -1.0]])
        A = sim.amplitude_from_trajectory(states)
        assert A == pytest.approx(2.0)

    def test_period_from_trajectory(self):
        """period_from_trajectory should detect period of sine-like signal."""
        sim = LienardSimulation(_make_config())
        dt = 0.01
        t = np.arange(0, 20 * np.pi, dt)
        x = np.sin(t)
        v = np.cos(t)
        states = np.column_stack([x, v])
        T = sim.period_from_trajectory(states, dt)
        assert abs(T - 2 * np.pi) < 0.1, (
            f"Period {T:.4f} not near 2*pi for sine"
        )

    def test_period_from_trajectory_no_crossings(self):
        """Should return inf if no zero crossings."""
        sim = LienardSimulation(_make_config())
        states = np.array([[1.0, 0.0], [2.0, 0.0], [1.5, 0.0]])
        T = sim.period_from_trajectory(states, 0.01)
        assert T == float("inf")

    def test_approximate_period_small_mu(self):
        """Approximate period for small mu should be near 2*pi."""
        sim = LienardSimulation(_make_config(mu=0.01, alpha=0.0))
        T_approx = sim.approximate_period
        assert abs(T_approx - 2 * np.pi) < 0.5

    def test_approximate_period_large_mu(self):
        """Approximate period for large mu should scale with mu."""
        sim = LienardSimulation(_make_config(mu=10.0, alpha=0.0))
        T_approx = sim.approximate_period
        T_theory = (3 - 2 * np.log(2)) * 10.0
        assert abs(T_approx - T_theory) < 2.0


# ---------------------------------------------------------------------------
# TestRediscovery -- data generation and sweep functions
# ---------------------------------------------------------------------------
class TestRediscovery:
    def test_ode_data_generation(self):
        """generate_ode_data should produce correct shapes."""
        from simulating_anything.rediscovery.lienard import generate_ode_data

        data = generate_ode_data(mu=1.0, alpha=0.1, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["mu"] == 1.0
        assert data["alpha"] == 0.1
        assert np.all(np.isfinite(data["states"]))

    def test_mu_sweep(self):
        """generate_mu_sweep should return correct array lengths."""
        from simulating_anything.rediscovery.lienard import generate_mu_sweep

        data = generate_mu_sweep(n_mu=5, alpha=0.1, dt=0.01)
        assert len(data["mu"]) == 5
        assert len(data["period"]) == 5
        assert len(data["amplitude"]) == 5
        assert data["alpha"] == 0.1

    def test_mu_sweep_periods_positive(self):
        """All periods from mu sweep should be positive and finite."""
        from simulating_anything.rediscovery.lienard import generate_mu_sweep

        data = generate_mu_sweep(n_mu=5, alpha=0.1, dt=0.005)
        valid = np.isfinite(data["period"])
        assert np.sum(valid) >= 3, "Most periods should be finite"
        assert np.all(data["period"][valid] > 0), "Periods should be positive"

    def test_alpha_sweep(self):
        """generate_alpha_sweep should return correct array lengths."""
        from simulating_anything.rediscovery.lienard import generate_alpha_sweep

        data = generate_alpha_sweep(n_alpha=5, mu=1.0, dt=0.01)
        assert len(data["alpha"]) == 5
        assert len(data["period"]) == 5
        assert len(data["amplitude"]) == 5
        assert data["mu"] == 1.0

    def test_alpha_sweep_amplitudes_positive(self):
        """Amplitudes from alpha sweep should be positive."""
        from simulating_anything.rediscovery.lienard import generate_alpha_sweep

        data = generate_alpha_sweep(n_alpha=5, mu=1.0, dt=0.005)
        valid = np.isfinite(data["amplitude"]) & (data["amplitude"] > 0)
        assert np.sum(valid) >= 3, "Most amplitudes should be positive"

    def test_energy_check_conserved(self):
        """Energy check with mu=0 should show conservation."""
        from simulating_anything.rediscovery.lienard import generate_energy_check

        data = generate_energy_check(mu=0.0, alpha=0.1, n_steps=1000, dt=0.001)
        assert data["max_relative_drift"] < 1e-6, (
            f"Energy drift too large: {data['max_relative_drift']:.2e}"
        )

    def test_energy_check_not_conserved(self):
        """Energy check with mu > 0 should show non-conservation."""
        from simulating_anything.rediscovery.lienard import generate_energy_check

        data = generate_energy_check(mu=1.0, alpha=0.1, n_steps=1000, dt=0.005)
        assert data["max_relative_drift"] > 1e-4, (
            "Energy should not be conserved for mu > 0"
        )

    def test_output_format(self):
        """generate_ode_data output should have expected keys."""
        from simulating_anything.rediscovery.lienard import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.01)
        expected_keys = {"time", "states", "x", "v", "mu", "alpha", "dt"}
        assert expected_keys.issubset(data.keys())
