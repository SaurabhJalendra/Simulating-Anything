"""Tests for the Glycolytic Oscillator (Higgins two-variable model)."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.glycolytic_oscillator import (
    GlycolyticOscillatorSimulation,
    compute_hopf_v_in,
    compute_hopf_v_in_numerical,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    v_in: float = 1.0,
    k1: float = 0.1,
    k2: float = 0.5,
    dt: float = 0.01,
    S_0: float = 2.5,
    P_0: float = 2.1,
    n_steps: int = 1000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.GLYCOLYTIC_OSCILLATOR,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "v_in": v_in,
            "k1": k1,
            "k2": k2,
            "S_0": S_0,
            "P_0": P_0,
        },
    )


# ---------------------------------------------------------------------------
# Creation and basic state
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorCreation:
    def test_create_simulation(self):
        sim = GlycolyticOscillatorSimulation(_make_config())
        assert sim.v_in == 1.0
        assert sim.k1 == 0.1
        assert sim.k2 == 0.5

    def test_default_initial_conditions(self):
        sim = GlycolyticOscillatorSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [2.5, 2.1])

    def test_custom_initial_conditions(self):
        sim = GlycolyticOscillatorSimulation(_make_config(S_0=3.0, P_0=2.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [3.0, 2.0])

    def test_initial_state_shape(self):
        sim = GlycolyticOscillatorSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_state_dtype(self):
        sim = GlycolyticOscillatorSimulation(_make_config())
        state = sim.reset()
        assert state.dtype == np.float64

    def test_custom_parameters(self):
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=1.5, k1=0.2, k2=0.8)
        )
        assert sim.v_in == 1.5
        assert sim.k1 == 0.2
        assert sim.k2 == 0.8


# ---------------------------------------------------------------------------
# Step and observe
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorDynamics:
    def test_step_advances_state(self):
        """Start away from fixed point so a single step is visible."""
        sim = GlycolyticOscillatorSimulation(
            _make_config(S_0=3.0, P_0=2.0)
        )
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = GlycolyticOscillatorSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_run_returns_trajectory(self):
        sim = GlycolyticOscillatorSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101

    def test_trajectory_timestamps_correct(self):
        dt = 0.01
        sim = GlycolyticOscillatorSimulation(_make_config(dt=dt))
        traj = sim.run(n_steps=50)
        expected = np.arange(51) * dt
        np.testing.assert_allclose(traj.timestamps, expected, rtol=1e-10)

    def test_deterministic_trajectory(self):
        """Same initial conditions should produce identical results."""
        config = _make_config(S_0=3.0, P_0=2.0)
        sim1 = GlycolyticOscillatorSimulation(config)
        sim2 = GlycolyticOscillatorSimulation(config)
        traj1 = sim1.run(n_steps=200)
        traj2 = sim2.run(n_steps=200)
        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_step_count_increments(self):
        sim = GlycolyticOscillatorSimulation(_make_config())
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2

    def test_short_trajectory_bounded(self):
        """Short trajectory should remain bounded for stable parameters."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 2.0  # Stable regime
        S_star = k2**2 / (k1 * v_in)
        P_star = v_in / k2
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, S_0=S_star + 0.1, P_0=P_star + 0.1, dt=0.01)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
            S, P = sim.observe()
            assert abs(S) < 100, f"S diverged: {S}"
            assert abs(P) < 100, f"P diverged: {P}"


# ---------------------------------------------------------------------------
# Non-negativity (in stable regime)
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorNonNegativity:
    def test_concentrations_non_negative_stable(self):
        """S and P should remain non-negative in the stable regime."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 2.0
        S_star = k2**2 / (k1 * v_in)
        P_star = v_in / k2
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, dt=0.005, S_0=S_star + 0.5, P_0=P_star + 0.5)
        )
        sim.reset()
        for _ in range(10000):
            sim.step()
            S, P = sim.observe()
            assert S > -0.01, f"S went negative: {S}"
            assert P > -0.01, f"P went negative: {P}"

    def test_concentrations_non_negative_near_threshold(self):
        """Near-threshold trajectory: P should stay positive for many cycles."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 0.95
        S_star = k2**2 / (k1 * v_in)
        P_star = v_in / k2
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, dt=0.005, S_0=S_star + 0.01, P_0=P_star + 0.01)
        )
        sim.reset()
        omega = sim.oscillation_frequency
        T = 2 * np.pi / omega
        # Run for several periods
        for _ in range(int(5 * T / 0.005)):
            sim.step()
            S, P = sim.observe()
            assert S > -0.01, f"S went negative: {S}"
            assert P > -0.01, f"P went negative: {P}"


# ---------------------------------------------------------------------------
# Fixed point
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorFixedPoint:
    def test_fixed_point_formula(self):
        """P* = v_in/k2, S* = k2^2/(k1*v_in)."""
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=1.0, k1=0.1, k2=0.5)
        )
        S_star, P_star = sim.fixed_point
        assert P_star == pytest.approx(1.0 / 0.5)  # v_in/k2 = 2.0
        assert S_star == pytest.approx(0.25 / 0.1)  # k2^2/(k1*v_in) = 2.5

    def test_fixed_point_different_params(self):
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=1.0, k1=0.2, k2=0.8)
        )
        S_star, P_star = sim.fixed_point
        assert P_star == pytest.approx(1.0 / 0.8, rel=1e-10)
        assert S_star == pytest.approx(0.64 / 0.2, rel=1e-10)

    def test_derivatives_at_fixed_point(self):
        """At the fixed point, derivatives should be zero."""
        v_in, k1, k2 = 1.0, 0.1, 0.5
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, k1=k1, k2=k2)
        )
        sim.reset()
        fp = np.array(sim.fixed_point)
        dy = sim._derivatives(fp)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-12)

    def test_derivatives_at_fixed_point_other_params(self):
        v_in, k1, k2 = 1.0, 0.2, 0.8
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, k1=k1, k2=k2)
        )
        sim.reset()
        fp = np.array(sim.fixed_point)
        dy = sim._derivatives(fp)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-12)

    def test_stable_spiral_converges_to_fixed_point(self):
        """Above Hopf threshold (large v_in), system converges to fixed point."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 2.0  # Stable spiral
        S_star = k2**2 / (k1 * v_in)
        P_star = v_in / k2
        sim = GlycolyticOscillatorSimulation(
            _make_config(
                v_in=v_in, k1=k1, k2=k2,
                S_0=S_star + 0.3, P_0=P_star + 0.3, dt=0.01,
            )
        )
        sim.reset()
        for _ in range(50000):
            sim.step()
        S, P = sim.observe()
        np.testing.assert_allclose([S, P], [S_star, P_star], atol=0.05)


# ---------------------------------------------------------------------------
# Hopf bifurcation
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorHopf:
    def test_hopf_v_in_positive(self):
        v_in_c = compute_hopf_v_in(0.1, 0.5)
        assert v_in_c > 0

    def test_hopf_v_in_formula(self):
        """v_in_c = k2*sqrt(k2/k1)."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        expected = k2 * np.sqrt(k2 / k1)
        assert v_in_c == pytest.approx(expected, rel=1e-10)

    def test_hopf_v_in_numerical_matches_analytical(self):
        """Numerical root-finding should match analytical formula."""
        k1, k2 = 0.1, 0.5
        v_in_c_anal = compute_hopf_v_in(k1, k2)
        v_in_c_num = compute_hopf_v_in_numerical(k1, k2)
        assert v_in_c_num == pytest.approx(v_in_c_anal, rel=0.05)

    def test_hopf_trace_zero_at_threshold(self):
        """At v_in_c, the Jacobian trace should be zero."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in_c, k1=k1, k2=k2)
        )
        trace = sim.jacobian_trace
        assert abs(trace) < 1e-10, f"Trace at Hopf: {trace}"

    def test_jacobian_det_positive(self):
        """Determinant should always be positive for positive parameters."""
        sim = GlycolyticOscillatorSimulation(_make_config())
        assert sim.jacobian_det > 0

    def test_stable_spiral_above_hopf(self):
        """Above v_in_c the fixed point is a stable spiral."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in_c * 2.0, k1=k1, k2=k2)
        )
        assert sim.is_stable_spiral
        assert not sim.is_unstable_spiral

    def test_unstable_spiral_below_hopf(self):
        """Below v_in_c the fixed point is an unstable spiral."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in_c * 0.5, k1=k1, k2=k2)
        )
        assert sim.is_unstable_spiral
        assert not sim.is_stable_spiral

    def test_default_params_near_threshold(self):
        """Default v_in=1.0 should be close to v_in_c~1.118."""
        sim = GlycolyticOscillatorSimulation(_make_config())
        v_in_c = compute_hopf_v_in(sim.k1, sim.k2)
        assert sim.v_in < v_in_c
        assert sim.v_in > v_in_c * 0.5

    def test_near_threshold_oscillates(self):
        """Near Hopf threshold (just below), system should show oscillation."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 0.95  # Just below threshold
        S_star = k2**2 / (k1 * v_in)
        P_star = v_in / k2
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, k1=k1, k2=k2, dt=0.01,
                         S_0=S_star + 0.01, P_0=P_star + 0.01)
        )
        sim.reset()
        # Measure for several oscillation periods
        omega = sim.oscillation_frequency
        T = 2 * np.pi / omega
        n_steps = int(3 * T / 0.01)
        P_vals = []
        for _ in range(n_steps):
            sim.step()
            P_vals.append(sim.observe()[1])
        amplitude = max(P_vals) - min(P_vals)
        assert amplitude > 0.01, f"No oscillation: amplitude={amplitude}"

    def test_hopf_v_in_increases_with_k2(self):
        """Larger k2 should increase the Hopf threshold."""
        v_in_c1 = compute_hopf_v_in(0.1, 0.3)
        v_in_c2 = compute_hopf_v_in(0.1, 0.8)
        assert v_in_c2 > v_in_c1

    def test_hopf_v_in_decreases_with_k1(self):
        """Larger k1 should decrease the Hopf threshold."""
        v_in_c1 = compute_hopf_v_in(0.05, 0.5)
        v_in_c2 = compute_hopf_v_in(0.2, 0.5)
        assert v_in_c2 < v_in_c1


# ---------------------------------------------------------------------------
# Eigenvalues and oscillation frequency
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorEigenvalues:
    def test_eigenvalues_complex_below_hopf(self):
        """Below Hopf, eigenvalues should be complex with positive real part."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in_c * 0.8, k1=k1, k2=k2)
        )
        lam1, lam2 = sim.eigenvalues
        assert lam1.imag != 0
        assert lam1.real > 0
        assert lam2 == pytest.approx(lam1.conjugate())

    def test_eigenvalues_complex_above_hopf(self):
        """Above Hopf, eigenvalues should be complex with negative real part."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in_c * 1.5, k1=k1, k2=k2)
        )
        lam1, lam2 = sim.eigenvalues
        assert lam1.imag != 0
        assert lam1.real < 0

    def test_oscillation_frequency_positive(self):
        """Should return positive frequency when eigenvalues are complex."""
        sim = GlycolyticOscillatorSimulation(_make_config())
        assert sim.oscillation_frequency > 0

    def test_oscillation_frequency_formula(self):
        """omega = sqrt(det - (tr/2)^2)."""
        v_in, k1, k2 = 1.0, 0.1, 0.5
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, k1=k1, k2=k2)
        )
        tr = sim.jacobian_trace
        det = sim.jacobian_det
        expected = np.sqrt(det - (tr / 2)**2)
        assert sim.oscillation_frequency == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Period measurement
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorPeriod:
    def test_measure_period_near_threshold(self):
        """Near threshold should return finite period."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 0.95
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, k1=k1, k2=k2, dt=0.005)
        )
        sim.reset()
        period = sim.measure_oscillation_period(n_cycles=3)
        assert np.isfinite(period)
        assert period > 0

    def test_measure_period_stable(self):
        """Should return inf when above Hopf threshold (stable spiral)."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 2.0  # Stable
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, k1=k1, k2=k2, dt=0.01)
        )
        sim.reset()
        period = sim.measure_oscillation_period()
        assert period == float("inf")

    def test_period_matches_eigenvalue_prediction(self):
        """Measured period should be close to 2*pi/omega from eigenvalues."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 0.95
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, k1=k1, k2=k2, dt=0.005)
        )
        sim.reset()
        measured = sim.measure_oscillation_period(n_cycles=3)
        predicted = 2 * np.pi / sim.oscillation_frequency
        # Near-threshold oscillations should match linearized prediction well
        assert measured == pytest.approx(predicted, rel=0.15)

    def test_measure_amplitude_near_threshold(self):
        """Should return positive amplitude near threshold."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 0.95
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, k1=k1, k2=k2, dt=0.005)
        )
        sim.reset()
        amp_S, amp_P = sim.measure_peak_amplitude(n_cycles=3)
        assert amp_S > 0.001
        assert amp_P > 0.001

    def test_amplitude_returns_two_values(self):
        """measure_peak_amplitude should return (S_amplitude, P_amplitude)."""
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=1.0, dt=0.005)
        )
        sim.reset()
        result = sim.measure_peak_amplitude(n_cycles=2)
        assert len(result) == 2

    def test_period_consistency(self):
        """Two independent period measurements should agree."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 0.95
        config = _make_config(v_in=v_in, k1=k1, k2=k2, dt=0.005)

        sim1 = GlycolyticOscillatorSimulation(config)
        sim1.reset()
        p1 = sim1.measure_oscillation_period(n_cycles=3)

        sim2 = GlycolyticOscillatorSimulation(config)
        sim2.reset()
        p2 = sim2.measure_oscillation_period(n_cycles=3)

        assert p1 == pytest.approx(p2, rel=0.05)


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorSweeps:
    def test_k2_sweep_fixed_point_changes(self):
        """Changing k2 should move the fixed point."""
        sim1 = GlycolyticOscillatorSimulation(
            _make_config(v_in=1.0, k1=0.1, k2=0.3)
        )
        sim2 = GlycolyticOscillatorSimulation(
            _make_config(v_in=1.0, k1=0.1, k2=0.8)
        )
        S1, P1 = sim1.fixed_point
        S2, P2 = sim2.fixed_point
        assert S1 != pytest.approx(S2, rel=0.01)
        assert P1 != pytest.approx(P2, rel=0.01)

    def test_k1_sweep_fixed_point_changes(self):
        """Changing k1 should move the fixed point (S* depends on k1)."""
        sim1 = GlycolyticOscillatorSimulation(
            _make_config(v_in=1.0, k1=0.05, k2=0.5)
        )
        sim2 = GlycolyticOscillatorSimulation(
            _make_config(v_in=1.0, k1=0.2, k2=0.5)
        )
        S1, _ = sim1.fixed_point
        S2, _ = sim2.fixed_point
        # Larger k1 => smaller S* since S* = k2^2/(k1*v_in)
        assert S1 > S2

    def test_stable_regime_small_amplitude(self):
        """v_in well above v_in_c: decaying oscillation should have small amplitude."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        v_in = v_in_c * 2.0
        S_star = k2**2 / (k1 * v_in)
        P_star = v_in / k2
        sim = GlycolyticOscillatorSimulation(
            _make_config(
                v_in=v_in, k1=k1, k2=k2, dt=0.005,
                S_0=S_star + 0.1, P_0=P_star + 0.1,
            )
        )
        sim.reset()
        # Run past transient
        for _ in range(int(500 / 0.005)):
            sim.step()
        # Measure residual oscillation
        S_vals = []
        P_vals = []
        for _ in range(int(100 / 0.005)):
            sim.step()
            S_vals.append(sim.observe()[0])
            P_vals.append(sim.observe()[1])
        assert max(S_vals) - min(S_vals) < 0.01
        assert max(P_vals) - min(P_vals) < 0.01


# ---------------------------------------------------------------------------
# Derivatives / ODE correctness
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorDerivatives:
    def test_derivatives_manual(self):
        """dS/dt = v_in - k1*S*P^2, dP/dt = k1*S*P^2 - k2*P."""
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=0.5, k1=0.1, k2=0.5)
        )
        sim.reset()
        state = np.array([4.0, 2.0])
        dy = sim._derivatives(state)
        # dS/dt = 0.5 - 0.1 * 4.0 * 4.0 = 0.5 - 1.6 = -1.1
        # dP/dt = 0.1 * 4.0 * 4.0 - 0.5 * 2.0 = 1.6 - 1.0 = 0.6
        np.testing.assert_allclose(dy, [-1.1, 0.6], atol=1e-12)

    def test_derivatives_zero_product(self):
        """With P=0, dS/dt = v_in, dP/dt = 0."""
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=0.5, k1=0.1, k2=0.5)
        )
        sim.reset()
        state = np.array([5.0, 0.0])
        dy = sim._derivatives(state)
        np.testing.assert_allclose(dy, [0.5, 0.0], atol=1e-12)

    def test_autocatalytic_term_scales_as_p_squared(self):
        """The autocatalytic term k1*S*P^2 should scale as P^2."""
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=0.0, k1=0.1, k2=0.0)
        )
        sim.reset()
        S = 1.0
        dy1 = sim._derivatives(np.array([S, 1.0]))
        dy2 = sim._derivatives(np.array([S, 2.0]))
        # Ratio should be (2^2)/(1^2) = 4
        ratio = dy2[1] / dy1[1]
        assert ratio == pytest.approx(4.0, rel=1e-10)

    def test_mass_conservation_at_fixed_point(self):
        """dS/dt + dP/dt = v_in - k2*P at any state."""
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=1.0, k1=0.1, k2=0.5)
        )
        sim.reset()
        state = np.array([3.0, 2.5])
        dy = sim._derivatives(state)
        # dS/dt + dP/dt = v_in - k2*P
        assert dy[0] + dy[1] == pytest.approx(1.0 - 0.5 * 2.5, rel=1e-10)


# ---------------------------------------------------------------------------
# Jacobian properties
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorJacobian:
    def test_jacobian_trace_simplified_formula(self):
        """tr(J) = k2 - k1*v_in^2/k2^2."""
        v_in, k1, k2 = 1.0, 0.1, 0.5
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, k1=k1, k2=k2)
        )
        expected = k2 - k1 * v_in**2 / k2**2
        assert sim.jacobian_trace == pytest.approx(expected, rel=1e-10)

    def test_jacobian_det_formula(self):
        """det(J) = k1*P*^2*k2."""
        v_in, k1, k2 = 1.0, 0.1, 0.5
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in, k1=k1, k2=k2)
        )
        _, P_star = sim.fixed_point
        expected = k1 * P_star**2 * k2
        assert sim.jacobian_det == pytest.approx(expected, rel=1e-10)

    def test_trace_positive_below_hopf(self):
        """Below v_in_c, the trace should be positive (unstable)."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in_c * 0.5, k1=k1, k2=k2)
        )
        assert sim.jacobian_trace > 0

    def test_trace_negative_above_hopf(self):
        """Above v_in_c, the trace should be negative (stable)."""
        k1, k2 = 0.1, 0.5
        v_in_c = compute_hopf_v_in(k1, k2)
        sim = GlycolyticOscillatorSimulation(
            _make_config(v_in=v_in_c * 2.0, k1=k1, k2=k2)
        )
        assert sim.jacobian_trace < 0

    def test_det_always_positive(self):
        """det(J) = k1*P*^2*k2 > 0 for any positive parameters."""
        for v_in in [0.1, 0.5, 1.0, 5.0]:
            for k1 in [0.01, 0.1, 1.0]:
                sim = GlycolyticOscillatorSimulation(
                    _make_config(v_in=v_in, k1=k1, k2=0.5)
                )
                assert sim.jacobian_det > 0


# ---------------------------------------------------------------------------
# Rediscovery data generation
# ---------------------------------------------------------------------------


class TestGlycolyticOscillatorRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.glycolytic_oscillator import (
            generate_ode_data,
        )

        data = generate_ode_data(v_in=1.0, k1=0.1, k2=0.5, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["v_in"] == 1.0
        assert data["k1"] == 0.1
        assert data["k2"] == 0.5

    def test_bifurcation_data_generation(self):
        from simulating_anything.rediscovery.glycolytic_oscillator import (
            generate_bifurcation_data,
        )

        data = generate_bifurcation_data(k1=0.1, k2=0.5, n_v_in=5, dt=0.01)
        assert len(data["v_in"]) == 5
        assert len(data["amplitude_S"]) == 5
        assert len(data["amplitude_P"]) == 5
        assert data["v_in_c_theory"] > 0

    def test_period_data_generation(self):
        from simulating_anything.rediscovery.glycolytic_oscillator import (
            generate_period_data,
        )

        data = generate_period_data(k1=0.1, k2=0.5, n_v_in=3, dt=0.01)
        assert len(data["v_in"]) == 3
        assert len(data["period"]) == 3
        assert data["v_in_c"] > 0

    def test_ode_data_S_and_P_arrays(self):
        from simulating_anything.rediscovery.glycolytic_oscillator import (
            generate_ode_data,
        )

        data = generate_ode_data(n_steps=100, dt=0.01)
        assert data["S"].shape == (101,)
        assert data["P"].shape == (101,)
        np.testing.assert_array_equal(data["S"], data["states"][:, 0])
        np.testing.assert_array_equal(data["P"], data["states"][:, 1])
