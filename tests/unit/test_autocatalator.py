"""Tests for the Autocatalator 3-species chemical oscillator."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.autocatalator import AutocatalatorSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    mu: float = 0.002,
    kappa: float = 65.0,
    sigma: float = 0.005,
    delta: float = 0.2,
    dt: float = 0.01,
    a_0: float = 0.5,
    b_0: float = 1.0,
    c_0: float = 0.5,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.AUTOCATALATOR,
        dt=dt,
        n_steps=1000,
        parameters={
            "mu": mu, "kappa": kappa, "sigma": sigma, "delta": delta,
            "a_0": a_0, "b_0": b_0, "c_0": c_0,
        },
    )


class TestAutocatalatorSimulation:
    def test_initial_state_shape(self):
        sim = AutocatalatorSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_values(self):
        sim = AutocatalatorSimulation(_make_config(a_0=0.3, b_0=0.7, c_0=0.4))
        state = sim.reset()
        np.testing.assert_allclose(state, [0.3, 0.7, 0.4])

    def test_initial_state_dtype(self):
        sim = AutocatalatorSimulation(_make_config())
        state = sim.reset()
        assert state.dtype == np.float64

    def test_step_advances_state(self):
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = AutocatalatorSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_parameter_defaults(self):
        config = SimulationConfig(
            domain=Domain.AUTOCATALATOR,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = AutocatalatorSimulation(config)
        assert sim.mu == 0.002
        assert sim.kappa == 65.0
        assert sim.sigma == 0.005
        assert sim.delta == 0.2

    def test_custom_parameters(self):
        sim = AutocatalatorSimulation(
            _make_config(mu=0.005, kappa=50.0, sigma=0.01, delta=0.3)
        )
        assert sim.mu == 0.005
        assert sim.kappa == 50.0
        assert sim.sigma == 0.01
        assert sim.delta == 0.3

    def test_run_returns_trajectory(self):
        sim = AutocatalatorSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_trajectory_stays_bounded(self):
        """Trajectory should remain bounded for default parameters."""
        sim = AutocatalatorSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"State became NaN/Inf: {state}"
            assert np.linalg.norm(state) < 1000, f"Trajectory diverged: {state}"

    def test_concentrations_non_negative(self):
        """Concentrations should stay non-negative for small enough dt."""
        sim = AutocatalatorSimulation(_make_config(dt=0.001))
        sim.reset()
        for _ in range(5000):
            sim.step()
            a, b, c = sim.observe()
            # Small numerical undershoot allowed
            assert a > -0.01, f"a went negative: {a}"
            assert b > -0.01, f"b went negative: {b}"
            assert c > -0.01, f"c went negative: {c}"


class TestAutocatalatorDerivatives:
    def test_derivatives_at_fixed_point(self):
        """Derivatives should be approximately zero at the fixed point."""
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1
        fp = fps[0]
        dy = sim._derivatives(fp)
        np.testing.assert_allclose(dy, [0.0, 0.0, 0.0], atol=1e-10)

    def test_derivatives_known_values(self):
        """Test derivatives at a specific known point."""
        sim = AutocatalatorSimulation(
            _make_config(mu=0.002, kappa=65.0, sigma=0.005, delta=0.2)
        )
        sim.reset()
        # At state [1, 1, 1]:
        # da = 0.002*(65+1) - 1*1 - 1 = 0.132 - 2 = -1.868
        # db = (1*1 + 1 - 1)/0.005 = (1)/0.005 = 200
        # dc = (1 - 1)/0.2 = 0
        dy = sim._derivatives(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(dy[0], -1.868, rtol=1e-10)
        np.testing.assert_allclose(dy[1], 200.0, rtol=1e-10)
        np.testing.assert_allclose(dy[2], 0.0, atol=1e-10)

    def test_c_equation_decoupled(self):
        """dc/dt depends only on b and c, not on a."""
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        # Two states differing only in a
        s1 = np.array([0.5, 1.0, 0.5])
        s2 = np.array([1.5, 1.0, 0.5])
        dy1 = sim._derivatives(s1)
        dy2 = sim._derivatives(s2)
        # dc/dt should be the same
        assert dy1[2] == pytest.approx(dy2[2])

    def test_b_equation_independent_of_c(self):
        """db/dt depends on a and b but not on c."""
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        s1 = np.array([0.5, 1.0, 0.3])
        s2 = np.array([0.5, 1.0, 0.9])
        dy1 = sim._derivatives(s1)
        dy2 = sim._derivatives(s2)
        assert dy1[1] == pytest.approx(dy2[1])


class TestAutocatalatorFixedPoints:
    def test_one_fixed_point(self):
        """Default parameters should give exactly one physical fixed point."""
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 1

    def test_fixed_point_formula(self):
        """Verify fixed point against analytical formula."""
        mu, kappa = 0.002, 65.0
        sim = AutocatalatorSimulation(_make_config(mu=mu, kappa=kappa))
        sim.reset()
        fps = sim.fixed_points
        fp = fps[0]

        b_star_expected = mu * kappa / (1.0 - mu)
        c_star_expected = b_star_expected
        a_star_expected = b_star_expected / (b_star_expected**2 + 1.0)

        np.testing.assert_allclose(fp[0], a_star_expected, rtol=1e-10)
        np.testing.assert_allclose(fp[1], b_star_expected, rtol=1e-10)
        np.testing.assert_allclose(fp[2], c_star_expected, rtol=1e-10)

    def test_fixed_point_different_params(self):
        """Fixed point changes correctly with parameters."""
        sim1 = AutocatalatorSimulation(_make_config(mu=0.001, kappa=65.0))
        sim2 = AutocatalatorSimulation(_make_config(mu=0.005, kappa=65.0))
        sim1.reset()
        sim2.reset()
        fp1 = sim1.fixed_points[0]
        fp2 = sim2.fixed_points[0]
        # Larger mu => larger b*
        assert fp2[1] > fp1[1]

    def test_no_fixed_point_mu_ge_1(self):
        """No physical fixed point when mu >= 1."""
        sim = AutocatalatorSimulation(_make_config(mu=1.0))
        sim.reset()
        fps = sim.fixed_points
        assert len(fps) == 0

    def test_c_star_equals_b_star(self):
        """At the fixed point, c* = b*."""
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        fp = sim.fixed_points[0]
        assert fp[2] == pytest.approx(fp[1])


class TestAutocatalatorJacobian:
    def test_jacobian_shape(self):
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        J = sim.jacobian(np.array([0.5, 1.0, 0.5]))
        assert J.shape == (3, 3)

    def test_jacobian_at_fixed_point(self):
        """Jacobian at fixed point should match analytical form."""
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        fp = sim.fixed_points[0]
        J = sim.jacobian(fp)
        a_s, b_s, _c_s = fp

        # Check specific entries
        assert J[0, 0] == pytest.approx(-(b_s**2 + 1.0))
        assert J[0, 1] == pytest.approx(-2.0 * a_s * b_s)
        assert J[0, 2] == pytest.approx(sim.mu)
        assert J[1, 0] == pytest.approx((b_s**2 + 1.0) / sim.sigma)
        assert J[2, 1] == pytest.approx(1.0 / sim.delta)
        assert J[2, 2] == pytest.approx(-1.0 / sim.delta)

    def test_jacobian_trace_matches_divergence(self):
        """Trace of Jacobian should equal the divergence."""
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        state = np.array([0.3, 0.8, 0.4])
        J = sim.jacobian(state)
        div = sim.compute_divergence(state)
        assert np.trace(J) == pytest.approx(div, rel=1e-10)


class TestAutocatalatorDivergence:
    def test_divergence_is_negative(self):
        """Divergence should be strongly negative (dissipative system)."""
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        div = sim.compute_divergence(np.array([0.5, 1.0, 0.5]))
        assert div < 0

    def test_divergence_formula(self):
        """Check divergence formula: -(b^2+1) + (2*a*b-1)/sigma - 1/delta."""
        sim = AutocatalatorSimulation(
            _make_config(sigma=0.005, delta=0.2)
        )
        sim.reset()
        state = np.array([0.5, 1.0, 0.5])
        a, b = state[0], state[1]
        expected = -(b**2 + 1.0) + (2.0 * a * b - 1.0) / 0.005 - 1.0 / 0.2
        actual = sim.compute_divergence(state)
        assert actual == pytest.approx(expected, rel=1e-10)

    def test_divergence_depends_on_state(self):
        """Divergence is state-dependent (unlike constant-divergence systems)."""
        sim = AutocatalatorSimulation(_make_config())
        sim.reset()
        div1 = sim.compute_divergence(np.array([0.1, 0.5, 0.3]))
        div2 = sim.compute_divergence(np.array([0.5, 2.0, 1.0]))
        assert div1 != pytest.approx(div2)


class TestAutocatalatorDynamics:
    def test_oscillation_detection(self):
        """System should produce oscillations above Hopf bifurcation.

        The Hopf bifurcation occurs at mu ~ 0.015 for default kappa/sigma/delta.
        With mu=0.016 and dt=0.0001 (stiff system), b oscillates with large
        relaxation amplitude.
        """
        sim = AutocatalatorSimulation(
            _make_config(mu=0.016, dt=0.0001)
        )
        sim.reset()
        # Skip transient
        for _ in range(100000):
            sim.step()
        # Collect b values
        b_vals = []
        for _ in range(50000):
            sim.step()
            b_vals.append(sim.observe()[1])
        amplitude = max(b_vals) - min(b_vals)
        assert amplitude > 1.0, f"No oscillation detected, amplitude={amplitude}"

    def test_multiple_timescales(self):
        """With sigma << 1, b should evolve much faster than a and c.

        At a generic state away from the b-nullcline (a*b^2 + a - b != 0),
        the b equation has a 1/sigma factor making it much faster.
        """
        sim = AutocatalatorSimulation(
            _make_config(mu=0.002, sigma=0.005, delta=0.2, dt=0.001)
        )
        sim.reset()
        # Choose a point where the b-numerator is not zero
        state = np.array([0.3, 0.8, 0.4])
        dy = sim._derivatives(state)
        # |db/dt| should be much larger than |da/dt| and |dc/dt|
        # because sigma is very small (0.005)
        assert abs(dy[1]) > abs(dy[0])

    def test_measure_amplitude_positive(self):
        """Amplitude measurement should return a value."""
        sim = AutocatalatorSimulation(_make_config(mu=0.002, dt=0.005))
        sim.reset()
        amp = sim.measure_amplitude(transient_time=200.0)
        assert amp >= 0


class TestAutocatalatorRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.autocatalator import generate_ode_data

        data = generate_ode_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert data["mu"] == 0.002
        assert data["kappa"] == 65.0
        assert data["sigma"] == 0.005
        assert data["delta"] == 0.2

    def test_ode_data_stays_finite(self):
        from simulating_anything.rediscovery.autocatalator import generate_ode_data

        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(np.isfinite(data["states"]))

    def test_bifurcation_data_generation(self):
        from simulating_anything.rediscovery.autocatalator import (
            generate_bifurcation_data,
        )

        data = generate_bifurcation_data(n_mu=5, dt=0.01)
        assert len(data["mu"]) == 5
        assert len(data["amplitude"]) == 5

    def test_period_data_generation(self):
        from simulating_anything.rediscovery.autocatalator import generate_period_data

        data = generate_period_data(n_mu=3, dt=0.01)
        assert len(data["mu"]) == 3
        assert len(data["period"]) == 3

    def test_sindy_ready_data_format(self):
        """Trajectory data should be in the right format for SINDy."""
        from simulating_anything.rediscovery.autocatalator import generate_ode_data

        data = generate_ode_data(n_steps=200, dt=0.01)
        states = data["states"]
        assert states.ndim == 2
        assert states.shape[1] == 3
        assert states.dtype == np.float64
        assert "dt" in data
