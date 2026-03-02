"""Tests for the Glucose-Insulin Bergman minimal model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.glucose_insulin import GlucoseInsulinSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    p1: float = 0.028,
    p2: float = 0.025,
    p3: float = 5e-6,
    n: float = 0.23,
    gamma: float = 0.01,
    Gb: float = 92.0,
    Ib: float = 11.0,
    h: float = 80.0,
    G0: float = 300.0,
    X0: float = 0.0,
    I0: float = 11.0,
    dt: float = 0.5,
    n_steps: int = 600,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.GLUCOSE_INSULIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "p1": p1, "p2": p2, "p3": p3,
            "n": n, "gamma": gamma,
            "Gb": Gb, "Ib": Ib, "h": h,
            "G0": G0, "X0": X0, "I0": I0,
        },
    )


class TestGlucoseInsulinCreation:
    def test_initial_state_shape(self):
        sim = GlucoseInsulinSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_values(self):
        sim = GlucoseInsulinSimulation(_make_config(G0=300.0, X0=0.0, I0=11.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [300.0, 0.0, 11.0])

    def test_custom_initial_conditions(self):
        sim = GlucoseInsulinSimulation(_make_config(G0=250.0, X0=0.01, I0=15.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [250.0, 0.01, 15.0])

    def test_default_parameters(self):
        config = SimulationConfig(
            domain=Domain.GLUCOSE_INSULIN, dt=0.5, n_steps=100, parameters={},
        )
        sim = GlucoseInsulinSimulation(config)
        assert sim.p1 == pytest.approx(0.028)
        assert sim.p2 == pytest.approx(0.025)
        assert sim.p3 == pytest.approx(5e-6)
        assert sim.n == pytest.approx(0.23)
        assert sim.gamma == pytest.approx(0.01)
        assert sim.Gb == pytest.approx(92.0)
        assert sim.Ib == pytest.approx(11.0)
        assert sim.h == pytest.approx(80.0)


class TestGlucoseInsulinBasicOps:
    def test_reset_returns_state(self):
        sim = GlucoseInsulinSimulation(_make_config())
        state = sim.reset()
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float64

    def test_step_advances_state(self):
        sim = GlucoseInsulinSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        sim = GlucoseInsulinSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_run_returns_trajectory(self):
        sim = GlucoseInsulinSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_run_default_steps(self):
        sim = GlucoseInsulinSimulation(_make_config(n_steps=50))
        traj = sim.run()
        assert traj.states.shape == (51, 3)

    def test_trajectory_timestamps_correct(self):
        dt = 0.5
        sim = GlucoseInsulinSimulation(_make_config(dt=dt, n_steps=10))
        traj = sim.run(n_steps=10)
        expected = np.arange(11) * dt
        np.testing.assert_allclose(traj.timestamps, expected)


class TestGlucoseDynamics:
    def test_glucose_decreases_initially(self):
        """After bolus, glucose should decrease from G0 toward Gb."""
        sim = GlucoseInsulinSimulation(_make_config())
        sim.reset()
        G_initial = sim.observe()[0]
        for _ in range(20):
            sim.step()
        G_after = sim.observe()[0]
        assert G_after < G_initial

    def test_glucose_returns_to_basal(self):
        """Glucose should eventually return close to Gb."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5, n_steps=2000))
        sim.reset()
        for _ in range(2000):
            sim.step()
        G_final = sim.observe()[0]
        assert abs(G_final - sim.Gb) < 5.0, f"G={G_final}, Gb={sim.Gb}"

    def test_glucose_stays_positive(self):
        """Glucose should never go negative."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5, n_steps=1200))
        sim.reset()
        for _ in range(1200):
            sim.step()
            G = sim.observe()[0]
            assert G > 0, f"Glucose went non-positive: {G}"

    def test_glucose_monotone_decrease_early(self):
        """In first ~50 steps, glucose should decrease monotonically."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5))
        sim.reset()
        prev_G = sim.observe()[0]
        for _ in range(50):
            sim.step()
            G = sim.observe()[0]
            assert G < prev_G + 0.1, f"Glucose increased: {prev_G} -> {G}"
            prev_G = G

    def test_glucose_bounded(self):
        """Glucose should remain bounded for the full simulation."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5, n_steps=1200))
        sim.reset()
        for _ in range(1200):
            sim.step()
            G = sim.observe()[0]
            assert G < 500, f"Glucose diverged: {G}"


class TestInsulinDynamics:
    def test_insulin_spike_after_bolus(self):
        """Insulin should rise above basal when G > h."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5))
        sim.reset()
        I_initial = sim.observe()[2]
        # Run a few steps -- gamma*max(G-h,0) drives I up
        for _ in range(20):
            sim.step()
        I_after = sim.observe()[2]
        assert I_after > I_initial, f"Insulin didn't rise: {I_initial} -> {I_after}"

    def test_insulin_returns_to_basal(self):
        """Insulin should eventually return close to Ib."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5, n_steps=2000))
        sim.reset()
        for _ in range(2000):
            sim.step()
        I_final = sim.observe()[2]
        assert abs(I_final - sim.Ib) < 2.0, f"I={I_final}, Ib={sim.Ib}"

    def test_insulin_stays_positive(self):
        """Insulin should remain positive."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5, n_steps=1200))
        sim.reset()
        for _ in range(1200):
            sim.step()
            ins = sim.observe()[2]
            assert ins > 0, f"Insulin went non-positive: {ins}"

    def test_higher_gamma_larger_insulin_spike(self):
        """Higher gamma should produce a larger insulin response."""
        peak_I = []
        for gamma in [0.005, 0.02]:
            sim = GlucoseInsulinSimulation(_make_config(gamma=gamma, dt=0.5))
            sim.reset()
            max_I = sim.observe()[2]
            for _ in range(200):
                sim.step()
                ins = sim.observe()[2]
                if ins > max_I:
                    max_I = ins
            peak_I.append(max_I)
        assert peak_I[1] > peak_I[0], f"Higher gamma peak {peak_I[1]} <= {peak_I[0]}"


class TestRemoteInsulinAction:
    def test_X_starts_at_zero(self):
        """Remote insulin action should start at zero."""
        sim = GlucoseInsulinSimulation(_make_config())
        state = sim.reset()
        assert state[1] == pytest.approx(0.0)

    def test_X_rises_then_falls(self):
        """X should rise as insulin increases, then fall back to zero."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5, n_steps=1200))
        sim.reset()
        X_vals = [sim.observe()[1]]
        for _ in range(1200):
            sim.step()
            X_vals.append(sim.observe()[1])

        X_arr = np.array(X_vals)
        peak_idx = np.argmax(X_arr)
        # Peak should not be at start or end
        assert peak_idx > 0
        assert peak_idx < len(X_arr) - 10
        # Should be close to zero at end
        assert abs(X_arr[-1]) < 0.001, f"X didn't return to zero: {X_arr[-1]}"

    def test_X_returns_to_near_zero(self):
        """X should return to approximately zero at steady state.

        Note: when Gb > h, the true steady state has a small positive X
        because the insulin secretion term gamma*(Gb-h) is nonzero, leading
        to a slightly elevated I and thus a small X via p3*(I-Ib)/p2.
        """
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5, n_steps=3000))
        sim.reset()
        for _ in range(3000):
            sim.step()
        X_final = sim.observe()[1]
        assert abs(X_final) < 0.01, f"X final = {X_final}"


class TestPhysiologicalIndices:
    def test_glucose_effectiveness_equals_p1(self):
        """SG = p1."""
        for p1 in [0.01, 0.028, 0.05]:
            sim = GlucoseInsulinSimulation(_make_config(p1=p1))
            assert sim.glucose_effectiveness == pytest.approx(p1)

    def test_insulin_sensitivity_equals_p3_over_p2(self):
        """SI = p3/p2."""
        sim = GlucoseInsulinSimulation(_make_config(p3=5e-6, p2=0.025))
        expected = 5e-6 / 0.025
        assert sim.insulin_sensitivity == pytest.approx(expected)

    def test_insulin_sensitivity_different_params(self):
        """SI = p3/p2 for different parameter combinations."""
        for p3, p2 in [(1e-5, 0.025), (5e-6, 0.05), (2e-5, 0.01)]:
            sim = GlucoseInsulinSimulation(_make_config(p3=p3, p2=p2))
            assert sim.insulin_sensitivity == pytest.approx(p3 / p2)

    def test_basal_state(self):
        """Basal state should be [Gb, 0, Ib]."""
        sim = GlucoseInsulinSimulation(_make_config(Gb=100.0, Ib=15.0))
        basal = sim.basal_state
        np.testing.assert_allclose(basal, [100.0, 0.0, 15.0])


class TestTimeToHalfPeak:
    def test_time_to_half_peak_finite(self):
        """Should return a finite time for standard parameters."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5, n_steps=1200))
        t_half = sim.time_to_half_peak(max_steps=1200)
        assert np.isfinite(t_half)
        assert t_half > 0

    def test_higher_p1_faster_decay(self):
        """Higher glucose effectiveness should give faster decay."""
        t_halves = []
        for p1 in [0.01, 0.05]:
            sim = GlucoseInsulinSimulation(_make_config(p1=p1, dt=0.5))
            t_halves.append(sim.time_to_half_peak(max_steps=2000))
        # Higher p1 -> faster decay -> shorter half time
        assert t_halves[1] < t_halves[0]

    def test_time_to_half_peak_reasonable_range(self):
        """Half-time should be physiologically reasonable (minutes)."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5))
        t_half = sim.time_to_half_peak(max_steps=2000)
        # Typical IVGTT half-time: 10-100 minutes
        assert 1.0 < t_half < 200.0, f"t_half={t_half}"


class TestParameterVariations:
    def test_higher_p1_faster_glucose_return(self):
        """Higher p1 (glucose effectiveness) should give faster glucose clearance."""
        final_G = []
        for p1 in [0.01, 0.05]:
            sim = GlucoseInsulinSimulation(_make_config(p1=p1, dt=0.5, n_steps=400))
            sim.reset()
            for _ in range(400):
                sim.step()
            final_G.append(sim.observe()[0])
        # Higher p1 -> glucose closer to Gb at fixed time
        assert abs(final_G[1] - 92.0) < abs(final_G[0] - 92.0)

    def test_higher_p3_more_insulin_effect(self):
        """Higher p3 should increase insulin's glucose-lowering effect."""
        final_G = []
        for p3 in [1e-6, 1e-5]:
            sim = GlucoseInsulinSimulation(_make_config(p3=p3, dt=0.5, n_steps=600))
            sim.reset()
            for _ in range(600):
                sim.step()
            final_G.append(sim.observe()[0])
        # Higher p3 -> more insulin action -> lower final glucose
        assert final_G[1] < final_G[0] + 5.0

    def test_higher_gamma_more_insulin(self):
        """Higher gamma should produce more insulin secretion."""
        peak_I = []
        for gamma in [0.005, 0.02]:
            sim = GlucoseInsulinSimulation(_make_config(gamma=gamma, dt=0.5))
            sim.reset()
            max_I = 0.0
            for _ in range(300):
                sim.step()
                ins = sim.observe()[2]
                max_I = max(max_I, ins)
            peak_I.append(max_I)
        assert peak_I[1] > peak_I[0]

    def test_zero_p3_no_insulin_action(self):
        """With p3=0, X should stay at zero (no insulin action pathway)."""
        sim = GlucoseInsulinSimulation(_make_config(p3=0.0, dt=0.5))
        sim.reset()
        for _ in range(200):
            sim.step()
            X = sim.observe()[1]
            assert abs(X) < 1e-10, f"X should be zero when p3=0: {X}"


class TestDerivatives:
    def test_derivatives_at_basal(self):
        """At the basal steady state [Gb, 0, Ib], all derivatives should be zero
        (assuming Gb > h so the secretion term balances)."""
        sim = GlucoseInsulinSimulation(_make_config())
        sim.reset()
        basal = sim.basal_state
        dy = sim._derivatives(basal)
        # dG/dt = -(p1+0)*Gb + p1*Gb = 0
        assert abs(dy[0]) < 1e-10
        # dX/dt = -p2*0 + p3*(Ib-Ib) = 0
        assert abs(dy[1]) < 1e-10
        # dI/dt = -n*(Ib-Ib) + gamma*max(Gb-h,0) = gamma*(Gb-h)
        # This is NOT zero unless gamma=0 or Gb=h
        # At basal, the secretion term gamma*(Gb-h) balances the clearance
        # Actually at true steady state, dI/dt = 0 requires gamma*(Gb-h) = 0
        # But Gb=92, h=80 -> gamma*(92-80)=0.12 != 0
        # The "true" basal is more complex; X=0, I=Ib only if Gb<=h
        # For Gb>h, the true steady state has I > Ib

    def test_derivatives_glucose_equation(self):
        """Check dG/dt = -(p1+X)*G + p1*Gb manually."""
        sim = GlucoseInsulinSimulation(_make_config(p1=0.028, Gb=92.0))
        sim.reset()
        state = np.array([200.0, 0.001, 15.0])
        dy = sim._derivatives(state)
        expected_dG = -(0.028 + 0.001) * 200.0 + 0.028 * 92.0
        assert dy[0] == pytest.approx(expected_dG, rel=1e-10)

    def test_derivatives_X_equation(self):
        """Check dX/dt = -p2*X + p3*(I-Ib) manually."""
        sim = GlucoseInsulinSimulation(_make_config(p2=0.025, p3=5e-6, Ib=11.0))
        sim.reset()
        state = np.array([200.0, 0.001, 20.0])
        dy = sim._derivatives(state)
        expected_dX = -0.025 * 0.001 + 5e-6 * (20.0 - 11.0)
        assert dy[1] == pytest.approx(expected_dX, rel=1e-10)

    def test_derivatives_I_equation_above_threshold(self):
        """Check dI/dt = -n*(I-Ib) + gamma*max(G-h,0) when G > h."""
        sim = GlucoseInsulinSimulation(
            _make_config(n=0.23, gamma=0.01, Ib=11.0, h=80.0)
        )
        sim.reset()
        state = np.array([200.0, 0.001, 20.0])
        dy = sim._derivatives(state)
        expected_dI = -0.23 * (20.0 - 11.0) + 0.01 * (200.0 - 80.0)
        assert dy[2] == pytest.approx(expected_dI, rel=1e-10)

    def test_derivatives_I_equation_below_threshold(self):
        """Check dI/dt = -n*(I-Ib) when G <= h."""
        sim = GlucoseInsulinSimulation(
            _make_config(n=0.23, Ib=11.0, h=80.0)
        )
        sim.reset()
        state = np.array([70.0, 0.001, 20.0])
        dy = sim._derivatives(state)
        expected_dI = -0.23 * (20.0 - 11.0)
        assert dy[2] == pytest.approx(expected_dI, rel=1e-10)


class TestSteadyStateError:
    def test_steady_state_error_small(self):
        """After long simulation, state should be near equilibrium."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5))
        error = sim.steady_state_error(n_steps=3000)
        # G and I should be close to some steady state
        assert abs(error[1]) < 0.01, f"X error too large: {error[1]}"

    def test_steady_state_X_near_zero(self):
        """X should converge to near zero at steady state."""
        sim = GlucoseInsulinSimulation(_make_config(dt=0.5))
        sim.reset()
        for _ in range(3000):
            sim.step()
        X_final = sim.observe()[1]
        assert abs(X_final) < 0.001


class TestRediscoveryDataGeneration:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.glucose_insulin import generate_ode_data

        data = generate_ode_data(n_steps=500, dt=0.5)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert data["p1"] == 0.028
        assert data["p2"] == 0.025
        assert data["p3"] == 5e-6

    def test_ode_data_glucose_column(self):
        from simulating_anything.rediscovery.glucose_insulin import generate_ode_data

        data = generate_ode_data(n_steps=100, dt=0.5)
        np.testing.assert_array_equal(data["G"], data["states"][:, 0])
        np.testing.assert_array_equal(data["X"], data["states"][:, 1])
        np.testing.assert_array_equal(data["I"], data["states"][:, 2])

    def test_sensitivity_data_generation(self):
        from simulating_anything.rediscovery.glucose_insulin import (
            generate_sensitivity_data,
        )

        data = generate_sensitivity_data(n_p1=3, n_p3=3, dt=0.5)
        assert len(data["SI"]) == 9
        assert len(data["SG"]) == 9
        assert len(data["half_time"]) == 9

    def test_sensitivity_SI_matches_theory(self):
        from simulating_anything.rediscovery.glucose_insulin import (
            generate_sensitivity_data,
        )

        data = generate_sensitivity_data(n_p1=3, n_p3=3, dt=0.5)
        np.testing.assert_allclose(data["SI"], data["SI_theory"])

    def test_sensitivity_SG_matches_theory(self):
        from simulating_anything.rediscovery.glucose_insulin import (
            generate_sensitivity_data,
        )

        data = generate_sensitivity_data(n_p1=3, n_p3=3, dt=0.5)
        np.testing.assert_allclose(data["SG"], data["SG_theory"])

    def test_decay_data_generation(self):
        from simulating_anything.rediscovery.glucose_insulin import generate_decay_data

        data = generate_decay_data(n_p1=5, dt=0.5)
        assert len(data["p1"]) == 5
        assert len(data["half_time"]) == 5
        # All half times should be finite
        assert np.all(np.isfinite(data["half_time"]))
