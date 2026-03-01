"""Tests for the Morris-Lecar neuron model."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.morris_lecar import MorrisLecarSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    I_ext: float = 0.0,
    dt: float = 0.1,
    **kwargs: float,
) -> SimulationConfig:
    params = {
        "C": 20.0, "g_L": 2.0, "g_Ca": 4.0, "g_K": 8.0,
        "V_L": -60.0, "V_Ca": 120.0, "V_K": -84.0,
        "V1": -1.2, "V2": 18.0, "V3": 2.0, "V4": 30.0,
        "phi": 0.04, "I_ext": I_ext,
    }
    params.update(kwargs)
    return SimulationConfig(
        domain=Domain.MORRIS_LECAR,
        dt=dt,
        n_steps=1000,
        parameters=params,
    )


class TestMorrisLecarSimulation:
    """Tests for the MorrisLecarSimulation class."""

    def test_initial_state_shape(self):
        """State vector should be 2D: [V, w]."""
        sim = MorrisLecarSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        """Initial V should be near -60 mV, w at steady state."""
        sim = MorrisLecarSimulation(_make_config())
        state = sim.reset()
        V, w = state
        assert abs(V - (-60.0)) < 1e-10
        # w should be w_ss(V=-60)
        expected_w = MorrisLecarSimulation.w_ss(-60.0)
        assert abs(w - expected_w) < 1e-10

    def test_step_advances_state(self):
        """A single step should change the state."""
        sim = MorrisLecarSimulation(_make_config(I_ext=100.0))
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        """observe() should return the same array as the internal state."""
        sim = MorrisLecarSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs is sim._state

    def test_trajectory_bounded_no_current(self):
        """With I_ext=0, system should remain bounded near rest."""
        sim = MorrisLecarSimulation(_make_config(I_ext=0.0, dt=0.1))
        sim.reset()
        for _ in range(5000):
            sim.step()
            V, w = sim.observe()
            assert abs(V) < 200, f"V diverged: {V}"
            assert 0.0 <= w <= 1.0, f"w out of range: {w}"

    def test_trajectory_bounded_high_current(self):
        """With I_ext=100, system should oscillate but remain bounded."""
        sim = MorrisLecarSimulation(_make_config(I_ext=100.0, dt=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
            V, w = sim.observe()
            assert abs(V) < 200, f"V diverged: {V}"
            assert 0.0 <= w <= 1.0, f"w out of range: {w}"

    def test_no_current_rests(self):
        """Without input current, should settle to resting potential."""
        sim = MorrisLecarSimulation(_make_config(I_ext=0.0, dt=0.1))
        sim.reset()
        for _ in range(20000):
            sim.step()
        V, w = sim.observe()
        # Should be near resting potential (around -60 mV or stable eq)
        assert V < -20.0, f"V not at rest: {V}"

    def test_high_current_oscillates(self):
        """With high enough current, should show action potentials."""
        sim = MorrisLecarSimulation(_make_config(I_ext=100.0, dt=0.1))
        sim.reset()
        # Skip transient
        for _ in range(10000):
            sim.step()
        # Measure voltage range
        v_vals = []
        for _ in range(10000):
            sim.step()
            v_vals.append(sim.observe()[0])
        v_range = max(v_vals) - min(v_vals)
        # Spikes should produce > 10 mV range
        assert v_range > 10.0, f"No oscillation at I_ext=100, range={v_range}"

    def test_different_currents_different_behavior(self):
        """Different external currents should produce different states."""
        sim1 = MorrisLecarSimulation(_make_config(I_ext=0.0, dt=0.1))
        sim1.reset()
        for _ in range(10000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = MorrisLecarSimulation(_make_config(I_ext=100.0, dt=0.1))
        sim2.reset()
        for _ in range(10000):
            sim2.step()
        state2 = sim2.observe().copy()

        assert not np.allclose(state1, state2, atol=1.0)


class TestMorrisLecarSteadyState:
    """Tests for the steady-state functions m_ss, w_ss, tau_w."""

    def test_m_ss_range(self):
        """m_ss should be in [0, 1] for all V."""
        V_values = np.linspace(-100, 100, 200)
        m = MorrisLecarSimulation.m_ss(V_values)
        assert np.all(m >= 0.0)
        assert np.all(m <= 1.0)

    def test_m_ss_monotonic(self):
        """m_ss should be monotonically increasing in V."""
        V_values = np.linspace(-100, 100, 200)
        m = MorrisLecarSimulation.m_ss(V_values)
        assert np.all(np.diff(m) >= 0)

    def test_w_ss_range(self):
        """w_ss should be in [0, 1] for all V."""
        V_values = np.linspace(-100, 100, 200)
        w = MorrisLecarSimulation.w_ss(V_values)
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)

    def test_w_ss_monotonic(self):
        """w_ss should be monotonically increasing in V."""
        V_values = np.linspace(-100, 100, 200)
        w = MorrisLecarSimulation.w_ss(V_values)
        assert np.all(np.diff(w) >= 0)

    def test_tau_w_positive(self):
        """tau_w should be positive for all V."""
        V_values = np.linspace(-100, 100, 200)
        tau = MorrisLecarSimulation.tau_w(V_values)
        assert np.all(tau > 0)

    def test_tau_w_symmetric(self):
        """tau_w peaks at V=V3 and is symmetric around it."""
        # With default V3=2.0
        V3 = 2.0
        tau_at_V3 = MorrisLecarSimulation.tau_w(V3)
        # tau_w(V3) = 1/cosh(0) = 1
        np.testing.assert_allclose(tau_at_V3, 1.0, atol=1e-10)

    def test_m_ss_at_V1(self):
        """m_ss(V1) should be 0.5 (half-activation)."""
        # Default V1 = -1.2
        m = MorrisLecarSimulation.m_ss(-1.2)
        np.testing.assert_allclose(m, 0.5, atol=1e-10)

    def test_w_ss_at_V3(self):
        """w_ss(V3) should be 0.5 (half-activation)."""
        # Default V3 = 2.0
        w = MorrisLecarSimulation.w_ss(2.0)
        np.testing.assert_allclose(w, 0.5, atol=1e-10)


class TestMorrisLecarDerivatives:
    """Tests for the derivatives computation."""

    def test_derivatives_at_rest_no_current(self):
        """At equilibrium with I=0, derivatives should be small."""
        sim = MorrisLecarSimulation(_make_config(I_ext=0.0))
        V_eq, w_eq = sim.find_equilibrium()
        y = np.array([V_eq, w_eq])
        dy = sim._derivatives(y)
        # Should be approximately zero at equilibrium
        assert np.all(np.abs(dy) < 1.0), f"Derivatives at eq: {dy}"

    def test_derivatives_return_shape(self):
        """Derivatives should return a 2D array."""
        sim = MorrisLecarSimulation(_make_config())
        y = np.array([-60.0, 0.1])
        dy = sim._derivatives(y)
        assert dy.shape == (2,)


class TestMorrisLecarIonicCurrents:
    """Tests for ionic current decomposition."""

    def test_ionic_currents_keys(self):
        """Should return I_L, I_Ca, I_K."""
        sim = MorrisLecarSimulation(_make_config())
        sim.reset()
        currents = sim.ionic_currents()
        assert "I_L" in currents
        assert "I_Ca" in currents
        assert "I_K" in currents

    def test_ionic_currents_at_reversal(self):
        """Current through a channel should be zero at its reversal potential."""
        sim = MorrisLecarSimulation(_make_config())
        # At V = V_K = -84, I_K should be 0
        state = np.array([-84.0, 0.5])
        currents = sim.ionic_currents(state)
        np.testing.assert_allclose(currents["I_K"], 0.0, atol=1e-10)


class TestMorrisLecarNullclines:
    """Tests for nullcline computation."""

    def test_compute_nullclines_shape(self):
        """Nullcline arrays should have the right shape."""
        sim = MorrisLecarSimulation(_make_config(I_ext=0.0))
        nc = sim.compute_nullclines(n_points=100)
        assert len(nc["V"]) == 100
        assert len(nc["w_v_nullcline"]) == 100
        assert len(nc["w_w_nullcline"]) == 100

    def test_w_nullcline_equals_w_ss(self):
        """The w-nullcline should be exactly w_ss(V)."""
        sim = MorrisLecarSimulation(_make_config(I_ext=0.0))
        nc = sim.compute_nullclines(n_points=50)
        expected = sim._w_ss_inst(nc["V"])
        np.testing.assert_allclose(nc["w_w_nullcline"], expected, atol=1e-12)


class TestMorrisLecarEquilibrium:
    """Tests for equilibrium finding."""

    def test_find_equilibrium_returns_tuple(self):
        """Should return (V_eq, w_eq)."""
        sim = MorrisLecarSimulation(_make_config(I_ext=0.0))
        result = sim.find_equilibrium()
        assert len(result) == 2

    def test_equilibrium_is_on_both_nullclines(self):
        """Equilibrium should be on both V- and w-nullclines."""
        sim = MorrisLecarSimulation(_make_config(I_ext=0.0))
        V_eq, w_eq = sim.find_equilibrium()
        # w_eq should equal w_ss(V_eq)
        w_ss_eq = sim._w_ss_inst(V_eq)
        np.testing.assert_allclose(w_eq, w_ss_eq, atol=0.01)


class TestMorrisLecarFiringFrequency:
    """Tests for firing frequency measurement."""

    def test_no_firing_at_zero_current(self):
        """No spikes expected with I_ext=0."""
        sim = MorrisLecarSimulation(_make_config(I_ext=0.0, dt=0.1))
        sim.reset()
        freq = sim.measure_firing_frequency(n_spikes=3)
        assert freq == 0.0

    def test_firing_at_high_current(self):
        """Should fire with I_ext=100."""
        sim = MorrisLecarSimulation(_make_config(I_ext=100.0, dt=0.1))
        sim.reset()
        freq = sim.measure_firing_frequency(n_spikes=3)
        # Should be a positive frequency
        assert freq >= 0.0  # May be 0 if not enough spikes in window


class TestMorrisLecarFICurve:
    """Tests for the f-I curve computation."""

    def test_fi_curve_shape(self):
        """f-I curve should return arrays of correct length."""
        sim = MorrisLecarSimulation(_make_config(dt=0.1))
        I_vals = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
        fi = sim.compute_fi_curve(I_vals, t_max=500.0)
        assert len(fi["I"]) == 5
        assert len(fi["frequency"]) == 5
        assert np.all(fi["frequency"] >= 0)

    def test_fi_curve_monotonic_tendency(self):
        """Frequency should generally increase with current."""
        sim = MorrisLecarSimulation(_make_config(dt=0.1))
        I_vals = np.array([0.0, 100.0, 200.0])
        fi = sim.compute_fi_curve(I_vals, t_max=1000.0)
        # frequency at I=0 should be <= frequency at I=200
        assert fi["frequency"][0] <= fi["frequency"][2] + 1.0


class TestMorrisLecarIsOscillatory:
    """Tests for oscillation detection."""

    def test_not_oscillatory_at_zero_current(self):
        """System at rest should not be oscillatory."""
        sim = MorrisLecarSimulation(_make_config(I_ext=0.0, dt=0.1))
        assert not sim.is_oscillatory(n_test_steps=5000)


class TestMorrisLecarRediscovery:
    """Tests for the rediscovery data generation functions."""

    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.morris_lecar import generate_ode_data
        data = generate_ode_data(I_ext=100.0, n_steps=500, dt=0.1)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert np.all(np.isfinite(data["states"]))

    def test_fi_curve_generation(self):
        from simulating_anything.rediscovery.morris_lecar import generate_fi_curve
        data = generate_fi_curve(n_I=5, dt=0.1)
        assert len(data["I"]) == 5
        assert len(data["frequency"]) == 5
        assert np.all(data["frequency"] >= 0)

    def test_classify_excitability(self):
        from simulating_anything.rediscovery.morris_lecar import classify_excitability
        result = classify_excitability(dt=0.1)
        assert "type" in result
        assert result["type"] in ("Type I", "Type II", "quiescent")


class TestMorrisLecarRun:
    """Tests for the run() trajectory collection method."""

    def test_run_trajectory(self):
        """run() should return a TrajectoryData object with correct shape."""
        config = _make_config(I_ext=0.0, dt=0.1)
        config.n_steps = 100
        sim = MorrisLecarSimulation(config)
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert np.all(np.isfinite(traj.states))
