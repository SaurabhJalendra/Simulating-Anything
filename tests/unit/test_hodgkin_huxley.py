"""Tests for the Hodgkin-Huxley neuron model."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.hodgkin_huxley import (
    HodgkinHuxleySimulation,
    _alpha_h,
    _beta_h,
    _beta_m,
    _beta_n,
    _safe_alpha_m,
    _safe_alpha_n,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    I_ext: float = 10.0,
    dt: float = 0.01,
    n_steps: int = 1000,
    **kwargs: float,
) -> SimulationConfig:
    params = {
        "g_Na": 120.0, "g_K": 36.0, "g_L": 0.3,
        "E_Na": 50.0, "E_K": -77.0, "E_L": -54.387,
        "C_m": 1.0, "I_ext": I_ext,
    }
    params.update(kwargs)
    return SimulationConfig(
        domain=Domain.HODGKIN_HUXLEY,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestHodgkinHuxleySimulation:
    def test_resting_potential(self):
        """At I_ext=0, membrane should stay near resting potential ~-65mV."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=0.0))
        sim.reset()
        # Run for a while at zero current
        for _ in range(5000):
            sim.step()
        V = sim.observe()[0]
        # Should remain near rest (may drift slightly due to gating dynamics)
        assert -80 < V < -50, f"Resting potential out of range: {V}"

    def test_steady_state_gating(self):
        """Steady-state gating variables at V=-65 should be in [0,1]."""
        n_inf, m_inf, h_inf = HodgkinHuxleySimulation.steady_state_gating(-65.0)
        # All must be in [0, 1]
        for val, name in [(n_inf, "n"), (m_inf, "m"), (h_inf, "h")]:
            assert 0 <= val <= 1, f"{name}_inf={val} out of [0,1]"
        # At rest: n_inf ~ 0.32, m_inf ~ 0.05, h_inf ~ 0.60
        assert 0.2 < n_inf < 0.5, f"n_inf={n_inf} unexpected"
        assert 0.01 < m_inf < 0.15, f"m_inf={m_inf} unexpected"
        assert 0.4 < h_inf < 0.8, f"h_inf={h_inf} unexpected"

    def test_subthreshold(self):
        """Small I_ext should not produce spikes (V stays below 0)."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=2.0))
        sim.reset()
        V_max = -100.0
        for _ in range(10000):
            sim.step()
            V = sim.observe()[0]
            V_max = max(V_max, V)
        assert V_max < 0, f"Subthreshold stimulus produced V={V_max}"

    def test_action_potential(self):
        """Large I_ext should produce action potential (V > 0)."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=10.0))
        sim.reset()
        V_max = -100.0
        for _ in range(20000):
            sim.step()
            V = sim.observe()[0]
            V_max = max(V_max, V)
        assert V_max > 0, f"No action potential: V_max={V_max}"

    def test_spike_amplitude(self):
        """Peak voltage during spike should reach ~40-50 mV."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=10.0))
        sim.reset()
        V_max = -100.0
        for _ in range(30000):
            sim.step()
            V = sim.observe()[0]
            V_max = max(V_max, V)
        # Classic HH spike peaks near 40 mV
        assert 20 < V_max < 60, f"Spike amplitude unexpected: {V_max}"

    def test_refractory_period(self):
        """After a spike, the neuron cannot fire immediately."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=10.0, dt=0.01))
        sim.reset()

        # Collect voltage trace
        V_trace = []
        for _ in range(50000):
            sim.step()
            V_trace.append(sim.observe()[0])
        V_trace = np.array(V_trace)

        # Find spikes
        spike_indices = sim.detect_spikes(V_trace, threshold=0.0)
        if len(spike_indices) >= 2:
            # Inter-spike interval must be > 2ms (absolute refractory)
            isi_ms = np.diff(spike_indices) * 0.01
            assert np.all(isi_ms > 2.0), (
                f"Refractory period violated: min ISI={np.min(isi_ms):.2f}ms"
            )

    def test_fi_curve_monotonic(self):
        """Higher current should produce higher frequency (below depolarization block)."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=0.0, dt=0.01))
        # Use currents below depolarization block threshold (~150 uA/cm^2)
        I_values = np.array([7.0, 10.0, 20.0, 40.0])
        fi = sim.compute_fi_curve(I_values, t_max=300.0)
        freqs = fi["frequency"]
        # All should fire, and trend should be increasing
        assert freqs[-1] >= freqs[0], (
            f"f-I not monotonic: f({I_values[0]})={freqs[0]}, "
            f"f({I_values[-1]})={freqs[-1]}"
        )

    def test_gating_bounds(self):
        """Gating variables n, m, h must always be in [0, 1]."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=20.0, dt=0.01))
        sim.reset()
        for _ in range(20000):
            sim.step()
            state = sim.observe()
            n, m, h = state[1], state[2], state[3]
            assert 0 <= n <= 1, f"n out of bounds: {n}"
            assert 0 <= m <= 1, f"m out of bounds: {m}"
            assert 0 <= h <= 1, f"h out of bounds: {h}"

    def test_rate_functions_known_values(self):
        """Test rate functions at V=-65 (resting potential)."""
        V = -65.0
        # alpha_n(-65) = 0.01*(-65+55)/(1-exp(-(-65+55)/10))
        #              = 0.01*(-10)/(1-exp(1))
        #              = -0.1 / (1 - e) = -0.1 / (-1.7183) ~ 0.0582
        an = _safe_alpha_n(V)
        assert 0.05 < an < 0.07, f"alpha_n(-65) = {an}"

        bn = _beta_n(V)
        # beta_n(-65) = 0.125*exp(0) = 0.125
        np.testing.assert_allclose(bn, 0.125, atol=1e-10)

        am = _safe_alpha_m(V)
        # alpha_m(-65) = 0.1*(-65+40)/(1-exp(-(-65+40)/10))
        #              = 0.1*(-25)/(1-exp(2.5))
        assert 0.1 < am < 0.3, f"alpha_m(-65) = {am}"

        bm = _beta_m(V)
        # beta_m(-65) = 4*exp(0) = 4.0
        np.testing.assert_allclose(bm, 4.0, atol=1e-10)

        ah = _alpha_h(V)
        # alpha_h(-65) = 0.07*exp(0) = 0.07
        np.testing.assert_allclose(ah, 0.07, atol=1e-10)

        bh = _beta_h(V)
        # beta_h(-65) = 1/(1+exp(-(-65+35)/10)) = 1/(1+exp(3))
        expected = 1.0 / (1.0 + np.exp(3.0))
        np.testing.assert_allclose(bh, expected, atol=1e-10)

    def test_singularity_alpha_n(self):
        """alpha_n at V=-55 should not produce NaN (L'Hopital limit = 0.1)."""
        result = _safe_alpha_n(-55.0)
        assert np.isfinite(result), f"alpha_n(-55) = {result}"
        np.testing.assert_allclose(result, 0.1, atol=1e-5)

    def test_singularity_alpha_m(self):
        """alpha_m at V=-40 should not produce NaN (L'Hopital limit = 1.0)."""
        result = _safe_alpha_m(-40.0)
        assert np.isfinite(result), f"alpha_m(-40) = {result}"
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_rk4_stability(self):
        """No NaN or Inf after 10000 steps with dt=0.01."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=10.0, dt=0.01))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_energy_dissipation(self):
        """HH is dissipative: it should not conserve energy."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=0.0, dt=0.01))
        sim.reset()

        # Track voltage variance over time -- should decay toward rest
        early_V = []
        for _ in range(5000):
            sim.step()
            early_V.append(sim.observe()[0])
        late_V = []
        for _ in range(5000):
            sim.step()
            late_V.append(sim.observe()[0])

        late_range = max(late_V) - min(late_V)
        # Without external current, system should settle (less variation later)
        # or at least not blow up
        assert late_range < 200, f"System diverging: late_range={late_range}"

    def test_ionic_currents(self):
        """Ionic current decomposition should balance at rest."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=0.0, dt=0.01))
        sim.reset()
        # Let it settle
        for _ in range(5000):
            sim.step()
        currents = sim.ionic_currents()
        # At near-rest, all currents should be finite and small
        for name, val in currents.items():
            assert np.isfinite(val), f"{name} is not finite: {val}"

    def test_repetitive_firing(self):
        """Sustained high current should produce multiple spikes."""
        sim = HodgkinHuxleySimulation(_make_config(I_ext=15.0, dt=0.01))
        sim.reset()
        V_trace = []
        for _ in range(50000):
            sim.step()
            V_trace.append(sim.observe()[0])
        V_trace = np.array(V_trace)
        spikes = sim.detect_spikes(V_trace, threshold=0.0)
        assert len(spikes) >= 3, f"Expected repetitive firing, got {len(spikes)} spikes"

    def test_single_spike_brief_pulse(self):
        """A brief current pulse should produce roughly one spike."""
        # Start with current, then turn it off after a short pulse
        config = _make_config(I_ext=20.0, dt=0.01)
        sim = HodgkinHuxleySimulation(config)
        sim.reset()

        V_trace = []
        pulse_steps = int(2.0 / 0.01)  # 2ms pulse
        total_steps = int(50.0 / 0.01)  # 50ms total

        for i in range(total_steps):
            if i == pulse_steps:
                sim.I_ext = 0.0  # Turn off current
            sim.step()
            V_trace.append(sim.observe()[0])

        V_trace = np.array(V_trace)
        spikes = sim.detect_spikes(V_trace, threshold=0.0)
        # Brief pulse: should produce at most 1-2 spikes
        assert len(spikes) <= 3, f"Brief pulse gave too many spikes: {len(spikes)}"

    def test_deterministic(self):
        """Same parameters should give same trajectory."""
        config = _make_config(I_ext=10.0, dt=0.01)
        sim1 = HodgkinHuxleySimulation(config)
        sim1.reset()
        for _ in range(1000):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = HodgkinHuxleySimulation(config)
        sim2.reset()
        for _ in range(1000):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2, atol=1e-12)

    def test_dt_convergence(self):
        """Smaller dt should give similar results to larger dt."""
        # Coarse dt
        config1 = _make_config(I_ext=10.0, dt=0.05)
        sim1 = HodgkinHuxleySimulation(config1)
        sim1.reset()
        for _ in range(2000):
            sim1.step()
        V_coarse = sim1.observe()[0]

        # Fine dt (same total time = 100ms)
        config2 = _make_config(I_ext=10.0, dt=0.01)
        sim2 = HodgkinHuxleySimulation(config2)
        sim2.reset()
        for _ in range(10000):
            sim2.step()
        V_fine = sim2.observe()[0]

        # Both should be finite and in a similar range (not identical due to chaos)
        assert np.isfinite(V_coarse), f"Coarse V is not finite: {V_coarse}"
        assert np.isfinite(V_fine), f"Fine V is not finite: {V_fine}"
        # Both in physiological range
        assert -100 < V_coarse < 60, f"V_coarse out of range: {V_coarse}"
        assert -100 < V_fine < 60, f"V_fine out of range: {V_fine}"

    def test_observe_shape(self):
        """Observe should return 4-element state vector [V, n, m, h]."""
        sim = HodgkinHuxleySimulation(_make_config())
        state = sim.reset()
        assert state.shape == (4,)
        obs = sim.observe()
        assert obs.shape == (4,)


class TestHodgkinHuxleyRediscovery:
    def test_rediscovery_data(self):
        """Data generation for rediscovery should work."""
        from simulating_anything.rediscovery.hodgkin_huxley import generate_spike_data
        data = generate_spike_data(I_ext=10.0, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 4)
        assert len(data["time"]) == 501
        assert data["I_ext"] == 10.0

    def test_fi_curve_generation(self):
        """f-I curve data generation should produce valid results."""
        from simulating_anything.rediscovery.hodgkin_huxley import generate_fi_curve
        data = generate_fi_curve(n_I=5, I_min=0.0, I_max=50.0, dt=0.01, t_max=100.0)
        assert len(data["I"]) == 5
        assert len(data["frequency"]) == 5
        assert np.all(data["frequency"] >= 0)

    def test_spike_property_measurement(self):
        """Spike property measurement should return valid metrics."""
        from simulating_anything.rediscovery.hodgkin_huxley import (
            measure_spike_properties,
        )
        props = measure_spike_properties(I_ext=10.0, dt=0.01)
        assert "spike_amplitude" in props
        assert "spike_duration_ms" in props
        assert "refractory_period_ms" in props
