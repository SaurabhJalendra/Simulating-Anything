"""Tests for the Wilson-Cowan neural population model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.wilson_cowan import WilsonCowanSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    I_ext_E: float = 1.5,
    tau_E: float = 1.0,
    tau_I: float = 2.0,
    w_EE: float = 16.0,
    w_EI: float = 12.0,
    w_IE: float = 15.0,
    w_II: float = 3.0,
    dt: float = 0.01,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.WILSON_COWAN,
        dt=dt,
        n_steps=1000,
        parameters={
            "tau_E": tau_E, "tau_I": tau_I,
            "w_EE": w_EE, "w_EI": w_EI,
            "w_IE": w_IE, "w_II": w_II,
            "a": 1.3, "theta": 4.0,
            "I_ext_E": I_ext_E, "I_ext_I": 0.0,
            "E_0": 0.1, "I_0": 0.05,
        },
    )


class TestWilsonCowanSigmoid:
    def test_sigmoid_at_threshold(self):
        """S(theta) should equal 0.5."""
        sim = WilsonCowanSimulation(_make_config())
        sim.reset()
        assert sim.sigmoid(sim.theta) == pytest.approx(0.5)

    def test_sigmoid_large_input(self):
        """S(large) should be close to 1."""
        sim = WilsonCowanSimulation(_make_config())
        sim.reset()
        assert sim.sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_sigmoid_small_input(self):
        """S(small) should be close to 0."""
        sim = WilsonCowanSimulation(_make_config())
        sim.reset()
        assert sim.sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_parameters(self):
        """Different steepness 'a' changes sigmoid shape."""
        sim_steep = WilsonCowanSimulation(
            SimulationConfig(
                domain=Domain.WILSON_COWAN,
                dt=0.01, n_steps=100,
                parameters={"a": 5.0, "theta": 4.0},
            )
        )
        sim_flat = WilsonCowanSimulation(
            SimulationConfig(
                domain=Domain.WILSON_COWAN,
                dt=0.01, n_steps=100,
                parameters={"a": 0.5, "theta": 4.0},
            )
        )
        # Both should be 0.5 at theta
        assert sim_steep.sigmoid(4.0) == pytest.approx(0.5)
        assert sim_flat.sigmoid(4.0) == pytest.approx(0.5)
        # Steep sigmoid should be closer to 1 above threshold
        x = 6.0
        assert sim_steep.sigmoid(x) > sim_flat.sigmoid(x)


class TestWilsonCowanSimulation:
    def test_reset(self):
        """Reset returns valid initial state."""
        sim = WilsonCowanSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)
        np.testing.assert_allclose(state, [0.1, 0.05])

    def test_observe_shape(self):
        """Observe returns 2-element state [E, I]."""
        sim = WilsonCowanSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_step_advances(self):
        """State changes after stepping."""
        sim = WilsonCowanSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same parameters produce same trajectory."""
        config = _make_config()
        sim1 = WilsonCowanSimulation(config)
        sim1.reset()
        for _ in range(200):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = WilsonCowanSimulation(config)
        sim2.reset()
        for _ in range(200):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2, atol=1e-12)

    def test_rk4_stability(self):
        """No NaN or Inf over a long trajectory."""
        sim = WilsonCowanSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_E_I_range(self):
        """E and I should stay roughly in [0, 1] (sigmoid output range)."""
        sim = WilsonCowanSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            exc, inh = sim.observe()
            assert -0.1 <= exc <= 1.1, f"E out of range: {exc}"
            assert -0.1 <= inh <= 1.1, f"I out of range: {inh}"

    def test_oscillation(self):
        """Default parameters (I_ext_E=1.5) should produce E-I oscillation."""
        sim = WilsonCowanSimulation(_make_config(I_ext_E=1.5, dt=0.005))
        sim.reset()
        # Skip transient
        for _ in range(10000):
            sim.step()
        # Measure variation
        E_vals = []
        for _ in range(5000):
            sim.step()
            E_vals.append(sim.observe()[0])
        amplitude = max(E_vals) - min(E_vals)
        assert amplitude > 0.05, f"No oscillation, amplitude={amplitude}"

    def test_steady_state(self):
        """Low I_ext_E should give steady state (small amplitude)."""
        sim = WilsonCowanSimulation(_make_config(I_ext_E=0.0, dt=0.005))
        sim.reset()
        # Run to convergence
        for _ in range(20000):
            sim.step()
        # Measure variation
        E_vals = []
        for _ in range(5000):
            sim.step()
            E_vals.append(sim.observe()[0])
        amplitude = max(E_vals) - min(E_vals)
        assert amplitude < 0.05, f"Should be steady, amplitude={amplitude}"

    def test_inhibition_effect(self):
        """Increasing w_EI should suppress excitatory activity."""
        # Run with normal w_EI
        sim1 = WilsonCowanSimulation(_make_config(w_EI=12.0, dt=0.005))
        sim1.reset()
        for _ in range(10000):
            sim1.step()
        E_vals_1 = []
        for _ in range(3000):
            sim1.step()
            E_vals_1.append(sim1.observe()[0])

        # Run with higher w_EI
        sim2 = WilsonCowanSimulation(_make_config(w_EI=20.0, dt=0.005))
        sim2.reset()
        for _ in range(10000):
            sim2.step()
        E_vals_2 = []
        for _ in range(3000):
            sim2.step()
            E_vals_2.append(sim2.observe()[0])

        # Stronger inhibition should give lower or equal mean E
        assert np.mean(E_vals_2) <= np.mean(E_vals_1) + 0.05

    def test_excitation_effect(self):
        """Increasing w_EE should amplify excitatory activity."""
        # Low w_EE: may not oscillate
        sim1 = WilsonCowanSimulation(_make_config(w_EE=5.0, dt=0.005))
        sim1.reset()
        for _ in range(10000):
            sim1.step()
        E_vals_1 = []
        for _ in range(3000):
            sim1.step()
            E_vals_1.append(sim1.observe()[0])

        # High w_EE
        sim2 = WilsonCowanSimulation(_make_config(w_EE=16.0, dt=0.005))
        sim2.reset()
        for _ in range(10000):
            sim2.step()
        E_vals_2 = []
        for _ in range(3000):
            sim2.step()
            E_vals_2.append(sim2.observe()[0])

        # Stronger excitation should give higher or equal mean E
        assert np.mean(E_vals_2) >= np.mean(E_vals_1) - 0.05

    def test_tau_ratio_changes_frequency(self):
        """Different tau_E/tau_I ratio should change oscillation frequency."""
        # Frequency with tau_E=1, tau_I=2
        sim1 = WilsonCowanSimulation(_make_config(
            tau_E=1.0, tau_I=2.0, dt=0.005
        ))
        spec1 = sim1.frequency_spectrum(n_steps=8000)
        freq1 = spec1["peak_freq"]

        # Frequency with tau_E=0.5, tau_I=2 (faster excitatory)
        sim2 = WilsonCowanSimulation(_make_config(
            tau_E=0.5, tau_I=2.0, dt=0.005
        ))
        spec2 = sim2.frequency_spectrum(n_steps=8000)
        freq2 = spec2["peak_freq"]

        # Frequencies should differ if both oscillate
        if freq1 > 0.1 and freq2 > 0.1:
            assert abs(freq1 - freq2) > 0.01


class TestWilsonCowanAnalysis:
    def test_nullclines(self):
        """Nullcline computation returns valid curves."""
        sim = WilsonCowanSimulation(_make_config())
        sim.reset()
        null_data = sim.nullclines(n_points=50)
        assert "E_null_E" in null_data
        assert "E_null_I" in null_data
        assert "I_null_E" in null_data
        assert "I_null_I" in null_data
        assert len(null_data["E_null_E"]) == 50
        # At least some points should be valid
        assert np.sum(np.isfinite(null_data["I_null_I"])) > 0

    def test_fixed_point_exists(self):
        """Should find at least one fixed point."""
        sim = WilsonCowanSimulation(_make_config())
        sim.reset()
        fps = sim.find_fixed_points()
        assert len(fps) >= 1
        # Fixed point should be in valid range
        for fp in fps:
            assert -0.1 <= fp[0] <= 1.1, f"E fixed point out of range: {fp[0]}"
            assert -0.1 <= fp[1] <= 1.1, f"I fixed point out of range: {fp[1]}"

    def test_eigenvalues(self):
        """Eigenvalues should be computable at fixed point."""
        sim = WilsonCowanSimulation(_make_config(I_ext_E=1.5))
        sim.reset()
        eigs = sim.compute_eigenvalues()
        assert eigs.shape == (2,)
        assert np.all(np.isfinite(eigs))

    def test_eigenvalues_oscillatory(self):
        """Complex eigenvalues indicate oscillatory regime."""
        sim = WilsonCowanSimulation(_make_config(I_ext_E=1.5))
        sim.reset()
        eigs = sim.compute_eigenvalues()
        # If the system oscillates, at least one eigenvalue should have
        # a non-zero imaginary part or positive real part
        has_imag = np.any(np.abs(np.imag(eigs)) > 1e-6)
        has_positive_real = np.any(np.real(eigs) > 0)
        # For oscillatory parameters, typically see complex eigs with + real
        # or a limit cycle driven by the nonlinearity
        assert has_imag or has_positive_real, (
            f"Expected oscillatory eigenvalues, got {eigs}"
        )

    def test_hopf_sweep(self):
        """Hopf sweep should detect transition from steady to oscillatory."""
        sim = WilsonCowanSimulation(_make_config(dt=0.005))
        result = sim.hopf_bifurcation_sweep(
            I_ext_values=np.linspace(0.0, 3.0, 10),
            n_test_steps=3000,
        )
        assert "I_ext" in result
        assert "amplitude" in result
        assert "frequency" in result
        assert len(result["I_ext"]) == 10
        assert len(result["amplitude"]) == 10
        # Should see some amplitude variation
        assert np.max(result["amplitude"]) > np.min(result["amplitude"])

    def test_frequency_spectrum(self):
        """Frequency spectrum should find measurable oscillation frequency."""
        sim = WilsonCowanSimulation(_make_config(I_ext_E=1.5, dt=0.005))
        spec = sim.frequency_spectrum(n_steps=6000)
        assert "freq" in spec
        assert "power" in spec
        assert "peak_freq" in spec
        assert spec["peak_freq"] >= 0

    def test_jacobian_shape(self):
        """Jacobian should be 2x2."""
        sim = WilsonCowanSimulation(_make_config())
        sim.reset()
        J = sim.compute_jacobian(np.array([0.3, 0.2]))
        assert J.shape == (2, 2)
        assert np.all(np.isfinite(J))


class TestWilsonCowanRediscovery:
    def test_ode_data_generation(self):
        from simulating_anything.rediscovery.wilson_cowan import generate_ode_data
        data = generate_ode_data(I_ext_E=1.5, n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 2)
        assert len(data["time"]) == 501
        assert data["I_ext_E"] == 1.5

    def test_hopf_data_generation(self):
        from simulating_anything.rediscovery.wilson_cowan import generate_hopf_data
        data = generate_hopf_data(n_I_ext=5, dt=0.01)
        assert len(data["I_ext"]) == 5
        assert len(data["amplitude"]) == 5

    def test_frequency_data_generation(self):
        from simulating_anything.rediscovery.wilson_cowan import (
            generate_frequency_data,
        )
        data = generate_frequency_data(n_I_ext=5, dt=0.01)
        assert len(data["I_ext"]) == 5
        assert len(data["frequency"]) == 5
        assert np.all(data["frequency"] >= 0)
