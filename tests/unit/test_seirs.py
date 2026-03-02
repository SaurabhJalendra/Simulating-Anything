"""Tests for the SEIRS (Susceptible-Exposed-Infectious-Recovered-Susceptible) epidemic model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.seirs import SEIRSSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.1,
    n_steps: int = 1000,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for SEIRS with optional overrides."""
    params: dict[str, float] = {
        "beta": 0.5,
        "sigma": 0.2,
        "gamma": 0.1,
        "xi": 0.05,
        "N0": 1000.0,
        "S_0": 990.0,
        "E_0": 5.0,
        "I_0": 5.0,
        "R_0_init": 0.0,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.SEIRS,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestSEIRSCreation:
    """Tests for simulation creation and parameter initialization."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = SEIRSSimulation(_make_config())
        assert sim.beta == 0.5
        assert sim.sigma == 0.2
        assert sim.gamma == 0.1
        assert sim.xi == 0.05
        assert sim.N0 == 1000.0

    def test_initial_state_shape(self):
        """State vector should be [S, E, I, R] with 4 components."""
        sim = SEIRSSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (4,)

    def test_initial_state_values(self):
        """Initial conditions should match parameters."""
        sim = SEIRSSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [990.0, 5.0, 5.0, 0.0])

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = SEIRSSimulation(_make_config(beta=0.6, gamma=0.2, xi=0.1))
        assert sim.beta == 0.6
        assert sim.gamma == 0.2
        assert sim.xi == 0.1

    def test_custom_initial_conditions(self):
        """Custom initial conditions should be applied."""
        sim = SEIRSSimulation(_make_config(
            S_0=800.0, E_0=50.0, I_0=50.0, R_0_init=100.0,
        ))
        state = sim.reset()
        assert state.shape == (4,)
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)


class TestSEIRSBasicSimulation:
    """Core simulation mechanics tests."""

    def test_step_advances(self):
        """State should change after a step."""
        sim = SEIRSSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_state(self):
        """Observe should return the current state."""
        sim = SEIRSSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_run_method(self):
        """run() should produce a TrajectoryData object."""
        sim = SEIRSSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)
        assert len(traj.timestamps) == 101

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = SEIRSSimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = SEIRSSimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_long_run_stability(self):
        """No NaN after many steps."""
        sim = SEIRSSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"

    def test_multiple_resets(self):
        """Multiple resets should produce identical initial states."""
        sim = SEIRSSimulation(_make_config())
        state1 = sim.reset().copy()
        for _ in range(50):
            sim.step()
        state2 = sim.reset().copy()
        np.testing.assert_allclose(state1, state2)


class TestSEIRSConservation:
    """Conservation law tests: S + E + I + R = N at all times."""

    def test_initial_conservation(self):
        """S + E + I + R should equal N0 at initialization."""
        sim = SEIRSSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)

    def test_conservation_throughout(self):
        """S + E + I + R = N0 should hold at all times."""
        sim = SEIRSSimulation(_make_config())
        sim.reset()
        for _ in range(500):
            state = sim.step()
            np.testing.assert_allclose(
                state.sum(), 1000.0, atol=1e-6,
                err_msg="Population conservation violated",
            )

    def test_conservation_long_run(self):
        """Conservation should hold over long simulation."""
        sim = SEIRSSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            state = sim.step()
        np.testing.assert_allclose(
            state.sum(), 1000.0, atol=1e-4,
            err_msg="Conservation violated over long run",
        )

    def test_conservation_various_parameters(self):
        """Conservation should hold across different parameter sets."""
        for beta, gamma, xi in [(0.3, 0.05, 0.01), (0.8, 0.3, 0.2), (0.2, 0.1, 0.1)]:
            sim = SEIRSSimulation(_make_config(beta=beta, gamma=gamma, xi=xi))
            sim.reset()
            for _ in range(500):
                state = sim.step()
            np.testing.assert_allclose(
                state.sum(), 1000.0, atol=1e-5,
                err_msg=f"Conservation violated for beta={beta}, gamma={gamma}, xi={xi}",
            )


class TestSEIRSNonNegativity:
    """All compartments must remain non-negative."""

    def test_non_negative_all_compartments(self):
        """S, E, I, R should all remain >= 0."""
        sim = SEIRSSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"S became negative: {state[0]}"
            assert state[1] >= 0, f"E became negative: {state[1]}"
            assert state[2] >= 0, f"I became negative: {state[2]}"
            assert state[3] >= 0, f"R became negative: {state[3]}"

    def test_non_negative_fast_dynamics(self):
        """Non-negativity should hold with fast waning immunity."""
        sim = SEIRSSimulation(_make_config(xi=0.5))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"


class TestSEIRSR0:
    """Tests for the basic reproduction number R0 = beta / gamma."""

    def test_r0_formula(self):
        """R0 should equal beta / gamma."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1))
        assert sim.R0 == pytest.approx(5.0)

    def test_r0_alias(self):
        """basic_reproduction_number should equal R0."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1))
        assert sim.basic_reproduction_number == sim.R0

    def test_r0_supercritical(self):
        """R0 > 1 with standard parameters."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1))
        assert sim.R0 > 1.0

    def test_r0_subcritical(self):
        """R0 < 1 when gamma > beta."""
        sim = SEIRSSimulation(_make_config(beta=0.05, gamma=0.1))
        assert sim.R0 < 1.0

    def test_r0_different_params(self):
        """R0 should vary correctly with different parameters."""
        sim = SEIRSSimulation(_make_config(beta=0.6, gamma=0.2))
        assert sim.R0 == pytest.approx(3.0)


class TestSEIRSEpidemicBehavior:
    """Tests for epidemic dynamics above and below threshold."""

    def test_epidemic_occurs_above_threshold(self):
        """With R0 > 1, infected should rise then fall toward endemic state."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1))
        sim.reset()
        max_I = 0
        for _ in range(10000):
            state = sim.step()
            max_I = max(max_I, state[2])
        # Significant epidemic peak
        assert max_I > 50.0, f"Peak I too low: {max_I}"

    def test_no_epidemic_below_threshold(self):
        """With R0 < 1, infected should decay."""
        sim = SEIRSSimulation(
            _make_config(beta=0.05, gamma=0.1, E_0=0.0)
        )
        sim.reset()
        initial_I = sim.observe()[2]
        for _ in range(1000):
            state = sim.step()
        assert state[2] < initial_I, (
            f"I should decay when R0 < 1: I={state[2]}, I_0={initial_I}"
        )

    def test_endemic_state_maintained(self):
        """With R0 > 1 and waning immunity, I should remain nonzero long-term."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1, xi=0.05))
        sim.reset()
        # Run long enough for endemic state to establish
        for _ in range(50000):
            sim.step()
        final_I = sim.observe()[2]
        # I should be nonzero at endemic equilibrium (unlike SEIR where I->0)
        assert final_I > 0.1, (
            f"I should be positive at endemic equilibrium: I={final_I}"
        )

    def test_waning_immunity_sustains_infection(self):
        """Without waning immunity (xi=0), I eventually reaches 0. With xi > 0, it persists."""
        # With xi=0: standard SEIR, disease burns out
        sim_no_wane = SEIRSSimulation(_make_config(
            beta=0.5, gamma=0.1, xi=0.0, E_0=0.0,
        ))
        sim_no_wane.reset()
        for _ in range(20000):
            sim_no_wane.step()
        I_no_wane = sim_no_wane.observe()[2]

        # With xi=0.05: SEIRS, endemic state
        sim_wane = SEIRSSimulation(_make_config(
            beta=0.5, gamma=0.1, xi=0.05, E_0=0.0,
        ))
        sim_wane.reset()
        for _ in range(20000):
            sim_wane.step()
        I_wane = sim_wane.observe()[2]

        assert I_wane > I_no_wane, (
            f"Waning immunity should sustain higher I: {I_wane} vs {I_no_wane}"
        )

    def test_exposed_precedes_infectious(self):
        """Exposed should peak before infectious (latent delay)."""
        sim = SEIRSSimulation(
            _make_config(beta=0.5, sigma=0.1, gamma=0.1, E_0=0.0)
        )
        sim.reset()
        peak_E_time = 0
        peak_I_time = 0
        peak_E = 0.0
        peak_I = 0.0
        for step in range(10000):
            state = sim.step()
            if state[1] > peak_E:
                peak_E = state[1]
                peak_E_time = step
            if state[2] > peak_I:
                peak_I = state[2]
                peak_I_time = step
        assert peak_E_time < peak_I_time, (
            f"Exposed peak ({peak_E_time}) should precede "
            f"infectious peak ({peak_I_time})"
        )


class TestSEIRSDampedOscillations:
    """Tests for damped oscillations in I(t) toward endemic state."""

    def test_oscillations_detected(self):
        """I(t) should exhibit damped oscillations toward endemic state."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1, xi=0.05))
        sim.reset()
        I_history = []
        for _ in range(10000):
            state = sim.step()
            I_history.append(state[2])

        I_arr = np.array(I_history)
        # Look for peaks (local maxima) with I > endemic/2 across full trajectory
        # The oscillations damp quickly, so look across the whole run
        peaks = []
        for j in range(1, len(I_arr) - 1):
            if I_arr[j] > I_arr[j - 1] and I_arr[j] > I_arr[j + 1]:
                if I_arr[j] > 10.0:  # Only count significant peaks
                    peaks.append(j)

        # Should have at least 2 oscillation peaks (initial overshoot + at least one more)
        assert len(peaks) >= 2, (
            f"Expected >= 2 oscillation peaks, found {len(peaks)}"
        )

    def test_oscillation_amplitude_decreases(self):
        """Oscillation amplitude should decrease over time (damping)."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1, xi=0.02))
        sim.reset()
        I_history = []
        for _ in range(50000):
            state = sim.step()
            I_history.append(state[2])

        I_arr = np.array(I_history)
        # Find first and second peak amplitudes (after initial peak)
        # Use the endemic equilibrium as baseline
        eq = sim.endemic_equilibrium()
        if eq is not None:
            I_eq = eq[2]
            # Peaks above equilibrium in first quarter vs last quarter
            first_q = I_arr[:len(I_arr) // 4]
            last_q = I_arr[3 * len(I_arr) // 4:]
            first_max_dev = np.max(np.abs(first_q - I_eq))
            last_max_dev = np.max(np.abs(last_q - I_eq))
            assert last_max_dev < first_max_dev, (
                f"Oscillations should be damped: first deviation {first_max_dev:.2f}, "
                f"last deviation {last_max_dev:.2f}"
            )


class TestSEIRSEndemicEquilibrium:
    """Tests for endemic equilibrium computation."""

    def test_endemic_exists_above_threshold(self):
        """Endemic equilibrium should exist when R0 > 1."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1))
        eq = sim.endemic_equilibrium()
        assert eq is not None, "Endemic equilibrium should exist for R0 = 5"

    def test_endemic_none_below_threshold(self):
        """Endemic equilibrium should be None when R0 < 1."""
        sim = SEIRSSimulation(_make_config(beta=0.05, gamma=0.1))
        eq = sim.endemic_equilibrium()
        assert eq is None, "No endemic equilibrium when R0 < 1"

    def test_endemic_conservation(self):
        """S* + E* + I* + R* should equal N0."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1, xi=0.05))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        np.testing.assert_allclose(eq.sum(), 1000.0, atol=1e-6)

    def test_endemic_all_positive(self):
        """All compartments should be positive at endemic equilibrium."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1, xi=0.05))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        assert np.all(eq > 0), f"All compartments should be positive: {eq}"

    def test_endemic_s_star_equals_n_over_r0(self):
        """At endemic equilibrium, S* = N / R0."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        expected_S = 1000.0 / sim.R0
        assert eq[0] == pytest.approx(expected_S, rel=1e-6)

    def test_simulation_converges_to_endemic(self):
        """Long simulation should converge to the endemic equilibrium."""
        sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1, xi=0.05))
        eq = sim.endemic_equilibrium()
        assert eq is not None

        sim.reset()
        for _ in range(100000):
            sim.step()
        final_state = sim.observe()

        # Allow some tolerance since oscillations may not have fully damped
        np.testing.assert_allclose(final_state, eq, rtol=0.1, atol=5.0)


class TestSEIRSProperties:
    """Tests for analytical properties and computed quantities."""

    def test_latent_period(self):
        """Latent period = 1 / sigma."""
        sim = SEIRSSimulation(_make_config(sigma=0.2))
        assert sim.latent_period == pytest.approx(5.0)

    def test_infectious_period(self):
        """Infectious period = 1 / gamma."""
        sim = SEIRSSimulation(_make_config(gamma=0.1))
        assert sim.infectious_period == pytest.approx(10.0)

    def test_immunity_duration(self):
        """Immunity duration = 1 / xi."""
        sim = SEIRSSimulation(_make_config(xi=0.05))
        assert sim.immunity_duration == pytest.approx(20.0)

    def test_immunity_duration_infinite(self):
        """Immunity duration should be infinite when xi = 0."""
        sim = SEIRSSimulation(_make_config(xi=0.0))
        assert sim.immunity_duration == float("inf")

    def test_generation_time(self):
        """Generation time = 1/sigma + 1/gamma."""
        sim = SEIRSSimulation(_make_config(sigma=0.2, gamma=0.1))
        assert sim.generation_time == pytest.approx(15.0)

    def test_disease_free_equilibrium(self):
        """DFE should be [N, 0, 0, 0]."""
        sim = SEIRSSimulation(_make_config(N0=1000.0))
        dfe = sim.disease_free_equilibrium()
        np.testing.assert_allclose(dfe, [1000.0, 0.0, 0.0, 0.0])

    def test_total_population_property(self):
        """total_population should return sum of compartments."""
        sim = SEIRSSimulation(_make_config(N0=1000.0))
        sim.reset()
        assert sim.total_population == pytest.approx(1000.0)

    def test_total_population_before_reset(self):
        """total_population should return N0 before reset."""
        sim = SEIRSSimulation(_make_config(N0=1000.0))
        assert sim.total_population == pytest.approx(1000.0)


class TestSEIRSNormalization:
    """Tests for initial condition normalization."""

    def test_normalization_on_reset(self):
        """Initial conditions should normalize to N0 even if sum differs."""
        sim = SEIRSSimulation(_make_config(
            S_0=500.0, E_0=100.0, I_0=100.0, R_0_init=100.0, N0=1000.0,
        ))
        state = sim.reset()
        # Sum should be N0 = 1000 (input sums to 800, so scale by 1.25)
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)

    def test_normalization_preserves_ratios(self):
        """Normalization should preserve ratios between compartments."""
        sim = SEIRSSimulation(_make_config(
            S_0=400.0, E_0=200.0, I_0=200.0, R_0_init=200.0, N0=1000.0,
        ))
        state = sim.reset()
        # E_0 / S_0 = 0.5 should be preserved
        assert state[1] / state[0] == pytest.approx(0.5)


class TestSEIRSParameterSweep:
    """Tests for parameter sweep behavior."""

    def test_higher_beta_higher_peak(self):
        """Higher beta (transmission) should produce higher peak I."""
        peaks = []
        for beta in [0.2, 0.5, 0.8]:
            sim = SEIRSSimulation(_make_config(beta=beta, gamma=0.1, E_0=0.0))
            sim.reset()
            max_I = 0
            for _ in range(10000):
                state = sim.step()
                max_I = max(max_I, state[2])
            peaks.append(max_I)
        # Higher beta should give higher peak
        assert peaks[1] > peaks[0], "beta=0.5 peak should exceed beta=0.2 peak"
        assert peaks[2] > peaks[1], "beta=0.8 peak should exceed beta=0.5 peak"

    def test_higher_xi_sustains_more_infection(self):
        """Higher waning rate should maintain higher endemic I."""
        endemic_I_values = []
        for xi in [0.01, 0.05, 0.2]:
            sim = SEIRSSimulation(_make_config(beta=0.5, gamma=0.1, xi=xi))
            sim.reset()
            for _ in range(50000):
                sim.step()
            endemic_I_values.append(sim.observe()[2])
        # Higher waning rate recycles more people back to S, sustaining infection
        assert endemic_I_values[2] > endemic_I_values[0], (
            f"Higher xi should sustain more infection: "
            f"xi=0.2 -> I={endemic_I_values[2]:.2f}, "
            f"xi=0.01 -> I={endemic_I_values[0]:.2f}"
        )

    def test_faster_sigma_sharper_peak(self):
        """Higher sigma (faster latent progression) should produce earlier peak."""
        peak_times = []
        for sigma in [0.1, 0.5]:
            sim = SEIRSSimulation(_make_config(
                beta=0.5, sigma=sigma, gamma=0.1, E_0=0.0,
            ))
            sim.reset()
            peak_time = 0
            peak_I = 0
            for step in range(10000):
                state = sim.step()
                if state[2] > peak_I:
                    peak_I = state[2]
                    peak_time = step
            peak_times.append(peak_time)
        assert peak_times[1] < peak_times[0], (
            f"Higher sigma should give earlier peak: "
            f"sigma=0.5 -> t={peak_times[1]}, sigma=0.1 -> t={peak_times[0]}"
        )


class TestSEIRSRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_sweep_data(self):
        """Sweep data generation should produce valid arrays."""
        from simulating_anything.rediscovery.seirs import (
            generate_seirs_sweep_data,
        )
        data = generate_seirs_sweep_data(n_samples=5, n_steps=500, dt=0.1)
        assert "beta" in data
        assert "sigma" in data
        assert "gamma" in data
        assert "xi" in data
        assert "R0" in data
        assert "peak_I" in data
        assert "endemic_I" in data
        assert "conservation_error" in data
        assert "oscillation_detected" in data
        assert len(data["beta"]) == 5
        assert np.all(data["R0"] > 0)
        assert np.all(np.isfinite(data["peak_I"]))

    def test_generate_ode_data(self):
        """ODE data generation should produce valid trajectory."""
        from simulating_anything.rediscovery.seirs import (
            generate_seirs_ode_data,
        )
        data = generate_seirs_ode_data(n_steps=100, dt=0.1)
        assert data["states"].shape == (101, 4)
        # Population should be conserved
        totals = data["states"].sum(axis=1)
        np.testing.assert_allclose(totals, 1000.0, atol=1.0)

    def test_sweep_r0_range(self):
        """Sweep should cover both sub- and super-critical R0."""
        from simulating_anything.rediscovery.seirs import (
            generate_seirs_sweep_data,
        )
        data = generate_seirs_sweep_data(n_samples=50, n_steps=500, dt=0.1)
        assert np.any(data["R0"] < 1.0), "Should have some R0 < 1 cases"
        assert np.any(data["R0"] > 1.0), "Should have some R0 > 1 cases"

    def test_ode_data_parameters(self):
        """ODE data should include ground truth parameters."""
        from simulating_anything.rediscovery.seirs import (
            generate_seirs_ode_data,
        )
        data = generate_seirs_ode_data(n_steps=50, dt=0.1)
        assert data["beta"] == 0.5
        assert data["sigma"] == 0.2
        assert data["gamma"] == 0.1
        assert data["xi"] == 0.05
        assert data["N0"] == 1000.0

    def test_conservation_data(self):
        """Conservation check data should show small errors."""
        from simulating_anything.rediscovery.seirs import (
            generate_seirs_conservation_data,
        )
        data = generate_seirs_conservation_data(n_steps=500, dt=0.1)
        assert "max_conservation_errors" in data
        assert "parameters" in data
        assert len(data["max_conservation_errors"]) == 20
        # Conservation errors should be small
        assert np.all(data["max_conservation_errors"] < 1e-3), (
            f"Conservation errors too large: {data['max_conservation_errors'].max()}"
        )

    def test_oscillation_data(self):
        """Oscillation data should detect peaks."""
        from simulating_anything.rediscovery.seirs import (
            generate_seirs_oscillation_data,
        )
        data = generate_seirs_oscillation_data(n_steps=5000, dt=0.1)
        assert "I_values" in data
        assert "times" in data
        assert "peak_indices" in data
        assert "periods" in data
        assert "endemic_I_theory" in data
        assert len(data["I_values"]) == 5001
        assert len(data["times"]) == 5001

    def test_sweep_conservation_errors_small(self):
        """Conservation errors in sweep should be small."""
        from simulating_anything.rediscovery.seirs import (
            generate_seirs_sweep_data,
        )
        data = generate_seirs_sweep_data(n_samples=10, n_steps=500, dt=0.1)
        assert np.all(data["conservation_error"] < 1e-3), (
            f"Conservation error too large: {data['conservation_error'].max()}"
        )

    def test_sweep_peak_I_positive_for_supercritical(self):
        """Peak I should be positive for R0 > 1 cases."""
        from simulating_anything.rediscovery.seirs import (
            generate_seirs_sweep_data,
        )
        data = generate_seirs_sweep_data(n_samples=20, n_steps=2000, dt=0.1)
        mask = data["R0"] > 2.0
        if mask.sum() > 0:
            assert np.all(data["peak_I"][mask] > 0), (
                "Peak I should be positive for R0 > 2"
            )
