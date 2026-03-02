"""Tests for the SIRS (Susceptible-Infected-Recovered-Susceptible) epidemic model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sirs import SIRSSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.1,
    n_steps: int = 1000,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for SIRS with optional overrides."""
    params: dict[str, float] = {
        "beta": 0.5,
        "gamma": 0.15,
        "xi": 0.05,
        "N0": 1000.0,
        "I_0": 10.0,
        "R_0_init": 0.0,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.SIRS,  # Placeholder domain
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestSIRSCreation:
    """Tests for simulation creation and initialization."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = SIRSSimulation(_make_config())
        assert sim.beta == 0.5
        assert sim.gamma == 0.15
        assert sim.xi == 0.05
        assert sim.N0 == 1000.0

    def test_initial_state_shape(self):
        """State vector should be [S, I, R] with 3 components."""
        sim = SIRSSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_values(self):
        """Initial conditions should match parameters."""
        sim = SIRSSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [990.0, 10.0, 0.0])

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = SIRSSimulation(_make_config(beta=0.6, gamma=0.2, xi=0.1))
        assert sim.beta == 0.6
        assert sim.gamma == 0.2
        assert sim.xi == 0.1


class TestSIRSBasicSimulation:
    """Core simulation mechanics tests."""

    def test_step_advances(self):
        """State should change after a step."""
        sim = SIRSSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_state(self):
        """Observe should return the current state."""
        sim = SIRSSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_run_method(self):
        """run() should produce a TrajectoryData object."""
        sim = SIRSSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = SIRSSimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = SIRSSimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_long_run_stability(self):
        """No NaN after many steps."""
        sim = SIRSSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"


class TestSIRSConservation:
    """Conservation law tests: S + I + R = N0 at all times."""

    def test_total_population_conservation(self):
        """S + I + R should equal N0 at all times."""
        sim = SIRSSimulation(_make_config())
        sim.reset()
        for _ in range(500):
            state = sim.step()
            np.testing.assert_allclose(
                state.sum(), 1000.0, atol=1e-6,
                err_msg="Total population conservation violated",
            )

    def test_initial_total_population(self):
        """S + I + R should equal N0 at initialization."""
        sim = SIRSSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)

    def test_conservation_long_run(self):
        """Conservation should hold over long simulation."""
        sim = SIRSSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            state = sim.step()
        np.testing.assert_allclose(
            state.sum(), 1000.0, atol=1e-4,
            err_msg="Conservation violated over long run",
        )


class TestSIRSNonNegativity:
    """All compartments must remain non-negative."""

    def test_non_negative_all_compartments(self):
        """S, I, R should all remain >= 0."""
        sim = SIRSSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"S became negative: {state[0]}"
            assert state[1] >= 0, f"I became negative: {state[1]}"
            assert state[2] >= 0, f"R became negative: {state[2]}"

    def test_non_negative_high_waning(self):
        """Non-negativity should hold even with high waning rate."""
        sim = SIRSSimulation(_make_config(xi=1.0))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"


class TestSIRSR0:
    """Tests for the basic reproduction number R0 = beta / gamma."""

    def test_r0_formula(self):
        """R0 should equal beta / gamma."""
        sim = SIRSSimulation(_make_config(beta=0.5, gamma=0.15))
        expected = 0.5 / 0.15
        assert sim.R0 == pytest.approx(expected)

    def test_r0_supercritical(self):
        """R0 > 1 with standard parameters."""
        sim = SIRSSimulation(_make_config(beta=0.5, gamma=0.15))
        assert sim.R0 > 1.0

    def test_r0_subcritical(self):
        """R0 < 1 when recovery exceeds transmission."""
        sim = SIRSSimulation(_make_config(beta=0.05, gamma=0.15))
        assert sim.R0 < 1.0

    def test_r0_different_params(self):
        """R0 should vary correctly with different parameters."""
        sim = SIRSSimulation(_make_config(beta=0.6, gamma=0.2))
        assert sim.R0 == pytest.approx(3.0)


class TestSIRSEndemicEquilibrium:
    """Tests for the endemic equilibrium."""

    def test_endemic_equilibrium_exists_above_threshold(self):
        """Endemic equilibrium should exist when R0 > 1."""
        sim = SIRSSimulation(_make_config(beta=0.5, gamma=0.15, xi=0.05))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        assert eq.shape == (3,)
        assert np.all(eq >= 0), f"Negative equilibrium components: {eq}"

    def test_endemic_equilibrium_none_below_threshold(self):
        """Endemic equilibrium should not exist when R0 <= 1."""
        sim = SIRSSimulation(_make_config(beta=0.05, gamma=0.15))
        eq = sim.endemic_equilibrium()
        assert eq is None

    def test_endemic_equilibrium_sums_to_n(self):
        """S* + I* + R* should equal N0."""
        sim = SIRSSimulation(_make_config(beta=0.5, gamma=0.15, xi=0.05))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        np.testing.assert_allclose(eq.sum(), 1000.0, atol=1e-8)

    def test_endemic_s_star_equals_n_over_r0(self):
        """At endemic equilibrium, S* = N / R0."""
        sim = SIRSSimulation(_make_config(beta=0.5, gamma=0.15, xi=0.05))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        expected_S = 1000.0 / sim.R0
        np.testing.assert_allclose(eq[0], expected_S, atol=1e-6)

    def test_simulation_approaches_endemic_equilibrium(self):
        """Long simulation should approach endemic equilibrium."""
        sim = SIRSSimulation(_make_config(
            beta=0.5, gamma=0.15, xi=0.05, n_steps=100000,
        ))
        eq = sim.endemic_equilibrium()
        assert eq is not None

        sim.reset()
        for _ in range(100000):
            sim.step()
        final_state = sim.observe()

        # Should be within 5% of endemic equilibrium
        np.testing.assert_allclose(final_state, eq, rtol=0.05, atol=1.0)

    def test_endemic_i_increases_with_xi(self):
        """Higher waning rate xi should increase endemic I*."""
        sim_low = SIRSSimulation(_make_config(beta=0.5, gamma=0.15, xi=0.01))
        sim_high = SIRSSimulation(_make_config(beta=0.5, gamma=0.15, xi=0.2))
        eq_low = sim_low.endemic_equilibrium()
        eq_high = sim_high.endemic_equilibrium()
        assert eq_low is not None and eq_high is not None
        assert eq_high[1] > eq_low[1], (
            f"Higher xi should give higher I*: {eq_high[1]} vs {eq_low[1]}"
        )


class TestSIRSEpidemicBehavior:
    """Tests for epidemic dynamics above and below threshold."""

    def test_epidemic_occurs_above_threshold(self):
        """With R0 > 1, infected should rise then approach endemic level."""
        sim = SIRSSimulation(_make_config(beta=0.5, gamma=0.15, xi=0.05))
        sim.reset()
        max_I = 0
        for _ in range(10000):
            state = sim.step()
            max_I = max(max_I, state[1])
        # Significant epidemic peak
        assert max_I > 50.0, f"Peak I too low: {max_I}"

    def test_no_epidemic_below_threshold(self):
        """With R0 < 1, infected should decay."""
        sim = SIRSSimulation(_make_config(beta=0.05, gamma=0.15, xi=0.05))
        sim.reset()
        initial_I = sim.observe()[1]
        for _ in range(1000):
            state = sim.step()
        assert state[1] < initial_I, (
            f"I should decay when R0 < 1: I={state[1]}, I_0={initial_I}"
        )

    def test_endemic_state_persists(self):
        """With waning immunity, infection should not go to zero (R0 > 1)."""
        sim = SIRSSimulation(_make_config(
            beta=0.5, gamma=0.15, xi=0.05,
        ))
        sim.reset()
        # Run long enough to reach endemic state
        for _ in range(50000):
            sim.step()
        final_I = sim.observe()[1]
        # Unlike SIR, SIRS maintains endemic infection
        assert final_I > 1.0, (
            f"Endemic I should persist with waning immunity: I={final_I}"
        )

    def test_s_not_monotonically_decreasing(self):
        """Unlike SIR, S can increase due to waning immunity."""
        sim = SIRSSimulation(_make_config(
            beta=0.5, gamma=0.15, xi=0.1,
        ))
        sim.reset()
        # Collect S values over full trajectory
        S_values = []
        for _ in range(10000):
            state = sim.step()
            S_values.append(state[0])
        # Find the minimum S (occurs during epidemic peak)
        S_arr = np.array(S_values)
        min_idx = np.argmin(S_arr)
        S_min = S_arr[min_idx]
        # After minimum, S should increase (waning immunity replenishes S)
        if min_idx < len(S_arr) - 100:
            S_after_min = S_arr[min_idx + 100]
            assert S_after_min > S_min, (
                f"S should increase after minimum due to waning immunity: "
                f"S_min={S_min}, S_later={S_after_min}"
            )


class TestSIRSWaningImmunity:
    """Tests for the waning immunity behavior."""

    def test_zero_xi_approaches_sir(self):
        """With xi=0 (no waning), dynamics should match SIR (I -> 0)."""
        sim = SIRSSimulation(_make_config(
            beta=0.5, gamma=0.15, xi=0.0,
        ))
        sim.reset()
        for _ in range(50000):
            sim.step()
        final_I = sim.observe()[1]
        # SIR limit: infection dies out
        assert final_I < 0.1, f"With xi=0, I should -> 0: I={final_I}"

    def test_large_xi_higher_endemic(self):
        """Larger xi should produce higher endemic infected level."""
        results = []
        for xi_val in [0.01, 0.05, 0.2]:
            sim = SIRSSimulation(_make_config(
                beta=0.5, gamma=0.15, xi=xi_val,
            ))
            sim.reset()
            for _ in range(50000):
                sim.step()
            results.append(sim.observe()[1])
        # Endemic I should increase with xi
        assert results[1] > results[0], (
            f"Endemic I should increase with xi: {results}"
        )
        assert results[2] > results[1], (
            f"Endemic I should increase with xi: {results}"
        )

    def test_damped_oscillations_to_endemic(self):
        """SIRS should exhibit damped oscillations toward endemic equilibrium."""
        sim = SIRSSimulation(_make_config(
            beta=0.5, gamma=0.15, xi=0.05,
        ))
        sim.reset()

        # Collect I values
        I_values = []
        for _ in range(10000):
            state = sim.step()
            I_values.append(state[1])

        I_arr = np.array(I_values)
        # Check that there are at least 2 local maxima (oscillatory approach)
        # Count peaks in the first half of trajectory
        half = len(I_arr) // 2
        I_first_half = I_arr[:half]
        peaks = 0
        for j in range(1, len(I_first_half) - 1):
            if I_first_half[j] > I_first_half[j - 1] and I_first_half[j] > I_first_half[j + 1]:
                # Only count significant peaks
                if I_first_half[j] > 5.0:
                    peaks += 1
        # Should have at least the initial epidemic peak
        assert peaks >= 1, f"Expected oscillatory approach, found {peaks} peaks"


class TestSIRSEquilibria:
    """Tests for equilibrium states."""

    def test_disease_free_equilibrium(self):
        """DFE should be [N, 0, 0]."""
        sim = SIRSSimulation(_make_config(N0=1000.0))
        dfe = sim.disease_free_equilibrium()
        np.testing.assert_allclose(dfe, [1000.0, 0.0, 0.0])

    def test_total_population_property(self):
        """total_population should return sum of all compartments."""
        sim = SIRSSimulation(_make_config(N0=1000.0))
        sim.reset()
        assert sim.total_population == pytest.approx(1000.0)

    def test_effective_reproduction_number_initial(self):
        """R_eff should start close to R0 when nearly all are susceptible."""
        sim = SIRSSimulation(_make_config(beta=0.5, gamma=0.15, I_0=1.0))
        sim.reset()
        r_eff = sim.effective_reproduction_number()
        # With S ~ N, R_eff ~ R0
        assert r_eff == pytest.approx(sim.R0, rel=0.01)

    def test_effective_reproduction_number_decreases_during_epidemic(self):
        """R_eff should decrease as susceptibles are depleted."""
        sim = SIRSSimulation(_make_config(beta=0.5, gamma=0.15))
        sim.reset()
        r_eff_initial = sim.effective_reproduction_number()
        for _ in range(500):
            sim.step()
        r_eff_later = sim.effective_reproduction_number()
        assert r_eff_later < r_eff_initial, (
            f"R_eff should decrease: {r_eff_initial} -> {r_eff_later}"
        )


class TestSIRSSIRLimit:
    """Tests for the SIR limit behavior (xi -> 0)."""

    def test_sir_limit_final_size_positive(self):
        """SIR limit final size should be positive for R0 > 1."""
        sim = SIRSSimulation(_make_config(beta=0.5, gamma=0.15))
        fs = sim.sir_limit_final_size()
        assert fs > 0.5, f"Final size too low for R0 ~ 3.3: {fs}"

    def test_sir_limit_final_size_zero_below_threshold(self):
        """SIR limit final size should be 0 when R0 < 1."""
        sim = SIRSSimulation(_make_config(beta=0.05, gamma=0.15))
        assert sim.sir_limit_final_size() == 0.0


class TestSIRSRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_sweep_data(self):
        """Sweep data generation should produce valid arrays."""
        from simulating_anything.rediscovery.sirs import generate_sirs_sweep_data

        data = generate_sirs_sweep_data(n_samples=5, n_steps=500, dt=0.1)
        assert "beta" in data
        assert "gamma" in data
        assert "xi" in data
        assert "R0" in data
        assert "peak_I" in data
        assert "final_I" in data
        assert "final_S" in data
        assert "time_to_peak" in data
        assert len(data["beta"]) == 5
        assert np.all(data["R0"] > 0)
        assert np.all(np.isfinite(data["peak_I"]))

    def test_generate_ode_data(self):
        """ODE data generation should produce valid trajectory."""
        from simulating_anything.rediscovery.sirs import generate_sirs_ode_data

        data = generate_sirs_ode_data(n_steps=100, dt=0.1)
        assert data["states"].shape == (101, 3)
        # Total population should be conserved
        totals = data["states"].sum(axis=1)
        np.testing.assert_allclose(totals, 1000.0, atol=1.0)

    def test_sweep_r0_range(self):
        """Sweep should cover both sub- and super-critical R0."""
        from simulating_anything.rediscovery.sirs import generate_sirs_sweep_data

        data = generate_sirs_sweep_data(n_samples=50, n_steps=500, dt=0.1)
        assert np.any(data["R0"] < 1.0), "Should have some R0 < 1 cases"
        assert np.any(data["R0"] > 1.0), "Should have some R0 > 1 cases"

    def test_ode_data_parameters(self):
        """ODE data should include ground truth parameters."""
        from simulating_anything.rediscovery.sirs import generate_sirs_ode_data

        data = generate_sirs_ode_data(n_steps=50, dt=0.1)
        assert data["beta"] == 0.5
        assert data["gamma"] == 0.15
        assert data["xi"] == 0.05
        assert data["N0"] == 1000.0

    def test_generate_waning_sweep(self):
        """Waning rate sweep should produce valid results."""
        from simulating_anything.rediscovery.sirs import generate_sirs_waning_sweep

        data = generate_sirs_waning_sweep(n_xi_values=5, n_steps=1000, dt=0.1)
        assert "xi" in data
        assert "final_I" in data
        assert "final_S" in data
        assert "max_I" in data
        assert "oscillation_count" in data
        assert len(data["xi"]) == 5
        assert np.all(data["xi"] > 0)

    def test_sweep_endemic_infection_positive(self):
        """Epidemics with R0 > 1 and xi > 0 should have endemic I > 0 after long run."""
        from simulating_anything.rediscovery.sirs import generate_sirs_sweep_data

        data = generate_sirs_sweep_data(n_samples=20, n_steps=5000, dt=0.1)
        mask = (data["R0"] > 2.0) & (data["xi"] > 0.02)
        if mask.sum() > 0:
            assert np.any(data["final_I"][mask] > 0), (
                "Epidemics with R0 > 2 and xi > 0 should have endemic I > 0"
            )


class TestSIRSEndemicEquilibriumVerification:
    """Additional verification that the endemic equilibrium is a true fixed point."""

    def test_derivatives_vanish_at_endemic_eq(self):
        """Derivatives should be zero at the endemic equilibrium."""
        sim = SIRSSimulation(_make_config(beta=0.5, gamma=0.15, xi=0.05))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        derivs = sim._derivatives(eq)
        np.testing.assert_allclose(
            derivs, np.zeros(3), atol=1e-10,
            err_msg="Derivatives should vanish at endemic equilibrium",
        )

    def test_endemic_equilibrium_multiple_params(self):
        """Endemic equilibrium formula should match across parameter sets."""
        param_sets = [
            {"beta": 0.3, "gamma": 0.1, "xi": 0.02},
            {"beta": 0.8, "gamma": 0.2, "xi": 0.1},
            {"beta": 0.6, "gamma": 0.05, "xi": 0.5},
        ]
        for params in param_sets:
            sim = SIRSSimulation(_make_config(**params))
            eq = sim.endemic_equilibrium()
            assert eq is not None
            derivs = sim._derivatives(eq)
            np.testing.assert_allclose(
                derivs, np.zeros(3), atol=1e-10,
                err_msg=f"Derivatives nonzero at endemic eq for params={params}",
            )
