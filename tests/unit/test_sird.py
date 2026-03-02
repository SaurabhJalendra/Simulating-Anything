"""Tests for the SIRD (Susceptible-Infected-Recovered-Deceased) epidemic model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sird import SIRDSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.1,
    n_steps: int = 1000,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for SIRD with optional overrides."""
    params: dict[str, float] = {
        "beta": 0.4,
        "gamma": 0.1,
        "mu": 0.02,
        "N0": 1000.0,
        "S_0": 990.0,
        "I_0": 10.0,
        "R_0_init": 0.0,
        "D_0": 0.0,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.SIRD,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestSIRDCreation:
    """Tests for simulation creation and initialization."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = SIRDSimulation(_make_config())
        assert sim.beta == 0.4
        assert sim.gamma == 0.1
        assert sim.mu == 0.02
        assert sim.N0 == 1000.0

    def test_initial_state_shape(self):
        """State vector should be [S, I, R, D] with 4 components."""
        sim = SIRDSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (4,)

    def test_initial_state_values(self):
        """Initial conditions should match parameters."""
        sim = SIRDSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [990.0, 10.0, 0.0, 0.0])

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = SIRDSimulation(_make_config(beta=0.6, gamma=0.2, mu=0.05))
        assert sim.beta == 0.6
        assert sim.gamma == 0.2
        assert sim.mu == 0.05


class TestSIRDBasicSimulation:
    """Core simulation mechanics tests."""

    def test_step_advances(self):
        """State should change after a step."""
        sim = SIRDSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_state(self):
        """Observe should return the current state."""
        sim = SIRDSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (4,)

    def test_run_method(self):
        """run() should produce a TrajectoryData object."""
        sim = SIRDSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 4)
        assert len(traj.timestamps) == 101

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = SIRDSimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = SIRDSimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_long_run_stability(self):
        """No NaN after many steps."""
        sim = SIRDSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"


class TestSIRDConservation:
    """Conservation law tests."""

    def test_total_population_conservation(self):
        """S + I + R + D should equal N0 at all times."""
        sim = SIRDSimulation(_make_config())
        sim.reset()
        for _ in range(500):
            state = sim.step()
            np.testing.assert_allclose(
                state.sum(), 1000.0, atol=1e-6,
                err_msg="Total population conservation violated",
            )

    def test_initial_total_population(self):
        """S + I + R + D should equal N0 at initialization."""
        sim = SIRDSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)

    def test_conservation_long_run(self):
        """Conservation should hold over long simulation."""
        sim = SIRDSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            state = sim.step()
        np.testing.assert_allclose(
            state.sum(), 1000.0, atol=1e-4,
            err_msg="Conservation violated over long run",
        )


class TestSIRDNonNegativity:
    """All compartments must remain non-negative."""

    def test_non_negative_all_compartments(self):
        """S, I, R, D should all remain >= 0."""
        sim = SIRDSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"S became negative: {state[0]}"
            assert state[1] >= 0, f"I became negative: {state[1]}"
            assert state[2] >= 0, f"R became negative: {state[2]}"
            assert state[3] >= 0, f"D became negative: {state[3]}"

    def test_non_negative_high_mortality(self):
        """Non-negativity should hold even with high mortality rate."""
        sim = SIRDSimulation(_make_config(mu=0.3))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"


class TestSIRDMonotonicity:
    """Monotonicity tests for S and D."""

    def test_s_monotonically_decreasing(self):
        """S should never increase (no births, no loss of immunity)."""
        sim = SIRDSimulation(_make_config())
        sim.reset()
        prev_S = sim.observe()[0]
        for _ in range(5000):
            state = sim.step()
            assert state[0] <= prev_S + 1e-10, (
                f"S increased: {prev_S} -> {state[0]}"
            )
            prev_S = state[0]

    def test_d_monotonically_increasing(self):
        """D should never decrease (deaths are irreversible)."""
        sim = SIRDSimulation(_make_config())
        sim.reset()
        prev_D = sim.observe()[3]
        for _ in range(5000):
            state = sim.step()
            assert state[3] >= prev_D - 1e-10, (
                f"D decreased: {prev_D} -> {state[3]}"
            )
            prev_D = state[3]


class TestSIRDR0:
    """Tests for the basic reproduction number R0 = beta / (gamma + mu)."""

    def test_r0_formula(self):
        """R0 should equal beta / (gamma + mu)."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        expected = 0.4 / (0.1 + 0.02)
        assert sim.R0 == pytest.approx(expected)

    def test_r0_supercritical(self):
        """R0 > 1 with standard parameters."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        assert sim.R0 > 1.0

    def test_r0_subcritical(self):
        """R0 < 1 when removal rate exceeds transmission."""
        sim = SIRDSimulation(_make_config(beta=0.05, gamma=0.1, mu=0.02))
        assert sim.R0 < 1.0

    def test_r0_different_params(self):
        """R0 should vary correctly with different parameters."""
        sim = SIRDSimulation(_make_config(beta=0.6, gamma=0.2, mu=0.1))
        assert sim.R0 == pytest.approx(0.6 / 0.3)


class TestSIRDCFR:
    """Tests for the case fatality rate CFR = mu / (gamma + mu)."""

    def test_cfr_formula(self):
        """CFR should equal mu / (gamma + mu)."""
        sim = SIRDSimulation(_make_config(gamma=0.1, mu=0.02))
        expected = 0.02 / (0.1 + 0.02)
        assert sim.case_fatality_rate() == pytest.approx(expected)

    def test_cfr_zero_mortality(self):
        """CFR should be 0 when mu = 0."""
        sim = SIRDSimulation(_make_config(mu=0.0))
        assert sim.case_fatality_rate() == pytest.approx(0.0)

    def test_cfr_high_mortality(self):
        """CFR should approach 1 as mu >> gamma."""
        sim = SIRDSimulation(_make_config(gamma=0.01, mu=10.0))
        assert sim.case_fatality_rate() > 0.99

    def test_cfr_equal_rates(self):
        """CFR should be 0.5 when mu = gamma."""
        sim = SIRDSimulation(_make_config(gamma=0.1, mu=0.1))
        assert sim.case_fatality_rate() == pytest.approx(0.5)


class TestSIRDEpidemicBehavior:
    """Tests for epidemic dynamics above and below threshold."""

    def test_epidemic_occurs_above_threshold(self):
        """With R0 > 1, infected should rise then fall."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        sim.reset()
        max_I = 0
        for _ in range(10000):
            state = sim.step()
            max_I = max(max_I, state[1])
        # Significant epidemic peak
        assert max_I > 50.0, f"Peak I too low: {max_I}"
        # Epidemic should end
        assert state[1] < 1.0, f"I should decay: {state[1]}"

    def test_no_epidemic_below_threshold(self):
        """With R0 < 1, infected should decay monotonically."""
        sim = SIRDSimulation(_make_config(beta=0.05, gamma=0.1, mu=0.02))
        sim.reset()
        initial_I = sim.observe()[1]
        for _ in range(1000):
            state = sim.step()
        assert state[1] < initial_I, (
            f"I should decay when R0 < 1: I={state[1]}, I_0={initial_I}"
        )

    def test_final_size_positive_above_threshold(self):
        """Final size should be positive when R0 > 1."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        fs = sim.final_size()
        assert fs > 0.5, f"Final size too low for R0 ~ 3.3: {fs}"

    def test_final_size_zero_below_threshold(self):
        """Final size should be 0 when R0 < 1."""
        sim = SIRDSimulation(_make_config(beta=0.05, gamma=0.1, mu=0.02))
        assert sim.final_size() == 0.0

    def test_s_infinity_positive(self):
        """Not everyone gets infected; some S remains at end."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        sim.reset()
        for _ in range(20000):
            sim.step()
        final_S = sim.observe()[0]
        assert final_S > 0, f"S_inf should be > 0, got {final_S}"


class TestSIRDDeathToll:
    """Tests for deceased compartment behavior."""

    def test_deaths_occur_during_epidemic(self):
        """D should be positive after an epidemic (R0 > 1)."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        sim.reset()
        for _ in range(10000):
            sim.step()
        assert sim.observe()[3] > 0, "No deaths occurred during epidemic"

    def test_deaths_proportional_to_cfr(self):
        """Final D / (R + D) should approximate CFR = mu / (gamma + mu)."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        sim.reset()
        for _ in range(20000):
            sim.step()
        state = sim.observe()
        R, D = state[2], state[3]
        if R + D > 1e-6:
            observed_cfr = D / (R + D)
            expected_cfr = sim.case_fatality_rate()
            assert observed_cfr == pytest.approx(expected_cfr, abs=0.01), (
                f"Observed CFR {observed_cfr:.4f} != expected {expected_cfr:.4f}"
            )

    def test_no_deaths_without_mortality(self):
        """D should remain 0 when mu = 0."""
        sim = SIRDSimulation(_make_config(mu=0.0))
        sim.reset()
        for _ in range(5000):
            sim.step()
        assert sim.observe()[3] == pytest.approx(0.0, abs=1e-10)

    def test_high_mortality_more_deaths(self):
        """Higher mu should produce more deaths."""
        sim_low = SIRDSimulation(_make_config(beta=0.4, mu=0.01))
        sim_low.reset()
        for _ in range(10000):
            sim_low.step()
        d_low = sim_low.observe()[3]

        sim_high = SIRDSimulation(_make_config(beta=0.4, mu=0.1))
        sim_high.reset()
        for _ in range(10000):
            sim_high.step()
        d_high = sim_high.observe()[3]

        assert d_high > d_low, (
            f"Higher mu should give more deaths: {d_high} vs {d_low}"
        )


class TestSIRDInfectionFatalityRate:
    """Tests for infection fatality rate computation."""

    def test_ifr_after_epidemic(self):
        """IFR should be computable after an epidemic."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        sim.reset()
        for _ in range(20000):
            sim.step()
        ifr = sim.infection_fatality_rate()
        assert 0 < ifr < 1, f"IFR out of range: {ifr}"

    def test_ifr_approximates_cfr(self):
        """IFR should approach CFR after epidemic completes."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        sim.reset()
        for _ in range(20000):
            sim.step()
        ifr = sim.infection_fatality_rate()
        cfr = sim.case_fatality_rate()
        assert ifr == pytest.approx(cfr, abs=0.01)

    def test_ifr_before_epidemic(self):
        """IFR should be 0 before any cases resolve."""
        sim = SIRDSimulation(_make_config())
        sim.reset()
        assert sim.infection_fatality_rate() == 0.0


class TestSIRDPeakInfected:
    """Tests for peak infected computation."""

    def test_peak_infected_positive(self):
        """Peak infected should be positive for R0 > 1."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        peak = sim.peak_infected()
        assert peak > 0, f"Peak infected should be > 0: {peak}"

    def test_peak_timing_occurs(self):
        """Simulated peak should occur within the simulation."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        sim.reset()
        max_I = 0
        peak_step = 0
        for step in range(10000):
            state = sim.step()
            if state[1] > max_I:
                max_I = state[1]
                peak_step = step
        assert peak_step > 0, "Peak should not be at step 0"
        assert peak_step < 9999, "Peak should occur before end of simulation"


class TestSIRDEquilibria:
    """Tests for equilibrium states."""

    def test_disease_free_equilibrium(self):
        """DFE should be [N0, 0, 0, 0]."""
        sim = SIRDSimulation(_make_config(N0=1000.0))
        dfe = sim.disease_free_equilibrium()
        np.testing.assert_allclose(dfe, [1000.0, 0.0, 0.0, 0.0])

    def test_living_population_decreases(self):
        """Living population N = S + I + R should decrease as people die."""
        sim = SIRDSimulation(_make_config(beta=0.4, gamma=0.1, mu=0.02))
        sim.reset()
        initial_living = sim.living_population
        for _ in range(10000):
            sim.step()
        assert sim.living_population < initial_living, (
            "Living population should decrease due to mortality"
        )

    def test_total_population_property(self):
        """total_population should return sum of all compartments."""
        sim = SIRDSimulation(_make_config(N0=1000.0))
        sim.reset()
        assert sim.total_population == pytest.approx(1000.0)


class TestSIRDNormalization:
    """Tests for initial condition normalization."""

    def test_normalization_on_reset(self):
        """Initial conditions should normalize to N0."""
        sim = SIRDSimulation(_make_config(
            S_0=500.0, I_0=100.0, R_0_init=100.0, D_0=0.0, N0=1000.0,
        ))
        state = sim.reset()
        # Living sum should be N0 - D_0 = 1000
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)

    def test_normalization_with_deceased(self):
        """Normalization should account for initial deceased."""
        sim = SIRDSimulation(_make_config(
            S_0=500.0, I_0=50.0, R_0_init=50.0, D_0=100.0, N0=1000.0,
        ))
        state = sim.reset()
        # Total should be N0 = 1000 (including D_0=100)
        # Living = S + I + R = 900, D = 100
        assert state[3] == pytest.approx(100.0)
        np.testing.assert_allclose(
            state[0] + state[1] + state[2], 900.0, atol=1e-10,
        )


class TestSIRDRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_sweep_data(self):
        """Sweep data generation should produce valid arrays."""
        from simulating_anything.rediscovery.sird import (
            generate_sird_sweep_data,
        )
        data = generate_sird_sweep_data(n_samples=5, n_steps=500, dt=0.1)
        assert "beta" in data
        assert "gamma" in data
        assert "mu" in data
        assert "R0" in data
        assert "CFR" in data
        assert "peak_I" in data
        assert "final_size" in data
        assert "final_D" in data
        assert "time_to_peak" in data
        assert len(data["beta"]) == 5
        assert np.all(data["R0"] > 0)
        assert np.all(np.isfinite(data["peak_I"]))

    def test_generate_ode_data(self):
        """ODE data generation should produce valid trajectory."""
        from simulating_anything.rediscovery.sird import (
            generate_sird_ode_data,
        )
        data = generate_sird_ode_data(n_steps=100, dt=0.1)
        assert data["states"].shape == (101, 4)
        # Total population should be conserved
        totals = data["states"].sum(axis=1)
        np.testing.assert_allclose(totals, 1000.0, atol=1.0)

    def test_sweep_r0_range(self):
        """Sweep should cover both sub- and super-critical R0."""
        from simulating_anything.rediscovery.sird import (
            generate_sird_sweep_data,
        )
        data = generate_sird_sweep_data(n_samples=50, n_steps=500, dt=0.1)
        assert np.any(data["R0"] < 1.0), "Should have some R0 < 1 cases"
        assert np.any(data["R0"] > 1.0), "Should have some R0 > 1 cases"

    def test_ode_data_parameters(self):
        """ODE data should include ground truth parameters."""
        from simulating_anything.rediscovery.sird import (
            generate_sird_ode_data,
        )
        data = generate_sird_ode_data(n_steps=50, dt=0.1)
        assert data["beta"] == 0.4
        assert data["gamma"] == 0.1
        assert data["mu"] == 0.02
        assert data["N0"] == 1000.0

    def test_sweep_cfr_values(self):
        """CFR values in sweep should match mu / (gamma + mu)."""
        from simulating_anything.rediscovery.sird import (
            generate_sird_sweep_data,
        )
        data = generate_sird_sweep_data(n_samples=10, n_steps=500, dt=0.1)
        expected_cfr = data["mu"] / (data["gamma"] + data["mu"])
        np.testing.assert_allclose(data["CFR"], expected_cfr, atol=1e-10)

    def test_sweep_final_deaths_positive(self):
        """Some sweeps should produce positive final deaths."""
        from simulating_anything.rediscovery.sird import (
            generate_sird_sweep_data,
        )
        data = generate_sird_sweep_data(n_samples=20, n_steps=2000, dt=0.1)
        mask = data["R0"] > 1.5
        if mask.sum() > 0:
            assert np.any(data["final_D"][mask] > 0), (
                "Epidemics with R0 > 1.5 should produce deaths"
            )
