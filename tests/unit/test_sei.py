"""Tests for the SEI (Susceptible-Exposed-Infected) epidemic model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sei import SEISimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.1,
    n_steps: int = 1000,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for SEI with optional overrides."""
    params: dict[str, float] = {
        "beta": 0.5,
        "sigma": 0.2,
        "mu": 0.05,
        "N0": 1000.0,
        "S_0": 985.0,
        "E_0": 10.0,
        "I_0": 5.0,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.SEI,  # Placeholder
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestSEICreation:
    """Tests for simulation creation and initialization."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = SEISimulation(_make_config())
        assert sim.beta == 0.5
        assert sim.sigma == 0.2
        assert sim.mu == 0.05
        assert sim.N0 == 1000.0

    def test_initial_state_shape(self):
        """State vector should be [S, E, I] with 3 components."""
        sim = SEISimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_values(self):
        """Initial conditions should match parameters."""
        sim = SEISimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [985.0, 10.0, 5.0])

    def test_initial_population_sum(self):
        """S + E + I should equal N0 at initialization."""
        sim = SEISimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = SEISimulation(_make_config(beta=0.8, sigma=0.3, mu=0.1))
        assert sim.beta == 0.8
        assert sim.sigma == 0.3
        assert sim.mu == 0.1


class TestSEIBasicSimulation:
    """Core simulation mechanics tests."""

    def test_step_advances(self):
        """State should change after a step."""
        sim = SEISimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_state(self):
        """Observe should return the current state."""
        sim = SEISimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_run_method(self):
        """run() should produce a TrajectoryData object."""
        sim = SEISimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = SEISimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = SEISimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_long_run_stability(self):
        """No NaN after many steps."""
        sim = SEISimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"


class TestSEINonNegativity:
    """All compartments must remain non-negative."""

    def test_non_negative_all_compartments(self):
        """S, E, I should all remain >= 0."""
        sim = SEISimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"S became negative: {state[0]}"
            assert state[1] >= 0, f"E became negative: {state[1]}"
            assert state[2] >= 0, f"I became negative: {state[2]}"

    def test_non_negative_high_mortality(self):
        """Non-negativity should hold even with high mortality rate."""
        sim = SEISimulation(_make_config(mu=0.3))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"

    def test_non_negative_high_transmission(self):
        """Non-negativity with high beta."""
        sim = SEISimulation(_make_config(beta=2.0, dt=0.05))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"


class TestSEIPopulationDynamics:
    """Tests for population conservation and decline."""

    def test_population_conservation_no_mortality(self):
        """S + E + I = N0 when mu = 0 (no deaths)."""
        sim = SEISimulation(_make_config(mu=0.0))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            np.testing.assert_allclose(
                state.sum(), 1000.0, atol=1e-6,
                err_msg="Population conservation violated when mu=0",
            )

    def test_population_decline_with_mortality(self):
        """S + E + I should decrease when mu > 0."""
        sim = SEISimulation(_make_config(beta=0.5, mu=0.05))
        sim.reset()
        initial_pop = sim.observe().sum()
        for _ in range(5000):
            sim.step()
        final_pop = sim.observe().sum()
        assert final_pop < initial_pop, (
            f"Population should decrease with mu > 0: "
            f"{initial_pop} -> {final_pop}"
        )

    def test_population_monotonic_decline(self):
        """Total population should never increase when mu > 0.

        dN/dt = -mu*I <= 0, so N is monotonically decreasing.
        """
        sim = SEISimulation(_make_config(beta=0.5, mu=0.05))
        sim.reset()
        prev_pop = sim.observe().sum()
        for _ in range(5000):
            state = sim.step()
            current_pop = state.sum()
            assert current_pop <= prev_pop + 1e-10, (
                f"Population increased: {prev_pop} -> {current_pop}"
            )
            prev_pop = current_pop


class TestSEISMonotonicity:
    """S should monotonically decrease (no births, no loss of immunity)."""

    def test_s_monotonically_decreasing(self):
        """S should never increase."""
        sim = SEISimulation(_make_config())
        sim.reset()
        prev_S = sim.observe()[0]
        for _ in range(5000):
            state = sim.step()
            assert state[0] <= prev_S + 1e-10, (
                f"S increased: {prev_S} -> {state[0]}"
            )
            prev_S = state[0]


class TestSEIR0:
    """Tests for the basic reproduction number R0 = beta / mu."""

    def test_r0_formula(self):
        """R0 should equal beta / mu."""
        sim = SEISimulation(_make_config(beta=0.5, mu=0.05))
        assert sim.R0 == pytest.approx(10.0)

    def test_r0_supercritical(self):
        """R0 > 1 with standard parameters."""
        sim = SEISimulation(_make_config(beta=0.5, mu=0.05))
        assert sim.R0 > 1.0

    def test_r0_subcritical(self):
        """R0 < 1 when mu > beta."""
        sim = SEISimulation(_make_config(beta=0.05, mu=0.1))
        assert sim.R0 < 1.0

    def test_r0_different_params(self):
        """R0 should vary correctly with different parameters."""
        sim = SEISimulation(_make_config(beta=0.6, mu=0.2))
        assert sim.R0 == pytest.approx(3.0)

    def test_r0_infinite_when_no_mortality(self):
        """R0 should be inf when mu = 0 (no removal)."""
        sim = SEISimulation(_make_config(mu=0.0))
        assert sim.R0 == float("inf")


class TestSEIEpidemicBehavior:
    """Tests for epidemic dynamics above and below threshold."""

    def test_epidemic_occurs_above_threshold(self):
        """With R0 > 1, exposed should rise then fall as susceptibles deplete."""
        sim = SEISimulation(_make_config(beta=0.5, mu=0.05))  # R0 = 10
        sim.reset()
        max_E = 0
        for _ in range(10000):
            state = sim.step()
            max_E = max(max_E, state[1])
        # Significant exposed peak
        assert max_E > 20.0, f"Peak E too low: {max_E}"

    def test_no_epidemic_below_threshold(self):
        """With R0 < 1, infected should decay."""
        sim = SEISimulation(
            _make_config(beta=0.05, mu=0.1, E_0=0.0)
        )  # R0 = 0.5
        sim.reset()
        initial_I = sim.observe()[2]
        for _ in range(1000):
            state = sim.step()
        assert state[2] < initial_I, (
            f"I should decay when R0 < 1: I={state[2]}, I_0={initial_I}"
        )

    def test_exposed_precedes_infected_peak(self):
        """Exposed peak should occur before infected peak (latency delay)."""
        sim = SEISimulation(
            _make_config(beta=0.5, sigma=0.1, mu=0.05, E_0=0.0, I_0=5.0)
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
            f"infected peak ({peak_I_time})"
        )

    def test_s_depletes_with_high_r0(self):
        """With R0 >> 1, nearly all susceptibles should be infected."""
        sim = SEISimulation(_make_config(beta=1.0, mu=0.01))  # R0 = 100
        sim.reset()
        for _ in range(20000):
            sim.step()
        final_S = sim.observe()[0]
        # S should be very small
        assert final_S < 100.0, (
            f"S should be nearly depleted with R0=100: S={final_S}"
        )

    def test_no_recovery_i_accumulates(self):
        """With mu = 0, infected accumulate permanently (no recovery)."""
        sim = SEISimulation(_make_config(mu=0.0, beta=0.5))
        sim.reset()
        initial_I = sim.observe()[2]
        for _ in range(5000):
            sim.step()
        final_I = sim.observe()[2]
        # I should grow when R0 > 1 and mu=0
        # Eventually all population moves to I
        assert final_I > initial_I, (
            f"I should accumulate with mu=0: {initial_I} -> {final_I}"
        )


class TestSEIProperties:
    """Tests for analytical properties and computed quantities."""

    def test_latent_period(self):
        """Latent period = 1 / sigma."""
        sim = SEISimulation(_make_config(sigma=0.2))
        assert sim.latent_period == pytest.approx(5.0)

    def test_infectious_duration(self):
        """Infectious duration = 1 / mu."""
        sim = SEISimulation(_make_config(mu=0.05))
        assert sim.infectious_duration == pytest.approx(20.0)

    def test_infectious_duration_infinite_no_mortality(self):
        """Infectious duration should be inf when mu = 0."""
        sim = SEISimulation(_make_config(mu=0.0))
        assert sim.infectious_duration == float("inf")

    def test_living_population_property(self):
        """living_population should return sum of compartments."""
        sim = SEISimulation(_make_config(N0=1000.0))
        sim.reset()
        assert sim.living_population == pytest.approx(1000.0)

    def test_disease_free_equilibrium(self):
        """DFE should be [N0, 0, 0]."""
        sim = SEISimulation(_make_config(N0=1000.0))
        dfe = sim.disease_free_equilibrium()
        np.testing.assert_allclose(dfe, [1000.0, 0.0, 0.0])

    def test_fraction_infected_total(self):
        """fraction_infected_total should return (E+I)/N0."""
        sim = SEISimulation(_make_config(N0=1000.0, E_0=10.0, I_0=5.0))
        sim.reset()
        frac = sim.fraction_infected_total()
        assert frac == pytest.approx(15.0 / 1000.0)

    def test_cumulative_deaths_initial(self):
        """Cumulative deaths should be 0 at start."""
        sim = SEISimulation(_make_config())
        sim.reset()
        assert sim.cumulative_deaths() == pytest.approx(0.0, abs=1e-10)

    def test_cumulative_deaths_positive_with_mortality(self):
        """Cumulative deaths should be positive after running with mu > 0."""
        sim = SEISimulation(_make_config(beta=0.5, mu=0.05))
        sim.reset()
        for _ in range(5000):
            sim.step()
        deaths = sim.cumulative_deaths()
        assert deaths > 0, f"Should have deaths with mu > 0: {deaths}"


class TestSEIJacobian:
    """Tests for the Jacobian matrix computation."""

    def test_jacobian_shape(self):
        """Jacobian should be 3x3."""
        sim = SEISimulation(_make_config())
        sim.reset()
        J = sim.jacobian()
        assert J.shape == (3, 3)

    def test_jacobian_at_dfe(self):
        """Jacobian at DFE should have known structure."""
        sim = SEISimulation(_make_config(beta=0.5, sigma=0.2, mu=0.05))
        dfe = sim.disease_free_equilibrium()
        J = sim.jacobian(dfe)
        # At DFE [N, 0, 0]: I=0, so force_of_infection vanishes
        # J[2,1] should be sigma
        assert J[2, 1] == pytest.approx(0.2)
        # J[2,2] should be -mu
        assert J[2, 2] == pytest.approx(-0.05)

    def test_jacobian_finite(self):
        """Jacobian values should be finite."""
        sim = SEISimulation(_make_config())
        sim.reset()
        for _ in range(100):
            sim.step()
        J = sim.jacobian()
        assert np.all(np.isfinite(J))

    def test_compute_divergence(self):
        """Divergence should be computable and finite."""
        sim = SEISimulation(_make_config())
        sim.reset()
        div = sim.compute_divergence()
        assert np.isfinite(div)


class TestSEIEndemicEquilibrium:
    """Tests for endemic equilibrium computation."""

    def test_endemic_equilibrium_exists_above_threshold(self):
        """Endemic equilibrium should exist when R0 > 1 and mu > 0."""
        sim = SEISimulation(_make_config(beta=0.5, mu=0.05))  # R0 = 10
        eq = sim.endemic_equilibrium()
        assert eq is not None
        assert eq.shape == (3,)
        assert np.all(eq >= 0), f"Negative equilibrium values: {eq}"

    def test_endemic_equilibrium_none_below_threshold(self):
        """Endemic equilibrium should be None when R0 < 1."""
        sim = SEISimulation(_make_config(beta=0.05, mu=0.1))  # R0 = 0.5
        eq = sim.endemic_equilibrium()
        assert eq is None

    def test_endemic_equilibrium_none_no_mortality(self):
        """Endemic equilibrium should be None when mu = 0."""
        sim = SEISimulation(_make_config(mu=0.0))
        eq = sim.endemic_equilibrium()
        assert eq is None

    def test_endemic_s_less_than_n0(self):
        """At endemic equilibrium, S* < N0."""
        sim = SEISimulation(_make_config(beta=0.5, mu=0.05))
        eq = sim.endemic_equilibrium()
        assert eq is not None
        assert eq[0] < sim.N0


class TestSEINormalization:
    """Tests for initial condition normalization."""

    def test_normalization_on_reset(self):
        """Initial conditions should normalize to N0."""
        sim = SEISimulation(_make_config(
            S_0=500.0, E_0=100.0, I_0=100.0, N0=1000.0,
        ))
        state = sim.reset()
        np.testing.assert_allclose(state.sum(), 1000.0, atol=1e-10)

    def test_normalization_preserves_ratios(self):
        """Normalization should preserve ratios between compartments."""
        sim = SEISimulation(_make_config(
            S_0=600.0, E_0=200.0, I_0=200.0, N0=1000.0,
        ))
        state = sim.reset()
        # Ratios: S:E:I = 3:1:1
        assert state[0] / state[1] == pytest.approx(3.0, rel=1e-10)
        assert state[1] / state[2] == pytest.approx(1.0, rel=1e-10)


class TestSEIRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_sweep_data(self):
        """Sweep data generation should produce valid arrays."""
        from simulating_anything.rediscovery.sei import (
            generate_sei_sweep_data,
        )
        data = generate_sei_sweep_data(n_samples=5, n_steps=500, dt=0.1)
        assert "beta" in data
        assert "sigma" in data
        assert "mu" in data
        assert "R0" in data
        assert "peak_E" in data
        assert "final_I" in data
        assert "final_S" in data
        assert "cumulative_deaths" in data
        assert "time_to_peak_E" in data
        assert len(data["beta"]) == 5
        assert np.all(data["R0"] > 0)
        assert np.all(np.isfinite(data["peak_E"]))

    def test_generate_ode_data(self):
        """ODE data generation should produce valid trajectory."""
        from simulating_anything.rediscovery.sei import (
            generate_sei_ode_data,
        )
        data = generate_sei_ode_data(n_steps=100, dt=0.1)
        assert data["states"].shape == (101, 3)
        # With mu > 0, population decreases; just check it's reasonable
        initial_pop = data["states"][0].sum()
        assert initial_pop == pytest.approx(1000.0, abs=1.0)

    def test_sweep_r0_range(self):
        """Sweep should cover both sub- and super-critical R0."""
        from simulating_anything.rediscovery.sei import (
            generate_sei_sweep_data,
        )
        data = generate_sei_sweep_data(n_samples=50, n_steps=500, dt=0.1)
        assert np.any(data["R0"] < 1.0), "Should have some R0 < 1 cases"
        assert np.any(data["R0"] > 1.0), "Should have some R0 > 1 cases"

    def test_ode_data_parameters(self):
        """ODE data should include ground truth parameters."""
        from simulating_anything.rediscovery.sei import (
            generate_sei_ode_data,
        )
        data = generate_sei_ode_data(n_steps=50, dt=0.1)
        assert data["beta"] == 0.5
        assert data["sigma"] == 0.2
        assert data["mu"] == 0.05
        assert data["N0"] == 1000.0

    def test_sweep_deaths_correlate_with_r0(self):
        """Higher R0 should produce more cumulative deaths."""
        from simulating_anything.rediscovery.sei import (
            generate_sei_sweep_data,
        )
        data = generate_sei_sweep_data(n_samples=30, n_steps=3000, dt=0.1)
        mask_high = data["R0"] > 5.0
        mask_low = (data["R0"] > 1.0) & (data["R0"] < 2.0)
        if mask_high.sum() > 0 and mask_low.sum() > 0:
            mean_deaths_high = data["cumulative_deaths"][mask_high].mean()
            mean_deaths_low = data["cumulative_deaths"][mask_low].mean()
            assert mean_deaths_high >= mean_deaths_low, (
                f"Higher R0 should have more deaths: "
                f"high={mean_deaths_high}, low={mean_deaths_low}"
            )
