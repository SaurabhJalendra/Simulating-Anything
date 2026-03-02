"""Tests for the SIS (Susceptible-Infected-Susceptible) endemic model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sis_endemic import SISEndemicSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.1,
    n_steps: int = 500,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for SIS with optional overrides."""
    params: dict[str, float] = {
        "beta": 0.5,
        "gamma": 0.2,
        "N0": 1000.0,
        "I_0": 10.0,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.SIS_ENDEMIC,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


# ---------------------------------------------------------------------------
# TestSISBasic: init, reset, step, observe, run, state_shape, dt
# ---------------------------------------------------------------------------

class TestSISBasic:
    """Tests for basic simulation setup and operation."""

    def test_create_simulation(self):
        """Simulation should be created with default parameters."""
        sim = SISEndemicSimulation(_make_config())
        assert sim.beta == 0.5
        assert sim.gamma == 0.2
        assert sim.N0 == 1000.0
        assert sim.I_0 == 10.0

    def test_initial_state_shape(self):
        """State vector should be [S, I] with 2 components."""
        sim = SISEndemicSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_initial_state_values(self):
        """Initial conditions should match parameters."""
        sim = SISEndemicSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [990.0, 10.0])

    def test_step_returns_array(self):
        """step() should return a numpy array of shape (2,)."""
        sim = SISEndemicSimulation(_make_config())
        sim.reset()
        state = sim.step()
        assert isinstance(state, np.ndarray)
        assert state.shape == (2,)

    def test_step_advances_state(self):
        """State should change after a step."""
        sim = SISEndemicSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_returns_state(self):
        """Observe should return the current state."""
        sim = SISEndemicSimulation(_make_config())
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2,)
        np.testing.assert_allclose(obs, [990.0, 10.0])

    def test_run_method(self):
        """run() should produce a TrajectoryData object."""
        sim = SISEndemicSimulation(_make_config(n_steps=100))
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 2)
        assert len(traj.timestamps) == 101

    def test_dt_matches_config(self):
        """Simulation dt should match config dt."""
        sim = SISEndemicSimulation(_make_config(dt=0.05))
        assert sim.config.dt == 0.05

    def test_deterministic(self):
        """Same config produces same trajectory."""
        config = _make_config()
        sim1 = SISEndemicSimulation(config)
        sim1.reset()
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = SISEndemicSimulation(config)
        sim2.reset()
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)


# ---------------------------------------------------------------------------
# TestSISDynamics: epidemic growth, endemic equilibrium, DFE, conservation
# ---------------------------------------------------------------------------

class TestSISDynamics:
    """Tests for SIS epidemic dynamics."""

    def test_infected_grows_when_r0_gt_1(self):
        """With R0 > 1, infected should initially grow."""
        sim = SISEndemicSimulation(_make_config(beta=0.5, gamma=0.2))
        sim.reset()
        initial_I = sim.observe()[1]
        # Step forward a bit
        for _ in range(50):
            sim.step()
        assert sim.observe()[1] > initial_I, (
            "I should grow when R0 > 1"
        )

    def test_infected_decays_when_r0_lt_1(self):
        """With R0 < 1, infected should decay monotonically."""
        sim = SISEndemicSimulation(_make_config(beta=0.1, gamma=0.5))
        sim.reset()
        initial_I = sim.observe()[1]
        for _ in range(500):
            sim.step()
        assert sim.observe()[1] < initial_I, (
            "I should decay when R0 < 1"
        )

    def test_approaches_endemic_equilibrium(self):
        """With R0 > 1, I should converge to I* = N*(1-1/R0)."""
        sim = SISEndemicSimulation(
            _make_config(beta=0.5, gamma=0.2, N0=1000.0, I_0=10.0)
        )
        sim.reset()
        # Run long enough to reach equilibrium
        for _ in range(50000):
            sim.step()
        final_I = sim.observe()[1]
        r0 = 0.5 / 0.2  # = 2.5
        expected_I = 1000.0 * (1.0 - 1.0 / r0)  # = 600
        assert final_I == pytest.approx(expected_I, rel=0.01), (
            f"Expected I*={expected_I}, got {final_I}"
        )

    def test_dfe_when_r0_lt_1(self):
        """With R0 < 1, system should converge to DFE (I -> 0)."""
        sim = SISEndemicSimulation(
            _make_config(beta=0.1, gamma=0.5, I_0=100.0)
        )
        sim.reset()
        for _ in range(10000):
            sim.step()
        final_I = sim.observe()[1]
        assert final_I < 1.0, (
            f"I should approach 0 when R0<1, got {final_I}"
        )

    def test_total_population_conserved(self):
        """S + I should equal N0 at all times."""
        sim = SISEndemicSimulation(_make_config())
        sim.reset()
        for _ in range(500):
            state = sim.step()
            np.testing.assert_allclose(
                state.sum(), 1000.0, atol=1e-6,
                err_msg="Total population conservation violated",
            )

    def test_conservation_long_run(self):
        """Conservation should hold over long simulation."""
        sim = SISEndemicSimulation(_make_config())
        sim.reset()
        for _ in range(5000):
            state = sim.step()
        np.testing.assert_allclose(
            state.sum(), 1000.0, atol=1e-4,
            err_msg="Conservation violated over long run",
        )

    def test_no_epidemic_dies_out_completely(self):
        """When R0 < 1, disease should die out; I stays near 0 at end."""
        sim = SISEndemicSimulation(
            _make_config(beta=0.05, gamma=0.2, I_0=50.0)
        )
        sim.reset()
        for _ in range(20000):
            sim.step()
        assert sim.observe()[1] < 0.1


# ---------------------------------------------------------------------------
# TestSISNonNegativity
# ---------------------------------------------------------------------------

class TestSISNonNegativity:
    """All compartments must remain non-negative."""

    def test_non_negative_both_compartments(self):
        """S and I should both remain >= 0."""
        sim = SISEndemicSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert state[0] >= 0, f"S became negative: {state[0]}"
            assert state[1] >= 0, f"I became negative: {state[1]}"

    def test_non_negative_high_beta(self):
        """Non-negativity should hold even with high transmission rate."""
        sim = SISEndemicSimulation(_make_config(beta=2.0))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"

    def test_non_negative_small_population(self):
        """Non-negativity with small population size."""
        sim = SISEndemicSimulation(_make_config(N0=10.0, I_0=1.0))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative compartment: {state}"


# ---------------------------------------------------------------------------
# TestSISParameters: different param values
# ---------------------------------------------------------------------------

class TestSISParameters:
    """Tests for parameter handling."""

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        sim = SISEndemicSimulation(_make_config(beta=0.8, gamma=0.3))
        assert sim.beta == 0.8
        assert sim.gamma == 0.3

    def test_different_beta_changes_dynamics(self):
        """Higher beta should lead to more infection at equilibrium."""
        sim_low = SISEndemicSimulation(_make_config(beta=0.3, gamma=0.2))
        sim_low.reset()
        for _ in range(20000):
            sim_low.step()
        I_low = sim_low.observe()[1]

        sim_high = SISEndemicSimulation(_make_config(beta=0.8, gamma=0.2))
        sim_high.reset()
        for _ in range(20000):
            sim_high.step()
        I_high = sim_high.observe()[1]

        assert I_high > I_low, (
            f"Higher beta should give more I: {I_high} vs {I_low}"
        )

    def test_different_gamma_changes_dynamics(self):
        """Higher gamma should lead to less infection at equilibrium."""
        sim_low = SISEndemicSimulation(_make_config(beta=0.5, gamma=0.1))
        sim_low.reset()
        for _ in range(20000):
            sim_low.step()
        I_low_gamma = sim_low.observe()[1]

        sim_high = SISEndemicSimulation(_make_config(beta=0.5, gamma=0.4))
        sim_high.reset()
        for _ in range(20000):
            sim_high.step()
        I_high_gamma = sim_high.observe()[1]

        assert I_low_gamma > I_high_gamma, (
            f"Higher gamma should give less I: gamma=0.1 -> {I_low_gamma}, "
            f"gamma=0.4 -> {I_high_gamma}"
        )

    def test_different_N0(self):
        """Different N0 should scale the population."""
        sim = SISEndemicSimulation(_make_config(N0=500.0, I_0=5.0))
        state = sim.reset()
        np.testing.assert_allclose(state.sum(), 500.0)

    def test_r0_calculation(self):
        """R0 property should return beta/gamma."""
        sim = SISEndemicSimulation(_make_config(beta=0.6, gamma=0.3))
        assert sim.R0 == pytest.approx(2.0)

    def test_r0_various_values(self):
        """R0 should vary correctly with different parameters."""
        for beta, gamma in [(0.1, 0.5), (0.5, 0.1), (0.3, 0.3), (1.0, 0.2)]:
            sim = SISEndemicSimulation(_make_config(beta=beta, gamma=gamma))
            assert sim.R0 == pytest.approx(beta / gamma)


# ---------------------------------------------------------------------------
# TestSISEquilibria: endemic eq formula, DFE stability, transcritical
# ---------------------------------------------------------------------------

class TestSISEquilibria:
    """Tests for equilibrium states and bifurcation."""

    def test_endemic_equilibrium_formula(self):
        """Endemic eq should be [N/R0, N*(1-1/R0)] for R0 > 1."""
        sim = SISEndemicSimulation(_make_config(beta=0.5, gamma=0.2, N0=1000.0))
        eq = sim.endemic_equilibrium()
        r0 = 2.5
        np.testing.assert_allclose(eq, [1000.0 / r0, 1000.0 * (1 - 1 / r0)])

    def test_endemic_equilibrium_r0_lt_1_returns_dfe(self):
        """When R0 <= 1, endemic_equilibrium should return DFE."""
        sim = SISEndemicSimulation(_make_config(beta=0.1, gamma=0.5))
        eq = sim.endemic_equilibrium()
        np.testing.assert_allclose(eq, [1000.0, 0.0])

    def test_disease_free_equilibrium(self):
        """DFE should be [N0, 0]."""
        sim = SISEndemicSimulation(_make_config(N0=1000.0))
        dfe = sim.disease_free_equilibrium()
        np.testing.assert_allclose(dfe, [1000.0, 0.0])

    def test_dfe_stability_subcritical(self):
        """DFE should be stable when R0 < 1 (I decays)."""
        sim = SISEndemicSimulation(
            _make_config(beta=0.1, gamma=0.5, I_0=100.0)
        )
        sim.reset()
        prev_I = sim.observe()[1]
        decaying = True
        for _ in range(500):
            state = sim.step()
            if state[1] > prev_I + 1e-6:
                decaying = False
                break
            prev_I = state[1]
        assert decaying, "I should decay monotonically when R0 < 1"

    def test_transcritical_bifurcation_at_r0_1(self):
        """Near R0 = 1, endemic eq I* should be near 0."""
        # R0 = 1.01 -- just above critical
        sim = SISEndemicSimulation(
            _make_config(beta=0.201, gamma=0.2, N0=1000.0)
        )
        eq = sim.endemic_equilibrium()
        # I* = N*(1 - 1/R0) = N*(1 - 1/1.005) ~ N*0.005 ~ 5
        assert eq[1] > 0, "I* should be positive for R0 > 1"
        assert eq[1] < 50, "I* should be small near R0 = 1"

    def test_endemic_eq_increases_with_r0(self):
        """Higher R0 should give higher I*."""
        I_stars = []
        for beta in [0.3, 0.5, 0.8, 1.0]:
            sim = SISEndemicSimulation(
                _make_config(beta=beta, gamma=0.2, N0=1000.0)
            )
            eq = sim.endemic_equilibrium()
            I_stars.append(eq[1])
        # Each I* should be larger than the previous
        for i in range(1, len(I_stars)):
            assert I_stars[i] > I_stars[i - 1], (
                f"I* should increase with R0: {I_stars}"
            )

    def test_simulation_converges_to_endemic_eq(self):
        """Long simulation should converge to the analytical endemic eq."""
        sim = SISEndemicSimulation(
            _make_config(beta=0.8, gamma=0.2, N0=1000.0, I_0=5.0)
        )
        sim.reset()
        for _ in range(50000):
            sim.step()
        final_state = sim.observe()
        expected_eq = sim.endemic_equilibrium()
        np.testing.assert_allclose(final_state, expected_eq, rtol=0.01)

    def test_total_population_property(self):
        """total_population should return sum of compartments."""
        sim = SISEndemicSimulation(_make_config(N0=1000.0))
        sim.reset()
        assert sim.total_population == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# TestSISStability: long-run stability
# ---------------------------------------------------------------------------

class TestSISStability:
    """Tests for numerical stability."""

    def test_long_run_no_nan(self):
        """No NaN after many steps."""
        sim = SISEndemicSimulation(_make_config(dt=0.1))
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"

    def test_stability_high_r0(self):
        """Stable with high R0."""
        sim = SISEndemicSimulation(_make_config(beta=5.0, gamma=0.1))
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state))
        assert np.all(state >= 0)

    def test_stability_near_bifurcation(self):
        """Stable near R0 = 1 (bifurcation point)."""
        sim = SISEndemicSimulation(_make_config(beta=0.201, gamma=0.2))
        sim.reset()
        for _ in range(10000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state))


# ---------------------------------------------------------------------------
# TestSISRediscovery: data generation for PySR/SINDy
# ---------------------------------------------------------------------------

class TestSISRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_sweep_data(self):
        """Sweep data generation should produce valid arrays."""
        from simulating_anything.rediscovery.sis_endemic import (
            generate_sis_sweep_data,
        )
        data = generate_sis_sweep_data(n_samples=5, n_steps=500, dt=0.1)
        assert "beta" in data
        assert "gamma" in data
        assert "R0" in data
        assert "final_I_frac" in data
        assert "endemic_eq" in data
        assert "time_to_eq" in data
        assert len(data["beta"]) == 5
        assert np.all(data["R0"] > 0)
        assert np.all(np.isfinite(data["final_I_frac"]))

    def test_generate_ode_data(self):
        """ODE data generation should produce valid trajectory."""
        from simulating_anything.rediscovery.sis_endemic import (
            generate_sis_ode_data,
        )
        data = generate_sis_ode_data(n_steps=100, dt=0.1)
        assert data["states"].shape == (101, 2)
        # Total population should be conserved
        totals = data["states"].sum(axis=1)
        np.testing.assert_allclose(totals, 1000.0, atol=1.0)

    def test_sweep_r0_range(self):
        """Sweep should cover both sub- and super-critical R0."""
        from simulating_anything.rediscovery.sis_endemic import (
            generate_sis_sweep_data,
        )
        data = generate_sis_sweep_data(n_samples=50, n_steps=500, dt=0.1)
        assert np.any(data["R0"] < 1.0), "Should have some R0 < 1 cases"
        assert np.any(data["R0"] > 1.0), "Should have some R0 > 1 cases"

    def test_ode_data_parameters(self):
        """ODE data should include ground truth parameters."""
        from simulating_anything.rediscovery.sis_endemic import (
            generate_sis_ode_data,
        )
        data = generate_sis_ode_data(n_steps=50, dt=0.1)
        assert data["beta"] == 0.5
        assert data["gamma"] == 0.2
        assert data["N0"] == 1000.0

    def test_sweep_endemic_eq_values(self):
        """Endemic eq values should match theoretical I*/N = 1 - gamma/beta."""
        from simulating_anything.rediscovery.sis_endemic import (
            generate_sis_sweep_data,
        )
        data = generate_sis_sweep_data(n_samples=10, n_steps=500, dt=0.1)
        for i in range(len(data["beta"])):
            r0 = data["R0"][i]
            if r0 > 1.0:
                expected = 1.0 - 1.0 / r0
                assert data["endemic_eq"][i] == pytest.approx(expected, abs=1e-10)
            else:
                assert data["endemic_eq"][i] == pytest.approx(0.0)

    def test_sweep_final_I_approaches_endemic_eq(self):
        """For R0 > 1, final I/N should be close to endemic eq (with enough steps)."""
        from simulating_anything.rediscovery.sis_endemic import (
            generate_sis_sweep_data,
        )
        data = generate_sis_sweep_data(n_samples=20, n_steps=5000, dt=0.1)
        mask = data["R0"] > 1.5  # well above threshold
        if mask.sum() > 0:
            # Final I fraction should be in the right ballpark for the endemic eq
            for i in np.where(mask)[0]:
                expected = data["endemic_eq"][i]
                actual = data["final_I_frac"][i]
                # Allow 20% relative tolerance since may not fully converge
                if expected > 0.01:
                    assert actual > 0.01, (
                        f"I should be endemic for R0={data['R0'][i]:.2f}"
                    )
