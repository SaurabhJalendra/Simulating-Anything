"""Tests for the SIS Adaptive Behavior epidemic model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.sis_adaptive import SISAdaptiveSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_sim(
    dt: float = 0.1,
    n_steps: int = 5000,
    **param_overrides: float,
) -> SISAdaptiveSimulation:
    """Create an SISAdaptiveSimulation with default parameters.

    Default parameters: beta0=0.5, gamma=0.1, kappa=5.0, I_0=0.01.
    """
    params: dict[str, float] = {
        "beta0": 0.5,
        "gamma": 0.1,
        "kappa": 5.0,
        "I_0": 0.01,
    }
    params.update(param_overrides)
    config = SimulationConfig(
        domain=Domain.SIS_ADAPTIVE,  # Placeholder
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )
    return SISAdaptiveSimulation(config)


class TestSimulation:
    """Core simulation initialization and interface tests."""

    def test_init_default_params(self):
        """Default parameters should be assigned correctly."""
        sim = _make_sim()
        assert sim.beta0 == 0.5
        assert sim.gamma == 0.1
        assert sim.kappa == 5.0
        assert sim.I_0 == 0.01

    def test_init_custom_params(self):
        """Custom parameters should override defaults."""
        sim = _make_sim(beta0=0.8, gamma=0.2, kappa=10.0, I_0=0.05)
        assert sim.beta0 == 0.8
        assert sim.gamma == 0.2
        assert sim.kappa == 10.0
        assert sim.I_0 == 0.05

    def test_reset_shape(self):
        """State after reset has shape (1,) containing I."""
        sim = _make_sim()
        state = sim.reset()
        assert state.shape == (1,)

    def test_reset_value(self):
        """State after reset equals I_0."""
        sim = _make_sim(I_0=0.05)
        state = sim.reset()
        assert state[0] == pytest.approx(0.05)

    def test_reset_clamps_I(self):
        """I_0 should be clamped to [0, 1]."""
        sim_over = _make_sim(I_0=1.5)
        state = sim_over.reset()
        assert state[0] == pytest.approx(1.0)

        sim_under = _make_sim(I_0=-0.1)
        state = sim_under.reset()
        assert state[0] == pytest.approx(0.0)

    def test_step_advances_state(self):
        """A single step should change the state."""
        sim = _make_sim()
        s0 = sim.reset().copy()
        s1 = sim.step()
        assert not np.allclose(s0, s1)

    def test_observe_shape(self):
        """observe() returns [I, S, beta_eff] with shape (3,)."""
        sim = _make_sim()
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_observe_contents(self):
        """observe() returns consistent [I, S, beta_eff]."""
        sim = _make_sim(I_0=0.1)
        sim.reset()
        obs = sim.observe()
        inf, sus, beta_eff = obs
        assert inf == pytest.approx(0.1)
        assert sus == pytest.approx(0.9)
        assert beta_eff == pytest.approx(0.5 / (1.0 + 5.0 * 0.1))

    def test_run_trajectory_shape(self):
        """run(n) returns TrajectoryData with states shape (n+1, 1)."""
        n = 200
        sim = _make_sim(n_steps=n)
        traj = sim.run(n_steps=n)
        assert traj.states.shape == (n + 1, 1)

    def test_run_trajectory_metadata(self):
        """TrajectoryData should contain parameters from the config."""
        sim = _make_sim(n_steps=50)
        traj = sim.run(n_steps=50)
        assert "beta0" in traj.parameters
        assert "gamma" in traj.parameters
        assert traj.parameters["beta0"] == pytest.approx(0.5)

    def test_step_count_increments(self):
        """Step counter should increment with each step."""
        sim = _make_sim()
        sim.reset()
        assert sim._step_count == 0
        sim.step()
        assert sim._step_count == 1
        sim.step()
        assert sim._step_count == 2


class TestRK4:
    """Tests for RK4 integration correctness and constraints."""

    def test_I_stays_in_bounds(self):
        """I should remain in [0, 1] throughout the simulation."""
        sim = _make_sim(dt=0.1, I_0=0.5)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert 0.0 <= state[0] <= 1.0, f"I out of bounds: {state[0]}"

    def test_S_equals_one_minus_I(self):
        """S = 1 - I should hold at all times."""
        sim = _make_sim(I_0=0.1)
        sim.reset()
        for _ in range(100):
            sim.step()
            obs = sim.observe()
            inf, sus, _ = obs
            assert sus == pytest.approx(1.0 - inf, abs=1e-12)

    def test_population_conservation(self):
        """S + I = 1 should hold throughout (normalized population)."""
        sim = _make_sim(I_0=0.3)
        sim.reset()
        for _ in range(1000):
            sim.step()
            obs = sim.observe()
            total = obs[0] + obs[1]  # I + S
            assert total == pytest.approx(1.0, abs=1e-12)

    def test_small_dt_accuracy(self):
        """Smaller dt should give more accurate results."""
        # Run with two different dt values and compare final state
        sim_coarse = _make_sim(dt=0.1, n_steps=1000, I_0=0.1)
        traj_coarse = sim_coarse.run(n_steps=1000)

        sim_fine = _make_sim(dt=0.01, n_steps=10000, I_0=0.1)
        traj_fine = sim_fine.run(n_steps=10000)

        # Both should reach similar equilibrium
        I_coarse = traj_coarse.states[-1, 0]
        I_fine = traj_fine.states[-1, 0]
        assert abs(I_coarse - I_fine) < 0.01

    def test_deterministic(self):
        """Same configuration produces identical trajectories."""
        sim1 = _make_sim()
        sim1.reset()
        for _ in range(100):
            sim1.step()
        s1 = sim1.observe().copy()

        sim2 = _make_sim()
        sim2.reset()
        for _ in range(100):
            sim2.step()
        s2 = sim2.observe().copy()

        np.testing.assert_allclose(s1, s2)

    def test_I_decreases_below_threshold(self):
        """When R0 < 1, infection should decay to zero."""
        sim = _make_sim(beta0=0.05, gamma=0.1, kappa=0.0, I_0=0.5)
        assert sim.basic_R0 < 1.0
        sim.reset()
        for _ in range(5000):
            sim.step()
        assert sim._state[0] < 1e-6


class TestBehavioralResponse:
    """Tests for the behavioral beta function."""

    def test_beta_at_zero_prevalence(self):
        """beta(0) = beta0."""
        sim = _make_sim(beta0=0.5, kappa=5.0)
        assert sim.behavioral_beta(0.0) == pytest.approx(0.5)

    def test_beta_decreases_with_I(self):
        """beta(I) should decrease as I increases."""
        sim = _make_sim(beta0=0.5, kappa=5.0)
        betas = [sim.behavioral_beta(p) for p in np.linspace(0, 0.5, 10)]
        for j in range(len(betas) - 1):
            assert betas[j] > betas[j + 1]

    def test_beta_formula(self):
        """beta(I) = beta0 / (1 + kappa*I)."""
        sim = _make_sim(beta0=0.8, kappa=3.0)
        prev = 0.2
        expected = 0.8 / (1.0 + 3.0 * 0.2)
        assert sim.behavioral_beta(prev) == pytest.approx(expected)

    def test_beta_kappa_zero(self):
        """With kappa=0, beta(I) = beta0 for all I."""
        sim = _make_sim(beta0=0.5, kappa=0.0)
        for prev in [0.0, 0.1, 0.5, 0.9]:
            assert sim.behavioral_beta(prev) == pytest.approx(0.5)

    def test_beta_large_kappa(self):
        """With large kappa, beta(I) is strongly suppressed for I > 0."""
        sim = _make_sim(beta0=0.5, kappa=100.0)
        # At I=0.1, beta = 0.5 / (1 + 10) = 0.0455
        assert sim.behavioral_beta(0.1) == pytest.approx(0.5 / 11.0)

    def test_effective_R0_at_zero(self):
        """R_eff(0) = beta0 / gamma = basic R0."""
        sim = _make_sim(beta0=0.5, gamma=0.1, kappa=5.0)
        assert sim.effective_R0(0.0) == pytest.approx(5.0)

    def test_effective_R0_decreases_with_I(self):
        """R_eff should decrease as I increases."""
        sim = _make_sim(beta0=0.5, gamma=0.1, kappa=5.0)
        R0_vals = [sim.effective_R0(p) for p in np.linspace(0, 0.5, 10)]
        for j in range(len(R0_vals) - 1):
            assert R0_vals[j] > R0_vals[j + 1]


class TestEndemicEquilibrium:
    """Tests for endemic equilibrium computation and convergence."""

    def test_equilibrium_formula(self):
        """I* = (beta0 - gamma) / (beta0 + kappa*gamma)."""
        sim = _make_sim(beta0=0.5, gamma=0.1, kappa=5.0)
        I_star = sim.endemic_equilibrium()
        assert I_star is not None
        expected = (0.5 - 0.1) / (0.5 + 5.0 * 0.1)
        assert I_star == pytest.approx(expected)

    def test_equilibrium_none_below_threshold(self):
        """No endemic equilibrium when R0 <= 1."""
        sim = _make_sim(beta0=0.05, gamma=0.1, kappa=5.0)
        assert sim.basic_R0 < 1.0
        assert sim.endemic_equilibrium() is None

    def test_convergence_to_equilibrium(self):
        """Simulation should converge to I* for R0 > 1."""
        sim = _make_sim(beta0=0.5, gamma=0.1, kappa=5.0, I_0=0.01, dt=0.1)
        I_star = sim.endemic_equilibrium()
        assert I_star is not None

        sim.reset()
        for _ in range(50000):
            sim.step()
        I_final = sim._state[0]
        assert I_final == pytest.approx(I_star, abs=0.005)

    def test_convergence_from_above(self):
        """Starting above I*, simulation should still converge."""
        sim = _make_sim(beta0=0.5, gamma=0.1, kappa=5.0, I_0=0.9, dt=0.1)
        I_star = sim.endemic_equilibrium()
        assert I_star is not None

        sim.reset()
        for _ in range(50000):
            sim.step()
        I_final = sim._state[0]
        assert I_final == pytest.approx(I_star, abs=0.005)

    def test_equilibrium_varies_with_kappa(self):
        """Different kappa values give different equilibria."""
        I_stars = []
        for kappa in [0.0, 2.0, 5.0, 10.0]:
            sim = _make_sim(beta0=0.5, gamma=0.1, kappa=kappa)
            I_star = sim.endemic_equilibrium()
            assert I_star is not None
            I_stars.append(I_star)
        # Higher kappa should give lower I*
        for i in range(len(I_stars) - 1):
            assert I_stars[i] > I_stars[i + 1]

    def test_equilibrium_at_boundary(self):
        """When beta0 = gamma, R0 = 1, no endemic equilibrium."""
        sim = _make_sim(beta0=0.1, gamma=0.1, kappa=5.0)
        assert sim.basic_R0 == pytest.approx(1.0)
        assert sim.endemic_equilibrium() is None


class TestKappaEffect:
    """Tests for the effect of behavioral sensitivity kappa."""

    def test_higher_kappa_lower_endemic(self):
        """Higher kappa should produce lower endemic prevalence."""
        I_stars = []
        for kappa in [0.0, 1.0, 5.0, 10.0, 20.0]:
            sim = _make_sim(beta0=0.5, gamma=0.1, kappa=kappa)
            I_star = sim.endemic_equilibrium()
            assert I_star is not None
            I_stars.append(I_star)
        for i in range(len(I_stars) - 1):
            assert I_stars[i] > I_stars[i + 1]

    def test_kappa_zero_standard_sis(self):
        """kappa=0 should give standard SIS equilibrium."""
        sim = _make_sim(beta0=0.5, gamma=0.1, kappa=0.0)
        I_star = sim.endemic_equilibrium()
        assert I_star is not None
        # Standard SIS: I* = 1 - gamma/beta0
        expected = 1.0 - 0.1 / 0.5
        assert I_star == pytest.approx(expected)

    def test_kappa_large_suppresses_strongly(self):
        """Very large kappa should reduce I* close to zero."""
        sim = _make_sim(beta0=0.5, gamma=0.1, kappa=100.0)
        I_star = sim.endemic_equilibrium()
        assert I_star is not None
        assert I_star < 0.05

    def test_kappa_does_not_change_threshold(self):
        """The epidemic threshold R0 = beta0/gamma is independent of kappa."""
        for kappa in [0.0, 5.0, 50.0]:
            sim = _make_sim(beta0=0.5, gamma=0.1, kappa=kappa)
            assert sim.basic_R0 == pytest.approx(5.0)

    def test_simulation_kappa_effect(self):
        """Simulated endemic level should decrease with kappa."""
        final_I_values = []
        for kappa in [0.0, 5.0, 15.0]:
            sim = _make_sim(
                beta0=0.5, gamma=0.1, kappa=kappa,
                I_0=0.1, dt=0.1,
            )
            sim.reset()
            for _ in range(30000):
                sim.step()
            final_I_values.append(float(sim._state[0]))
        for i in range(len(final_I_values) - 1):
            assert final_I_values[i] > final_I_values[i + 1]


class TestThreshold:
    """Tests for the epidemic threshold R0 = beta0/gamma."""

    def test_basic_R0(self):
        """R0 = beta0 / gamma."""
        sim = _make_sim(beta0=0.5, gamma=0.1)
        assert sim.basic_R0 == pytest.approx(5.0)

    def test_R0_above_one_grows(self):
        """When R0 > 1, infection should grow from small initial value."""
        sim = _make_sim(beta0=0.5, gamma=0.1, kappa=5.0, I_0=0.001)
        assert sim.basic_R0 > 1.0
        sim.reset()
        I_initial = sim._state[0]
        for _ in range(100):
            sim.step()
        I_after = sim._state[0]
        assert I_after > I_initial

    def test_R0_below_one_decays(self):
        """When R0 < 1, infection should decay to zero."""
        sim = _make_sim(beta0=0.05, gamma=0.1, kappa=5.0, I_0=0.3)
        assert sim.basic_R0 < 1.0
        sim.reset()
        for _ in range(10000):
            sim.step()
        assert sim._state[0] < 1e-6

    def test_R0_exactly_one(self):
        """When R0 = 1, infection should not grow."""
        sim = _make_sim(beta0=0.1, gamma=0.1, kappa=0.0, I_0=0.01)
        assert sim.basic_R0 == pytest.approx(1.0)
        sim.reset()
        for _ in range(10000):
            sim.step()
        # Should decay or stay near zero
        assert sim._state[0] < 0.02

    def test_effective_R0_at_equilibrium(self):
        """At endemic equilibrium, R_eff should equal 1."""
        sim = _make_sim(beta0=0.5, gamma=0.1, kappa=5.0)
        I_star = sim.endemic_equilibrium()
        assert I_star is not None
        R_eff = sim.effective_R0(I_star)
        assert R_eff == pytest.approx(1.0, abs=1e-10)


class TestNoAdaptation:
    """Tests verifying kappa=0 matches standard SIS model."""

    def test_standard_sis_equilibrium(self):
        """kappa=0: I* = 1 - gamma/beta0."""
        sim = _make_sim(beta0=0.4, gamma=0.1, kappa=0.0)
        I_star = sim.endemic_equilibrium()
        assert I_star is not None
        expected = 1.0 - 0.1 / 0.4
        assert I_star == pytest.approx(expected)

    def test_standard_sis_convergence(self):
        """kappa=0: simulation converges to standard SIS equilibrium."""
        sim = _make_sim(
            beta0=0.4, gamma=0.1, kappa=0.0,
            I_0=0.01, dt=0.1,
        )
        I_star = 1.0 - 0.1 / 0.4
        sim.reset()
        for _ in range(30000):
            sim.step()
        assert sim._state[0] == pytest.approx(I_star, abs=0.01)

    def test_standard_sis_beta_constant(self):
        """kappa=0: beta_eff = beta0 throughout simulation."""
        sim = _make_sim(beta0=0.5, kappa=0.0, I_0=0.1)
        sim.reset()
        for _ in range(100):
            sim.step()
            obs = sim.observe()
            assert obs[2] == pytest.approx(0.5)

    def test_standard_sis_no_epidemic(self):
        """kappa=0, R0 < 1: infection decays to zero."""
        sim = _make_sim(beta0=0.05, gamma=0.1, kappa=0.0, I_0=0.3)
        sim.reset()
        for _ in range(10000):
            sim.step()
        assert sim._state[0] < 1e-6


class TestRediscovery:
    """Tests for the rediscovery data generation functions."""

    def test_kappa_sweep_data_shape(self):
        """Kappa sweep should return arrays of expected length."""
        from simulating_anything.rediscovery.sis_adaptive import (
            generate_kappa_sweep_data,
        )

        data = generate_kappa_sweep_data(
            n_kappa=5, n_steps=1000, dt=0.1,
        )
        assert "kappa" in data
        assert "measured_I_star" in data
        assert "theory_I_star" in data
        assert len(data["kappa"]) == 5
        assert len(data["measured_I_star"]) == 5
        assert len(data["theory_I_star"]) == 5

    def test_kappa_sweep_non_negative(self):
        """Measured I* values should be non-negative."""
        from simulating_anything.rediscovery.sis_adaptive import (
            generate_kappa_sweep_data,
        )

        data = generate_kappa_sweep_data(
            n_kappa=5, n_steps=1000, dt=0.1,
        )
        assert np.all(data["measured_I_star"] >= 0)

    def test_beta_sweep_data_shape(self):
        """Beta sweep should return arrays of expected length."""
        from simulating_anything.rediscovery.sis_adaptive import (
            generate_beta_sweep_data,
        )

        data = generate_beta_sweep_data(
            n_beta=5, n_steps=1000, dt=0.1,
        )
        assert "beta0" in data
        assert "R0" in data
        assert "measured_I_star" in data
        assert "theory_I_star" in data
        assert len(data["beta0"]) == 5
        assert len(data["R0"]) == 5

    def test_beta_sweep_R0_values(self):
        """R0 = beta0 / gamma should be computed correctly."""
        from simulating_anything.rediscovery.sis_adaptive import (
            generate_beta_sweep_data,
        )

        data = generate_beta_sweep_data(
            n_beta=5, gamma=0.1, n_steps=100, dt=0.1,
        )
        expected_R0 = data["beta0"] / 0.1
        np.testing.assert_allclose(data["R0"], expected_R0)

    def test_run_rediscovery_structure(self):
        """run_sis_adaptive_rediscovery returns a dict with required keys."""
        import tempfile

        from simulating_anything.rediscovery.sis_adaptive import (
            run_sis_adaptive_rediscovery,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_sis_adaptive_rediscovery(
                output_dir=tmpdir,
                n_iterations=5,
            )

        assert isinstance(results, dict)
        assert results["domain"] == "sis_adaptive"
        assert "targets" in results
        assert "kappa_sweep" in results
        assert "beta_sweep" in results
        assert "formula_verification" in results

    def test_theory_matches_formula(self):
        """Theory I* in sweep data should match the analytical formula."""
        from simulating_anything.rediscovery.sis_adaptive import (
            generate_kappa_sweep_data,
        )

        beta0, gamma = 0.5, 0.1
        data = generate_kappa_sweep_data(
            n_kappa=10, beta0=beta0, gamma=gamma,
            n_steps=100, dt=0.1,
        )
        for i, kappa in enumerate(data["kappa"]):
            expected = (beta0 - gamma) / (beta0 + kappa * gamma)
            assert data["theory_I_star"][i] == pytest.approx(expected, rel=1e-10)
