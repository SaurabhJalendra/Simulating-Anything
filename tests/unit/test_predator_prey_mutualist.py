"""Tests for the Predator-Prey-Mutualist 3-species model."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.predator_prey_mutualist import (
    PredatorPreyMutualistSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    m: float = 0.5,
    dt: float = 0.01,
    **overrides: float,
) -> SimulationConfig:
    params = {
        "r": 1.0, "K": 10.0, "a": 1.0, "b": 0.1,
        "m": m, "n": 0.2, "d": 0.4, "e": 0.6,
        "s": 0.8, "C": 8.0, "p": 0.3,
        "x_0": 5.0, "y_0": 2.0, "z_0": 3.0,
    }
    params.update(overrides)
    return SimulationConfig(
        domain=Domain.PREDATOR_PREY_MUTUALIST,
        dt=dt,
        n_steps=1000,
        parameters=params,
    )


class TestPredatorPreyMutualistSimulation:
    """Tests for the simulation engine."""

    def test_initial_state_shape(self):
        """State should be a 3D vector [x, y, z]."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_values(self):
        """State should match initial conditions."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        state = sim.reset()
        np.testing.assert_allclose(state, [5.0, 2.0, 3.0])

    def test_step_advances_state(self):
        """One step should change the state."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_non_negative_populations(self):
        """All populations must remain non-negative."""
        sim = PredatorPreyMutualistSimulation(_make_config(dt=0.01))
        sim.reset()
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert np.all(state >= 0.0), f"Negative population: {state}"

    def test_trajectory_bounded(self):
        """Populations should remain bounded and finite."""
        sim = PredatorPreyMutualistSimulation(_make_config(dt=0.005))
        sim.reset()
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            assert np.all(state < 100), f"State diverged: {state}"

    def test_observe_returns_current_state(self):
        """observe() should return the same state as step()."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        sim.reset()
        step_result = sim.step()
        obs_result = sim.observe()
        np.testing.assert_array_equal(step_result, obs_result)

    def test_run_trajectory(self):
        """run() should produce a full trajectory."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_total_population_property(self):
        """total_population should return sum of all species."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        sim.reset()
        total = sim.total_population
        assert total == pytest.approx(10.0)  # 5 + 2 + 3

    def test_is_coexisting_property(self):
        """is_coexisting should be True when all species are positive."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        sim.reset()
        assert sim.is_coexisting

    def test_is_coexisting_false_for_none_state(self):
        """is_coexisting should be False before reset."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        assert not sim.is_coexisting

    def test_total_population_zero_before_reset(self):
        """total_population should be 0 before reset."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        assert sim.total_population == 0.0


class TestDerivatives:
    """Tests for the ODE right-hand side."""

    def test_zero_state_gives_zero_derivatives(self):
        """At origin, all derivatives should be zero."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        dy = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(dy, [0.0, 0.0, 0.0])

    def test_prey_only_logistic(self):
        """With no predator or mutualist, prey follows logistic growth."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        x = 5.0
        dy = sim._derivatives(np.array([x, 0.0, 0.0]))
        # dx/dt = r*x*(1 - x/K) = 1.0*5.0*(1 - 5/10) = 2.5
        assert dy[0] == pytest.approx(2.5)
        # dy/dt = -d*y = 0 (no predator)
        assert dy[1] == pytest.approx(0.0)
        # dz/dt = 0 (no mutualist)
        assert dy[2] == pytest.approx(0.0)

    def test_holling_type_ii_response(self):
        """Functional response a*x/(1+b*x) with correct saturation."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        # At x=10, a=1, b=0.1: functional_response = 10/(1+1) = 5.0
        x, y, z = 10.0, 1.0, 0.0
        dy = sim._derivatives(np.array([x, y, z]))
        # dx/dt = 1*10*(1-10/10) - 5*1 + 0 = 0 - 5 = -5
        assert dy[0] == pytest.approx(-5.0)
        # dy/dt = -0.4*1 + 0.6*5*1 = -0.4 + 3.0 = 2.6
        assert dy[1] == pytest.approx(2.6)

    def test_mutualism_term(self):
        """Mutualism term m*x*z/(1+n*z) increases prey growth."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        # At x=5, y=0, z=4: mutualism = 0.5*5*4/(1+0.2*4) = 10/(1.8) ~ 5.556
        x, y, z = 5.0, 0.0, 4.0
        dy = sim._derivatives(np.array([x, y, z]))
        logistic = 1.0 * 5.0 * (1.0 - 5.0 / 10.0)  # 2.5
        mutualism_prey = 0.5 * 5.0 * 4.0 / (1.0 + 0.2 * 4.0)  # 5.556
        expected_dx = logistic + mutualism_prey
        assert dy[0] == pytest.approx(expected_dx, rel=1e-6)


class TestParameterVariation:
    """Tests that parameter changes produce expected effects."""

    def test_higher_mutualism_increases_prey(self):
        """Higher m should increase steady-state prey population."""
        means = []
        for m_val in [0.0, 1.0]:
            sim = PredatorPreyMutualistSimulation(_make_config(m=m_val, dt=0.005))
            sim.reset()
            # Run to steady state
            for _ in range(20000):
                sim.step()
            # Measure mean over more steps
            prey_vals = []
            for _ in range(5000):
                sim.step()
                prey_vals.append(sim.observe()[0])
            means.append(np.mean(prey_vals))
        # With mutualism, prey should have higher mean
        assert means[1] > means[0], (
            f"Expected higher prey with m=1.0 ({means[1]:.3f}) "
            f"than m=0.0 ({means[0]:.3f})"
        )

    def test_no_mutualism_reverts_to_predator_prey(self):
        """With m=0 and p=0, mutualist decouples from predator-prey."""
        config = _make_config(m=0.0, p=0.0, z_0=3.0)
        sim = PredatorPreyMutualistSimulation(config)
        sim.reset()
        for _ in range(10000):
            sim.step()
        z_final = sim.observe()[2]
        # Mutualist should converge to its own carrying capacity C=8
        assert z_final == pytest.approx(8.0, rel=0.05)

    def test_predator_death_rate(self):
        """Very high predator death rate should drive predator to extinction."""
        config = _make_config(d=10.0, dt=0.005)
        sim = PredatorPreyMutualistSimulation(config)
        sim.reset()
        for _ in range(20000):
            sim.step()
        y_final = sim.observe()[1]
        assert y_final < 0.01, f"Predator should be extinct: y={y_final}"

    def test_different_initial_conditions(self):
        """Different initial conditions should produce different trajectories."""
        config1 = _make_config(x_0=2.0, y_0=1.0, z_0=1.0)
        config2 = _make_config(x_0=8.0, y_0=4.0, z_0=6.0)

        sim1 = PredatorPreyMutualistSimulation(config1)
        sim2 = PredatorPreyMutualistSimulation(config2)
        sim1.reset()
        sim2.reset()

        for _ in range(100):
            sim1.step()
            sim2.step()

        assert not np.allclose(sim1.observe(), sim2.observe(), atol=0.1)


class TestEquilibriumAnalysis:
    """Tests for equilibrium finding and stability analysis."""

    def test_find_equilibria_returns_list(self):
        """find_equilibria should return a list of numpy arrays."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        sim.reset()
        equilibria = sim.find_equilibria(n_initial=10)
        assert isinstance(equilibria, list)
        assert len(equilibria) > 0
        for eq in equilibria:
            assert isinstance(eq, np.ndarray)
            assert eq.shape == (3,)

    def test_trivial_equilibrium_found(self):
        """The origin (0,0,0) should always be an equilibrium."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        sim.reset()
        equilibria = sim.find_equilibria(n_initial=5)
        found_origin = any(
            np.allclose(eq, [0.0, 0.0, 0.0], atol=1e-6) for eq in equilibria
        )
        assert found_origin, "Origin equilibrium not found"

    def test_equilibria_are_fixed_points(self):
        """At each equilibrium, derivatives should be approximately zero."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        sim.reset()
        equilibria = sim.find_equilibria(n_initial=15)
        for eq in equilibria:
            dy = sim._derivatives(eq)
            np.testing.assert_allclose(
                dy, [0.0, 0.0, 0.0], atol=1e-6,
                err_msg=f"Not a fixed point: eq={eq}, dy={dy}",
            )

    def test_equilibria_non_negative(self):
        """All equilibrium components should be non-negative."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        sim.reset()
        equilibria = sim.find_equilibria(n_initial=15)
        for eq in equilibria:
            assert np.all(eq >= -1e-8), f"Negative equilibrium: {eq}"

    def test_stability_eigenvalues_shape(self):
        """stability_eigenvalues should return 3 eigenvalues."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        sim.reset()
        equilibria = sim.find_equilibria(n_initial=5)
        for eq in equilibria:
            eigs = sim.stability_eigenvalues(eq)
            assert eigs.shape == (3,)

    def test_origin_is_unstable(self):
        """The origin should be unstable (prey and mutualist grow)."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        sim.reset()
        eigs = sim.stability_eigenvalues(np.array([0.0, 0.0, 0.0]))
        max_real = np.max(np.real(eigs))
        assert max_real > 0, (
            f"Origin should be unstable, max eigenvalue real part = {max_real}"
        )

    def test_jacobian_shape(self):
        """Jacobian should be 3x3."""
        sim = PredatorPreyMutualistSimulation(_make_config())
        J = sim._jacobian(np.array([5.0, 2.0, 3.0]))
        assert J.shape == (3, 3)


class TestRediscovery:
    """Tests for the rediscovery data generation functions."""

    def test_trajectory_data_generation(self):
        from simulating_anything.rediscovery.predator_prey_mutualist import (
            generate_trajectory_data,
        )
        data = generate_trajectory_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert data["r"] == 1.0
        assert data["K"] == 10.0

    def test_mutualism_sweep_data(self):
        from simulating_anything.rediscovery.predator_prey_mutualist import (
            generate_mutualism_sweep,
        )
        data = generate_mutualism_sweep(n_m=5, n_steps=2000, dt=0.01)
        assert len(data["m"]) == 5
        assert len(data["prey_amplitude"]) == 5
        assert len(data["pred_amplitude"]) == 5
        assert np.all(data["prey_amplitude"] >= 0)

    def test_equilibrium_data_generation(self):
        from simulating_anything.rediscovery.predator_prey_mutualist import (
            generate_equilibrium_data,
        )
        data = generate_equilibrium_data(n_initial=10)
        assert "n_equilibria" in data
        assert data["n_equilibria"] > 0
        assert len(data["equilibria"]) == data["n_equilibria"]
        for eq in data["equilibria"]:
            assert "point" in eq
            assert "is_stable" in eq
            assert "max_real_eigenvalue" in eq
