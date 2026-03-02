"""Tests for the predator-prey-parasite eco-epidemiological model."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.predator_prey_parasite import (
    PredatorPreyParasiteSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    dt: float = 0.01,
    n_steps: int = 1000,
    **param_overrides: float,
) -> SimulationConfig:
    """Create a SimulationConfig for PredatorPreyParasite with optional overrides."""
    params: dict[str, float] = {
        "r": 1.0, "K": 10.0, "a": 0.1, "e": 0.5, "d": 0.2,
        "lambda_": 0.15, "mu": 0.3, "alpha": 0.05,
        "N_0": 5.0, "P_0": 2.0, "Z_0": 1.0,
    }
    params.update(param_overrides)
    return SimulationConfig(
        domain=Domain.PREDATOR_PREY_PARASITE,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestPredatorPreyParasiteSimulation:
    """Core simulation tests."""

    def test_reset_state_shape(self):
        """Initial state should be 3D: [N, P, Z]."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        state = sim.reset(seed=42)
        assert state.shape == (3,)

    def test_reset_state_values(self):
        """Initial state should be close to [N_0, P_0, Z_0]."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        state = sim.reset(seed=42)
        np.testing.assert_allclose(state[0], 5.0, atol=0.02)
        np.testing.assert_allclose(state[1], 2.0, atol=0.02)
        np.testing.assert_allclose(state[2], 1.0, atol=0.02)

    def test_observe_shape(self):
        """Observe should return 3-element state [N, P, Z]."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        sim.reset(seed=42)
        obs = sim.observe()
        assert obs.shape == (3,)

    def test_step_advances(self):
        """State should change after a step."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        sim.reset(seed=42)
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_deterministic(self):
        """Same seed produces identical trajectories."""
        config = _make_config()
        sim1 = PredatorPreyParasiteSimulation(config)
        sim1.reset(seed=42)
        for _ in range(100):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = PredatorPreyParasiteSimulation(config)
        sim2.reset(seed=42)
        for _ in range(100):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_allclose(state1, state2)

    def test_non_negative(self):
        """All populations should stay non-negative."""
        sim = PredatorPreyParasiteSimulation(_make_config(dt=0.01))
        sim.reset(seed=42)
        for _ in range(5000):
            sim.step()
            state = sim.observe()
            assert state[0] >= 0, f"N became negative: {state[0]}"
            assert state[1] >= 0, f"P became negative: {state[1]}"
            assert state[2] >= 0, f"Z became negative: {state[2]}"

    def test_boundedness(self):
        """Populations should not diverge (stay bounded by ~10*K)."""
        sim = PredatorPreyParasiteSimulation(_make_config(dt=0.01))
        sim.reset(seed=42)
        cap = sim.K
        for _ in range(10000):
            sim.step()
            state = sim.observe()
            total = np.sum(state)
            assert total < cap * 20, f"Population diverged: {total}"

    def test_long_run_stability(self):
        """No NaN after 50000 steps."""
        sim = PredatorPreyParasiteSimulation(_make_config(dt=0.01))
        sim.reset(seed=42)
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"

    def test_run_returns_trajectory(self):
        """run() should return TrajectoryData with correct shapes."""
        config = _make_config(dt=0.01, n_steps=100)
        sim = PredatorPreyParasiteSimulation(config)
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert traj.timestamps.shape == (101,)


class TestODETerms:
    """Tests verifying individual ODE terms and subsystem dynamics."""

    def test_prey_logistic_growth_alone(self):
        """Without predator or parasite, prey grows logistically to K."""
        sim = PredatorPreyParasiteSimulation(
            _make_config(
                P_0=0.0, Z_0=0.0, N_0=1.0,
                a=0.0, lambda_=0.0, dt=0.01,
            )
        )
        sim.reset(seed=42)
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        np.testing.assert_allclose(state[0], 10.0, atol=0.5)

    def test_prey_predator_subsystem(self):
        """With Z=0, reduces to Lotka-Volterra-like predator-prey."""
        sim = PredatorPreyParasiteSimulation(
            _make_config(Z_0=0.0, lambda_=0.0, dt=0.01)
        )
        sim.reset(seed=42)
        for _ in range(20000):
            sim.step()
        state = sim.observe()
        # Parasite should remain near zero
        assert state[2] < 0.01, f"Parasite should stay at zero: Z={state[2]}"
        # Prey and predator should both be positive (coexistence)
        assert state[0] > 0.1, f"Prey should survive: N={state[0]}"
        assert state[1] > 0.01, f"Predator should survive: P={state[1]}"

    def test_prey_parasite_subsystem(self):
        """With P=0 and d large, reduces to prey-parasite dynamics."""
        sim = PredatorPreyParasiteSimulation(
            _make_config(P_0=0.0, d=10.0, alpha=0.0, dt=0.01)
        )
        sim.reset(seed=42)
        for _ in range(20000):
            sim.step()
        state = sim.observe()
        # Predator should remain near zero
        assert state[1] < 0.01, f"Predator should die: P={state[1]}"

    def test_predator_death(self):
        """With a=0, predator has no food and dies out."""
        sim = PredatorPreyParasiteSimulation(
            _make_config(a=0.0, dt=0.01)
        )
        sim.reset(seed=42)
        for _ in range(20000):
            sim.step()
        state = sim.observe()
        assert state[1] < 0.01, f"Predator should die without prey: P={state[1]}"

    def test_parasite_death(self):
        """With lambda_=0, parasite has no host and dies out."""
        sim = PredatorPreyParasiteSimulation(
            _make_config(lambda_=0.0, dt=0.01)
        )
        sim.reset(seed=42)
        for _ in range(20000):
            sim.step()
        state = sim.observe()
        assert state[2] < 0.01, f"Parasite should die: Z={state[2]}"

    def test_derivatives_at_origin(self):
        """Derivatives at [0, 0, 0] should be [0, 0, 0]."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        sim.reset(seed=42)
        deriv = sim._derivatives(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(deriv, [0.0, 0.0, 0.0], atol=1e-15)

    def test_derivatives_prey_only(self):
        """At [K, 0, 0], dN/dt=0 (at carrying capacity with no consumers)."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        sim.reset(seed=42)
        deriv = sim._derivatives(np.array([sim.K, 0.0, 0.0]))
        np.testing.assert_allclose(deriv[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(deriv[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(deriv[2], 0.0, atol=1e-12)

    def test_predator_growth_term(self):
        """dP/dt = e*a*N*P - d*P: verify at specific state."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        sim.reset(seed=42)
        N, P, Z = 5.0, 2.0, 1.0
        deriv = sim._derivatives(np.array([N, P, Z]))
        expected_dP = sim.e * sim.a * N * P - sim.d * P
        np.testing.assert_allclose(deriv[1], expected_dP, rtol=1e-12)

    def test_parasite_growth_term(self):
        """dZ/dt = lambda_*N*Z - mu*Z - alpha*Z*P: verify at specific state."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        sim.reset(seed=42)
        N, P, Z = 5.0, 2.0, 1.0
        deriv = sim._derivatives(np.array([N, P, Z]))
        expected_dZ = sim.lambda_ * N * Z - sim.mu * Z - sim.alpha * Z * P
        np.testing.assert_allclose(deriv[2], expected_dZ, rtol=1e-12)

    def test_prey_growth_term(self):
        """dN/dt = r*N*(1-N/K) - a*N*P - lambda_*N*Z: verify at specific state."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        sim.reset(seed=42)
        N, P, Z = 5.0, 2.0, 1.0
        deriv = sim._derivatives(np.array([N, P, Z]))
        expected_dN = (
            sim.r * N * (1.0 - N / sim.K)
            - sim.a * N * P
            - sim.lambda_ * N * Z
        )
        np.testing.assert_allclose(deriv[0], expected_dN, rtol=1e-12)


class TestEquilibria:
    """Tests for equilibrium computation."""

    def test_trivial_equilibrium(self):
        """Trivial equilibrium [0, 0, 0] always exists."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        eq = sim.compute_equilibria()
        np.testing.assert_allclose(eq["trivial"], [0.0, 0.0, 0.0])

    def test_prey_only_equilibrium(self):
        """Prey-only equilibrium [K, 0, 0] always exists."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        eq = sim.compute_equilibria()
        np.testing.assert_allclose(eq["prey_only"], [10.0, 0.0, 0.0])

    def test_prey_predator_equilibrium(self):
        """Prey-predator equilibrium should satisfy fixed-point conditions."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        eq = sim.compute_equilibria()
        if eq["prey_predator"] is not None:
            state = eq["prey_predator"]
            assert state[2] == 0.0  # Z = 0
            assert state[0] > 0  # N > 0
            assert state[1] > 0  # P > 0
            # Verify it is a fixed point
            deriv = sim._derivatives(state)
            np.testing.assert_allclose(deriv, [0.0, 0.0, 0.0], atol=1e-10)

    def test_prey_parasite_equilibrium(self):
        """Prey-parasite equilibrium should satisfy fixed-point conditions."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        eq = sim.compute_equilibria()
        if eq["prey_parasite"] is not None:
            state = eq["prey_parasite"]
            assert state[1] == 0.0  # P = 0
            assert state[0] > 0  # N > 0
            assert state[2] > 0  # Z > 0
            # Verify it is a fixed point
            deriv = sim._derivatives(state)
            np.testing.assert_allclose(deriv, [0.0, 0.0, 0.0], atol=1e-10)

    def test_coexistence_equilibrium_check(self):
        """If coexistence equilibrium exists, it should be a true fixed point."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        eq = sim.compute_equilibria()
        if eq["coexistence"] is not None:
            state = eq["coexistence"]
            assert np.all(state > 0), f"Coexistence needs all positive: {state}"
            deriv = sim._derivatives(state)
            np.testing.assert_allclose(deriv, [0.0, 0.0, 0.0], atol=1e-10)

    def test_prey_predator_N_star(self):
        """Prey-predator N* = d/(e*a)."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        eq = sim.compute_equilibria()
        if eq["prey_predator"] is not None:
            expected_N = sim.d / (sim.e * sim.a)
            np.testing.assert_allclose(eq["prey_predator"][0], expected_N, rtol=1e-10)

    def test_prey_parasite_N_star(self):
        """Prey-parasite N* = mu/lambda_."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        eq = sim.compute_equilibria()
        if eq["prey_parasite"] is not None:
            expected_N = sim.mu / sim.lambda_
            np.testing.assert_allclose(eq["prey_parasite"][0], expected_N, rtol=1e-10)


class TestParameterSweep:
    """Tests for parameter sweep functionality."""

    def test_parameter_sweep_shape(self):
        """Sweep should return arrays of correct length."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        result = sim.parameter_sweep(
            "lambda_", np.linspace(0.0, 0.3, 5), n_steps=1000,
        )
        assert len(result["param_values"]) == 5
        assert len(result["N_avg"]) == 5
        assert len(result["P_avg"]) == 5
        assert len(result["Z_avg"]) == 5

    def test_parameter_sweep_finite(self):
        """Sweep results should be finite."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        result = sim.parameter_sweep(
            "lambda_", np.linspace(0.0, 0.3, 5), n_steps=1000,
        )
        assert np.all(np.isfinite(result["N_avg"]))
        assert np.all(np.isfinite(result["P_avg"]))
        assert np.all(np.isfinite(result["Z_avg"]))

    def test_sweep_zero_lambda_no_parasite(self):
        """At lambda=0, parasite should be near zero."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        result = sim.parameter_sweep(
            "lambda_", np.array([0.0]), n_steps=5000,
        )
        assert result["Z_avg"][0] < 0.1, (
            f"Parasite should be near zero at lambda=0: Z={result['Z_avg'][0]}"
        )


class TestProperties:
    """Tests for computed properties."""

    def test_total_population(self):
        """Total population should equal sum of all species."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        sim.reset(seed=42)
        total = sim.total_population
        state = sim.observe()
        np.testing.assert_allclose(total, np.sum(state))

    def test_total_population_none_state(self):
        """Total population should be 0 when state is None."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        assert sim.total_population == 0.0

    def test_is_coexisting(self):
        """With default params, system should start in coexistence."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        sim.reset(seed=42)
        assert sim.is_coexisting

    def test_is_coexisting_false_when_none(self):
        """Coexistence should be False when state is None."""
        sim = PredatorPreyParasiteSimulation(_make_config())
        assert not sim.is_coexisting


class TestRediscoveryDataGeneration:
    """Tests for rediscovery data generation functions."""

    def test_generate_ode_data(self):
        """ODE data generation should produce valid arrays."""
        from simulating_anything.rediscovery.predator_prey_parasite import (
            generate_ode_data,
        )
        data = generate_ode_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
        assert len(data["N"]) == 501
        assert len(data["P"]) == 501
        assert len(data["Z"]) == 501
        assert np.all(np.isfinite(data["states"]))

    def test_generate_lambda_sweep(self):
        """Lambda sweep should produce results for each value."""
        from simulating_anything.rediscovery.predator_prey_parasite import (
            generate_lambda_sweep,
        )
        data = generate_lambda_sweep(n_lambda=5, n_steps=1000, dt=0.01)
        assert len(data["lambda_values"]) == 5
        assert len(data["N_avg"]) == 5
        assert len(data["P_avg"]) == 5
        assert len(data["Z_avg"]) == 5
        assert np.all(np.isfinite(data["N_avg"]))

    def test_classify_behavior(self):
        """Behavior classifier should return a valid classification."""
        from simulating_anything.rediscovery.predator_prey_parasite import (
            classify_long_time_behavior,
        )
        # Constant data -> fixed point
        const_data = np.ones((1000, 3)) * 5.0
        assert classify_long_time_behavior(const_data) == "fixed_point"

    def test_classify_behavior_oscillating(self):
        """Oscillating data should not be classified as fixed point."""
        from simulating_anything.rediscovery.predator_prey_parasite import (
            classify_long_time_behavior,
        )
        t = np.linspace(0, 20 * np.pi, 2000)
        data = np.column_stack([
            5.0 + 2.0 * np.sin(t),
            2.0 + 0.5 * np.cos(t),
            1.0 + 0.3 * np.sin(t + 1),
        ])
        behavior = classify_long_time_behavior(data)
        assert behavior != "fixed_point"
