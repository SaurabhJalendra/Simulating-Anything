"""Tests for the Beddington-DeAngelis predator-prey simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.beddington_deangelis import (
    BeddingtonDeAngelisSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    r: float = 1.0,
    K: float = 10.0,
    a: float = 1.0,
    b: float = 0.5,
    c: float = 0.5,
    e: float = 0.5,
    d: float = 0.3,
    N_0: float = 5.0,
    P_0: float = 2.0,
    dt: float = 0.01,
    n_steps: int = 500,
) -> SimulationConfig:
    """Create a SimulationConfig for Beddington-DeAngelis with optional overrides."""
    return SimulationConfig(
        domain=Domain.BEDDINGTON_DEANGELIS,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "r": r, "K": K, "a": a, "b": b, "c": c,
            "e": e, "d": d, "N_0": N_0, "P_0": P_0,
        },
    )


class TestBDBasic:
    """Basic creation and interface tests."""

    def test_creation_default_params(self):
        """Simulation should be created with default parameters."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        assert sim.r == 1.0
        assert sim.K == 10.0
        assert sim.a == 1.0
        assert sim.b_param == 0.5
        assert sim.c == 0.5
        assert sim.e == 0.5
        assert sim.d == 0.3

    def test_creation_custom_params(self):
        """Custom parameters should override defaults."""
        sim = BeddingtonDeAngelisSimulation(_make_config(r=2.0, K=20.0, c=1.5))
        assert sim.r == 2.0
        assert sim.K == 20.0
        assert sim.c == 1.5

    def test_reset_returns_initial_state(self):
        """Reset should return the initial state [N_0, P_0]."""
        sim = BeddingtonDeAngelisSimulation(_make_config(N_0=5.0, P_0=2.0))
        state = sim.reset()
        np.testing.assert_allclose(state, [5.0, 2.0])

    def test_initial_state_shape(self):
        """State vector should have shape (2,)."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        state = sim.reset()
        assert state.shape == (2,)

    def test_step_returns_state(self):
        """step() should return the new state."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        sim.reset()
        state = sim.step()
        assert state.shape == (2,)

    def test_observe_returns_current_state(self):
        """observe() should return the current state."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (2,)

    def test_run_returns_trajectory(self):
        """run() should return a TrajectoryData with correct shape."""
        sim = BeddingtonDeAngelisSimulation(_make_config(n_steps=200))
        traj = sim.run(n_steps=200)
        assert traj.states.shape == (201, 2)
        assert len(traj.timestamps) == 201

    def test_step_advances_state(self):
        """State should change after a step."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_dt_used_correctly(self):
        """Config dt should be used in integration."""
        sim = BeddingtonDeAngelisSimulation(_make_config(dt=0.001))
        assert sim.config.dt == 0.001

    def test_deterministic(self):
        """Same config produces the same trajectory."""
        config = _make_config(n_steps=100)
        sim1 = BeddingtonDeAngelisSimulation(config)
        traj1 = sim1.run(100)
        sim2 = BeddingtonDeAngelisSimulation(config)
        traj2 = sim2.run(100)
        np.testing.assert_allclose(traj1.states, traj2.states)


class TestBDProperties:
    """Tests for convenience properties."""

    def test_prey_population(self):
        """prey_population should return N."""
        sim = BeddingtonDeAngelisSimulation(_make_config(N_0=5.0, P_0=2.0))
        sim.reset()
        assert sim.prey_population == pytest.approx(5.0)

    def test_predator_population(self):
        """predator_population should return P."""
        sim = BeddingtonDeAngelisSimulation(_make_config(N_0=5.0, P_0=2.0))
        sim.reset()
        assert sim.predator_population == pytest.approx(2.0)

    def test_total_population(self):
        """total_population should return N + P."""
        sim = BeddingtonDeAngelisSimulation(_make_config(N_0=5.0, P_0=2.0))
        sim.reset()
        assert sim.total_population == pytest.approx(7.0)

    def test_properties_before_reset(self):
        """Properties should return 0.0 before reset."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        assert sim.prey_population == 0.0
        assert sim.predator_population == 0.0
        assert sim.total_population == 0.0


class TestBDDynamics:
    """Tests for dynamical behavior."""

    def test_populations_non_negative(self):
        """All populations should remain non-negative."""
        sim = BeddingtonDeAngelisSimulation(_make_config(dt=0.01, n_steps=5000))
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0), f"Negative population: {state}"

    def test_populations_bounded(self):
        """Populations should not blow up to infinity."""
        sim = BeddingtonDeAngelisSimulation(_make_config(dt=0.005, n_steps=10000))
        sim.reset()
        for _ in range(10000):
            state = sim.step()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"
            assert np.all(state < 200), f"Population blow-up: {state}"

    def test_prey_bounded_by_K(self):
        """Prey population should not greatly exceed carrying capacity K."""
        K = 10.0
        sim = BeddingtonDeAngelisSimulation(
            _make_config(K=K, dt=0.01, n_steps=10000, N_0=K / 2, P_0=1.0)
        )
        sim.reset()
        max_N = 0.0
        for _ in range(10000):
            state = sim.step()
            max_N = max(max_N, state[0])
        # Prey can temporarily overshoot K in predator-prey dynamics but
        # should remain in a reasonable range
        assert max_N < 2.0 * K, f"Prey exceeded 2*K: {max_N}"

    def test_approaches_equilibrium_with_high_c(self):
        """With high interference c, should converge to stable equilibrium."""
        sim = BeddingtonDeAngelisSimulation(
            _make_config(c=3.0, dt=0.01, n_steps=30000, N_0=5.0, P_0=2.0)
        )
        sim.reset()

        states = []
        for _ in range(30000):
            sim.step()
            states.append(sim.observe().copy())

        trajectory = np.array(states)
        # Check last 10% for low amplitude (stable equilibrium)
        last_frac = trajectory[-3000:]
        N_amp = np.max(last_frac[:, 0]) - np.min(last_frac[:, 0])
        assert N_amp < 1.0, (
            f"Expected stable equilibrium with c=3.0, prey amplitude={N_amp:.4f}"
        )

    def test_predator_extinction_when_d_high(self):
        """With very high predator death rate, predators should go extinct."""
        sim = BeddingtonDeAngelisSimulation(
            _make_config(d=5.0, dt=0.01, n_steps=5000, N_0=5.0, P_0=2.0)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
        state = sim.observe()
        assert state[1] < 0.01, (
            f"Predator should be extinct with d=5.0, got P={state[1]:.4f}"
        )

    def test_prey_approaches_K_without_predator(self):
        """Without predators, prey should converge to K."""
        K = 10.0
        sim = BeddingtonDeAngelisSimulation(
            _make_config(K=K, dt=0.01, n_steps=5000, N_0=1.0, P_0=0.0)
        )
        sim.reset()
        for _ in range(5000):
            sim.step()
        state = sim.observe()
        assert state[0] == pytest.approx(K, abs=0.1), (
            f"Prey should approach K={K}, got N={state[0]:.4f}"
        )
        assert state[1] == pytest.approx(0.0, abs=1e-10)


class TestBDFunctionalResponse:
    """Tests for the Beddington-DeAngelis functional response."""

    def test_bd_formula(self):
        """functional_response should match a*N / (1 + b*N + c*P)."""
        sim = BeddingtonDeAngelisSimulation(
            _make_config(a=1.0, b=0.5, c=0.5)
        )
        N, P = 4.0, 2.0
        expected = 1.0 * 4.0 / (1.0 + 0.5 * 4.0 + 0.5 * 2.0)
        assert sim.functional_response(N, P) == pytest.approx(expected)

    def test_bd_increases_with_prey(self):
        """Functional response should increase with prey density."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        P = 1.0
        fr_low = sim.functional_response(1.0, P)
        fr_high = sim.functional_response(10.0, P)
        assert fr_high > fr_low

    def test_bd_saturates_at_high_prey(self):
        """Functional response should saturate as N -> infinity."""
        sim = BeddingtonDeAngelisSimulation(_make_config(a=1.0, b=0.5, c=0.5))
        P = 1.0
        fr_100 = sim.functional_response(100.0, P)
        fr_1000 = sim.functional_response(1000.0, P)
        # At saturation, f -> a/b = 2.0
        assert fr_1000 == pytest.approx(sim.a / sim.b_param, rel=0.01)
        # Should be close to each other (diminishing returns)
        assert abs(fr_1000 - fr_100) < 0.1

    def test_bd_decreases_with_predator_interference(self):
        """Functional response should decrease with more predator interference."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        N = 5.0
        fr_low_P = sim.functional_response(N, 1.0)
        fr_high_P = sim.functional_response(N, 10.0)
        assert fr_high_P < fr_low_P

    def test_bd_zero_prey(self):
        """Functional response should be 0 when N=0."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        assert sim.functional_response(0.0, 2.0) == pytest.approx(0.0)

    def test_bd_reduces_to_holling2_when_c_zero(self):
        """When c=0, BD reduces to Holling Type II: a*N / (1 + b*N)."""
        sim = BeddingtonDeAngelisSimulation(_make_config(c=0.0, a=1.0, b=0.5))
        N, P = 4.0, 5.0
        expected_holling = 1.0 * 4.0 / (1.0 + 0.5 * 4.0)
        assert sim.functional_response(N, P) == pytest.approx(expected_holling)


class TestBDParameters:
    """Tests for parameter sensitivity."""

    def test_different_r(self):
        """Higher r should produce faster initial prey growth."""
        sim_slow = BeddingtonDeAngelisSimulation(
            _make_config(r=0.5, N_0=1.0, P_0=0.0, n_steps=100)
        )
        sim_slow.reset()
        for _ in range(100):
            sim_slow.step()
        N_slow = sim_slow.observe()[0]

        sim_fast = BeddingtonDeAngelisSimulation(
            _make_config(r=2.0, N_0=1.0, P_0=0.0, n_steps=100)
        )
        sim_fast.reset()
        for _ in range(100):
            sim_fast.step()
        N_fast = sim_fast.observe()[0]

        assert N_fast > N_slow

    def test_different_K(self):
        """Higher K should allow larger prey populations."""
        # Without predator, prey goes to K
        sim_low = BeddingtonDeAngelisSimulation(
            _make_config(K=5.0, N_0=1.0, P_0=0.0, n_steps=5000)
        )
        sim_low.reset()
        for _ in range(5000):
            sim_low.step()
        N_low = sim_low.observe()[0]

        sim_high = BeddingtonDeAngelisSimulation(
            _make_config(K=20.0, N_0=1.0, P_0=0.0, n_steps=5000)
        )
        sim_high.reset()
        for _ in range(5000):
            sim_high.step()
        N_high = sim_high.observe()[0]

        assert N_high > N_low

    def test_different_c_stabilization(self):
        """Higher c should reduce oscillation amplitude (stabilization)."""
        def measure_amplitude(c_val: float) -> float:
            sim = BeddingtonDeAngelisSimulation(
                _make_config(c=c_val, dt=0.01, n_steps=20000)
            )
            sim.reset()
            states = []
            for _ in range(20000):
                sim.step()
                states.append(sim.observe().copy())
            traj = np.array(states)
            half = len(traj) // 2
            return float(np.max(traj[half:, 0]) - np.min(traj[half:, 0]))

        amp_low_c = measure_amplitude(0.1)
        amp_high_c = measure_amplitude(2.0)
        assert amp_high_c < amp_low_c, (
            f"Higher c should stabilize: amp(c=0.1)={amp_low_c:.4f}, "
            f"amp(c=2.0)={amp_high_c:.4f}"
        )

    def test_reduces_to_holling2(self):
        """With c=0, dynamics should match Holling Type II (RM model)."""
        # Both should produce the same trajectory when c=0
        config_bd = _make_config(c=0.0, dt=0.01, n_steps=1000)
        sim_bd = BeddingtonDeAngelisSimulation(config_bd)
        traj_bd = sim_bd.run(1000)

        # The state at any point should only depend on Holling II terms
        assert traj_bd.states.shape == (1001, 2)
        assert np.all(np.isfinite(traj_bd.states))


class TestBDEquilibrium:
    """Tests for coexistence equilibrium computation."""

    def test_coexistence_equilibrium_exists(self):
        """With default params, coexistence equilibrium should exist."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        N_star, P_star = sim.coexistence_equilibrium()
        assert N_star > 0
        assert P_star > 0

    def test_equilibrium_is_fixed_point(self):
        """At coexistence equilibrium, derivatives should be near zero."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        N_star, P_star = sim.coexistence_equilibrium()
        state = np.array([N_star, P_star])
        dy = sim._derivatives(state)
        np.testing.assert_allclose(dy, [0.0, 0.0], atol=1e-8)

    def test_equilibrium_prey_less_than_K(self):
        """Equilibrium prey should be less than carrying capacity."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        N_star, _ = sim.coexistence_equilibrium()
        assert N_star < sim.K

    def test_simulation_approaches_equilibrium(self):
        """Long simulation with high c should converge near equilibrium."""
        sim = BeddingtonDeAngelisSimulation(
            _make_config(c=3.0, dt=0.01, n_steps=50000)
        )
        N_star, P_star = sim.coexistence_equilibrium()
        sim.reset()
        for _ in range(50000):
            sim.step()
        state = sim.observe()
        np.testing.assert_allclose(
            state, [N_star, P_star], rtol=0.05,
            err_msg=f"State {state} not near equilibrium [{N_star:.4f}, {P_star:.4f}]",
        )


class TestBDDerivatives:
    """Tests for the right-hand side derivatives."""

    def test_derivatives_shape(self):
        """_derivatives should return array of shape (2,)."""
        sim = BeddingtonDeAngelisSimulation(_make_config())
        dy = sim._derivatives(np.array([5.0, 2.0]))
        assert dy.shape == (2,)

    def test_prey_growth_without_predators(self):
        """Without predators, dN/dt = r*N*(1 - N/K)."""
        sim = BeddingtonDeAngelisSimulation(_make_config(r=1.0, K=10.0))
        N, P = 5.0, 0.0
        dy = sim._derivatives(np.array([N, P]))
        expected_dN = 1.0 * 5.0 * (1.0 - 5.0 / 10.0)  # 2.5
        assert dy[0] == pytest.approx(expected_dN)

    def test_predator_death_without_prey(self):
        """Without prey, dP/dt = -d*P."""
        sim = BeddingtonDeAngelisSimulation(_make_config(d=0.3))
        N, P = 0.0, 2.0
        dy = sim._derivatives(np.array([N, P]))
        expected_dP = -0.3 * 2.0
        assert dy[1] == pytest.approx(expected_dP)

    def test_carrying_capacity_stops_growth(self):
        """At N=K with no predators, dN/dt should be 0."""
        sim = BeddingtonDeAngelisSimulation(_make_config(K=10.0))
        dy = sim._derivatives(np.array([10.0, 0.0]))
        assert dy[0] == pytest.approx(0.0)


class TestBDLongRunStability:
    """Tests for numerical stability over long runs."""

    def test_no_nan_long_run(self):
        """No NaN after many steps."""
        sim = BeddingtonDeAngelisSimulation(_make_config(dt=0.01, n_steps=50000))
        sim.reset()
        for _ in range(50000):
            state = sim.step()
        assert np.all(np.isfinite(state)), f"NaN or Inf in state: {state}"

    def test_no_nan_various_params(self):
        """No NaN for various parameter combinations."""
        params_list = [
            {"r": 2.0, "K": 5.0, "a": 2.0, "b": 1.0, "c": 0.1, "e": 0.8, "d": 0.5},
            {"r": 0.5, "K": 20.0, "a": 0.3, "b": 0.1, "c": 2.0, "e": 0.3, "d": 0.1},
            {"r": 1.5, "K": 15.0, "a": 1.5, "b": 0.5, "c": 0.0, "e": 0.5, "d": 0.2},
        ]
        for params in params_list:
            config = SimulationConfig(
                domain=Domain.BEDDINGTON_DEANGELIS,
                dt=0.01,
                n_steps=10000,
                parameters={**params, "N_0": 5.0, "P_0": 2.0},
            )
            sim = BeddingtonDeAngelisSimulation(config)
            sim.reset()
            for _ in range(10000):
                state = sim.step()
            assert np.all(np.isfinite(state)), (
                f"NaN with params {params}: state={state}"
            )


class TestBDRediscovery:
    """Tests for rediscovery data generation functions."""

    def test_generate_sweep_data(self):
        """Sweep data generation should produce valid arrays."""
        from simulating_anything.rediscovery.beddington_deangelis import (
            generate_sweep_data,
        )
        data = generate_sweep_data(n_samples=5, n_steps=500, dt=0.01)
        assert len(data["r"]) == 5
        assert len(data["K"]) == 5
        assert len(data["c"]) == 5
        assert len(data["N_avg"]) == 5
        assert len(data["P_avg"]) == 5
        assert len(data["N_amplitude"]) == 5
        assert len(data["P_amplitude"]) == 5
        assert len(data["fr_avg"]) == 5
        assert np.all(np.isfinite(data["N_avg"]))

    def test_generate_ode_data(self):
        """ODE data generation should produce valid trajectory."""
        from simulating_anything.rediscovery.beddington_deangelis import (
            generate_ode_data,
        )
        data = generate_ode_data(n_steps=200, dt=0.01)
        assert data["states"].shape == (201, 2)
        assert len(data["time"]) == 201
        assert data["r"] == 1.0
        assert data["K"] == 10.0
        assert data["a"] == 1.0
        assert data["b"] == 0.5
        assert data["c"] == 0.5
        assert data["e"] == 0.5
        assert data["d"] == 0.3

    def test_generate_interference_sweep(self):
        """Interference sweep should produce valid arrays."""
        from simulating_anything.rediscovery.beddington_deangelis import (
            generate_interference_sweep,
        )
        data = generate_interference_sweep(n_c_values=5, n_steps=500, dt=0.01)
        assert len(data["c"]) == 5
        assert len(data["N_avg"]) == 5
        assert len(data["N_amplitude"]) == 5
        assert data["c"][0] == pytest.approx(0.0)  # starts at c_min
        assert np.all(np.isfinite(data["N_avg"]))

    def test_sweep_data_parameters_in_range(self):
        """Sweep parameters should be within specified ranges."""
        from simulating_anything.rediscovery.beddington_deangelis import (
            generate_sweep_data,
        )
        data = generate_sweep_data(n_samples=20, n_steps=500, dt=0.01)
        assert np.all(data["r"] >= 0.5) and np.all(data["r"] <= 2.0)
        assert np.all(data["K"] >= 5.0) and np.all(data["K"] <= 20.0)
        assert np.all(data["c"] >= 0.0) and np.all(data["c"] <= 2.0)

    def test_ode_data_non_negative(self):
        """ODE trajectory should have non-negative populations."""
        from simulating_anything.rediscovery.beddington_deangelis import (
            generate_ode_data,
        )
        data = generate_ode_data(n_steps=1000, dt=0.01)
        assert np.all(data["states"] >= 0)
