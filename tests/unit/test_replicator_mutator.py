"""Tests for the replicator-mutator equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.replicator_mutator import (
    ReplicatorMutatorSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    n_strategies: int = 3,
    payoff: list[list[float]] | None = None,
    mutation_rate: float = 0.0,
    dt: float = 0.01,
    n_steps: int = 1000,
    x_init: list[float] | None = None,
) -> SimulationConfig:
    """Build a SimulationConfig for the replicator-mutator simulation."""
    params: dict[str, float] = {
        "n_strategies": float(n_strategies),
        "mutation_rate": mutation_rate,
    }
    if payoff is not None:
        for i, row in enumerate(payoff):
            for j, val in enumerate(row):
                params[f"A_{i}_{j}"] = val
    if x_init is not None:
        for i, val in enumerate(x_init):
            params[f"x_0_{i}"] = val
    return SimulationConfig(
        domain=Domain.CUSTOM,
        dt=dt,
        n_steps=n_steps,
        parameters=params,
    )


class TestReplicatorMutatorSimulation:
    """Core simulation tests."""

    def test_initial_state_shape(self):
        """State vector has correct length."""
        sim = ReplicatorMutatorSimulation(_make_config(n_strategies=3))
        state = sim.reset()
        assert state.shape == (3,)

    def test_initial_state_sums_to_one(self):
        """Initial frequencies lie on the simplex."""
        sim = ReplicatorMutatorSimulation(_make_config(n_strategies=5))
        state = sim.reset()
        assert state.sum() == pytest.approx(1.0, abs=1e-12)

    def test_simplex_preserved_after_steps(self):
        """Frequencies remain on the simplex throughout integration."""
        sim = ReplicatorMutatorSimulation(
            _make_config(n_strategies=3, dt=0.01, x_init=[0.5, 0.3, 0.2])
        )
        sim.reset()
        for _ in range(2000):
            state = sim.step()
            assert state.sum() == pytest.approx(1.0, abs=1e-8), (
                f"Simplex violation: sum = {state.sum()}"
            )
            assert np.all(state >= 0), f"Negative frequency: {state}"

    def test_step_advances_state(self):
        """State changes after a step."""
        sim = ReplicatorMutatorSimulation(
            _make_config(n_strategies=3, x_init=[0.5, 0.3, 0.2])
        )
        sim.reset()
        s0 = sim.observe().copy()
        sim.step()
        s1 = sim.observe()
        assert not np.allclose(s0, s1)

    def test_observe_returns_current_state(self):
        """observe() matches internal state."""
        sim = ReplicatorMutatorSimulation(_make_config())
        sim.reset()
        sim.step()
        state = sim.observe()
        assert state.shape == (3,)
        assert state.sum() == pytest.approx(1.0, abs=1e-10)


class TestRPSHeteroclinic:
    """Rock-Paper-Scissors specific tests."""

    def test_rps_interior_fixed_point(self):
        """RPS interior fixed point is (1/3, 1/3, 1/3)."""
        sim = ReplicatorMutatorSimulation(_make_config(n_strategies=3))
        sim.reset()
        fp = sim.interior_fixed_point
        assert fp is not None
        np.testing.assert_allclose(fp, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)

    def test_rps_derivatives_at_fixed_point(self):
        """Derivatives vanish at the interior fixed point."""
        sim = ReplicatorMutatorSimulation(_make_config(n_strategies=3))
        sim.reset()
        fp = np.array([1 / 3, 1 / 3, 1 / 3])
        dx = sim._derivatives(fp)
        np.testing.assert_allclose(dx, 0.0, atol=1e-14)

    def test_rps_oscillation_detected(self):
        """RPS dynamics produce oscillatory behavior."""
        sim = ReplicatorMutatorSimulation(
            _make_config(n_strategies=3, dt=0.005, x_init=[0.5, 0.3, 0.2])
        )
        sim.reset()

        # Collect trajectory
        x0_vals = []
        for _ in range(10000):
            sim.step()
            x0_vals.append(sim.observe()[0])

        # Check that x0 oscillates (not monotone)
        x0_arr = np.array(x0_vals)
        diffs = np.diff(x0_arr)
        n_sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        assert n_sign_changes > 5, "Expected oscillatory behavior in RPS"

    def test_rps_bounded_dynamics(self):
        """All frequencies stay bounded in [0, 1] for RPS."""
        sim = ReplicatorMutatorSimulation(
            _make_config(n_strategies=3, dt=0.01, x_init=[0.6, 0.3, 0.1])
        )
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(state >= 0.0), f"Negative frequency: {state}"
            assert np.all(state <= 1.0), f"Frequency > 1: {state}"


class TestHawkDove:
    """Hawk-Dove game tests."""

    def _hawk_dove_payoff(self, V: float, C: float) -> list[list[float]]:
        """Standard Hawk-Dove payoff matrix."""
        return [[(V - C) / 2.0, V], [0.0, V / 2.0]]

    def test_hawk_dove_ess_theory(self):
        """Theoretical ESS frequency is V/C for V < C."""
        V, C = 4.0, 10.0
        sim = ReplicatorMutatorSimulation(
            _make_config(n_strategies=2, payoff=self._hawk_dove_payoff(V, C))
        )
        sim.reset()
        ess = sim.hawk_dove_ess()
        assert ess is not None
        assert ess == pytest.approx(V / C, abs=1e-10)

    def test_hawk_dove_convergence(self):
        """Simulation converges to the mixed ESS for V < C."""
        V, C = 3.0, 10.0
        sim = ReplicatorMutatorSimulation(
            _make_config(
                n_strategies=2,
                payoff=self._hawk_dove_payoff(V, C),
                dt=0.005,
                x_init=[0.5, 0.5],
            )
        )
        sim.reset()
        sim.simulate_to_steady_state(max_steps=100000)
        p_hawk = sim.observe()[0]
        # Should converge to V/C = 0.3
        assert p_hawk == pytest.approx(V / C, abs=0.05)

    def test_hawk_dove_pure_ess_when_v_exceeds_c(self):
        """When V >= C, Hawk dominates (pure ESS)."""
        V, C = 12.0, 10.0
        sim = ReplicatorMutatorSimulation(
            _make_config(
                n_strategies=2,
                payoff=self._hawk_dove_payoff(V, C),
                dt=0.005,
                x_init=[0.5, 0.5],
            )
        )
        sim.reset()
        ess = sim.hawk_dove_ess()
        assert ess == pytest.approx(1.0)


class TestPrisonersDilemma:
    """Prisoner's Dilemma tests."""

    def test_defection_dominates(self):
        """In PD, defection frequency increases toward 1."""
        payoff = [[3.0, 0.0], [5.0, 1.0]]  # R=3, S=0, T=5, P=1
        sim = ReplicatorMutatorSimulation(
            _make_config(
                n_strategies=2,
                payoff=payoff,
                dt=0.01,
                x_init=[0.8, 0.2],
            )
        )
        sim.reset()
        for _ in range(20000):
            sim.step()

        # Cooperation should collapse, defection should dominate
        p_defect = sim.observe()[1]
        assert p_defect > 0.99, f"Defection should dominate, got p_defect={p_defect}"


class TestMutationEffects:
    """Tests for the mutation mechanism."""

    def test_mutation_matrix_is_stochastic(self):
        """Each row of the mutation matrix sums to 1."""
        sim = ReplicatorMutatorSimulation(
            _make_config(n_strategies=4, mutation_rate=0.2)
        )
        sim.reset()
        Q = sim.mutation_matrix
        row_sums = Q.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-14)

    def test_no_mutation_is_identity(self):
        """Q = I when mutation_rate = 0."""
        sim = ReplicatorMutatorSimulation(
            _make_config(n_strategies=3, mutation_rate=0.0)
        )
        sim.reset()
        np.testing.assert_allclose(sim.mutation_matrix, np.eye(3), atol=1e-14)

    def test_mutation_stabilizes_rps(self):
        """Mutation drives RPS toward uniform equilibrium."""
        # Without mutation: cycles around (1/3, 1/3, 1/3)
        sim_no_mut = ReplicatorMutatorSimulation(
            _make_config(n_strategies=3, mutation_rate=0.0, dt=0.01,
                         x_init=[0.5, 0.3, 0.2])
        )
        sim_no_mut.reset()

        # With mutation: should converge closer to (1/3, 1/3, 1/3)
        sim_mut = ReplicatorMutatorSimulation(
            _make_config(n_strategies=3, mutation_rate=0.1, dt=0.01,
                         x_init=[0.5, 0.3, 0.2])
        )
        sim_mut.reset()

        for _ in range(30000):
            sim_no_mut.step()
            sim_mut.step()

        uniform = np.ones(3) / 3.0
        dev_no_mut = np.linalg.norm(sim_no_mut.observe() - uniform)
        dev_mut = np.linalg.norm(sim_mut.observe() - uniform)

        # Mutation should reduce deviation from uniform
        assert dev_mut < dev_no_mut + 0.01, (
            f"Mutation should stabilize: dev_mut={dev_mut}, dev_no_mut={dev_no_mut}"
        )


class TestFitnessComputation:
    """Tests for fitness-related methods."""

    def test_fitness_at_uniform_rps(self):
        """At uniform distribution in RPS, all fitnesses are zero."""
        sim = ReplicatorMutatorSimulation(_make_config(n_strategies=3))
        sim.reset()
        sim._state = np.array([1 / 3, 1 / 3, 1 / 3])
        f = sim.fitness()
        np.testing.assert_allclose(f, 0.0, atol=1e-14)

    def test_average_fitness_at_uniform_rps(self):
        """Average fitness at uniform RPS is zero."""
        sim = ReplicatorMutatorSimulation(_make_config(n_strategies=3))
        sim.reset()
        sim._state = np.array([1 / 3, 1 / 3, 1 / 3])
        phi = sim.average_fitness()
        assert phi == pytest.approx(0.0, abs=1e-14)

    def test_ess_check_rps(self):
        """No pure strategy is ESS in RPS (each is beaten by another)."""
        sim = ReplicatorMutatorSimulation(_make_config(n_strategies=3))
        sim.reset()
        for i in range(3):
            assert not sim.is_ess(i), f"Strategy {i} should not be ESS in RPS"


class TestTrajectoryCollection:
    """Tests for the run() method and trajectory output."""

    def test_run_returns_trajectory(self):
        """run() returns TrajectoryData with correct shapes."""
        sim = ReplicatorMutatorSimulation(
            _make_config(n_strategies=3, dt=0.01, n_steps=100)
        )
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 3)
        assert len(traj.timestamps) == 101

    def test_trajectory_simplex_throughout(self):
        """All states in trajectory lie on the simplex."""
        sim = ReplicatorMutatorSimulation(
            _make_config(n_strategies=4, dt=0.01, n_steps=500,
                         x_init=[0.4, 0.3, 0.2, 0.1])
        )
        traj = sim.run(n_steps=500)
        sums = np.sum(traj.states, axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-8)
        assert np.all(traj.states >= 0.0)


class TestRediscoveryDataGeneration:
    """Tests for rediscovery data generation functions."""

    def test_rps_trajectory_generation(self):
        """RPS trajectory has correct shape and simplex property."""
        from simulating_anything.rediscovery.replicator_mutator import (
            generate_rps_trajectory,
        )
        data = generate_rps_trajectory(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        sums = np.sum(data["states"], axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-8)

    def test_hawk_dove_data_generation(self):
        """Hawk-Dove sweep returns correct number of data points."""
        from simulating_anything.rediscovery.replicator_mutator import (
            generate_hawk_dove_data,
        )
        data = generate_hawk_dove_data(n_V=5, n_steps=5000, dt=0.01)
        assert len(data["V"]) == 5
        assert len(data["p_star_measured"]) == 5
        assert len(data["p_star_theory"]) == 5

    def test_pd_trajectory_generation(self):
        """PD trajectory shows cooperation decline."""
        from simulating_anything.rediscovery.replicator_mutator import (
            generate_pd_trajectory,
        )
        data = generate_pd_trajectory(n_steps=5000, dt=0.01)
        assert data["states"].shape == (5001, 2)
        # Cooperation should decrease
        assert data["cooperation"][-1] < data["cooperation"][0]

    def test_mutation_sweep_generation(self):
        """Mutation sweep returns correct structure."""
        from simulating_anything.rediscovery.replicator_mutator import (
            generate_mutation_sweep_data,
        )
        data = generate_mutation_sweep_data(n_mu=5, n_steps=5000, dt=0.01)
        assert len(data["mutation_rate"]) == 5
        assert len(data["deviation"]) == 5
        assert data["final_states"].shape == (5, 3)

    def test_ode_data_generation(self):
        """ODE data has correct shape for SINDy input."""
        from simulating_anything.rediscovery.replicator_mutator import (
            generate_ode_data,
        )
        data = generate_ode_data(n_steps=500, dt=0.01)
        assert data["states"].shape == (501, 3)
        assert len(data["time"]) == 501
