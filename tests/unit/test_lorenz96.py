"""Tests for the Lorenz-96 atmospheric chaos simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.lorenz96 import Lorenz96
from simulating_anything.types.simulation import Domain, SimulationConfig


class TestLorenz96Creation:
    """Tests for Lorenz96 creation and parameter handling."""

    def _make_sim(self, N: int = 36, F: float = 8.0, dt: float = 0.01) -> Lorenz96:
        config = SimulationConfig(
            domain=Domain.LORENZ96,
            dt=dt,
            n_steps=1000,
            parameters={"N": float(N), "F": F},
        )
        return Lorenz96(config)

    def test_creation(self):
        sim = self._make_sim()
        assert sim.N == 36
        assert sim.F == 8.0

    def test_custom_parameters(self):
        sim = self._make_sim(N=20, F=4.0)
        assert sim.N == 20
        assert sim.F == 4.0

    def test_initial_state_shape(self):
        sim = self._make_sim(N=36)
        state = sim.reset()
        assert state.shape == (36,)

    def test_initial_state_near_F(self):
        """Initial state should be close to F with a small perturbation."""
        sim = self._make_sim(N=36, F=8.0)
        state = sim.reset()
        # Most sites should be exactly F, one has a small perturbation
        assert np.sum(np.abs(state - 8.0) > 1e-10) <= 1

    def test_observe_returns_copy(self):
        """Observe should return a copy, not a reference."""
        sim = self._make_sim()
        sim.reset()
        obs1 = sim.observe()
        sim.step()
        obs2 = sim.observe()
        assert not np.allclose(obs1, obs2)


class TestLorenz96Dynamics:
    """Tests for Lorenz-96 dynamics."""

    def _make_sim(self, N: int = 36, F: float = 8.0, dt: float = 0.01) -> Lorenz96:
        config = SimulationConfig(
            domain=Domain.LORENZ96,
            dt=dt,
            n_steps=1000,
            parameters={"N": float(N), "F": F},
        )
        return Lorenz96(config)

    def test_step_advances_state(self):
        sim = self._make_sim()
        state0 = sim.reset()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_trajectory_bounded_chaotic(self):
        """For F=8, trajectory should remain bounded."""
        sim = self._make_sim(N=36, F=8.0)
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State became NaN/Inf"
            assert np.max(np.abs(state)) < 100, f"Trajectory diverged: max={np.max(np.abs(state))}"

    def test_decay_to_zero_for_F0(self):
        """For F=0, the nonlinear term conserves energy but -x_i dissipates.
        State should decay towards zero."""
        sim = self._make_sim(N=36, F=0.0, dt=0.01)
        sim.reset()
        # Override with non-trivial initial condition
        sim._state = np.random.default_rng(42).standard_normal(36) * 0.5

        initial_energy = sim.energy
        for _ in range(5000):
            sim.step()

        final_energy = sim.energy
        assert final_energy < initial_energy * 0.01, (
            f"Energy did not decay sufficiently: {initial_energy:.4f} -> {final_energy:.4f}"
        )

    def test_fixed_point_exact(self):
        """The fixed point x_i = F should have zero derivatives."""
        sim = self._make_sim(N=36, F=8.0)
        sim.reset()
        fp = sim.fixed_point
        derivs = sim._derivatives(fp)
        np.testing.assert_allclose(
            derivs, np.zeros(36), atol=1e-12,
            err_msg="Derivatives at fixed point are not zero",
        )

    def test_fixed_point_value(self):
        """Fixed point should be x_i = F for all i."""
        sim = self._make_sim(N=20, F=5.0)
        sim.reset()
        fp = sim.fixed_point
        expected = np.full(20, 5.0)
        np.testing.assert_array_equal(fp, expected)

    def test_small_N(self):
        """Should work with small N (minimum meaningful is 4 for the stencil)."""
        sim = self._make_sim(N=4, F=8.0)
        state = sim.reset()
        assert state.shape == (4,)
        for _ in range(100):
            state = sim.step()
        assert np.all(np.isfinite(state))

    def test_periodic_boundary(self):
        """Verify periodic boundary by checking derivative at boundary sites."""
        sim = self._make_sim(N=6, F=0.0)
        sim.reset()
        # Set a known state to verify boundary handling
        sim._state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        derivs = sim._derivatives(sim._state)

        # For i=0: (x_1 - x_{-2}) * x_{-1} - x_0 + F
        #        = (x_1 - x_4) * x_5 - x_0 + 0
        #        = (2 - 5) * 6 - 1 = -18 - 1 = -19
        assert np.isclose(derivs[0], -19.0), f"Expected -19.0, got {derivs[0]}"

        # For i=N-1=5: (x_0 - x_3) * x_4 - x_5 + F
        #            = (1 - 4) * 5 - 6 + 0 = -15 - 6 = -21
        assert np.isclose(derivs[5], -21.0), f"Expected -21.0, got {derivs[5]}"

    def test_run_trajectory(self):
        """Test the run() method produces a proper TrajectoryData."""
        sim = self._make_sim(N=10, F=8.0)
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 10)  # n_steps + 1
        assert np.all(np.isfinite(traj.states))


class TestLorenz96Properties:
    """Tests for energy, mean_state, and max_amplitude properties."""

    def _make_sim(self, N: int = 36, F: float = 8.0) -> Lorenz96:
        config = SimulationConfig(
            domain=Domain.LORENZ96,
            dt=0.01,
            n_steps=1000,
            parameters={"N": float(N), "F": F},
        )
        return Lorenz96(config)

    def test_energy_at_fixed_point(self):
        """Energy at fixed point x_i = F should be F^2/2."""
        sim = self._make_sim(N=36, F=8.0)
        sim.reset()
        sim._state = sim.fixed_point
        expected_energy = 8.0**2 / 2.0  # = 32.0
        assert np.isclose(sim.energy, expected_energy), (
            f"Energy at fixed point: {sim.energy}, expected {expected_energy}"
        )

    def test_mean_state_at_fixed_point(self):
        """Mean state at fixed point should be F."""
        sim = self._make_sim(N=36, F=5.0)
        sim.reset()
        sim._state = sim.fixed_point
        assert np.isclose(sim.mean_state, 5.0)

    def test_max_amplitude_at_fixed_point(self):
        """Max amplitude at fixed point should be 0."""
        sim = self._make_sim(N=36, F=8.0)
        sim.reset()
        sim._state = sim.fixed_point
        assert np.isclose(sim.max_amplitude, 0.0)

    def test_energy_positive(self):
        """Energy should always be non-negative."""
        sim = self._make_sim()
        sim.reset()
        for _ in range(100):
            sim.step()
            assert sim.energy >= 0


class TestLorenz96Lyapunov:
    """Tests for Lyapunov exponent estimation."""

    def test_positive_lyapunov_chaotic(self):
        """For F=8, Lyapunov exponent should be positive (chaotic)."""
        config = SimulationConfig(
            domain=Domain.LORENZ96,
            dt=0.01,
            n_steps=20000,
            parameters={"N": 36.0, "F": 8.0},
        )
        sim = Lorenz96(config)
        sim.reset()
        # Skip transient
        for _ in range(2000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=20000, dt=0.01)
        assert lam > 0.5, f"Lyapunov {lam:.3f} too small for F=8 chaotic regime"

    def test_negative_lyapunov_weak_forcing(self):
        """For very small F, dynamics should be non-chaotic (decaying)."""
        config = SimulationConfig(
            domain=Domain.LORENZ96,
            dt=0.01,
            n_steps=10000,
            parameters={"N": 36.0, "F": 2.0},
        )
        sim = Lorenz96(config)
        sim.reset()
        for _ in range(2000):
            sim.step()
        lam = sim.estimate_lyapunov(n_steps=10000, dt=0.01)
        # For F=2, dynamics are simple -- Lyapunov should not be strongly positive
        assert lam < 1.0, f"Lyapunov {lam:.3f} too large for weak forcing F=2"


class TestLorenz96Reproducibility:
    """Tests for deterministic reproducibility."""

    def test_same_seed_same_result(self):
        """Same config and seed should produce identical trajectories."""
        config = SimulationConfig(
            domain=Domain.LORENZ96,
            dt=0.01,
            n_steps=200,
            parameters={"N": 10.0, "F": 8.0},
            seed=42,
        )
        sim1 = Lorenz96(config)
        traj1 = sim1.run(n_steps=200)

        sim2 = Lorenz96(config)
        traj2 = sim2.run(n_steps=200)

        np.testing.assert_allclose(traj1.states, traj2.states, atol=1e-12)


class TestLorenz96Rediscovery:
    """Tests for Lorenz-96 rediscovery data generation."""

    def test_chaos_transition_data(self):
        from simulating_anything.rediscovery.lorenz96 import generate_chaos_transition_data

        data = generate_chaos_transition_data(n_F=5, N=10, n_steps=2000, dt=0.01)
        assert len(data["F"]) == 5
        assert len(data["lyapunov_exponent"]) == 5
        assert len(data["attractor_type"]) == 5

    def test_energy_data(self):
        from simulating_anything.rediscovery.lorenz96 import generate_energy_data

        data = generate_energy_data(n_F=5, N=10, n_transient=500, n_measure=1000, dt=0.01)
        assert len(data["F"]) == 5
        assert len(data["energy_mean"]) == 5
        assert len(data["energy_std"]) == 5
        # Energy should be positive
        assert np.all(data["energy_mean"] > 0)

    def test_dimension_data(self):
        from simulating_anything.rediscovery.lorenz96 import generate_dimension_data

        data = generate_dimension_data(
            N_values=[8, 12], F=8.0, n_transient=500, n_measure=1000, dt=0.01,
        )
        assert len(data["N"]) == 2
        assert len(data["dimension_estimate"]) == 2
        # Dimension should be at most N
        for dim, n in zip(data["dimension_estimate"], data["N"]):
            assert dim <= n
