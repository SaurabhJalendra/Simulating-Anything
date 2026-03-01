"""Tests for the 2D Ising model simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.ising_model import IsingModel2D
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    N: int = 8,
    J: float = 1.0,
    h: float = 0.0,
    T: float = 2.0,
    dt: float = 1.0,
    n_steps: int = 100,
    seed: int = 42,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.ISING_MODEL,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        parameters={"N": float(N), "J": J, "h": h, "T": T},
    )


class TestIsingModelCreation:
    def test_creation(self):
        sim = IsingModel2D(_make_config())
        assert sim.N == 8
        assert sim.J == 1.0
        assert sim.h == 0.0
        assert sim.T == 2.0

    def test_default_params(self):
        config = SimulationConfig(
            domain=Domain.ISING_MODEL,
            dt=1.0,
            n_steps=100,
            parameters={},
        )
        sim = IsingModel2D(config)
        assert sim.N == 16
        assert sim.J == 1.0
        assert sim.h == 0.0
        assert sim.T == 2.0

    def test_custom_params(self):
        sim = IsingModel2D(_make_config(N=32, J=2.0, h=0.5, T=3.0))
        assert sim.N == 32
        assert sim.J == 2.0
        assert sim.h == 0.5
        assert sim.T == 3.0

    def test_lattice_size(self):
        for N in [4, 8, 16]:
            sim = IsingModel2D(_make_config(N=N))
            sim.reset(seed=42)
            lattice = sim.spin_lattice
            assert lattice.shape == (N, N)


class TestIsingModelObserve:
    def test_observe_shape(self):
        N = 8
        sim = IsingModel2D(_make_config(N=N))
        state = sim.reset(seed=42)
        assert state.shape == (N * N,)

    def test_observe_dtype(self):
        sim = IsingModel2D(_make_config())
        state = sim.reset(seed=42)
        assert state.dtype == np.float64

    def test_all_spins_plus_minus_one(self):
        """All spin values must be exactly +1 or -1."""
        sim = IsingModel2D(_make_config(N=16))
        sim.reset(seed=42)
        state = sim.observe()
        assert np.all(np.isin(state, [-1.0, 1.0]))

    def test_spins_remain_valid_after_steps(self):
        """After multiple MC sweeps, spins must still be +1 or -1."""
        sim = IsingModel2D(_make_config(N=8))
        sim.reset(seed=42)
        for _ in range(50):
            sim.step()
        state = sim.observe()
        assert np.all(np.isin(state, [-1.0, 1.0]))


class TestIsingModelPhysics:
    def test_magnetization_bounds(self):
        """Magnetization per spin must be in [-1, 1]."""
        sim = IsingModel2D(_make_config(N=8, T=2.0))
        sim.reset(seed=42)
        for _ in range(100):
            sim.step()
            m = sim.magnetization
            assert -1.0 <= m <= 1.0, f"Magnetization {m} out of bounds"

    def test_energy_bounds_no_field(self):
        """For h=0: -2*J*N^2 <= E <= 2*J*N^2."""
        N = 8
        J = 1.0
        sim = IsingModel2D(_make_config(N=N, J=J, h=0.0, T=2.0))
        sim.reset(seed=42)
        E_max = 2.0 * J * N * N
        for _ in range(50):
            sim.step()
            E = sim.energy
            assert -E_max <= E <= E_max, (
                f"Energy {E} outside bounds [{-E_max}, {E_max}]"
            )

    def test_low_temperature_ordered(self):
        """At very low T, system should be ordered: |m| close to 1."""
        N = 8
        sim = IsingModel2D(_make_config(N=N, T=0.5, n_steps=3000))
        result = sim.measure_equilibrium(
            n_equil_sweeps=500, n_measure_sweeps=500, seed=42,
        )
        # At T=0.5 << T_c~2.27, expect |m| > 0.8
        assert result["magnetization"] > 0.8, (
            f"Low T magnetization {result['magnetization']:.3f} too low"
        )

    def test_high_temperature_disordered(self):
        """At high T, system should be disordered: |m| close to 0."""
        N = 16
        sim = IsingModel2D(_make_config(N=N, T=5.0, n_steps=3000))
        result = sim.measure_equilibrium(
            n_equil_sweeps=500, n_measure_sweeps=500, seed=42,
        )
        # At T=5.0 >> T_c~2.27, expect |m| < 0.3
        assert result["magnetization"] < 0.3, (
            f"High T magnetization {result['magnetization']:.3f} too high"
        )

    def test_step_changes_spins(self):
        """A Monte Carlo sweep should flip at least some spins at moderate T."""
        sim = IsingModel2D(_make_config(N=8, T=2.0))
        s0 = sim.reset(seed=42).copy()
        s1 = sim.step()
        # Not all spins should be the same (very unlikely at T=2)
        assert not np.allclose(s0, s1), "No spins changed in a MC sweep"

    def test_energy_decreases_low_T_from_random(self):
        """Starting from random at low T, energy should decrease on average."""
        sim = IsingModel2D(_make_config(N=8, T=0.5))
        sim.reset(seed=42)
        E_initial = sim.energy

        for _ in range(200):
            sim.step()

        E_final = sim.energy
        # At low T, system should find lower energy states
        assert E_final < E_initial, (
            f"Energy did not decrease: {E_initial:.1f} -> {E_final:.1f}"
        )

    def test_external_field_biases_magnetization(self):
        """Positive h should bias magnetization positive."""
        N = 8
        sim = IsingModel2D(_make_config(N=N, T=2.0, h=2.0))
        sim.reset(seed=42)
        for _ in range(500):
            sim.step()
        # With strong positive field, most spins should be +1
        assert sim.magnetization > 0.5, (
            f"Positive field should give positive m, got {sim.magnetization:.3f}"
        )

    def test_energy_per_spin_property(self):
        """energy_per_spin should equal energy / N^2."""
        N = 8
        sim = IsingModel2D(_make_config(N=N))
        sim.reset(seed=42)
        for _ in range(10):
            sim.step()
        assert abs(sim.energy_per_spin - sim.energy / (N * N)) < 1e-10


class TestIsingModelCriticalTemperature:
    def test_critical_temperature_value(self):
        """T_c should be approximately 2.269 for J=1."""
        T_c = IsingModel2D.critical_temperature(1.0)
        assert abs(T_c - 2.269) < 0.001

    def test_critical_temperature_scaling(self):
        """T_c should scale linearly with J."""
        T_c_1 = IsingModel2D.critical_temperature(1.0)
        T_c_2 = IsingModel2D.critical_temperature(2.0)
        assert abs(T_c_2 / T_c_1 - 2.0) < 1e-10

    def test_onsager_magnetization_zero_above_Tc(self):
        """Onsager formula should give M=0 for T > T_c."""
        T_c = IsingModel2D.critical_temperature(1.0)
        assert IsingModel2D.onsager_magnetization(T_c + 0.1) == 0.0
        assert IsingModel2D.onsager_magnetization(5.0) == 0.0

    def test_onsager_magnetization_positive_below_Tc(self):
        """Onsager formula should give M > 0 for T < T_c."""
        M_low = IsingModel2D.onsager_magnetization(1.0)
        assert M_low > 0.9  # Near saturation at low T
        M_mid = IsingModel2D.onsager_magnetization(2.0)
        assert 0.0 < M_mid < 1.0


class TestIsingModelTrajectory:
    def test_run_trajectory_shape(self):
        N = 8
        n_steps = 50
        sim = IsingModel2D(_make_config(N=N, n_steps=n_steps))
        traj = sim.run(n_steps=n_steps)
        assert traj.states.shape == (n_steps + 1, N * N)

    def test_trajectory_all_valid_spins(self):
        """Every state in the trajectory should have only +1/-1 spins."""
        N = 8
        n_steps = 20
        sim = IsingModel2D(_make_config(N=N, n_steps=n_steps))
        traj = sim.run(n_steps=n_steps)
        assert np.all(np.isin(traj.states, [-1.0, 1.0]))

    def test_reproducibility(self):
        """Same seed should give identical trajectories."""
        cfg = _make_config(N=8, n_steps=20, seed=42)
        sim1 = IsingModel2D(cfg)
        traj1 = sim1.run(n_steps=20)

        sim2 = IsingModel2D(cfg)
        traj2 = sim2.run(n_steps=20)

        assert np.array_equal(traj1.states, traj2.states)


class TestIsingModelRediscovery:
    def test_magnetization_data_generation(self):
        from simulating_anything.rediscovery.ising_model import (
            generate_magnetization_vs_temperature,
        )

        data = generate_magnetization_vs_temperature(
            n_T=5, N=8, n_equil=100, n_measure=200,
        )
        assert len(data["T"]) == 5
        assert len(data["magnetization"]) == 5
        assert len(data["theory_magnetization"]) == 5
        assert data["N"] == 8
        # All magnetizations should be non-negative (absolute value)
        assert np.all(data["magnetization"] >= 0)
        assert np.all(data["magnetization"] <= 1)

    def test_susceptibility_data_generation(self):
        from simulating_anything.rediscovery.ising_model import (
            generate_susceptibility_data,
        )

        data = generate_susceptibility_data(
            n_T=5, N=8, n_equil=100, n_measure=200,
        )
        assert len(data["T"]) == 5
        assert len(data["susceptibility"]) == 5
        # Susceptibility should be non-negative
        assert np.all(data["susceptibility"] >= 0)

    def test_energy_data_generation(self):
        from simulating_anything.rediscovery.ising_model import (
            generate_energy_vs_temperature,
        )

        data = generate_energy_vs_temperature(
            n_T=5, N=8, n_equil=100, n_measure=200,
        )
        assert len(data["T"]) == 5
        assert len(data["energy_per_spin"]) == 5
        # Energy per spin should be in [-2*J, 0] roughly for h=0
        assert np.all(data["energy_per_spin"] <= 0.5)  # generous bound
        assert np.all(data["energy_per_spin"] >= -2.5)
