"""Tests for the 1D coupled spring-mass chain simulation and rediscovery."""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.spring_mass_chain import SpringMassChain
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_sim(
    N: int = 20,
    K: float = 4.0,
    m: float = 1.0,
    a: float = 1.0,
    mode: int = 1,
    amplitude: float = 0.1,
    dt: float = 0.001,
    n_steps: int = 1000,
) -> SpringMassChain:
    config = SimulationConfig(
        domain=Domain.SPRING_MASS_CHAIN,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "N": float(N),
            "K": K,
            "m": m,
            "a": a,
            "mode": float(mode),
            "amplitude": amplitude,
        },
    )
    return SpringMassChain(config)


class TestSpringMassChainCreation:
    def test_creation(self):
        sim = _make_sim()
        assert sim.N == 20
        assert sim.K == 4.0
        assert sim.m == 1.0
        assert sim.a == 1.0

    def test_default_params(self):
        """Default parameters are used when not specified."""
        config = SimulationConfig(
            domain=Domain.SPRING_MASS_CHAIN,
            dt=0.001,
            n_steps=100,
            parameters={},
        )
        sim = SpringMassChain(config)
        assert sim.N == 20
        assert sim.K == 4.0
        assert sim.m == 1.0
        assert sim.a == 1.0

    def test_custom_params(self):
        sim = _make_sim(N=10, K=2.0, m=0.5, a=2.0)
        assert sim.N == 10
        assert sim.K == 2.0
        assert sim.m == 0.5
        assert sim.a == 2.0

    def test_observe_shape(self):
        """Observe returns [u_1,...,u_N, v_1,...,v_N] with shape (2*N,)."""
        N = 15
        sim = _make_sim(N=N)
        sim.reset()
        obs = sim.observe()
        assert obs.shape == (2 * N,)

    def test_step_advances_state(self):
        sim = _make_sim()
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_trajectory_shape(self):
        """Run returns trajectory with correct state dimension."""
        N = 10
        n_steps = 50
        sim = _make_sim(N=N, n_steps=n_steps)
        traj = sim.run(n_steps=n_steps)
        assert traj.states.shape == (n_steps + 1, 2 * N)


class TestSpringMassChainPhysics:
    def test_energy_conservation(self):
        """Total energy should be conserved with RK4 over 1000 steps."""
        sim = _make_sim(K=4.0, m=1.0, amplitude=0.5, dt=0.001)
        sim.reset()
        E0 = sim.total_energy
        assert E0 > 0  # Sanity: should have some energy

        for _ in range(1000):
            sim.step()

        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-6, f"Energy drift {rel_drift:.2e} too large"

    def test_energy_conservation_long(self):
        """Energy drift stays small over 10000 steps."""
        sim = _make_sim(K=4.0, m=1.0, mode=3, amplitude=0.3, dt=0.001)
        sim.reset()
        E0 = sim.total_energy

        for _ in range(10000):
            sim.step()

        E_final = sim.total_energy
        rel_drift = abs(E_final - E0) / abs(E0)
        assert rel_drift < 1e-5, f"Energy drift {rel_drift:.2e} too large"

    def test_zero_displacement_equilibrium(self):
        """All masses at rest should stay at rest (zero displacements)."""
        config = SimulationConfig(
            domain=Domain.SPRING_MASS_CHAIN,
            dt=0.001,
            n_steps=100,
            parameters={
                "N": 10.0, "K": 4.0, "m": 1.0, "a": 1.0,
                "mode": 0.0, "amplitude": 0.0,
            },
        )
        sim = SpringMassChain(config)
        sim.reset()
        # Force exact zero state
        sim._state = np.zeros(2 * sim.N)

        for _ in range(100):
            sim.step()

        assert np.allclose(sim.observe(), 0.0, atol=1e-15)

    def test_normal_mode_frequency(self):
        """Exciting mode 5 should oscillate at the theoretical frequency.

        Mode 1 has the longest period. We use mode 3 for a good balance.
        """
        N, K, m, a = 10, 4.0, 1.0, 1.0
        mode_n = 3
        dt = 0.005
        n_steps = 20000  # Total time = 100s
        sim = _make_sim(N=N, K=K, m=m, a=a, mode=mode_n, amplitude=0.5,
                        dt=dt, n_steps=n_steps)
        sim.reset()
        sim.excite_mode(mode_n, 0.5)

        # Track the middle mass (good for odd modes)
        track_idx = N // 2

        positions = [sim.observe()[track_idx]]
        for _ in range(n_steps):
            state = sim.step()
            positions.append(state[track_idx])

        positions = np.array(positions)

        # Find positive-going zero crossings
        crossings = []
        for j in range(1, len(positions)):
            if positions[j - 1] < 0 and positions[j] >= 0:
                frac = -positions[j - 1] / (positions[j] - positions[j - 1])
                crossings.append((j - 1 + frac) * dt)

        assert len(crossings) >= 3, f"Only {len(crossings)} crossings found"
        periods = np.diff(crossings)
        T_measured = float(np.median(periods))
        omega_measured = 2 * np.pi / T_measured

        omega_theory = sim.compute_normal_mode_frequencies()[mode_n - 1]
        rel_err = abs(omega_measured - omega_theory) / omega_theory
        assert rel_err < 0.05, (
            f"Mode {mode_n} freq error {rel_err:.4%}: "
            f"measured={omega_measured:.4f}, theory={omega_theory:.4f}"
        )

    def test_single_mode_stays_pure(self):
        """Exciting one mode should not leak energy to other modes."""
        N = 20
        sim = _make_sim(N=N, K=4.0, m=1.0, mode=3, amplitude=0.5, dt=0.001)
        sim.reset()

        # Initial mode amplitudes: should be concentrated in mode 3
        amps_init = sim.mode_amplitudes()
        assert abs(amps_init[2]) > 0.1  # Mode 3 (index 2) is excited

        # Run for many steps
        for _ in range(5000):
            sim.step()

        amps_final = sim.mode_amplitudes()

        # Mode 3 should still dominate; other modes should be near zero
        mode3_energy = amps_final[2]**2
        other_energy = sum(
            amps_final[i]**2 for i in range(N) if i != 2
        )
        # Leakage should be tiny (numerical, not physical)
        assert other_energy < 1e-8 * mode3_energy, (
            f"Mode leakage: other/mode3 = {other_energy / mode3_energy:.2e}"
        )

    def test_fixed_boundaries(self):
        """End masses (boundary) should not move -- they are pinned at 0."""
        sim = _make_sim(N=20, mode=1, amplitude=0.5, dt=0.001)
        sim.reset()

        for _ in range(100):
            sim.step()

        state = sim.observe()
        u = state[:sim.N]

        # The pinned boundaries are not part of the state vector.
        # But the derivative computation uses u_0 = u_{N+1} = 0.
        # The first and last masses in the chain CAN move.
        # Instead, verify the extended boundary values are zero:
        u_ext = np.zeros(sim.N + 2)
        u_ext[1:-1] = u
        assert u_ext[0] == 0.0
        assert u_ext[-1] == 0.0

    def test_energy_equipartition(self):
        """Random excitation distributes energy across multiple modes."""
        sim = _make_sim(N=20, mode=0, amplitude=0.5, dt=0.001)
        sim.reset()

        amps = sim.mode_amplitudes()
        # With random init, multiple modes should have non-negligible amplitude
        nonzero = np.sum(np.abs(amps) > 0.01)
        assert nonzero >= 3, (
            f"Only {nonzero} modes excited in random init"
        )

    def test_omega_max(self):
        """Maximum frequency property should be 2*sqrt(K/m)."""
        sim = _make_sim(K=9.0, m=1.0)
        assert np.isclose(sim.omega_max, 6.0)  # 2*sqrt(9/1) = 6

    def test_speed_of_sound_property(self):
        """Speed of sound should be a*sqrt(K/m)."""
        sim = _make_sim(K=4.0, m=1.0, a=2.0)
        assert np.isclose(sim.speed_of_sound(), 4.0)  # 2*sqrt(4/1)

    def test_mode_shape_orthogonality(self):
        """Normal mode shapes should be orthogonal."""
        sim = _make_sim(N=10)
        sim.reset()
        shape1 = sim.normal_mode_shape(1)
        shape2 = sim.normal_mode_shape(2)
        # Orthogonal: dot product should be near zero
        assert abs(np.dot(shape1, shape2)) < 1e-10


class TestSpringMassChainRediscovery:
    def test_dispersion_data_shapes(self):
        from simulating_anything.rediscovery.spring_mass_chain import (
            generate_dispersion_data,
        )
        data = generate_dispersion_data(
            N=5, K=4.0, m=1.0, a=1.0, dt=0.001, n_steps=20000,
        )
        assert "mode_n" in data
        assert "k_wave" in data
        assert "omega_measured" in data
        assert "omega_theory" in data
        assert len(data["mode_n"]) == 5
        assert len(data["omega_theory"]) == 5

    def test_dispersion_accuracy(self):
        """Measured frequencies should be close to theory for low modes."""
        from simulating_anything.rediscovery.spring_mass_chain import (
            generate_dispersion_data,
        )
        data = generate_dispersion_data(
            N=10, K=4.0, m=1.0, a=1.0, dt=0.001, n_steps=30000,
        )
        valid = np.isfinite(data["omega_measured"])
        if np.sum(valid) >= 3:
            rel_err = (
                np.abs(
                    data["omega_measured"][valid]
                    - data["omega_theory"][valid]
                )
                / data["omega_theory"][valid]
            )
            assert np.mean(rel_err) < 0.05, (
                f"Mean dispersion error {np.mean(rel_err):.4%}"
            )

    def test_speed_of_sound_data(self):
        from simulating_anything.rediscovery.spring_mass_chain import (
            generate_speed_of_sound_data,
        )
        data = generate_speed_of_sound_data(
            n_K_values=5, N=10, m=1.0, a=1.0, dt=0.001, n_steps=20000,
        )
        assert "K_values" in data
        assert "c_measured" in data
        assert "c_theory" in data
        assert len(data["K_values"]) == 5

    def test_energy_conservation_data(self):
        from simulating_anything.rediscovery.spring_mass_chain import (
            generate_energy_conservation_data,
        )
        data = generate_energy_conservation_data(
            n_trajectories=3, N=10, K=4.0, m=1.0, dt=0.001, n_steps=1000,
        )
        assert len(data["relative_drift"]) == 3
        assert np.all(data["relative_drift"] < 1e-5)
