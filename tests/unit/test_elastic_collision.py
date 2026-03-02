"""Tests for the 1D elastic collision chain simulation and rediscovery."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.elastic_collision import (
    ElasticCollisionSimulation,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    n_particles: int = 5,
    mass: float = 1.0,
    restitution: float = 1.0,
    spacing: float = 2.0,
    init_velocity_0: float = 1.0,
    dt: float = 0.001,
    n_steps: int = 5000,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.ELASTIC_COLLISION,  
        dt=dt,
        n_steps=n_steps,
        parameters={
            "n_particles": float(n_particles),
            "mass": mass,
            "restitution": restitution,
            "spacing": spacing,
            "init_velocity_0": init_velocity_0,
        },
    )


class TestElasticCollisionCreation:
    def test_creation_defaults(self):
        """Default parameters are correctly loaded."""
        sim = ElasticCollisionSimulation(_make_config())
        assert sim.n_particles == 5
        assert sim.restitution == 1.0

    def test_custom_particles(self):
        """Custom particle count is correctly set."""
        sim = ElasticCollisionSimulation(_make_config(n_particles=10))
        assert sim.n_particles == 10

    def test_custom_restitution(self):
        """Custom restitution coefficient is correctly set."""
        sim = ElasticCollisionSimulation(_make_config(restitution=0.5))
        assert sim.restitution == 0.5

    def test_default_params_from_empty(self):
        """Default parameters work when config has empty parameters dict."""
        config = SimulationConfig(
            domain=Domain.ELASTIC_COLLISION,
            dt=0.01,
            n_steps=100,
            parameters={},
        )
        sim = ElasticCollisionSimulation(config)
        assert sim.n_particles == 5
        assert sim.restitution == 1.0

    def test_set_masses(self):
        """set_masses correctly assigns individual masses."""
        sim = ElasticCollisionSimulation(_make_config(n_particles=3))
        sim.set_masses([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(sim.masses, [1.0, 2.0, 3.0])

    def test_set_masses_wrong_length(self):
        """set_masses raises ValueError for wrong length."""
        sim = ElasticCollisionSimulation(_make_config(n_particles=3))
        with pytest.raises(ValueError, match="Expected 3 masses"):
            sim.set_masses([1.0, 2.0])


class TestElasticCollisionState:
    def test_state_shape(self):
        """State has shape (2*N,) with interleaved [x, v] pairs."""
        N = 5
        sim = ElasticCollisionSimulation(_make_config(n_particles=N))
        state = sim.reset()
        assert state.shape == (2 * N,)

    def test_state_shape_various_n(self):
        """State shape is correct for various particle counts."""
        for N in [2, 3, 7, 10]:
            sim = ElasticCollisionSimulation(_make_config(n_particles=N))
            state = sim.reset()
            assert state.shape == (2 * N,), f"Failed for N={N}"

    def test_initial_positions_ordered(self):
        """Initial positions are in ascending order."""
        sim = ElasticCollisionSimulation(_make_config(n_particles=5))
        sim.reset()
        pos = sim.positions
        assert np.all(np.diff(pos) > 0), "Positions not ordered"

    def test_initial_velocities(self):
        """First particle has init_velocity_0, others are at rest."""
        sim = ElasticCollisionSimulation(
            _make_config(n_particles=5, init_velocity_0=3.0)
        )
        sim.reset()
        vel = sim.velocities
        assert vel[0] == 3.0
        assert np.all(vel[1:] == 0.0)

    def test_observe_matches_state(self):
        """observe() returns same data as state from reset/step."""
        sim = ElasticCollisionSimulation(_make_config())
        state = sim.reset()
        obs = sim.observe()
        np.testing.assert_array_equal(state, obs)

    def test_step_changes_state(self):
        """State changes after a step."""
        sim = ElasticCollisionSimulation(_make_config())
        s0 = sim.reset()
        s1 = sim.step()
        assert not np.allclose(s0, s1)


class TestMomentumConservation:
    def test_momentum_conserved_equal_mass(self):
        """Total momentum is conserved for equal-mass elastic collisions."""
        sim = ElasticCollisionSimulation(
            _make_config(n_particles=5, mass=1.0, restitution=1.0, dt=0.001)
        )
        sim.reset()
        p0 = sim.compute_momentum()

        for _ in range(5000):
            sim.step()

        pf = sim.compute_momentum()
        np.testing.assert_allclose(
            pf, p0, rtol=1e-6,
            err_msg="Momentum not conserved (equal mass, elastic)",
        )

    def test_momentum_conserved_unequal_mass(self):
        """Total momentum is conserved with different masses."""
        sim = ElasticCollisionSimulation(
            _make_config(n_particles=3, mass=1.0, restitution=1.0, dt=0.001)
        )
        sim.reset()
        sim.set_masses([1.0, 3.0, 0.5])
        sim._vel[0] = 2.0
        sim._state = sim._pack_state()
        p0 = sim.compute_momentum()

        for _ in range(5000):
            sim.step()

        pf = sim.compute_momentum()
        np.testing.assert_allclose(
            pf, p0, rtol=1e-6,
            err_msg="Momentum not conserved (unequal mass)",
        )

    def test_momentum_conserved_inelastic(self):
        """Momentum is conserved even for inelastic collisions (e < 1)."""
        sim = ElasticCollisionSimulation(
            _make_config(
                n_particles=5, mass=1.0, restitution=0.5,
                init_velocity_0=2.0, dt=0.001,
            )
        )
        sim.reset()
        p0 = sim.compute_momentum()

        for _ in range(3000):
            sim.step()

        pf = sim.compute_momentum()
        np.testing.assert_allclose(
            pf, p0, rtol=1e-5,
            err_msg="Momentum not conserved (inelastic)",
        )

    def test_momentum_value(self):
        """compute_momentum returns correct value."""
        sim = ElasticCollisionSimulation(
            _make_config(n_particles=3, mass=2.0, init_velocity_0=3.0)
        )
        sim.reset()
        # p = m1*v1 + m2*v2 + m3*v3 = 2*3 + 2*0 + 2*0 = 6
        assert abs(sim.compute_momentum() - 6.0) < 1e-10


class TestEnergyConservation:
    def test_energy_conserved_elastic(self):
        """Kinetic energy is conserved for elastic collisions (e=1.0)."""
        sim = ElasticCollisionSimulation(
            _make_config(
                n_particles=5, mass=1.0, restitution=1.0,
                init_velocity_0=2.0, dt=0.001,
            )
        )
        sim.reset()
        KE0 = sim.compute_kinetic_energy()

        for _ in range(5000):
            sim.step()

        KEf = sim.compute_kinetic_energy()
        np.testing.assert_allclose(
            KEf, KE0, rtol=1e-6,
            err_msg="Energy not conserved (elastic)",
        )

    def test_energy_conserved_unequal_mass_elastic(self):
        """Energy is conserved for elastic collisions with different masses."""
        sim = ElasticCollisionSimulation(
            _make_config(n_particles=3, mass=1.0, restitution=1.0, dt=0.001)
        )
        sim.reset()
        sim.set_masses([2.0, 1.0, 3.0])
        sim._vel[0] = 2.0
        sim._state = sim._pack_state()
        KE0 = sim.compute_kinetic_energy()

        for _ in range(5000):
            sim.step()

        KEf = sim.compute_kinetic_energy()
        np.testing.assert_allclose(
            KEf, KE0, rtol=1e-5,
            err_msg="Energy not conserved (unequal mass, elastic)",
        )

    def test_energy_loss_inelastic(self):
        """Kinetic energy decreases for inelastic collisions (e < 1)."""
        sim = ElasticCollisionSimulation(
            _make_config(
                n_particles=5, mass=1.0, restitution=0.5,
                init_velocity_0=2.0, dt=0.001,
            )
        )
        sim.reset()
        KE0 = sim.compute_kinetic_energy()

        for _ in range(5000):
            sim.step()

        KEf = sim.compute_kinetic_energy()
        assert KEf < KE0, (
            f"Energy should decrease for inelastic: "
            f"KE0={KE0:.4f}, KEf={KEf:.4f}"
        )

    def test_energy_value(self):
        """compute_kinetic_energy returns correct value."""
        sim = ElasticCollisionSimulation(
            _make_config(n_particles=3, mass=2.0, init_velocity_0=3.0)
        )
        sim.reset()
        # KE = 0.5*m*v^2 = 0.5*2*9 = 9.0
        assert abs(sim.compute_kinetic_energy() - 9.0) < 1e-10

    def test_more_inelastic_loses_more_energy(self):
        """Lower restitution leads to more energy loss."""
        KE_ratios = []
        for e_val in [0.3, 0.7, 1.0]:
            sim = ElasticCollisionSimulation(
                _make_config(
                    n_particles=5, mass=1.0, restitution=e_val,
                    init_velocity_0=2.0, dt=0.001,
                )
            )
            sim.reset()
            KE0 = sim.compute_kinetic_energy()
            for _ in range(5000):
                sim.step()
            KEf = sim.compute_kinetic_energy()
            KE_ratios.append(KEf / KE0)

        # Higher restitution should retain more energy
        assert KE_ratios[0] <= KE_ratios[1] + 0.01
        assert KE_ratios[1] <= KE_ratios[2] + 0.01


class TestNewtonsCradle:
    def test_cradle_setup(self):
        """Newton's cradle setup creates correct configuration."""
        sim = ElasticCollisionSimulation(
            _make_config(n_particles=5, init_velocity_0=1.0)
        )
        sim.reset()
        sim.newtons_cradle_setup(n_moving=1)

        vel = sim.velocities
        assert vel[0] == 1.0, "First particle should be moving"
        assert np.all(vel[1:] == 0.0), "Other particles should be stationary"

    def test_cradle_velocity_transfer(self):
        """In Newton's cradle, last ball gets nearly all the velocity."""
        sim = ElasticCollisionSimulation(
            _make_config(
                n_particles=5, mass=1.0, restitution=1.0,
                init_velocity_0=1.0, dt=0.0005,
            )
        )
        sim.reset()
        sim.newtons_cradle_setup(n_moving=1)

        for _ in range(20000):
            sim.step()

        vel = sim.velocities
        # Last ball should have velocity close to 1.0
        assert abs(vel[-1] - 1.0) < 0.1, (
            f"Last ball velocity {vel[-1]:.4f} should be near 1.0"
        )
        # Intermediate balls should be nearly stationary
        assert np.all(np.abs(vel[:-1]) < 0.15), (
            f"Intermediate velocities should be near 0: {vel[:-1]}"
        )

    def test_cradle_3_particles(self):
        """Newton's cradle works for 3 particles."""
        sim = ElasticCollisionSimulation(
            _make_config(
                n_particles=3, mass=1.0, restitution=1.0,
                init_velocity_0=1.0, dt=0.0005,
            )
        )
        sim.reset()
        sim.newtons_cradle_setup(n_moving=1)

        for _ in range(15000):
            sim.step()

        vel = sim.velocities
        assert abs(vel[-1] - 1.0) < 0.1


class TestTwoBodyCollision:
    def test_equal_mass_velocity_swap(self):
        """Two equal-mass particles swap velocities in elastic collision."""
        sim = ElasticCollisionSimulation(
            _make_config(
                n_particles=2, mass=1.0, restitution=1.0,
                init_velocity_0=2.0, spacing=1.0, dt=0.0005,
            )
        )
        sim.reset()
        v0_1 = sim.velocities[0]

        for _ in range(10000):
            sim.step()

        vel = sim.velocities
        # After collision: v1 should be ~0, v2 should be ~v0_1
        assert abs(vel[0]) < 0.1, f"v1 should be ~0 after swap, got {vel[0]}"
        assert abs(vel[1] - v0_1) < 0.1, (
            f"v2 should be ~{v0_1} after swap, got {vel[1]}"
        )

    def test_heavy_hits_light(self):
        """Heavy particle hitting light stationary: both move right."""
        sim = ElasticCollisionSimulation(
            _make_config(
                n_particles=2, mass=1.0, restitution=1.0,
                init_velocity_0=1.0, spacing=1.0, dt=0.0005,
            )
        )
        sim.reset()
        sim.set_masses([5.0, 1.0])
        sim._vel[0] = 1.0
        sim._vel[1] = 0.0
        sim._state = sim._pack_state()

        p0 = sim.compute_momentum()
        KE0 = sim.compute_kinetic_energy()

        for _ in range(10000):
            sim.step()

        vel = sim.velocities
        # Heavy particle should still move right (slowed down)
        assert vel[0] > 0, f"Heavy particle should still move right: {vel[0]}"
        # Light particle should move right faster
        assert vel[1] > vel[0], (
            f"Light should be faster: v_light={vel[1]}, v_heavy={vel[0]}"
        )

        # Check conservation
        pf = sim.compute_momentum()
        KEf = sim.compute_kinetic_energy()
        np.testing.assert_allclose(pf, p0, rtol=1e-5)
        np.testing.assert_allclose(KEf, KE0, rtol=1e-5)

    def test_light_hits_heavy(self):
        """Light particle hitting heavy stationary: light bounces back."""
        sim = ElasticCollisionSimulation(
            _make_config(
                n_particles=2, mass=1.0, restitution=1.0,
                init_velocity_0=1.0, spacing=1.0, dt=0.0005,
            )
        )
        sim.reset()
        sim.set_masses([1.0, 5.0])
        sim._vel[0] = 1.0
        sim._vel[1] = 0.0
        sim._state = sim._pack_state()

        for _ in range(10000):
            sim.step()

        vel = sim.velocities
        # Light particle should bounce back (negative velocity)
        assert vel[0] < 0, f"Light should bounce back: {vel[0]}"
        # Heavy particle should move right slowly
        assert vel[1] > 0, f"Heavy should move right: {vel[1]}"

    def test_two_body_elastic_formula(self):
        """Verify the elastic collision formula: v1' = ((m1-m2)v1+2m2*v2)/(m1+m2)."""
        m1, m2 = 3.0, 1.0
        v1_init, v2_init = 2.0, 0.0

        sim = ElasticCollisionSimulation(
            _make_config(
                n_particles=2, mass=1.0, restitution=1.0,
                init_velocity_0=v1_init, spacing=2.0, dt=0.0005,
            )
        )
        sim.reset()
        sim.set_masses([m1, m2])
        sim._vel[0] = v1_init
        sim._vel[1] = v2_init
        sim._state = sim._pack_state()

        for _ in range(15000):
            sim.step()

        vel = sim.velocities
        M = m1 + m2
        v1_theory = ((m1 - m2) * v1_init + 2 * m2 * v2_init) / M
        v2_theory = ((m2 - m1) * v2_init + 2 * m1 * v1_init) / M

        np.testing.assert_allclose(
            vel[0], v1_theory, atol=0.15,
            err_msg=f"v1: {vel[0]:.4f} vs theory {v1_theory:.4f}",
        )
        np.testing.assert_allclose(
            vel[1], v2_theory, atol=0.15,
            err_msg=f"v2: {vel[1]:.4f} vs theory {v2_theory:.4f}",
        )


class TestCollisionDetection:
    def test_no_collisions_initially(self):
        """No collisions detected right after reset (particles well spaced)."""
        sim = ElasticCollisionSimulation(
            _make_config(n_particles=5, spacing=2.0)
        )
        sim.reset()
        collisions = sim.detect_collisions()
        assert len(collisions) == 0

    def test_collision_after_approach(self):
        """Collision detected when particles overlap."""
        sim = ElasticCollisionSimulation(
            _make_config(n_particles=2, spacing=0.5, dt=0.001)
        )
        sim.reset()
        # Place particles so they overlap
        sim._pos[0] = 1.0
        sim._pos[1] = 0.5
        collisions = sim.detect_collisions()
        assert len(collisions) == 1
        assert collisions[0] == (0, 1)


class TestTrajectoryCollection:
    def test_run_returns_trajectory(self):
        """run() returns valid TrajectoryData."""
        config = _make_config(n_particles=3, n_steps=100)
        sim = ElasticCollisionSimulation(config)
        traj = sim.run(n_steps=100)
        assert traj.states.shape == (101, 6)  # 100+1 steps, 2*3 state
        assert np.all(np.isfinite(traj.states))

    def test_trajectory_timestamps(self):
        """Trajectory timestamps are correct."""
        config = _make_config(n_particles=2, dt=0.01, n_steps=50)
        sim = ElasticCollisionSimulation(config)
        traj = sim.run(n_steps=50)
        assert len(traj.timestamps) == 51
        np.testing.assert_allclose(
            traj.timestamps[-1], 50 * 0.01, rtol=1e-10,
        )


class TestReproducibility:
    def test_deterministic(self):
        """Same config produces identical trajectories."""
        config = _make_config(n_particles=5, init_velocity_0=2.0)
        sim1 = ElasticCollisionSimulation(config)
        sim2 = ElasticCollisionSimulation(config)
        sim1.reset()
        sim2.reset()

        for _ in range(100):
            s1 = sim1.step()
            s2 = sim2.step()
            np.testing.assert_array_equal(s1, s2)

    def test_state_finite(self):
        """State remains finite throughout simulation."""
        sim = ElasticCollisionSimulation(
            _make_config(n_particles=5, init_velocity_0=5.0, dt=0.001)
        )
        sim.reset()
        for _ in range(5000):
            state = sim.step()
            assert np.all(np.isfinite(state)), "State has non-finite values"


class TestSetState:
    def test_set_state(self):
        """set_state correctly overrides positions and velocities."""
        sim = ElasticCollisionSimulation(_make_config(n_particles=3))
        sim.reset()
        new_pos = np.array([0.0, 5.0, 10.0])
        new_vel = np.array([1.0, -1.0, 0.5])
        sim.set_state(new_pos, new_vel)
        np.testing.assert_array_equal(sim.positions, new_pos)
        np.testing.assert_array_equal(sim.velocities, new_vel)

    def test_positions_property(self):
        """positions property returns a copy."""
        sim = ElasticCollisionSimulation(_make_config(n_particles=3))
        sim.reset()
        pos = sim.positions
        pos[0] = 999.0  # modify copy
        assert sim.positions[0] != 999.0, "positions should return a copy"

    def test_velocities_property(self):
        """velocities property returns a copy."""
        sim = ElasticCollisionSimulation(_make_config(n_particles=3))
        sim.reset()
        vel = sim.velocities
        vel[0] = 999.0  # modify copy
        assert sim.velocities[0] != 999.0, "velocities should return a copy"


class TestRediscovery:
    def test_momentum_conservation_data(self):
        """Momentum conservation data generation works."""
        from simulating_anything.rediscovery.elastic_collision import (
            generate_momentum_conservation_data,
        )

        data = generate_momentum_conservation_data(
            n_trials=3, n_particles=3, n_steps=1000, dt=0.001,
        )
        assert len(data["p_initial"]) == 3
        assert len(data["p_final"]) == 3
        assert len(data["relative_drift"]) == 3
        # Drift should be small for elastic collisions
        assert np.all(data["relative_drift"] < 0.01)

    def test_energy_conservation_data(self):
        """Energy conservation data generation works."""
        from simulating_anything.rediscovery.elastic_collision import (
            generate_energy_conservation_data,
        )

        data = generate_energy_conservation_data(
            n_trials=3, n_particles=3, n_steps=1000, dt=0.001,
        )
        assert len(data["KE_initial"]) == 3
        assert len(data["KE_final"]) == 3
        assert np.all(data["relative_drift"] < 0.01)

    def test_newtons_cradle_data(self):
        """Newton's cradle data generation works."""
        from simulating_anything.rediscovery.elastic_collision import (
            generate_newtons_cradle_data,
        )

        data = generate_newtons_cradle_data(
            n_particles=3, n_steps=5000, dt=0.001,
        )
        assert "transfer_ratio" in data
        assert "max_intermediate_v" in data
        assert data["transfer_ratio"] > 0.5  # should transfer most velocity

    def test_energy_loss_vs_restitution_data(self):
        """Energy loss vs restitution data generation works."""
        from simulating_anything.rediscovery.elastic_collision import (
            generate_energy_loss_vs_restitution_data,
        )

        data = generate_energy_loss_vs_restitution_data(
            n_e_values=5, n_particles=3, n_steps=1000, dt=0.001,
        )
        assert len(data["e_values"]) == 5
        assert len(data["KE_ratio"]) == 5
        # At e=1.0 (last value), KE should be conserved
        assert data["KE_ratio"][-1] > 0.95

    def test_two_body_collision_data(self):
        """Two-body collision data generation works."""
        from simulating_anything.rediscovery.elastic_collision import (
            generate_two_body_collision_data,
        )

        data = generate_two_body_collision_data(
            n_mass_ratios=5, dt=0.0005, n_steps=5000,
        )
        assert len(data["mass_ratio"]) == 5
        assert len(data["v1_measured"]) == 5
        assert len(data["v2_measured"]) == 5
        assert len(data["v1_theory"]) == 5
        assert len(data["v2_theory"]) == 5
