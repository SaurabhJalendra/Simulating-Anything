"""Tests for the 1D shallow water equations simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.shallow_water import ShallowWater
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    g: float = 9.81,
    h0: float = 1.0,
    N: int = 128,
    L: float = 10.0,
    dt: float = 0.001,
    n_steps: int = 1000,
    perturbation_amplitude: float = 0.1,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.SHALLOW_WATER,
        dt=dt,
        n_steps=n_steps,
        parameters={
            "g": g,
            "h0": h0,
            "N": float(N),
            "L": L,
            "perturbation_amplitude": perturbation_amplitude,
        },
    )


def _make_sim(**kwargs) -> ShallowWater:
    return ShallowWater(_make_config(**kwargs))


class TestShallowWaterCreation:
    def test_creation(self):
        sim = _make_sim()
        assert sim.g == 9.81
        assert sim.h0 == 1.0
        assert sim.N == 128
        assert sim.L == 10.0

    def test_custom_params(self):
        sim = _make_sim(g=5.0, h0=2.0, N=64, L=20.0)
        assert sim.g == 5.0
        assert sim.h0 == 2.0
        assert sim.N == 64
        assert sim.L == 20.0

    def test_default_params(self):
        """Default parameters are used when not specified."""
        config = SimulationConfig(
            domain=Domain.SHALLOW_WATER,
            dt=0.001,
            n_steps=100,
            parameters={},
        )
        sim = ShallowWater(config)
        assert sim.g == 9.81
        assert sim.h0 == 1.0
        assert sim.N == 128

    def test_observe_shape(self):
        """Observe returns [h_1,...,h_N, u_1,...,u_N] with shape (2*N,)."""
        N = 128
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
        N = 64
        n_steps = 50
        sim = _make_sim(N=N, n_steps=n_steps)
        traj = sim.run(n_steps=n_steps)
        assert traj.states.shape == (n_steps + 1, 2 * N)


class TestShallowWaterFlatSurface:
    def test_flat_surface_stays_flat(self):
        """A flat surface with no velocity should remain flat."""
        N = 64
        sim = _make_sim(N=N, perturbation_amplitude=0.0)
        sim.reset()

        h0 = sim.height_field.copy()

        for _ in range(500):
            sim.step()

        h_final = sim.height_field
        assert np.allclose(h_final, h0, atol=1e-12), (
            f"Flat surface drifted: max change = {np.max(np.abs(h_final - h0)):.2e}"
        )


class TestShallowWaterConservation:
    def test_mass_conservation(self):
        """Total mass should be conserved to high accuracy."""
        sim = _make_sim(N=128, dt=0.001)
        sim.reset()
        mass0 = sim.total_mass

        for _ in range(2000):
            sim.step()

        mass_final = sim.total_mass
        rel_drift = abs(mass_final - mass0) / abs(mass0)
        assert rel_drift < 1e-4, (
            f"Mass drift {rel_drift:.2e} too large (mass0={mass0:.6f}, "
            f"final={mass_final:.6f})"
        )

    def test_mass_conservation_various_params(self):
        """Mass conservation holds for different g and h0 values."""
        for g_val, h0_val in [(5.0, 0.5), (15.0, 2.0)]:
            sim = _make_sim(g=g_val, h0=h0_val, N=128, dt=0.001)
            sim.reset()
            mass0 = sim.total_mass

            for _ in range(1000):
                sim.step()

            mass_final = sim.total_mass
            rel_drift = abs(mass_final - mass0) / abs(mass0)
            assert rel_drift < 1e-4, (
                f"Mass drift {rel_drift:.2e} for g={g_val}, h0={h0_val}"
            )

    def test_energy_positive(self):
        """Total energy should be positive."""
        sim = _make_sim()
        sim.reset()
        for _ in range(100):
            sim.step()
        assert sim.total_energy > 0
        assert sim.kinetic_energy >= 0
        assert sim.potential_energy > 0


class TestShallowWaterPhysics:
    def test_wave_propagation(self):
        """A perturbation should propagate outward as gravity waves."""
        N = 256
        sim = _make_sim(N=N, h0=1.0, g=9.81, perturbation_amplitude=0.05)
        sim.reset()

        # The initial bump is centered at L/2
        h_init = sim.height_field.copy()
        center = N // 2
        assert h_init[center] > sim.h0  # Bump is present

        # After some time, the bump should have split and moved away
        for _ in range(500):
            sim.step()

        h_later = sim.height_field
        # The center height should have decreased as waves propagate away
        assert h_later[center] < h_init[center], (
            "Center height should decrease as waves propagate outward"
        )

    def test_wave_speed_approximately_correct(self):
        """Measured wave speed should be approximately sqrt(g*h0).

        We use a small perturbation (linear regime) and track the
        rightward-traveling peak.
        """
        g_val = 9.81
        h0_val = 1.0
        N = 256
        L = 10.0
        dt = 0.001

        sim = _make_sim(
            g=g_val, h0=h0_val, N=N, L=L, dt=dt,
            perturbation_amplitude=0.01,
        )
        sim.reset()

        c_theory = np.sqrt(g_val * h0_val)

        # Wait for the bump to split
        sep_time = (L / 4) / c_theory
        sep_steps = int(sep_time / dt)
        for _ in range(sep_steps):
            sim.step()

        # Track the right-traveling peak
        h = sim.height_field
        mid = N // 2
        right_half = h[mid:]
        peak1_idx = mid + int(np.argmax(right_half))
        pos1 = sim.x[peak1_idx]
        t1 = sep_steps * dt

        # Run more steps
        more_steps = sep_steps
        for _ in range(more_steps):
            sim.step()

        h = sim.height_field
        right_half = h[mid:]
        peak2_idx = mid + int(np.argmax(right_half))
        pos2 = sim.x[peak2_idx]
        t2 = (sep_steps + more_steps) * dt

        delta_x = pos2 - pos1
        if delta_x < -L / 2:
            delta_x += L
        delta_t = t2 - t1

        if delta_t > 0 and abs(delta_x) > 1e-10:
            speed = abs(delta_x) / delta_t
            rel_err = abs(speed - c_theory) / c_theory
            assert rel_err < 0.15, (
                f"Wave speed error {rel_err:.2%}: "
                f"measured={speed:.4f}, theory={c_theory:.4f}"
            )

    def test_height_stays_positive(self):
        """Water depth h should remain positive throughout simulation."""
        sim = _make_sim(N=128, dt=0.001, perturbation_amplitude=0.05)
        sim.reset()

        for _ in range(2000):
            sim.step()
            h = sim.height_field
            assert np.all(h > -1e-10), (
                f"Negative height detected: min(h) = {np.min(h):.6f}"
            )

    def test_wave_speed_property(self):
        """The wave_speed property should return sqrt(g*h0)."""
        sim = _make_sim(g=9.81, h0=2.0)
        sim.reset()
        expected = np.sqrt(9.81 * 2.0)
        assert sim.wave_speed == pytest.approx(expected, rel=1e-10)

    def test_cfl_number(self):
        """CFL number should be computable and less than 1 for stable dt."""
        sim = _make_sim(g=9.81, h0=1.0, N=128, L=10.0, dt=0.001)
        sim.reset()
        cfl = sim.cfl_number()
        assert cfl > 0
        assert cfl < 1.0, f"CFL number {cfl} >= 1, simulation may be unstable"

    def test_mean_height_near_h0(self):
        """Mean height should remain close to h0 (mass conservation implies this)."""
        sim = _make_sim(h0=1.5, N=128, dt=0.001)
        sim.reset()
        for _ in range(500):
            sim.step()
        assert sim.mean_height == pytest.approx(1.5, rel=0.01)

    def test_max_height_decreases_with_time(self):
        """The peak height should decrease as the initial bump disperses.

        Lax-Friedrichs has numerical diffusion that smooths the solution,
        so the peak should decrease from its initial value.
        """
        sim = _make_sim(N=128, dt=0.001, perturbation_amplitude=0.1)
        sim.reset()
        max_h_init = sim.max_height

        for _ in range(1000):
            sim.step()

        max_h_later = sim.max_height
        assert max_h_later < max_h_init, (
            "Max height should decrease as bump disperses"
        )


class TestShallowWaterRediscovery:
    def test_wave_speed_data_generation(self):
        from simulating_anything.rediscovery.shallow_water import (
            generate_wave_speed_data,
        )
        data = generate_wave_speed_data(
            n_g=3, n_h=3, n_steps=500, dt=0.001, N=128,
        )
        assert len(data["g_values"]) == 9  # 3 * 3
        assert len(data["speed_measured"]) == 9
        assert len(data["speed_theory"]) == 9

    def test_mass_conservation_data_generation(self):
        from simulating_anything.rediscovery.shallow_water import (
            generate_mass_conservation_data,
        )
        data = generate_mass_conservation_data(
            n_runs=3, n_steps=500, dt=0.001, N=64,
        )
        assert len(data["relative_drift"]) == 3
        # Mass drift should be small
        assert np.all(data["relative_drift"] < 0.01)
