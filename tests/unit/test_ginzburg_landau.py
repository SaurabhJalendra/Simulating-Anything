"""Tests for the Complex Ginzburg-Landau equation simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.ginzburg_landau import GinzburgLandau
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(
    c1: float = 1.0,
    c2: float = -1.2,
    L: float = 50.0,
    N: int = 128,
    dt: float = 0.05,
    n_steps: int = 100,
) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.GINZBURG_LANDAU,
        dt=dt,
        n_steps=n_steps,
        parameters={"c1": c1, "c2": c2, "L": L, "N": float(N)},
    )


class TestGinzburgLandauCreation:
    def test_initial_state_shape(self):
        sim = GinzburgLandau(_make_config(N=128))
        state = sim.reset()
        assert state.shape == (256,)  # 2*N: Re and Im parts

    def test_default_parameters(self):
        sim = GinzburgLandau(_make_config())
        assert sim.c1 == 1.0
        assert sim.c2 == -1.2
        assert sim.L == 50.0
        assert sim.N == 128

    def test_custom_parameters(self):
        sim = GinzburgLandau(_make_config(c1=2.5, c2=-0.8, L=100.0, N=64))
        assert sim.c1 == 2.5
        assert sim.c2 == -0.8
        assert sim.L == 100.0
        assert sim.N == 64

    def test_state_is_real_valued(self):
        sim = GinzburgLandau(_make_config())
        state = sim.reset()
        assert state.dtype in (np.float64, np.float32)


class TestGinzburgLandauDynamics:
    def test_step_advances_state(self):
        sim = GinzburgLandau(_make_config())
        state0 = sim.reset().copy()
        state1 = sim.step()
        assert not np.allclose(state0, state1)

    def test_observe_matches_state(self):
        sim = GinzburgLandau(_make_config())
        sim.reset()
        sim.step()
        obs = sim.observe()
        assert obs.shape == (256,)
        np.testing.assert_array_equal(obs, sim._state)

    def test_amplitude_bounded_no_blowup(self):
        """Amplitude should remain finite -- no blow-up over many steps."""
        sim = GinzburgLandau(_make_config(dt=0.05, n_steps=500))
        sim.reset()
        for _ in range(500):
            sim.step()
            assert np.all(np.isfinite(sim.observe())), "State became NaN/Inf"
            assert sim.amplitude < 100, f"Amplitude diverged: {sim.amplitude}"

    def test_energy_stays_finite(self):
        """Energy (L2 norm) should remain finite."""
        sim = GinzburgLandau(_make_config(dt=0.05, n_steps=200))
        sim.reset()
        for _ in range(200):
            sim.step()
            assert np.isfinite(sim.energy), f"Energy became non-finite: {sim.energy}"

    def test_real_gl_stable_uniform(self):
        """For c1=0, c2=0 (real GL), the uniform |A|=1 solution should be stable.

        The real GL has no Benjamin-Feir instability, so starting near |A|=1
        should remain close to |A|=1.
        """
        sim = GinzburgLandau(_make_config(c1=0.0, c2=0.0, dt=0.05, N=64))
        sim.reset()
        # Run to steady state
        for _ in range(500):
            sim.step()
        # Amplitude should be close to 1
        assert abs(sim.amplitude - 1.0) < 0.05, (
            f"Uniform solution not stable: <|A|>={sim.amplitude}"
        )
        # Spatial variation should be small
        assert sim.spatial_std() < 0.02, (
            f"Too much spatial variation: std={sim.spatial_std()}"
        )

    def test_benjamin_feir_unstable_becomes_nonuniform(self):
        """When 1+c1*c2 < 0, the uniform solution should become spatially non-uniform.

        Using c1=2, c2=-1.2 gives 1+c1*c2 = 1-2.4 = -1.4 < 0 (BF unstable).
        """
        sim = GinzburgLandau(_make_config(c1=2.0, c2=-1.2, dt=0.05, N=128))
        assert sim.is_benjamin_feir_unstable
        sim.reset()
        # Run long enough for instability to develop
        for _ in range(2000):
            sim.step()
        # Spatial std should be non-trivial
        assert sim.spatial_std() > 0.01, (
            f"Expected non-uniform solution but std={sim.spatial_std()}"
        )


class TestGinzburgLandauProperties:
    def test_benjamin_feir_parameter(self):
        sim = GinzburgLandau(_make_config(c1=2.0, c2=-1.2))
        assert sim.benjamin_feir_parameter == pytest.approx(1.0 + 2.0 * (-1.2))
        assert sim.benjamin_feir_parameter < 0

    def test_benjamin_feir_stable(self):
        sim = GinzburgLandau(_make_config(c1=0.5, c2=0.5))
        assert sim.benjamin_feir_parameter == pytest.approx(1.25)
        assert not sim.is_benjamin_feir_unstable

    def test_phase_coherence_near_one_for_uniform(self):
        """Phase coherence should be close to 1 for the initial near-uniform state."""
        sim = GinzburgLandau(_make_config(c1=0.0, c2=0.0))
        sim.reset()
        # Initial state is A ~ 1 + noise, so phases are near 0
        assert sim.phase_coherence > 0.9

    def test_amplitude_near_one_initially(self):
        """Initial amplitude should be near 1 (uniform A=1 + small noise)."""
        sim = GinzburgLandau(_make_config())
        sim.reset()
        assert abs(sim.amplitude - 1.0) < 0.1

    def test_count_phase_defects_nonnegative(self):
        sim = GinzburgLandau(_make_config())
        sim.reset()
        assert sim.count_phase_defects() >= 0


class TestGinzburgLandauTrajectory:
    def test_trajectory_reproducibility(self):
        """Same seed should give identical trajectories."""
        sim1 = GinzburgLandau(_make_config(N=64))
        sim1.reset(seed=123)
        for _ in range(50):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = GinzburgLandau(_make_config(N=64))
        sim2.reset(seed=123)
        for _ in range(50):
            sim2.step()
        state2 = sim2.observe().copy()

        np.testing.assert_array_equal(state1, state2)

    def test_run_returns_trajectory(self):
        sim = GinzburgLandau(_make_config(N=64, n_steps=50))
        traj = sim.run(n_steps=50)
        assert traj.states.shape == (51, 128)  # 51 frames, 2*64 state dim

    def test_different_seeds_differ(self):
        """Different seeds should produce different trajectories."""
        sim1 = GinzburgLandau(_make_config(N=64))
        sim1.reset(seed=1)
        for _ in range(20):
            sim1.step()
        state1 = sim1.observe().copy()

        sim2 = GinzburgLandau(_make_config(N=64))
        sim2.reset(seed=2)
        for _ in range(20):
            sim2.step()
        state2 = sim2.observe().copy()

        assert not np.allclose(state1, state2)


class TestGinzburgLandauRediscovery:
    def test_amplitude_data_generation(self):
        from simulating_anything.rediscovery.ginzburg_landau import (
            generate_amplitude_data,
        )
        data = generate_amplitude_data(n_c2=5, c1=2.0, n_steps=200, dt=0.05, N=64)
        assert len(data["c2"]) == 5
        assert len(data["mean_amplitude"]) == 5
        assert len(data["std_amplitude"]) == 5
        assert all(np.isfinite(data["mean_amplitude"]))

    def test_benjamin_feir_data_generation(self):
        from simulating_anything.rediscovery.ginzburg_landau import (
            generate_benjamin_feir_data,
        )
        data = generate_benjamin_feir_data(
            n_c1=3, n_c2=3, n_steps=200, dt=0.05, N=32
        )
        assert data["uniformity"].shape == (3, 3)
        assert data["bf_parameter"].shape == (3, 3)
        assert len(data["c1"]) == 3
        assert len(data["c2"]) == 3

    def test_defect_data_generation(self):
        from simulating_anything.rediscovery.ginzburg_landau import (
            generate_defect_data,
        )
        data = generate_defect_data(
            c1=2.0, c2=-1.2, n_steps=200, dt=0.05, N=64
        )
        assert len(data["time"]) == 500
        assert len(data["defect_counts"]) == 500
        assert all(d >= 0 for d in data["defect_counts"])
