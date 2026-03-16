"""Tests for extended V8 novel domains."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.types.simulation import Domain, SimulationConfig


def _run_sim(sim_class, dt=0.01, steps=500, **kwargs):
    config = SimulationConfig(domain=Domain.CUSTOM, dt=dt, n_steps=100, parameters=kwargs)
    sim = sim_class(config)
    sim.reset(seed=42)
    for _ in range(steps):
        sim.step()
    return sim.observe()


class TestTumorImmune:
    def test_shape(self):
        from simulating_anything.simulation.tumor_immune import TumorImmuneSimulation
        s = _run_sim(TumorImmuneSimulation, dt=0.1)
        assert s.shape == (4,)

    def test_no_nan(self):
        from simulating_anything.simulation.tumor_immune import TumorImmuneSimulation
        s = _run_sim(TumorImmuneSimulation, dt=0.1, steps=2000)
        assert not np.any(np.isnan(s))

    def test_positive(self):
        from simulating_anything.simulation.tumor_immune import TumorImmuneSimulation
        s = _run_sim(TumorImmuneSimulation, dt=0.1)
        assert np.all(s >= 0)


class TestGeneMetabolism:
    def test_shape(self):
        from simulating_anything.simulation.gene_metabolism import GeneMetabolismSimulation
        s = _run_sim(GeneMetabolismSimulation)
        assert s.shape == (4,)

    def test_no_nan(self):
        from simulating_anything.simulation.gene_metabolism import GeneMetabolismSimulation
        s = _run_sim(GeneMetabolismSimulation, steps=2000)
        assert not np.any(np.isnan(s))

    def test_positive(self):
        from simulating_anything.simulation.gene_metabolism import GeneMetabolismSimulation
        s = _run_sim(GeneMetabolismSimulation)
        assert np.all(s >= 0)


class TestPlanktonOcean:
    def test_shape(self):
        from simulating_anything.simulation.plankton_ocean import PlanktonOceanSimulation
        s = _run_sim(PlanktonOceanSimulation, dt=0.1)
        assert s.shape == (4,)

    def test_no_nan(self):
        from simulating_anything.simulation.plankton_ocean import PlanktonOceanSimulation
        s = _run_sim(PlanktonOceanSimulation, dt=0.1, steps=2000)
        assert not np.any(np.isnan(s))

    def test_positive(self):
        from simulating_anything.simulation.plankton_ocean import PlanktonOceanSimulation
        s = _run_sim(PlanktonOceanSimulation, dt=0.1)
        assert np.all(s >= 0)


class TestSocialEpidemic:
    def test_shape(self):
        from simulating_anything.simulation.social_epidemic import SocialEpidemicSimulation
        s = _run_sim(SocialEpidemicSimulation, dt=0.1)
        assert s.shape == (4,)

    def test_bounds(self):
        from simulating_anything.simulation.social_epidemic import SocialEpidemicSimulation
        s = _run_sim(SocialEpidemicSimulation, dt=0.1)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_no_nan(self):
        from simulating_anything.simulation.social_epidemic import SocialEpidemicSimulation
        s = _run_sim(SocialEpidemicSimulation, dt=0.1, steps=2000)
        assert not np.any(np.isnan(s))


class TestPredatorPreyPollution:
    def test_shape(self):
        from simulating_anything.simulation.predator_prey_pollution import PredatorPreyPollutionSimulation
        s = _run_sim(PredatorPreyPollutionSimulation)
        assert s.shape == (4,)

    def test_no_nan(self):
        from simulating_anything.simulation.predator_prey_pollution import PredatorPreyPollutionSimulation
        s = _run_sim(PredatorPreyPollutionSimulation, steps=2000)
        assert not np.any(np.isnan(s))

    def test_positive(self):
        from simulating_anything.simulation.predator_prey_pollution import PredatorPreyPollutionSimulation
        s = _run_sim(PredatorPreyPollutionSimulation)
        assert np.all(s >= 0)


class TestCircadianMetabolism:
    def test_shape(self):
        from simulating_anything.simulation.circadian_metabolism import CircadianMetabolismSimulation
        s = _run_sim(CircadianMetabolismSimulation, dt=0.1)
        assert s.shape == (4,)

    def test_no_nan(self):
        from simulating_anything.simulation.circadian_metabolism import CircadianMetabolismSimulation
        s = _run_sim(CircadianMetabolismSimulation, dt=0.1, steps=2000)
        assert not np.any(np.isnan(s))

    def test_positive(self):
        from simulating_anything.simulation.circadian_metabolism import CircadianMetabolismSimulation
        s = _run_sim(CircadianMetabolismSimulation, dt=0.1)
        assert np.all(s >= 0)


class TestPreyDiseasePredator:
    def test_shape(self):
        from simulating_anything.simulation.prey_disease_predator import PreyDiseasePredatorSimulation
        s = _run_sim(PreyDiseasePredatorSimulation)
        assert s.shape == (4,)

    def test_no_nan(self):
        from simulating_anything.simulation.prey_disease_predator import PreyDiseasePredatorSimulation
        s = _run_sim(PreyDiseasePredatorSimulation, steps=2000)
        assert not np.any(np.isnan(s))

    def test_positive(self):
        from simulating_anything.simulation.prey_disease_predator import PreyDiseasePredatorSimulation
        s = _run_sim(PreyDiseasePredatorSimulation)
        assert np.all(s[:3] >= 0)
