"""Tests for V8 Batch 3 domains: LaserAbsorber, AtmosphereVegetation, BatteryThermal,
InfectionImmunity, ResourceConsumerWaste, PredatorPreyMigration,
PredatorPreyFear, NutrientPhageBacteria."""
from __future__ import annotations
import numpy as np
from simulating_anything.types.simulation import Domain, SimulationConfig


def _run(sim_class, dt=0.01, steps=500, **kw):
    config = SimulationConfig(domain=Domain.CUSTOM, dt=dt, n_steps=100, parameters=kw)
    sim = sim_class(config)
    sim.reset(seed=42)
    for _ in range(steps):
        sim.step()
    return sim.observe()


class TestLaserAbsorber:
    def test_shape(self):
        from simulating_anything.simulation.laser_absorber import LaserAbsorberSimulation
        assert _run(LaserAbsorberSimulation).shape == (4,)

    def test_no_nan(self):
        from simulating_anything.simulation.laser_absorber import LaserAbsorberSimulation
        assert not np.any(np.isnan(_run(LaserAbsorberSimulation, steps=2000)))

    def test_positive(self):
        from simulating_anything.simulation.laser_absorber import LaserAbsorberSimulation
        assert np.all(_run(LaserAbsorberSimulation) >= 0)


class TestAtmosphereVegetation:
    def test_shape(self):
        from simulating_anything.simulation.atmosphere_vegetation import AtmosphereVegetationSimulation
        assert _run(AtmosphereVegetationSimulation).shape == (4,)

    def test_no_nan(self):
        from simulating_anything.simulation.atmosphere_vegetation import AtmosphereVegetationSimulation
        assert not np.any(np.isnan(_run(AtmosphereVegetationSimulation, steps=2000)))


class TestBatteryThermal:
    def test_shape(self):
        from simulating_anything.simulation.battery_thermal import BatteryThermalSimulation
        assert _run(BatteryThermalSimulation).shape == (4,)

    def test_no_nan(self):
        from simulating_anything.simulation.battery_thermal import BatteryThermalSimulation
        assert not np.any(np.isnan(_run(BatteryThermalSimulation, steps=2000)))

    def test_soc_bounded(self):
        from simulating_anything.simulation.battery_thermal import BatteryThermalSimulation
        s = _run(BatteryThermalSimulation)
        assert 0 <= s[3] <= 1


class TestInfectionImmunity:
    def test_shape(self):
        from simulating_anything.simulation.infection_immunity import InfectionImmunitySimulation
        assert _run(InfectionImmunitySimulation, dt=0.1).shape == (4,)

    def test_sir_conservation(self):
        from simulating_anything.simulation.infection_immunity import InfectionImmunitySimulation
        s = _run(InfectionImmunitySimulation, dt=0.1)
        assert abs(s[0] + s[1] + s[2] - 1.0) < 0.05

    def test_no_nan(self):
        from simulating_anything.simulation.infection_immunity import InfectionImmunitySimulation
        assert not np.any(np.isnan(_run(InfectionImmunitySimulation, dt=0.1, steps=2000)))


class TestResourceConsumerWaste:
    def test_shape(self):
        from simulating_anything.simulation.resource_consumer_waste import ResourceConsumerWasteSimulation
        assert _run(ResourceConsumerWasteSimulation).shape == (4,)

    def test_positive(self):
        from simulating_anything.simulation.resource_consumer_waste import ResourceConsumerWasteSimulation
        assert np.all(_run(ResourceConsumerWasteSimulation)[:3] >= 0)


class TestPredatorPreyMigration:
    def test_shape(self):
        from simulating_anything.simulation.predator_prey_migration import PredatorPreyMigrationSimulation
        assert _run(PredatorPreyMigrationSimulation).shape == (4,)

    def test_positive(self):
        from simulating_anything.simulation.predator_prey_migration import PredatorPreyMigrationSimulation
        assert np.all(_run(PredatorPreyMigrationSimulation) >= 0)


class TestPredatorPreyFear:
    def test_shape(self):
        from simulating_anything.simulation.predator_prey_fear import PredatorPreyFearSimulation
        assert _run(PredatorPreyFearSimulation).shape == (4,)

    def test_fear_bounded(self):
        from simulating_anything.simulation.predator_prey_fear import PredatorPreyFearSimulation
        s = _run(PredatorPreyFearSimulation)
        assert 0 <= s[2] <= 1


class TestNutrientPhageBacteria:
    def test_shape(self):
        from simulating_anything.simulation.nutrient_phage_bacteria import NutrientPhageBacteriaSimulation
        assert _run(NutrientPhageBacteriaSimulation).shape == (4,)

    def test_positive(self):
        from simulating_anything.simulation.nutrient_phage_bacteria import NutrientPhageBacteriaSimulation
        assert np.all(_run(NutrientPhageBacteriaSimulation) >= 0)
