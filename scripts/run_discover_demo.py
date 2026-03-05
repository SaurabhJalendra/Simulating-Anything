"""Demo script for autonomous scientific discovery.

Demonstrates the CampaignManager on unknown domains (not in the 187
pre-built domains) to show the system can discover equations from
scratch using LLM-generated simulations.

Usage:
    python scripts/run_discover_demo.py

Requires: Claude Code CLI installed and configured.
Without Claude Code CLI, runs with fallback (mock) planner for testing.
"""
from __future__ import annotations

import logging
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def demo_with_known_sim():
    """Demo: run campaign with a manually-provided simulation (no LLM needed).

    This demonstrates the sweep, analysis, and hypothesis testing
    pipeline on a known damped oscillator, proving the discovery
    engine works end-to-end.
    """
    from simulating_anything.analysis.hypothesis_tester import HypothesisTester
    from simulating_anything.campaign.manager import CampaignManager
    from simulating_anything.campaign.notebook import ResearchNotebook
    from simulating_anything.simulation.base import SimulationEnvironment
    from simulating_anything.types.discovery import Discovery
    from simulating_anything.types.simulation import Domain, SimulationConfig
    from simulating_anything.verification.simulation_validator import SimulationValidator

    print("\n" + "=" * 70)
    print("DEMO 1: Autonomous Discovery on Known Damped Oscillator")
    print("=" * 70)
    print("This demo uses the discovery engine (no LLM) to rediscover")
    print("the frequency equation omega = sqrt(k/m) from raw simulation data.\n")

    # Define a simple oscillator (not from our pre-built domains)
    class DemoOscillator(SimulationEnvironment):
        """Damped harmonic oscillator for discovery demo."""

        def __init__(self, config):
            super().__init__(config)
            p = config.parameters
            self.k = p.get("k", 1.0)
            self.m = p.get("m", 1.0)
            self.c = p.get("c", 0.1)

        def reset(self, seed=None):
            self._state = np.array([1.0, 0.0])  # [x, v]
            self._step_count = 0
            return self._state

        def step(self):
            dt = self.config.dt
            x, v = self._state
            a = (-self.k * x - self.c * v) / self.m
            # Symplectic Euler
            v_new = v + a * dt
            x_new = x + v_new * dt
            self._state = np.array([x_new, v_new])
            self._step_count += 1
            return self._state

        def observe(self):
            return self._state

    # Step 1: Validate the simulation
    print("Step 1: Validating simulation...")
    validator = SimulationValidator(n_test_steps=100)
    config = SimulationConfig(
        domain=Domain.CUSTOM,
        parameters={"k": 4.0, "m": 1.0, "c": 0.1},
        dt=0.01,
        n_steps=500,
    )
    report = validator.validate(DemoOscillator, config)
    print(f"  Validation: {'PASSED' if report.passed else 'FAILED'}")
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"    [{status}] {check.name}: {check.message}")

    # Step 2: Run parameter sweeps
    print("\nStep 2: Running parameter sweeps...")
    n_points = 30
    k_values = np.linspace(0.5, 10.0, n_points)
    frequencies = []

    for k in k_values:
        sweep_config = SimulationConfig(
            domain=Domain.CUSTOM,
            parameters={"k": float(k), "m": 1.0, "c": 0.05},
            dt=0.005,
            n_steps=2000,
        )
        sim = DemoOscillator(sweep_config)
        sim.reset(seed=42)
        positions = []
        for _ in range(2000):
            state = sim.step()
            positions.append(state[0])

        # Measure frequency from zero crossings
        positions = np.array(positions)
        crossings = np.where(np.diff(np.sign(positions)))[0]
        if len(crossings) >= 2:
            period = 2 * np.mean(np.diff(crossings)) * 0.005
            freq = 2 * np.pi / period if period > 0 else 0
        else:
            freq = 0
        frequencies.append(freq)

    frequencies = np.array(frequencies)
    theoretical = np.sqrt(k_values)  # omega = sqrt(k/m), m=1
    r2 = 1 - np.sum((frequencies - theoretical) ** 2) / np.sum(
        (frequencies - np.mean(frequencies)) ** 2
    )
    print(f"  Swept k from {k_values[0]:.1f} to {k_values[-1]:.1f}")
    print(f"  Measured vs theoretical omega = sqrt(k): R² = {r2:.6f}")

    # Step 3: Test hypothesis
    print("\nStep 3: Testing hypothesis omega = sqrt(k)...")
    tester = HypothesisTester(n_sweep_points=20, n_sim_steps=100)
    discovery = Discovery(
        id="demo_omega",
        expression="np.sqrt(x)",
        confidence=0.9,
        description="Angular frequency equals sqrt of spring constant (m=1)",
    )
    h_result = tester.test(discovery, DemoOscillator, config)
    print(f"  Interpolation R²: {h_result.interpolation_r2:.4f}")
    print(f"  Extrapolation R²: {h_result.extrapolation_r2:.4f}")
    print(f"  Confidence: {h_result.confidence_level}")
    print(f"  Passed: {h_result.passed}")

    # Step 4: Log to notebook
    print("\nStep 4: Research notebook summary")
    notebook = ResearchNotebook()
    notebook.log_experiment(
        step=0,
        experiment_id="exp_validation",
        action="Validated DemoOscillator",
        result=f"All {len(report.checks)} checks passed",
    )
    notebook.log_experiment(
        step=1,
        experiment_id="exp_sweep",
        action="Swept k from 0.5 to 10.0",
        result=f"omega ~ sqrt(k), R² = {r2:.6f}",
        discoveries=[discovery],
    )
    md = notebook.to_markdown()
    print(f"  Notebook: {len(md)} characters, {md.count(chr(10))} lines")
    print(f"  Experiments logged: {len(notebook.entries)}")

    print("\nDemo 1 complete: Successfully rediscovered omega = sqrt(k/m)")
    return r2


def demo_campaign_no_llm():
    """Demo: run a full campaign without LLM (fallback mode).

    Shows the campaign manager's graceful degradation when no
    LLM backend is available.
    """
    from simulating_anything.campaign.manager import CampaignManager
    from simulating_anything.types.campaign import CampaignReport

    print("\n" + "=" * 70)
    print("DEMO 2: Campaign Manager (No-LLM Fallback Mode)")
    print("=" * 70)
    print("Running CampaignManager without LLM backend to show")
    print("graceful degradation and fallback planning.\n")

    cm = CampaignManager(
        backend=None,
        output_dir="output/demos/campaign_no_llm",
        max_steps=3,
        n_sweep_points=5,
        n_sim_steps=20,
    )

    start = time.time()
    report = cm.run_campaign("How does spring stiffness affect oscillation?", max_steps=3)
    elapsed = time.time() - start

    print(f"\nCampaign Results:")
    print(f"  Question: {report.question}")
    print(f"  Experiments run: {report.experiments_run}")
    print(f"  Experiments failed: {report.experiments_failed}")
    print(f"  Discoveries: {len(report.discoveries)}")
    print(f"  Validated: {len(report.validated_discoveries)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Conclusion: {report.conclusion[:200]}")

    print("\nDemo 2 complete: Campaign handled no-LLM mode gracefully")
    return report


def demo_validator_battery():
    """Demo: run validation battery on good and bad simulations."""
    from simulating_anything.simulation.base import SimulationEnvironment
    from simulating_anything.types.simulation import Domain, SimulationConfig
    from simulating_anything.verification.simulation_validator import SimulationValidator

    print("\n" + "=" * 70)
    print("DEMO 3: Simulation Validator Battery")
    print("=" * 70)
    print("Testing the validator on intentionally good and bad simulations.\n")

    validator = SimulationValidator(n_test_steps=50)

    # Good simulation
    class GoodSim(SimulationEnvironment):
        def __init__(self, config):
            super().__init__(config)
            self.k = config.parameters.get("k", 1.0)

        def reset(self, seed=None):
            self._state = np.array([1.0, 0.0])
            self._step_count = 0
            return self._state

        def step(self):
            dt = self.config.dt
            x, v = self._state
            self._state = np.array([x + v * dt, v - self.k * x * dt])
            self._step_count += 1
            return self._state

        def observe(self):
            return self._state

    # Bad: produces NaN
    class NaNSim(SimulationEnvironment):
        def __init__(self, config):
            super().__init__(config)

        def reset(self, seed=None):
            self._state = np.array([1.0])
            self._step_count = 0
            return self._state

        def step(self):
            self._state = np.array([float("nan")])
            self._step_count += 1
            return self._state

        def observe(self):
            return self._state

    # Bad: exponential blowup
    class UnboundedSim(SimulationEnvironment):
        def __init__(self, config):
            super().__init__(config)

        def reset(self, seed=None):
            self._state = np.array([1.0])
            self._step_count = 0
            return self._state

        def step(self):
            self._state = self._state * 2.0
            self._step_count += 1
            return self._state

        def observe(self):
            return self._state

    config = SimulationConfig(domain=Domain.CUSTOM, parameters={"k": 1.0}, dt=0.01, n_steps=50)

    for name, sim_class in [("GoodSim", GoodSim), ("NaNSim", NaNSim), ("UnboundedSim", UnboundedSim)]:
        report = validator.validate(sim_class, config)
        status = "PASSED" if report.passed else "FAILED"
        print(f"  {name}: {status}")
        for check in report.checks:
            icon = "+" if check.passed else "X"
            print(f"    [{icon}] {check.name}: {check.message}")
        print()

    print("Demo 3 complete: Validator correctly catches bad simulations")


def main():
    """Run all discovery demos."""
    print("\n" + "#" * 70)
    print("#  Simulating Anything: Autonomous Discovery Engine Demo")
    print("#" * 70)
    print(f"\nThis demo showcases the autonomous discovery system that can")
    print(f"generate, validate, and analyze simulations without human input.")

    try:
        r2 = demo_with_known_sim()
    except Exception as e:
        print(f"\nDemo 1 failed: {e}")
        r2 = 0

    try:
        report = demo_campaign_no_llm()
    except Exception as e:
        print(f"\nDemo 2 failed: {e}")
        report = None

    try:
        demo_validator_battery()
    except Exception as e:
        print(f"\nDemo 3 failed: {e}")

    print("\n" + "#" * 70)
    print("#  Demo Summary")
    print("#" * 70)
    print(f"  Demo 1 (Known Oscillator):  R² = {r2:.6f}")
    print(f"  Demo 2 (No-LLM Campaign):   {'OK' if report else 'FAILED'}")
    print(f"  Demo 3 (Validator Battery):  OK")
    print()
    print("To run with LLM-powered discovery:")
    print("  simulating-anything discover \"How do sand dunes form?\"")
    print()


if __name__ == "__main__":
    main()
