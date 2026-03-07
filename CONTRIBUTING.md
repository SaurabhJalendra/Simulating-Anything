# Contributing to Simulating Anything

## Adding a New Simulation Domain

The pipeline's core design principle is that adding a new domain requires only
implementing a `SimulationEnvironment` subclass (~50-200 lines). Everything
else -- world model, exploration, analysis, reporting -- works automatically.

### Step 1: Implement the Simulation

Create `src/simulating_anything/simulation/your_domain.py`:

```python
"""Your Domain simulation.

Brief description of the physical system and key equations.
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class YourDomainSimulation(SimulationEnvironment):
    """Simulation of [your domain].

    State: [x, y, ...] where x = ..., y = ...
    Parameters: a (default=1.0), b (default=2.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        self.a = config.parameters.get("a", 1.0)
        self.b = config.parameters.get("b", 2.0)
        self.dt = config.dt
        self.state = np.zeros(2)

    def reset(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self.state = np.array([
            self.config.parameters.get("x0", 1.0),
            self.config.parameters.get("y0", 0.0),
        ])
        return self.observe()

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        x, y = state
        dxdt = self.a * x - self.b * x * y
        dydt = -self.a * y + self.b * x * y
        return np.array([dxdt, dydt])

    def step(self) -> np.ndarray:
        # RK4 integration
        dt = self.dt
        k1 = self._derivatives(self.state)
        k2 = self._derivatives(self.state + 0.5 * dt * k1)
        k3 = self._derivatives(self.state + 0.5 * dt * k2)
        k4 = self._derivatives(self.state + dt * k3)
        self.state = self.state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self.observe()

    def observe(self) -> np.ndarray:
        return self.state.copy()
```

Key conventions:
- Use `from __future__ import annotations` in every file
- Parameters come from `config.parameters` dict (values must be `float`)
- Use RK4 for ODEs, spectral FFT for PDEs, symplectic for Hamiltonian systems
- State must be a numpy array
- `observe()` returns a copy, not a reference
- `reset()` returns the initial observation

### Step 2: Add a Rediscovery Module

Create `src/simulating_anything/rediscovery/your_domain.py`:

```python
"""Your Domain rediscovery.

Targets:
- [What equation/law you expect to recover]
- [What analysis methods to use: PySR, SINDy, or both]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from simulating_anything.simulation.your_domain import YourDomainSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


def _make_config(**params) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CUSTOM,
        n_steps=5000,
        dt=0.01,
        parameters=params,
    )


def generate_sweep_data(
    param_name: str = "a",
    param_values: np.ndarray | None = None,
    n_steps: int = 5000,
) -> dict:
    """Sweep one parameter and measure observables."""
    if param_values is None:
        param_values = np.linspace(0.5, 5.0, 30)

    results = {"param_values": param_values.tolist(), "observable": []}

    for val in param_values:
        config = _make_config(**{param_name: val})
        sim = YourDomainSimulation(config)
        sim.reset(seed=42)
        # Run and collect data
        for _ in range(n_steps):
            sim.step()
        obs = sim.observe()
        results["observable"].append(float(obs[0]))

    return results


def run_your_domain_rediscovery(
    output_dir: str | Path = "output/rediscovery/your_domain",
    **kwargs,
) -> dict:
    """Run rediscovery analysis for your domain."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = generate_sweep_data()

    results = {
        "domain": "your_domain",
        "sweep_data": data,
        "target_equation": "y = f(a)",
    }

    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results
```

### Step 3: Add Tests

Create `tests/unit/test_your_domain.py`:

```python
"""Tests for YourDomain simulation."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.simulation.your_domain import YourDomainSimulation
from simulating_anything.types.simulation import Domain, SimulationConfig


def _make_config(**params) -> SimulationConfig:
    return SimulationConfig(
        domain=Domain.CUSTOM,
        n_steps=1000,
        dt=0.01,
        parameters=params,
    )


class TestYourDomainBasics:
    def test_init(self):
        sim = YourDomainSimulation(_make_config())
        assert sim is not None

    def test_reset_shape(self):
        sim = YourDomainSimulation(_make_config())
        obs = sim.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2,)

    def test_step_shape(self):
        sim = YourDomainSimulation(_make_config())
        sim.reset(seed=42)
        obs = sim.step()
        assert obs.shape == (2,)

    def test_no_nan(self):
        sim = YourDomainSimulation(_make_config())
        sim.reset(seed=42)
        for _ in range(1000):
            obs = sim.step()
        assert not np.any(np.isnan(obs))

    def test_deterministic(self):
        sim1 = YourDomainSimulation(_make_config())
        sim2 = YourDomainSimulation(_make_config())
        obs1 = sim1.reset(seed=42)
        obs2 = sim2.reset(seed=42)
        for _ in range(100):
            obs1 = sim1.step()
            obs2 = sim2.step()
        np.testing.assert_array_equal(obs1, obs2)

    def test_parameter_sensitivity(self):
        sim1 = YourDomainSimulation(_make_config(a=1.0))
        sim2 = YourDomainSimulation(_make_config(a=2.0))
        sim1.reset(seed=42)
        sim2.reset(seed=42)
        for _ in range(100):
            obs1 = sim1.step()
            obs2 = sim2.step()
        assert not np.allclose(obs1, obs2)
```

### Step 4: Register the Domain

1. Add an import to `src/simulating_anything/rediscovery/__init__.py`
2. Add the domain to the runner in `src/simulating_anything/rediscovery/runner.py`
3. Update the `Domain` enum in `src/simulating_anything/types/simulation.py` if needed

### Running Tests

```bash
# Run your tests
python -m pytest tests/unit/test_your_domain.py -v

# Run lint
ruff check src/simulating_anything/simulation/your_domain.py

# Run full suite to check nothing is broken
python -m pytest tests/unit/ -q --tb=short
```

## Code Style

- Line length: 99 characters
- Python 3.11+ with `from __future__ import annotations`
- Type hints on all functions using `|` union syntax
- Google-style docstrings
- Ruff for linting: `ruff check src/ tests/`

## Testing Guidelines

- Every simulation domain needs at minimum:
  - Instantiation test
  - State shape test
  - NaN/Inf check over 1000 steps
  - Determinism test (same seed = same output)
  - Parameter sensitivity test
- Conservation laws should be verified where applicable
- Rediscovery tests should verify data generation shapes
