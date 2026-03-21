---
name: discover
description: Run a discovery campaign on a novel domain to find bifurcations, scaling laws, and phase boundaries
user_invocable: true
---

# /discover — Run Discovery Campaign

Run a discovery campaign on a specified domain to find genuine scientific discoveries.

## Usage
```
/discover <domain_name> <parameter> <lo> <hi> [--question "..."]
```

## Examples
```
/discover social_epidemic beta 0.1 0.8 --question "What transmission rate causes epidemic oscillations?"
/discover tumor_immune a_t 0.05 0.5 --question "What growth rate overwhelms immune control?"
/discover coral_reef N_in 0.1 3.0 --question "What nutrient level triggers algal phase shift?"
```

## What It Does
1. Sweeps the specified parameter across [lo, hi] at 200-300 points
2. Extracts observables at each point (mean, amplitude, period, Lyapunov, classification)
3. Detects bifurcations via gradient discontinuities
4. Fits scaling laws (polynomial degree 1-3) on parameter-observable relationships
5. Validates Hopf/InvHopf bifurcations across 5 random seeds
6. Generates bifurcation diagram figure
7. Saves results to output/discoveries/<domain>/campaign_results.json

## Implementation
```python
import sys
sys.path.insert(0, 'src')
from simulating_anything.analysis.campaign_runner import CampaignConfig, DiscoveryCampaignRunner

config = CampaignConfig(
    domain_name='<domain>',
    sim_module='<domain>',
    sim_class='<DomainSimulation>',
    question='<question>',
    sweep_params={'<param>': (<lo>, <hi>)},
    n_points=200,
    n_steps=5000,
    dt=0.01,
    base_params={},
)
runner = DiscoveryCampaignRunner(config)
result = runner.run()
```

## Requirements
- Domain must exist in `src/simulating_anything/simulation/`
- Class must follow `<CamelCase>Simulation` naming convention
- For GPU-dependent analysis: use WSL2
