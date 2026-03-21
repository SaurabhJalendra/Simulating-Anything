---
name: sweep
description: Run a 2D parameter sweep to generate phase diagrams with classified regions
user_invocable: true
---

# /sweep — 2D Phase Diagram

Run a 2D parameter sweep to map out the full phase diagram of a coupled dynamical system.

## Usage
```
/sweep <domain> <param1> <lo1> <hi1> <param2> <lo2> <hi2> [--points 30]
```

## Examples
```
/sweep social_epidemic beta 0.1 0.8 v_max 0.0 0.15
/sweep tumor_immune a_t 0.05 0.4 d_ti 0.01 0.12
/sweep coral_reef N_in 0.2 2.5 g_max 0.1 1.5
```

## What It Produces
1. Classification grid: each (param1, param2) point labeled steady/oscillatory/chaotic/divergent
2. Phase boundary detection between regions
3. 1D slice bifurcation analysis at the midpoint
4. Publication-quality phase diagram figure
5. Results saved to output/discoveries/<domain>/campaign_results.json

## Key Insight
2D sweeps reveal PHASE BOUNDARIES — the curves in parameter space separating different dynamical regimes. These are more scientifically valuable than 1D bifurcation points because they give the full picture of when a system transitions.
