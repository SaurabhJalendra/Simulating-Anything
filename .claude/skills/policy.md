---
name: policy
description: Find a policy equation — the control boundary between success and failure in a coupled system
user_invocable: true
---

# /policy — Find Policy Equation

Find the control boundary equation: given one parameter, what value of a second parameter is needed to achieve a desired outcome (e.g., epidemic elimination, tumor control).

## Usage
```
/policy <domain> <control_param> <sweep_param> <sweep_lo> <sweep_hi> --target "<condition>"
```

## Examples
```
/policy social_epidemic v_max beta 0.1 0.8 --target "final_I < 1e-6"
/policy tumor_immune d_ti a_t 0.05 0.5 --target "final_tumor < 10"
/policy vegetation_hydrology P_rain coupling_vw 0.0 3.0 --target "final_veg > 0.1"
```

## What It Produces
1. For each value of sweep_param, finds the minimum control_param that achieves the target
2. Fits a polynomial (degree 1-3) to the boundary: control_min = f(sweep_param)
3. Validates across 5 seeds at 5 test points
4. Generates policy figure: green=success region, red=failure region

## Why This Matters
Policy equations are the most impactful discoveries because they directly answer:
"Given X, how much Y do I need?" — actionable for decision-makers.

## Existing Policy Equations
1. v_min = 0.443*beta - 0.039 (epidemic elimination, R²=0.9999)
2. d_ti_min = 0.736*a_t - 0.046 (tumor control, R²=0.99)
3. P_rain_min = 0.208 (dryland survival, R²=0.93)
