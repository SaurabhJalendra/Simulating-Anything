---
name: validate
description: Validate a discovered bifurcation across 5 random seeds to confirm it's genuine
user_invocable: true
---

# /validate — Validate Discovery

Validate a bifurcation discovery across 5 random seeds to confirm it's genuine and reproducible.

## Usage
```
/validate <domain> <parameter> <critical_value> [--dt 0.01] [--steps 5000]
```

## Examples
```
/validate social_epidemic v_max 0.08
/validate neuron_astrocyte coupling_an 0.20
/validate tumor_immune d_ti 0.089
```

## What It Does
1. Runs simulation at critical_value - 10% and critical_value + 10%
2. For each: runs 5 random seeds
3. Extracts observable classification at each seed
4. CONFIRMED if: all seeds agree below AND all seeds agree above AND they differ
5. Reports: below classifications, above classifications, validation status

## Validation Criteria
- **CONFIRMED**: All 5 seeds give same classification below AND same (different) above
- **PARTIAL**: 4/5 seeds agree on each side
- **UNCONFIRMED**: Seeds disagree or no transition detected
