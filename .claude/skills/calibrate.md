---
name: calibrate
description: Run discovery campaigns with literature-calibrated parameters and compare to published thresholds
user_invocable: true
---

# /calibrate — Literature-Calibrated Discovery

Run discovery campaigns using published experimental parameter values and compare discovered bifurcation thresholds to known results.

## What It Does
1. Loads literature parameter values for a domain (NPB, Tumor-Immune, or Brusselator)
2. Runs discovery campaign with those parameters as base_params
3. Compares discovered bifurcation thresholds to published values
4. Reports % error for each comparison

## Calibrated Domains
- **NPB** (nutrient_phage_bacteria): Levin 1977, Lenski 1988, Bohannan & Lenski 2000
- **Tumor-Immune**: Kuznetsov 1994, de Pillis 2005
- **Brusselator**: Exact analytical b_c = 1 + a^2

## Implementation
```bash
python scripts/run_calibrated_discoveries.py
```

## Key Files
- `src/simulating_anything/analysis/literature_calibration.py` — calibration module
- `scripts/run_calibrated_discoveries.py` — runner script
