---
name: predict
description: Generate specific testable experimental predictions from discovered bifurcations
user_invocable: true
---

# /predict — Generate Testable Predictions

Generate specific, falsifiable experimental predictions based on discovered bifurcations.

## What It Does
1. Takes calibrated discovery results
2. Generates predictions with: hypothesis, conditions, expected outcome, experimental protocol
3. Assigns confidence levels (high/medium/low)
4. Links to supporting bifurcation evidence

## Current Predictions
- **NPB burst threshold**: T4 phage burst >50 triggers oscillations (HIGH confidence)
- **NPB dual dilution windows**: Two oscillatory regimes in D_dilution (HIGH)
- **NPB paradox of enrichment**: Re-entry into oscillations at high nutrient (MEDIUM)
- **Tumor immune escape**: Growth rate >0.15/day overwhelms immune control (MEDIUM)
- **Brusselator validation**: Detected b_c within 10-15% of exact (validated)

## Implementation
```python
from simulating_anything.analysis.prediction_generator import generate_all_predictions
predictions = generate_all_predictions()
```
