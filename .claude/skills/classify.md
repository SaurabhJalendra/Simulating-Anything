---
name: classify
description: Classify all simulation domains into tiers (core, hand-crafted, template)
user_invocable: true
---

# /classify — Classify Domains

Run domain classification to get honest counts of real vs template-generated domains.

## What It Does
1. Scans all files in `src/simulating_anything/simulation/`
2. Classifies into 3 tiers:
   - **Tier 1 (Core)**: 14 domains with complete PySR/SINDy rediscovery evidence
   - **Tier 2 (Hand-crafted)**: >50 lines, named physics parameters, domain-specific docstrings
   - **Tier 3 (Template)**: 37-line templates with random coefficients
3. Outputs to `output/domain_classification.json`

## Implementation
```bash
python scripts/classify_domains.py
```
