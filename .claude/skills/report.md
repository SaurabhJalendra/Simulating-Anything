---
name: report
description: Generate a comprehensive discovery report summarizing all findings
user_invocable: true
---

# /report — Discovery Report

Generate a comprehensive report of all discoveries made by the engine.

## What It Reports
1. Total campaigns, discoveries, validations
2. All validated bifurcations with domain, type, critical value
3. All policy equations with R² and validation status
4. All structural invariants
5. Phase diagrams found
6. Summary statistics by discovery type

## Implementation
```python
import json, os
for d in sorted(os.listdir('output/discoveries')):
    p = f'output/discoveries/{d}/campaign_results.json'
    if os.path.exists(p):
        with open(p) as f: r = json.load(f)
        # aggregate all discoveries
```

## Output
- Console: formatted report
- File: output/discoveries/DISCOVERY_REPORT.md
