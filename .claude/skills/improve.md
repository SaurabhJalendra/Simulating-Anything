---
name: improve
description: Analyze discovery failures and improve the pipeline architecture
user_invocable: true
---

# /improve — Pipeline Improvement

Analyze what's failing in the discovery pipeline and make targeted improvements.

## What It Checks
1. **Validation rate**: How many bifurcations pass 5-seed validation? (target: >40%)
2. **Classification accuracy**: Are "unknown" bifurcations misclassified Hopf/fold?
3. **False positives**: Are gradient-based detections finding real transitions?
4. **Missing discoveries**: Are there domains with known bifurcations we're not detecting?
5. **Parameter coverage**: Are we sweeping the right axes for each domain?

## Improvement Actions
- Lower SINDy threshold for domains with slow dynamics
- Increase n_steps for domains with long transients
- Refine around detected bifurcations with higher resolution
- Add domain-specific observable extraction
- Improve Lyapunov proxy for chaos detection

## Self-Improvement Protocol
After each improvement:
1. Re-run 5 representative campaigns
2. Compare validation rate before/after
3. If improved: commit and update lessons.md
4. If degraded: revert and try different approach
