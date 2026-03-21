---
name: discovery-loop
description: Run the autonomous discovery loop — continuously discover, validate, improve
user_invocable: true
---

# /discovery-loop — Autonomous Discovery Loop

Run the continuous discovery loop that picks domains, runs campaigns, validates findings, and improves the pipeline.

## The Loop
```
FOREVER:
  1. Pick a novel domain with untested parameter axes
  2. Run discovery campaign (200-300 points)
  3. Auto-validate any Hopf/InvHopf bifurcations (5 seeds)
  4. If policy equation possible: find control boundary
  5. Log results to output/discoveries/
  6. Update CLAUDE.md with new counts
  7. Commit and push
  8. Pick next domain, repeat
```

## Domain Priority
1. Domains with hand-crafted physics that haven't been fully explored
2. Parameter axes not yet swept on previously explored domains
3. 2D sweeps on domains with rich 1D structure
4. Refinement near borderline bifurcations

## Stop Conditions
- User says stop
- Context window approaching limit (compress and continue next session)
- All 35 hand-crafted domains fully explored on all axes

## Current Status
Run /status to see how many campaigns remain.
