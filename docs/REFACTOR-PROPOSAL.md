# Refactor Proposal: From Discovery Pipeline to Cognitive Architecture

> **Status:** PROPOSAL. No file moves have been executed. This document describes the proposed new directory layout for `src/simulating_anything/` and the migration plan. Approval required before execution.

## Why refactor

The current directory layout reflects the project's original framing as a 7-stage scientific-discovery pipeline. The new framing (per `IDEA.md` and `VISION.md`) is a modular cognitive architecture in the AMI tradition. The directory layout should expose the cognitive structure so that:

1. New contributors immediately see the architectural backbone
2. Each cognitive component has a clear home
3. Stub modules for not-yet-built components occupy their permanent locations
4. Experiments and ablations can be wired against component boundaries

The current layout works. It is not broken. But it makes the AMI framing invisible.

## Current layout (today)

```
src/simulating_anything/
├── __init__.py
├── pipeline.py
├── agents/
│   ├── base.py
│   ├── problem_architect.py
│   ├── domain_classifier.py
│   ├── simulation_builder.py
│   └── communicator.py
├── simulation/
│   └── (1500+ simulator files)
├── world_model/
│   ├── rssm.py
│   ├── rssm_v2.py
│   ├── ensemble.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── advanced_encoders.py
│   ├── trainer.py
│   └── trainer_v2.py
├── exploration/
│   ├── base.py
│   └── uncertainty_driven.py
├── analysis/
│   ├── symbolic_regression.py
│   ├── equation_discovery.py
│   ├── ablation.py
│   ├── pipeline_ablation.py
│   ├── sensitivity.py
│   ├── cross_domain.py
│   ├── dream_debate.py
│   ├── domain_statistics.py
│   ├── error_analysis.py
│   ├── scaling_analysis.py
│   ├── baselines.py
│   ├── significance.py
│   ├── robustness.py
│   ├── computational_cost.py
│   ├── observable_extractor.py
│   ├── bifurcation_detector.py
│   ├── literature_calibration.py
│   ├── dt_invariance.py
│   ├── discovery_baselines.py
│   └── prediction_generator.py
├── rediscovery/
│   └── (per-domain rediscovery scripts)
├── knowledge/
│   ├── trajectory_store.py
│   ├── discovery_log.py
│   └── knowledge_base.py
├── discovery/
│   ├── open_problems.py
│   └── discovery_runner.py
├── verification/
│   ├── dimensional.py
│   ├── conservation.py
│   ├── transfer_validation.py
│   └── simulation_validator.py
├── types/
│   ├── problem_spec.py
│   ├── simulation.py
│   ├── trajectory.py
│   ├── discovery.py
│   └── campaign.py
├── utils/
│   └── config.py
└── campaign/
    ├── manager.py
    └── notebook.py
```

## Proposed layout

```
src/simulating_anything/
├── __init__.py
├── pipeline.py                    # KEEP — high-level orchestrator
│
├── perception/                    # NEW — Component 2.1
│   ├── encoder.py                 # MOVED from world_model/encoder.py
│   ├── advanced_encoders.py       # MOVED from world_model/
│   ├── multimodal.py              # NEW stub — unified interface
│   └── uncertainty.py             # NEW — per-modality uncertainty
│
├── world_model/                   # KEEP — Component 2.2
│   ├── rssm.py
│   ├── rssm_v2.py
│   ├── ensemble.py
│   ├── decoder.py
│   ├── trainer.py
│   ├── trainer_v2.py
│   ├── lewm.py                    # NEW (per ADR-0001)
│   ├── pde_fm.py                  # NEW (per ADR-0001)
│   └── router.py                  # NEW — picks which model
│
├── causal/                        # NEW — Component 2.3
│   ├── scm.py                     # Structural causal model
│   ├── intervention.py            # do-calculus, do(X)
│   ├── identification.py          # Pearl's identifiability checks
│   └── discovery.py               # PC, GES, NOTEARS, deep-causal
│
├── planning/                      # NEW — Component 2.4
│   ├── hierarchical.py            # Top-level orchestrator
│   ├── llm_planner.py             # Top level: LLM-as-planner
│   ├── option_critic.py           # Mid level: options
│   ├── mcts.py                    # Bottom level: MCTS over rollouts
│   └── verification.py            # World-model checks at every level
│
├── reasoning/                     # CONSOLIDATE — Component 2.5
│   ├── forward.py                 # Rollouts (NEW interface over world_model)
│   ├── backward.py                # NEW — inverse planning
│   ├── symbolic.py                # MOVED from analysis/symbolic_regression.py
│   ├── equation_discovery.py      # MOVED from analysis/
│   ├── kan_sr.py                  # NEW (per ADR-0001)
│   └── theorem_search.py          # NEW stub
│
├── skills/                        # NEW — Component 2.6
│   ├── library.py                 # Storage, retrieval, pruning
│   ├── voyager_loop.py            # Discovery via curiosity-driven play
│   ├── composition.py             # Hierarchical composition
│   ├── verification.py            # Self-test runner
│   └── retrieval.py               # NL-description semantic search
│
├── memory/                        # CONSOLIDATE — Component 2.7
│   ├── working.py                 # NEW
│   ├── episodic.py                # MOVED from knowledge/trajectory_store.py
│   ├── semantic.py                # NEW — programmatic interface to wiki/
│   └── procedural.py              # Reference to skills/
│
├── reflection/                    # NEW — Component 2.8
│   ├── self_model.py
│   ├── capability_tracker.py
│   ├── failure_analyzer.py
│   └── meta_learning.py
│
├── curiosity/                     # RENAMED from exploration/ — Component 2.9
│   ├── base.py
│   ├── uncertainty_driven.py      # KEEP
│   ├── empowerment.py             # NEW
│   ├── learning_progress.py       # NEW
│   └── free_energy.py             # NEW (depends on Component 2.17)
│
├── actuator/                      # NEW (formalizes campaign/) — Component 2.10
│   ├── executor.py                # MOVED from campaign/manager.py (subset)
│   ├── monitor.py                 # NEW — predicted-vs-actual tracking
│   └── replanner.py               # NEW — abort & re-plan logic
│
├── concepts/                      # NEW — Component 2.11
│   ├── formation.py
│   ├── compression.py             # MDL-based candidate generation
│   ├── naming.py                  # LLM-assisted concept naming
│   └── wiki_integration.py        # Auto-generate wiki pages
│
├── agents/                        # KEEP — Component 2.12 (LLM utility)
│   ├── base.py                    # ClaudeCodeBackend
│   ├── problem_architect.py
│   ├── domain_classifier.py
│   ├── simulation_builder.py
│   ├── simulation_generator.py
│   ├── research_planner.py
│   ├── communicator.py
│   ├── skeptic.py                 # NEW (per ADR-0001 Robin pattern)
│   ├── literature_grounding.py    # NEW (per ADR-0001)
│   └── replication_planner.py     # NEW (per ADR-0001)
│
├── theory_of_mind/                # NEW stub — Component 2.13 (deferred Phase 4+)
│   └── README.md                  # Placeholder; defer implementation
│
├── resource/                      # NEW — Component 2.14
│   ├── manager.py                 # Compute/memory/time budget allocation
│   └── profiler.py                # Tracks actual usage
│
├── goals/                         # NEW — Component 2.15
│   ├── stack.py                   # Multi-horizon goal tracking
│   └── decomposition.py           # Long → mid → short goal expansion
│
├── counterfactual/                # NEW — Component 2.16 (small)
│   └── reasoner.py                # Coordinates causal/ + memory/episodic
│
├── objective/                     # NEW — Component 2.17 (active inference glue)
│   └── free_energy.py             # Unified objective formulation
│
├── constitutional/                # NEW — Component 2.18
│   ├── constraints.py             # Hard rules system won't violate
│   └── value_layer.py             # Output filters
│
├── simulation/                    # KEEP UNCHANGED — environment substrate
│   └── (1500+ simulator files — untouched)
│
├── rediscovery/                   # KEEP — Phase 1 deliverable
│   └── (existing scripts)
│
├── verification/                  # KEEP — overlaps with constitutional/ and reflection/
│   ├── dimensional.py
│   ├── conservation.py
│   ├── transfer_validation.py
│   └── simulation_validator.py
│
├── types/                         # KEEP — Pydantic schemas
│   ├── problem_spec.py
│   ├── simulation.py
│   ├── trajectory.py
│   ├── discovery.py
│   └── campaign.py
│
├── utils/                         # KEEP
│   └── config.py
│
└── campaign/                      # KEEP for now — overlaps with planning/ and actuator/
    ├── manager.py                 # Eventually deprecated; functions migrate
    └── notebook.py
```

## Mapping summary (what moves)

| From | To | Rationale |
|---|---|---|
| `world_model/encoder.py`, `advanced_encoders.py` | `perception/` | Encoders are perception, not world-model dynamics |
| `analysis/symbolic_regression.py`, `equation_discovery.py` | `reasoning/` | Symbolic reasoning is a reasoning channel, not analysis |
| `analysis/observable_extractor.py`, `bifurcation_detector.py`, `literature_calibration.py`, `dt_invariance.py`, `discovery_baselines.py`, `prediction_generator.py` | `analysis/` (stays) | These ARE Phase 1 discovery analysis; not cognitive components |
| `analysis/cross_domain.py`, `dream_debate.py` | `reasoning/` (or stay) | Cross-domain analogy is reasoning; debate is meta-reasoning |
| `knowledge/trajectory_store.py` | `memory/episodic.py` | Trajectories are episodic memory |
| `knowledge/knowledge_base.py`, `discovery_log.py` | `memory/semantic.py` (interface) + stay | Knowledge base aligns with semantic memory; programmatic wiki interface lives in `memory/` |
| `exploration/` | `curiosity/` | Renamed for clarity; same role |
| `campaign/manager.py` | partially → `actuator/executor.py`, partially → `planning/` | Campaign management splits between execution and planning |

## Migration plan (3 days, when approved)

**Day 1: directory + stubs**
- Create all new directories
- Add `__init__.py` to each with module docstring referencing `ARCHITECTURE.md` section
- Create stub files (just docstrings + class skeletons) for missing components
- Add `# TODO(arch-2.X)` markers tying every stub back to architecture spec

**Day 2: moves**
- Move encoder files: `world_model/encoder.py` → `perception/encoder.py`
- Move analysis-vs-reasoning split: symbolic + equation discovery → `reasoning/`
- Move episodic memory: `knowledge/trajectory_store.py` → `memory/episodic.py`
- Rename `exploration/` → `curiosity/`
- Update all internal imports (Python tooling: `python -m libcst.tool` or `rope`)

**Day 3: validation**
- Run full test suite (7900+ tests must still pass)
- Run discovery campaigns to verify Phase 1 functionality unbroken
- Update `CLAUDE.md` directory map (Section 12)
- Update Sphinx docs (`docs/architecture.rst`, `docs/index.rst`)

**Risk:** breaking 7900+ tests. Mitigation: do moves incrementally with full-test-suite checks after each. Use `git mv` to preserve history. Keep old paths as re-exports for one release, then remove.

## What NOT to refactor

These stay exactly as-is:

- `simulation/` — 1500+ simulator files. Touching them risks breaking working domains.
- `types/` — Pydantic schemas are stable interfaces.
- `rediscovery/` — Phase 1 deliverable; no reason to disturb.
- `tests/` — internal imports get updated, but test layout stays.
- `docs/conf.py`, Sphinx setup — unchanged.

## Approval gate

This is a 3-day operation that touches ~50 files. It is reversible (`git revert`) but disruptive. Approval required before execution.

If approved: run as one focused 3-day sprint. Do not interleave with feature work.

If rejected: keep current layout, add new components as siblings to existing structure (`world_model/lewm.py`, `analysis/causal_layer.py`, etc.). Less clean but functional.

If partial: agree on which moves are in-scope. Execute those subset over 1 day.

## Decision

Awaiting your call on:
1. Full refactor (3 days, clean architecture exposed)
2. Partial refactor (1 day, only the most-used moves)
3. No refactor (add new components alongside old)

Recommendation: **Full refactor**, but only **after Phase 1** (the discovery paper) ships. Don't refactor mid-paper. Land the paper, then refactor cleanly into the new architecture for Phase 2.
