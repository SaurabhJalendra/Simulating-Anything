---
name: Cognitive Architecture Rationale (April 2026)
description: Why Simulating-Anything pivots from a discovery pipeline to an AMI-style modular cognitive architecture. Cross-cutting synthesis tying ingested sources to design choices.
type: synthesis
sources:
  - lewm-2026
  - the-well-2024
  - wsindy-2021
  - time-warp-attend-2024
  - robin-2025
last_updated: 2026-04-29
---

# Cognitive Architecture Rationale — April 2026

## What this synthesis answers

Why is Simulating-Anything pivoting from a scientific-discovery pipeline to a modular cognitive architecture in the LeCun Autonomous Machine Intelligence tradition? What does the literature actually support? What's the evidence chain from current sources to architectural decisions?

This page traces the reasoning. It is not the design (see `docs/ARCHITECTURE.md`); it is the *why*.

## The trigger

The user posed an architecture-level question: *"What else should we have to make this an AGI-capable system?"*

Honest framing: AGI is not buildable as a solo project on consumer hardware. But a credible reference implementation of LeCun's AMI vision, demonstrated on a physics substrate, *is* buildable. That is the project's actual ceiling, and it is high enough to matter.

The pivot from "discovery engine" to "cognitive architecture" is not a re-scoping of ambition — it is a change of *framing* that better fits both the literature and the existing project assets.

## The five-stream literature survey (April 2026)

Six parallel research agents surveyed:

1. **World models**: state of JEPA, RSSM, predictive coding, world-model foundation models
2. **Symbolic regression**: AI-Feynman, KAN, WSINDy, ODEFormer, neural-symbolic distillation
3. **Multi-physics datasets**: The Well, PDEBench, MPP, Aurora, GraphCast, JHTDB, BLAST-Net
4. **Autonomous research loops**: Karpathy autoresearch, Sakana AI Scientist v1/v2, FunSearch, Bayesian experimental design, Coscientist, ChemCrow
5. **Bifurcation detection**: classical EWS, deep EWS (Bury 2021/2023), Time-Warp-Attend, TDA approaches, equation-free, reservoir computing
6. **Discovery agents**: Sakana, Anthropic multi-agent, Robin, PaperQA2, ChemCrow, MLE-Bench, ResearchBench, LLM-SRBench

~50 papers reviewed; 5 ingested as sources to date.

## What the literature actually says

Three convergent findings across the streams:

### Finding 1: The pieces exist; the integration does not

Every component the AMI vision requires has at least one credible implementation in the literature:

- World model with stable training: [LeWM](../sources/lewm-2026.md) (March 2026, first JEPA stable end-to-end from pixels)
- Symbolic reasoning robust to noise: [WSINDy](../sources/wsindy-2021.md) (10–20% noise tolerance; pysindy library-ready)
- Bifurcation detection with topology awareness: [Time-Warp-Attend](../sources/time-warp-attend-2024.md) (87% on Brusselator-class)
- Multi-agent committees that actually produce real discoveries: [Robin](../sources/robin-2025.md) (FutureHouse, ripasudil for dry-AMD, validated wet-lab)
- Multi-physics substrate: [The Well](../sources/the-well-2024.md) (15 TB, 16 PDE classes, NeurIPS 2024)
- Skill libraries: Voyager (NVIDIA, 2023, not yet ingested but central reference)
- Curriculum / autoresearch: Karpathy autoresearch (March 2026, not yet ingested)
- Active inference / free-energy: Friston framework (2009-present)

What does *not* exist in the literature: an open-source reference implementation that integrates all of these into one cognitive loop, on a substrate of comparable diversity to ours.

### Finding 2: The "discovery" framing is too narrow

Sakana's AI Scientist v1 (linear pipeline, 42% experiment failures) versus Robin (specialist committee, real discovery) is the canonical contrast. The architectural difference is *committee at validation*, *not at build*. Our 7-stage discovery pipeline is closer to Sakana v1 than to Robin.

The fix is not a more complicated discovery pipeline. The fix is a different architectural pattern: cognitive loop with reflection, skill library, planner, world-model rollouts. Discovery becomes one capability emergent from this loop, not the architecture.

### Finding 3: Frontier labs are pursuing this trajectory in closed code

OpenAI o1/o3, Anthropic agent SDK, DeepMind SIMA / AdA / Genie, Meta V-JEPA 2 — all proprietary, all narrowly scoped, none open as a complete architecture. The open-source community needs a credible reference implementation. Our project is positioned to be that, on a substrate (1500+ physics simulators) no other group has.

## Why the pivot is *the* right move

Three reasons, in order of weight:

### Reason 1: The substrate is irreplaceable

1500+ simulators built over two months. 277 trained RSSM world models. 316 discovery campaigns. This is real value that took real time to produce. Throwing it away to "restart cleanly with cognitive architecture" would be wasteful and is unnecessary — the existing assets fit naturally as the **substrate** of the new framing.

### Reason 2: The discovery paper becomes Phase 1 leverage, not the goal

If the project ships only as "scientific discovery engine," it is one paper. If it ships as "cognitive architecture, demonstrated *first* on scientific discovery," the discovery paper is Phase 1 of a multi-paper, multi-year research program. Same code, different framing, dramatically different ceiling.

### Reason 3: The architectural layers map cleanly to existing components

| AMI component | Existing project asset |
|---|---|
| World model | RSSM ensemble (277 trained) |
| Reasoning (symbolic) | SINDy, PySR, soon WSINDy + KAN-SR |
| Episodic memory | `knowledge/trajectory_store.py` |
| Semantic memory | the wiki (just installed) |
| Curiosity / exploration | `exploration/uncertainty_driven.py` |
| LLM as scaffold | `agents/*.py` (already constrained) |
| Environment substrate | `simulation/*` (1500+ domains) |
| Autoresearch loop | `campaign/manager.py` |

Roughly 30% of the architecture is already solidly in place. 30% is partial. 40% is missing — but the 40% is a clear, scoped engineering effort.

## What's not supported by the literature (honest)

The literature does NOT support claims that:

- AGI is achievable in 6–12 months by a solo dev on consumer hardware. It isn't.
- Universal problem-solving systems exist or are imminent. They don't.
- Any single paper provides a complete blueprint. None do — the AMI position paper (LeCun 2022) is a vision document, not an implementation guide.
- The "5 properties of intelligence" (causal counterfactual, sample efficiency, continuous learning, self-knowledge, multi-horizon goals) can be implemented to human-level performance now. They cannot.

What the literature DOES support: each property can be implemented to *useful* performance on bounded domains. That is the achievable ceiling.

## The architectural decisions traced to sources

Every component decision in `docs/ARCHITECTURE.md` traces to ingested sources:

- **World-model ensemble + router** ← [LeWM source](../sources/lewm-2026.md) (different latents capture different physics) + DreamerV3 (Hafner Nature 2025)
- **Symbolic reasoning as first-class channel** ← [WSINDy source](../sources/wsindy-2021.md) + KAN-SR + ODEFormer (extractable, auditable reasoning)
- **Validation committee at discovery boundary** ← [Robin source](../sources/robin-2025.md) (real discoveries) vs Sakana v1 (linear pipeline failures)
- **Bifurcation detection ensemble** ← [Time-Warp-Attend source](../sources/time-warp-attend-2024.md) + Bury 2021 deep EWS + classical statistics
- **External grounding via standard datasets** ← [The Well source](../sources/the-well-2024.md) + PDEBench + MPP

Components without yet-ingested sources (deferred ingestion priority):

- **Voyager-style skill library** ← Wang et al, *Voyager*, NVIDIA 2023 (priority next ingest)
- **Curiosity / free-energy** ← Friston, Pathak et al (priority ingest)
- **Causal layer** ← Pearl, Schölkopf et al (priority ingest)
- **Autoresearch loop pattern** ← Karpathy autoresearch March 2026 (priority ingest)

Once these four additional sources are ingested, every architectural component will have its supporting literature in the wiki.

## What this means for next session

1. **Ship the discovery paper as Phase 1.** Execute ADR-0001 changes. The discovery work proves the substrate; that proof is leverage for the broader pivot.
2. **Refactor the directory structure** (per `docs/REFACTOR-PROPOSAL.md`) to expose cognitive components. Stub all 18 components.
3. **Ingest the four priority remaining sources** (Voyager, Friston, Pearl, Karpathy autoresearch) to complete the architectural literature base.
4. **Begin Phase 3 implementation** with planner + skill library + causal layer as the highest-leverage new components.

## Open questions this synthesis cannot resolve

- **What is the right benchmark?** Existing benchmarks are LLM-shaped. A fair benchmark for an embodied cognitive architecture on physics reasoning does not exist; we may need to build one.
- **Where does the project's edge come from at scale?** Frontier labs have more compute. Our edge is architectural clarity — but at what point is "architectural clarity" *not* sufficient against "bigger model"?
- **Which is the first non-trivial cognitive task to demonstrate?** Discovery is Phase 1. After that — what specifically does the agent do that an LLM cannot? This needs a concrete proposal in Phase 2.
- **How is the LLM kept genuinely subordinate?** As soon as Claude is in the loop, there is temptation to lean on it for reasoning that should be done by world-model rollouts. The constitutional layer must hold this line.

These are not blockers. They are the questions Phase 2 work will answer.

## Closing

The pivot is justified by the literature, by the existing assets, and by the user's actual stated goal. The risk is in execution, not in concept.

The synthesis stands. The architecture is documented. The phasing is realistic. The next step is to commit, refactor when Phase 1 ships, and begin building.
