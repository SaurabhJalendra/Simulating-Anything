---
name: Robin (FutureHouse)
description: Multi-agent scientific discovery system that produced a wet-lab-validated novel hypothesis (ripasudil for dry-AMD) in 2.5 months. May 2025.
type: source
sources: ["arXiv:2505.13400"]
last_updated: 2026-04-29
---

# Robin: End-to-End Scientific Discovery with a Multi-Agent System

**Authors:** FutureHouse team
**arXiv:** [2505.13400](https://arxiv.org/abs/2505.13400)
**Project:** [futurehouse.org Robin announcement](https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system)

## One-line claim

Three specialist agents (Crow, Falcon, Finch) orchestrated by a planner produced a wet-lab-validated novel hypothesis: ripasudil (a ROCK inhibitor) repurposed for dry age-related macular degeneration. Validated in 2.5 months.

## Architecture

| Agent | Role |
|---|---|
| **Crow** | Concise literature search (PaperQA2-based) |
| **Falcon** | Deep literature search |
| **Finch** | RNA-seq / flow-cytometry data analysis |
| **Planner** | Orchestrator that synthesizes specialist outputs |

## Why it matters for Simulating-Anything

**Robin produced a real discovery; Sakana v1 produced 42% experiment failures.** The architectural difference: Robin uses a *committee* of specialists at the discovery-validation boundary, while Sakana v1 used a linear template-driven pipeline.

Robin pattern is the recommended model for the project's LLM-agent layer — specifically at the validation boundary, not the build boundary.

## Recommended adaptation for this project

Replace the project's linear 7-stage pipeline (`problem_architect → domain_classifier → simulation_builder → ... → communicator`) with a **specialist committee at validation**:

| Robin agent | Simulating-Anything analogue |
|---|---|
| Crow + Falcon | Literature-grounding agent — PaperQA2 retrieval over calibration corpus (NPB chemostat, Brusselator, tumor-immune) |
| Finch | Replication-planner — generates dt/2, dt×2, 5-seed perturbations *before* committing to validated discoveries |
| (new) | Skeptic agent — must produce ≥3 alternative explanations for every candidate bifurcation |
| Planner | Orchestrator (you) — synthesizes, never delegates understanding (CLAUDE.md rule) |

Each candidate bifurcation/scaling-law passes through the committee before entering `validated_discoveries.jsonl`.

## Why committee-at-validation, not build

Anthropic's documented "wrong domain" for multi-agent is shared-context tasks. `simulation_builder` is one such task — keep it single-agent. Validation is the right place for adversarial committee structure: Skeptic disagrees with Lit-grounding, Replication-planner adds orthogonal evidence, orchestrator synthesizes.

## Honest contrast

- **Robin (this paper):** real discovery, wet-lab validated
- **Sakana AI Scientist v1** (arXiv:2408.06292): 42% experiments failed from coding errors; placeholder text shipped to PDFs; hallucinated numbers
- **Sakana AI Scientist v2** (arXiv:2504.08066): one paper passed ICLR workshop review (avg score 6.33). Better than v1 but not yet a real-world novel discovery.

## Related concepts

- [Multi-agent orchestrator-worker pattern](#concepts/orchestrator-worker)
- [Specialist committee](#concepts/specialist-committee)
- [Adversarial review](#concepts/adversarial-review)

## Related sources

- [PaperQA2 (Skarlinski 2024)](#sources/paperqa2-2024) — Crow/Falcon's underlying engine
- [Sakana AI Scientist v2](#sources/sakana-v2-2025) — counter-example
- [Anthropic multi-agent research system (Jun 2025)](#sources/anthropic-multi-agent-2025) — the orchestrator-worker reference

## Related entities

- [FutureHouse](#entities/futurehouse) — releasing organization
