# Wiki Index

The LLM-maintained knowledge base for Simulating-Anything. Drop sources into `raw/`, ask questions, and Claude updates this wiki page-by-page over time.

## Categories

- **entities/** — people, organizations, products, institutions referenced in sources
- **concepts/** — ideas, theories, mathematical methods, dynamical-systems concepts
- **sources/** — one page per ingested paper/article/book (summary + key takeaways)
- **syntheses/** — cross-cutting analyses, comparisons, theses, novel framings

---

## Entities
(no entities yet — will be populated as sources are ingested)

## Concepts
(no concepts yet — will be populated as sources are ingested)

## Sources

- [LeWorldModel (LeWM)](sources/lewm-2026.md) — first JEPA stable end-to-end from pixels, 192-dim latent, 15M params, March 2026 (LeCun et al)
- [The Well](sources/the-well-2024.md) — 15 TB / 16 physics simulation datasets, NeurIPS 2024 (Polymathic AI)
- [Weak SINDy (WSINDy)](sources/wsindy-2021.md) — weak-form symbolic regression, consistent at 10–20% noise (Messenger & Bortz 2021)
- [Time-Warp-Attend (TWA)](sources/time-warp-attend-2024.md) — topology-aware Hopf classifier, 87% accuracy at σ=0.1 (Talmon et al, ICLR 2024)
- [Robin (FutureHouse)](sources/robin-2025.md) — multi-agent system that produced wet-lab validated dry-AMD treatment (May 2025)

## Syntheses

- [Cognitive Architecture Rationale (April 2026)](syntheses/cognitive-architecture-rationale-2026-04.md) — why we pivot from discovery pipeline to AMI-style modular cognitive architecture; traces every architectural decision to ingested sources

---

## How to use

1. **Ingest**: drop a file into `raw/` (PDFs, articles, clippings). Ask Claude: "Ingest the new file in raw/."
2. **Query**: ask Claude any question. It reads `index.md` first, drills into relevant pages, answers with citations.
3. **Lint**: run `/wiki-lint` periodically to find contradictions, orphan pages, stale claims.

See CLAUDE.md "LLM Wiki Operations" section for the full protocol.
