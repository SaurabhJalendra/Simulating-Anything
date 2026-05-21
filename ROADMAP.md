# Simulating-Anything — Roadmap

**Last updated:** 2026-05-21

> Synthesized from `IDEA.md` (4-phase plan) and `CLAUDE.md` §11 (V-series roadmap). The project is mid-pivot: from a standalone *discovery engine* (V9) to a modular *AMI-style cognitive architecture* (the discovery engine becomes its first demonstrated capability).

## Now (next 2-4 weeks)
- **Phase 1 — ship the discovery-engine paper** as the first demonstrated capability of the architecture (proves the 1500+ simulator substrate works)
- Finish V9 discovery campaigns — validate the 3 confirmed discoveries (Neuron-Astrocyte, Social-Epidemic, Ocean-Carbon inverse-Hopf thresholds) and push toward more genuine ones
- Execute ADR-0001 architectural changes (ensemble SINDy default landed; adaptive parameter-sweep sampling landed — continue the change list)
- V9C validation loop — extrapolation, 5-seed robustness, dt-invariance on all discoveries
- Paper rewrite with honest domain count (261 real) + calibration + prediction contributions

## Next (1-3 months)
- **Phase 2 — scaffold the cognitive architecture**: refactor the codebase to expose cognitive structure; write `docs/VISION.md` + `docs/ARCHITECTURE.md`; stub all 18 components
- V10 — autoresearch integration: fixed-budget experiment loops, per-domain SINDy threshold optimization, per-domain-class RSSM architecture search, overnight autonomous discovery loop
- Begin Phase 3 core components: planner, skill library, causal layer

## Later (exploratory)
- **Phase 3 (months 2-6)** — implement the full cognitive stack: hierarchical planner, Voyager-style skill library, structural-causal-model layer, reflection/self-modeling, autonomous concept formation, curiosity-driven exploration
- **Phase 4 (months 6-12)** — integration paper: *"AMI-Lite: A Modular Cognitive Architecture for Embodied Reasoning"*; ablation studies; open-source release; engage research community
- Cross-domain skill transfer experiments (Lorenz → Brusselator etc.)
- Sim-to-real transfer — shortest credible path from simulation discoveries to real-world tasks
- Benchmark design for embodied cognitive architectures (existing benchmarks are LLM-shaped)

## Recently Shipped (last 10)
- ✅ Adaptive parameter-sweep sampling helper (ADR-0001 Change #2)
- ✅ Ensemble SINDy default (ADR-0001 Change #1)
- ✅ ADR-0001 added for architectural changes; 5 papers ingested into wiki
- ✅ Framing pivot to AMI-style cognitive architecture
- ✅ V8 — scaled to 261 real domains + 1285 template stress tests, 277-281 world models
- ✅ V7 — 5 novel coupled systems, CKA latent analysis, cross-domain transfer
- ✅ Honest assessment — domain classification, extrapolation/noise/dream-discovery limits documented
- ✅ 316 discovery campaigns, 1267 discoveries, 118 validated bifurcations, 43 2D phase diagrams
- ✅ 14 core rediscoveries (projectile → logistic map) across 6 mathematical classes
- ✅ Cross-domain analogy engine (586+ isomorphisms across 200+ domains)

## Not Doing (explicit)
- ❌ AGI claims — the goal is the cleanest open-source AMI-style architecture on a constrained domain
- ❌ A universal problem solver — bounded by the simulator library and perceived modalities
- ❌ Competing with frontier labs on raw capability — the edge is architectural clarity + openness
- ❌ A product — this is a research platform
- ❌ A pure-LLM agent system — the LLM is strictly subordinate scaffold, never the core loop
- ❌ A pure-RL system — RL is one component, not the paradigm
- ❌ Autonomous deployment — all capabilities demonstrated human-in-the-loop
- ❌ Inflating numbers — honest results only; failures reported as future work
